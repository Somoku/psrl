"""Moving trainer weights off the GPU at a colocate stage boundary, and back (design §3.7.3).

At a colocate switch the trainer's GPU memory has to go to the rollout's KV cache: the
design's 72B accounting is ~50 GB of KV cache with the trainer resident against ~68 GB
without it, a 26% cut in rollout concurrency to save 18 GB of host memory per card.

Two things the existing sleep path got wrong, both silently:

* **It discarded the weights.** `_sleep_*_model` resizes each storage to zero and the
  wake path resizes it back *without restoring data*, relying on a later pull from the
  parameter server. Under direct supply (`ps_mode=nixl_direct`) there is no parameter
  server -- the trainer *is* the only copy -- so a switch destroyed the model and the next
  training step ran on whatever the allocator handed back.
* **It left the optimizer on the GPU.** Adam state is 2x the fp32 master weights, the
  single largest item a colocated trainer holds, and none of it was moved.

**Why storages and not `module.to("cpu")`.** Every consumer of these weights holds a
*view*: Megatron's parameters are views into its DDP buffers, and PSRL's converters hand
the weight transport views into the parameters (`unified_state_dict`). Moving a module
replaces each parameter's storage with a new CPU one, so the views keep the old GPU
storage alive -- the memory is never returned, and the weight transport keeps reading
stale bytes. Resizing the *same* storage to zero and back keeps its identity, so every
view, however derived, sees the restored bytes at the new address. The address does
change (the allocator chooses it), which is why the wake path re-registers with NIXL and
re-publishes client info; nothing here pretends otherwise.

**Pinned buffers are allocated once and reused.** `cudaHostAlloc` of 18 GB takes seconds
and allocating per switch also briefly doubles host usage (old and new buffer alive at
once), the failure verl's own offload hit as cgroup OOM kills. One buffer per storage,
kept for the life of the process, makes every switch after the first a pure copy.
"""

from __future__ import annotations

import ctypes
import logging
import os
import time
import weakref
from collections.abc import Iterable
from dataclasses import dataclass, field

import torch

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))


@dataclass
class OffloadStats:
    """What one direction of a switch moved, and how fast."""

    storages: int = 0
    bytes: int = 0
    seconds: float = 0.0
    #: GPU memory the driver reports free before and after, so a caller can tell whether
    #: the bytes actually went back to the device rather than into the caching allocator.
    free_before: int = 0
    free_after: int = 0

    @property
    def gbps(self) -> float:
        return self.bytes / self.seconds / 1e9 if self.seconds > 0 else 0.0

    @property
    def freed(self) -> int:
        return self.free_after - self.free_before


@dataclass
class _Entry:
    storage: torch.UntypedStorage
    nbytes: int
    #: Pinned host copy. None in discard mode, where the bytes are restored by someone
    #: else (a parameter-server pull) and only the allocation needs to come back.
    host: torch.Tensor | None = None


def storage_key(tensor: torch.Tensor) -> int:
    """Identity of the storage behind ``tensor``, stable across `resize_`."""
    return tensor.untyped_storage()._cdata


@dataclass
class StorageOffloader:
    """Offload and restore a set of storages, preserving their identity.

    ``preserve=False`` is the parameter-server case: the bytes are refilled by a pull,
    so only the allocation is released and restored. It is the old behaviour, kept
    because copying 18 GB to host and back would be wasted work there.

    ``zero_on_restore`` is for gradients: nobody reads their old contents, but leaving
    whatever the allocator hands back would make the first accumulation after a switch
    add to garbage wherever a code path assumes a zeroed buffer.
    """

    preserve: bool = True
    zero_on_restore: bool = False
    _entries: list[_Entry] = field(default_factory=list)
    _offloaded: bool = False

    @classmethod
    def from_tensors(
        cls,
        tensors: Iterable[torch.Tensor],
        *,
        preserve: bool = True,
        zero_on_restore: bool = False,
        exclude: set[int] | None = None,
    ) -> StorageOffloader:
        """Track the distinct storages behind ``tensors``.

        Deduplicated by the storage's identity rather than by tensor, because many
        tensors share one storage (every parameter in a DDP bucket) and offloading it
        once per view would copy the same bytes repeatedly.

        ``exclude`` names storages another offloader owns. Two offloaders resizing one
        storage would have the second copy zero bytes over the first's saved ones --
        which is what happens with FSDP, whose optimizer's param groups *are* the model
        parameters.
        """
        offloader = cls(preserve=preserve, zero_on_restore=zero_on_restore)
        offloader.retarget(tensors, exclude=exclude)
        return offloader

    def retarget(self, tensors: Iterable[torch.Tensor], *, exclude: set[int] | None = None) -> None:
        """Re-collect the storages to track, keeping pinned buffers for those still here.

        Needed because the set changes: Adam creates its moments on the first step, so
        an offloader built before training starts sees no optimizer state at all.
        Rebuilding from scratch would re-pin every buffer on each switch.
        """
        if self._offloaded:
            raise RuntimeError("cannot retarget while offloaded; restore first")
        previous = {entry.storage._cdata: entry for entry in self._entries}
        exclude = exclude or set()
        seen: set[int] = set()
        entries: list[_Entry] = []
        for tensor in tensors:
            if tensor is None or tensor.device.type != "cuda":
                continue
            storage = tensor.untyped_storage()
            key = storage._cdata
            if key in seen or key in exclude:
                continue
            seen.add(key)
            nbytes = storage.nbytes()
            if nbytes == 0:
                continue
            old = previous.get(key)
            host = old.host if old is not None and old.nbytes == nbytes else None
            entries.append(_Entry(storage=storage, nbytes=nbytes, host=host))
        self._entries = entries

    @property
    def keys(self) -> set[int]:
        return {entry.storage._cdata for entry in self._entries}

    @property
    def nbytes(self) -> int:
        return sum(entry.nbytes for entry in self._entries)

    @property
    def offloaded(self) -> bool:
        return self._offloaded

    def offload(self) -> OffloadStats:
        """Copy every storage to host (if preserving), then release it on the device."""
        if self._offloaded:
            raise RuntimeError(
                "offload called twice without a restore; the second call would copy the "
                "zero-size storages over the saved bytes"
            )
        stats = OffloadStats(storages=len(self._entries), bytes=self.nbytes)
        torch.cuda.synchronize()
        stats.free_before = torch.cuda.mem_get_info()[0]
        start = time.perf_counter()
        if self.preserve:
            for entry in self._entries:
                if entry.host is None:
                    entry.host = pinned_host_buffer(entry.nbytes)
            for entry in self._entries:
                entry.host.copy_(_byte_view(entry.storage), non_blocking=True)
            # One sync for the whole batch: the copies are queued on the current stream
            # in order, and a per-copy sync would serialise PCIe behind host latency.
            torch.cuda.synchronize()
        for entry in self._entries:
            entry.storage.resize_(0)
        torch.cuda.empty_cache()
        stats.seconds = time.perf_counter() - start
        stats.free_after = torch.cuda.mem_get_info()[0]
        self._offloaded = True
        return stats

    def restore(self) -> OffloadStats:
        """Reallocate every storage and, if preserving, copy its bytes back."""
        if not self._offloaded:
            raise RuntimeError("restore called without a matching offload")
        stats = OffloadStats(storages=len(self._entries), bytes=self.nbytes)
        torch.cuda.synchronize()
        stats.free_before = torch.cuda.mem_get_info()[0]
        start = time.perf_counter()
        for entry in self._entries:
            entry.storage.resize_(entry.nbytes)
            if self.preserve:
                _byte_view(entry.storage).copy_(entry.host, non_blocking=True)
            elif self.zero_on_restore:
                _byte_view(entry.storage).zero_()
        torch.cuda.synchronize()
        stats.seconds = time.perf_counter() - start
        stats.free_after = torch.cuda.mem_get_info()[0]
        self._offloaded = False
        return stats

    def release_host(self) -> int:
        """Free the pinned buffers. Returns bytes released. Only valid while restored."""
        if self._offloaded:
            raise RuntimeError("releasing host buffers while offloaded would lose the weights")
        released = sum(entry.host.numel() for entry in self._entries if entry.host is not None)
        for entry in self._entries:
            entry.host = None
        return released


def _byte_view(storage: torch.UntypedStorage) -> torch.Tensor:
    """A flat uint8 tensor over the whole of ``storage``."""
    return torch.empty(0, dtype=torch.uint8, device=storage.device).set_(storage)


_PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
_MPOL_PREFERRED = 1
_SYS_MBIND = 237  # x86_64


def pinned_host_buffer(nbytes: int) -> torch.Tensor:
    """Page-locked host memory of exactly ``nbytes``, on the current GPU's NUMA node.

    Not `torch.empty(..., pin_memory=True)`, for two measured reasons (H20, torch 2.11):

    * **It rounds up to a power of two.** A 9 GiB request took 16 GiB of host memory --
      the caching host allocator's size classes. For a trainer offloading tens of GB per
      rank, eight ranks a node, that is the difference between fitting and an OOM kill.
    * **It is slow to allocate**: 3.65 s per 8 GiB against 0.79 s for registering a plain
      allocation, with identical copy bandwidth afterwards (51 / 57 GB/s).

    NUMA placement is set on the range *before* `cudaHostRegister` faults it in, so the
    pages land next to the GPU's root complex whichever thread touches them. Measured
    with eight ranks switching at once: 373 GB/s node aggregate local versus 266 remote.
    Best effort -- a kernel that refuses `mbind` still gets a correct, if slower, buffer.
    """
    raw = torch.empty(nbytes, dtype=torch.uint8)
    ptr = raw.data_ptr()
    # `mbind` wants page alignment; bind the aligned interior, which is all but at most
    # two partial pages.
    head = (-ptr) % _PAGE
    _bind_to_node(ptr + head, (nbytes - head) // _PAGE * _PAGE, current_gpu_numa_node())
    # Registered from the allocation's own base: `Tensor.is_pinned` asks about the
    # storage's base pointer, so registering an aligned interior view leaves PyTorch
    # believing the buffer is pageable (measured) and treating copies accordingly.
    error = torch.cuda.cudart().cudaHostRegister(ptr, nbytes, 0)
    if int(error) != 0:
        raise RuntimeError(f"cudaHostRegister of {nbytes} bytes failed: {error}")
    # A separate tensor object over the same storage, so the finalizer can hold `raw`
    # and unregister before the allocation is freed rather than after.
    view = raw[:]
    weakref.finalize(view, _unregister, ptr, raw)
    return view


def _unregister(ptr: int, _keepalive: torch.Tensor) -> None:
    torch.cuda.cudart().cudaHostUnregister(ptr)


def current_gpu_numa_node() -> int | None:
    """NUMA node of the current CUDA device's PCIe root, or None if unknown."""
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        bus = f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
        with open(f"/sys/bus/pci/devices/{bus}/numa_node") as f:
            node = int(f.read().strip())
        return node if node >= 0 else None
    except (OSError, ValueError, AttributeError):
        return None


def _bind_to_node(ptr: int, nbytes: int, node: int | None) -> bool:
    if node is None or node > 63:
        return False
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        mask = ctypes.c_ulong(1 << node)
        result = libc.syscall(
            _SYS_MBIND,
            ctypes.c_void_p(ptr),
            ctypes.c_ulong(nbytes),
            ctypes.c_int(_MPOL_PREFERRED),
            ctypes.byref(mask),
            ctypes.c_ulong(64),
            ctypes.c_uint(0),
        )
        return result == 0
    except (OSError, AttributeError):
        return False


def megatron_weight_tensors(models) -> list[torch.Tensor]:
    """The tensors whose storages hold a Megatron model's weights.

    Under DDP the parameters are views into contiguous buffers, so the buffers are the
    unit -- a handful of large copies instead of thousands of small ones. Without DDP
    (`wrap_with_ddp=false`, used at 235B to drop the gradient buffer) each parameter owns
    its storage.

    Gradients are deliberately excluded: they are zeroed at the start of every step, so
    preserving them would copy bytes nobody reads. The caller releases them separately.
    """
    try:
        from megatron.core import DistributedDataParallel as DDP  # noqa: PLC0415
    except ImportError:  # pragma: no cover - exercised only without Megatron
        DDP = None

    from megatron.bridge.models.conversion.utils import unwrap_model  # noqa: PLC0415

    tensors: list[torch.Tensor] = []
    for chunk in models:
        if DDP is not None and isinstance(chunk, DDP):
            for buffers in (chunk.buffers, chunk.expert_parallel_buffers):
                for buffer in buffers:
                    if buffer.param_data is not None:
                        tensors.append(buffer.param_data)
        else:
            tensors.extend(param.data for _, param in unwrap_model(chunk).named_parameters())
        tensors.extend(buf for _, buf in unwrap_model(chunk).named_buffers() if buf is not None)
    return tensors


def megatron_grad_tensors(models) -> list[torch.Tensor]:
    """Gradient storages, released without backup and zeroed on restore."""
    try:
        from megatron.core import DistributedDataParallel as DDP  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        DDP = None

    from megatron.bridge.models.conversion.utils import unwrap_model  # noqa: PLC0415

    tensors: list[torch.Tensor] = []
    for chunk in models:
        if DDP is not None and isinstance(chunk, DDP):
            for buffers in (chunk.buffers, chunk.expert_parallel_buffers):
                for buffer in buffers:
                    if buffer.grad_data is not None:
                        tensors.append(buffer.grad_data)
        else:
            tensors.extend(p.grad for _, p in unwrap_model(chunk).named_parameters() if p.grad is not None)
    return tensors


def fsdp_weight_tensors(model) -> list[torch.Tensor]:
    """Local shards of an FSDP2 model's parameters and buffers."""
    from torch.distributed.tensor import DTensor  # noqa: PLC0415

    tensors = [p._local_tensor if isinstance(p, DTensor) else p for _, p in model.named_parameters()]
    tensors.extend(b for _, b in model.named_buffers() if b is not None)
    return tensors


def _local(tensor: torch.Tensor) -> torch.Tensor:
    from torch.distributed.tensor import DTensor  # noqa: PLC0415

    return tensor._local_tensor if isinstance(tensor, DTensor) else tensor


def _torch_optimizers(optimizer) -> list[torch.optim.Optimizer]:
    """The plain `torch.optim.Optimizer`s inside whatever wrapper a backend uses.

    Megatron nests: a `ChainedOptimizer` holds one per param group kind (dense, expert),
    each a mixed-precision wrapper whose `.optimizer` is the real Adam; a
    `HybridDeviceOptimizer` holds several. FSDP2 hands over the Adam directly.
    """
    if optimizer is None:
        return []
    if isinstance(optimizer, torch.optim.Optimizer) and not hasattr(optimizer, "sub_optimizers"):
        return [optimizer]
    found: list[torch.optim.Optimizer] = []
    for child in getattr(optimizer, "chained_optimizers", None) or []:
        found.extend(_torch_optimizers(child))
    for child in getattr(optimizer, "sub_optimizers", None) or []:
        found.extend(_torch_optimizers(child))
    inner = getattr(optimizer, "optimizer", None)
    if inner is not None and inner is not optimizer:
        found.extend(_torch_optimizers(inner))
    return found


def optimizer_tensors(optimizer) -> list[torch.Tensor]:
    """Every tensor an optimizer owns: its param groups' parameters and its state.

    The param groups matter as much as the state. Under Megatron's mixed precision the
    inner Adam steps the **fp32 master copies**, which live only there; the bf16 weights
    are rebuilt from them after each step. Offloading the moments but not the masters
    leaves the largest single item on the GPU, and *dropping* the masters (as the TMS
    sleep path did) makes the next step write zeros, or whatever the allocator returned,
    over the model.

    Under FSDP2 those parameters are the model's own, so the caller excludes storages
    the weight offloader already owns.
    """
    tensors: list[torch.Tensor] = []
    for inner in _torch_optimizers(optimizer):
        for group in inner.param_groups:
            tensors.extend(_local(param) for param in group["params"])
        for state in inner.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    tensors.append(_local(value))
    return tensors


@dataclass(frozen=True)
class StagePolicy:
    """What the trainer does with each kind of GPU memory at a colocate switch."""

    #: Copy weights to host and back. Otherwise they are released and refilled by a pull.
    preserve_weights: bool
    #: Copy optimizer state (fp32 masters and moments) to host and back.
    offload_optimizer: bool
    #: Let torch_memory_saver release everything else (activations, caches). Preserved
    #: storages are offloaded before the pause and restored after the resume, because a
    #: TMS pause with CPU backup disabled -- how PSRL configures it -- drops contents.
    tms_pause: bool

    def describe(self) -> str:
        weights = "preserved on host" if self.preserve_weights else "released, refilled by pull"
        optimizer = "preserved on host" if self.offload_optimizer else "left on GPU"
        return f"weights {weights}; optimizer {optimizer}; tms_pause={self.tms_pause}"


def stage_policy(*, has_ps_data_plane: bool, offload_optimizer: bool, tms_pause: bool) -> StagePolicy:
    """Decide the switch behaviour from facts about the deployment.

    Weights are preserved exactly when nobody else holds them. With a parameter server
    the switch-back pulls them, so copying to host would be 2x PCIe for nothing; under
    direct supply the trainer *is* the only copy, and releasing it is data loss.

    The optimizer has no such second home in any mode, so it is preserved whenever it is
    moved at all. Leaving it on the GPU is the opt-out, for a deployment whose KV cache
    does not need the room and would rather skip the copy -- **except under a TMS pause**,
    which would drop it: PSRL configures TMS without CPU backup, so the fp32 masters came
    back as whatever the driver mapped and the next step wrote them over the model. The
    opt-out is overridden there rather than honoured into silent corruption.
    """
    return StagePolicy(
        preserve_weights=not has_ps_data_plane,
        offload_optimizer=offload_optimizer or tms_pause,
        tms_pause=tms_pause,
    )


class TrainerStage:
    """The trainer's side of a colocate switch: what to keep, what to drop, and both ways.

    Shared by the production train worker and the e2e harness, so the harness validates
    the code that ships rather than a description of it. TMS pausing stays with the
    caller, because only the worker owns the TMS context.
    """

    def __init__(self) -> None:
        #: Persistent across switches so pinned host buffers are allocated once.
        self._keep: StorageOffloader | None = None
        #: Rebuilt per switch; owns no host memory.
        self._drop: StorageOffloader | None = None

    @property
    def offloaded(self) -> bool:
        return self._keep is not None and self._keep.offloaded

    @property
    def host_bytes(self) -> int:
        if self._keep is None:
            return 0
        return sum(e.host.numel() for e in self._keep._entries if e.host is not None)

    def offload(
        self,
        *,
        weights: list[torch.Tensor],
        grads: list[torch.Tensor],
        optimizer: list[torch.Tensor],
        policy: StagePolicy,
    ) -> tuple[OffloadStats, OffloadStats]:
        keep = list(optimizer) if policy.offload_optimizer else []
        if policy.preserve_weights:
            keep = list(weights) + keep
        if self._keep is None:
            self._keep = StorageOffloader.from_tensors(keep)
        else:
            # Adam's state appears after the first step, so the set is re-collected each
            # time; pinned buffers for storages already seen are kept.
            self._keep.retarget(keep)
        dropped = list(grads) + ([] if policy.preserve_weights else list(weights))
        self._drop = StorageOffloader.from_tensors(
            dropped, preserve=False, zero_on_restore=True, exclude=self._keep.keys
        )
        return self._keep.offload(), self._drop.offload()

    def restore(self) -> OffloadStats | None:
        """Kept storages first: they hold the bytes nobody can recreate."""
        restored = self._keep.restore() if self._keep is not None and self._keep.offloaded else None
        if self._drop is not None:
            self._drop.restore()
            self._drop = None
        return restored
