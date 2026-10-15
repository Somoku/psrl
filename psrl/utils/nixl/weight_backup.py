"""The reference policy as a host-side snapshot of the actor, swapped in place (design §3.10.2).

PPO and GRPO with a KL term need the reference policy's log-probs every step. PSRL ran
it as a separate worker group -- verl's `ActorRolloutRefWorker` in the `ref` role --
which reads the checkpoint at startup and then holds a **second full copy of the model
on the training GPUs for the whole run**, to execute one forward pass per step. Under
colocate that copy is also the third tenant of a card already shared by the trainer and
the rollout.

The ref never trains, and it starts as the same bytes as the actor. So it is kept the way
miles keeps it (`TensorBackuper`): a pinned host copy of the actor's weights, swapped
into the actor's own storages for the forward pass and swapped back out after. That
replaces a model's worth of GPU memory with a model's worth of host memory, removes a
worker group and its disk read, and removes the precision-matching question the design
used to carry -- the snapshot is of the trainer's parameters, unrelated to what the
rollout is quantised to.

**Swapping in place, not loading a second module.** The forward pass runs on the actor's
own parameters, so it inherits every parallel layout, fused kernel and sequence-packing
path the actor already has; a separate module would need all of them reproduced. The
storages keep their identity, so nothing registered against them moves. (The weight
transport is not reading during a ref pass: rollout pulls happen at publish, and the
version barrier keeps readers out of an in-progress step.)

**Cost per swap is one D2H plus two H2D of the rank's weight shard**, at the measured
~46 GB/s per rank with eight ranks switching together (`bench_stage_offload.py`). The
actor's capture is skipped when nothing has written the weights since the last one,
which is the common case of several ref passes within one step.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from .stage_offload import _byte_view, pinned_host_buffer


@dataclass
class SwapStats:
    captured_bytes: int = 0
    loaded_bytes: int = 0
    seconds: float = 0.0


class WeightBackup:
    """Named host snapshots ("tags") of a fixed set of GPU storages.

    Built from the tensors the actor trains, deduplicated by storage exactly as the stage
    offloader is: a DDP bucket is one storage however many parameters view it.
    """

    def __init__(self, tensors: Iterable[torch.Tensor]) -> None:
        seen: set[int] = set()
        self._storages: list[torch.UntypedStorage] = []
        for tensor in tensors:
            if tensor is None or tensor.device.type != "cuda":
                continue
            storage = tensor.untyped_storage()
            if storage._cdata in seen or storage.nbytes() == 0:
                continue
            seen.add(storage._cdata)
            self._storages.append(storage)
        self._tags: dict[str, list[torch.Tensor]] = {}
        #: The tag whose bytes are on the GPU right now, if known.
        self._resident: str | None = None
        #: Whether the GPU bytes changed since `_resident` was last captured or loaded.
        self._dirty = True

    @property
    def nbytes(self) -> int:
        return sum(s.nbytes() for s in self._storages)

    @property
    def tags(self) -> set[str]:
        return set(self._tags)

    @property
    def resident(self) -> str | None:
        return self._resident

    def host_bytes(self) -> int:
        return sum(buf.numel() for bufs in self._tags.values() for buf in bufs)

    def mark_modified(self) -> None:
        """The GPU weights were written (an optimizer step). The next swap re-captures."""
        self._dirty = True

    def capture(self, tag: str) -> int:
        """Copy the GPU bytes into ``tag``'s host buffers. Returns bytes copied."""
        # Check liveness before allocating: pinned_host_buffer(0) would call
        # cudaHostRegister(0) which fails with a CUDA error before we reach _check_live.
        for storage in self._storages:
            self._check_live(storage)
        buffers = self._tags.get(tag)
        if buffers is None:
            buffers = [pinned_host_buffer(s.nbytes()) for s in self._storages]
            self._tags[tag] = buffers
        for storage, host in zip(self._storages, buffers):
            host.copy_(_byte_view(storage), non_blocking=True)
        torch.cuda.synchronize()
        self._resident = tag
        self._dirty = False
        return self.nbytes

    def load(self, tag: str) -> int:
        """Copy ``tag``'s host bytes onto the GPU storages, in place. Returns bytes copied."""
        if tag not in self._tags:
            raise KeyError(f"no snapshot named {tag!r}; captured: {sorted(self._tags)}")
        for storage, host in zip(self._storages, self._tags[tag]):
            self._check_live(storage)
            _byte_view(storage).copy_(host, non_blocking=True)
        torch.cuda.synchronize()
        self._resident = tag
        self._dirty = False
        return self.nbytes

    @contextmanager
    def swapped(self, tag: str, *, home: str = "actor") -> Iterator[SwapStats]:
        """Run the body with ``tag`` on the GPU, then put ``home`` back.

        ``home`` is captured first unless it is already resident and unmodified -- the
        GPU bytes then *are* the snapshot, and copying them out again is pure PCIe.
        The restore runs even if the body raises: leaving the ref's weights in the
        actor's storages would make the next optimizer step train the ref.
        """
        stats = SwapStats()
        start = time.perf_counter()
        if self._resident != home or self._dirty:
            stats.captured_bytes = self.capture(home)
        stats.loaded_bytes += self.load(tag)
        try:
            yield stats
        finally:
            stats.loaded_bytes += self.load(home)
            stats.seconds = time.perf_counter() - start

    @staticmethod
    def _check_live(storage: torch.UntypedStorage) -> None:
        if storage.nbytes() == 0:
            raise RuntimeError(
                "a backed-up storage is offloaded (size 0); swapping now would read or "
                "write nothing and leave the wrong weights resident. Restore the stage first."
            )
