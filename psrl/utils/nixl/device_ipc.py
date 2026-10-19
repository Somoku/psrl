"""
CUDA IPC for readers that share a device -- or only a node -- with their source (3.7.2.1, 3.7.4).

A rollout that shares a GPU with its trainer can read the trainer's bytes with a device copy,
which the hardware does at HBM speed. Going through NIXL instead costs 7.7x: a same device NIXL
READ measures 52 GB/s against 399 to 447 GB/s for `cudaMemcpy` of the same bytes, because UCX
drives the copy through its rendezvous protocol rather than handing over a pointer. NIXL also
needs an active message transport for its control path, so an IPC only UCX configuration is
refused outright.

The same handle works across GPUs of one node when NVLink (or PCIe peer access) connects them:
a colocate reader whose GPU differs from the trainer rank serving it copies at 361 to 377 GB/s
instead of taking the RDMA loopback. Only the exporter's ordinal has to be translated through
the device UUID, which is what `local_device` and `import_region` do.

What is shared is one handle per *allocation*, not per tensor. A region already holds many
shards, so the handle count is a handful whatever the model size, which is the property the
design's memory accounting rests on. The importer maps a region on first touch and then
resolves any address inside it by offset.
"""

from __future__ import annotations

import logging

import torch
from torch.multiprocessing.reductions import reduce_tensor

psrl_logger = logging.getLogger("psrl")


def device_uuid(device: torch.device) -> str:
    """A device identity that holds across processes, unlike the CUDA ordinal under `CUDA_VISIBLE_DEVICES`."""
    if device.type != "cuda":
        return ""
    properties = torch.cuda.get_device_properties(device)
    return str(getattr(properties, "uuid", None) or properties.pci_bus_id)


def region_view(tensor: torch.Tensor) -> torch.Tensor:
    """A uint8 tensor over the whole storage behind `tensor`, which is the unit that is shared."""
    storage = tensor.untyped_storage()
    whole = torch.empty(0, dtype=torch.uint8, device=tensor.device)
    whole.set_(storage, 0, (storage.nbytes(),), (1,))
    return whole


def export_region(tensor: torch.Tensor, explain: bool = False):
    """
    An IPC handle for the region behind `tensor`, or None when it cannot be shared.

    Export fails for memory the caching allocator did not hand out, which is not an error: the
    reader then falls back to NIXL, which is correct everywhere and only slower. With
    `explain`, returns `(handle, error)` so a caller can report why an allocation was skipped.
    """
    if tensor.device.type != "cuda":
        return (None, "not a CUDA tensor") if explain else None
    try:
        handle = reduce_tensor(region_view(tensor))
        return (handle, None) if explain else handle
    except Exception as exc:  # best effort, NIXL remains the fallback
        psrl_logger.debug(f"CUDA IPC export failed for a region on {tensor.device}: {exc!r}")
        return (None, repr(exc)) if explain else None


def local_device(uuid: str) -> torch.device | None:
    """
    This process's device with `uuid`, or None when it does not see that GPU.

    Exporter and importer may number the same physical GPU differently: a trainer pinned to
    one GPU sees it as ordinal 0, while a rollout process with the whole node visible sees
    it as `nvidia-smi` numbers it. The UUID is the only identity that survives that, and it
    is also what decides whether a peer copy is possible at all (3.7.4).
    """
    if not uuid:
        return None
    for index in range(torch.cuda.device_count()):
        device = torch.device("cuda", index)
        if device_uuid(device) == uuid:
            return device
    return None


def import_region(handle: object, device: torch.device | None = None) -> torch.Tensor | None:
    """
    The exporter's region mapped into this process as a uint8 tensor, or None if it cannot be mapped.

    `device` is the local ordinal of the exporter's GPU. The handle records the exporter's
    own numbering, which is wrong here whenever the two processes were pinned differently;
    `rebuild` takes the storage device as an argument, so rewriting it attributes the mapped
    bytes to the right local device. The pointer itself is the exporter's allocation either
    way -- this only fixes which device's context drives a peer copy out of it.
    """
    try:
        rebuild, args = handle
        if device is not None and device.type == "cuda":
            args = (*args[:6], device.index or 0, *args[7:])
        return rebuild(*args)
    except Exception as exc:
        psrl_logger.debug(f"CUDA IPC import failed: {exc!r}")
        return None


def locate(regions: dict[int, int], address: int, length: int) -> tuple[int, int] | None:
    """
    The `(region base, offset into it)` of `[address, address + length)`, or None if no region holds it.

    `regions` maps each region's base address to its size. A read whose span escapes its region
    would alias whatever follows it, so the bound is checked rather than assumed.
    """
    for base, nbytes in regions.items():
        if base <= address and address + length <= base + nbytes:
            return base, address - base
    return None


class RegionMapper:
    """
    Regions imported from sources this process can reach without a NIC, addressed by the source's pointers.

    Two topologies share this mapping: a source on this process's own device (design 3.7.2.1)
    and a source on another device of this node (design 3.7.4). The second one is why
    `adopt` takes a device: the imported bytes live on the *source's* GPU either way, and
    the local ordinal that drives the copy is resolved from the source's UUID.

    Import is lazy. A source may publish hundreds of allocations while the reader's shard
    touches a handful; opening a handle costs ~0.2 ms and the first peer copy out of a new
    mapping pays page-table setup, so paying both only for regions an actual read lands in
    keeps the switch O(bytes read), not O(bytes published).

    A source exports `{base address: (handle, size)}`. Cataloguing is done once per source
    and kept until the source re-registers, since an imported mapping outliving its
    exporter's allocation is the same hazard `registration_epoch` guards against remotely.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.uuid = device_uuid(device)
        self._regions: dict[str, dict[int, tuple[object, int]]] = {}
        self._devices: dict[str, torch.device | None] = {}
        self._mapped: dict[str, dict[int, tuple[torch.Tensor, int]]] = {}

    def adopt(self, source: str, regions: dict[int, tuple[object, int]], device: torch.device | None = None) -> bool:
        """Record `source`'s regions and the device they live on. Nothing is imported yet."""
        catalog = {base: (handle, int(nbytes)) for base, (handle, nbytes) in regions.items()}
        if not catalog:
            return False
        self._regions[source] = catalog
        self._devices[source] = device
        return True

    def forget(self, source: str) -> None:
        self._regions.pop(source, None)
        self._devices.pop(source, None)
        self._mapped.pop(source, None)

    def knows(self, source: str) -> bool:
        return source in self._regions

    def resolve(self, source: str, address: int, length: int) -> tuple[torch.Tensor, int] | None:
        """This process's `(region tensor, byte offset)` for a span of `source`, or None if unreachable."""
        catalog = self._regions.get(source)
        if not catalog:
            return None
        found = locate({base: nbytes for base, (_, nbytes) in catalog.items()}, address, length)
        if found is None:
            return None
        base, offset = found
        mapped = self._mapped.setdefault(source, {})
        if base not in mapped:
            handle, nbytes = catalog[base]
            imported = import_region(handle, self._devices.get(source))
            if imported is None:
                self._mapped.pop(source, None)
                return None
            mapped[base] = (imported, nbytes)
        return mapped[base][0], offset
