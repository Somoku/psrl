"""
CUDA IPC for readers that share a device with their source (colocate, design 3.7.2.1).

A rollout that shares a GPU with its trainer can read the trainer's bytes with a device copy,
which the hardware does at HBM speed. Going through NIXL instead costs 7.7x: a same device NIXL
READ measures 52 GB/s against 399 to 447 GB/s for `cudaMemcpy` of the same bytes, because UCX
drives the copy through its rendezvous protocol rather than handing over a pointer. NIXL also
needs an active message transport for its control path, so an IPC only UCX configuration is
refused outright.

What is shared is one handle per *registered region*, not per tensor. A region already holds many
shards, so the handle count is a handful whatever the model size, which is the property the
design's memory accounting rests on. The importer maps a region once and then resolves any
address inside it by offset.
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


def export_region(tensor: torch.Tensor) -> object | None:
    """
    An IPC handle for the region behind `tensor`, or None when it cannot be shared.

    Export fails for memory the caching allocator did not hand out, which is not an error: the
    reader then falls back to NIXL, which is correct everywhere and only slower.
    """
    if tensor.device.type != "cuda":
        return None
    try:
        return reduce_tensor(region_view(tensor))
    except Exception as exc:  # best effort, NIXL remains the fallback
        psrl_logger.debug(f"CUDA IPC export failed for a region on {tensor.device}: {exc!r}")
        return None


def import_region(handle: object) -> torch.Tensor | None:
    """The exporter's region mapped into this process as a uint8 tensor, or None if it cannot be mapped."""
    try:
        rebuild, args = handle
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
    Regions imported from sources that share this process's device, addressed by the source's pointers.

    A source exports `{base address: handle}`. Mapping is done once per source and kept until the
    source re-registers, since an imported mapping outliving its exporter's allocation is the same
    hazard `registration_epoch` guards against remotely.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.uuid = device_uuid(device)
        self._mapped: dict[str, dict[int, tuple[torch.Tensor, int]]] = {}

    def adopt(self, source: str, regions: dict[int, object]) -> bool:
        """Map `source`'s regions into this process. Returns whether every one of them mapped."""
        mapped: dict[int, tuple[torch.Tensor, int]] = {}
        for base, handle in regions.items():
            imported = import_region(handle)
            if imported is None:
                self._mapped.pop(source, None)
                return False
            mapped[base] = (imported, imported.untyped_storage().nbytes())
        self._mapped[source] = mapped
        return bool(mapped)

    def forget(self, source: str) -> None:
        self._mapped.pop(source, None)

    def knows(self, source: str) -> bool:
        return source in self._mapped

    def resolve(self, source: str, address: int, length: int) -> tuple[torch.Tensor, int] | None:
        """This process's `(region tensor, byte offset)` for a span of `source`, or None if unmapped."""
        mapped = self._mapped.get(source)
        if not mapped:
            return None
        found = locate({base: nbytes for base, (_, nbytes) in mapped.items()}, address, length)
        if found is None:
            return None
        base, offset = found
        return mapped[base][0], offset
