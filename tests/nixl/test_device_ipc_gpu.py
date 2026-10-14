"""
Sharing a device: a reader on the same GPU copies its source's bytes instead of transferring them.

The colocate path (design 3.7.2.1) rests on two claims this pins down. First, a region exported by
one process is addressable in another by the exporter's own pointers, which is what lets the reader
reuse the descriptors the planner already built. Second, the copy is worth making: a same device
NIXL READ measures 52 GB/s here against 399 to 447 GB/s for `cudaMemcpy` of the same bytes, so the
transfer costs 7.7x what the hardware needs.

Run cross process with spawn, since that is where it is used: the trainer and the rollout are
separate processes that happen to share a card.
"""

from __future__ import annotations

import pytest
import torch
import torch.multiprocessing as mp
from psrl.utils.nixl.device_ipc import RegionMapper, device_uuid, export_region, locate, region_view

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA IPC needs a GPU")


def _reader(handle, base, offset_bytes, shape, strides, queue):
    """Child: map the exporter's region and read a strided view through the exporter's addresses."""
    try:
        torch.cuda.set_device(0)
        mapper = RegionMapper(torch.device("cuda:0"))
        adopted = mapper.adopt("source", {base: handle})
        address = base + offset_bytes
        nbytes = (max((n - 1) * o for n, o in zip(shape, strides)) + 1) * 2
        found = mapper.resolve("source", address, nbytes)
        region, offset = found
        view = torch.as_strided(region.view(torch.bfloat16), shape, strides, offset // 2)
        out = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
        out.copy_(view)
        queue.put((adopted, True, out.float().cpu().tolist()))
    except Exception as exc:  # a dead child would otherwise look like a timeout
        queue.put((False, False, repr(exc)))


def _run_reader(exported, base, offset_bytes, shape, strides):
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_reader, args=(exported, base, offset_bytes, shape, strides, queue))
    process.start()
    try:
        return queue.get(timeout=120)
    finally:
        process.join(timeout=60)


def test_a_reader_on_the_same_device_sees_the_exporter_bytes():
    torch.cuda.set_device(0)
    arena = torch.arange(1 << 20, dtype=torch.bfloat16, device="cuda")
    exported = export_region(arena)
    assert exported is not None, "a tensor from the caching allocator must be exportable"
    # A column cut of a row major matrix, which is what a finer reader asks of a train shard.
    matrix = arena[: 64 * 128].view(64, 128)
    column = matrix[:, 32:64]
    offset_bytes = column.data_ptr() - arena.untyped_storage().data_ptr()
    adopted, resolved, values = _run_reader(
        exported,
        arena.untyped_storage().data_ptr(),
        offset_bytes,
        tuple(column.shape),
        tuple(column.stride()),
    )
    assert adopted and resolved
    assert values == column.float().cpu().tolist()


def test_a_span_outside_its_region_does_not_resolve():
    # Pure address arithmetic, so it is checked directly rather than through a mapping: CUDA
    # refuses to import a handle the same process exported, which is why `adopt` is not used here.
    regions = {1000: 500, 4000: 100}
    assert locate(regions, 1000, 500) == (1000, 0)
    assert locate(regions, 1200, 300) == (1000, 200)
    assert locate(regions, 4000, 100) == (4000, 0)
    assert locate(regions, 1000, 501) is None  # one byte past the end
    assert locate(regions, 1500, 1) is None  # the gap between regions
    assert locate(regions, 900, 200) is None  # starts before any region
    assert locate({}, 1000, 1) is None


def test_the_device_identity_survives_cuda_visible_devices():
    torch.cuda.set_device(0)
    uuid = device_uuid(torch.device("cuda:0"))
    assert uuid and uuid != "0", "the identity must not be the ordinal, which is per process"
    assert device_uuid(torch.device("cpu")) == ""


def test_the_region_view_covers_the_whole_storage():
    arena = torch.empty(1000, dtype=torch.bfloat16, device="cuda")
    view = region_view(arena[100:200])
    assert view.numel() == arena.untyped_storage().nbytes()
    assert view.data_ptr() == arena.untyped_storage().data_ptr()
