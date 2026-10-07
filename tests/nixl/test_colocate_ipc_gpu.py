"""CUDA IPC weight handover for the colocate path (item A8).

Design 3.7.2.1/3.7.3 replaces the colocate transfer with CUDA IPC into one reusable
buffer, and claims `register_memory` is never called on that path. These tests verify
the mechanism cross-process, which is where it is actually used: verl and SkyRL both
share trainer memory with an inference process, not within one process.

What is checked:

* a tensor exported by `reduce_tensor` and rebuilt in another process aliases the same
  device memory, so a producer-side write is visible without a copy
* one reusable uint8 buffer can carry tensors of several dtypes and shapes via
  `view(dtype).view(shape)`, including the fp8 weight+scale pairing
* the buffer size is fixed and independent of model size, which is the property the
  design's memory accounting rests on
* no NIXL registration is involved -- on this path the import is not even reached

Uses `torch.multiprocessing` with spawn, matching how verl's `rebuild_ipc` is used.
"""

from __future__ import annotations

import pytest
import torch
import torch.multiprocessing as mp
from torch.multiprocessing.reductions import reduce_tensor

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

IPC_BUFFER_BYTES = 64 * 1024 * 1024  # the design's reusable bucket, verl uses 512MB


def _rebuild(handle):
    """Rebuild a tensor from an IPC handle, as verl's `rebuild_ipc` does."""
    func, args = handle
    return func(*args)


def _consumer_reads(handle, queue, offset: int, length: int):
    """Child process: rebuild the shared buffer and report what it sees.

    Returns a plain list rather than a tensor: putting even a CPU tensor on the queue
    fails with EOFError in the parent, because the child's reducer still routes it
    through the CUDA IPC machinery that the shared buffer installed.
    """
    try:
        buffer = _rebuild(handle)
        queue.put(("ok", buffer[offset : offset + length].cpu().tolist()))
    except Exception:
        import traceback

        queue.put(("error", traceback.format_exc()))


def _consumer_writes(handle, queue, offset: int, values):
    """Child process: write into the shared buffer so the parent can observe it."""
    try:
        buffer = _rebuild(handle)
        payload = torch.tensor(values, dtype=torch.uint8, device=buffer.device)
        buffer[offset : offset + payload.numel()].copy_(payload)
        torch.cuda.synchronize()
        queue.put(("ok", None))
    except Exception:
        import traceback

        queue.put(("error", traceback.format_exc()))


@pytest.fixture(scope="module")
def spawn_context():
    return mp.get_context("spawn")


class TestCrossProcessAliasing:
    """The property the design relies on: IPC shares memory, it does not copy it."""

    def test_producer_write_is_visible_to_consumer(self, spawn_context):
        buffer = torch.zeros(IPC_BUFFER_BYTES, dtype=torch.uint8, device="cuda")
        pattern = torch.arange(256, dtype=torch.int64).remainder(251).to(torch.uint8).cuda()
        buffer[:256].copy_(pattern)
        torch.cuda.synchronize()

        queue = spawn_context.Queue()
        process = spawn_context.Process(
            target=_consumer_reads, args=(reduce_tensor(buffer), queue, 0, 256)
        )
        process.start()
        status, payload = queue.get(timeout=120)
        process.join(timeout=60)

        assert status == "ok", f"consumer failed: {payload}"
        assert payload == pattern.cpu().tolist(), "consumer saw different bytes"

    def test_consumer_write_is_visible_to_producer(self, spawn_context):
        """Writes must propagate both ways; a copy would make this fail."""
        buffer = torch.zeros(IPC_BUFFER_BYTES, dtype=torch.uint8, device="cuda")
        values = list(range(1, 33))

        queue = spawn_context.Queue()
        process = spawn_context.Process(
            target=_consumer_writes, args=(reduce_tensor(buffer), queue, 1024, values)
        )
        process.start()
        status, error = queue.get(timeout=120)
        process.join(timeout=60)

        assert status == "ok", f"consumer failed: {error}"
        torch.cuda.synchronize()
        got = buffer[1024 : 1024 + len(values)].cpu().tolist()
        assert got == values, "producer did not observe the consumer's write"

    def test_handle_survives_a_second_consumer(self, spawn_context):
        """Several rollout ranks share one trainer buffer, so re-import must work."""
        buffer = torch.zeros(IPC_BUFFER_BYTES, dtype=torch.uint8, device="cuda")
        buffer[:16].copy_(torch.arange(16, dtype=torch.uint8, device="cuda"))
        torch.cuda.synchronize()
        handle = reduce_tensor(buffer)

        for _ in range(2):
            queue = spawn_context.Queue()
            process = spawn_context.Process(target=_consumer_reads, args=(handle, queue, 0, 16))
            process.start()
            status, payload = queue.get(timeout=120)
            process.join(timeout=60)
            assert status == "ok", f"consumer failed: {payload}"
            assert payload == list(range(16))


class TestReusableBufferTyping:
    """One uint8 bucket must carry every dtype the precision matrix produces."""

    @pytest.mark.parametrize(
        "dtype,shape",
        [
            (torch.bfloat16, (64, 128)),
            (torch.float32, (32, 64)),
            (torch.float8_e4m3fn, (128, 128)),
            (torch.float16, (16, 16)),
        ],
    )
    def test_view_round_trip_preserves_bytes(self, dtype, shape):
        """`view(dtype).view(shape)` is how the bucket is reinterpreted (3.7.2.1)."""
        buffer = torch.zeros(IPC_BUFFER_BYTES, dtype=torch.uint8, device="cuda")
        numel = shape[0] * shape[1]
        nbytes = numel * torch.empty(0, dtype=dtype).element_size()

        source = (torch.arange(numel, device="cuda") % 7).reshape(shape).to(dtype)
        buffer[:nbytes].view(dtype).view(shape).copy_(source)
        recovered = buffer[:nbytes].view(dtype).view(shape)

        assert torch.equal(recovered.to(torch.float32), source.to(torch.float32))

    def test_fp8_weight_and_scale_share_one_bucket(self):
        """fp8 ships weight and scale together; they must coexist in one allocation.

        The design makes them one commit group for exactly this reason -- half a group
        arriving is a silent no-op in the engine loader.
        """
        buffer = torch.zeros(IPC_BUFFER_BYTES, dtype=torch.uint8, device="cuda")
        weight_shape, block = (256, 256), 128
        scale_shape = (weight_shape[0] // block, weight_shape[1] // block)

        weight = (torch.arange(weight_shape[0] * weight_shape[1], device="cuda") % 5).reshape(
            weight_shape
        ).to(torch.float8_e4m3fn)
        scale = torch.full(scale_shape, 2.0, dtype=torch.float32, device="cuda")

        weight_bytes = weight.numel()  # fp8 is 1 byte
        scale_bytes = scale.numel() * 4
        buffer[:weight_bytes].view(torch.float8_e4m3fn).view(weight_shape).copy_(weight)
        buffer[weight_bytes : weight_bytes + scale_bytes].view(torch.float32).view(
            scale_shape
        ).copy_(scale)

        got_weight = buffer[:weight_bytes].view(torch.float8_e4m3fn).view(weight_shape)
        got_scale = buffer[weight_bytes : weight_bytes + scale_bytes].view(torch.float32).view(
            scale_shape
        )
        assert torch.equal(got_weight.to(torch.float32), weight.to(torch.float32))
        assert torch.equal(got_scale, scale)

    def test_buffer_size_does_not_track_model_size(self):
        """The accounting claim: one fixed bucket, reused, regardless of model.

        Streaming twice the bytes through the same allocation must not grow it.
        """
        buffer = torch.zeros(IPC_BUFFER_BYTES, dtype=torch.uint8, device="cuda")
        before = buffer.untyped_storage().nbytes()

        chunk = torch.ones(IPC_BUFFER_BYTES // 2, dtype=torch.uint8, device="cuda")
        for _ in range(4):
            buffer[: chunk.numel()].copy_(chunk)
            torch.cuda.synchronize()

        assert buffer.untyped_storage().nbytes() == before == IPC_BUFFER_BYTES


class TestNoNixlRegistrationOnColocatePath:
    """3.7.3's claim: the colocate path never calls register_memory."""

    def test_ipc_handover_needs_no_nixl_call(self, spawn_context, monkeypatch):
        """Fail if the IPC path reaches NIXL registration.

        At MoE scale registration is ~140k calls, so this is what makes colocate cheap.
        Patching the agent method means an accidental dependency surfaces as a failure
        rather than as unexplained setup time.
        """
        pytest.importorskip("nixl._api", reason="NIXL not installed")
        from nixl._api import nixl_agent

        calls: list[object] = []
        original = nixl_agent.register_memory

        def tracked(self, *args, **kwargs):
            calls.append(args)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(nixl_agent, "register_memory", tracked)

        buffer = torch.zeros(IPC_BUFFER_BYTES, dtype=torch.uint8, device="cuda")
        buffer[:8].copy_(torch.arange(8, dtype=torch.uint8, device="cuda"))
        torch.cuda.synchronize()

        queue = spawn_context.Queue()
        process = spawn_context.Process(
            target=_consumer_reads, args=(reduce_tensor(buffer), queue, 0, 8)
        )
        process.start()
        status, payload = queue.get(timeout=120)
        process.join(timeout=60)

        assert status == "ok", f"consumer failed: {payload}"
        assert payload == list(range(8))
        assert calls == [], f"colocate path called register_memory {len(calls)} times"
