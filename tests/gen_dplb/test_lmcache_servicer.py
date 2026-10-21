"""Merging the native vLLM event stream and the LMCache off-GPU stream."""

import asyncio
from types import SimpleNamespace

import pytest
from psrl.workers.gen.lmcache_servicer import PSRLVllmEngineServicer
from smg_grpc_proto.generated import common_pb2
from smg_grpc_servicer.vllm.servicer import VllmEngineServicer


class _FakeRequest:
    def HasField(self, name: str) -> bool:
        return False

    dp_rank = 0


class _FakeContext:
    def __init__(self) -> None:
        self.metadata_sent = False

    async def send_initial_metadata(self, metadata) -> None:
        self.metadata_sent = True

    def cancelled(self) -> bool:
        return False


class _FakeBridge:
    async def stream(self, is_cancelled):
        yield common_pb2.KvEventBatch(sequence_number=99, timestamp=0.0)


def _fake_engine() -> SimpleNamespace:
    vllm_config = SimpleNamespace(
        kv_events_config=None,
        parallel_config=SimpleNamespace(data_parallel_size=1),
    )
    return SimpleNamespace(vllm_config=vllm_config)


@pytest.mark.asyncio
async def test_both_sources_share_one_monotonic_sequence(monkeypatch):
    async def fake_subscribe(self, request, context):
        await context.send_initial_metadata(())
        yield common_pb2.KvEventBatch(sequence_number=1, timestamp=0.0)
        yield common_pb2.KvEventBatch(sequence_number=2, timestamp=0.0)

    monkeypatch.setattr(VllmEngineServicer, "SubscribeKvEvents", fake_subscribe)
    servicer = PSRLVllmEngineServicer(_fake_engine(), 0.0, lmcache_bridge=_FakeBridge())

    batches = [batch async for batch in servicer.SubscribeKvEvents(_FakeRequest(), _FakeContext())]

    assert [batch.sequence_number for batch in batches] == [1, 2, 3]
    assert all(batch.dp_rank == 0 for batch in batches)


@pytest.mark.asyncio
async def test_without_a_bridge_only_the_native_stream_is_served(monkeypatch):
    async def fake_subscribe(self, request, context):
        yield common_pb2.KvEventBatch(sequence_number=7, timestamp=0.0)

    monkeypatch.setattr(VllmEngineServicer, "SubscribeKvEvents", fake_subscribe)
    servicer = PSRLVllmEngineServicer(_fake_engine(), 0.0, lmcache_bridge=None)

    batches = [batch async for batch in servicer.SubscribeKvEvents(_FakeRequest(), _FakeContext())]

    assert [batch.sequence_number for batch in batches] == [7]
