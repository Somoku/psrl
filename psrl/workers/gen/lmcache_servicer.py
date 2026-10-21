"""SMG engine servicer that adds the LMCache off-GPU tier to KV events.

The router's cache-aware policy scores GPU-resident blocks and off-GPU blocks
through one positional index, fed by the servicer's `SubscribeKvEvents` stream.
vLLM reports only its own blocks, so this servicer merges the node's LMCache
event stream into the same gRPC stream, rewriting sequence numbers so the two
sources form one monotonically increasing sequence (the router drops any batch
whose number is not greater than the last one it saw).

LMCache events are emitted once per node, on data-parallel rank 0, because the
node's MP server owns the KV of every local rank.
"""

from __future__ import annotations

import asyncio
import logging

from smg_grpc_servicer.vllm.servicer import VllmEngineServicer

from psrl.workers.gen.lmcache_kv_events import LMCacheKvEventBridge

psrl_logger = logging.getLogger(__file__)

# Bounded so a slow router applies backpressure to the converters instead of
# growing an unbounded queue.
_QUEUE_SIZE = 256


class _MetadataGate:
    """Context proxy that records when the wrapped generator sent metadata.

    The vLLM event generator sends initial metadata itself. The LMCache
    producer must not produce a batch before that call, or gRPC would send
    metadata implicitly and the later explicit call would fail.
    """

    def __init__(self, context, sent: asyncio.Event) -> None:
        self._context = context
        self._sent = sent

    async def send_initial_metadata(self, metadata) -> None:
        self._sent.set()
        await self._context.send_initial_metadata(metadata)

    def __getattr__(self, name):
        return getattr(self._context, name)


class PSRLVllmEngineServicer(VllmEngineServicer):
    """`VllmEngineServicer` that also forwards the node's LMCache events."""

    def __init__(self, *args, lmcache_bridge: LMCacheKvEventBridge | None = None, **kwargs) -> None:
        """
        Initialize the servicer.

        Args:
            *args: Positional arguments forwarded to `VllmEngineServicer`.
            lmcache_bridge: Bridge for this node's off-GPU events, or None to
                serve only the native vLLM stream.
            **kwargs: Keyword arguments forwarded to `VllmEngineServicer`.
        """
        super().__init__(*args, **kwargs)
        self._lmcache_bridge = lmcache_bridge

    async def SubscribeKvEvents(self, request, context):
        """Stream GPU and off-GPU cache events as one monotonic sequence."""
        if self._lmcache_bridge is None or (request.HasField("dp_rank") and request.dp_rank != 0):
            async for batch in super().SubscribeKvEvents(request, context):
                yield batch
            return

        dp_rank = request.dp_rank if request.HasField("dp_rank") else 0
        queue: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_SIZE)
        metadata_sent = asyncio.Event()
        vllm_done = asyncio.Event()
        gate = _MetadataGate(context, metadata_sent)

        async def pump_vllm() -> None:
            try:
                async for batch in VllmEngineServicer.SubscribeKvEvents(self, request, gate):
                    await queue.put(("batch", batch))
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - the client sees the gRPC status
                await queue.put(("vllm_error", e))
            finally:
                vllm_done.set()
                await queue.put(("vllm_done", None))

        async def pump_lmcache() -> None:
            try:
                # Wait for the native generator to establish the stream. If it
                # ends first (for example, events were not enabled), do not
                # emit an orphan stream the router cannot correlate.
                while not metadata_sent.is_set():
                    if vllm_done.is_set():
                        return
                    try:
                        await asyncio.wait_for(metadata_sent.wait(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue
                async for batch in self._lmcache_bridge.stream(context.cancelled):
                    await queue.put(("batch", batch))
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - advisory stream, never fail the RPC
                psrl_logger.warning(f"[LMCache] KV event bridge stopped: {e}")
            finally:
                await queue.put(("lmcache_done", None))

        tasks = [asyncio.create_task(pump_vllm()), asyncio.create_task(pump_lmcache())]
        sequence = 0
        vllm_finished = False
        lmcache_finished = False
        try:
            while not (vllm_finished and lmcache_finished):
                kind, item = await queue.get()
                if kind == "batch":
                    sequence += 1
                    item.sequence_number = sequence
                    item.dp_rank = dp_rank
                    yield item
                    continue
                if kind == "vllm_error":
                    psrl_logger.warning(f"[KVEvents] Native stream ended: {item}")
                    continue
                if kind == "lmcache_done":
                    lmcache_finished = True
                    tasks[1] = None
                    continue
                if kind == "vllm_done":
                    vllm_finished = True
                    tasks[0] = None
                    continue
                raise AssertionError(f"Unknown event queue item {kind!r}")
        finally:
            for task in tasks:
                if task is not None:
                    task.cancel()
