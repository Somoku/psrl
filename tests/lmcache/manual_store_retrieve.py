"""Manual check: LMCache serves a repeated prefix from L1 instead of re-prefilling.

Starts a coordinator and one node MP server with PSRL's runtime, runs the same
long prompt twice against a real vLLM engine with vLLM's own prefix cache
disabled, and prints the two latencies. The MP server log reports the store and
retrieve byte counts.

Run on one free GPU:

    CUDA_VISIBLE_DEVICES=0 python tests/lmcache/manual_store_retrieve.py

Override the model with `PSRL_LMCACHE_TEST_MODEL`; the default is the small
checkpoint shipped under `tests/nixl/models`. Set `LMCACHE_CHUNK` to change the
chunk size. Set `LMCACHE_MP_TRANSFER_MODE=lmcache_driven` to compare the
zero-copy path on a deployment whose KV memory is exportable.
"""

from __future__ import annotations

import asyncio
import os
import socket
import sys
import tempfile
import time
from pathlib import Path

from psrl.utils.kv_cache.config import LMCacheConfig
from psrl.utils.kv_cache.runtime import LMCacheMPRuntime, start_coordinator

REPO = Path(__file__).resolve().parents[2]
MODEL = os.environ.get("PSRL_LMCACHE_TEST_MODEL", str(REPO / "tests/nixl/models/Qwen2.5-0.5B-Instruct"))


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def start_runtime(instance_id: str = "store_retrieve", log_dir: str = ""):
    config = LMCacheConfig(
        enable=True,
        offload_size_gb=4.0,
        l1_init_size_gb=1,
        chunk_size=int(os.environ.get("LMCACHE_CHUNK", "256")),
        mp_transfer_mode=os.environ.get("LMCACHE_MP_TRANSFER_MODE", "engine_driven"),
        server_host="127.0.0.1",
        server_port=free_port(),
        http_port=free_port(),
        coordinator_host="127.0.0.1",
        coordinator_port=free_port(),
        lmcache_instance_id=instance_id,
        replica_id=instance_id,
        coordinator_event_reporting=True,
        enable_kv_events=True,
        coordinator_health_timeout_s=180.0,
    )
    os.makedirs(log_dir, exist_ok=True)
    coord = await start_coordinator(config, os.path.join(log_dir, "coordinator.log"))
    runtime = LMCacheMPRuntime(config)
    await runtime.start(os.path.join(log_dir, "server.log"))
    return config, coord, runtime


def run_engine(config: LMCacheConfig) -> list[float]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        enforce_eager=True,
        gpu_memory_utilization=0.35,
        max_model_len=2048,
        max_num_seqs=4,
        enable_prefix_caching=False,
        **config.to_engine_kwargs(),
    )
    # About 1k tokens of unique text; the model only prefills it.
    prompt = (
        "You are a careful assistant. Read the following notes and answer the question.\n\n"
        + "\n".join(f"Note {i}: the value of item {i} is {i * 7 % 101}." for i in range(100))
        + "\n\nQuestion: what is the value of item 42? Answer with the number."
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=16)
    timings = []
    for _ in range(2):
        start = time.perf_counter()
        out = llm.generate([prompt], sampling)
        timings.append(time.perf_counter() - start)
        print(f"ttft_s={timings[-1]:.3f} text={out[0].outputs[0].text!r}")
    return timings


def main() -> None:
    log_dir = tempfile.mkdtemp(prefix="lmcache_store_retrieve_")
    config, coord, runtime = asyncio.run(start_runtime(log_dir=log_dir))
    try:
        cold, warm = run_engine(config)
        print(f"cold={cold:.3f}s warm={warm:.3f}s speedup={cold / warm:.2f}x logs={log_dir}")
    finally:
        runtime.stop()
        coord.terminate()


if __name__ == "__main__":
    sys.exit(main())
