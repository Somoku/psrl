"""Manual check: the throughput cost of enabling LMCache on a no-reuse batch.

Runs a batch of unique prompts through one vLLM engine and compares total time
with LMCache off, eager stores, and lazy (batched) stores. The LMCache server
log reports how many chunks were stored and how long the stores took.

Examples:

    CUDA_VISIBLE_DEVICES=0 python tests/lmcache/manual_overhead.py
    LMCACHE_MODE=lazy CUDA_VISIBLE_DEVICES=0 python tests/lmcache/manual_overhead.py
    LMCACHE_MODE=eager LMCACHE_PIN=all CUDA_VISIBLE_DEVICES=0 python tests/lmcache/manual_overhead.py

`SLEEP_MODE=0` runs the zero-copy path on exportable KV memory; the default
runs vLLM sleep mode, which forces the engine-driven copy path.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import socket
import tempfile
import time
from pathlib import Path

import psrl
from psrl.utils.kv_cache.config import LMCacheConfig
from psrl.utils.kv_cache.runtime import LMCacheMPRuntime, start_coordinator

REPO = Path(psrl.__file__).resolve().parents[1]
MODEL = os.environ.get("PSRL_LMCACHE_TEST_MODEL", str(REPO / "models/Qwen2.5-0.5B-Instruct"))
MODE = os.environ.get("LMCACHE_MODE", "off").lower()
PIN = os.environ.get("LMCACHE_PIN", "off").lower()
L1_GB = float(os.environ.get("LMCACHE_L1_GB", "8"))
INIT_GB = int(os.environ.get("LMCACHE_INIT_GB", "1"))
SLEEP_MODE = os.environ.get("SLEEP_MODE", "1") == "1"
NUM_PROMPTS = int(os.environ.get("NUM_PROMPTS", "64"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "64"))


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def prompts() -> list[str]:
    """Unique prefixes, so no request can reuse another's cache."""
    return [
        f"Case {i}: {' '.join(str((i * 7919 + j) % 100003) for j in range(180))}. Reply with the last number."
        for i in range(NUM_PROMPTS)
    ]


def summarize_server_log(path: str) -> dict:
    stores = []
    if os.path.exists(path):
        for line in Path(path).read_text(errors="ignore").splitlines():
            match = re.search(r"Stored (\d+) tokens in ([0-9.]+) seconds", line)
            if match:
                stores.append((int(match.group(1)), float(match.group(2))))
    return {
        "store_batches": len(stores),
        "stored_tokens": sum(tokens for tokens, _ in stores),
        "store_seconds_total": round(sum(seconds for _, seconds in stores), 3),
        "store_seconds_max": round(max((seconds for _, seconds in stores), default=0.0), 3),
    }


async def run() -> None:
    log_dir = tempfile.mkdtemp(prefix=f"lmcache_overhead_{MODE}_")
    server_log = os.path.join(log_dir, "server.log")
    coordinator = runtime = None
    engine_kwargs = {}
    if MODE != "off":
        config = LMCacheConfig(
            enable=True,
            offload_size_gb=L1_GB,
            l1_init_size_gb=INIT_GB,
            pin_policy=PIN,
            mp_transfer_mode="engine_driven" if SLEEP_MODE else "lmcache_driven",
            server_host="127.0.0.1",
            server_port=free_port(),
            http_port=free_port(),
            coordinator_host="127.0.0.1",
            coordinator_port=free_port(),
            lmcache_instance_id="overhead",
            replica_id="overhead",
            coordinator_health_timeout_s=180.0,
        )
        coordinator = await start_coordinator(config, os.path.join(log_dir, "coordinator.log"))
        runtime = LMCacheMPRuntime(config)
        await runtime.start(server_log)
        engine_kwargs = config.to_engine_kwargs()
        if MODE == "lazy":
            extra = engine_kwargs["kv_transfer_config"]["kv_connector_extra_config"]
            extra.update(
                {
                    "lmcache.mp.lazy_offload": True,
                    "lmcache.mp.lazy_offload_threshold": int(os.environ.get("LMCACHE_LAZY_THRESHOLD", "4")),
                    "lmcache.mp.lazy_offload_select_count": int(os.environ.get("LMCACHE_LAZY_SELECT", "8")),
                }
            )
    try:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=MODEL,
            enforce_eager=False,
            enable_sleep_mode=SLEEP_MODE,
            gpu_memory_utilization=0.6,
            max_model_len=2048,
            max_num_seqs=NUM_PROMPTS,
            **engine_kwargs,
        )
        sampling = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
        batch = prompts()
        # Warm up the kernels and, for LMCache, the server connection.
        llm.generate(batch[:1], sampling)
        start = time.perf_counter()
        llm.generate(batch, sampling)
        elapsed = time.perf_counter() - start
        print(
            json.dumps(
                {
                    "mode": MODE,
                    "pin": PIN,
                    "sleep_mode": SLEEP_MODE,
                    "l1_gb": L1_GB,
                    "prompts": NUM_PROMPTS,
                    "max_tokens": MAX_TOKENS,
                    "generate_seconds": round(elapsed, 3),
                    "prompts_per_second": round(NUM_PROMPTS / elapsed, 2),
                    "tokens_per_second": round(NUM_PROMPTS * MAX_TOKENS / elapsed, 2),
                }
            )
        )
        time.sleep(3)
        print(json.dumps(summarize_server_log(server_log)))
        print(f"logs={log_dir}")
    finally:
        if runtime is not None:
            runtime.stop()
        if coordinator is not None:
            coordinator.terminate()


if __name__ == "__main__":
    asyncio.run(run())
