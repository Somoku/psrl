"""Manual check: a peer warms its L1 over NIXL and then serves the migrated prefix.

Starts a coordinator and two MP servers with P2P enabled, stores a prefix from
an engine attached to server A, asks server B to prefetch it from A through the
NIXL transfer channel, then generates the same prompt from an engine attached
to B. Server B's log must report the warm prefetch and the retrieved token
count.

Run on one node with two free GPUs:

    PSRL_LMCACHE_TEST_GPUS=0,1 python tests/lmcache/manual_p2p.py

Both servers advertise the node IP, so the same script can be split across two
nodes later without changing the transfer code.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import socket
import sys
import tempfile
import time
from pathlib import Path

from psrl.utils.kv_cache.config import LMCacheConfig
from psrl.utils.kv_cache.manager import KVCacheManager
from psrl.utils.kv_cache.runtime import LMCacheMPRuntime, start_coordinator

REPO = Path(__file__).resolve().parents[2]
MODEL = os.environ.get("PSRL_LMCACHE_TEST_MODEL", str(REPO / "tests/nixl/models/Qwen2.5-0.5B-Instruct"))
_DEVICES = [int(part) for part in os.environ.get("PSRL_LMCACHE_TEST_GPUS", "0,1").split(",")]
_USED_PORTS: set[int] = set()


def free_port() -> int:
    while True:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        if port not in _USED_PORTS:
            _USED_PORTS.add(port)
            return port


def build_config(instance: str, coordinator_port: int, node_ip: str) -> LMCacheConfig:
    return LMCacheConfig(
        enable=True,
        offload_size_gb=8.0,
        l1_init_size_gb=1,
        l1_align_bytes=65536,
        server_host="0.0.0.0",
        server_port=free_port(),
        http_host="0.0.0.0",
        http_port=free_port(),
        http_advertise_host=node_ip,
        lmcache_instance_id=instance,
        replica_id=instance,
        enable_p2p=True,
        p2p_advertise_host=node_ip,
        p2p_transfer_port=free_port(),
        p2p_transfer_engine="nixl",
        coordinator_host=node_ip,
        coordinator_port=coordinator_port,
        coordinator_health_timeout_s=300.0,
    )


def engine_worker(device: int, config_dict: dict, prompt: str, queue) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    from vllm import LLM, SamplingParams

    config = LMCacheConfig(**config_dict)
    llm = LLM(
        model=MODEL,
        enforce_eager=True,
        gpu_memory_utilization=0.25,
        max_model_len=2048,
        max_num_seqs=4,
        enable_prefix_caching=False,
        **config.to_engine_kwargs(),
    )
    start = time.perf_counter()
    out = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=8))
    queue.put(
        {
            "elapsed_s": time.perf_counter() - start,
            "text": out[0].outputs[0].text,
            "tokens": list(out[0].prompt_token_ids),
        }
    )


def run_engine(device: int, config: LMCacheConfig, prompt: str) -> dict:
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=engine_worker,
        args=(device, {field: getattr(config, field) for field in config.__dataclass_fields__}, prompt, queue),
    )
    process.start()
    result = queue.get(timeout=900)
    process.join(timeout=120)
    return result


async def main() -> None:
    node_ip = socket.gethostbyname(socket.gethostname())
    coordinator_port = free_port()
    config_a = build_config("p2p_a", coordinator_port, node_ip)
    config_b = build_config("p2p_b", coordinator_port, node_ip)
    log_dir = tempfile.mkdtemp(prefix="lmcache_p2p_")
    coord = await start_coordinator(config_a, os.path.join(log_dir, "coordinator.log"))
    runtime_a = LMCacheMPRuntime(config_a)
    runtime_b = LMCacheMPRuntime(config_b)
    await runtime_a.start(os.path.join(log_dir, "server_a.log"))
    await runtime_b.start(os.path.join(log_dir, "server_b.log"))
    try:
        prompt = "Remember these values: " + ", ".join(str(i) for i in range(250)) + ". What is 42?"
        stored = run_engine(_DEVICES[0], config_a, prompt)
        tokens = stored["tokens"]
        print(f"engine A generate_s={stored['elapsed_s']:.3f} tokens={len(tokens)}")

        # A destination can only lay out and accept a prefetch once one of its
        # engines has registered its KV cache.
        warmup = run_engine(_DEVICES[1], config_b, "hello")
        print(f"engine B warmup_s={warmup['elapsed_s']:.3f}")

        manager = KVCacheManager(config_b)
        manager.set_parallel_geometry(MODEL, 1)
        start = time.perf_counter()
        ok = await manager.transfer_direct(tokens, ("p2p_a", ""), ("p2p_b", ""), copy=True)
        print(f"transfer ok={ok} elapsed_s={time.perf_counter() - start:.3f} error={manager._last_transfer_error!r}")

        fetched = run_engine(_DEVICES[1], config_b, prompt)
        print(f"engine B generate_s={fetched['elapsed_s']:.3f} text={fetched['text']!r}")
        await manager.close()
        print(f"logs={log_dir}")
        assert ok, "P2P transfer failed"
    finally:
        runtime_a.stop()
        runtime_b.stop()
        coord.terminate()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
