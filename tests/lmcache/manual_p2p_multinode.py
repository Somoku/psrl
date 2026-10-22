"""Manual check: cross-node LMCache P2P transfer over NIXL.

Runs on an existing Ray cluster with at least two free GPUs on two nodes:

    RAY_ADDRESS=<head>:6399 python tests/lmcache/manual_p2p_multinode.py

The driver pins server A and engine A to the first node and server B and
engine B to the second. Engine A stores a prefix; the driver asks server B to
warm its L1 from A; engine B then generates the same prompt. Server B's log
must report the warm prefetch and the retrieved token count.

Node selection comes from `PSRL_LMCACHE_NODES` (comma-separated IPs). Without
it, the first two alive nodes in Ray's node table are used.
"""

from __future__ import annotations

import os
import socket
import sys
import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from psrl.utils.kv_cache.config import LMCacheConfig
from psrl.utils.kv_cache.manager import KVCacheManager
from psrl.utils.kv_cache.runtime import LMCacheMPRuntime, start_coordinator

MODEL = os.environ.get(
    "PSRL_LMCACHE_TEST_MODEL",
    "/apdcephfs_zwfy10/share_303541817/ls/psrl-wip/psrl/models/Qwen2.5-0.5B-Instruct",
)
PROMPT = "Remember these values: " + ", ".join(str(i) for i in range(250)) + ". What is 42?"
COLD_PROMPT = "Different values: " + ", ".join(str(i + 13) for i in range(250)) + ". What is 42?"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("0.0.0.0", 0))
        return sock.getsockname()[1]


def config_dict(instance: str, node_ip: str, coordinator_host: str, coordinator_port: int) -> dict:
    return {
        "enable": True,
        "offload_size_gb": 8.0,
        "l1_init_size_gb": 1,
        "l1_align_bytes": 65536,
        "server_host": "0.0.0.0",
        "server_port": free_port(),
        "http_host": "0.0.0.0",
        "http_port": free_port(),
        "http_advertise_host": node_ip,
        "lmcache_instance_id": instance,
        "replica_id": instance,
        "enable_p2p": True,
        "p2p_advertise_host": node_ip,
        "p2p_transfer_port": free_port(),
        "p2p_transfer_engine": "nixl",
        "coordinator_host": coordinator_host,
        "coordinator_port": coordinator_port,
        "coordinator_health_timeout_s": 300.0,
    }


@ray.remote(num_cpus=1)
class CoordinatorActor:
    def __init__(self) -> None:
        self._loop = None
        self._process = None

    def start(self, node_ip: str) -> int:
        import asyncio

        port = free_port()
        config = LMCacheConfig(**config_dict("coordinator_probe", node_ip, node_ip, port))
        self._loop = asyncio.new_event_loop()
        self._process = self._loop.run_until_complete(start_coordinator(config, "/tmp/lmcache_p2p_coordinator.log"))
        return port

    def stop(self) -> None:
        if self._process is not None:
            self._process.terminate()
            self._process = None


@ray.remote(num_cpus=1)
class ServerActor:
    def __init__(self) -> None:
        self._loop = None
        self._runtime = None
        self.config = None
        self.log_path = ""

    def start(self, instance: str, node_ip: str, coordinator_host: str, coordinator_port: int) -> dict:
        import asyncio

        config = LMCacheConfig(**config_dict(instance, node_ip, coordinator_host, coordinator_port))
        self.config = config
        self.log_path = f"/tmp/lmcache_p2p_{instance}.log"
        runtime = LMCacheMPRuntime(config)
        asyncio.run(runtime.start(self.log_path))
        # Keep the loop alive for later stop() by storing the runtime only; the
        # subprocess does not depend on the temporary loop.
        self._runtime = runtime
        return {field: getattr(config, field) for field in config.__dataclass_fields__}

    def stop(self) -> None:
        if self._runtime is not None:
            self._runtime.stop()
            self._runtime = None

    def tail_log(self, pattern: str) -> list[str]:
        matches = []
        try:
            with open(self.log_path, errors="ignore") as handle:
                for line in handle:
                    if pattern in line:
                        matches.append(line.strip())
        except FileNotFoundError:
            return []
        return matches[-8:]


@ray.remote(num_gpus=1)
def run_engine(config_data: dict, prompt: str) -> dict:
    from vllm import LLM, SamplingParams

    config = LMCacheConfig(**config_data)
    llm = LLM(
        model=MODEL,
        enforce_eager=True,
        gpu_memory_utilization=0.2,
        max_model_len=2048,
        max_num_seqs=4,
        enable_prefix_caching=False,
        **config.to_engine_kwargs(),
    )
    start = time.perf_counter()
    out = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=8))
    return {
        "elapsed_s": time.perf_counter() - start,
        "text": out[0].outputs[0].text,
        "tokens": list(out[0].prompt_token_ids),
    }


def node_ids() -> tuple[str, str, str, str]:
    nodes = {
        node["NodeManagerAddress"]: node["NodeID"]
        for node in ray.nodes()
        if node["Alive"] and node["Resources"].get("GPU", 0) >= 1
    }
    preferred = [ip for ip in os.environ.get("PSRL_LMCACHE_NODES", "").split(",") if ip]
    ordered = [ip for ip in preferred if ip in nodes] or sorted(nodes)
    if len(ordered) < 2:
        raise SystemExit(f"need two GPU nodes, found {sorted(nodes)}")
    node_a, node_b = ordered[0], ordered[1]
    return node_a, nodes[node_a], node_b, nodes[node_b]


def pin(node_id: str):
    return {"scheduling_strategy": NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)}


def main() -> None:
    ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR")
    ip_a, id_a, ip_b, id_b = node_ids()
    print(f"nodes: A={ip_a} B={ip_b}")

    coordinator = CoordinatorActor.options(**pin(id_a)).remote()
    coordinator_port = ray.get(coordinator.start.remote(ip_a))
    server_a = ServerActor.options(**pin(id_a)).remote()
    server_b = ServerActor.options(**pin(id_b)).remote()
    cfg_a = ray.get(server_a.start.remote("p2p_a", ip_a, ip_a, coordinator_port))
    cfg_b = ray.get(server_b.start.remote("p2p_b", ip_b, ip_a, coordinator_port))
    try:
        stored = ray.get(run_engine.options(**pin(id_a)).remote(cfg_a, PROMPT))
        tokens = stored["tokens"]
        print(f"engine A generate_s={stored['elapsed_s']:.3f} tokens={len(tokens)}")

        warm = ray.get(run_engine.options(**pin(id_b)).remote(cfg_b, COLD_PROMPT))
        print(f"engine B cold generate_s={warm['elapsed_s']:.3f}")

        manager = KVCacheManager(LMCacheConfig(**cfg_b))
        manager.set_parallel_geometry(MODEL, 1)

        async def transfer() -> tuple[bool, str]:
            try:
                ok = await manager.transfer_direct(tokens, ("p2p_a", ""), ("p2p_b", ""), copy=True)
                return ok, manager._last_transfer_error
            finally:
                await manager.close()

        start = time.perf_counter()
        ok, error = asyncio_run(transfer())
        print(f"transfer ok={ok} elapsed_s={time.perf_counter() - start:.3f} error={error!r}")

        fetched = ray.get(run_engine.options(**pin(id_b)).remote(cfg_b, PROMPT))
        print(f"engine B warm generate_s={fetched['elapsed_s']:.3f} text={fetched['text']!r}")
        print(f"warm/cold speedup={warm['elapsed_s'] / max(fetched['elapsed_s'], 1e-9):.2f}x")
        print("server B warm prefetch:", ray.get(server_b.tail_log.remote("Warm prefetch")))
        print("server B retrieved:", ray.get(server_b.tail_log.remote("Retrieved ")))
        assert ok, "P2P transfer failed"
    finally:
        ray.get(server_a.stop.remote())
        ray.get(server_b.stop.remote())
        ray.get(coordinator.stop.remote())


def asyncio_run(coro):
    import asyncio

    return asyncio.new_event_loop().run_until_complete(coro)


if __name__ == "__main__":
    sys.exit(main())
