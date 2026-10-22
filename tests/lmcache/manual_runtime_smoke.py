"""Manual check: the MP runtime starts a coordinator and node server and serves its API.

This exercises port allocation, readiness polling, coordinator registration,
the management HTTP endpoints, and the off-GPU event stream without a GPU
engine. Logs go to a temporary directory printed at the end.

Run with no arguments:

    python tests/lmcache/manual_runtime_smoke.py
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import tempfile

import aiohttp

from psrl.utils.kv_cache.config import LMCacheConfig
from psrl.utils.kv_cache.runtime import LMCacheMPRuntime, start_coordinator


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def get(session: aiohttp.ClientSession, url: str) -> tuple[int, str]:
    async with session.get(url) as response:
        return response.status, (await response.text())[:400]


async def main() -> None:
    coordinator_port = free_port()
    config = LMCacheConfig(
        enable=True,
        offload_size_gb=2.0,
        l1_init_size_gb=1,
        server_host="127.0.0.1",
        server_port=free_port(),
        http_port=free_port(),
        coordinator_host="127.0.0.1",
        coordinator_port=coordinator_port,
        coordinator_event_reporting=True,
        enable_kv_events=True,
        lmcache_instance_id="smoke_instance",
        replica_id="smoke_replica",
        coordinator_health_timeout_s=180.0,
    )
    log_dir = tempfile.mkdtemp(prefix="lmcache_runtime_smoke_")
    coordinator = await start_coordinator(config, os.path.join(log_dir, "coordinator.log"))
    runtime = LMCacheMPRuntime(config)
    await runtime.start(os.path.join(log_dir, "server.log"))
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            endpoints = {
                "coordinator_instances": f"http://127.0.0.1:{coordinator_port}/instances",
                "server_health": f"{config.http_base_url}/healthcheck",
                "event_stats": f"{config.http_base_url}/cache/events/stats",
                "pin_stats": f"{config.http_base_url}/cache/l1/pins/stats",
            }
            for name, url in endpoints.items():
                status, body = await get(session, url)
                print(f"{name}: {status} {body}")
            for method, path in (("DELETE", "/cache/l1/pins/groups"), ("POST", "/cache/clear")):
                async with session.request(method, f"{config.http_base_url}{path}") as response:
                    print(f"{method} {path}: {response.status} {(await response.text())[:200]}")
            async with session.get(
                f"{config.http_base_url}/cache/events/stream",
                timeout=aiohttp.ClientTimeout(total=5, sock_read=3),
            ) as response:
                print(f"event_stream: {response.status} content-type={response.content_type}")
                first = await response.content.readline()
                print(f"event_stream first line: {json.loads(first) if first.strip() else None}")
    finally:
        runtime.stop()
        coordinator.terminate()
        print(f"logs={log_dir}")


if __name__ == "__main__":
    asyncio.run(main())
