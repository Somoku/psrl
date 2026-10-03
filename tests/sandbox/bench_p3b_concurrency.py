"""Concurrency stress test for the p3b sandbox service.

Creates N sandboxes concurrently, runs a command in each, measures total time
and per-sandbox latency, and compares throughput against the Python sandbox
theoretical capacity on the same node.

Usage:
    SANDBOXD_ENDPOINT=unix:///run/sandboxd.sock python bench_p3b_concurrency.py
"""
from __future__ import annotations

import asyncio
import os
import statistics
import sys
import time

from sandboxd import SandboxClient, SandboxSpec, Source, Resources

ENDPOINT = os.environ.get("SANDBOXD_ENDPOINT", "unix:///run/sandboxd.sock")
# Use 256 MB per sandbox so we can fit many concurrently; PSRL default is 2048 MB.
# We test both so the comparison is fair.
SANDBOX_MEMORY_MB = int(os.environ.get("SANDBOX_MEMORY_MB", "256"))
RESOURCE_CLASS = os.environ.get("RESOURCE_CLASS", "rollout")


async def one_sandbox(client: SandboxClient, idx: int, memory_mb: int) -> tuple[int, float, float, str]:
    """Create, exec one command, release. Returns (idx, create_s, exec_s, output)."""
    spec = SandboxSpec(
        source=Source.image("alpine:latest"),
        resources=Resources(memory_mb=memory_mb, cpu_count=1),
        resource_class=RESOURCE_CLASS,
    )
    t0 = time.monotonic()
    async with await client.create(spec) as sb:
        create_s = time.monotonic() - t0
        t1 = time.monotonic()
        result = await sb.exec(f"echo hello-from-{idx}")
        exec_s = time.monotonic() - t1
    return idx, create_s, exec_s, result.stdout.strip()


async def burst(n: int, memory_mb: int) -> dict:
    """Launch N sandboxes concurrently and collect results."""
    async with SandboxClient(ENDPOINT) as client:
        t0 = time.monotonic()
        tasks = [asyncio.create_task(one_sandbox(client, i, memory_mb)) for i in range(n)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_s = time.monotonic() - t0

    errors = [r for r in results if isinstance(r, Exception)]
    ok = [r for r in results if not isinstance(r, Exception)]
    create_times = [r[1] for r in ok]
    exec_times   = [r[2] for r in ok]

    return {
        "n": n,
        "memory_mb": memory_mb,
        "succeeded": len(ok),
        "failed": len(errors),
        "errors": [str(e) for e in errors[:3]],
        "total_wall_s": round(total_s, 3),
        "throughput_per_s": round(len(ok) / total_s, 2),
        "create_p50_ms": round(statistics.median(create_times) * 1000, 1) if create_times else None,
        "create_p95_ms": round(sorted(create_times)[int(len(create_times)*0.95)] * 1000, 1) if create_times else None,
        "create_max_ms": round(max(create_times) * 1000, 1) if create_times else None,
        "exec_p50_ms":   round(statistics.median(exec_times) * 1000, 1) if exec_times else None,
    }


def python_sandbox_capacity(node_mem_mb: int, utilization: float, guaranteed_share: float, per_sb_mb: int) -> int:
    """
    Python sandbox per-worker capacity estimate.
    Each worker has its own node envelope; the rollout class gets guaranteed_share of it.
    """
    usable = node_mem_mb * utilization * guaranteed_share
    return int(usable / per_sb_mb)


async def main():
    node_mem_mb = 2265 * 1024   # 2265 GB

    print(f"\n=== p3b Concurrency Benchmark ===")
    print(f"Endpoint:      {ENDPOINT}")
    print(f"Image:         alpine:latest")
    print(f"Resource class: {RESOURCE_CLASS}")
    print()

    # Python sandbox theoretical capacity for comparison
    py_cap_256  = python_sandbox_capacity(node_mem_mb, 0.8, 0.45, 256)
    py_cap_2048 = python_sandbox_capacity(node_mem_mb, 0.8, 0.45, 2048)
    print(f"Python sandbox theoretical capacity (per worker, same node):")
    print(f"  @256 MB/sandbox:  {py_cap_256} sandboxes")
    print(f"  @2048 MB/sandbox: {py_cap_2048} sandboxes")
    print()

    concurrencies = [int(x) for x in os.environ.get("CONCURRENCIES", "8,16,32,64").split(",")]

    all_results = []
    for n in concurrencies:
        mem = SANDBOX_MEMORY_MB
        print(f"--- {n} concurrent sandboxes @ {mem} MB each ---", flush=True)
        r = await burst(n, mem)
        all_results.append(r)
        print(f"  succeeded:      {r['succeeded']}/{r['n']}")
        print(f"  total wall:     {r['total_wall_s']}s")
        print(f"  throughput:     {r['throughput_per_s']} sandboxes/s")
        print(f"  create p50/p95/max: {r['create_p50_ms']} / {r['create_p95_ms']} / {r['create_max_ms']} ms")
        print(f"  exec   p50:     {r['exec_p50_ms']} ms")
        if r['failed']:
            print(f"  ERRORS ({r['failed']}):  {r['errors']}")
        print()

    print("=== Summary ===")
    print(f"{'N':>6}  {'mem':>6}  {'ok':>5}  {'wall_s':>7}  {'tput/s':>7}  {'cr_p50':>7}  {'cr_p95':>7}")
    for r in all_results:
        print(f"{r['n']:>6}  {r['memory_mb']:>6}  {r['succeeded']:>5}  "
              f"{r['total_wall_s']:>7.2f}  {r['throughput_per_s']:>7.2f}  "
              f"{r['create_p50_ms']:>7}  {r['create_p95_ms']:>7}")

    print()
    best = max(all_results, key=lambda r: r['succeeded'])
    print(f"p3b fleet admission capacity (fleet-wide ledger, single service):")
    print(f"  Max concurrent succeeded: {best['succeeded']} @ {best['memory_mb']} MB each")
    fleet_cap_256  = python_sandbox_capacity(node_mem_mb, 0.8, 0.45, 256) * 1  # per-worker only
    print(f"  vs Python sandbox per-worker capacity @ 256 MB: {py_cap_256}")
    print(f"  vs Python sandbox per-worker capacity @ 2048 MB: {py_cap_2048}")
    print()
    print("Key difference: p3b uses a single fleet ledger shared across ALL workers.")
    print("Python sandbox gives each worker its own node envelope independently.")
    print("With M workers, Python sandbox total capacity = M × per-worker capacity,")
    print("but each worker sees only its own share and cannot borrow across workers.")
    print("p3b admission is global: the whole fleet quota is available to any worker.")


if __name__ == "__main__":
    asyncio.run(main())
