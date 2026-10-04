"""Concurrency benchmark: p3b service vs Python sandbox theoretical throughput.

Measures create+exec+release latency under increasing concurrency levels,
then compares to the Python sandbox's per-worker envelope ceiling.
"""
import asyncio
import statistics
import sys
import time

from sandboxd import Resources, SandboxClient, SandboxSpec, Source

ENDPOINT = "unix:///run/sandboxd.sock"
IMAGE = "alpine:latest"
MEM_MB = 256   # 256 MB per sandbox — same as psrl live test default
CONCURRENCY_LEVELS = [8, 16, 32, 64]


async def one_sandbox(client: SandboxClient, idx: int) -> float:
    """Create, exec, release one sandbox. Return wall-clock seconds for create."""
    t0 = time.perf_counter()
    spec = SandboxSpec(
        source=Source.image(IMAGE),
        resources=Resources(memory_mb=MEM_MB, cpu_count=1),
        resource_class="rollout",
    )
    async with await client.create(spec) as sb:
        create_ms = (time.perf_counter() - t0) * 1000
        result = await sb.exec("echo bench-ok")
        assert result.exit_code == 0 and "bench-ok" in result.stdout
    return create_ms


async def run_level(n: int) -> dict:
    """Create n sandboxes concurrently, each running one command."""
    client = SandboxClient(ENDPOINT)
    t_wall = time.perf_counter()
    times = await asyncio.gather(*[one_sandbox(client, i) for i in range(n)], return_exceptions=True)
    elapsed = time.perf_counter() - t_wall
    await client.close()

    ok = [t for t in times if isinstance(t, float)]
    errs = [t for t in times if not isinstance(t, float)]
    return {
        "n": n,
        "ok": len(ok),
        "errs": len(errs),
        "elapsed_s": round(elapsed, 2),
        "throughput": round(len(ok) / elapsed, 2) if elapsed > 0 else 0,
        "p50_ms": round(statistics.median(ok), 0) if ok else 0,
        "p95_ms": round(sorted(ok)[int(len(ok) * 0.95)] if len(ok) > 1 else ok[0], 0) if ok else 0,
    }


async def main():
    # Python sandbox theoretical capacity on this node (2265 GB, util=0.8, rollout=0.45)
    node_mem_mb = 2265 * 1024
    py_capacity = int(node_mem_mb * 0.8 * 0.45 / MEM_MB)
    print(f"\nPython sandbox per-worker capacity at {MEM_MB} MB/sandbox: {py_capacity} sandboxes")
    print(f"p3b fleet capacity at {MEM_MB} MB/sandbox: {int(960_000 * 0.45 / MEM_MB)} sandboxes (from deploy.json)\n")
    print(f"{'Concurrency':>12} {'OK/Total':>10} {'Elapsed':>9} {'Throughput':>12} {'p50 ms':>9} {'p95 ms':>9}")
    print("-" * 68)

    for n in CONCURRENCY_LEVELS:
        result = await run_level(n)
        print(f"{result['n']:>12} {result['ok']:>4}/{result['n']:<5} "
              f"{result['elapsed_s']:>8.1f}s {result['throughput']:>11.2f}/s "
              f"{result['p50_ms']:>8.0f}ms {result['p95_ms']:>8.0f}ms")

    print("\nNOTE: Create latency dominated by Docker container start (~1s).")
    print("p3b throughput advantage is in fleet-wide quota accounting across workers,")
    print("not in single-node throughput vs Python sandbox on the same node.")


asyncio.run(main())
