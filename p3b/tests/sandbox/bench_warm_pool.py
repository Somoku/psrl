"""Measure what the warm pool actually buys on a deployed node.

Two batches of the same size are created against the same node. One matches the pool's
template exactly, so its first `size` claims are served from inventory and the rest are
cold; the other differs in one field a shared entry cannot have, so every create is cold.
The difference between the two batches is the pool.

Run it against a control plane whose node has a `warm_pool` block configured:

    python tests/sandbox/bench_warm_pool.py 28.49.198.250:9000 --batches 16 --pool-size 8

It prints one JSON line per batch and a manifest, so a result can be tied to the
deployment it was measured on rather than to a paragraph.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

from sandboxd import ExecMode, Resources, SandboxClient, SandboxSpec, Source


def template_spec(image: str, memory_mb: int, cpu: float, lifetime_s: float) -> SandboxSpec:
    """The spec the pool's template declares, exactly."""
    return SandboxSpec(
        source=Source.image(image),
        resources=Resources(memory_mb=memory_mb, cpu_count=cpu),
        resource_class="rollout",
        exec_mode=ExecMode.ONE_SHOT,
        lifetime_s=lifetime_s,
    )


def cold_spec(image: str, memory_mb: int, cpu: float, lifetime_s: float) -> SandboxSpec:
    """A spec the pool cannot serve: one dimension differs, so every create must build."""
    return SandboxSpec(
        source=Source.image(image),
        resources=Resources(memory_mb=memory_mb + 128, cpu_count=cpu),
        resource_class="rollout",
        exec_mode=ExecMode.ONE_SHOT,
        lifetime_s=lifetime_s,
    )


async def batch(client: SandboxClient, make_spec, count: int) -> dict:
    """Create `count` sandboxes concurrently and report each create's latency."""
    latencies: list[float] = []
    sandboxes = []

    async def one() -> None:
        spec = make_spec()
        started = time.perf_counter()
        sandbox = await client.create(spec)
        latencies.append(time.perf_counter() - started)
        sandboxes.append(sandbox)

    await asyncio.gather(*(one() for _ in range(count)))
    for sandbox in sandboxes:
        try:
            await sandbox.release()
        except Exception:  # noqa: BLE001 - a release failure is reported by the node
            pass
    latencies.sort()
    return {
        "count": len(latencies),
        "p50_s": round(statistics.median(latencies), 4),
        "p95_s": round(latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))], 4),
        "min_s": round(latencies[0], 4),
        "max_s": round(latencies[-1], 4),
        "total_s": round(sum(latencies), 3),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("endpoint")
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--pool-size", type=int, default=8)
    parser.add_argument("--image", default="alpine:latest")
    parser.add_argument("--memory-mb", type=int, default=512)
    parser.add_argument("--cpu", type=float, default=1.0)
    parser.add_argument("--lifetime-s", type=float, default=7200)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    manifest = {
        "endpoint": args.endpoint,
        "image": args.image,
        "memory_mb": args.memory_mb,
        "cpu": args.cpu,
        "lifetime_s": args.lifetime_s,
        "pool_size": args.pool_size,
        "batches": args.batches,
        "repeat": args.repeat,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    print(json.dumps({"manifest": manifest}, sort_keys=True))

    client = SandboxClient(args.endpoint)
    try:
        for run in range(args.repeat):
            # Alternate the order so a warming daemon does not favour one batch.
            first, second = ("cold", "warm") if run % 2 else ("warm", "cold")
            for kind in (first, second):
                make = (
                    (lambda: template_spec(args.image, args.memory_mb, args.cpu, args.lifetime_s))
                    if kind == "warm"
                    else (lambda: cold_spec(args.image, args.memory_mb, args.cpu, args.lifetime_s))
                )
                result = await batch(client, make, args.batches)
                print(json.dumps({"run": run, "kind": kind, **result}, sort_keys=True))
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
