"""Small repeatable latency and density benchmark for the Docker backend.

It measures the three phases an episode pays for, one at a time, and projects how
many sandboxes one node's envelope can hold from the footprint it actually
observed. The projection is a floor over a run, not a capacity plan: it exists so
a change to the exec model, the image path, or the envelope has something to be
compared against.

Run it on a node with a working daemon:

    python -m tests.sandbox.benchmark_docker_backend --iterations 20 --out baseline.json

Then compare a later run against a recorded one:

    python -m tests.sandbox.benchmark_docker_backend --baseline baseline.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
import uuid
from pathlib import Path

from psrl.sandbox import SandboxManager, SandboxSource, SandboxSpec
from psrl.sandbox.backends.docker import DockerBackend
from psrl.sandbox.capacity import SandboxCapacityConfig, resolve_sandbox_capacity


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def _summarize(samples: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(samples),
        "p50": _percentile(samples, 0.50),
        "p95": _percentile(samples, 0.95),
        "max": max(samples),
    }


async def _benchmark(image: str, iterations: int, concurrency: int, *, command: str) -> dict:
    backend = DockerBackend(auto_pull=True)
    manager = SandboxManager({"docker": backend}, "docker")
    gate = asyncio.Semaphore(concurrency)

    async def run_one(index: int) -> tuple[float, float, float, int]:
        async with gate:
            started = time.perf_counter()
            lease = await manager.acquire(
                SandboxSpec(
                    SandboxSource.image(image),
                    idempotency_key=f"benchmark-{uuid.uuid4().hex}-{index}",
                    # A lifetime, not an idle window: a stuck benchmark still has to end.
                    lifetime_timeout_s=300,
                )
            )
            created = time.perf_counter()
            result = await lease.session.exec(command)
            executed = time.perf_counter()
            if result.stdout != "ok":
                raise RuntimeError(f"Unexpected Docker output: {result!r}.")
            usage = await lease.session.stats()
            await lease.release()
            released = time.perf_counter()
            return created - started, executed - created, released - executed, usage.peak_memory_bytes

    try:
        samples = await asyncio.gather(*(run_one(index) for index in range(iterations)))
    finally:
        await manager.shutdown()
    creates = [sample[0] for sample in samples]
    execs = [sample[1] for sample in samples]
    releases = [sample[2] for sample in samples]
    peak_mib = [sample[3] / (1024 * 1024) for sample in samples]
    envelope = resolve_sandbox_capacity(SandboxCapacityConfig())
    worst_peak = max(peak_mib)
    # A floor over the observed footprint, which is what makes the number comparable
    # between runs rather than a claim about a specific workload.
    density = {
        "memory_limited": math.floor(envelope.resources.memory_mb / worst_peak) if worst_peak > 0 else None,
        "envelope_memory_mb": envelope.resources.memory_mb,
        "envelope_cpu_cores": envelope.resources.cpu_millis / 1000,
        "peak_memory_mib_per_sandbox_max": worst_peak,
    }
    return {
        "iterations": iterations,
        "concurrency": concurrency,
        "create_s": _summarize(creates),
        "exec_s": _summarize(execs),
        "release_s": _summarize(releases),
        "density": density,
    }


def _compare(baseline: dict, current: dict) -> list[str]:
    """Return one line per metric that moved, worst regression first."""
    regressions: list[tuple[float, str]] = []
    for phase in ("create_s", "exec_s", "release_s"):
        for percentile in ("p50", "p95"):
            before = baseline.get(phase, {}).get(percentile)
            after = current.get(phase, {}).get(percentile)
            if not before or after is None:
                continue
            delta = (after - before) / before
            detail = f"{phase}.{percentile}: {before * 1000:.1f}ms -> {after * 1000:.1f}ms ({delta:+.1%})"
            regressions.append((delta, detail))
    before_density = baseline.get("density", {}).get("memory_limited")
    after_density = current.get("density", {}).get("memory_limited")
    if before_density and after_density:
        delta = (after_density - before_density) / before_density
        regressions.append((-delta, f"density.memory_limited: {before_density} -> {after_density} ({delta:+.1%})"))
    return [line for _, line in sorted(regressions, key=lambda item: -item[0])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="python:3.11-slim")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--command", default="printf ok")
    parser.add_argument("--out", type=Path, default=None, help="Write the measured baseline here.")
    parser.add_argument("--baseline", type=Path, default=None, help="Compare against a recorded baseline.")
    args = parser.parse_args()
    if args.iterations < 1 or args.concurrency < 1:
        parser.error("--iterations and --concurrency must be positive")
    result = asyncio.run(_benchmark(args.image, args.iterations, args.concurrency, command=args.command))
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text())
        lines = _compare(baseline, result)
        print("Comparison, worst regression first:")
        print("\n".join(f"  {line}" for line in lines) if lines else "  no comparable metric changed")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
