"""How much a container-create burst starves the operations sharing the node.

The training loop admits a whole batch of sandboxes at one step boundary, and every
operation that shares the node's daemon, kernel locks, and disk queues behind those
creates. That queue is what turns a bounded command into a failed one, so these are
the two numbers that decide what to do about it:

* create latency as a burst grows, which is the supply curve the burst is queued against
* exec latency on an *existing* container while that burst is in flight, which is the
  starvation an unrelated operation actually experiences

Both matter on their own. A create curve that flattens says the daemon is saturated and
the fix is to stop asking for so many at once. An exec latency that rises with the burst
says the fix is to keep creates off the path of everything else, whatever the curve.

Usage:
    python tests/sandbox/bench_create_contention.py
    python tests/sandbox/bench_create_contention.py --burst 32 --network host
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

DOCKER = ["docker"]


def _run(args: list[str]) -> float:
    """Run one docker command and return its wall time in milliseconds."""
    started = time.monotonic()
    subprocess.run([*DOCKER, *args], capture_output=True, check=False)
    return (time.monotonic() - started) * 1000


def _percentile(samples: list[float], fraction: float) -> float:
    if not samples:
        return float("nan")
    ordered = sorted(samples)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _summarise(samples: list[float]) -> str:
    return (
        f"p50={_percentile(samples, 0.50):7.1f}ms  p95={_percentile(samples, 0.95):7.1f}ms  max={max(samples):7.1f}ms"
    )


def create_burst(image: str, burst: int, network: str | None) -> list[float]:
    """Create and start `burst` containers at once, and time each one."""
    extra = ["--network", network] if network else []

    def one(index: int) -> float:
        # `--rm` tears the container down too, which both modes pay equally.
        return _run(["run", "--rm", "--name", f"bench-{index}", *extra, image, "true"])

    with ThreadPoolExecutor(max_workers=burst) as pool:
        return list(pool.map(one, range(burst)))


def exec_latency(target: str, samples: int) -> list[float]:
    """Time `docker exec` on one existing container."""
    return [_run(["exec", target, "true"]) for _ in range(samples)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", default="python:3.12-slim")
    parser.add_argument("--burst", type=int, default=32)
    parser.add_argument("--bursts", type=int, default=2, help="How many create bursts to run.")
    parser.add_argument("--exec-samples", type=int, default=15)
    parser.add_argument("--network", default=None, help="Docker network mode for the burst, if any.")
    args = parser.parse_args()

    idle = _run(["run", "-d", "--rm", "--name", "bench-idle", args.image, "sleep", "900"])
    print(f"idle container started in {idle:.0f}ms")
    try:
        print("\n== exec on an existing container, no burst ==")
        quiet = exec_latency("bench-idle", args.exec_samples)
        print("   " + _summarise(quiet))

        for burst_index in range(args.bursts):
            print(f"\n== create burst {burst_index + 1}: {args.burst} concurrent creates ==")
            created: list[list[float]] = []
            burst = threading.Thread(
                target=lambda sink=created: sink.append(create_burst(args.image, args.burst, args.network))
            )
            burst.start()
            # Sample exec while the burst is in flight, which is the whole point: this is
            # what a probe or a file write experiences at a step boundary.
            loaded = exec_latency("bench-idle", args.exec_samples)
            burst.join()
            create_samples = created[0]
            print("   create  " + _summarise(create_samples))
            print("   exec    " + _summarise(loaded))
            slowdown = _percentile(loaded, 0.50) / max(_percentile(quiet, 0.50), 0.001)
            print(f"   exec p50 is {slowdown:.2f}x its idle value")
    finally:
        subprocess.run([*DOCKER, "rm", "-f", "bench-idle"], capture_output=True, check=False)

    print(f"\nexec idle p50 across the run: {statistics.median(quiet):.1f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
