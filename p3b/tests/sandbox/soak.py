#!/usr/bin/env python3
"""Hold an agent-shaped load for a while and watch for leaks.

A burst proves throughput; a soak proves the service does not accumulate state.
The load is the shape an agent episode makes -- one long-lived sandbox with
several short commands and a file read, then release -- repeated by a fixed
number of concurrent workers.

Every sample prints one JSON line with the fleet's live count, the control's
renewal counters, each node's sandboxd file-descriptor count, and the daemon's
container count. At the end the script asserts that all of them returned to
zero.

    python p3b/tests/sandbox/soak.py --endpoint 127.0.0.1:9200 \
        --nodes node-a=root@host-a,node-b=root@host-b \
        --image alpine:latest --seconds 900 --concurrency 128
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from collections import Counter

from sandboxd import Resources, SandboxClient, SandboxSpec, Source


def parse_nodes(value: str) -> dict[str, str]:
    return dict(item.split("=", 1) for item in value.split(","))


def ssh(target: str, command: str) -> str:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", target, command],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.stdout.strip()


def node_pid(target: str) -> str:
    """The node plane's pid, not the control plane's: both run the same binary."""
    return ssh(target, "pgrep -f 'sandbox[d] -config /tmp/p3b-node-' | head -1")


def fd_count(target: str, pid: str) -> int:
    if not pid:
        return -1
    out = ssh(target, f"ls /proc/{pid}/fd 2>/dev/null | wc -l")
    return int(out or 0)


def container_count(target: str) -> int:
    out = ssh(target, "docker ps -q | wc -l")
    return int(out or 0)


def spec(image: str) -> SandboxSpec:
    return SandboxSpec(
        source=Source.image(image),
        resources=Resources(memory_mb=256, cpu_count=1),
        resource_class="rollout",
        lifetime_s=900,
    )


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--image", default="alpine:latest")
    parser.add_argument("--seconds", type=int, default=900)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--sample-s", type=int, default=30)
    args = parser.parse_args()

    nodes = parse_nodes(args.nodes)
    pids = {name: node_pid(target) for name, target in nodes.items()}
    failures = 0
    completed = 0
    failure_kinds: Counter[str] = Counter()
    stop = asyncio.Event()
    client = SandboxClient(args.endpoint)

    async def worker(semaphore: asyncio.Semaphore) -> None:
        nonlocal failures, completed
        while not stop.is_set():
            async with semaphore:
                sandbox = None
                try:
                    sandbox = await client.create(spec(args.image))
                    for _ in range(4):
                        if (await sandbox.exec("echo soak")).exit_code != 0:
                            raise RuntimeError("exec failed")
                    await sandbox.read_bytes("/etc/hostname")
                    completed += 1
                except Exception as exc:  # noqa: BLE001 - counted and named
                    failures += 1
                    failure_kinds[f"{type(exc).__name__}: {str(exc)[:120]}"] += 1
                finally:
                    if sandbox is not None:
                        try:
                            await sandbox.release()
                        except Exception as exc:  # noqa: BLE001
                            failures += 1
                            failure_kinds[f"release:{type(exc).__name__}: {str(exc)[:120]}"] += 1

    async def sampler() -> None:
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline and not stop.is_set():
            await asyncio.sleep(args.sample_s)
            try:
                fleet = await client.fleet_detail()
                live = sum(int(n.get("live_sandboxes", 0)) for n in fleet["nodes"])
                diagnostics = await client.diagnostics()
            except Exception as exc:  # noqa: BLE001 - a sample failure is reported
                print(json.dumps({"sample": "error", "error": repr(exc)}))
                continue
            print(
                json.dumps(
                    {
                        "sample": "tick",
                        "elapsed_s": round(args.seconds - (deadline - time.monotonic()), 1),
                        "completed": completed,
                        "failures": failures,
                        "live": live,
                        "renew_failures": diagnostics["renew"]["failures"],
                        "stale_nodes": diagnostics["fleet"]["stale_nodes"],
                        "fds": {name: fd_count(target, pids[name]) for name, target in nodes.items()},
                        "containers": {
                            name: container_count(target) for name, target in nodes.items()
                        },
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [asyncio.create_task(worker(semaphore)) for _ in range(args.concurrency)]
    try:
        await asyncio.wait_for(sampler(), timeout=args.seconds + 60)
    except asyncio.TimeoutError:
        pass
    finally:
        stop.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # Drain: anything the load created must be releasable, and the fleet must
    # return to empty.
    for _ in range(30):
        fleet = await client.fleet_detail()
        live = sum(int(n.get("live_sandboxes", 0)) for n in fleet["nodes"])
        containers = sum(container_count(target) for target in nodes.values())
        if live == 0 and containers == 0:
            break
        await asyncio.sleep(2)
    fds = {name: fd_count(target, pids[name]) for name, target in nodes.items()}
    final = {
        "sample": "final",
        "completed": completed,
        "failures": failures,
        "failure_kinds": dict(failure_kinds),
        "live": live,
        "containers": containers,
        "fds": fds,
    }
    print(json.dumps(final, sort_keys=True))
    await client.close()
    return 0 if failures == 0 and live == 0 and containers == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
