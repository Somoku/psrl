#!/usr/bin/env python3
"""Exercise the sandbox service's failure semantics against a real deployment.

The unit suites pin each decision in isolation. This drives the same decisions
through a real Docker daemon, because the failures that cost a training run were
the ones that only appear when a container, a node process, and a caller's
deadline interact.

Run it from the repository root against a control plane:

    python p3b/tests/sandbox/fault_injection.py \
        --endpoint 127.0.0.1:9000 --node-ssh root@127.0.0.1 --image alpine:latest

Every case prints one JSON line with `case`, `ok`, and the measurements it made,
so a result can be compared across revisions rather than described in prose.
The script exits non-zero if any case failed. `--skip-reclamation` skips the
two-minute owner-silence case; `--concurrency` sizes the load case.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import sys
import time

from sandboxd import (
    Resources,
    SandboxClient,
    SandboxPathNotFound,
    SandboxSessionLost,
    SandboxSpec,
    SandboxStatus,
    Source,
)


def emit(case: str, ok: bool, **fields: object) -> bool:
    print(json.dumps({"case": case, "ok": ok, **fields}, sort_keys=True, default=str))
    return ok


def parse_nodes(value: str) -> dict[str, str]:
    """Parse `node-a=root@host,node-b=root@host`, or a single `root@host`."""
    if "=" not in value:
        return {"": value}
    return dict(item.split("=", 1) for item in value.split(","))


def ssh_for(nodes: dict[str, str], node_id: str) -> str:
    if node_id in nodes:
        return nodes[node_id]
    if "" in nodes:
        return nodes[""]
    return next(iter(nodes.values()))


def docker(node_ssh: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", node_ssh, "docker", *arguments],
        capture_output=True,
        text=True,
        timeout=60,
    )


def spec(image: str, *, owner: str | None = None, lifetime_s: float = 600) -> SandboxSpec:
    metadata = {"owner_id": owner} if owner else {}
    return SandboxSpec(
        source=Source.image(image),
        resources=Resources(memory_mb=256, cpu_count=1),
        resource_class="rollout",
        lifetime_s=lifetime_s,
        metadata=metadata,
    )


async def wait_for_live(client: SandboxClient, expected: int, deadline_s: float) -> int:
    """Poll the fleet until the node holds `expected` sandboxes, or time out."""
    deadline = time.monotonic() + deadline_s
    live = -1
    while time.monotonic() < deadline:
        fleet = await client.fleet_detail()
        live = sum(int(node.get("live_sandboxes", 0)) for node in fleet.get("nodes", []))
        if live == expected:
            return live
        await asyncio.sleep(2)
    return live


async def case_stdout_stderr(client: SandboxClient, image: str) -> bool:
    sandbox = await client.create(spec(image))
    try:
        result = await sandbox.exec("printf 'out\\n'; printf 'err\\n' >&2; exit 7")
        ok = result.exit_code == 7 and result.stdout == "out\n" and result.stderr == "err\n"
        return emit(
            "exec-separates-streams",
            ok,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    finally:
        await sandbox.release()


async def case_missing_path(client: SandboxClient, image: str) -> bool:
    sandbox = await client.create(spec(image))
    try:
        try:
            await sandbox.read_bytes("/definitely/not/here")
            missing = None
        except SandboxPathNotFound as exc:
            missing = exc
        after = await sandbox.exec("echo still-usable")
        ok = missing is not None and after.exit_code == 0 and "still-usable" in after.stdout
        return emit(
            "missing-path-is-not-a-lost-session",
            ok,
            raised=type(missing).__name__ if missing else None,
            sandbox_survived=after.exit_code == 0,
        )
    finally:
        await sandbox.release()


async def case_container_dies(client: SandboxClient, image: str, nodes: dict[str, str]) -> bool:
    sandbox = await client.create(spec(image))
    target = ssh_for(nodes, sandbox.node_id)
    try:
        if (await sandbox.exec("true")).exit_code != 0:
            return emit("container-death-is-attributed", False, reason="sandbox did not start")
        killed = docker(target, "kill", sandbox.sandbox_id)
        if killed.returncode != 0:
            return emit("container-death-is-attributed", False, reason=killed.stderr.strip())
        started = time.monotonic()
        try:
            await sandbox.exec("true")
            raised = None
        except SandboxSessionLost as exc:
            raised = exc
        latency = time.monotonic() - started
        # The node must release the charge, not merely report the loss.
        live = await wait_for_live(client, 0, deadline_s=20)
        ok = raised is not None and live == 0
        return emit(
            "container-death-is-attributed",
            ok,
            exit_reason=raised.exit_reason if raised else None,
            detected_s=round(latency, 3),
            live_after=live,
        )
    finally:
        try:
            await sandbox.release()
        except Exception:  # noqa: BLE001 - the sandbox may already be gone
            pass


async def case_owner_silence(client: SandboxClient, image: str) -> bool:
    """A caller that stops renewing loses its sandbox at the node's silence budget."""
    sandbox = await client.create(spec(image, owner="silent-owner", lifetime_s=3600))
    started = time.monotonic()
    reclaimed_s = None
    try:
        while time.monotonic() - started < 140:
            await asyncio.sleep(5)
            try:
                status = await sandbox.status()
            except Exception:  # noqa: BLE001 - a released sandbox may fail the lookup
                status = SandboxStatus.TERMINATED
            if status is SandboxStatus.TERMINATED:
                reclaimed_s = round(time.monotonic() - started, 1)
                break
        ok = reclaimed_s is not None and 90 <= reclaimed_s <= 140
        return emit("owner-silence-reclaims", ok, reclaimed_s=reclaimed_s)
    finally:
        try:
            await sandbox.release()
        except Exception:  # noqa: BLE001 - already reclaimed
            pass


async def case_load(
    client: SandboxClient, image: str, concurrency: int, nodes: dict[str, str],
) -> bool:
    """The shape an agent episode makes: create, command, file read, release."""
    latencies: list[float] = []
    failures = 0
    semaphore = asyncio.Semaphore(concurrency)

    async def one() -> None:
        nonlocal failures
        async with semaphore:
            started = time.perf_counter()
            sandbox = None
            try:
                sandbox = await client.create(spec(image, lifetime_s=300))
                result = await sandbox.exec("echo load")
                if result.exit_code != 0:
                    raise RuntimeError(f"exec exited {result.exit_code}")
                await sandbox.read_bytes("/etc/hostname")
                latencies.append(time.perf_counter() - started)
            except Exception:  # noqa: BLE001 - counted, not classified
                failures += 1
            finally:
                if sandbox is not None:
                    try:
                        await sandbox.release()
                    except Exception:  # noqa: BLE001
                        failures += 1

    total = concurrency * 4
    started = time.monotonic()
    await asyncio.gather(*(one() for _ in range(total)))
    elapsed = time.monotonic() - started
    latencies.sort()
    live = await wait_for_live(client, 0, deadline_s=30)
    remaining = 0
    for target in set(nodes.values()):
        containers = docker(target, "ps", "-q")
        remaining += len([line for line in containers.stdout.splitlines() if line.strip()])
    p50 = statistics.median(latencies) if latencies else 0.0
    p99 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.99))] if latencies else 0.0
    ok = failures == 0 and live == 0 and remaining == 0
    return emit(
        "episode-shaped-load",
        ok,
        total=total,
        concurrency=concurrency,
        failures=failures,
        p50_s=round(p50, 3),
        p99_s=round(p99, 3),
        throughput_per_s=round(total / elapsed, 2),
        live_after=live,
        containers_after=remaining,
    )


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--node-ssh", required=True)
    parser.add_argument("--image", default="alpine:latest")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--skip-reclamation", action="store_true")
    args = parser.parse_args()

    results: list[bool] = []
    nodes = parse_nodes(args.node_ssh)
    client = SandboxClient(args.endpoint)
    try:
        results.append(await case_stdout_stderr(client, args.image))
        results.append(await case_missing_path(client, args.image))
        results.append(await case_container_dies(client, args.image, nodes))
        if not args.skip_reclamation:
            results.append(await case_owner_silence(client, args.image))
        results.append(await case_load(client, args.image, args.concurrency, nodes))
    finally:
        await client.close()
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
