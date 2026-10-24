#!/usr/bin/env python3
"""Two fleet-level failures that a single node cannot show.

1. A node process dies. The control plane must stop targeting it for renewals
   after its TTL, keep renewing the surviving nodes inside the heartbeat budget,
   stop placing new work on it, and pick its sandboxes back up when it returns
   (the containers outlive the process).
2. A container disappears out of band. The node's runtime prober must release
   its charge, and the control plane's quota ledger must converge to the node's
   inventory rather than holding a phantom grant for the lease TTL.

Run against a control plane with a named fleet:

    python p3b/tests/sandbox/fleet_faults.py \
        --endpoint 127.0.0.1:9200 \
        --nodes node-a=root@host-a,node-b=root@host-b \
        --target node-b --image alpine:latest

The restart command is built from the node's config path, which must already
exist on that host: `<binary> -config <config>`. Pass `--restart-cmd` when the
deployment needs something else.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time

from sandboxd import Resources, SandboxClient, SandboxSpec, Source


def emit(case: str, ok: bool, **fields: object) -> bool:
    print(json.dumps({"case": case, "ok": ok, **fields}, sort_keys=True, default=str))
    return ok


def parse_nodes(value: str) -> dict[str, str]:
    return dict(item.split("=", 1) for item in value.split(","))


def ssh(target: str, command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", target, command],
        capture_output=True,
        text=True,
        timeout=60,
    )


def spec(image: str, owner: str) -> SandboxSpec:
    return SandboxSpec(
        source=Source.image(image),
        resources=Resources(memory_mb=256, cpu_count=1),
        resource_class="rollout",
        lifetime_s=1800,
        metadata={"owner_id": owner},
    )


async def create_until(client: SandboxClient, image: str, owner: str, node_id: str):
    """Create sandboxes until one lands on `node_id`, then return it and the rest."""
    sandboxes = []
    for _ in range(24):
        sandbox = await client.create(spec(image, owner))
        sandboxes.append(sandbox)
        if sandbox.node_id == node_id:
            return sandbox, sandboxes
    return None, sandboxes


async def renew_loop(client: SandboxClient, owner: str, stop: asyncio.Event, samples: list[float]):
    while not stop.is_set():
        started = time.monotonic()
        try:
            await client.renew_owner(owner)
        except Exception:  # noqa: BLE001 - the control answers ok even on partial failure
            pass
        samples.append(time.monotonic() - started)
        try:
            await asyncio.wait_for(stop.wait(), timeout=2)
        except asyncio.TimeoutError:
            continue


async def case_node_outage(
    client: SandboxClient, image: str, nodes: dict[str, str], target_id: str,
    restart_cmd: str | None,
) -> bool:
    target_ssh = nodes[target_id]
    owner = f"outage-{int(time.time())}"
    sandbox, siblings = await create_until(client, image, owner, target_id)
    if sandbox is None:
        for s in siblings:
            await s.release()
        return emit("node-outage", False, reason=f"no sandbox landed on {target_id}")

    stop = asyncio.Event()
    renewals: list[float] = []
    renew_task = asyncio.create_task(renew_loop(client, owner, stop, renewals))
    try:
        # Kill the node process; its containers keep running.
        killed = ssh(target_ssh, f"pkill -f 'sandbox[d] -config .*{target_id}'")
        killed_ok = killed.returncode == 0
        await asyncio.sleep(3)

        # A renewal that targets a dead node must still answer inside the budget:
        # the control fans out concurrently and the surviving nodes answer.
        batch_started = time.monotonic()
        await client.renew_owner(owner)
        renew_latency = time.monotonic() - batch_started

        # The sandbox on the dead node is unreachable; it must not look like a
        # task result.
        try:
            await sandbox.exec("true")
            data_plane = "unexpected-success"
        except Exception as exc:  # noqa: BLE001 - the type is the measurement
            data_plane = type(exc).__name__

        # Let the monitor age the dead node out.
        await asyncio.sleep(125)
        diagnostics = await client.diagnostics()
        stale = int(diagnostics["fleet"]["stale_nodes"])

        # New creates must not land on the stale node.
        fresh = [await client.create(spec(image, owner)) for _ in range(9)]
        placed_on_dead = sum(1 for s in fresh if s.node_id == target_id)
        for s in fresh:
            await s.release()
        if siblings:
            for s in siblings:
                if s.node_id != target_id:
                    await s.release()

        # Bring the node back. The containers outlived it, so the restarted node
        # adopts them and the same sandbox is usable again.
        restarted = False
        if restart_cmd:
            restarted = ssh(target_ssh, restart_cmd).returncode == 0
        if restarted:
            await asyncio.sleep(45)
        recovered = False
        if restarted:
            for _ in range(6):
                try:
                    recovered = (await sandbox.exec("echo back")).exit_code == 0
                except Exception:  # noqa: BLE001 - the node may still be reconciling
                    recovered = False
                if recovered:
                    break
                await asyncio.sleep(5)

        ok = (
            killed_ok
            and renew_latency < 6.0
            and data_plane != "unexpected-success"
            and stale >= 1
            and placed_on_dead == 0
            and (not restarted or recovered)
        )
        return emit(
            "node-outage",
            ok,
            killed=killed_ok,
            renew_latency_s=round(renew_latency, 3),
            renew_max_s=round(max(renewals) if renewals else 0.0, 3),
            stale_nodes=stale,
            placed_on_dead=placed_on_dead,
            data_plane_error=data_plane,
            restarted=restarted,
            recovered=recovered,
        )
    finally:
        stop.set()
        renew_task.cancel()
        try:
            await renew_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        for s in [sandbox, *siblings]:
            try:
                await s.release()
            except Exception:  # noqa: BLE001 - already gone or unreachable
                pass


async def case_container_vanishes(
    client: SandboxClient, image: str, nodes: dict[str, str],
) -> bool:
    owner = f"vanish-{int(time.time())}"
    sandbox = await client.create(spec(image, owner))
    target_ssh = nodes[sandbox.node_id]
    before = await client.quota()
    granted_before = sum(q.granted_memory_mb for q in before.classes.values())
    # The node reports on its own cadence; wait until the fleet view actually
    # shows the sandbox, or a removal would be compared against a zero baseline
    # that never contained it.
    node_before = 0
    for _ in range(15):
        fleet = await client.fleet_detail()
        node_before = sum(int(n.get("live_sandboxes", 0)) for n in fleet["nodes"])
        if node_before >= 1:
            break
        await asyncio.sleep(3)
    if node_before < 1:
        return emit("container-vanishes", False, reason="the fleet never reported the sandbox")

    removed = ssh(target_ssh, f"docker rm -f {sandbox.sandbox_id}")
    started = time.monotonic()
    node_released_s = None
    quota_released_s = None
    while time.monotonic() - started < 180:
        await asyncio.sleep(5)
        elapsed = time.monotonic() - started
        if node_released_s is None:
            fleet = await client.fleet_detail()
            live = sum(int(n.get("live_sandboxes", 0)) for n in fleet["nodes"])
            if live < node_before:
                node_released_s = round(elapsed, 1)
        if quota_released_s is None:
            quota = await client.quota()
            granted = sum(q.granted_memory_mb for q in quota.classes.values())
            if granted < granted_before:
                quota_released_s = round(elapsed, 1)
        if node_released_s is not None and quota_released_s is not None:
            break

    ok = (
        removed.returncode == 0
        and node_released_s is not None
        and quota_released_s is not None
        and quota_released_s <= 150
    )
    return emit(
        "container-vanishes",
        ok,
        removed=removed.returncode == 0,
        node_released_s=node_released_s,
        quota_released_s=quota_released_s,
        granted_before_mb=granted_before,
    )


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--image", default="alpine:latest")
    parser.add_argument("--restart-cmd", default=None)
    parser.add_argument("--skip-outage", action="store_true")
    parser.add_argument("--skip-vanish", action="store_true")
    args = parser.parse_args()

    nodes = parse_nodes(args.nodes)
    results: list[bool] = []
    client = SandboxClient(args.endpoint)
    try:
        if not args.skip_vanish:
            results.append(await case_container_vanishes(client, args.image, nodes))
        if not args.skip_outage:
            results.append(
                await case_node_outage(
                    client, args.image, nodes, args.target, args.restart_cmd
                )
            )
    finally:
        await client.close()
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
