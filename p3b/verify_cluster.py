"""Verify a cluster-level sandboxd deployment places across every fleet node.

Run this against the control plane, not a node plane. It answers the question a
split deployment exists to answer: does one control plane spread work over the
nodes it was given, and does the data plane reach a sandbox on a node the
control plane does not live on?

    python3 verify_cluster.py <host:port> [sandbox_count] [--lifecycle]

The count defaults to the number of fleet nodes times two, which is the
smallest number that can show a placement spread at all.

`--lifecycle` adds the reclamation check a placement sweep cannot show: a
sandbox whose owner stops heartbeating must be reclaimed by its node, on the
real timing the deployment runs with. It costs about two minutes, because it
waits out the node's owner-silence budget rather than shortening it.

The node process restart is not checked here. It needs the node's own host and
configuration, and the two-process test in `tests/rollout` covers it with a real
process and a real daemon.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from collections import Counter

from sandboxd import ExecMode, Resources, SandboxClient, SandboxSpec, SandboxStatus, Source

# The metadata key a node reads the owning caller from. The SDK carries metadata
# opaquely, so this script states it the way a caller does.
OWNER_METADATA_KEY = "owner_id"

# The node's own constants, restated so the wait is derived rather than guessed.
# They are the deployment's, not this script's: shortening them here would check
# a timing no node runs with.
OWNER_SILENCE_TTL_S = 90
SWEEP_INTERVAL_S = 15


async def check_reclamation(client: SandboxClient) -> int:
    """Verify a node reclaims a sandbox whose owner stopped heartbeating."""
    print("== reclamation: a silent owner's sandbox is reclaimed ==")
    owner = f"verify-silent-{uuid.uuid4().hex[:12]}"
    spec = SandboxSpec(
        source=Source.image("alpine:latest"),
        resources=Resources(memory_mb=256, cpu_count=1),
        resource_class="rollout",
        exec_mode=ExecMode.ONE_SHOT,
        # Long enough that only owner silence can explain a reclamation.
        lifetime_s=3600,
        # The owner the node keys liveness on. Nothing heartbeats it.
        metadata={OWNER_METADATA_KEY: owner},
    )
    sandbox = await client.create(spec)
    try:
        before = await sandbox.status()
        print(f"  {sandbox.sandbox_id[:12]} on {sandbox.node_id}: {before.value}")
        if before != SandboxStatus.RUNNING:
            print(f"FAIL: a freshly created sandbox is {before.value}")
            return 1

        wait = OWNER_SILENCE_TTL_S + SWEEP_INTERVAL_S + 15
        print(f"  waiting {wait}s for the silence budget and a sweep, with no heartbeat")
        await asyncio.sleep(wait)

        after = await sandbox.status()
        print(f"  {sandbox.sandbox_id[:12]} on {sandbox.node_id}: {after.value}")
        if after == SandboxStatus.RUNNING:
            print(
                "FAIL: the node still holds a sandbox whose owner has been silent for "
                f"{OWNER_SILENCE_TTL_S}s, so nothing reclaims a caller that walked away"
            )
            return 1
    finally:
        # The control plane still holds a record, so this must succeed and must
        # not resurrect anything.
        await sandbox.release()

    print("PASS: the node reclaimed the silent owner's sandbox")
    return 0


async def main() -> int:
    argv = [arg for arg in sys.argv[1:] if arg != "--lifecycle"]
    lifecycle = "--lifecycle" in sys.argv
    endpoint = argv[0] if argv else "127.0.0.1:9000"
    count_arg = int(argv[1]) if len(argv) > 1 else None

    async with SandboxClient(endpoint) as client:
        report = await client.fleet()
        nodes = report.nodes
        count = count_arg if count_arg is not None else max(2, len(nodes) * 2)
        print(f"fleet: {len(nodes)} node(s)")
        for node in nodes:
            print(
                f"  {node.node_id:10s} backends={list(node.backends)} "
                f"live={node.live_sandboxes} mem_used_pct={node.mem_used_pct:.3f}"
            )
        if len(nodes) < 2:
            print("FAIL: a cluster-level deployment must report more than one node")
            return 1

        # Created concurrently, because placement's in-flight overlay is what
        # keeps a burst from choosing one node; a serial loop would pass even
        # with the overlay broken.
        spec = SandboxSpec(
            source=Source.image("alpine:latest"),
            resources=Resources(memory_mb=256, cpu_count=1),
            resource_class="rollout",
        )
        sandboxes = await asyncio.gather(*(client.create(spec) for _ in range(count)), return_exceptions=True)

        failures = [s for s in sandboxes if isinstance(s, BaseException)]
        live = [s for s in sandboxes if not isinstance(s, BaseException)]
        for failure in failures:
            print(f"  create failed: {type(failure).__name__}: {failure}")

        try:
            placed = Counter(sandbox.node_id for sandbox in live)
            print(f"placement over {len(live)} sandbox(es): {dict(placed)}")

            # One exec per node, so the verdict covers the data plane on a node
            # the control plane does not share a machine with.
            seen: set[str] = set()
            for sandbox in live:
                if sandbox.node_id in seen:
                    continue
                seen.add(sandbox.node_id)
                result = await sandbox.exec("echo ok")
                status = "ok" if result.stdout.strip() == "ok" else f"unexpected {result.stdout!r}"
                print(f"  exec on {sandbox.node_id}: {status}")
        finally:
            await asyncio.gather(*(sandbox.release() for sandbox in live), return_exceptions=True)

        if failures:
            print(f"FAIL: {len(failures)} of {count} creates failed")
            return 1
        if len(placed) < 2:
            print("FAIL: every sandbox landed on one node; placement is not spreading")
            return 1
        print("PASS: the control plane placed across nodes and exec reached each one")

        if lifecycle:
            return await check_reclamation(client)
        return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
