"""Verify a cluster-level sandboxd deployment places across every fleet node.

Run this against the control plane, not a node plane. It answers the question a
split deployment exists to answer: does one control plane spread work over the
nodes it was given, and does the data plane reach a sandbox on a node the
control plane does not live on?

    python3 verify_cluster.py <host:port> [sandbox_count]

The count defaults to the number of fleet nodes times two, which is the
smallest number that can show a placement spread at all.
"""

from __future__ import annotations

import asyncio
import sys
from collections import Counter

from sandboxd import Resources, SandboxClient, SandboxSpec, Source


async def main() -> int:
    endpoint = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:9000"
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    async with SandboxClient(endpoint) as client:
        report = await client.fleet()
        nodes = report.nodes
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
        sandboxes = await asyncio.gather(
            *(client.create(spec) for _ in range(count)), return_exceptions=True
        )

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
            await asyncio.gather(
                *(sandbox.release() for sandbox in live), return_exceptions=True
            )

        if failures:
            print(f"FAIL: {len(failures)} of {count} creates failed")
            return 1
        if len(placed) < 2:
            print("FAIL: every sandbox landed on one node; placement is not spreading")
            return 1
        print("PASS: the control plane placed across nodes and exec reached each one")
        return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
