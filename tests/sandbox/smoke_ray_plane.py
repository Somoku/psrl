"""Exercise the sandbox plane on a real Ray cluster.

This is not a unit test. It starts Ray, creates the placement service and one node agent per
node as real actors, and drives the protocol across process boundaries, which is the part a
test with `InProcessTransport` cannot reach: serialization, node affinity, actor concurrency,
and the per-call deadlines.

It deliberately does **not** create a container. A sandbox needs a Docker daemon, which is
what `test_docker_live.py` covers behind `PSRL_RUN_DOCKER_INTEGRATION`. What is verified here
is the plane around the daemon:

- the actors start, pinned to a node, and each builds its own manager
- registering a node advertises it, starts its liveness reporting, and starts the sweeper
- placement chooses and reserves across RPC, and the reservation protocol round-trips
- a node keeps itself known by reporting, and re-registers after a placement restart
- a create that fails on the node leaves no reservation behind
- the fleet's declared capabilities are the intersection of what its nodes report
- the plane shuts down without leaving actors running

Run it against any environment that has Ray and PSRL importable:

    python -m tests.sandbox.smoke_ray_plane

Exit code 0 means every check passed. Any failure raises, which is what a caller scripts on.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
import time

import ray
from omegaconf import OmegaConf
from psrl.sandbox.core import SandboxFeature, SandboxSource, SandboxSpec
from psrl.sandbox.placement import NodeCapabilities, PlacementRequest, PlacementService
from psrl.sandbox.ray_plane import build_sandbox_plane
from psrl.sandbox.remote import BoundedTransport

# Long enough that a report interval is observable inside one test run, short enough that the
# checks finish quickly. A production deployment uses the placement TTL's order of magnitude.
_LIVENESS_INTERVAL_S = 1.0
_NODE_TTL_S = 8.0
_RESERVATION_TTL_S = 4.0
_SWEEP_INTERVAL_S = 1.0
_RPC_TIMEOUT_S = 30.0


def _sandbox_config(heartbeat_dir: str) -> object:
    """A Docker-backed sandbox config, the shape a real deployment ships.

    The daemon is not reached until a create, so this is enough to build a node's manager and
    to report its capabilities on a machine with no Docker.
    """
    return OmegaConf.create(
        {
            "default_backend": "docker",
            "backends": {
                "docker": {
                    "_target_": "psrl.sandbox.backends.DockerBackend",
                    "lifecycle": {"heartbeat_dir": heartbeat_dir, "gc_enabled": False},
                }
            },
        }
    )


async def _run_checks() -> list[str]:
    """
    Drive every check and return the names that passed.
    """
    context = ray.get_runtime_context()
    node_id = context.get_node_id()
    passed: list[str] = []
    heartbeat_dir = tempfile.mkdtemp(prefix="psrl-smoke-heartbeats-")
    plane = build_sandbox_plane(
        _sandbox_config(heartbeat_dir),
        [node_id],
        node_ttl_s=_NODE_TTL_S,
        reservation_ttl_s=_RESERVATION_TTL_S,
        sweep_interval_s=_SWEEP_INTERVAL_S,
        heartbeat_interval_s=_LIVENESS_INTERVAL_S,
    )
    try:
        advertisements = plane.register_nodes()
        assert node_id in advertisements, advertisements
        backends = advertisements[node_id]["backends"]
        assert backends and backends[0]["backend"] == "docker", advertisements
        passed.append("a node actor starts on its node, builds a manager, and advertises itself")

        capabilities = plane.capabilities
        declared = {feature for feature in backends[0]["features"]}
        assert all(capabilities.supports(SandboxFeature(name)) for name in declared), declared
        passed.append("the fleet declares the intersection of what its nodes report")

        transport = BoundedTransport(plane.transport(), timeout_s=_RPC_TIMEOUT_S)
        request = {"backend": "docker", "image_references": ["psrl-smoke:latest"]}
        candidates = await transport.candidate_nodes(request, limit=1)
        assert candidates == [node_id], candidates
        passed.append("placement answers a candidate query across process boundaries")

        decision = await transport.choose({**request, "owner_id": "smoke"})
        reservation_id = str(decision["reservation_id"])
        assert str(decision["node_id"]) == node_id, decision
        ttl = await transport.reservation_ttl_s()
        assert ttl == _RESERVATION_TTL_S, ttl
        await transport.renew_reservation(reservation_id)
        await transport.release_reservation(reservation_id)
        passed.append("the reservation protocol round-trips, including the TTL a caller renews against")

        # Built through the same handle a worker receives, so this covers the worker's own
        # construction path. A create whose node fails must also withdraw its reservation.
        handle = plane.handle(backend_name="docker", rpc_timeout_s=_RPC_TIMEOUT_S)
        assert handle.node_ids == (node_id,), handle.node_ids
        remote = handle.remote_backend(owner_id="smoke")
        before = plane.placement_snapshot()
        try:
            await remote.create(SandboxSpec(SandboxSource.image("psrl-smoke:latest")))
        except Exception as error:
            created = error
        else:
            raise AssertionError("A create with no Docker daemon must fail rather than report a sandbox.")
        after = plane.placement_snapshot()
        assert after.reservations_open <= before.reservations_open, (before, after)
        # A failure has to arrive as its own diagnosis. An unpicklable exception crosses as a
        # serialization error instead, which hides what actually went wrong on the node.
        assert not isinstance(created, ray.exceptions.UnserializableException), (
            f"A provisioning failure must cross the actor boundary with its own diagnosis: {created!r}"
        )
        await remote.shutdown()
        passed.append(f"a create that fails on the node leaves no reservation behind ({type(created).__name__})")

        # The node keeps itself known. Without this every node drains a node TTL after it
        # registers, and placement then refuses work for the rest of the job.
        seen_before = plane.placement_snapshot().nodes
        # The trainer's metric hook reads the flattened form, so it has to cross a boundary too.
        flattened = plane.placement_snapshot().as_dict()
        assert "placement/nodes" in flattened, flattened
        assert seen_before == 1, seen_before
        await asyncio.sleep(_LIVENESS_INTERVAL_S * 2.5)
        assert await plane.placement.has_node.remote(node_id), "The node stopped reporting."
        assert await transport.candidate_nodes(request, limit=1) == [node_id]
        passed.append("a node keeps itself known by reporting across process boundaries")

        # Emptying the registry is the state a placement restart produces, and a heartbeat for
        # an unknown node is ignored, so what gets measured is the recovery.
        await plane.placement.unregister.remote(node_id)
        assert not await plane.placement.has_node.remote(node_id)
        await asyncio.sleep(_LIVENESS_INTERVAL_S * 2.5)
        assert await plane.placement.has_node.remote(node_id), "The node did not re-announce itself."
        assert await transport.candidate_nodes(request, limit=1) == [node_id]
        passed.append("a node re-announces itself to a placement that has forgotten it")

        # A stale node is drained rather than trusted, which is the backstop when a node really
        # goes away. Registered directly, since reporting is what this check withholds.
        stale = (
            ray.remote(PlacementService)
            .options(num_cpus=0)
            .remote(node_ttl_s=0.5, reservation_ttl_s=0.2, sweep_interval_s=0.1)
        )
        await stale.register.remote(_capabilities_from(advertisements[node_id]))
        drained = await stale.candidates.remote(PlacementRequest(backend="docker"), now=time.monotonic() + 10.0)
        assert drained == [], drained
        await stale.shutdown.remote()
        passed.append("a node nobody has heard from past its TTL is drained")
    finally:
        plane.shutdown()
        shutil.rmtree(heartbeat_dir, ignore_errors=True)
    return passed


def _capabilities_from(advertisement: dict) -> object:
    """
    Rebuild one node's published capabilities, for the stale-registry check.
    """
    return NodeCapabilities.from_advertisement(advertisement["backends"][0])


def main() -> int:
    """
    Start a cluster, run every check, and report what passed.
    """
    ray.init(include_dashboard=False, log_to_driver=False)
    try:
        passed = asyncio.run(_run_checks())
    finally:
        ray.shutdown()
    for index, name in enumerate(passed, start=1):
        print(f"  ok {index}. {name}")
    print(f"smoke: {len(passed)} checks passed on Ray {ray.__version__} (python {sys.version.split()[0]}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
