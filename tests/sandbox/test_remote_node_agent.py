"""The caller side of a remote node: placement, the agent, and the session protocol.

The property under test is that a caller holding a lease cannot tell a remote
sandbox from a local one, and that a provision which never happened leaves no
reservation behind.
"""

from __future__ import annotations

import pytest
from psrl.sandbox import (
    SandboxFeature,
    SandboxManager,
    SandboxSession,
    SandboxSource,
    SandboxSpec,
    SandboxStatus,
    SnapshotKind,
)
from psrl.sandbox.core import ResumeLevel
from psrl.sandbox.node_agent import SandboxNodeAgent, spec_from_payload, spec_to_payload
from psrl.sandbox.placement import NoPlacementCandidate, PlacementService
from psrl.sandbox.remote import (
    BoundedTransport,
    InProcessTransport,
    NodeAgentTimeout,
    NodeAgentTransport,
    RemoteNodeError,
    RemoteSandboxBackend,
)

from tests.sandbox.test_manager import FakeBackend, FakeSession

pytestmark = pytest.mark.cpu_test


class FailingBackend(FakeBackend):
    """A backend whose create always fails after the reservation was taken."""

    async def create(self, spec: SandboxSpec):
        raise RuntimeError("image is missing")


def _cluster(*backends: tuple[str, FakeBackend]) -> tuple[PlacementService, RemoteSandboxBackend, dict]:
    """Build a placement service, node agents, and the caller's remote backend."""
    placement = PlacementService(node_ttl_s=100.0, reservation_ttl_s=20.0)
    agents = {
        node_id: SandboxNodeAgent(SandboxManager({backend.name: backend}, backend.name), node_id=node_id)
        for node_id, backend in backends
    }
    transport = InProcessTransport(placement, agents)
    remote = RemoteSandboxBackend(transport, name="fake", owner_id="worker-1")
    return placement, remote, agents


def _spec(**overrides) -> SandboxSpec:
    payload = {"source": SandboxSource.image("image")}
    payload.update(overrides)
    return SandboxSpec(**payload)


async def _registered(placement, agents, transport) -> None:
    for node_id in agents:
        await transport.register_node(node_id)


def test_a_spec_survives_a_round_trip_through_the_wire() -> None:
    # The control channel carries mappings rather than live objects, so the codec is
    # the contract a non-Ray caller would implement.
    spec = _spec(
        required_features=frozenset({SandboxFeature.RESUME_ANYWHERE}),
        required_resume_level=ResumeLevel.FILESYSTEM,
        workflow_id="task-1",
        exec_mode=None,
        metadata={"run_id": "run-1"},
    )

    assert spec_from_payload(spec_to_payload(spec)) == spec


async def test_a_caller_acquires_and_drives_a_remote_sandbox() -> None:
    placement, remote, agents = _cluster(("node-a", FakeBackend({SandboxFeature.RESTORE}, uses_node_capacity=True)))
    transport = remote.transport
    await _registered(placement, agents, transport)

    session = await remote.create(_spec(required_features=frozenset({SandboxFeature.RESTORE})))

    result = await session.exec("echo hi")
    assert result.stdout == "echo hi"
    assert await session.status() is SandboxStatus.RUNNING
    assert (await session.read_bytes("/tmp/x")) == b"/tmp/x"

    await session.terminate()

    assert await session.status() is SandboxStatus.TERMINATED
    assert placement.snapshot().reservations_open == 0


async def test_a_remote_backend_never_charges_the_callers_node() -> None:
    # Admission has to live with the daemon it guards, so the node charges and the
    # caller does not.
    placement, remote, agents = _cluster(("node-a", FakeBackend(set(), uses_node_capacity=True)))
    await _registered(placement, agents, remote.transport)

    assert remote.uses_node_capacity is False


async def test_a_requirement_no_node_satisfies_is_refused_before_provisioning() -> None:
    placement, remote, agents = _cluster(("node-a", FakeBackend(set())))
    await _registered(placement, agents, remote.transport)

    with pytest.raises(NoPlacementCandidate, match="required features"):
        await remote.create(_spec(required_features=frozenset({SandboxFeature.NATIVE_FORK})))


async def test_a_failed_provision_leaves_no_reservation_behind() -> None:
    # A reservation nobody releases starves the cluster one slot at a time.
    placement, remote, agents = _cluster(("node-a", FailingBackend(set())))
    await _registered(placement, agents, remote.transport)

    with pytest.raises(RuntimeError, match="image is missing"):
        await remote.create(_spec())

    assert placement.snapshot().reservations_open == 0


async def test_a_live_remote_sandbox_keeps_its_reservation_by_renewing() -> None:
    # Placement sweeps a reservation its owner stopped renewing, and it cannot tell a live
    # long-lived sandbox from an abandoned one by age alone. Renewing keeps the slot, stopping loses it.
    import time

    placement, remote, agents = _cluster(("node-a", FakeBackend(set())))
    await _registered(placement, agents, remote.transport)
    session = await remote.create(_spec())
    baseline = time.monotonic()

    assert await remote.renew_reservations() == 1
    assert placement.sweep(now=baseline + placement.reservation_ttl_s() * 0.5) == []
    assert placement.snapshot().reservations_open == 1

    await session.terminate()

    # A finished sandbox leaves the index, so the worker stops renewing it.
    assert remote._handles == {}
    assert await remote.renew_reservations() == 0
    assert placement.snapshot().reservations_open == 0
    await remote.shutdown()


async def test_a_reservation_nobody_renews_is_swept_and_frees_the_slot() -> None:
    # The backstop for a worker that died between the reserve and the provision.
    import time

    placement, remote, agents = _cluster(("node-a", FakeBackend(set())))
    await _registered(placement, agents, remote.transport)
    await remote.create(_spec())
    baseline = time.monotonic()

    swept = placement.sweep(now=baseline + placement.reservation_ttl_s() * 10)

    assert len(swept) == 1
    assert placement.snapshot().reservations_open == 0
    await remote.shutdown()


async def test_the_session_reports_the_capabilities_the_node_gave_it() -> None:
    backend = FakeBackend({SandboxFeature.RESTORE, SandboxFeature.FREEZE})
    placement, remote, agents = _cluster(("node-a", backend))
    await _registered(placement, agents, remote.transport)

    session = await remote.create(_spec(required_features=frozenset({SandboxFeature.RESTORE})))

    assert session.capabilities.supports(SandboxFeature.RESTORE)
    assert session.capabilities.supports(SandboxFeature.FREEZE)
    assert not session.capabilities.supports(SandboxFeature.NATIVE_FORK)


async def test_a_paused_remote_sandbox_is_paused_on_its_node() -> None:
    placement, remote, agents = _cluster(("node-a", FakeBackend({SandboxFeature.FREEZE})))
    await _registered(placement, agents, remote.transport)
    session = await remote.create(_spec(required_features=frozenset({SandboxFeature.FREEZE})))

    from psrl.sandbox import PauseMode

    await session.pause(PauseMode.FREEZE)

    lease = next(iter(agents["node-a"]._handles.values()))
    assert lease.session.paused_with is PauseMode.FREEZE


async def test_diagnostics_come_back_from_the_hosting_node() -> None:
    placement, remote, agents = _cluster(("node-a", FakeBackend(set())))
    await _registered(placement, agents, remote.transport)
    session = await remote.create(_spec())

    diagnosis = await session.diagnostics()

    assert diagnosis.ref == session.ref
    assert diagnosis.status is SandboxStatus.RUNNING


async def test_a_node_refuses_a_sandbox_it_does_not_own() -> None:
    # A caller may only reach a sandbox through the node that provisioned it.
    placement, remote, agents = _cluster(("node-a", FakeBackend(set())))
    await _registered(placement, agents, remote.transport)

    with pytest.raises(RuntimeError, match="does not own"):
        await agents["node-a"].exec("fake", "someone-elses", "echo hi")


async def test_a_group_is_provisioned_on_one_node() -> None:
    placement, remote, agents = _cluster(("node-a", FakeBackend(set())))
    await _registered(placement, agents, remote.transport)
    members = [_spec(workflow_id=f"task-1#{index}") for index in range(3)]

    sessions = await remote.acquire_group(members)

    assert len(sessions) == 3
    assert {session.remote.node_id for session in sessions} == {"node-a"}
    assert placement.snapshot().reservations_open == 1
    for session in sessions:
        await session.terminate()
    assert placement.snapshot().reservations_open == 0


async def test_an_empty_group_is_refused() -> None:
    placement, remote, agents = _cluster(("node-a", FakeBackend(set())))
    await _registered(placement, agents, remote.transport)

    with pytest.raises(ValueError, match="at least one member"):
        await remote.acquire_group([])


async def test_connect_finds_a_sandbox_this_worker_already_holds() -> None:
    placement, remote, agents = _cluster(("node-a", FakeBackend(set())))
    await _registered(placement, agents, remote.transport)
    session = await remote.create(_spec())

    reconnected = await remote.connect(session.remote.sandbox_id)

    assert reconnected.ref == session.ref


async def test_connect_names_why_an_unknown_sandbox_cannot_be_found() -> None:
    placement, remote, agents = _cluster(("node-a", FakeBackend(set())))
    await _registered(placement, agents, remote.transport)

    with pytest.raises(RemoteNodeError, match="handle index"):
        await remote.connect("unknown")


async def test_the_node_agent_advertises_what_placement_needs() -> None:
    placement, remote, agents = _cluster(("node-a", FakeBackend({SandboxFeature.HOST_MOUNT})))
    await _registered(placement, agents, remote.transport)

    advertised = placement.nodes()[0]

    assert advertised.node_id == "node-a"
    assert advertised.backend == "fake"
    assert advertised.host_mounts
    assert advertised.reachable_session_server


async def test_a_restore_is_placed_by_the_level_the_snapshot_promises() -> None:
    # A filesystem snapshot must not be treated as a full-state resume.
    from psrl.sandbox.core import SnapshotRef

    placement, remote, agents = _cluster(
        ("fs-only", FakeBackend({SandboxFeature.FILESYSTEM_SNAPSHOT, SandboxFeature.RESTORE}))
    )
    await _registered(placement, agents, remote.transport)
    snapshot = SnapshotRef("fake", "snapshot-1", SnapshotKind.FULL_STATE, resume_level=ResumeLevel.FULL_STATE)

    with pytest.raises(NoPlacementCandidate):
        await remote.restore(snapshot, _spec())


async def test_the_node_agent_reports_its_plane_metrics() -> None:
    placement, remote, agents = _cluster(("node-a", FakeBackend(set())))
    await _registered(placement, agents, remote.transport)
    session = await remote.create(_spec())

    metrics = agents["node-a"].snapshot()

    assert metrics["ownership/leases"] == 1.0
    await session.terminate()


async def test_a_node_fake_session_is_only_reachable_after_acquire() -> None:
    # A guard against a handle index that quietly accepts anything.
    agent = SandboxNodeAgent(SandboxManager({"fake": FakeBackend(set())}, "fake"), node_id="node-a")

    assert isinstance(FakeSession, type)
    with pytest.raises(RuntimeError, match="does not own"):
        await agent.status("fake", "unknown")


class WarmableBackend(FakeBackend):
    """A backend that can materialize an image ahead of a task."""

    def __init__(self, features=frozenset(), **kwargs) -> None:
        super().__init__(set(features), **kwargs)
        self.prefetched: list[tuple[str, ...]] = []

    async def prefetch_images(self, references, *, concurrency: int = 2) -> int:
        self.prefetched.append(tuple(references))
        return len(references)


async def test_a_prefetch_warms_only_the_candidate_nodes() -> None:
    # A prefetch warms the working set where a task is likely to land, not on every node:
    # warming the fleet would pay a pull per node for images most nodes never serve.
    placement, remote, agents = _cluster(
        ("node-a", WarmableBackend()),
        ("node-b", WarmableBackend()),
    )
    await _registered(placement, agents, remote.transport)

    warmed = await remote.prefetch_images(["python:3.11"], nodes=1)

    assert warmed == 1
    warmed_nodes = [
        node_id for node_id, agent in agents.items() if agent.manager._backends["fake"].prefetched
    ]
    assert warmed_nodes == ["node-a"]
    await remote.shutdown()


async def test_a_node_whose_backend_materializes_lazily_warms_nothing() -> None:
    # A provider loads layers on demand, so the node reports zero rather than failing a
    # step that is only an optimization.
    placement, remote, agents = _cluster(("node-a", FakeBackend(set())))
    await _registered(placement, agents, remote.transport)

    assert await remote.prefetch_images(["python:3.11"]) == 0
    await remote.shutdown()


class HostedSession(FakeSession):
    """A session whose node rewrites loopback URLs, the way the Docker backend does."""

    @property
    def callback_host_alias(self) -> str | None:
        return "host.docker.internal"


class HostedBackend(FakeBackend):
    """A backend whose sessions reach the caller through their node's gateway alias."""

    async def create(self, spec: SandboxSpec) -> SandboxSession:
        session = HostedSession(self, f"session-{len(self.created)}", spec)
        self.created.append(session)
        return session


async def test_a_callback_url_resolves_through_the_nodes_forwarder() -> None:
    # A sandbox on another node reaches the caller through a node-side forwarder, so the
    # URL it is handed is the same whether it landed locally or elsewhere.
    placement, remote, agents = _cluster(("node-a", HostedBackend(set())))
    await _registered(placement, agents, remote.transport)
    remote.callback_target = "10.0.0.7:9000"

    session = await remote.create(_spec())

    assert session.remote.callback_port is not None
    assert session.resolve_callback_url("http://127.0.0.1:9000/session/1") == (
        f"http://host.docker.internal:{session.remote.callback_port}/session/1"
    )
    await session.terminate()
    await remote.shutdown()


async def test_without_a_target_the_url_keeps_its_own_port() -> None:
    # The single node path needs no forwarder: the sandbox is a sibling of the worker, so
    # only the host is rewritten.
    placement, remote, agents = _cluster(("node-a", HostedBackend(set())))
    await _registered(placement, agents, remote.transport)

    session = await remote.create(_spec())

    assert session.remote.callback_port is None
    assert session.resolve_callback_url("http://127.0.0.1:9000/x") == "http://host.docker.internal:9000/x"
    await session.terminate()
    await remote.shutdown()


async def test_one_forwarder_serves_every_sandbox_the_worker_places() -> None:
    # The forwarder is per caller, not per sandbox, so a node does not open a port per
    # trajectory.
    placement, remote, agents = _cluster(("node-a", HostedBackend(set())))
    await _registered(placement, agents, remote.transport)
    remote.callback_target = "10.0.0.7:9000"

    first = await remote.create(_spec())
    second = await remote.create(_spec())

    assert first.remote.callback_port == second.remote.callback_port
    assert len(agents["node-a"]._forwarders) == 1
    await first.terminate()
    await second.terminate()
    await remote.shutdown()


async def test_a_node_shutdown_stops_forwarding() -> None:
    placement, remote, agents = _cluster(("node-a", HostedBackend(set())))
    await _registered(placement, agents, remote.transport)
    remote.callback_target = "10.0.0.7:9000"
    await remote.create(_spec())

    await agents["node-a"].shutdown()

    assert agents["node-a"]._forwarders == {}
    await remote.shutdown()


async def test_a_slow_admission_keeps_its_reservation_by_renewing() -> None:
    # Placement charges the slot before the sandbox exists, so an admission that outlasts
    # the reservation TTL would lose the slot while this caller is still waiting for it.
    import asyncio

    class SlowBackend(FakeBackend):
        """A backend whose create takes longer than the reservation TTL."""

        def __init__(self) -> None:
            super().__init__(set())
            self.released = asyncio.Event()
            self.started = asyncio.Event()

        async def create(self, spec: SandboxSpec):
            self.started.set()
            await self.released.wait()
            return await super().create(spec)

    backend = SlowBackend()
    placement = PlacementService(node_ttl_s=100.0, reservation_ttl_s=0.05, sweep_interval_s=0.01)
    agents = {"node-a": SandboxNodeAgent(SandboxManager({backend.name: backend}, backend.name), node_id="node-a")}
    transport = InProcessTransport(placement, agents)
    remote = RemoteSandboxBackend(transport, name="fake", owner_id="worker-1", renew_interval_s=0.01)
    await _registered(placement, agents, transport)

    creating = asyncio.create_task(remote.create(_spec()))
    await backend.started.wait()
    # Past several reservations' worth of TTL, which is where the old code lost the slot.
    await asyncio.sleep(0.2)
    # The real clock asks "is anything older than the TTL right now", which is the claim.
    # A jumped clock would only prove that nobody renewed for the jump.
    assert placement.sweep() == [], "A renewed reservation must survive its own admission wait."
    assert remote._pending_reservations, "The reservation has no handle yet, so it is still pending."
    assert placement.snapshot().reservations_open == 1

    backend.released.set()
    session = await creating
    await session.terminate()
    await remote.shutdown()


async def test_a_failed_admission_releases_its_pending_reservation() -> None:
    # Protection starts before the sandbox exists, so the failure path has to withdraw the
    # reservation rather than leave a slot charged for a create that never happened.
    placement, remote, agents = _cluster(("node-a", FailingBackend(set())))
    await _registered(placement, agents, remote.transport)

    with pytest.raises(RuntimeError, match="image is missing"):
        await remote.create(_spec())

    assert placement.snapshot().reservations_open == 0
    assert remote._pending_reservations == set()
    assert remote._live_reservations() == []
    await remote.shutdown()


async def test_the_renew_loop_stops_once_nothing_is_left_to_renew() -> None:
    # The loop is started per reservation, so it must end when the last one clears rather
    # than idle for the life of the worker.
    import asyncio

    placement, remote, agents = _cluster(("node-a", FakeBackend(set())))
    await _registered(placement, agents, remote.transport)
    remote._renew_interval_s = 0.01
    session = await remote.create(_spec())
    await asyncio.sleep(0.05)

    await session.terminate()
    await asyncio.sleep(0.05)

    assert remote._renew_task is not None
    assert remote._renew_task.done(), "The renewal loop must end with its last reservation."
    await remote.shutdown()


async def test_an_unreadable_placement_ttl_does_not_stop_renewal() -> None:
    # One slow RPC must not leave every reservation to expire.
    placement, remote, agents = _cluster(("node-a", FakeBackend(set())))
    await _registered(placement, agents, remote.transport)

    async def refuse() -> float:
        raise RuntimeError("placement is unreachable")

    remote.transport.reservation_ttl_s = refuse
    assert await remote._reservation_ttl_s() == 60.0


async def test_a_call_that_never_answers_raises_a_named_timeout() -> None:
    # A remote call has no default deadline, so without one a wedged agent blocks its
    # caller for the rest of the run and the reservation renewer stops with it.
    import asyncio

    placement, remote, agents = _cluster(("node-a", FakeBackend(set())))
    await _registered(placement, agents, remote.transport)
    released = asyncio.Event()

    async def hang(*args, **kwargs):
        await released.wait()

    remote.transport.acquire = hang
    remote.transport = BoundedTransport(remote.transport, timeout_s=0.02)

    with pytest.raises(NodeAgentTimeout, match="'acquire'") as raised:
        await remote.create(_spec())

    # It is a TimeoutError, which is what a caller already knows how to handle.
    assert isinstance(raised.value, TimeoutError)
    # The reservation the timed-out create was holding is withdrawn, not left charged.
    assert placement.snapshot().reservations_open == 0
    released.set()


async def test_the_deadline_covers_every_call_the_protocol_declares() -> None:
    # A policy applied to some calls is a policy with a hole, and a hole is where the
    # caller that never returns gets through.
    bounded = BoundedTransport(_cluster(("node-a", FakeBackend(set())))[1].transport, timeout_s=1.0)
    declared = sorted(
        name
        for base in NodeAgentTransport.__mro__
        for name, member in vars(base).items()
        if not name.startswith("_") and callable(member)
    )

    missing = [name for name in declared if not hasattr(bounded, name)]

    assert declared, "The protocol must declare the calls this wrapper is meant to cover."
    assert missing == [], f"Every node agent call needs a deadline, and these have none: {missing}"


def test_a_bounded_transport_refuses_a_useless_deadline() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        BoundedTransport(None, timeout_s=0)


async def test_a_slow_placement_ttl_lookup_is_bounded_too() -> None:
    # The renewer's first call is the TTL lookup, so a deadline that skips it would let one
    # hung lookup stop every renewal.
    import asyncio

    placement, remote, agents = _cluster(("node-a", FakeBackend(set())))
    released = asyncio.Event()

    async def hang() -> float:
        await released.wait()
        return 1.0

    bounded = BoundedTransport(remote.transport, timeout_s=0.02)
    bounded.transport.reservation_ttl_s = hang
    remote.transport = bounded

    with pytest.raises(NodeAgentTimeout, match="'reservation_ttl_s'"):
        await remote.transport.reservation_ttl_s()

    released.set()


async def test_reporting_moves_the_moment_placement_ages_a_node_from() -> None:
    # A node is trusted relative to its own last word, which only a report moves forward.
    # Measured from the record, because a jumped clock cannot show when the last report was.
    import asyncio

    from psrl.sandbox.placement import PlacementRequest

    placement, remote, agents = _cluster(("node-a", FakeBackend(set())))
    agent = agents["node-a"]
    agent.liveness_interval_s = 0.02
    await _registered(placement, agents, remote.transport)
    registered_at = placement._nodes["node-a"].seen_at

    await asyncio.sleep(0.1)

    reported_at = placement._nodes["node-a"].seen_at
    request = PlacementRequest(backend="fake")
    assert reported_at > registered_at, "The node must keep refreshing its liveness."
    assert agent._liveness_task is not None, "Registering a node must start its reporting."
    # A node is trusted relative to its own last word, and only a report moves that word.
    assert placement.candidates(request, now=reported_at + placement.node_ttl_s * 0.5) == ["node-a"]
    assert placement.candidates(request, now=reported_at + placement.node_ttl_s * 1.5) == []

    await agent.shutdown()
    assert agent._liveness_task is None
    await remote.shutdown()


async def test_a_node_that_never_reports_is_drained_at_its_ttl() -> None:
    # The backstop for a node that went away: placement ages it out rather than trusting it
    # forever. Registered directly, because the transport is what installs the reporter.
    import time

    from psrl.sandbox.placement import NodeCapabilities, PlacementRequest

    placement, remote, agents = _cluster(("node-a", FakeBackend(set())))
    placement.register(NodeCapabilities(node_id="node-a", backend="fake"))
    registered_at = time.monotonic()
    request = PlacementRequest(backend="fake")

    assert agents["node-a"]._liveness_task is None
    assert placement.candidates(request, now=registered_at + placement.node_ttl_s * 0.5) == ["node-a"]
    assert placement.candidates(request, now=registered_at + placement.node_ttl_s * 1.5) == []
    await remote.shutdown()


async def test_a_node_outliving_a_placement_restart_registers_again() -> None:
    # Heartbeating into a registry that never heard of the node is ignored, so a restarted
    # placement would otherwise leave the whole fleet drained for the rest of the job.
    import asyncio

    placement, remote, agents = _cluster(("node-a", FakeBackend(set())))
    agent = agents["node-a"]
    agent.liveness_interval_s = 0.02
    await _registered(placement, agents, remote.transport)

    # A restarted placement service, empty, while the node keeps running.
    restarted = PlacementService(node_ttl_s=100.0, reservation_ttl_s=20.0)
    remote.transport.placement = restarted
    await asyncio.sleep(0.1)

    assert restarted.has_node("node-a"), "The node must re-announce itself to an empty registry."
    await agent.shutdown()
    await remote.shutdown()


async def test_a_placement_that_refuses_a_report_does_not_stop_the_node() -> None:
    # One unreachable call must not end the reporting loop, because that would drain the
    # node permanently for a transient reason.
    import asyncio

    placement, remote, agents = _cluster(("node-a", FakeBackend(set())))
    agent = agents["node-a"]
    agent.liveness_interval_s = 0.01
    attempts = []

    async def refuse() -> None:
        attempts.append(1)
        raise RuntimeError("placement is unreachable")

    agent.liveness_reporter = refuse
    agent.start_liveness()
    await asyncio.sleep(0.05)

    assert len(attempts) >= 2, "The loop must keep trying after a failed report."
    await agent.shutdown()


async def test_a_node_without_a_reporter_reports_nothing() -> None:
    # A directly constructed agent has no transport to report through, and must not start a
    # task it cannot complete.
    import asyncio

    placement, remote, agents = _cluster(("node-a", FakeBackend(set())))
    agent = agents["node-a"]
    agent.liveness_reporter = None

    await agent.advertise()

    assert agent._liveness_task is None
    await asyncio.sleep(0.01)


def test_placement_refuses_a_sweep_slower_than_its_reservation_ttl() -> None:
    # The sweep is what enforces the reservation TTL, so a slower sweep frees a dead owner's
    # slot later than the TTL promises while the node keeps admitting against it.
    with pytest.raises(ValueError, match="sweep interval"):
        PlacementService(node_ttl_s=100.0, reservation_ttl_s=5.0, sweep_interval_s=10.0)
