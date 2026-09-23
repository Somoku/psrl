from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping

import pytest
from psrl.sandbox import (
    ExecResult,
    PauseMode,
    ResourceSpec,
    SandboxBackend,
    SandboxCapabilities,
    SandboxFeature,
    SandboxManager,
    SandboxRef,
    SandboxSession,
    SandboxSource,
    SandboxSpec,
    SandboxStatePolicy,
    SandboxStatus,
    SnapshotKind,
    SnapshotRef,
)
from psrl.sandbox.capacity import SandboxCapacityConfig


class FakeSession(SandboxSession):
    def __init__(self, backend: FakeBackend, sandbox_id: str, spec: SandboxSpec | None = None) -> None:
        self.backend = backend
        self.sandbox_id = sandbox_id
        self._spec = spec
        self.terminated = False
        self.paused_with: PauseMode | None = None

    @property
    def ref(self) -> SandboxRef:
        return SandboxRef(self.backend.name, self.sandbox_id)

    @property
    def capabilities(self) -> SandboxCapabilities:
        return self.backend.capabilities

    @property
    def spec(self) -> SandboxSpec | None:
        return self._spec

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> ExecResult:
        return ExecResult(0, command, "")

    async def read_bytes(self, path: str) -> bytes:
        return path.encode()

    async def write_bytes(self, path: str, data: bytes) -> None:
        return None

    async def status(self) -> SandboxStatus:
        return SandboxStatus.TERMINATED if self.terminated else SandboxStatus.RUNNING

    async def terminate(self) -> None:
        self.terminated = True

    async def pause(self, mode: PauseMode) -> None:
        self.paused_with = mode

    async def resume(self) -> None:
        self.paused_with = None

    async def snapshot(self, kind: SnapshotKind) -> SnapshotRef:
        return SnapshotRef(self.backend.name, f"snapshot-{self.sandbox_id}", kind)

    async def fork(self) -> SandboxSession:
        return FakeSession(self.backend, f"fork-{self.sandbox_id}", self._spec)


class FakeBackend(SandboxBackend):
    def __init__(self, features: set[SandboxFeature], uses_node_capacity: bool = False) -> None:
        self._capabilities = SandboxCapabilities(frozenset(features))
        self._uses_node_capacity = uses_node_capacity
        self.created: list[FakeSession] = []
        self.deleted_snapshots: list[SnapshotRef] = []

    @property
    def name(self) -> str:
        return "fake"

    @property
    def capabilities(self) -> SandboxCapabilities:
        return self._capabilities

    @property
    def uses_node_capacity(self) -> bool:
        return self._uses_node_capacity

    async def create(self, spec: SandboxSpec) -> SandboxSession:
        session = FakeSession(self, f"session-{len(self.created)}", spec)
        self.created.append(session)
        return session

    async def connect(self, sandbox_id: str) -> SandboxSession:
        return FakeSession(self, sandbox_id)

    async def restore(self, snapshot: SnapshotRef, spec: SandboxSpec | None = None) -> SandboxSession:
        return FakeSession(self, f"restored-{snapshot.snapshot_id}")

    async def delete_snapshot(self, snapshot: SnapshotRef) -> None:
        self.deleted_snapshots.append(snapshot)


@pytest.mark.asyncio
async def test_lease_terminates_exactly_once_and_unregisters() -> None:
    backend = FakeBackend(set())
    manager = SandboxManager({"fake": backend}, "fake")
    lease = await manager.acquire(SandboxSpec(SandboxSource.image("image")))

    await lease.release()
    await lease.release()

    assert isinstance(lease.session, FakeSession)
    assert lease.session.terminated
    assert not manager._leases


@pytest.mark.asyncio
async def test_failed_terminate_retains_capacity_until_reclaimed(monkeypatch) -> None:
    released = []

    async def release_capacity(lease_id):
        released.append(lease_id)

    backend = FakeBackend(set())
    manager = SandboxManager({"fake": backend}, "fake")
    lease = await manager.acquire(SandboxSpec(SandboxSource.image("image")))
    lease._capacity_lease_id = "capacity-1"
    lease._release_capacity = release_capacity
    terminate = lease.session.terminate

    async def fail_terminate():
        raise RuntimeError("Temporary cleanup failure.")

    monkeypatch.setattr(lease.session, "terminate", fail_terminate)
    await lease.release()
    assert not released
    assert not lease.released
    assert lease in manager._pending_releases
    monkeypatch.setattr(lease.session, "terminate", terminate)
    await asyncio.wait_for(manager._reaper_task, timeout=2)
    assert released == ["capacity-1"]
    assert lease.released
    assert not manager._leases
    await manager.shutdown()


@pytest.mark.asyncio
async def test_failed_capacity_release_remains_owned_for_shutdown_retry(monkeypatch) -> None:
    """A failed capacity release keeps the lease owned so shutdown can retry it."""
    attempts: list[str] = []

    async def flaky_release_capacity(lease_id: str) -> None:
        attempts.append(lease_id)
        raise RuntimeError("capacity coordinator unreachable")

    backend = FakeBackend(set())
    manager = SandboxManager({"fake": backend}, "fake")
    lease = await manager.acquire(SandboxSpec(SandboxSource.image("image")))
    lease._capacity_lease_id = "capacity-2"
    lease._release_capacity = flaky_release_capacity

    await manager.release(lease)

    assert attempts == ["capacity-2"]
    assert lease in manager._leases
    assert not lease.released

    async def release_capacity(lease_id: str) -> None:
        attempts.append(lease_id)

    lease._release_capacity = release_capacity
    await manager.release(lease)

    assert attempts == ["capacity-2", "capacity-2"]
    assert lease.released
    assert lease not in manager._leases


@pytest.mark.asyncio
async def test_shutdown_terminates_an_explicitly_paused_session() -> None:
    backend = FakeBackend({SandboxFeature.HIBERNATE})
    manager = SandboxManager({"fake": backend}, "fake")
    lease = await manager.acquire(SandboxSpec(SandboxSource.image("image")))

    await lease.session.pause(PauseMode.HIBERNATE)

    assert lease in manager._leases
    assert not lease.session.terminated

    await manager.shutdown()

    assert lease.session.terminated


@pytest.mark.asyncio
async def test_branch_prefers_native_fork() -> None:
    backend = FakeBackend({SandboxFeature.NATIVE_FORK})
    manager = SandboxManager({"fake": backend}, "fake")
    spec = SandboxSpec(
        SandboxSource.image("image"),
        state_policy=SandboxStatePolicy(enabled=True),
    )
    parent = await manager.acquire(spec)

    child = await manager.branch(parent.session)

    assert child.ref.sandbox_id == f"fork-{parent.ref.sandbox_id}"
    await manager.shutdown()


@pytest.mark.asyncio
async def test_branch_preserves_requested_snapshot_semantics() -> None:
    backend = FakeBackend({SandboxFeature.FILESYSTEM_SNAPSHOT, SandboxFeature.RESTORE})
    manager = SandboxManager({"fake": backend}, "fake")
    spec = SandboxSpec(
        SandboxSource.image("image"),
        state_policy=SandboxStatePolicy(enabled=True),
    )
    parent = await manager.acquire(spec)

    with pytest.raises(RuntimeError, match="full_state_snapshot"):
        await manager.branch(parent.session, snapshot_kind=SnapshotKind.FULL_STATE)

    child = await manager.branch(parent.session, snapshot_kind=SnapshotKind.FILESYSTEM)
    assert child.ref.sandbox_id.startswith("restored-snapshot-")
    assert len(backend.deleted_snapshots) == 1
    await manager.shutdown()


@pytest.mark.asyncio
async def test_concurrent_idempotent_create_returns_one_owned_lease() -> None:
    backend = FakeBackend(set())
    manager = SandboxManager({"fake": backend}, "fake")
    spec = SandboxSpec(SandboxSource.image("image"), idempotency_key="trajectory-1")

    first, second = await asyncio.gather(manager.acquire(spec), manager.acquire(spec))

    assert first is second
    assert len(backend.created) == 1
    await first.release()
    assert not manager._leases


@pytest.mark.asyncio
async def test_idempotency_key_rejects_a_different_spec() -> None:
    backend = FakeBackend(set())
    manager = SandboxManager({"fake": backend}, "fake")
    first = SandboxSpec(SandboxSource.image("image-a"), idempotency_key="trajectory-1")
    second = SandboxSpec(SandboxSource.image("image-b"), idempotency_key="trajectory-1")

    lease = await manager.acquire(first)
    with pytest.raises(RuntimeError, match="different spec"):
        await manager.acquire(second)

    await lease.release()


@pytest.mark.asyncio
async def test_snapshot_policy_rejects_secrets_and_post_command_external_effects() -> None:
    backend = FakeBackend({SandboxFeature.FULL_STATE_SNAPSHOT, SandboxFeature.RESTORE})
    manager = SandboxManager({"fake": backend}, "fake")
    secret_spec = SandboxSpec(
        SandboxSource.image("image"),
        env={"API_TOKEN": "secret"},
        state_policy=SandboxStatePolicy(enabled=True),
    )
    secret_lease = await manager.acquire(secret_spec)

    with pytest.raises(RuntimeError, match="secret-bearing"):
        await manager.checkpoint(secret_lease.session, SnapshotKind.FULL_STATE)

    class DirtySession(FakeSession):
        @property
        def command_count(self) -> int:
            return 1

    dirty = DirtySession(
        backend,
        "dirty",
        SandboxSpec(
            SandboxSource.image("image"),
            state_policy=SandboxStatePolicy(enabled=True),
        ),
    )
    with pytest.raises(RuntimeError, match="external"):
        await manager.checkpoint(dirty, SnapshotKind.FULL_STATE)

    await manager.shutdown()


@pytest.mark.asyncio
async def test_restore_requires_explicit_state_policy() -> None:
    backend = FakeBackend({SandboxFeature.RESTORE})
    manager = SandboxManager({"fake": backend}, "fake")
    snapshot = SnapshotRef("fake", "snapshot-1", SnapshotKind.FULL_STATE)

    with pytest.raises(RuntimeError, match="explicitly enabled"):
        await manager.restore(snapshot)

    await manager.shutdown()


@pytest.mark.asyncio
async def test_shutdown_waits_for_inflight_idempotent_create_and_reclaims_it() -> None:
    started = asyncio.Event()
    allow_create = asyncio.Event()

    class DelayedBackend(FakeBackend):
        async def create(self, spec: SandboxSpec) -> SandboxSession:
            started.set()
            await allow_create.wait()
            return await super().create(spec)

    backend = DelayedBackend(set())
    manager = SandboxManager({"fake": backend}, "fake")
    acquire_task = asyncio.create_task(
        manager.acquire(SandboxSpec(SandboxSource.image("image"), idempotency_key="inflight"))
    )
    await started.wait()
    shutdown_task = asyncio.create_task(manager.shutdown())
    await asyncio.sleep(0)
    allow_create.set()

    result = (await asyncio.gather(acquire_task, return_exceptions=True))[0]
    await shutdown_task
    assert isinstance(result, asyncio.CancelledError) or result.session.terminated
    assert not manager._leases


@pytest.mark.asyncio
async def test_cancelled_create_waiter_does_not_poison_idempotency_key() -> None:
    started = asyncio.Event()
    allow_create = asyncio.Event()

    class DelayedBackend(FakeBackend):
        async def create(self, spec: SandboxSpec) -> SandboxSession:
            started.set()
            await allow_create.wait()
            return await super().create(spec)

    backend = DelayedBackend(set())
    manager = SandboxManager({"fake": backend}, "fake")
    spec = SandboxSpec(SandboxSource.image("image"), idempotency_key="cancelled-waiter")
    waiter = asyncio.create_task(manager.acquire(spec))
    await started.wait()
    waiter.cancel()
    allow_create.set()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert not manager._leases

    replacement = await manager.acquire(spec)

    assert len(backend.created) == 2
    assert backend.created[0].terminated
    await replacement.release()
    await manager.shutdown()


@pytest.mark.asyncio
async def test_shutdown_also_tracks_non_idempotent_create() -> None:
    started = asyncio.Event()
    allow_create = asyncio.Event()

    class DelayedBackend(FakeBackend):
        async def create(self, spec: SandboxSpec) -> SandboxSession:
            started.set()
            await allow_create.wait()
            return await super().create(spec)

    backend = DelayedBackend(set())
    manager = SandboxManager({"fake": backend}, "fake")
    acquire_task = asyncio.create_task(manager.acquire(SandboxSpec(SandboxSource.image("image"))))
    await started.wait()
    shutdown_task = asyncio.create_task(manager.shutdown())
    allow_create.set()

    result = (await asyncio.gather(acquire_task, return_exceptions=True))[0]
    await shutdown_task
    assert isinstance(result, asyncio.CancelledError) or result.session.terminated
    assert not manager._leases


class FakeRemoteMethod:
    def __init__(self, method) -> None:
        self.method = method

    def remote(self, *args):
        return self.method(*args)


class FakeCapacityCoordinator:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str, int, float, str]] = []
        self.released: list[str] = []
        self.released_owners: list[str] = []
        self.acquire = FakeRemoteMethod(self._acquire)
        self.release = FakeRemoteMethod(self._release)
        self.cancel = FakeRemoteMethod(self._release)
        self.renew_owner = FakeRemoteMethod(self._renew_owner)
        self.release_owner = FakeRemoteMethod(self._release_owner)

    async def _acquire(
        self,
        lease_id: str,
        owner_id: str,
        memory_mb: int,
        cpu_count: float,
        resource_class: str,
    ) -> None:
        self.requests.append((lease_id, owner_id, memory_mb, cpu_count, resource_class))

    async def _release(self, lease_id: str) -> None:
        self.released.append(lease_id)

    async def _renew_owner(self, owner_id: str) -> None:
        return None

    async def _release_owner(self, owner_id: str) -> None:
        self.released_owners.append(owner_id)


def _capacity_manager(backend: FakeBackend, coordinator: FakeCapacityCoordinator) -> SandboxManager:
    return SandboxManager(
        {"fake": backend},
        "fake",
        capacity_coordinator=coordinator,
        capacity_owner_id="worker-1",
        capacity_heartbeat_interval_s=SandboxCapacityConfig().heartbeat_interval_s,
    )


@pytest.mark.asyncio
async def test_local_backend_charges_actual_spec_until_sandbox_release() -> None:
    backend = FakeBackend(set(), uses_node_capacity=True)
    capacity = FakeCapacityCoordinator()
    manager = _capacity_manager(backend, capacity)
    resources = ResourceSpec(cpu_count=1.5, memory_mb=4096)

    lease = await manager.acquire(
        SandboxSpec(SandboxSource.image("image"), resources=resources, resource_class="grader")
    )

    assert len(capacity.requests) == 1
    lease_id, owner_id, memory_mb, cpu_count, resource_class = capacity.requests[0]
    assert (owner_id, memory_mb, cpu_count, resource_class) == ("worker-1", 4096, 1.5, "grader")
    assert capacity.released == []
    await lease.release()
    assert capacity.released == [lease_id]
    await manager.shutdown()
    assert capacity.released_owners == ["worker-1"]


@pytest.mark.asyncio
async def test_failed_local_create_returns_capacity() -> None:
    class FailingBackend(FakeBackend):
        async def create(self, spec: SandboxSpec) -> SandboxSession:
            raise RuntimeError("create failed")

    backend = FailingBackend(set(), uses_node_capacity=True)
    capacity = FakeCapacityCoordinator()
    manager = _capacity_manager(backend, capacity)
    spec = SandboxSpec(
        SandboxSource.image("image"),
        resources=ResourceSpec(cpu_count=1, memory_mb=1024),
    )

    with pytest.raises(RuntimeError, match="create failed"):
        await manager.acquire(spec)

    assert capacity.released == [capacity.requests[0][0]]
    await manager.shutdown()


@pytest.mark.asyncio
async def test_shutdown_frees_capacity_before_joining_waiting_creates() -> None:
    class BlockingCapacityCoordinator(FakeCapacityCoordinator):
        def __init__(self) -> None:
            self.active_lease: str | None = None
            self.waiting = asyncio.Event()
            self.available = asyncio.Event()
            super().__init__()

        async def _acquire(
            self,
            lease_id: str,
            owner_id: str,
            memory_mb: int,
            cpu_count: float,
            resource_class: str,
        ) -> None:
            await super()._acquire(lease_id, owner_id, memory_mb, cpu_count, resource_class)
            if self.active_lease is not None:
                self.waiting.set()
                await self.available.wait()
            self.active_lease = lease_id

        async def _release(self, lease_id: str) -> None:
            await super()._release(lease_id)
            if self.active_lease == lease_id:
                self.active_lease = None
                self.available.set()

    backend = FakeBackend(set(), uses_node_capacity=True)
    capacity = BlockingCapacityCoordinator()
    manager = _capacity_manager(backend, capacity)
    spec = SandboxSpec(
        SandboxSource.image("image"),
        resources=ResourceSpec(cpu_count=1, memory_mb=1024),
    )
    first = await manager.acquire(spec)
    second_task = asyncio.create_task(manager.acquire(spec))
    await capacity.waiting.wait()

    await asyncio.wait_for(manager.shutdown(), timeout=1)
    with pytest.raises(asyncio.CancelledError):
        await second_task
    assert first.session.terminated
    assert not manager._leases


@pytest.mark.asyncio
async def test_hold_and_wait_is_rejected_before_admission() -> None:
    backend = FakeBackend(set(), uses_node_capacity=True)
    capacity = FakeCapacityCoordinator()
    manager = _capacity_manager(backend, capacity)
    spec = SandboxSpec(
        SandboxSource.image("image"),
        workflow_id="task-1",
        resources=ResourceSpec(cpu_count=1, memory_mb=1),
    )
    lease = await manager.acquire(spec)
    with pytest.raises(RuntimeError, match="must release"):
        await manager.acquire(spec)
    assert len(capacity.requests) == 1
    await lease.release()
    await manager.shutdown()


@pytest.mark.asyncio
async def test_released_workflow_slot_does_not_report_hold_and_wait() -> None:
    """The supported multi-phase order, release then acquire, is not reported."""
    backend = FakeBackend(set())
    manager = SandboxManager({"fake": backend}, "fake")

    rollout = await manager.acquire(SandboxSpec(SandboxSource.image("image"), workflow_id="task-1"))
    await manager.release(rollout)
    await manager.acquire(SandboxSpec(SandboxSource.image("image"), workflow_id="task-1"))

    assert len(manager._workflow_leases["task-1"]) == 1
    await manager.shutdown()


@pytest.mark.asyncio
async def test_deferred_cleanup_does_not_block_the_next_workflow_phase(monkeypatch) -> None:
    """A grading phase must still start when its rollout container resisted deletion.

    The workflow slot exists to stop concurrent phases. Cleanup the manager has taken over
    is not a concurrent phase, and its capacity reservation already protects the node, so
    failing the episode here would lose finished work over a container nobody can delete.
    """
    backend = FakeBackend(set())
    manager = SandboxManager({"fake": backend}, "fake")
    rollout = await manager.acquire(SandboxSpec(SandboxSource.image("image"), workflow_id="task-1"))

    async def fail_terminate():
        raise RuntimeError("device or resource busy")

    monkeypatch.setattr(rollout.session, "terminate", fail_terminate)
    await manager.release(rollout)

    assert rollout in manager._pending_releases
    assert not rollout.released
    # The physical reservation is retained, the logical phase slot is not.
    assert rollout in manager._leases
    assert "task-1" not in manager._workflow_leases

    grader = await manager.acquire(SandboxSpec(SandboxSource.image("image"), workflow_id="task-1"))

    assert grader is not rollout
    manager._reaper_task.cancel()
    await asyncio.gather(manager._reaper_task, return_exceptions=True)
    await grader.release()


@pytest.mark.asyncio
async def test_repeated_destruction_failure_is_escalated_once_it_stops_being_a_retry(monkeypatch, caplog) -> None:
    backend = FakeBackend(set())
    manager = SandboxManager({"fake": backend}, "fake")
    lease = await manager.acquire(SandboxSpec(SandboxSource.image("image")))

    async def fail_terminate():
        raise RuntimeError("device or resource busy")

    monkeypatch.setattr(lease.session, "terminate", fail_terminate)
    with caplog.at_level(logging.ERROR):
        for _ in range(5):
            await lease._release()

    assert manager.ownership_snapshot()["max_release_attempts"] == 5
    assert any("resisted 5 destruction attempts" in record.message for record in caplog.records)
    manager._reaper_task.cancel()
    await asyncio.gather(manager._reaper_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_a_cancelled_admission_withdraws_its_queued_request() -> None:
    """
    Cancelled admission withdraws queued or concurrently granted capacity.
    """

    class BlockingAdmissionCoordinator(FakeCapacityCoordinator):
        def __init__(self) -> None:
            super().__init__()
            self.admission_started = asyncio.Event()

        async def _acquire(
            self,
            lease_id: str,
            owner_id: str,
            memory_mb: int,
            cpu_count: float,
            resource_class: str,
        ) -> None:
            await super()._acquire(lease_id, owner_id, memory_mb, cpu_count, resource_class)
            self.admission_started.set()
            await asyncio.Event().wait()

    backend = FakeBackend(set(), uses_node_capacity=True)
    capacity = BlockingAdmissionCoordinator()
    manager = _capacity_manager(backend, capacity)
    spec = SandboxSpec(
        SandboxSource.image("image"),
        resources=ResourceSpec(cpu_count=1, memory_mb=1024),
    )

    create_task = asyncio.create_task(manager._acquire_new(spec, backend))
    await capacity.admission_started.wait()

    create_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await create_task

    assert capacity.released == [capacity.requests[0][0]], (
        "A cancelled admission left its request queued, so the node stays reserved for a "
        "sandbox that will never exist."
    )
    await manager.shutdown()


@pytest.mark.asyncio
async def test_capacity_release_completes_even_when_the_caller_is_cancelled() -> None:
    """
    Capacity return finishes before cancellation reaches the caller.
    """

    class SlowReleaseCoordinator(FakeCapacityCoordinator):
        def __init__(self) -> None:
            super().__init__()
            self.release_started = asyncio.Event()
            self.release_may_finish = asyncio.Event()

        async def _release(self, lease_id: str) -> None:
            self.release_started.set()
            await self.release_may_finish.wait()
            await super()._release(lease_id)

    backend = FakeBackend(set(), uses_node_capacity=True)
    capacity = SlowReleaseCoordinator()
    manager = _capacity_manager(backend, capacity)
    lease = await manager.acquire(
        SandboxSpec(SandboxSource.image("image"), resources=ResourceSpec(cpu_count=1, memory_mb=1024))
    )
    lease_id = capacity.requests[0][0]

    release_task = asyncio.create_task(lease.release())
    await capacity.release_started.wait()

    release_task.cancel()
    await asyncio.sleep(0)
    assert not release_task.done(), "The release finished before the cancellation was delivered."
    assert capacity.released == []

    capacity.release_may_finish.set()
    with pytest.raises(asyncio.CancelledError):
        await release_task

    assert capacity.released == [lease_id], (
        "A cancelled teardown skipped its capacity release, so the node stays charged for a "
        "sandbox that no longer exists."
    )
    await manager.shutdown()


@pytest.mark.asyncio
async def test_a_cancelled_teardown_completes_the_entire_ownership_transition() -> None:
    """
    Cancellation propagates after destruction, accounting, and ownership removal.
    """

    class SlowReleaseCoordinator(FakeCapacityCoordinator):
        def __init__(self) -> None:
            super().__init__()
            self.release_started = asyncio.Event()
            self.release_may_finish = asyncio.Event()
            self.attempts = 0

        async def _release(self, lease_id: str) -> None:
            self.attempts += 1
            self.release_started.set()
            await self.release_may_finish.wait()
            await super()._release(lease_id)

    backend = FakeBackend(set(), uses_node_capacity=True)
    capacity = SlowReleaseCoordinator()
    manager = _capacity_manager(backend, capacity)
    lease = await manager.acquire(
        SandboxSpec(SandboxSource.image("image"), resources=ResourceSpec(cpu_count=1, memory_mb=1024))
    )
    lease_id = capacity.requests[0][0]

    release_task = asyncio.create_task(manager.release(lease))
    await capacity.release_started.wait()

    release_task.cancel()
    capacity.release_may_finish.set()
    with pytest.raises(asyncio.CancelledError):
        await release_task

    assert capacity.released == [lease_id]
    assert lease not in manager._leases
    assert lease.released

    await manager.shutdown()

    assert lease.released
    assert not manager._leases
    assert set(capacity.released) == {lease_id}, "The shutdown retry charged or released a different lease."


@pytest.mark.asyncio
async def test_repeated_cancellation_cannot_interrupt_termination(monkeypatch):
    backend = FakeBackend(set(), uses_node_capacity=True)
    capacity = FakeCapacityCoordinator()
    manager = _capacity_manager(backend, capacity)
    lease = await manager.acquire(SandboxSpec(SandboxSource.image("image"), resources=ResourceSpec(1, 1)))
    entered, finish = asyncio.Event(), asyncio.Event()

    async def terminate():
        entered.set()
        await finish.wait()

    monkeypatch.setattr(lease.session, "terminate", terminate)
    task = asyncio.create_task(lease.release())
    await entered.wait()
    for _ in range(3):
        task.cancel()
        await asyncio.sleep(0)
    assert not task.done()
    assert not capacity.released
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert lease.released
    assert len(capacity.released) == 1
    await manager.shutdown()


@pytest.mark.asyncio
async def test_cancelled_shared_waiter_does_not_cancel_remaining_owner():
    entered, finish = asyncio.Event(), asyncio.Event()

    class Backend(FakeBackend):
        async def create(self, spec):
            entered.set()
            await finish.wait()
            return await super().create(spec)

    backend = Backend(set())
    manager = SandboxManager({"fake": backend}, "fake")
    spec = SandboxSpec(SandboxSource.image("image"), idempotency_key="shared")
    first = asyncio.create_task(manager.acquire(spec))
    await entered.wait()
    second = asyncio.create_task(manager.acquire(spec))
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    finish.set()
    lease = await second
    assert len(backend.created) == 1
    assert not lease.released
    await manager.shutdown()


@pytest.mark.asyncio
async def test_two_pending_requests_cannot_reserve_the_same_workflow():
    entered, finish = asyncio.Event(), asyncio.Event()

    class Backend(FakeBackend):
        async def create(self, spec):
            entered.set()
            await finish.wait()
            return await super().create(spec)

    backend = Backend(set())
    manager = SandboxManager({"fake": backend}, "fake")
    spec = SandboxSpec(SandboxSource.image("image"), workflow_id="workflow")
    first = asyncio.create_task(manager.acquire(spec))
    await entered.wait()
    with pytest.raises(RuntimeError, match="must release"):
        await manager.acquire(spec)
    finish.set()
    await first
    await manager.shutdown()


@pytest.mark.asyncio
async def test_failed_create_retains_a_failed_capacity_return_for_retry(monkeypatch):
    class Backend(FakeBackend):
        async def create(self, spec):
            raise RuntimeError("Creation failed.")

    backend = Backend(set(), uses_node_capacity=True)
    capacity = FakeCapacityCoordinator()
    manager = _capacity_manager(backend, capacity)
    release = capacity.release

    async def fail_release(lease_id):
        raise OSError("Coordinator unavailable.")

    capacity.release = FakeRemoteMethod(fail_release)
    with pytest.raises(RuntimeError, match="Creation failed"):
        await manager.acquire(SandboxSpec(SandboxSource.image("image"), resources=ResourceSpec(1, 1)))
    assert manager.ownership_snapshot()["pending_capacity"] == 1
    assert not manager._leases
    capacity.release = release
    await asyncio.wait_for(manager._reaper_task, timeout=2)
    assert len(capacity.released) == 1
    assert manager.ownership_snapshot()["pending_capacity"] == 0
    await manager.shutdown()


@pytest.mark.asyncio
async def test_cancelled_create_key_stays_reserved_until_rollback_finishes():
    entered, finish = asyncio.Event(), asyncio.Event()

    class Backend(FakeBackend):
        async def create(self, spec):
            entered.set()
            await finish.wait()
            return await super().create(spec)

    backend = Backend(set())
    manager = SandboxManager({"fake": backend}, "fake")
    spec = SandboxSpec(SandboxSource.image("image"), idempotency_key="key")
    task = asyncio.create_task(manager.acquire(spec))
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="reclaimed"):
        await manager.acquire(spec)
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert backend.created[0].terminated
    await manager.shutdown()


@pytest.mark.asyncio
async def test_connect_to_owned_session_does_not_charge_capacity_twice():
    backend = FakeBackend(set(), uses_node_capacity=True)
    capacity = FakeCapacityCoordinator()
    manager = _capacity_manager(backend, capacity)
    spec = SandboxSpec(SandboxSource.image("image"), resources=ResourceSpec(1, 1))
    lease = await manager.acquire(spec)
    connected = await manager.connect(lease.ref, resources=spec.resources)
    assert connected is lease
    assert len(capacity.requests) == 1
    await manager.shutdown()
