from __future__ import annotations

import asyncio
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
async def test_failed_terminate_still_returns_capacity(monkeypatch, caplog) -> None:
    """A container that cannot be removed must not keep its capacity lease.

    Capacity has no second recovery path: the owner heartbeat keeps renewing the owning
    worker, so a lease a failed release forgot is never reclaimed. The container is left
    to the Docker ownership sweep, which force-removes it when the owner stops.
    """
    released: list[str] = []

    async def release_capacity(lease_id: str) -> None:
        released.append(lease_id)

    backend = FakeBackend(set())
    manager = SandboxManager({"fake": backend}, "fake")
    lease = await manager.acquire(SandboxSpec(SandboxSource.image("image")))
    lease._capacity_lease_id = "capacity-1"
    lease._release_capacity = release_capacity

    async def fail_terminate() -> None:
        raise RuntimeError("temporary cleanup failure")

    monkeypatch.setattr(lease.session, "terminate", fail_terminate)
    with caplog.at_level("ERROR"):
        await manager.release(lease)

    assert released == ["capacity-1"]
    assert lease.released
    assert lease not in manager._leases
    assert any("temporary cleanup failure" in record.message for record in caplog.records)


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

    with pytest.raises(RuntimeError, match="capacity coordinator unreachable"):
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

    lease = await acquire_task
    await shutdown_task

    assert isinstance(lease.session, FakeSession)
    assert lease.session.terminated


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
    with pytest.raises(asyncio.CancelledError):
        await waiter
    allow_create.set()
    while not manager._leases:
        await asyncio.sleep(0)
    abandoned = next(iter(manager._leases))
    await abandoned.release()

    replacement = await manager.acquire(spec)

    assert len(backend.created) == 2
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

    lease = await acquire_task
    await shutdown_task

    assert isinstance(lease.session, FakeSession)
    assert lease.session.terminated


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
    second = await second_task

    assert first.session.terminated
    assert second.session.terminated


@pytest.mark.asyncio
async def test_hold_and_wait_is_reported_when_a_workflow_holds_a_lease(caplog) -> None:
    """A job that asks for a second sandbox before releasing the first is reported.

    That shape deadlocks a shared envelope: the job occupies one class's capacity while
    waiting for another class, and no admission order can fix it. The manager only warns
    and counts, because a backend that hands back a warm sandbox may one day do it on purpose.
    """
    backend = FakeBackend(set())
    manager = SandboxManager({"fake": backend}, "fake")
    first = SandboxSpec(SandboxSource.image("image"), workflow_id="task-1")
    second = SandboxSpec(SandboxSource.image("image"), workflow_id="task-1")

    with caplog.at_level("ERROR"):
        lease = await manager.acquire(first)
        assert manager.hold_and_wait_observed == 0
        await manager.acquire(second)

    assert manager.hold_and_wait_observed == 1
    assert any("Hold-and-wait observed" in record.message for record in caplog.records)

    await manager.release(lease)
    await manager.shutdown()


@pytest.mark.asyncio
async def test_released_workflow_slot_does_not_report_hold_and_wait() -> None:
    """The supported multi-phase order, release then acquire, is not reported."""
    backend = FakeBackend(set())
    manager = SandboxManager({"fake": backend}, "fake")

    rollout = await manager.acquire(SandboxSpec(SandboxSource.image("image"), workflow_id="task-1"))
    await manager.release(rollout)
    await manager.acquire(SandboxSpec(SandboxSource.image("image"), workflow_id="task-1"))

    assert manager.hold_and_wait_observed == 0
    await manager.shutdown()


@pytest.mark.asyncio
async def test_a_cancelled_admission_withdraws_its_queued_request() -> None:
    """An abandoned request must not stay queued under an owner that keeps renewing it.

    A queued request occupies a place in its class queue and counts as that class waiting,
    which also blocks every other class from borrowing the elastic pool. If the withdrawal
    were skippable, one cancelled rollout could hold capacity the node had already promised
    to a sandbox that will never be created.
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
    """Returning capacity must not be interruptible.

    A cancellation that lands inside the release skips it, and a lease whose worker is
    still alive has no other recovery path: the owner keeps renewing it, so the coordinator
    holds the node charged and every later sandbox waits out the full admission deadline.
    That window is what the failed end-to-end run fell into, so the release is now awaited
    to completion before the cancellation is re-raised.
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
async def test_a_cancelled_teardown_leaves_the_lease_owned_so_shutdown_retries_it() -> None:
    """Cancellation must not be able to leave a lease half-disposed.

    The capacity is returned by the time the cancellation surfaces, but the disposition is
    not marked complete, so the lease stays owned and `shutdown` retries it. That retry has
    to be clean, because the release is what a worker's teardown runs after a cancellation.
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
    assert lease in manager._leases, (
        "A cancelled teardown claimed the disposition was complete, so no retry can happen."
    )
    assert not lease.released

    await manager.shutdown()

    assert lease.released
    assert not manager._leases
    assert set(capacity.released) == {lease_id}, (
        "The shutdown retry charged or released a different lease."
    )
