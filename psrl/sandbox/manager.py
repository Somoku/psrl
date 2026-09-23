from __future__ import annotations

import asyncio
import logging
import secrets
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from psrl.sandbox.async_utils import complete_cleanup
from psrl.sandbox.capacity import ResourceQuantity
from psrl.sandbox.core import (
    ResourceSpec,
    SandboxBackend,
    SandboxFeature,
    SandboxProvisionError,
    SandboxRef,
    SandboxSession,
    SandboxSpec,
    SandboxStatePolicy,
    SnapshotKind,
    SnapshotRef,
)
from psrl.sandbox.metrics import SandboxMetricsSnapshot

if TYPE_CHECKING:
    from psrl.sandbox.sync import SyncSandboxManager

psrl_logger = logging.getLogger(__file__)

# Destruction attempts before a deferred cleanup is reported as a fault rather than a retry.
_RELEASE_ATTEMPT_ALERT = 5


class SandboxLease:
    """
    Single-owner lifecycle guard for a sandbox session.
    """

    def __init__(
        self,
        session: SandboxSession,
        capacity_lease_id: str | None = None,
        release_capacity: Callable[[str], Awaitable[None]] | None = None,
        on_released: Callable[[SandboxLease], None] | None = None,
        workflow_id: str | None = None,
        on_cleanup_failed: Callable[[SandboxLease], None] | None = None,
    ) -> None:
        self.session = session
        self.workflow_id = workflow_id
        self._capacity_lease_id = capacity_lease_id
        self._release_capacity = release_capacity
        self._terminated = False
        self._released = False
        self._lock = asyncio.Lock()
        self._on_released = on_released
        self._on_cleanup_failed = on_cleanup_failed

    @property
    def ref(self) -> SandboxRef:
        """
        Return the leased sandbox reference.
        """
        return self.session.ref

    @property
    def released(self) -> bool:
        """
        Return whether the exit disposition has already been applied.
        """
        return self._released

    async def release(self) -> None:
        """
        Destroy the sandbox before returning capacity, even during cancellation.

        A manager may take over failed destruction for background reclamation.
        Until destruction succeeds, both ownership and capacity remain held.
        """
        await complete_cleanup(self._release())

    async def _release(self) -> None:
        async with self._lock:
            if self._released:
                return
            if not self._terminated:
                try:
                    await self.session.terminate()
                except Exception:
                    if self._on_cleanup_failed is None:
                        raise
                    self._on_cleanup_failed(self)
                    return
                self._terminated = True
            if self._capacity_lease_id is not None and self._release_capacity is not None:
                try:
                    await self._release_capacity(self._capacity_lease_id)
                except Exception:
                    if self._on_cleanup_failed is None:
                        raise
                    self._on_cleanup_failed(self)
                    return
            self._released = True
            if self._on_released is not None:
                self._on_released(self)

    async def __aenter__(self) -> SandboxSession:
        return self.session

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.release()


@dataclass
class SandboxTask:
    """
    One shared create task and the spec bound to its idempotency key.
    """

    spec: SandboxSpec
    task: asyncio.Task[SandboxLease]
    waiters: int = 0
    delivered: bool = False
    abandoned: bool = False


class SandboxManager:
    """
    Resolve backends and own active sandbox leases for one worker.
    """

    def __init__(
        self,
        backends: Mapping[str, SandboxBackend],
        default_backend: str,
        capacity_coordinator=None,
        capacity_owner_id: str | None = None,
        capacity_heartbeat_interval_s: float | None = None,
    ) -> None:
        self._backends = dict(backends)
        self.default_backend = default_backend
        if default_backend not in self._backends:
            raise ValueError(f"Default sandbox backend {default_backend!r} is not configured.")
        for name, backend in self._backends.items():
            if name != backend.name:
                raise ValueError(f"Sandbox backend key {name!r} does not match backend.name={backend.name!r}.")
        self._leases: set[SandboxLease] = set()
        self._workflow_leases: dict[str, set[SandboxLease]] = {}
        self._idempotent_leases: dict[tuple[str, str], SandboxLease] = {}
        self._create_tasks: dict[tuple[str, str], SandboxTask] = {}
        self._provision_tasks: set[asyncio.Task[SandboxLease]] = set()
        self._create_lock = asyncio.Lock()
        self._closed = False
        self._shutdown_task: asyncio.Task[None] | None = None
        self._pending_releases: set[SandboxLease] = set()
        self._release_attempts: dict[SandboxLease, int] = {}
        self._pending_capacity: dict[str, bool] = {}
        self._reaper_task: asyncio.Task[None] | None = None
        self._pending_workflows: set[str] = set()
        self._capacity_coordinator = capacity_coordinator
        self._capacity_owner_id = capacity_owner_id
        self._capacity_heartbeat_interval_s = capacity_heartbeat_interval_s
        self._capacity_heartbeat_task: asyncio.Task[None] | None = None
        if capacity_coordinator is not None and (not capacity_owner_id or capacity_heartbeat_interval_s is None):
            raise ValueError("Sandbox capacity coordination requires an owner id and heartbeat interval.")

    def _register_lease(self, lease: SandboxLease) -> None:
        """
        Track the lease and its workflow reservation.
        """
        self._leases.add(lease)
        workflow_id = lease.workflow_id
        if workflow_id is None:
            return
        held = self._workflow_leases.setdefault(workflow_id, set())
        held.add(lease)

    def _defer_release(self, lease: SandboxLease) -> None:
        """Take over a failed destruction without releasing its reservation.

        The backend has already proven the container is neither gone nor removable, so the
        memory it holds is real. Releasing the reservation here would over-commit the node
        and turn one stuck container into a host OOM.
        """
        attempts = self._release_attempts.get(lease, 0) + 1
        self._release_attempts[lease] = attempts
        self._pending_releases.add(lease)
        # The episode is over either way, so let its workflow start the next phase.
        self._release_workflow_slot(lease)
        if attempts == 1:
            psrl_logger.warning(
                f"Sandbox cleanup deferred for {lease.ref!r}. Capacity remains reserved.", exc_info=True
            )
        elif attempts >= _RELEASE_ATTEMPT_ALERT:
            psrl_logger.error(
                f"Sandbox {lease.ref!r} has resisted {attempts} destruction attempts. Its node capacity "
                "reservation stays charged until it is removed or this worker exits. Check the Docker daemon "
                "and the container's storage driver.",
                exc_info=True,
            )
        self._ensure_reaper()

    def _ensure_reaper(self) -> None:
        if not self._closed and (self._reaper_task is None or self._reaper_task.done()):
            self._reaper_task = asyncio.create_task(self._reap_releases())

    async def _reap_releases(self) -> None:
        delay = 1.0
        while self._pending_releases or self._pending_capacity:
            await asyncio.sleep(delay)
            results = await asyncio.gather(
                *(lease.release() for lease in list(self._pending_releases)),
                *(self._retry_capacity(lease_id) for lease_id in list(self._pending_capacity)),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, Exception):
                    psrl_logger.warning(f"Sandbox reclamation failed: {result!r}.")
            delay = min(delay * 2, 30.0)

    async def _reclaim_unassigned_capacity(self, lease_id: str, *, cancel: bool = False) -> None:
        self._pending_capacity[lease_id] = cancel
        try:
            await self._retry_capacity(lease_id)
        except Exception:
            psrl_logger.warning(f"Sandbox capacity reclamation deferred for {lease_id!r}.", exc_info=True)
        finally:
            if lease_id in self._pending_capacity:
                self._ensure_reaper()

    async def _retry_capacity(self, lease_id: str) -> None:
        cancel = self._pending_capacity[lease_id]
        method = self._capacity_coordinator.cancel if cancel else self._capacity_coordinator.release
        await complete_cleanup(method.remote(lease_id))
        self._pending_capacity.pop(lease_id, None)

    def _reserve_workflow(self, workflow_id: str | None) -> None:
        """Hold one sandbox phase per workflow, counting requests still waiting.

        Only a phase that is still running blocks the next one. Cleanup this manager has
        taken over does not, because its reservation already protects the node.
        """
        if workflow_id is None:
            return
        if workflow_id in self._pending_workflows or self._workflow_leases.get(workflow_id):
            raise RuntimeError(
                f"Workflow {workflow_id!r} must release its sandbox before requesting another, or its phases "
                "will starve each other's resource class."
            )
        self._pending_workflows.add(workflow_id)

    def backend(self, name: str | None = None) -> SandboxBackend:
        """
        Resolve a configured backend by name or use the default backend.
        """
        backend_name = name or self.default_backend
        return self._backends[backend_name]

    def sync(
        self,
        backend: str | None = None,
    ) -> SyncSandboxManager:
        """
        Create a facade for synchronous code running outside the current event loop.
        """
        from psrl.sandbox.sync import SyncSandboxManager

        self._require_open()
        self.backend(backend)
        return SyncSandboxManager(self, asyncio.get_running_loop(), backend=backend)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Sandbox manager is closed.")

    async def _acquire_capacity(
        self,
        backend: SandboxBackend,
        resources: ResourceSpec | None,
        resource_class: str = "default",
    ) -> str | None:
        """
        Admit one node-local resource request through the shared coordinator.

        The coordinator owns the queue deadline and reports an unserved request as
        a ``SandboxCapacityTimeout``, so the caller never has to guess whether a
        failed wait was a timeout or a cancellation.
        """
        if not backend.uses_node_capacity or self._capacity_coordinator is None:
            return None
        if resources is None:
            raise ValueError("Node-capacity admission requires a ResourceSpec.")
        request = ResourceQuantity.from_spec(resources)
        if self._capacity_heartbeat_task is None:
            self._capacity_heartbeat_task = asyncio.create_task(self._heartbeat_capacity_owner())
        lease_id = uuid.uuid4().hex
        try:
            await self._capacity_coordinator.acquire.remote(
                lease_id,
                self._capacity_owner_id,
                request.memory_mb,
                request.cpu_millis / 1000,
                resource_class,
            )
        except BaseException:
            # An RPC failure can race a grant, so withdraw the reservation explicitly.
            await complete_cleanup(self._reclaim_unassigned_capacity(lease_id, cancel=True))
            raise
        return lease_id

    async def _release_capacity(self, lease_id: str) -> None:
        """Return one capacity lease, completing even if this caller is cancelled.

        Every release path in this module goes through here, including the one on
        `SandboxLease`, so no acquire failure or cancellation can leave the envelope
        charged for a sandbox that does not exist.
        """
        await complete_cleanup(self._capacity_coordinator.release.remote(lease_id))

    async def _heartbeat_capacity_owner(self) -> None:
        while True:
            await asyncio.sleep(self._capacity_heartbeat_interval_s)
            try:
                await self._capacity_coordinator.renew_owner.remote(self._capacity_owner_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                psrl_logger.warning(
                    f"Sandbox capacity heartbeat failed for owner {self._capacity_owner_id!r}.",
                    exc_info=True,
                )

    async def _close_capacity(self) -> None:
        if self._capacity_heartbeat_task is not None:
            self._capacity_heartbeat_task.cancel()
            await asyncio.gather(self._capacity_heartbeat_task, return_exceptions=True)
        await self._capacity_coordinator.release_owner.remote(self._capacity_owner_id)
        self._pending_capacity.clear()

    async def prepare(
        self,
        spec: SandboxSpec,
        backend: str | None = None,
    ) -> None:
        """
        Warm backend artifacts without reserving container CPU or memory.

        Callers may schedule this alongside rollout and must join or cancel
        their preparation task before releasing task ownership.
        """
        self._require_open()
        await self.backend(backend).prepare(spec)

    async def acquire(
        self,
        spec: SandboxSpec,
        backend: str | None = None,
    ) -> SandboxLease:
        """
        Provision one owned sandbox, sharing only identical idempotent requests.
        """
        self._require_open()
        selected = self.backend(backend)
        key = (selected.name, spec.idempotency_key) if spec.idempotency_key else None
        async with self._create_lock:
            self._require_open()
            operation = self._create_tasks.get(key) if key else None
            if operation is not None and operation.abandoned:
                raise RuntimeError("The sandbox idempotency key is still being reclaimed.")
            existing = self._idempotent_leases.get(key)
            if existing is not None and not existing.released:
                if existing.session.spec != spec:
                    raise RuntimeError("A sandbox idempotency key cannot be reused with a different spec.")
                if existing in self._pending_releases:
                    raise RuntimeError("The sandbox idempotency key is still being reclaimed.")
                return existing
            if operation is None:
                operation = SandboxTask(spec, asyncio.create_task(self._acquire_new(spec, selected, key)))
                if key:
                    self._create_tasks[key] = operation
                self._provision_tasks.add(operation.task)
            elif operation.spec != spec:
                raise RuntimeError("A sandbox idempotency key cannot be reused with a different spec.")
            operation.waiters += 1
        try:
            lease = await asyncio.shield(operation.task)
            operation.delivered = True
            return lease
        finally:
            operation.waiters -= 1
            if operation.waiters == 0:
                try:
                    if not operation.delivered:
                        operation.abandoned = True
                        await complete_cleanup(self._abandon_create(operation.task))
                finally:
                    if key and self._create_tasks.get(key) is operation:
                        self._create_tasks.pop(key)
                    self._provision_tasks.discard(operation.task)

    async def _abandon_create(self, task: asyncio.Task[SandboxLease]) -> None:
        if not task.done():
            task.cancel()
        results = await asyncio.gather(task, return_exceptions=True)
        if isinstance(results[0], SandboxLease):
            await results[0].release()

    async def _acquire_new(
        self,
        spec: SandboxSpec,
        selected: SandboxBackend,
        task_key: tuple[str, str] | None = None,
    ) -> SandboxLease:
        required = set(spec.required_features)
        if spec.mounts:
            required.add(SandboxFeature.HOST_MOUNT)
        selected.capabilities.require(*required)
        await selected.prepare(spec)
        return await self._provision(
            selected,
            spec.resources,
            spec.resource_class,
            spec.workflow_id,
            lambda: selected.create(spec),
            task_key=task_key,
            required=required,
        )

    async def _provision(
        self,
        backend: SandboxBackend,
        resources: ResourceSpec | None,
        resource_class: str,
        workflow_id: str | None,
        create: Callable[[], Awaitable[SandboxSession]],
        *,
        task_key: tuple[str, str] | None = None,
        required: set[SandboxFeature] | None = None,
        policy: SandboxStatePolicy | None = None,
    ) -> SandboxLease:
        """
        Own admission, provisioning, validation, and rollback as one transaction.
        """
        self._reserve_workflow(workflow_id)
        capacity_id = None
        lease = None
        try:
            capacity_id = await self._acquire_capacity(backend, resources, resource_class)
            try:
                creation = asyncio.ensure_future(create())
                cancelled = False
                try:
                    session = await asyncio.shield(creation)
                except asyncio.CancelledError:
                    cancelled = True
                    try:
                        await complete_cleanup(asyncio.gather(creation, return_exceptions=True))
                    except asyncio.CancelledError:
                        pass
                    session = creation.result()
            except SandboxProvisionError as exc:
                lease = self._adopt(exc.session, capacity_id, workflow_id, task_key)
                raise
            lease = self._adopt(session, capacity_id, workflow_id, task_key)
            if cancelled:
                raise asyncio.CancelledError
            session.capabilities.require(*(required or ()))
            if policy is not None:
                await self._sanitize_restored_session(session, policy)
            return lease
        except BaseException:
            if lease is not None:
                await lease.release()
            elif capacity_id is not None:
                await complete_cleanup(self._reclaim_unassigned_capacity(capacity_id))
            raise
        finally:
            self._pending_workflows.discard(workflow_id)

    def _adopt(
        self,
        session: SandboxSession,
        capacity_id: str | None,
        workflow_id: str | None,
        task_key: tuple[str, str] | None = None,
    ) -> SandboxLease:
        lease = SandboxLease(
            session,
            capacity_id,
            self._release_capacity,
            on_released=lambda released: self._forget_lease(released, task_key),
            workflow_id=workflow_id,
            on_cleanup_failed=self._defer_release,
        )
        self._register_lease(lease)
        if task_key is not None:
            self._idempotent_leases[task_key] = lease
        return lease

    async def _run_owned(self, operation: Awaitable[SandboxLease]) -> SandboxLease:
        task = asyncio.create_task(operation)
        self._provision_tasks.add(task)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await complete_cleanup(self._abandon_create(task))
            raise
        finally:
            self._provision_tasks.discard(task)

    def _release_workflow_slot(self, lease: SandboxLease) -> None:
        """Free a workflow's phase slot without giving up physical ownership.

        The slot exists to stop one workflow from holding the sandbox its next phase needs.
        That is a logical constraint, and the physical resource stays protected by the
        capacity reservation, so a cleanup this manager is still retrying must not also
        block the next phase and fail the episode.
        """
        if lease.workflow_id is None:
            return
        held = self._workflow_leases.get(lease.workflow_id)
        if held is None:
            return
        held.discard(lease)
        if not held:
            self._workflow_leases.pop(lease.workflow_id, None)

    def _forget_lease(
        self,
        lease: SandboxLease,
        task_key: tuple[str, str] | None = None,
    ) -> None:
        """
        Forget a released lease, its workflow slot, and its idempotency key.
        """
        self._leases.discard(lease)
        self._pending_releases.discard(lease)
        self._release_attempts.pop(lease, None)
        self._release_workflow_slot(lease)
        if task_key is not None and self._idempotent_leases.get(task_key) is lease:
            self._idempotent_leases.pop(task_key, None)

    async def connect(
        self,
        ref: SandboxRef,
        resources: ResourceSpec | None = None,
    ) -> SandboxLease:
        """
        Reconnect to a backend-owned sandbox and assume lifecycle ownership.
        """
        self._require_open()
        backend = self.backend(ref.backend)
        async with self._create_lock:
            self._require_open()
            for lease in self._leases:
                if lease.ref == ref:
                    if lease in self._pending_releases:
                        raise RuntimeError("The sandbox is still being reclaimed.")
                    if resources is not None and lease.session.spec is not None:
                        if resources != lease.session.spec.resources:
                            raise ValueError("Connected sandbox resources do not match the owned session.")
                    return lease
            return await self._run_owned(
                self._provision(
                    backend,
                    resources,
                    "default",
                    None,
                    lambda: backend.connect(ref.sandbox_id),
                )
            )

    async def restore(
        self,
        snapshot: SnapshotRef,
        spec: SandboxSpec | None = None,
        state_policy: SandboxStatePolicy | None = None,
    ) -> SandboxLease:
        """
        Restore a snapshot through its owning backend.
        """
        self._require_open()
        backend = self.backend(snapshot.backend)
        backend.capabilities.require(SandboxFeature.RESTORE)
        policy = state_policy or (spec.state_policy if spec is not None else SandboxStatePolicy())
        if not policy.enabled:
            raise RuntimeError("Sandbox restore requires an explicitly enabled SandboxStatePolicy.")
        resources = spec.resources if spec is not None else self._resources_from_snapshot(snapshot)
        return await self._run_owned(
            self._provision(
                backend,
                resources,
                spec.resource_class if spec else "default",
                spec.workflow_id if spec else None,
                lambda: backend.restore(snapshot, spec),
                policy=policy,
            )
        )

    async def branch(
        self,
        session: SandboxSession,
        snapshot_kind: SnapshotKind = SnapshotKind.FULL_STATE,
        state_policy: SandboxStatePolicy | None = None,
    ) -> SandboxLease:
        """
        Branch state without silently weakening snapshot semantics.
        """
        self._require_open()
        policy = state_policy or (session.spec.state_policy if session.spec is not None else SandboxStatePolicy())
        self._validate_state_policy(session, policy)
        if session.capabilities.supports(SandboxFeature.NATIVE_FORK):
            backend = self.backend(session.ref.backend)
            return await self._run_owned(
                self._provision(
                    backend,
                    session.spec.resources,
                    session.spec.resource_class,
                    session.spec.workflow_id,
                    session.fork,
                    policy=policy,
                )
            )

        snapshot_feature = {
            SnapshotKind.FILESYSTEM: SandboxFeature.FILESYSTEM_SNAPSHOT,
            SnapshotKind.FULL_STATE: SandboxFeature.FULL_STATE_SNAPSHOT,
        }[snapshot_kind]
        session.capabilities.require(snapshot_feature, SandboxFeature.RESTORE)
        snapshot = await self.checkpoint(session, snapshot_kind, policy)
        try:
            child = await self.restore(snapshot, spec=replace(session.spec, idempotency_key=None), state_policy=policy)
        except BaseException as restore_error:
            try:
                await self.delete_snapshot(snapshot)
            except BaseException as delete_error:
                restore_error.add_note(f"Temporary snapshot cleanup also failed: {delete_error!r}")
            raise
        try:
            await self.delete_snapshot(snapshot)
        except BaseException:
            await child.release()
            raise
        return child

    async def checkpoint(
        self,
        session: SandboxSession,
        kind: SnapshotKind,
        state_policy: SandboxStatePolicy | None = None,
    ) -> SnapshotRef:
        """
        Create a capability-gated snapshot after enforcing RL safety policy.
        """
        policy = state_policy or (session.spec.state_policy if session.spec is not None else SandboxStatePolicy())
        self._validate_state_policy(session, policy)
        feature = {
            SnapshotKind.FILESYSTEM: SandboxFeature.FILESYSTEM_SNAPSHOT,
            SnapshotKind.FULL_STATE: SandboxFeature.FULL_STATE_SNAPSHOT,
        }[kind]
        session.capabilities.require(feature, SandboxFeature.RESTORE)
        try:
            snapshot = await session.snapshot(kind)
        except BaseException as snapshot_error:
            # Snapshotting invalidates command streams on E2B-compatible
            # providers. Reconnect the live parent before its next command.
            try:
                await session.refresh_transport()
            except BaseException as refresh_error:
                snapshot_error.add_note(f"Parent transport refresh also failed: {refresh_error!r}")
            raise
        try:
            await session.refresh_transport()
        except BaseException as refresh_error:
            try:
                await self.delete_snapshot(snapshot)
            except BaseException as delete_error:
                refresh_error.add_note(f"Snapshot cleanup also failed: {delete_error!r}")
            raise
        spec = session.spec
        metadata = dict(snapshot.metadata)
        metadata.update(
            {
                "psrl.network_connections_restored": False,
                "psrl.external_side_effects_restored": False,
            }
        )
        if spec is not None:
            metadata.update(
                {
                    "psrl.source_kind": spec.source.kind.value,
                    "psrl.source_reference": spec.source.reference,
                    "psrl.cpu_count": spec.resources.cpu_count,
                    "psrl.memory_mb": spec.resources.memory_mb,
                    "psrl.disk_mb": spec.resources.disk_mb,
                }
            )
        return replace(snapshot, metadata=metadata)

    @staticmethod
    def _validate_state_policy(session: SandboxSession, policy: SandboxStatePolicy) -> None:
        if not policy.enabled:
            raise RuntimeError("Sandbox state operations require an explicitly enabled SandboxStatePolicy.")
        spec = session.spec
        if spec is None:
            raise RuntimeError("Sandbox state safety cannot be verified for a session without its creation spec.")
        if not policy.allow_external_side_effects and session.command_count:
            raise RuntimeError(
                "Sandbox state capture after user commands requires allow_external_side_effects=True because external "
                "effects cannot be rolled back."
            )
        if policy.allow_secret_capture:
            return
        secret_markers = ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "API_KEY", "PRIVATE_KEY", "CREDENTIAL")
        sensitive_keys = [key for key in spec.env if any(marker in key.upper() for marker in secret_markers)]
        proxy_secrets = [
            key for key, value in spec.env.items() if "PROXY" in key.upper() and urlsplit(value).username is not None
        ]
        if sensitive_keys or proxy_secrets:
            names = ", ".join(sorted(set(sensitive_keys + proxy_secrets)))
            raise RuntimeError(f"Sandbox snapshot would capture secret-bearing environment variables: {names}.")

    @staticmethod
    async def _sanitize_restored_session(session: SandboxSession, policy: SandboxStatePolicy) -> None:
        """
        Reset stale transports and mix host entropy into a restored microVM.
        """
        await session.refresh_transport()
        if not policy.reseed_after_restore:
            return
        seed_path = f"/tmp/psrl-branch-seed-{secrets.token_hex(8)}"
        await session.write_bytes(seed_path, secrets.token_bytes(64))
        result = await session.exec(
            f"cat {seed_path} > /dev/urandom && rm -f {seed_path}",
            timeout_s=10,
        )
        if result.exit_code != 0:
            raise RuntimeError(f"Restored sandbox entropy reseed failed: {result.stderr.strip()}.")

    async def delete_snapshot(self, snapshot: SnapshotRef) -> None:
        """
        Delete a snapshot through its owning backend.
        """
        await self.backend(snapshot.backend).delete_snapshot(snapshot)

    @staticmethod
    def _resources_from_snapshot(snapshot: SnapshotRef) -> ResourceSpec:
        """
        Recover the original resource request recorded by checkpoint().
        """
        return ResourceSpec(
            cpu_count=snapshot.metadata.get("psrl.cpu_count"),
            memory_mb=snapshot.metadata.get("psrl.memory_mb"),
            disk_mb=snapshot.metadata.get("psrl.disk_mb"),
        )

    async def release(self, lease: SandboxLease) -> None:
        """
        Release one lease and remove it from worker ownership.
        """
        await lease.release()

    def metrics_snapshot(self) -> dict[str, SandboxMetricsSnapshot]:
        """
        Return one metrics snapshot per configured backend.
        """
        return {name: backend.metrics_snapshot() for name, backend in self._backends.items()}

    def ownership_snapshot(self) -> dict[str, int]:
        """
        Report worker ownership, including cleanup that still holds capacity.
        """
        return {
            "leases": len(self._leases),
            "provisioning": len(self._provision_tasks),
            "pending_releases": len(self._pending_releases),
            "pending_capacity": len(self._pending_capacity),
            "max_release_attempts": max(self._release_attempts.values(), default=0),
        }

    async def shutdown(self) -> None:
        """
        Cancel admission, reclaim owned sessions, and close backend transports.
        """
        if self._shutdown_task is None:
            self._closed = True
            self._shutdown_task = asyncio.create_task(self._shutdown())
        await complete_cleanup(self._shutdown_task)

    async def _shutdown(self) -> None:
        tasks = list(self._provision_tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._reaper_task is not None:
            self._reaper_task.cancel()
            await asyncio.gather(self._reaper_task, return_exceptions=True)
        results = await asyncio.gather(*(lease.release() for lease in list(self._leases)), return_exceptions=True)
        results.extend(
            await asyncio.gather(
                *(backend.shutdown() for backend in self._backends.values()),
                return_exceptions=True,
            )
        )
        # Backend shutdown may have completed a previously failed destruction.
        results.extend(
            await asyncio.gather(*(lease.release() for lease in list(self._leases)), return_exceptions=True)
        )
        results.extend(
            await asyncio.gather(
                *(self._retry_capacity(lease_id) for lease_id in list(self._pending_capacity)),
                return_exceptions=True,
            )
        )
        if self._capacity_coordinator is not None:
            if self._leases:
                # Keep reservations charged until owner expiry if destruction is unconfirmed.
                if self._capacity_heartbeat_task is not None:
                    self._capacity_heartbeat_task.cancel()
                    await asyncio.gather(self._capacity_heartbeat_task, return_exceptions=True)
            else:
                results.extend(await asyncio.gather(self._close_capacity(), return_exceptions=True))
        self._create_tasks.clear()
        self._provision_tasks.clear()
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors or self._leases or self._pending_capacity:
            raise RuntimeError(
                f"Sandbox shutdown has {len(errors)} failure(s), {len(self._leases)} unreclaimed lease(s), "
                f"and {len(self._pending_capacity)} pending capacity return(s)."
            ) from (errors[0] if errors else None)
