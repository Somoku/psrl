"""Sandbox lease ownership, admission, and the workflow phase reservation."""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from psrl.sandbox.async_utils import complete_cleanup
from psrl.sandbox.capacity import CapacityGrant, ResourceQuantity
from psrl.sandbox.core import (
    ResourceSpec,
    SandboxBackend,
    SandboxBusyError,
    SandboxCapabilityError,
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
from psrl.sandbox.ownership import LeaseState, LeaseStateMachine
from psrl.sandbox.prefetch import PrefetchPlan, PrefetchReport

if TYPE_CHECKING:
    from psrl.sandbox.sync import SyncSandboxManager

psrl_logger = logging.getLogger(__file__)

# Destruction attempts before a deferred cleanup is reported as a fault rather than a retry.
_RELEASE_ATTEMPT_ALERT = 5

# The resource class a fork parent is admitted under. It is not a trajectory, so it
# must not be able to consume a class share a grader or a rollout is waiting for.
_FORK_PARENT_CLASS = "prepare"


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
        state_machine: LeaseStateMachine | None = None,
    ) -> None:
        self.session = session
        self.workflow_id = workflow_id
        self._capacity_lease_id = capacity_lease_id
        self._release_capacity = release_capacity
        self._lock = asyncio.Lock()
        self._on_released = on_released
        self._on_cleanup_failed = on_cleanup_failed
        self._state = state_machine or LeaseStateMachine()
        self._capacity_released = False

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
        return self._state.terminal

    @property
    def state(self) -> LeaseState:
        """
        Return the lease's position in its lifecycle.
        """
        return self._state.state

    @property
    def state_machine(self) -> LeaseStateMachine:
        """
        Return the lifecycle machine, which exposes the trail a stuck lease needs.
        """
        return self._state

    async def release(self) -> None:
        """
        Destroy the sandbox before returning capacity, even during cancellation.

        A manager may take over failed destruction for background reclamation.
        Until destruction succeeds, both ownership and capacity remain held.
        """
        await complete_cleanup(self._release())

    async def _release(self) -> None:
        async with self._lock:
            if self._state.terminal:
                return
            if self._state.state is not LeaseState.RECLAIMING:
                self._state.transition(LeaseState.RECLAIMING)
            try:
                await self.session.terminate()
            except Exception:
                if self._on_cleanup_failed is None:
                    raise
                self._on_cleanup_failed(self)
                return
            if self._capacity_lease_id is not None and self._release_capacity is not None:
                try:
                    await self._release_capacity(self._capacity_lease_id)
                except Exception:
                    if self._on_cleanup_failed is None:
                        raise
                    self._on_cleanup_failed(self)
                    return
                self._capacity_released = True
            self._state.transition(LeaseState.RELEASED)
            if self._on_released is not None:
                self._on_released(self)

    async def __aenter__(self) -> SandboxSession:
        return self.session

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.release()


class AdmissionGate:
    """The node's capacity envelope, reached the same way whether it is local or remote.

    The manager holds this rather than a Ray actor handle, so admission is a library
    by default and a remote call only when the envelope belongs to another process.
    A caller cannot tell the difference, which is what keeps the ownership code free
    of deployment knowledge.
    """

    def __init__(self, coordinator) -> None:
        self._coordinator = coordinator
        self._remote = hasattr(coordinator, "acquire") and hasattr(coordinator.acquire, "remote")

    async def acquire(self, *args):
        """
        Admit one request and return its grant.
        """
        if self._remote:
            return await self._coordinator.acquire.remote(*args)
        return await self._coordinator.acquire(*args)

    async def release(self, lease_id: str) -> None:
        """
        Return one admitted vector.
        """
        if self._remote:
            await self._coordinator.release.remote(lease_id)
            return
        await self._coordinator.release(lease_id)

    async def cancel(self, lease_id: str) -> None:
        """
        Withdraw a request that was never admitted.
        """
        if self._remote:
            await self._coordinator.cancel.remote(lease_id)
            return
        await self._coordinator.cancel(lease_id)

    async def renew_owner(self, owner_id: str) -> None:
        """
        Refresh one owner's lease.
        """
        if self._remote:
            await self._coordinator.renew_owner.remote(owner_id)
            return
        await self._coordinator.renew_owner(owner_id)

    async def release_owner(self, owner_id: str) -> None:
        """
        Release everything one owner held.
        """
        if self._remote:
            await self._coordinator.release_owner.remote(owner_id)
            return
        await self._coordinator.release_owner(owner_id)

    async def snapshot(self) -> dict:
        """
        Return the envelope's accounting.
        """
        if self._remote:
            return await self._coordinator.snapshot.remote()
        return await self._coordinator.snapshot()


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
        self._capacity_coordinator = AdmissionGate(capacity_coordinator) if capacity_coordinator else None
        self._capacity_owner_id = capacity_owner_id
        self._capacity_heartbeat_interval_s = capacity_heartbeat_interval_s
        self._capacity_heartbeat_task: asyncio.Task[None] | None = None
        # Counters behind the metric hook. The plane reports, and it never logs a metric.
        self._counters: dict[str, int] = {}
        self._cumulative_wait_s = 0.0
        self._wait_samples = 0
        self._idle_pause_window_s: float | None = None
        self._idle_reap_window_s: float | None = None
        self._default_lifetime_s: float | None = None
        self._idle_task: asyncio.Task[None] | None = None
        if capacity_coordinator is not None and (not capacity_owner_id or capacity_heartbeat_interval_s is None):
            raise ValueError("Sandbox capacity coordination requires an owner id and heartbeat interval.")

    def _count(self, key: str, amount: int = 1) -> None:
        """
        Increment one outcome counter.
        """
        self._counters[key] = self._counters.get(key, 0) + amount

    def configure_lifetime(self, lifetime_s: float | None) -> None:
        """
        Set the absolute lifetime applied to a spec that does not name one.

        The lifetime is the backstop behind the reaper: a sandbox that keeps
        running commands and is therefore never idle still has to end, or a stuck
        episode holds a node slot for the rest of the run.
        """
        if lifetime_s is not None and lifetime_s <= 0:
            raise ValueError("Sandbox lifetime_s must be greater than zero when set.")
        self._default_lifetime_s = lifetime_s

    def configure_idle_policy(self, pause_window_s: float | None, reap_window_s: float | None) -> None:
        """Set the idle windows that release compute and reclaim leaks.

        The pause window is shorter than the reap window, so an idle sandbox is
        paused before it is destroyed and a running episode is never the first
        thing the reaper sees. The ordering is asserted here rather than described,
        because a configuration that inverts it produces a slow leak or a sandbox
        reclaimed while in use, and neither failure points at the configuration.
        """
        for name, window in (("pause_window_s", pause_window_s), ("reap_window_s", reap_window_s)):
            if window is not None and window <= 0:
                raise ValueError(f"Sandbox idle {name} must be greater than zero when set.")
        if pause_window_s is not None and reap_window_s is not None and pause_window_s >= reap_window_s:
            raise ValueError(
                f"Sandbox idle pause_window_s ({pause_window_s:g}) must be shorter than reap_window_s "
                f"({reap_window_s:g}), or a sandbox is destroyed before it is ever paused."
            )
        self._idle_pause_window_s = pause_window_s
        self._idle_reap_window_s = reap_window_s

    def _register_lease(self, lease: SandboxLease) -> None:
        """
        Track the lease, its workflow reservation, and the idle sweep it needs.
        """
        self._leases.add(lease)
        # Started by the first lease, so a deployment that configures idle windows does
        # not silently lose pause-on-idle and leak reclamation.
        self.start_idle_reaper()
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
        await complete_cleanup(method(lease_id))
        self._pending_capacity.pop(lease_id, None)

    def _reserve_workflow(self, workflow_id: str | None) -> None:
        """Hold one sandbox phase per workflow, counting requests still waiting.

        Only a phase that is still running blocks the next one. Cleanup this manager has
        taken over does not, because its reservation already protects the node.

        A group is several workflows, one per member, because `workflow_id` and
        `idempotency_key` are per trajectory. The fork parent carries no workflow at
        all, so it can neither collide with a member's reservation nor hold a slot a
        member needs.
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
        """Resolve a configured backend by name or use the default backend.

        None means the default. An empty name is a caller that lost the name it meant
        to pass, so it is refused rather than falling through to the default, which
        would restore one backend's snapshot on another.

        Raises:
            ValueError: When the name is empty.
            KeyError: When no backend is registered under the name.
        """
        if name is not None and not name.strip():
            raise ValueError(
                "A sandbox backend name cannot be empty. Pass None for the default, because an empty name "
                "is a lost reference rather than a request for the default."
            )
        return self._backends[name or self.default_backend]

    def select_backend(self, spec: SandboxSpec, name: str | None = None) -> SandboxBackend:
        """Choose the backend that will serve a spec.

        An explicit name, a spec pin, or the first backend whose declaration covers
        what the spec requires. The default backend is tried first so a deployment
        with one backend never pays for the search.

        Raises:
            SandboxCapabilityError: When no configured backend can serve the spec,
                naming what was missing rather than degrading to a weaker one.
        """
        pinned = name or spec.backend
        if pinned is not None:
            if pinned not in self._backends:
                raise SandboxCapabilityError(
                    f"Sandbox backend {pinned!r} is not configured (configured: {sorted(self._backends)})."
                )
            return self._backends[pinned]
        required = self._required_features(spec)
        ordered = [self.default_backend, *sorted(name for name in self._backends if name != self.default_backend)]
        missing: dict[str, str] = {}
        for candidate in ordered:
            backend = self._backends[candidate]
            problem = self._capability_gap(backend, spec, required)
            if problem is None:
                return backend
            missing[candidate] = problem
        raise SandboxCapabilityError(
            f"No configured sandbox backend can serve this spec. Reasons: {missing}. "
            "Declare the requirement on a backend that supports it, or pin one."
        )

    @staticmethod
    def _required_features(spec: SandboxSpec) -> set[SandboxFeature]:
        """
        Return every capability a spec needs, derived rather than restated.
        """
        required = set(spec.required_features)
        if spec.mounts:
            required.add(SandboxFeature.HOST_MOUNT)
        if spec.volumes:
            required.add(SandboxFeature.VOLUME)
        if spec.egress is not None:
            required.add(SandboxFeature.EGRESS_POLICY)
        if spec.credentials:
            required.add(SandboxFeature.CREDENTIAL_INJECTION)
        if spec.credential_bindings:
            # A binding is only useful with a policy to reach its host and a broker to
            # attach the value, so a backend that lacks either is not a candidate.
            required.add(SandboxFeature.CREDENTIAL_INJECTION)
            required.add(SandboxFeature.EGRESS_POLICY)
        return required

    @staticmethod
    def _capability_gap(
        backend: SandboxBackend,
        spec: SandboxSpec,
        required: set[SandboxFeature],
    ) -> str | None:
        """Return why one backend cannot serve a spec, or None when it can.

        The resume level is checked here as well as at provisioning, because a
        selection that ignores it would hand a full-state requirement to a backend
        that only restores a filesystem.
        """
        capabilities = backend.capabilities
        absent = sorted(feature.value for feature in required - capabilities.features)
        if absent:
            return f"missing features {absent}"
        if spec.required_resume_level is not None:
            level = capabilities.resume_level
            if level is None or not level.satisfies(spec.required_resume_level):
                return (
                    f"resumes at {level.value if level else 'no declared level'}, which does not satisfy "
                    f"{spec.required_resume_level.value}"
                )
        return None

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
        *,
        deadline_s: float | None = None,
    ) -> CapacityGrant | None:
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
        started_at = time.monotonic()
        try:
            grant = await self._capacity_coordinator.acquire(
                lease_id,
                self._capacity_owner_id,
                request.memory_mb,
                request.cpu_millis / 1000,
                resource_class,
                deadline_s,
                request.gpu_count,
                request.disk_mb,
            )
        except BaseException:
            # An RPC failure can race a grant, so withdraw the reservation explicitly.
            await complete_cleanup(self._reclaim_unassigned_capacity(lease_id, cancel=True))
            raise
        self._cumulative_wait_s += time.monotonic() - started_at
        self._wait_samples += 1
        return CapacityGrant(
            lease_id=grant.lease_id,
            resources=grant.resources,
            gpu_indices=tuple(grant.gpu_indices),
        )

    async def _release_capacity(self, lease_id: str) -> None:
        """Return one capacity lease, completing even if this caller is cancelled.

        Every release path in this module goes through here, including the one on
        `SandboxLease`, so no acquire failure or cancellation can leave the envelope
        charged for a sandbox that does not exist.
        """
        await complete_cleanup(self._capacity_coordinator.release(lease_id))

    async def _heartbeat_capacity_owner(self) -> None:
        while True:
            await asyncio.sleep(self._capacity_heartbeat_interval_s)
            try:
                await self._capacity_coordinator.renew_owner(self._capacity_owner_id)
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
        await self._capacity_coordinator.release_owner(self._capacity_owner_id)
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
        selected = self.select_backend(spec, backend)
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

    async def acquire_group(
        self,
        specs: Sequence[SandboxSpec],
        backend: str | None = None,
    ) -> list[SandboxLease]:
        """Provision one GRPO group as independent leases.

        A group is a completion unit, not a scheduling unit: the algorithm needs
        its trajectories to finish together, not to start together. Members are
        therefore admitted independently, and no member holds capacity while
        waiting for a sibling, which is what makes two concurrent groups unable to
        starve each other.

        Args:
            specs (Sequence[SandboxSpec]): One spec per member, differing only in
                `workflow_id`, `idempotency_key`, and other per-trajectory identity.
            backend (str | None): Backend name, or the configured default.

        Returns:
            list[SandboxLease]: One lease per member, in the order given.

        Raises:
            ValueError: When the members do not agree on a source and a resource
                request, or when two members share an idempotency key.
        """
        members = list(specs)
        if not members:
            raise ValueError("A sandbox group requires at least one member spec.")
        if len(members) == 1:
            return [await self.acquire(members[0], backend=backend)]
        self._validate_group(members)
        selected = self.select_backend(members[0], backend)
        # The fork path is a provider path. A backend that also charges local node
        # capacity admits members instead, because a fork makes ungranted children.
        if selected.capabilities.supports(SandboxFeature.NATIVE_FORK) and not selected.uses_node_capacity:
            return await self._acquire_group_by_fork(members, selected)
        return await self._acquire_group_by_create(members, selected)

    @staticmethod
    def _validate_group(members: Sequence[SandboxSpec]) -> None:
        """Refuse a group whose members cannot share one prepared environment.

        A fork clones one sandbox, so members that want different images or
        different resources have no common parent, and a caller that believes they
        do would silently train on the wrong environment.
        """
        first = members[0]
        for index, member in enumerate(members[1:], start=1):
            if member.source != first.source:
                raise ValueError(
                    f"Sandbox group member {index} has source {member.source!r} but member 0 has "
                    f"{first.source!r}. A group shares one prepared environment."
                )
            if member.resources != first.resources:
                raise ValueError(
                    f"Sandbox group member {index} requests {member.resources!r} but member 0 requests "
                    f"{first.resources!r}. A group shares one prepared environment."
                )
        keys = [member.idempotency_key for member in members if member.idempotency_key]
        if len(set(keys)) != len(keys):
            raise ValueError(
                "Sandbox group members must not share an idempotency key, or a retried acquire would "
                "hand two trajectories the same sandbox."
            )

    async def _acquire_group_by_create(
        self,
        members: Sequence[SandboxSpec],
        backend: SandboxBackend,
    ) -> list[SandboxLease]:
        """Provision each member independently, which is the portable group path.

        Members are created concurrently so the window in which one member holds
        capacity while a sibling is still queued stays short, and a member that
        cannot be admitted fails on its own. Any failure releases the members that
        did succeed, so a failed group leaves nothing charged.
        """
        results = await asyncio.gather(
            *(self.acquire(member, backend=backend.name) for member in members),
            return_exceptions=True,
        )
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            leases = [result for result in results if isinstance(result, SandboxLease)]
            await asyncio.gather(*(lease.release() for lease in leases), return_exceptions=True)
            self._count("outcome/group_failed")
            raise failures[0]
        return list(results)

    async def _acquire_group_by_fork(
        self,
        members: Sequence[SandboxSpec],
        backend: SandboxBackend,
    ) -> list[SandboxLease]:
        """Prepare once and fork the group from one parent, keeping one member recoverable.

        The parent belongs to no trajectory: it carries no workflow and no
        idempotency key, and it is admitted under its own resource class, so it can
        neither collide with a member's reservation nor consume a class share a
        grader is waiting for. It is destroyed as soon as the children are adopted,
        because its slot is held for the fork and not for the episode.

        One member is created directly rather than forked. A forked child carries no
        idempotency key, so a lost response leaves it unrecoverable. The directly
        created member is an ordinary idempotent acquire, so a retried group acquire
        adopts it instead of building a second group for one prompt.
        """
        parent_spec = replace(
            members[0],
            workflow_id=None,
            idempotency_key=None,
            resource_class=_FORK_PARENT_CLASS,
        )
        parent = await self._provision(
            backend,
            parent_spec.resources,
            parent_spec.resource_class,
            None,
            lambda grant: backend.create(self._apply_grant(parent_spec, grant)),
        )
        child_count = len(members) - 1
        children: list[SandboxSession] = []
        leases: list[SandboxLease] = []
        try:
            children = list(await parent.session.fork(child_count))
            if len(children) != child_count:
                raise RuntimeError(f"Sandbox fork produced {len(children)} child(ren) for a group of {len(members)}.")
            try:
                for member, child in zip(members, children, strict=False):
                    # A fork clones memory, so a child that skipped sanitization would
                    # start from the parent's random stream and transport state.
                    await self._sanitize_restored_session(child, member.state_policy)
                    leases.append(self._adopt(child, None, member.workflow_id, None))
                leases.append(await self.acquire(members[-1], backend=backend.name))
            except BaseException:
                await asyncio.gather(*(lease.release() for lease in leases), return_exceptions=True)
                self._count("outcome/group_failed")
                raise
            self._count("outcome/group_forked")
            return leases
        finally:
            await parent.release()

    @staticmethod
    def _apply_grant(spec: SandboxSpec, grant: CapacityGrant | None) -> SandboxSpec:
        """
        Attach the devices admission granted to a spec before the backend sees it.
        """
        if grant is None or not grant.gpu_indices:
            return spec
        return replace(spec, assigned_gpus=grant.gpu_indices)

    async def acquire_prepared(
        self,
        spec: SandboxSpec,
        *,
        task_id: str,
        cache,
        setup: str | None = None,
        state_policy: SandboxStatePolicy | None = None,
        setup_timeout_s: float | None = None,
        node_id: str | None = None,
        backend: str | None = None,
    ) -> SandboxLease:
        """Return a lease for a task environment, from a capture when one exists.

        A task's setup is a property of the task rather than of the group that runs
        it, so the first caller pays it and every later caller, in any later step,
        reuses the capture. The first caller keeps the lease it already built instead
        of restoring its own capture, which would pay a round trip for nothing.

        Args:
            spec (SandboxSpec): The environment the task needs.
            task_id (str): The dataset task, which is half the cache key.
            cache: A `TaskSnapshotCache`.
            setup (str | None): The preparation command, run once per task.
            state_policy (SandboxStatePolicy | None): Must be enabled, because
                capturing is the point. A setup command has external side effects, so
                a caller that wants the capture needs `allow_external_side_effects`.
            setup_timeout_s (float | None): Deadline for the preparation command.
            node_id (str | None): The node asking, so a capture that lives on one node
                is not served to another.
            backend (str | None): Backend name, or the configured default.

        Returns:
            SandboxLease: A lease whose environment has the task's setup applied.

        Raises:
            SandboxSetupError: When the preparation command failed, in which case
                nothing is captured.
            RuntimeError: When the state policy is not enabled.
        """
        from psrl.sandbox.core import SandboxSetupError
        from psrl.sandbox.task_snapshot import resource_fingerprint

        policy = state_policy or spec.state_policy
        if not policy.enabled:
            raise RuntimeError(
                "A task environment capture requires an explicitly enabled SandboxStatePolicy, because "
                "capturing is what makes the reuse possible."
            )
        fingerprint = resource_fingerprint(
            {
                "cpu_count": spec.resources.cpu_count,
                "memory_mb": spec.resources.memory_mb,
                "disk_mb": spec.resources.disk_mb,
                "gpu_count": spec.resources.gpu_count,
            }
        )
        captured = cache.get(
            task_id=task_id,
            image=spec.source.reference,
            setup=setup,
            node_id=node_id,
            resources=fingerprint,
        )
        if captured is not None:
            self._count("outcome/task_snapshot_hit")
            return await self.restore(
                captured,
                spec=replace(spec, idempotency_key=None),
                state_policy=policy,
            )
        self._count("outcome/task_snapshot_miss")
        lease = await self.acquire(spec, backend=backend)
        try:
            if setup:
                result = await lease.session.exec(setup, timeout_s=setup_timeout_s)
                if result.exit_code != 0:
                    raise SandboxSetupError(
                        f"Sandbox setup for task {task_id!r} exited {result.exit_code}: {result.stderr.strip()[:400]}"
                    )
            snapshot = await self.checkpoint(lease.session, SnapshotKind.FILESYSTEM, policy)
        except BaseException:
            # A failed setup must not leave a sandbox behind, and must not be
            # captured as a good environment.
            await lease.release()
            raise
        cache.put(
            task_id=task_id,
            image=spec.source.reference,
            setup=setup,
            snapshot=snapshot,
            node_id=node_id or "",
            resources=fingerprint,
        )
        self._count("outcome/task_snapshot_captured")
        return lease

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
        # A spec that names no lifetime inherits the deployment's backstop, so a
        # stuck episode cannot hold a node slot for the rest of the run.
        if spec.lifetime_timeout_s is None and self._default_lifetime_s is not None:
            spec = replace(spec, lifetime_timeout_s=self._default_lifetime_s)
        required = self._required_features(spec)
        selected.capabilities.require(*required)
        if spec.required_resume_level is not None:
            selected.capabilities.require_resume_level(spec.required_resume_level)
        await selected.prepare(spec)
        return await self._provision(
            selected,
            spec.resources,
            spec.resource_class,
            spec.workflow_id,
            lambda grant: selected.create(self._apply_grant(spec, grant)),
            task_key=task_key,
            required=required,
        )

    async def _provision(
        self,
        backend: SandboxBackend,
        resources: ResourceSpec | None,
        resource_class: str,
        workflow_id: str | None,
        create: Callable[[CapacityGrant | None], Awaitable[SandboxSession]],
        *,
        task_key: tuple[str, str] | None = None,
        required: set[SandboxFeature] | None = None,
        policy: SandboxStatePolicy | None = None,
        deadline_s: float | None = None,
    ) -> SandboxLease:
        """
        Own admission, provisioning, validation, and rollback as one transaction.

        The lease moves `prepare -> queued -> provisioning -> leased`, so each
        place ownership or capacity is held is a state rather than a flag.
        """
        self._reserve_workflow(workflow_id)
        machine = LeaseStateMachine()
        grant: CapacityGrant | None = None
        lease: SandboxLease | None = None
        try:
            machine.transition(LeaseState.QUEUED)
            grant = await self._acquire_capacity(backend, resources, resource_class, deadline_s=deadline_s)
            machine.transition(LeaseState.PROVISIONING)
            try:
                creation = asyncio.ensure_future(create(grant))
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
                lease = self._adopt(exc.session, grant, workflow_id, task_key, machine)
                raise
            lease = self._adopt(session, grant, workflow_id, task_key, machine)
            if cancelled:
                raise asyncio.CancelledError
            session.capabilities.require(*(required or ()))
            if policy is not None:
                await self._sanitize_restored_session(session, policy)
            machine.transition(LeaseState.LEASED)
            self._count("outcome/provisioned")
            return lease
        except BaseException:
            if lease is not None:
                await lease.release()
            elif grant is not None:
                await complete_cleanup(self._reclaim_unassigned_capacity(grant.lease_id))
            raise
        finally:
            self._pending_workflows.discard(workflow_id)

    def _adopt(
        self,
        session: SandboxSession,
        grant: CapacityGrant | None,
        workflow_id: str | None,
        task_key: tuple[str, str] | None = None,
        machine: LeaseStateMachine | None = None,
    ) -> SandboxLease:
        # A session adopted here is already provisioning or leased, so a fresh
        # machine starts in the matching state rather than replaying the trail.
        if machine is None:
            machine = LeaseStateMachine()
            machine.transition(LeaseState.QUEUED)
            machine.transition(LeaseState.PROVISIONING)
            machine.transition(LeaseState.LEASED)
        lease = SandboxLease(
            session,
            grant.lease_id if grant is not None else None,
            self._release_capacity,
            on_released=lambda released: self._forget_lease(released, task_key),
            workflow_id=workflow_id,
            on_cleanup_failed=self._defer_release,
            state_machine=machine,
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
        self._count("outcome/released")
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
                    lambda grant: backend.connect(ref.sandbox_id),
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
        required_level = spec.required_resume_level if spec is not None else None
        if required_level is not None:
            backend.capabilities.require_resume_level(required_level)
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
                lambda grant: backend.restore(snapshot, spec),
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
            spec = session.spec

            async def fork_one(grant: CapacityGrant | None) -> SandboxSession:
                children = await session.fork(1)
                child = children[0]
                await self._sanitize_restored_session(child, policy)
                return child

            return await self._run_owned(
                self._provision(
                    backend,
                    spec.resources if spec is not None else None,
                    spec.resource_class if spec is not None else "default",
                    spec.workflow_id if spec is not None else None,
                    fork_one,
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
        level = session.capabilities.resume_level
        if spec is not None:
            metadata.update(
                {
                    "psrl.source_kind": spec.source.kind.value,
                    "psrl.source_reference": spec.source.reference,
                    "psrl.cpu_count": spec.resources.cpu_count,
                    "psrl.memory_mb": spec.resources.memory_mb,
                    "psrl.disk_mb": spec.resources.disk_mb,
                    "psrl.gpu_count": spec.resources.gpu_count,
                }
            )
        if level is not None:
            metadata["psrl.resume_level"] = level.value
        return replace(snapshot, metadata=metadata, resume_level=level or snapshot.resume_level)

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
        declared = set(spec.env) | set(session.protected_env_names)
        sensitive_keys = [key for key in declared if any(marker in key.upper() for marker in secret_markers)]
        proxy_secrets = [
            key for key, value in spec.env.items() if "PROXY" in key.upper() and urlsplit(value).username is not None
        ]
        if sensitive_keys or proxy_secrets:
            names = ", ".join(sorted(set(sensitive_keys + proxy_secrets)))
            raise RuntimeError(f"Sandbox snapshot would capture secret-bearing environment variables: {names}.")

    @staticmethod
    async def _sanitize_restored_session(session: SandboxSession, policy: SandboxStatePolicy) -> None:
        """
        Reset stale transports and mix host entropy into a session that began from captured state.

        Used for both a restore and a fork. A fork clones memory, so without this
        the whole group would resume one seeded random stream and collide on
        temporary names, test order, and ephemeral ports.
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
            gpu_count=snapshot.metadata.get("psrl.gpu_count"),
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

    def start_idle_reaper(self) -> None:
        """Start the loop that releases compute from idle sandboxes and reclaims leaks.

        The interval follows from the shortest window rather than being configured: a
        sweep slower than the window it enforces reports an idle sandbox long after
        the fact, and one much faster wakes the node for nothing.

        Started by the first lease this manager registers, so a deployment that
        configures idle windows always gets the sweep. Calling it again is a no-op.
        """
        if self._idle_task is not None and not self._idle_task.done():
            return
        windows = [w for w in (self._idle_pause_window_s, self._idle_reap_window_s) if w is not None]
        if not windows:
            return
        self._idle_task = asyncio.create_task(self._idle_loop(max(1.0, min(windows) / 4)))

    async def _idle_loop(self, interval_s: float) -> None:
        """
        Sweep the idle windows once per interval until the manager closes.
        """
        while not self._closed:
            await asyncio.sleep(interval_s)
            try:
                if self._idle_pause_window_s is not None:
                    await self.pause_idle()
                if self._idle_reap_window_s is not None:
                    await self.reap_idle()
            except asyncio.CancelledError:
                raise
            except Exception:
                psrl_logger.warning("Sandbox idle sweep failed. Retrying next interval.", exc_info=True)

    async def stop_idle_reaper(self) -> None:
        """
        Stop the idle sweep, leaving the sandboxes it was watching in place.
        """
        task, self._idle_task = self._idle_task, None
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def pause_idle(
        self,
        window_s: float | None = None,
        *,
        now: float | None = None,
    ) -> list[SandboxRef]:
        """Pause every sandbox that has been idle for its window.

        Idle is two conditions, not one. A command must not be in flight, and no
        command boundary may be more recent than the window. Age alone is not
        enough: a backend that stamps activity when a command returns leaves the
        stamp stale for the whole duration of a long command, so a reaper reading
        only the stamp sees a busy sandbox as idle and pauses a running test suite.
        """
        window = self._idle_pause_window_s if window_s is None else window_s
        if window is None:
            return []
        current = time.monotonic() if now is None else now
        paused: list[SandboxRef] = []
        for lease in list(self._leases):
            session = lease.session
            if not self._is_idle(session, window, current):
                continue
            try:
                await session.pause(self._pause_mode(session))
            except (NotImplementedError, SandboxBusyError):
                # Not idle in fact, or no pause to offer. Either way the next pass
                # decides again, and neither is a failure to report.
                continue
            except Exception:
                psrl_logger.warning(f"Sandbox idle pause failed for {lease.ref!r}.", exc_info=True)
                continue
            self._count("outcome/paused_idle")
            paused.append(lease.ref)
        return paused

    async def reap_idle(
        self,
        window_s: float | None = None,
        *,
        now: float | None = None,
    ) -> list[SandboxRef]:
        """Release every sandbox idle past the reap window.

        The window is longer than the pause window, so a sandbox is always paused
        before it is destroyed, and a caller that only wants to reclaim leaks does
        not also lose the state a pause would have kept.
        """
        window = self._idle_reap_window_s if window_s is None else window_s
        if window is None:
            return []
        current = time.monotonic() if now is None else now
        reaped: list[SandboxRef] = []
        for lease in list(self._leases):
            if lease in self._pending_releases:
                continue
            if not self._is_idle(lease.session, window, current):
                continue
            self._count("outcome/reaped_idle")
            reaped.append(lease.ref)
            await lease.release()
        return reaped

    @staticmethod
    def _is_idle(session: SandboxSession, window_s: float, now: float) -> bool:
        """
        Return whether a session is idle for the whole window.
        """
        if session.busy:
            return False
        last = session.last_activity_at
        if last is None:
            # A backend that reports no activity is never treated as idle, because
            # the policy cannot tell a quiet sandbox from an unreported one.
            return False
        return (now - last) >= window_s

    @staticmethod
    def _pause_mode(session: SandboxSession):
        """
        Choose the strongest pause the backend declares, preferring a resident freeze.
        """
        from psrl.sandbox.core import PauseMode

        if session.capabilities.supports(SandboxFeature.FREEZE):
            return PauseMode.FREEZE
        return PauseMode.HIBERNATE

    async def capacity_snapshot(self) -> dict:
        """Return the node envelope's accounting, or an empty mapping without one.

        Exposed so a node side caller can describe the envelope it advertises without
        reaching into the coordinator this manager holds.
        """
        if self._capacity_coordinator is None:
            return {}
        return await self._capacity_coordinator.snapshot()

    def ownership_snapshot(self) -> dict[str, int]:
        """Report worker ownership, including cleanup that still holds capacity.

        `holding_resources` counts the leases that still own a runtime object or a
        reservation, which is the question an operator actually asks: a lease in a
        terminal state costs nothing, and one stuck in reclaiming costs a slot.
        """
        by_state: dict[str, int] = {state.value: 0 for state in LeaseState}
        oldest_idle = 0.0
        holding = 0
        for lease in self._leases:
            by_state[lease.state.value] = by_state.get(lease.state.value, 0) + 1
            oldest_idle = max(oldest_idle, lease.state_machine.age_s)
            holding += int(lease.state_machine.holds_resources)
        return {
            "leases": len(self._leases),
            "holding_resources": holding,
            "provisioning": len(self._provision_tasks),
            "pending_releases": len(self._pending_releases),
            "pending_capacity": len(self._pending_capacity),
            "max_release_attempts": max(self._release_attempts.values(), default=0),
            "oldest_lease_state_age_s": oldest_idle,
            **{f"leases_{state}": count for state, count in by_state.items()},
        }

    def snapshot(self) -> dict[str, float]:
        """Return the ownership plane's metrics for the trainer's per-step hook.

        The plane reports and never logs. One reader keeps the sandbox metric group
        on the training step axis, and a collection failure cannot stall a step.
        """
        ownership = self.ownership_snapshot()
        metrics: dict[str, float] = {f"ownership/{key}": float(value) for key, value in ownership.items()}
        metrics["admission/waits"] = float(self._wait_samples)
        metrics["admission/wait_s_mean"] = self._cumulative_wait_s / self._wait_samples if self._wait_samples else 0.0
        metrics.update({key: float(value) for key, value in self._counters.items()})
        metrics.update(self._backend_plane_metrics())
        return metrics

    def _backend_plane_metrics(self) -> dict[str, float]:
        """Return the counters backends keep for themselves, such as a warm pool.

        A backend exposes a `snapshot()` when it has a plane of its own to report, and
        the manager is the only reader, which keeps every sandbox metric on one hook.
        """
        metrics: dict[str, float] = {}
        for backend in self._backends.values():
            reporter = getattr(backend, "warm_pool_snapshot", None)
            if callable(reporter):
                metrics.update({str(key): float(value) for key, value in reporter().items()})
            usage = getattr(backend, "usage_snapshot", None)
            if callable(usage):
                metrics.update({str(key): float(value) for key, value in usage().items()})
            image = getattr(backend, "image_snapshot", None)
            if callable(image):
                metrics.update({str(key): float(value) for key, value in image().items()})
        return metrics

    async def prefetch(
        self,
        plan: PrefetchPlan,
        *,
        backend: str | None = None,
        concurrency: int = 2,
    ) -> PrefetchReport:
        """Warm a run's working set before rollout, so the first create does not pull.

        A backend that materializes images lazily has nothing to warm and reports zero.
        A reference that could not be warmed is reported rather than raised, because a
        prefetch is an optimization: a task whose image missed it still runs.

        Args:
            plan (PrefetchPlan): The working set the run's task set asks for.
            backend (str | None): Backend name, or the configured default.
            concurrency (int): How many images may be warmed at once.

        Returns:
            PrefetchReport: What the step achieved, including its coverage.
        """
        self._require_open()
        if not plan.references:
            return PrefetchReport()
        selected = self.backend(backend)
        warmer = getattr(selected, "prefetch_images", None)
        if not callable(warmer):
            return PrefetchReport(requested=plan.size)
        warmed = await warmer(plan.references, concurrency=concurrency)
        self._count("outcome/prefetched")
        return PrefetchReport(requested=plan.size, warmed=int(warmed))

    def end_run(self, run_id: str | None = None) -> list[str]:
        """Drop the snapshots a run published, for a run-scoped retention.

        A snapshot whose retention is the run must not outlive it, and this is the
        only signal that says the run is over: a store cannot tell a finished run from
        a quiet one. `run_id` of None forgets every run the stores hold, which is what
        a worker shutting down has to say.

        Args:
            run_id (str | None): The run that ended, or None for every run.

        Returns:
            list[str]: The digest references forgotten.
        """
        forgotten: list[str] = []
        for backend in self._backends.values():
            store = getattr(backend, "snapshot_store", None)
            if store is None:
                continue
            forgotten.extend(store.forget_run(run_id))
        return forgotten

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
        await self.stop_idle_reaper()
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
