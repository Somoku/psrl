"""Worker-local sandbox registry and lifecycle ownership."""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from psrl.sandbox.core import (
    SandboxBackend,
    SandboxFeature,
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


class SandboxLease:
    """Single-owner lifecycle guard for a sandbox session."""

    def __init__(
        self,
        session: SandboxSession,
        on_released: Callable[[SandboxLease], None] | None = None,
    ) -> None:
        self.session = session
        self._released = False
        self._lock = asyncio.Lock()
        self._on_released = on_released

    @property
    def ref(self) -> SandboxRef:
        """Return the leased sandbox reference."""
        return self.session.ref

    @property
    def released(self) -> bool:
        """Return whether the exit disposition has already been applied."""
        return self._released

    async def release(self) -> None:
        """Terminate the sandbox exactly once."""
        async with self._lock:
            if self._released:
                return
            await self.session.terminate()
            self._released = True
            if self._on_released is not None:
                self._on_released(self)

    async def __aenter__(self) -> SandboxSession:
        return self.session

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.release()


@dataclass(frozen=True)
class SandboxTask:
    """One shared create task and the spec bound to its idempotency key."""

    spec: SandboxSpec
    task: asyncio.Task[SandboxLease]


class SandboxManager:
    """Resolve backends and own active sandbox leases for one worker."""

    def __init__(self, backends: Mapping[str, SandboxBackend], default_backend: str) -> None:
        self._backends = dict(backends)
        self.default_backend = default_backend
        if default_backend not in self._backends:
            raise ValueError(f"Default sandbox backend {default_backend!r} is not configured.")
        for name, backend in self._backends.items():
            if name != backend.name:
                raise ValueError(f"Sandbox backend key {name!r} does not match backend.name={backend.name!r}.")
        self._leases: set[SandboxLease] = set()
        self._idempotent_leases: dict[tuple[str, str], SandboxLease] = {}
        self._create_tasks: dict[tuple[str, str], SandboxTask] = {}
        self._anonymous_create_tasks: set[asyncio.Task[SandboxLease]] = set()
        self._create_lock = asyncio.Lock()
        self._closed = False

    def backend(self, name: str | None = None) -> SandboxBackend:
        """Resolve a configured backend by name or use the default backend."""
        backend_name = name or self.default_backend
        try:
            return self._backends[backend_name]
        except KeyError as exc:
            raise ValueError(f"Sandbox backend {backend_name!r} is not configured.") from exc

    def sync(
        self,
        backend: str | None = None,
    ) -> SyncSandboxManager:
        """Create a facade for synchronous code running outside the current event loop."""
        from psrl.sandbox.sync import SyncSandboxManager

        self._require_open()
        self.backend(backend)
        return SyncSandboxManager(self, asyncio.get_running_loop(), backend=backend)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Sandbox manager is closed.")

    async def acquire(
        self,
        spec: SandboxSpec,
        backend: str | None = None,
    ) -> SandboxLease:
        """Provision a sandbox and return its explicit ownership guard."""
        self._require_open()
        selected = self.backend(backend)
        idempotency_key = spec.idempotency_key
        if idempotency_key is None:
            async with self._create_lock:
                self._require_open()
                task = asyncio.create_task(self._acquire_new(spec, selected))
                self._anonymous_create_tasks.add(task)
                task.add_done_callback(self._anonymous_create_tasks.discard)
            return await asyncio.shield(task)

        task_key = (selected.name, idempotency_key)
        async with self._create_lock:
            self._require_open()
            existing = self._idempotent_leases.get(task_key)
            if existing is not None and not existing.released:
                if existing.session.spec != spec:
                    raise RuntimeError("A sandbox idempotency key cannot be reused with a different spec.")
                return existing
            create_task = self._create_tasks.get(task_key)
            if create_task is not None and create_task.task.done():
                self._create_tasks.pop(task_key, None)
                create_task = None
            if create_task is None:
                task = asyncio.create_task(
                    self._acquire_new(spec, selected, task_key),
                )
                self._create_tasks[task_key] = SandboxTask(spec, task)
            elif create_task.spec != spec:
                raise RuntimeError("A sandbox idempotency key cannot be reused with a different spec.")
            else:
                task = create_task.task
        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                async with self._create_lock:
                    create_task = self._create_tasks.get(task_key)
                    if create_task is not None and create_task.task is task:
                        self._create_tasks.pop(task_key, None)

    async def _acquire_new(
        self,
        spec: SandboxSpec,
        selected: SandboxBackend,
        task_key: tuple[str, str] | None = None,
    ) -> SandboxLease:
        """Create and register one lease after admission checks."""
        required = set(spec.required_features)
        if spec.mounts:
            required.add(SandboxFeature.HOST_MOUNT)
        selected.capabilities.require(*required)
        session = await selected.create(spec)
        try:
            session.capabilities.require(*required)
        except BaseException:
            await session.terminate()
            raise
        lease = SandboxLease(
            session,
            on_released=lambda released: self._forget_lease(released, task_key),
        )
        self._leases.add(lease)
        if task_key is not None:
            self._idempotent_leases[task_key] = lease
        return lease

    def _forget_lease(
        self,
        lease: SandboxLease,
        task_key: tuple[str, str] | None = None,
    ) -> None:
        """Forget a released lease and its process-local idempotency key."""
        self._leases.discard(lease)
        if task_key is not None and self._idempotent_leases.get(task_key) is lease:
            self._idempotent_leases.pop(task_key, None)

    async def connect(
        self,
        ref: SandboxRef,
    ) -> SandboxLease:
        """Reconnect to a backend-owned sandbox and assume lifecycle ownership."""
        self._require_open()
        session = await self.backend(ref.backend).connect(ref.sandbox_id)
        lease = SandboxLease(session, on_released=self._leases.discard)
        self._leases.add(lease)
        return lease

    async def restore(
        self,
        snapshot: SnapshotRef,
        spec: SandboxSpec | None = None,
        state_policy: SandboxStatePolicy | None = None,
    ) -> SandboxLease:
        """Restore a snapshot through its owning backend."""
        self._require_open()
        backend = self.backend(snapshot.backend)
        backend.capabilities.require(SandboxFeature.RESTORE)
        policy = state_policy or (spec.state_policy if spec is not None else SandboxStatePolicy())
        if not policy.enabled:
            raise RuntimeError("Sandbox restore requires an explicitly enabled SandboxStatePolicy.")
        session = await backend.restore(snapshot, spec)
        await self._sanitize_restored_session(session, policy)
        lease = SandboxLease(session, on_released=self._leases.discard)
        self._leases.add(lease)
        return lease

    async def branch(
        self,
        session: SandboxSession,
        snapshot_kind: SnapshotKind = SnapshotKind.FULL_STATE,
        state_policy: SandboxStatePolicy | None = None,
    ) -> SandboxLease:
        """Branch state without silently weakening snapshot semantics."""
        self._require_open()
        policy = state_policy or (session.spec.state_policy if session.spec is not None else SandboxStatePolicy())
        self._validate_state_policy(session, policy)
        if session.capabilities.supports(SandboxFeature.NATIVE_FORK):
            child = await session.fork()
            await self._sanitize_restored_session(child, policy)
            lease = SandboxLease(child, on_released=self._leases.discard)
            self._leases.add(lease)
            return lease

        snapshot_feature = {
            SnapshotKind.FILESYSTEM: SandboxFeature.FILESYSTEM_SNAPSHOT,
            SnapshotKind.FULL_STATE: SandboxFeature.FULL_STATE_SNAPSHOT,
        }[snapshot_kind]
        session.capabilities.require(snapshot_feature, SandboxFeature.RESTORE)
        snapshot = await self.checkpoint(session, snapshot_kind, policy)
        try:
            child = await self.restore(snapshot, state_policy=policy)
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
        """Create a capability-gated snapshot after enforcing RL safety policy."""
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
        """Reset stale transports and mix host entropy into a restored microVM."""
        try:
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
        except BaseException:
            try:
                await session.terminate()
            except BaseException:
                pass
            raise

    async def delete_snapshot(self, snapshot: SnapshotRef) -> None:
        """Delete a snapshot through its owning backend."""
        await self.backend(snapshot.backend).delete_snapshot(snapshot)

    async def release(self, lease: SandboxLease) -> None:
        """Release one lease and remove it from worker ownership."""
        await lease.release()

    def metrics_snapshot(self) -> dict[str, SandboxMetricsSnapshot]:
        """Return one metrics snapshot per configured backend."""
        return {name: backend.metrics_snapshot() for name, backend in self._backends.items()}

    async def shutdown(self) -> None:
        """Terminate unreleased leases and close all backends."""
        if self._closed:
            return
        self._closed = True
        async with self._create_lock:
            create_tasks = [
                *(create_task.task for create_task in self._create_tasks.values()),
                *self._anonymous_create_tasks,
            ]
        create_results = await asyncio.gather(*create_tasks, return_exceptions=True)
        leases = list(self._leases)
        results = await asyncio.gather(*(lease.release() for lease in leases), return_exceptions=True)
        self._leases.clear()
        self._idempotent_leases.clear()
        self._create_tasks.clear()
        self._anonymous_create_tasks.clear()
        backend_results = await asyncio.gather(
            *(backend.shutdown() for backend in self._backends.values()),
            return_exceptions=True,
        )
        errors = [
            result for result in [*create_results, *results, *backend_results] if isinstance(result, BaseException)
        ]
        if errors:
            raise RuntimeError(f"Sandbox manager shutdown had {len(errors)} failure(s).") from errors[0]
