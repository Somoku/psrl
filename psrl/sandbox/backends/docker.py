"""Docker Engine API sandbox backend."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import aiohttp

from psrl.sandbox.async_utils import complete_cleanup
from psrl.sandbox.backends.docker_engine import (
    DockerEngine,
    DockerEngineClient,
    DockerEngineError,
)
from psrl.sandbox.backends.docker_events import ContainerEventWatcher
from psrl.sandbox.backends.docker_lifecycle import DockerLifecycle, DockerLifecycleConfig
from psrl.sandbox.backends.docker_policy import (
    PROXY_URL_ENV_KEYS,
    DockerDiskAdmissionConfig,
    DockerPolicyProfile,
    DockerSecurityConfig,
    append_no_proxy_alias,
    rewrite_loopback_proxy,
)
from psrl.sandbox.backends.docker_session import DockerSession
from psrl.sandbox.core import (
    SandboxBackend,
    SandboxCapabilities,
    SandboxFeature,
    SandboxProvisionError,
    SandboxSession,
    SandboxSource,
    SandboxSourceKind,
    SandboxSpec,
    SandboxStatus,
    SnapshotKind,
    SnapshotRef,
)
from psrl.sandbox.metrics import SandboxMetrics, SandboxMetricsSnapshot
from psrl.sandbox.utils.docker_utils import lease_store_id

psrl_logger = logging.getLogger(__file__)

# FILESYSTEM_SNAPSHOT + RESTORE: docker commit turns a container's writable layer into a reusable
# image, and restore creates one from it: a cheap, self-contained grader snapshot with no cold start.
_DOCKER_CAPABILITIES = SandboxCapabilities(
    frozenset(
        {
            SandboxFeature.FREEZE,
            SandboxFeature.HOST_MOUNT,
            SandboxFeature.FILESYSTEM_SNAPSHOT,
            SandboxFeature.RESTORE,
        }
    )
)
# Grace for a creation already past admission to settle before shutdown cancels it, so a
# container that was just created still gets an owner that can destroy it.
_CREATE_DRAIN_TIMEOUT_S = 30.0


class DockerBackend(SandboxBackend):
    """
    Local Docker backend using one persistent Engine API connection pool.
    """

    def __init__(
        self,
        name: str = "docker",
        docker_host: str | None = None,
        policy_profiles: Mapping[str, DockerPolicyProfile | Mapping[str, Any]] | None = None,
        security: DockerSecurityConfig | Mapping[str, Any] | None = None,
        keepalive_command: Sequence[str] = ("tail", "-f", "/dev/null"),
        command_interpreter: Sequence[str] = ("bash", "-lc"),
        request_timeout_s: float = 180.0,
        container_watch_interval_s: float = 15.0,
        connection_limit: int = 128,
        image_pull_concurrency: int = 2,
        max_exec_output_bytes: int = 16 * 1024 * 1024,
        auto_pull: bool = True,
        registry_auth: Mapping[str, str] | None = None,
        owner_id_env: str = "PSRL_ACTOR_ID",
        lifecycle: DockerLifecycleConfig | Mapping[str, Any] | None = None,
        disk_admission: DockerDiskAdmissionConfig | Mapping[str, Any] | None = None,
        cgroup_parent: str | None = None,
        engine: DockerEngine | None = None,
    ) -> None:
        self._name = name
        self.docker_host = docker_host
        self.policy_profiles = {
            key: DockerPolicyProfile.from_value(value) for key, value in (policy_profiles or {}).items()
        }
        self.security = (
            security if isinstance(security, DockerSecurityConfig) else DockerSecurityConfig(**dict(security or {}))
        )
        self.keepalive_command = tuple(keepalive_command)
        self.command_interpreter = tuple(command_interpreter)
        # Retained so a session can describe its own truncation without reaching into the
        # engine client, which test doubles and other engines do not have to expose.
        self.max_exec_output_bytes = max_exec_output_bytes
        if container_watch_interval_s <= 0:
            raise ValueError("Docker container_watch_interval_s must be greater than zero.")
        self.container_watch_interval_s = container_watch_interval_s
        if not self.keepalive_command or not self.command_interpreter:
            raise ValueError("Docker keepalive_command and command_interpreter cannot be empty.")
        self.owner_id = os.getenv(owner_id_env, "")
        self.lifecycle = DockerLifecycle(
            self.owner_id,
            DockerLifecycleConfig.from_value(lifecycle, docker_host=docker_host),
        )
        self.disk_admission = DockerDiskAdmissionConfig.from_value(disk_admission)
        self.cgroup_parent = cgroup_parent
        self.auto_pull = auto_pull
        self.registry_auth = dict(registry_auth or {})
        self.engine = engine or DockerEngineClient(
            docker_host,
            request_timeout_s=request_timeout_s,
            connection_limit=connection_limit,
            max_exec_output_bytes=max_exec_output_bytes,
        )
        self.metrics = SandboxMetrics()
        self.container_events = ContainerEventWatcher(self.engine, self.engine.inspect_container)
        self._rootless_checked = False
        self._rootless_lock = asyncio.Lock()
        if image_pull_concurrency < 1:
            raise ValueError("Docker image_pull_concurrency must be positive.")
        self._image_pull_slots = asyncio.Semaphore(image_pull_concurrency)
        self._image_tasks: dict[str, asyncio.Task[None]] = {}
        self._closed = False
        self._close_event: asyncio.Event | None = None
        self._sessions: dict[str, DockerSession] = {}
        self._creates: set[asyncio.Task] = set()

    async def prepare(self, spec: SandboxSpec) -> None:
        """
        Warm an image with bounded, shared downloads on this backend instance.
        """
        if self._closed:
            raise RuntimeError("Docker backend is closed.")
        if spec.source.kind != SandboxSourceKind.IMAGE:
            raise RuntimeError("DockerBackend requires an image source.")
        reference = spec.source.reference
        task = self._image_tasks.get(reference)
        if task is None:
            task = asyncio.create_task(self._prepare_image(reference))
            self._image_tasks[reference] = task
            task.add_done_callback(lambda done: self._finish_image_prepare(reference, done))
        await asyncio.shield(task)

    def _finish_image_prepare(self, reference: str, task: asyncio.Task[None]) -> None:
        """
        Forget completed downloads and observe failures after waiter cancellation.
        """
        self._image_tasks.pop(reference, None)
        if not task.cancelled():
            task.exception()

    async def _prepare_image(self, reference: str) -> None:
        """
        Recheck the daemon cache after admission before downloading an image.
        """
        async with self._image_pull_slots:
            if await self.engine.image_exists(reference):
                return
            if not self.auto_pull:
                raise RuntimeError(f"Docker image {reference!r} is unavailable and auto_pull is disabled.")
            await self._await_disk_headroom()
            with self.metrics.measure("pull_image"):
                await self.engine.pull_image(reference, self.registry_auth or None)

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> SandboxCapabilities:
        return _DOCKER_CAPABILITIES

    @property
    def uses_node_capacity(self) -> bool:
        """
        Return that Docker sessions share the worker node's resources.
        """
        return True

    @property
    def is_open(self) -> bool:
        """
        Return whether this backend still accepts work and owns its sessions.
        """
        return not self._closed

    def forget_session(self, container_id: str) -> None:
        """
        Drop a destroyed session from the backend's ownership registry.
        """
        self._sessions.pop(container_id, None)
        self.container_events.forget(container_id)

    async def wait_for_stop(self, container_id: str) -> str | None:
        """
        Wait for a container stop, or report that the daemon does not serve events.
        """
        return await self.container_events.wait_for_stop(container_id)

    def metrics_snapshot(self) -> SandboxMetricsSnapshot:
        """
        Return Docker lifecycle, latency, and memory metrics.
        """
        return self.metrics.snapshot()

    async def _await_disk_headroom(self) -> None:
        """
        Block until the Docker data path has enough free space, then give up.
        """
        policy = self.disk_admission
        if not policy.path or policy.min_free_mb <= 0:
            return
        loop = asyncio.get_running_loop()
        deadline = loop.time() + policy.wait_timeout_s
        while True:
            try:
                usage = await asyncio.to_thread(shutil.disk_usage, policy.path)
                free_mb = usage.free // (1024 * 1024)
            except OSError as exc:
                raise RuntimeError(f"Could not inspect Docker data path {policy.path!r}.") from exc
            if free_mb >= policy.min_free_mb:
                return
            if loop.time() >= deadline:
                with self.metrics.measure("disk_admission_failure"):
                    raise RuntimeError(
                        f"Docker data path {policy.path!r} has {free_mb} MiB free, below the "
                        f"{policy.min_free_mb} MiB admission threshold, after waiting "
                        f"{policy.wait_timeout_s:.0f}s."
                    )
            with self.metrics.measure("disk_admission_wait"):
                # Wake on shutdown as well as on the poll interval, or a closing worker waits
                # out the full admission timeout for a sandbox it is no longer going to use.
                await self._wait_or_close(min(policy.poll_interval_s, max(0.0, deadline - loop.time())))
            if self._closed:
                raise RuntimeError("Docker backend is closed.")

    async def _wait_or_close(self, timeout_s: float) -> None:
        """
        Sleep for an interval unless the backend closes first.
        """
        if self._close_event is None:
            self._close_event = asyncio.Event()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._close_event.wait(), timeout=timeout_s)

    async def _check_rootless(self) -> None:
        if not self.security.require_rootless or self._rootless_checked:
            return
        async with self._rootless_lock:
            if self._rootless_checked:
                return
            info = await self.engine.info()
            security_options = [str(option).lower() for option in info.get("SecurityOptions", [])]
            if not any("rootless" in option for option in security_options):
                raise RuntimeError(
                    "Docker backend requires a rootless daemon, but Docker did not report rootless mode."
                )
            self._rootless_checked = True

    def _container_name(self, spec: SandboxSpec) -> str:
        if spec.idempotency_key:
            # Scoped to this worker: two workers sampling the same task share an idempotency
            # key, and sharing one container would put two trajectories in one sandbox.
            digest = hashlib.sha256(f"{self.owner_id}\0{self.name}\0{spec.idempotency_key}".encode()).hexdigest()[:20]
            return f"psrl-sandbox-{digest}"
        return f"psrl-sandbox-{uuid.uuid4().hex[:20]}"

    def _resolve_policy(self, spec: SandboxSpec) -> DockerPolicyProfile:
        if spec.policy_profile is None:
            return DockerPolicyProfile()
        try:
            return self.policy_profiles[spec.policy_profile]
        except KeyError as exc:
            raise RuntimeError(f"Unknown Docker policy profile {spec.policy_profile!r}.") from exc

    def _build_container_config(self, spec: SandboxSpec, policy: DockerPolicyProfile) -> dict[str, Any]:
        labels = {**dict(spec.metadata), "psrl.sandbox": "true"}
        labels["psrl.lease_store"] = lease_store_id(self.lifecycle.config.heartbeat_dir)
        if self.owner_id:
            labels["psrl.actor_id"] = self.owner_id
        if spec.idempotency_key:
            labels["psrl.idempotency_key"] = spec.idempotency_key

        mounts = []
        for mount in spec.mounts:
            if not Path(mount.source).is_absolute() or not Path(mount.target).is_absolute():
                raise ValueError("Docker bind-mount source and target must be absolute paths.")
            mounts.append(
                {
                    "Type": "bind",
                    "Source": mount.source,
                    "Target": mount.target,
                    "ReadOnly": mount.read_only,
                }
            )

        cap_drop = self.security.cap_drop if policy.cap_drop is None else policy.cap_drop
        cap_add = self.security.cap_add if policy.cap_add is None else policy.cap_add
        no_new_privileges = (
            self.security.no_new_privileges if policy.no_new_privileges is None else policy.no_new_privileges
        )
        read_only = self.security.read_only_rootfs if policy.read_only_rootfs is None else policy.read_only_rootfs
        pids_limit = self.security.pids_limit if policy.pids_limit is None else policy.pids_limit
        seccomp_profile = policy.seccomp_profile or self.security.seccomp_profile
        if seccomp_profile == "unconfined":
            raise ValueError("Docker seccomp cannot be disabled for PSRL sandboxes.")
        security_options = ["no-new-privileges"] if no_new_privileges else []
        if seccomp_profile:
            security_options.append(f"seccomp={seccomp_profile}")
        tmpfs = {**dict(self.security.tmpfs), **dict(policy.tmpfs)}

        host_config: dict[str, Any] = {
            "AutoRemove": False,
            "Init": True,
            "CapDrop": list(cap_drop),
            "CapAdd": list(cap_add),
            "ReadonlyRootfs": read_only,
            "PidsLimit": pids_limit,
            "SecurityOpt": security_options,
            "Mounts": mounts,
        }
        if policy.network_mode:
            host_config["NetworkMode"] = policy.network_mode
        extra_hosts = list(policy.extra_hosts)
        if policy.host_gateway_alias and not any(
            entry.partition(":")[0] == policy.host_gateway_alias for entry in extra_hosts
        ):
            extra_hosts.append(f"{policy.host_gateway_alias}:host-gateway")
        if extra_hosts:
            host_config["ExtraHosts"] = extra_hosts
        if tmpfs:
            host_config["Tmpfs"] = tmpfs
        if self.cgroup_parent:
            host_config["CgroupParent"] = self.cgroup_parent
        if spec.resources.cpu_count is not None:
            host_config["NanoCpus"] = int(spec.resources.cpu_count * 1_000_000_000)
        if spec.resources.memory_mb is not None:
            host_config["Memory"] = spec.resources.memory_mb * 1024 * 1024
            host_config["MemorySwap"] = host_config["Memory"]
        oom_score_adj = self.security.oom_score_adj if policy.oom_score_adj is None else policy.oom_score_adj
        if oom_score_adj is not None:
            host_config["OomScoreAdj"] = oom_score_adj

        environment = dict(spec.env)
        if policy.host_gateway_alias:
            if policy.rewrite_loopback_proxies:
                for key in PROXY_URL_ENV_KEYS:
                    if key in environment:
                        environment[key] = rewrite_loopback_proxy(environment[key], policy.host_gateway_alias)
            no_proxy = append_no_proxy_alias(
                ",".join(filter(None, (environment.get("no_proxy"), environment.get("NO_PROXY")))),
                policy.host_gateway_alias,
            )
            environment["no_proxy"] = no_proxy
            environment["NO_PROXY"] = no_proxy

        config: dict[str, Any] = {
            "Image": spec.source.reference,
            "Cmd": list(self.keepalive_command),
            "Entrypoint": [],
            "Env": [f"{key}={value}" for key, value in sorted(environment.items())],
            "Labels": labels,
            "HostConfig": host_config,
        }
        if spec.workdir:
            config["WorkingDir"] = spec.workdir
        user = policy.user or self.security.user
        if user:
            config["User"] = user
        identity = {
            "container": config,
            "idle_timeout_s": spec.idle_timeout_s,
        }
        labels["psrl.spec_hash"] = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return config

    async def create(self, spec: SandboxSpec) -> SandboxSession:
        if self._closed:
            raise RuntimeError("Docker backend is closed.")
        task = asyncio.create_task(self._create_session(spec))
        self._creates.add(task)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await complete_cleanup(self._discard_create(task))
            raise
        finally:
            self._creates.discard(task)

    async def _discard_create(self, task: asyncio.Task) -> None:
        session = await task
        try:
            await session.terminate()
        except Exception as exc:
            raise SandboxProvisionError(session, exc) from exc

    def _track_session(
        self,
        container_id: str,
        spec: SandboxSpec | None = None,
        policy: DockerPolicyProfile | None = None,
    ) -> DockerSession:
        existing = self._sessions.get(container_id)
        if existing is not None:
            return existing
        session = DockerSession(
            self,
            container_id,
            spec=spec,
            policy=policy or DockerPolicyProfile(),
            lifetime_timeout_s=spec.idle_timeout_s if spec else None,
        )
        self._sessions[container_id] = session
        self.metrics.session_started()
        return session

    async def _create_session(self, spec: SandboxSpec) -> SandboxSession:
        if self._closed:
            raise RuntimeError("Docker backend is closed.")
        if spec.source.kind != SandboxSourceKind.IMAGE:
            raise RuntimeError("DockerBackend requires an image source.")
        if spec.resources.disk_mb is not None:
            raise RuntimeError("DockerBackend does not implement a portable disk size limit.")
        await self._await_disk_headroom()
        await self._check_rootless()
        await asyncio.to_thread(self.lifecycle.start)
        policy = self._resolve_policy(spec)
        name = self._container_name(spec)
        config = self._build_container_config(spec, policy)
        with self.metrics.measure("create"):
            container_id = await self._create_or_recover(name, config, spec)
        return self._track_session(container_id, spec, policy)

    async def _create_or_recover(
        self,
        name: str,
        config: Mapping[str, Any],
        spec: SandboxSpec,
    ) -> str:
        """
        Create once, pull a missing image once, or join an exact retry.
        """
        try:
            return await self._create_and_start(name, config)
        except DockerEngineError as exc:
            create_error = exc
        if create_error.status == 404 and self.auto_pull:
            await self.prepare(spec)
            try:
                return await self._create_and_start(name, config)
            except DockerEngineError as exc:
                create_error = exc
        if create_error.status == 409 and spec.idempotency_key:
            return await self._recover_idempotent_conflict(name, config, spec, create_error)
        raise create_error

    async def _create_and_start(self, name: str, config: Mapping[str, Any]) -> str:
        """
        Create and start a container without leaking a failed start.
        """
        try:
            container_id = await self.engine.create_container(name, config)
        except (aiohttp.ClientError, TimeoutError, asyncio.TimeoutError) as exc:
            # A lost response can hide a successful create. Keep the unique name owned.
            raise SandboxProvisionError(self._track_session(name), exc) from exc
        try:
            await self.engine.start_container(container_id)
        except BaseException as error:
            session = self._track_session(container_id)
            try:
                await session.terminate()
            except Exception as cleanup_error:
                raise SandboxProvisionError(session, error) from cleanup_error
            raise
        return container_id

    async def _recover_idempotent_conflict(
        self,
        name: str,
        config: Mapping[str, Any],
        spec: SandboxSpec,
        conflict: DockerEngineError,
    ) -> str:
        """Adopt this worker's own earlier attempt at the same sandbox.

        The container name is scoped to this worker, so a conflict is always a retry of a
        request this process already made, never another worker's. It may still be mid-start,
        which is why a created container is waited on rather than replaced.
        """
        deadline = asyncio.get_running_loop().time() + 5.0
        while True:
            existing = await self.engine.inspect_container(name)
            if existing is None:
                return await self._create_and_start(name, config)
            labels = (existing.get("Config") or {}).get("Labels") or {}
            if labels.get("psrl.idempotency_key") != spec.idempotency_key:
                raise RuntimeError(f"Docker container name {name!r} has an unrelated owner.") from conflict
            expected_hash = config["Labels"]["psrl.spec_hash"]
            if labels.get("psrl.spec_hash") != expected_hash:
                raise RuntimeError(
                    f"Docker idempotency key {spec.idempotency_key!r} was reused with a different spec."
                ) from conflict
            container_id = str(existing["Id"])
            status = str((existing.get("State") or {}).get("Status", "unknown"))
            if status == "running":
                return container_id
            if status == "paused":
                await self.engine.unpause_container(container_id)
                return container_id
            if status in {"created", "restarting"}:
                if asyncio.get_running_loop().time() >= deadline:
                    raise RuntimeError(f"Docker sandbox {container_id!r} is still starting.")
                await asyncio.sleep(0.05)
                continue
            await self.engine.remove_container(container_id)
            return await self._create_and_start(name, config)

    async def connect(self, sandbox_id: str) -> SandboxSession:
        if self._closed:
            raise RuntimeError("Docker backend is closed.")
        session = self._track_session(sandbox_id)
        status = await session.status()
        if status == SandboxStatus.PAUSED:
            await session.resume()
        elif status != SandboxStatus.RUNNING:
            raise RuntimeError(f"Docker sandbox {sandbox_id!r} is not running (status={status.value}).")
        return session

    async def restore(self, snapshot: SnapshotRef, spec: SandboxSpec | None = None) -> SandboxSession:
        """
        Create a session from a filesystem snapshot (a committed image).
        """
        if snapshot.kind != SnapshotKind.FILESYSTEM:
            raise NotImplementedError(f"DockerBackend only restores FILESYSTEM snapshots, got {snapshot.kind.value}.")
        image = snapshot.metadata.get("psrl.docker.image") or snapshot.snapshot_id
        if not image:
            raise ValueError(f"Snapshot {snapshot.snapshot_id!r} has no docker image reference.")
        if spec is None:
            raise ValueError("DockerBackend.restore requires a SandboxSpec.")
        restored_spec = replace(spec, source=SandboxSource.image(str(image)))
        return await self.create(restored_spec)

    async def delete_snapshot(self, snapshot: SnapshotRef) -> None:
        """
        Remove the committed image backing a filesystem snapshot.
        """
        if snapshot.kind != SnapshotKind.FILESYSTEM:
            return
        image = snapshot.metadata.get("psrl.docker.image") or snapshot.snapshot_id
        if image:
            await self.engine.remove_image(str(image))

    async def shutdown(self) -> None:
        """Stop crash recovery and close the persistent Engine connection pool.

        Image downloads and disk admission are released before in-flight creations are
        joined, because `create` shields its own work: a create parked on either of those
        would otherwise hold shutdown for the whole pull or the whole admission timeout.
        """
        self._closed = True
        if self._close_event is not None:
            self._close_event.set()
        await self.container_events.close()
        image_tasks = list(self._image_tasks.values())
        for task in image_tasks:
            task.cancel()
        await asyncio.gather(*image_tasks, return_exceptions=True)
        creates = list(self._creates)
        if creates:
            _, pending = await asyncio.wait(creates, timeout=_CREATE_DRAIN_TIMEOUT_S)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        session_results = await asyncio.gather(
            *(session.terminate() for session in list(self._sessions.values())),
            return_exceptions=True,
        )
        try:
            await asyncio.to_thread(self.lifecycle.close)
        finally:
            await self.engine.close()
        errors = [result for result in session_results if isinstance(result, BaseException)]
        if errors:
            raise RuntimeError("Docker shutdown could not confirm container cleanup.") from errors[0]
