"""Docker Engine API sandbox backend."""

from __future__ import annotations

import asyncio
import atexit
import hashlib
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from psrl.sandbox.backends.docker_engine import DockerEngine, DockerEngineClient, DockerEngineError
from psrl.sandbox.core import (
    ExecResult,
    PauseMode,
    ResourceUsage,
    SandboxBackend,
    SandboxCapabilities,
    SandboxFeature,
    SandboxRef,
    SandboxSession,
    SandboxSource,
    SandboxSourceKind,
    SandboxSpec,
    SandboxStatus,
    SnapshotKind,
    SnapshotRef,
)
from psrl.sandbox.metrics import SandboxMetrics, SandboxMetricsSnapshot
from psrl.sandbox.utils.docker_utils import force_remove_containers_by_label, spawn_actor_reaper

# FILESYSTEM_SNAPSHOT + RESTORE: a docker commit turns a container's writable
# layer into a reusable image, and restore creates a container from it — the
# cheap, self-contained "clean snapshot" used to seed the grader without a
# fresh cold start.
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
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_PROXY_URL_ENV_KEYS = (
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
)


def _rewrite_loopback_proxy(value: str, host_alias: str) -> str:
    """Replace a proxy URL's loopback host with the Docker host gateway alias."""
    has_scheme = "://" in value
    parsed = urlsplit(value if has_scheme else f"//{value}")
    if parsed.hostname not in _LOOPBACK_HOSTS:
        return value
    try:
        port = f":{parsed.port}" if parsed.port is not None else ""
    except ValueError:
        return value
    userinfo, separator, _ = parsed.netloc.rpartition("@")
    authority = f"{userinfo}{separator}{host_alias}{port}"
    rewritten = urlunsplit((parsed.scheme, authority, parsed.path, parsed.query, parsed.fragment))
    return rewritten if has_scheme else rewritten.removeprefix("//")


def _append_no_proxy_alias(value: str, host_alias: str) -> str:
    entries = [entry.strip() for entry in value.split(",") if entry.strip()]
    return ",".join(dict.fromkeys([*entries, host_alias]))


@dataclass(frozen=True)
class DockerSecurityConfig:
    """Security controls applied to every Docker sandbox."""

    require_rootless: bool = False
    pids_limit: int = 4096
    cap_drop: tuple[str, ...] = ("ALL",)
    cap_add: tuple[str, ...] = ()
    no_new_privileges: bool = True
    read_only_rootfs: bool = False
    seccomp_profile: str | None = None
    user: str | None = None
    tmpfs: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pids_limit <= 0:
            raise ValueError("Docker pids_limit must be greater than zero.")


@dataclass(frozen=True)
class DockerPolicyProfile:
    """Typed per-workload Docker policy overrides."""

    network_mode: str | None = None
    extra_hosts: tuple[str, ...] = ()
    host_gateway_alias: str | None = None
    rewrite_loopback_proxies: bool = False
    pids_limit: int | None = None
    cap_drop: tuple[str, ...] | None = None
    cap_add: tuple[str, ...] | None = None
    no_new_privileges: bool | None = None
    read_only_rootfs: bool | None = None
    seccomp_profile: str | None = None
    user: str | None = None
    tmpfs: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pids_limit is not None and self.pids_limit <= 0:
            raise ValueError("Docker policy pids_limit must be greater than zero.")
        if self.rewrite_loopback_proxies and not self.host_gateway_alias:
            raise ValueError("Docker proxy rewriting requires host_gateway_alias.")

    @classmethod
    def from_value(cls, value: DockerPolicyProfile | Mapping[str, Any]) -> DockerPolicyProfile:
        """Normalize Hydra mappings into an immutable policy."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("Docker policy profiles must use typed mappings, not raw CLI arguments.")
        normalized = dict(value)
        for key in ("extra_hosts", "cap_drop", "cap_add"):
            if normalized.get(key) is not None:
                normalized[key] = tuple(normalized[key])
        return cls(**normalized)


class DockerBackend(SandboxBackend):
    """Local Docker backend using one persistent Engine API connection pool."""

    def __init__(
        self,
        name: str = "docker",
        docker_host: str | None = None,
        policy_profiles: Mapping[str, DockerPolicyProfile | Mapping[str, Any]] | None = None,
        security: DockerSecurityConfig | Mapping[str, Any] | None = None,
        keepalive_command: Sequence[str] = ("tail", "-f", "/dev/null"),
        command_interpreter: Sequence[str] = ("bash", "-lc"),
        request_timeout_s: float = 180.0,
        connection_limit: int = 128,
        auto_pull: bool = True,
        registry_auth: Mapping[str, str] | None = None,
        owner_id_env: str = "PSRL_ACTOR_ID",
        reaper_binary: str = "docker",
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
        if not self.keepalive_command or not self.command_interpreter:
            raise ValueError("Docker keepalive_command and command_interpreter cannot be empty.")
        self.owner_id = os.getenv(owner_id_env, "")
        self.reaper_binary = reaper_binary
        self.auto_pull = auto_pull
        self.registry_auth = dict(registry_auth or {})
        self.engine = engine or DockerEngineClient(
            docker_host,
            request_timeout_s=request_timeout_s,
            connection_limit=connection_limit,
        )
        self.metrics = SandboxMetrics()
        self._rootless_checked = False
        self._rootless_lock = asyncio.Lock()
        self._reaper_proc = None
        self._atexit_registered = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> SandboxCapabilities:
        return _DOCKER_CAPABILITIES

    def metrics_snapshot(self) -> SandboxMetricsSnapshot:
        """Return Docker lifecycle, latency, and memory metrics."""
        return self.metrics.snapshot()

    def _ensure_reaper(self) -> None:
        """Start crash recovery before the first owned container is created."""
        if not self.owner_id or self._reaper_proc is not None:
            return
        self._reaper_proc = spawn_actor_reaper(self.owner_id, binary=self.reaper_binary)
        if not self._atexit_registered:
            atexit.register(self._shutdown_sync)
            self._atexit_registered = True

    def _shutdown_sync(self) -> None:
        """Best-effort cleanup for interpreter and Ray actor shutdown."""
        if self.owner_id and self._reaper_proc is not None:
            force_remove_containers_by_label("psrl.actor_id", self.owner_id, binary=self.reaper_binary)
        if self._reaper_proc is not None and self._reaper_proc.poll() is None:
            self._reaper_proc.terminate()

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
            digest = hashlib.sha256(f"{self.name}\0{spec.idempotency_key}".encode()).hexdigest()[:20]
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
            "AutoRemove": True,
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
        if spec.resources.cpu_count is not None:
            host_config["NanoCpus"] = int(spec.resources.cpu_count * 1_000_000_000)
        if spec.resources.memory_mb is not None:
            host_config["Memory"] = spec.resources.memory_mb * 1024 * 1024

        environment = dict(spec.env)
        if policy.host_gateway_alias:
            if policy.rewrite_loopback_proxies:
                for key in _PROXY_URL_ENV_KEYS:
                    if key in environment:
                        environment[key] = _rewrite_loopback_proxy(environment[key], policy.host_gateway_alias)
            no_proxy = _append_no_proxy_alias(
                ",".join(filter(None, (environment.get("no_proxy"), environment.get("NO_PROXY")))),
                policy.host_gateway_alias,
            )
            environment["no_proxy"] = no_proxy
            environment["NO_PROXY"] = no_proxy

        config: dict[str, Any] = {
            "Image": spec.source.reference,
            "Cmd": list(self.keepalive_command),
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
        if spec.source.kind != SandboxSourceKind.IMAGE:
            raise RuntimeError("DockerBackend requires an image source.")
        if spec.resources.disk_mb is not None:
            raise RuntimeError("DockerBackend does not implement a portable disk size limit.")
        await self._check_rootless()
        self._ensure_reaper()
        policy = self._resolve_policy(spec)
        name = self._container_name(spec)
        config = self._build_container_config(spec, policy)
        with self.metrics.measure("create"):
            try:
                container_id = await self._create_and_start(name, config)
            except DockerEngineError as exc:
                if exc.status == 404 and self.auto_pull:
                    with self.metrics.measure("pull_image"):
                        await self.engine.pull_image(spec.source.reference, self.registry_auth or None)
                    container_id = await self._create_and_start(name, config)
                elif exc.status != 409 or not spec.idempotency_key:
                    raise
                else:
                    container_id = await self._recover_idempotent_conflict(name, config, spec, exc)
        self.metrics.session_started()
        return DockerSession(self, container_id, spec=spec, lifetime_timeout_s=spec.idle_timeout_s)

    async def _create_and_start(self, name: str, config: Mapping[str, Any]) -> str:
        """Create and start a container without leaking a failed start."""
        container_id = await self.engine.create_container(name, config)
        try:
            await self.engine.start_container(container_id)
        except BaseException:
            await self.engine.remove_container(container_id)
            raise
        return container_id

    async def _recover_idempotent_conflict(
        self,
        name: str,
        config: Mapping[str, Any],
        spec: SandboxSpec,
        conflict: DockerEngineError,
    ) -> str:
        """Reuse an exact retry without racing another process's start call."""
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
            if status in {"created", "restarting"} and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.05)
                continue
            await self.engine.remove_container(container_id)
            return await self._create_and_start(name, config)

    async def connect(self, sandbox_id: str) -> SandboxSession:
        session = DockerSession(self, sandbox_id)
        status = await session.status()
        if status == SandboxStatus.PAUSED:
            await session.resume()
        elif status != SandboxStatus.RUNNING:
            raise RuntimeError(f"Docker sandbox {sandbox_id!r} is not running (status={status.value}).")
        self.metrics.session_started()
        return session

    async def restore(self, snapshot: SnapshotRef, spec: SandboxSpec | None = None) -> SandboxSession:
        """Create a session from a filesystem snapshot (a committed image)."""
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
        """Remove the committed image backing a filesystem snapshot."""
        if snapshot.kind != SnapshotKind.FILESYSTEM:
            return
        image = snapshot.metadata.get("psrl.docker.image")
        if image:
            try:
                await self.engine.remove_image(str(image))
            except DockerEngineError as exc:
                if exc.status != 404:
                    raise

    async def shutdown(self) -> None:
        """Stop crash recovery and close the persistent Engine connection pool."""
        await asyncio.to_thread(self._shutdown_sync)
        await self.engine.close()


class DockerSession(SandboxSession):
    """One Docker container session."""

    def __init__(
        self,
        backend: DockerBackend,
        sandbox_id: str,
        *,
        spec: SandboxSpec | None = None,
        lifetime_timeout_s: float | None = None,
    ) -> None:
        self.backend = backend
        self.sandbox_id = sandbox_id
        self._spec = spec
        self._command_count = 0
        self._terminate_lock = asyncio.Lock()
        self._terminated = False
        self._timeout_task = (
            asyncio.create_task(self._terminate_at_deadline(lifetime_timeout_s))
            if lifetime_timeout_s is not None
            else None
        )

    async def _terminate_at_deadline(self, timeout_s: float) -> None:
        """Enforce Docker lifetime locally because Engine has no native TTL."""
        try:
            await asyncio.sleep(timeout_s)
            with self.backend.metrics.measure("lifetime_timeout"):
                await self.terminate()
        except asyncio.CancelledError:
            return

    @property
    def ref(self) -> SandboxRef:
        return SandboxRef(self.backend.name, self.sandbox_id)

    @property
    def capabilities(self) -> SandboxCapabilities:
        return self.backend.capabilities

    async def snapshot(self, kind: SnapshotKind) -> SnapshotRef:
        """Capture a filesystem snapshot by committing the container's writable layer."""
        if kind != SnapshotKind.FILESYSTEM:
            raise NotImplementedError(f"DockerSession only supports FILESYSTEM snapshots, got {kind.value}.")
        repo = f"psrl/snapshot/{self.sandbox_id}"
        image_id = await self.backend.engine.commit_container(self.sandbox_id, repo, "latest")
        image_tag = f"{repo}:latest"
        return SnapshotRef(
            backend=self.backend.name,
            snapshot_id=image_tag,
            kind=SnapshotKind.FILESYSTEM,
            metadata={"psrl.docker.image": image_tag, "psrl.docker.image_id": image_id},
        )

    @property
    def spec(self) -> SandboxSpec | None:
        return self._spec

    @property
    def command_count(self) -> int:
        return self._command_count

    def resolve_callback_url(self, url: str) -> str:
        """Rewrite worker-loopback URLs through the configured host gateway."""
        if self._spec is None or self._spec.policy_profile is None:
            return url
        policy = self.backend.policy_profiles.get(self._spec.policy_profile)
        if policy is None or not policy.host_gateway_alias:
            return url
        return _rewrite_loopback_proxy(url, policy.host_gateway_alias)

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> ExecResult:
        self._command_count += 1
        try:
            with self.backend.metrics.measure("exec"):
                exit_code, stdout, stderr = await self.backend.engine.exec(
                    self.sandbox_id,
                    [*self.backend.command_interpreter, command],
                    cwd=cwd,
                    env=env,
                    timeout_s=timeout_s,
                )
        except asyncio.CancelledError:
            try:
                await self.terminate()
            except Exception:
                pass
            raise
        except (TimeoutError, asyncio.TimeoutError):
            # The Engine API cannot cancel an exec after its start request has
            # timed out. Destroy the disposable session so no runaway process
            # survives or races subsequent commands.
            try:
                await self.terminate()
            except Exception:
                pass
            raise TimeoutError(f"Docker command exceeded {timeout_s} seconds.") from None
        return ExecResult(
            exit_code=exit_code,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
        )

    async def read_bytes(self, path: str) -> bytes:
        with self.backend.metrics.measure("read_bytes"):
            return await self.backend.engine.read_file(self.sandbox_id, path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        with self.backend.metrics.measure("write_bytes"):
            await self.backend.engine.write_file(self.sandbox_id, path, data)

    async def status(self) -> SandboxStatus:
        if self._terminated:
            return SandboxStatus.TERMINATED
        with self.backend.metrics.measure("status"):
            inspection = await self.backend.engine.inspect_container(self.sandbox_id)
        if inspection is None:
            return SandboxStatus.TERMINATED
        state = str((inspection.get("State") or {}).get("Status", "unknown"))
        return {
            "running": SandboxStatus.RUNNING,
            "paused": SandboxStatus.PAUSED,
            "exited": SandboxStatus.EXITED,
            "dead": SandboxStatus.EXITED,
        }.get(state, SandboxStatus.UNKNOWN)

    async def stats(self) -> ResourceUsage:
        with self.backend.metrics.measure("stats"):
            stats = await self.backend.engine.stats(self.sandbox_id)
        memory = stats.get("memory_stats") or {}
        current = int(memory.get("usage", 0) or 0)
        peak = int(memory.get("max_usage", 0) or (memory.get("stats") or {}).get("peak", 0) or current)
        cpu_total = int(((stats.get("cpu_stats") or {}).get("cpu_usage") or {}).get("total_usage", 0) or 0)
        self.backend.metrics.observe_memory(current, peak)
        return ResourceUsage(memory_bytes=current, peak_memory_bytes=peak, cpu_total_ns=cpu_total)

    async def terminate(self) -> None:
        async with self._terminate_lock:
            if self._terminated:
                return
            with self.backend.metrics.measure("terminate"):
                await self.backend.engine.remove_container(self.sandbox_id)
            self._terminated = True
            current_task = asyncio.current_task()
            if self._timeout_task is not None and self._timeout_task is not current_task:
                self._timeout_task.cancel()
            self.backend.metrics.session_stopped()

    async def pause(self, mode: PauseMode) -> None:
        if mode != PauseMode.FREEZE:
            raise RuntimeError("DockerBackend supports freeze, not hibernation.")
        with self.backend.metrics.measure("pause"):
            await self.backend.engine.pause_container(self.sandbox_id)

    async def resume(self) -> None:
        with self.backend.metrics.measure("resume"):
            await self.backend.engine.unpause_container(self.sandbox_id)
