"""One sandbox node's control surface, and the wire format its callers use.

The agent is what makes the internal Docker backend external to the rest of PSRL.
It owns the daemon, the envelope, and the reclaimer, because admission has to live
with the daemon it guards: a caller that reached another node's daemon over TCP
would create containers whose memory its own accounting never sees.

Everything crossing the boundary is a plain mapping rather than a live object, so
the protocol is documented by its payloads and a non-Ray caller can speak it. The
payload codecs live here, beside the service that answers them.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from psrl.sandbox.core import (
    CredentialAuth,
    CredentialAuthType,
    CredentialBinding,
    CredentialRef,
    CredentialSubstitution,
    EgressAction,
    EgressPolicy,
    EgressRule,
    ExecMode,
    MountSpec,
    PauseMode,
    ResourceSpec,
    ResumeLevel,
    SandboxDiagnostics,
    SandboxFeature,
    SandboxProvisionError,
    SandboxRef,
    SandboxSession,
    SandboxSource,
    SandboxSourceKind,
    SandboxSpec,
    SandboxStatePolicy,
    SnapshotKind,
    SnapshotRef,
    VolumeSpec,
)

psrl_logger = logging.getLogger(__file__)

# The resource class a fork parent is admitted under, kept in step with the manager.
FORK_PARENT_CLASS = "prepare"

# How often a node reports that it is still there. Placement's own node TTL is minutes, so
# several missed reports fit inside one TTL and a single slow call does not drain a node.
_DEFAULT_LIVENESS_INTERVAL_S = 30.0


@dataclass(frozen=True)
class CallbackTarget:
    """
    Where the worker that owns a trajectory serves its TITO session server.
    """

    host: str
    port: int

    @classmethod
    def parse(cls, value: str) -> CallbackTarget:
        """
        Read a `host:port` authority.
        """
        host, separator, port = value.rpartition(":")
        if not separator or not host or not port.isdigit():
            raise ValueError(f"A sandbox callback target must be host:port, got {value!r}.")
        return cls(host=host, port=int(port))

    def as_str(self) -> str:
        return f"{self.host}:{self.port}"


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """
    Copy one direction of a forwarded connection until it ends.
    """
    try:
        while chunk := await reader.read(64 * 1024):
            writer.write(chunk)
            await writer.drain()
    except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError, OSError):
        return


class CallbackForwarder:
    """A node-side listener that forwards to one caller's session server.

    A sandbox reaches its callback through its own node's gateway alias, which is the
    wrong machine once the caller is elsewhere. The node therefore listens on a port of
    its own and forwards to the caller, so the URL a sandbox is handed is the same
    whether it landed on the caller's node or on another one. The port is chosen by the
    node and reported to the caller, because a port the caller picked could already be
    taken on the node.
    """

    def __init__(self, target: CallbackTarget) -> None:
        self.target = target
        self.port: int | None = None
        self._server: asyncio.AbstractServer | None = None
        self._pumps: set[asyncio.Task[None]] = set()

    async def start(self) -> int:
        """
        Listen on a free node port and return it.
        """
        if self._server is not None and self.port is not None:
            return self.port
        self._server = await asyncio.start_server(self._pump, host="0.0.0.0", port=0)
        sockets = self._server.sockets or ()
        if not sockets:
            raise RuntimeError("Sandbox callback forwarder could not bind a listening port.")
        self.port = int(sockets[0].getsockname()[1])
        psrl_logger.info(
            f"Sandbox callback forwarder on this node is listening on port {self.port} for "
            f"{self.target.as_str()}."
        )
        return self.port

    async def _pump(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """
        Forward one accepted connection to the caller.
        """
        task = asyncio.current_task()
        if task is not None:
            self._pumps.add(task)
        upstream: asyncio.StreamWriter | None = None
        try:
            try:
                upstream_reader, upstream = await asyncio.open_connection(self.target.host, self.target.port)
            except OSError:
                psrl_logger.warning(
                    f"Sandbox callback forwarder could not reach {self.target.as_str()}. A sandbox on this "
                    "node cannot record its tokens."
                )
                return
            await asyncio.gather(_pipe(reader, upstream), _pipe(upstream_reader, writer))
        finally:
            writer.close()
            if upstream is not None:
                upstream.close()
            if task is not None:
                self._pumps.discard(task)

    async def close(self) -> None:
        """Stop listening and drop the connections in flight.

        The pumps are cancelled before the server is awaited, because
        `wait_closed` waits for every live handler and a pump is parked on a read
        that only the peer can end. Awaiting first would hang until the sandbox on
        the other side happened to disconnect, which is never during a shutdown.
        """
        pumps = list(self._pumps)
        for task in pumps:
            task.cancel()
        if pumps:
            await asyncio.gather(*pumps, return_exceptions=True)
        self._pumps.clear()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
            self.port = None


def spec_to_payload(spec: SandboxSpec) -> dict[str, Any]:
    """
    Serialize a spec for a control channel.
    """
    return {
        "source_kind": spec.source.kind.value,
        "source_reference": spec.source.reference,
        "resources": {
            "cpu_count": spec.resources.cpu_count,
            "memory_mb": spec.resources.memory_mb,
            "disk_mb": spec.resources.disk_mb,
            "gpu_count": spec.resources.gpu_count,
        },
        "workdir": spec.workdir,
        "metadata": dict(spec.metadata),
        "env": dict(spec.env),
        "mounts": [
            {"source": item.source, "target": item.target, "read_only": item.read_only} for item in spec.mounts
        ],
        "volumes": [{"name": item.name, "target": item.target, "read_only": item.read_only} for item in spec.volumes],
        "policy_profile": spec.policy_profile,
        "idle_timeout_s": spec.idle_timeout_s,
        "lifetime_timeout_s": spec.lifetime_timeout_s,
        "idempotency_key": spec.idempotency_key,
        "required_features": sorted(feature.value for feature in spec.required_features),
        "required_resume_level": spec.required_resume_level.value if spec.required_resume_level else None,
        "resource_class": spec.resource_class,
        "workflow_id": spec.workflow_id,
        "state_policy": {
            "enabled": spec.state_policy.enabled,
            "allow_secret_capture": spec.state_policy.allow_secret_capture,
            "allow_external_side_effects": spec.state_policy.allow_external_side_effects,
            "reseed_after_restore": spec.state_policy.reseed_after_restore,
        },
        "exec_mode": spec.exec_mode.value if spec.exec_mode else None,
        "egress": (
            None
            if spec.egress is None
            else {
                "default_action": spec.egress.default_action.value,
                "rules": [
                    {"action": rule.action.value, "target": rule.target, "ports": list(rule.ports)}
                    for rule in spec.egress.rules
                ],
            }
        ),
        "credentials": [
            {"source_env": item.source_env, "target_env": item.target_env, "name": item.name}
            for item in spec.credentials
        ],
        "credential_bindings": [
            {
                "name": binding.name,
                "hosts": list(binding.hosts),
                "credential": binding.credential,
                "schemes": list(binding.schemes),
                "methods": list(binding.methods),
                "paths": list(binding.paths),
                "auth": {
                    "type": binding.auth.type.value,
                    "header_name": binding.auth.header_name,
                    "headers": [list(item) for item in binding.auth.headers],
                    "substitutions": [
                        {
                            "credential": item.credential,
                            "placeholder": item.placeholder,
                            "surfaces": list(item.surfaces),
                        }
                        for item in binding.auth.substitutions
                    ],
                },
            }
            for binding in spec.credential_bindings
        ],
    }


def _egress_from_payload(egress: Mapping[str, Any] | None) -> EgressPolicy | None:
    """
    Rebuild an egress policy from its payload.
    """
    if not egress:
        return None
    return EgressPolicy(
        default_action=EgressAction(str(egress["default_action"])),
        rules=tuple(
            EgressRule(
                action=EgressAction(str(rule["action"])),
                target=str(rule["target"]),
                ports=tuple(rule.get("ports") or ()),
            )
            for rule in egress.get("rules") or ()
        ),
    )


def _auth_from_payload(payload: Mapping[str, Any]) -> CredentialAuth:
    """
    Rebuild a credential auth rule from its payload.
    """
    return CredentialAuth(
        type=CredentialAuthType(str(payload.get("type") or CredentialAuthType.BEARER.value)),
        header_name=payload.get("header_name"),
        headers=tuple((str(item[0]), str(item[1])) for item in payload.get("headers") or ()),
        substitutions=tuple(
            CredentialSubstitution(
                credential=str(item["credential"]),
                placeholder=str(item["placeholder"]),
                surfaces=tuple(item.get("surfaces") or ("body",)),
            )
            for item in payload.get("substitutions") or ()
        ),
    )


def spec_from_payload(payload: Mapping[str, Any]) -> SandboxSpec:
    """
    Rebuild a spec from its payload.
    """
    source_kind = SandboxSourceKind(str(payload["source_kind"]))
    source = SandboxSource(source_kind, str(payload["source_reference"]))
    resources = payload.get("resources") or {}
    state_policy = payload.get("state_policy") or {}
    return SandboxSpec(
        source=source,
        resources=ResourceSpec(
            cpu_count=resources.get("cpu_count"),
            memory_mb=resources.get("memory_mb"),
            disk_mb=resources.get("disk_mb"),
            gpu_count=resources.get("gpu_count"),
        ),
        workdir=payload.get("workdir"),
        metadata=dict(payload.get("metadata") or {}),
        env=dict(payload.get("env") or {}),
        mounts=tuple(MountSpec(**item) for item in payload.get("mounts") or ()),
        volumes=tuple(VolumeSpec(**item) for item in payload.get("volumes") or ()),
        policy_profile=payload.get("policy_profile"),
        idle_timeout_s=payload.get("idle_timeout_s"),
        lifetime_timeout_s=payload.get("lifetime_timeout_s"),
        idempotency_key=payload.get("idempotency_key"),
        state_policy=SandboxStatePolicy(**state_policy) if state_policy else SandboxStatePolicy(),
        required_features=frozenset(SandboxFeature(item) for item in payload.get("required_features") or ()),
        required_resume_level=(
            ResumeLevel(str(payload["required_resume_level"])) if payload.get("required_resume_level") else None
        ),
        resource_class=str(payload.get("resource_class") or "default"),
        workflow_id=payload.get("workflow_id"),
        exec_mode=ExecMode(str(payload["exec_mode"])) if payload.get("exec_mode") else None,
        egress=_egress_from_payload(payload.get("egress")),
        credentials=tuple(CredentialRef(**item) for item in payload.get("credentials") or ()),
        credential_bindings=tuple(
            CredentialBinding(
                name=str(item["name"]),
                hosts=tuple(str(host) for host in item.get("hosts") or ()),
                credential=str(item["credential"]),
                auth=_auth_from_payload(item.get("auth") or {}),
                schemes=tuple(str(value) for value in item.get("schemes") or ("https",)),
                methods=tuple(str(value) for value in item.get("methods") or ()),
                paths=tuple(str(value) for value in item.get("paths") or ("/*",)),
            )
            for item in payload.get("credential_bindings") or ()
        ),
    )


def snapshot_to_payload(snapshot: SnapshotRef) -> dict[str, Any]:
    """
    Serialize a snapshot reference for a control channel.
    """
    return {
        "backend": snapshot.backend,
        "snapshot_id": snapshot.snapshot_id,
        "kind": snapshot.kind.value,
        "metadata": dict(snapshot.metadata),
        "resume_level": snapshot.resume_level.value if snapshot.resume_level else None,
    }


def snapshot_from_payload(payload: Mapping[str, Any]) -> SnapshotRef:
    """
    Rebuild a snapshot reference from its payload.
    """
    level = payload.get("resume_level")
    return SnapshotRef(
        backend=str(payload["backend"]),
        snapshot_id=str(payload["snapshot_id"]),
        kind=SnapshotKind(str(payload["kind"])),
        metadata=dict(payload.get("metadata") or {}),
        resume_level=ResumeLevel(str(level)) if level else None,
    )


@dataclass(frozen=True)
class SandboxHandle:
    """
    What a node returns for a provisioned sandbox.

    The capabilities travel with the handle so the caller's session reports what
    the hosting node actually offers, rather than what the deployment declared.
    """

    backend: str
    sandbox_id: str
    node_id: str
    features: frozenset[SandboxFeature] = frozenset()
    resume_level: ResumeLevel | None = None
    exec_mode: str | None = None
    callback_host_alias: str | None = None
    # The port this node forwards to the caller's session server on, when the caller
    # asked for a forwarder. The sandbox reaches it through `callback_host_alias`.
    callback_port: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "sandbox_id": self.sandbox_id,
            "node_id": self.node_id,
            "features": sorted(feature.value for feature in self.features),
            "resume_level": self.resume_level.value if self.resume_level else None,
            "exec_mode": self.exec_mode,
            "callback_host_alias": self.callback_host_alias,
            "callback_port": self.callback_port,
        }


class NodeDrainingError(RuntimeError):
    """
    Raised when a node refuses work because it cannot confirm previous cleanup.

    The node still holds containers whose memory its envelope has already returned, so
    admitting against it would over-commit the host. Refusing is the only safe answer.
    """


class NodeProvisionError(RuntimeError):
    """Raised when a node could not provision a sandbox it now has to clean up.

    `SandboxProvisionError` carries the live session, because an in-process caller releases
    it. A session cannot cross an actor boundary, so the node keeps its own cleanup and
    reports this instead: the node's manager already owns the partial sandbox, which is why
    the message is a diagnosis rather than a request for the caller to act.

    It is a plain string-carrying error for exactly that reason, so it survives the wire.
    """


class SandboxNodeAgent:
    """The session protocol of one sandbox node.

    Every method is deliberately thin. The agent adds what only a node can know,
    which is where the sandbox is and what it holds, and delegates everything else
    to the manager that already owns ownership and admission.
    """

    def __init__(
        self,
        manager,
        *,
        node_id: str,
        labels: Sequence[str] = (),
        reachable_session_server: bool = True,
        reclaimer=None,
        liveness_interval_s: float = _DEFAULT_LIVENESS_INTERVAL_S,
    ) -> None:
        self.manager = manager
        self.node_id = node_id
        self.labels = frozenset(labels)
        self.reachable_session_server = reachable_session_server
        self.reclaimer = reclaimer
        # The node reports its own liveness, because liveness belongs to the node rather
        # than to any one worker: a worker dying must not drain a node others are using.
        #
        # A transport sets the reporter before it advertises, because the two are one
        # handshake. A node known without a way to stay known is drained at its TTL.
        if liveness_interval_s <= 0:
            raise ValueError("Sandbox node liveness_interval_s must be greater than zero.")
        self.liveness_reporter: Callable[[], Awaitable[None]] | None = None
        self.liveness_interval_s = liveness_interval_s
        self._liveness_task: asyncio.Task[None] | None = None
        self._handles: dict[tuple[str, str], Any] = {}
        # Containers this node asked the runtime to destroy and could not. While any are
        # outstanding it holds returned memory, so it refuses work until a sweep confirms them gone.
        self._unremovable: set[str] = set()
        # One forwarder per caller session server, so every sandbox a worker places on
        # this node reaches the worker that owns its trajectory.
        self._forwarders: dict[str, CallbackForwarder] = {}

    def start_liveness(self) -> None:
        """Start reporting that this node is still there.

        Started by `advertise`, because that is the call that makes the node known to
        placement and so the first moment there is liveness to report. Placement ages a node
        from its last word, so a node that registers once and then goes quiet is drained and
        stops receiving work.
        """
        if self.liveness_reporter is None or self._liveness_task is not None:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No loop means no sandbox this node could be running either.
            return
        self._liveness_task = asyncio.create_task(self._liveness_loop())

    def stop_liveness(self) -> None:
        """
        Stop reporting this node's liveness, so placement can drain it.
        """
        if self._liveness_task is not None:
            self._liveness_task.cancel()
            self._liveness_task = None

    async def _liveness_loop(self) -> None:
        while True:
            await asyncio.sleep(self.liveness_interval_s)
            try:
                await self.liveness_reporter()
            except Exception:
                # A placement that cannot be reached is not this node's failure to fix, and
                # the next tick retries. Stopping here would drain the node permanently.
                psrl_logger.warning("Sandbox node %s could not report its liveness.", self.node_id, exc_info=True)

    async def advertise(self) -> dict[str, Any]:
        """Report what this node can do, so placement can decide without asking twice.

        A draining node advertises that fact rather than its capabilities, because
        placement must not pick it until it can destroy what it already holds.
        """
        self.start_liveness()
        payloads: list[dict[str, Any]] = []
        for name, backend in self.manager._backends.items():
            capabilities = backend.capabilities
            index = await self._backend_image_index(backend)
            payloads.append(
                {
                    "node_id": self.node_id,
                    "backend": name,
                    "features": sorted(feature.value for feature in capabilities.features),
                    "resume_level": capabilities.resume_level.value if capabilities.resume_level else None,
                    "gpu_count": await self._gpu_count(),
                    "host_mounts": capabilities.supports(SandboxFeature.HOST_MOUNT),
                    "labels": sorted(self.labels),
                    "reachable_session_server": self.reachable_session_server,
                    "draining": self.draining,
                    "image_digests": sorted(index["digests"]),
                    "image_references": sorted(index["references"]),
                }
            )
        return {"node_id": self.node_id, "backends": payloads}

    async def _gpu_count(self) -> int:
        """
        Return the device count this node's envelope admits.
        """
        envelope = await self.manager.capacity_snapshot()
        return int((envelope.get("capacity") or {}).get("gpu_count", 0) or 0)

    async def image_index(self) -> dict[str, set[str]]:
        """
        Return the images this node's daemons already hold.
        """
        merged: dict[str, set[str]] = {"digests": set(), "references": set()}
        for backend in self.manager._backends.values():
            index = await self._backend_image_index(backend)
            merged["digests"] |= index["digests"]
            merged["references"] |= index["references"]
        return merged

    @staticmethod
    async def _backend_image_index(backend) -> dict[str, set[str]]:
        """
        Read one backend's image index, when it can report one.
        """
        index = getattr(backend, "image_index", None)
        if index is None:
            return {"digests": set(), "references": set()}
        reported = await index() if asyncio.iscoroutinefunction(index) else index()
        if isinstance(reported, dict):
            return {
                "digests": {str(item) for item in reported.get("digests", ())},
                "references": {str(item) for item in reported.get("references", ())},
            }
        return {"digests": {str(item) for item in reported}, "references": set()}

    @property
    def draining(self) -> bool:
        """
        Return whether this node is refusing new work.
        """
        return bool(self._unremovable)

    def _require_writable(self) -> None:
        """
        Refuse work while this node still holds containers it could not destroy.
        """
        if self._unremovable:
            raise NodeDrainingError(
                f"Sandbox node {self.node_id!r} is refusing new work while it still holds "
                f"{len(self._unremovable)} container(s) it could not destroy. Its daemon or collector has "
                "to recover first, or new sandboxes would be admitted against memory that is still in use."
            )

    async def acquire(self, spec_payload: Mapping[str, Any], *, callback: str | None = None) -> dict[str, Any]:
        """
        Provision one sandbox on this node.
        """
        self._require_writable()
        try:
            lease = await self.manager.acquire(spec_from_payload(spec_payload))
        except SandboxProvisionError as exc:
            raise self._provision_failed("acquire", exc) from None
        return await self._remember(lease, callback=callback)

    async def acquire_group(
        self,
        spec_payloads: Sequence[Mapping[str, Any]],
        *,
        callback: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Provision one group on this node.
        """
        self._require_writable()
        try:
            leases = await self.manager.acquire_group([spec_from_payload(payload) for payload in spec_payloads])
        except SandboxProvisionError as exc:
            raise self._provision_failed("acquire_group", exc) from None
        return [await self._remember(lease, callback=callback) for lease in leases]

    async def connect(self, backend: str, sandbox_id: str) -> dict[str, Any]:
        """
        Assume ownership of a sandbox this node already hosts.
        """
        lease = await self.manager.connect(SandboxRef(backend, sandbox_id))
        return await self._remember(lease)

    async def forward_callback(self, target: str) -> int:
        """Listen on this node for a caller's session server and return the port.

        The port is chosen here and reported to the caller, so a sandbox is handed a URL
        that resolves from whichever node it landed on. One forwarder serves every
        sandbox a worker places on this node, because the target is per caller.
        """
        existing = self._forwarders.get(target)
        if existing is not None and existing.port is not None:
            return existing.port
        forwarder = existing or CallbackForwarder(CallbackTarget.parse(target))
        self._forwarders[target] = forwarder
        return await forwarder.start()

    async def _remember(self, lease, *, callback: str | None = None) -> dict[str, Any]:
        capabilities = lease.session.capabilities
        callback_port = await self.forward_callback(callback) if callback else None
        handle = SandboxHandle(
            backend=lease.session.ref.backend,
            sandbox_id=lease.session.ref.sandbox_id,
            node_id=self.node_id,
            features=frozenset(capabilities.features),
            resume_level=capabilities.resume_level,
            callback_host_alias=lease.session.callback_host_alias,
            callback_port=callback_port,
            exec_mode=(
                lease.session.spec.exec_mode.value
                if lease.session.spec is not None and lease.session.spec.exec_mode is not None
                else None
            ),
        )
        self._handles[(handle.backend, handle.sandbox_id)] = lease
        return handle.as_dict()

    def _session(self, backend: str, sandbox_id: str) -> SandboxSession:
        """
        Resolve an owned handle to its session.
        """
        lease = self._handles.get((backend, sandbox_id))
        if lease is None:
            raise RuntimeError(
                f"Sandbox node {self.node_id!r} does not own {backend}/{sandbox_id}. A caller may only reach "
                "a sandbox through the node that provisioned it."
            )
        return lease.session

    async def exec(
        self,
        backend: str,
        sandbox_id: str,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
        silence_timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """
        Run one command on a sandbox this node owns.
        """
        result = await self._session(backend, sandbox_id).exec(
            command,
            cwd=cwd,
            env=env,
            timeout_s=timeout_s,
            silence_timeout_s=silence_timeout_s,
        )
        return {
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "truncated": result.truncated,
        }

    async def read_bytes(self, backend: str, sandbox_id: str, path: str) -> bytes:
        """
        Read one file from a sandbox this node owns.
        """
        return await self._session(backend, sandbox_id).read_bytes(path)

    async def write_bytes(self, backend: str, sandbox_id: str, path: str, data: bytes) -> None:
        """
        Write one file into a sandbox this node owns.
        """
        await self._session(backend, sandbox_id).write_bytes(path, data)

    async def status(self, backend: str, sandbox_id: str) -> str:
        """
        Return a sandbox's portable status.
        """
        return (await self._session(backend, sandbox_id).status()).value

    async def busy(self, backend: str, sandbox_id: str) -> bool:
        """
        Return whether a command is running on a sandbox this node owns.
        """
        return self._session(backend, sandbox_id).busy

    async def diagnostics(self, backend: str, sandbox_id: str) -> dict[str, Any]:
        """Return read-only evidence about one sandbox.

        A finished sandbox is still diagnosable through its handle, because the
        handle is what the node kept when the container went away.
        """
        session = self._session(backend, sandbox_id)
        diagnosis: SandboxDiagnostics = await session.diagnostics()
        return diagnosis.as_dict()

    async def resolve_callback_url(self, backend: str, sandbox_id: str, url: str) -> str:
        """Translate a caller URL into one this sandbox can reach.

        Resolved on the node, because the translation is node policy: it depends on
        that node's gateway alias and network mode, which the caller cannot know.
        """
        return self._session(backend, sandbox_id).resolve_callback_url(url)

    async def pause(self, backend: str, sandbox_id: str, mode: str = PauseMode.HIBERNATE.value) -> None:
        """
        Pause a sandbox this node owns.
        """
        await self._session(backend, sandbox_id).pause(PauseMode(mode))

    async def resume(self, backend: str, sandbox_id: str) -> None:
        """
        Resume a paused sandbox this node owns.
        """
        await self._session(backend, sandbox_id).resume()

    async def checkpoint(self, backend: str, sandbox_id: str, kind: str, policy_payload: Mapping[str, Any]) -> dict:
        """
        Capture state through the manager, so capability gating and safety policy apply.
        """
        snapshot = await self.manager.checkpoint(
            self._session(backend, sandbox_id),
            SnapshotKind(kind),
            SandboxStatePolicy(**dict(policy_payload)) if policy_payload else None,
        )
        return snapshot_to_payload(snapshot)

    async def restore(
        self,
        snapshot_payload: Mapping[str, Any],
        spec_payload: Mapping[str, Any] | None,
        *,
        callback: str | None = None,
    ) -> dict:
        """
        Restore a snapshot on this node.
        """
        try:
            lease = await self.manager.restore(
                snapshot_from_payload(snapshot_payload),
                spec_from_payload(spec_payload) if spec_payload else None,
            )
        except SandboxProvisionError as exc:
            raise self._provision_failed("restore", exc) from None
        return await self._remember(lease, callback=callback)

    def _provision_failed(self, operation: str, exc: SandboxProvisionError) -> NodeProvisionError:
        """Turn a provision failure into one this node can report across a boundary.

        The partial sandbox stays this node's, because its manager adopted it, so the
        caller is told what happened rather than asked to clean up something it cannot
        reach.
        """
        return NodeProvisionError(
            f"Sandbox node {self.node_id!r} failed to {operation}: {exc.args[0] if exc.args else exc!r}. "
            "This node keeps the partial sandbox and releases it once it can."
        )

    async def release(self, backend: str, sandbox_id: str) -> None:
        """
        Destroy one sandbox and return its capacity.
        """
        lease = self._handles.pop((backend, sandbox_id), None)
        if lease is None:
            return
        await lease.release()

    async def release_group(self, handles: Sequence[Mapping[str, Any]]) -> None:
        """
        Destroy a group, tolerating handles that are already gone.
        """
        await asyncio.gather(
            *(self.release(str(handle["backend"]), str(handle["sandbox_id"])) for handle in handles),
            return_exceptions=True,
        )

    async def sweep(self) -> dict[str, Any]:
        """Run one reclamation sweep and update whether this node still holds work.

        A container the runtime refuses to destroy keeps the node drained. A sweep that
        finds nothing unremovable clears it, so recovery is automatic once the daemon
        cooperates again.
        """
        if self.reclaimer is None:
            return {"removed": [], "unremovable": [], "remaining": None, "draining": self.draining}
        outcome = self.reclaimer.sweep()
        self._unremovable = set(outcome.unremovable)
        if self.draining:
            psrl_logger.warning(
                f"Sandbox node {self.node_id!r} is draining: it could not destroy {sorted(self._unremovable)}."
            )
        evicted = await self.prune_snapshot_cache()
        return {
            "removed": list(outcome.removed),
            "unremovable": list(outcome.unremovable),
            "remaining": outcome.remaining,
            "draining": self.draining,
            "snapshot_cache_evicted": evicted,
        }

    async def prune_snapshot_cache(self) -> list[str]:
        """Drop the least recently used local snapshots that exceed the cache budget.

        The allowance comes from this node's envelope, because only the envelope knows
        how much disk the node has for sandboxes.
        """
        envelope = await self.manager.capacity_snapshot()
        allowance_mb = int((envelope.get("capacity") or {}).get("disk_mb", 0) or 0)
        if allowance_mb <= 0:
            return []
        evicted: list[str] = []
        for backend in self.manager._backends.values():
            evict = getattr(backend, "evict_local_snapshots", None)
            if evict is None:
                continue
            evicted.extend(await evict(allowance_mb))
        return evicted

    async def prefetch(self, references: Sequence[str], *, concurrency: int = 2) -> int:
        """Warm a run's working set on this node, for the backends that can.

        A backend that materializes images lazily has nothing to warm, so a node
        reports zero for it rather than failing a step that is only an optimization.
        """
        warmed = 0
        for backend in self.manager._backends.values():
            warmer = getattr(backend, "prefetch_images", None)
            if callable(warmer):
                warmed += int(await warmer(references, concurrency=concurrency))
        return warmed

    def snapshot(self) -> dict[str, float]:
        """
        Return this node's plane metrics for the trainer's per-step hook.
        """
        return self.manager.snapshot()

    async def shutdown(self) -> None:
        """
        Destroy every sandbox this node owns and close its backends.
        """
        self.stop_liveness()
        await asyncio.gather(
            *(lease.release() for lease in list(self._handles.values())),
            return_exceptions=True,
        )
        self._handles.clear()
        await asyncio.gather(*(forwarder.close() for forwarder in self._forwarders.values()), return_exceptions=True)
        self._forwarders.clear()
        await self.manager.shutdown()
