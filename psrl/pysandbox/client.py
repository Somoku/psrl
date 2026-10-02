"""The client and the sandbox handle a caller actually uses.

Two channels, and the split is the whole performance argument.

- **Control plane**, over the service socket: create, release, snapshot, pause,
  and the reports. Roughly two calls per episode.
- **Data plane**, straight to the sandbox: commands and files. Dozens of calls
  per episode, and none of them needs a decision from the fleet, so none of them
  pays for a hop through it.

`create` returns the resolved agent endpoint with the handle, so the data plane
costs no extra round trip to discover.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any
from urllib.parse import urlsplit

from psrl.pysandbox.errors import (
    SandboxCapabilityError,
    SandboxCapacityTimeout,
    SandboxCommandTimeout,
    SandboxError,
    SandboxSessionLost,
    SandboxTransportError,
)
from psrl.pysandbox.types import (
    AgentEndpoint,
    Capabilities,
    ClassQuota,
    ExecResult,
    Feature,
    FleetReport,
    NodeReport,
    PauseMode,
    QuotaReport,
    ResumeLevel,
    SandboxHandle,
    SandboxSpec,
    SandboxStatus,
    SnapshotKind,
    SnapshotRef,
)

DEFAULT_ENDPOINT = "unix:///run/sandboxd.sock"

# A control call asks a question or moves a reservation, so a healthy one answers
# in milliseconds and one that has not answered in a minute is a fault. A command
# carries its own budget instead, because its duration is the work itself.
DEFAULT_CONTROL_TIMEOUT_S = 60.0


class Transport:
    """One connection to the service, speaking length-prefixed JSON.

    JSON rather than protobuf on the Python side: the SDK's job is to serialize
    and forward, the control path is two calls per episode, and a generated stub
    would make the SDK's install depend on a protobuf runtime for no measurable
    gain. The wire shape is still the one the proto defines.
    """

    def __init__(self, endpoint: str, *, timeout_s: float = DEFAULT_CONTROL_TIMEOUT_S) -> None:
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._lock = asyncio.Lock()

    async def _connect(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        if self._reader is not None and self._writer is not None and not self._writer.is_closing():
            return self._reader, self._writer
        if self.endpoint.startswith("unix://"):
            path = self.endpoint[len("unix://") :]
            self._reader, self._writer = await asyncio.open_unix_connection(path)
        else:
            parsed = urlsplit(self.endpoint if "://" in self.endpoint else f"//{self.endpoint}")
            if parsed.hostname is None or parsed.port is None:
                raise SandboxTransportError(f"A service endpoint must be unix:// or host:port, got {self.endpoint!r}.")
            self._reader, self._writer = await asyncio.open_connection(parsed.hostname, parsed.port)
        return self._reader, self._writer

    async def call(self, method: str, payload: Mapping[str, Any], *, timeout_s: float | None = None) -> dict[str, Any]:
        """Send one request and return its reply.

        Serialized on one connection, because the control path is low rate and a
        connection per call would cost more than it saves.
        """
        budget = self.timeout_s if timeout_s is None else timeout_s
        async with self._lock:
            try:
                return await asyncio.wait_for(self._roundtrip(method, payload), timeout=budget)
            except asyncio.TimeoutError as exc:
                # The service may still be acting on it, so the connection is
                # dropped rather than reused with an unread reply in the pipe.
                await self.close()
                raise SandboxTransportError(
                    f"The sandbox service did not answer {method!r} within {budget:g}s."
                ) from exc
            except (ConnectionError, OSError) as exc:
                await self.close()
                raise SandboxTransportError(f"The sandbox service connection failed during {method!r}: {exc}") from exc

    async def _roundtrip(self, method: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        reader, writer = await self._connect()
        request = json.dumps({"method": method, "payload": dict(payload)}).encode()
        writer.write(len(request).to_bytes(4, "big") + request)
        await writer.drain()
        header = await reader.readexactly(4)
        body = await reader.readexactly(int.from_bytes(header, "big"))
        reply = json.loads(body)
        if error := reply.get("error"):
            raise _error_for(error.get("code", ""), error.get("message", ""))
        return reply.get("result") or {}

    async def close(self) -> None:
        """Drop the connection."""
        writer, self._writer, self._reader = self._writer, None, None
        if writer is None:
            return
        writer.close()
        try:
            await writer.wait_closed()
        except (ConnectionError, OSError):
            # The peer is already gone, which is the state close wanted.
            pass


def _error_for(code: str, message: str) -> SandboxError:
    """Rebuild a typed failure from the service's reply.

    The type is the service's classification, not a guess from the message: the
    caller's response differs per type, and an infrastructure fault read as a
    task result corrupts a reward.
    """
    return {
        "capacity_timeout": SandboxCapacityTimeout,
        "capability": SandboxCapabilityError,
        "session_lost": SandboxSessionLost,
        "command_timeout": SandboxCommandTimeout,
        "transport": SandboxTransportError,
    }.get(code, SandboxError)(message)


class Sandbox:
    """One sandbox. Commands go straight to it; lifecycle goes to the service."""

    def __init__(
        self,
        client: SandboxClient,
        handle: SandboxHandle,
        capabilities: Capabilities,
        agent: AgentEndpoint,
        *,
        warm_start: bool = False,
    ) -> None:
        self.handle = handle
        self.capabilities = capabilities
        self.agent = agent
        self.warm_start = warm_start
        self._client = client
        self._released = False

    @property
    def sandbox_id(self) -> str:
        """Return the backend's id for this sandbox."""
        return self.handle.sandbox_id

    @property
    def node_id(self) -> str:
        """Return the node holding this sandbox."""
        return self.handle.node_id

    # -- Data plane: straight to the sandbox -----------------------------------

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
        silence_timeout_s: float | None = None,
    ) -> ExecResult:
        """Run one command and collect its result.

        `silence_timeout_s` exists beside `timeout_s` because a command that
        prints nothing for a long stretch is stuck rather than slow, and the two
        need different answers.

        Raises:
            SandboxCommandTimeout: When the deadline expires. Its
                `sandbox_preserved` says whether the sandbox survived, which
                decides between continuing with a fresh shell and replacing it.
            SandboxOomError: When the kernel killed the sandbox mid-command.
        """
        return await self._client._exec(
            self.handle,
            self.agent,
            command,
            cwd=cwd,
            env=env,
            timeout_s=timeout_s,
            silence_timeout_s=silence_timeout_s,
        )

    async def read_bytes(self, path: str) -> bytes:
        """Read one file, without assuming a text encoding."""
        return await self._client._read_bytes(self.handle, self.agent, path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        """Write one complete file."""
        await self._client._write_bytes(self.handle, self.agent, path, data)

    # -- Control plane ---------------------------------------------------------

    async def status(self) -> SandboxStatus:
        """Return the sandbox's portable state."""
        reply = await self._client._call("status", _handle_payload(self.handle))
        return SandboxStatus(reply.get("status", "unknown"))

    async def snapshot(self, kind: SnapshotKind = SnapshotKind.FILESYSTEM) -> SnapshotRef:
        """Capture this sandbox's state."""
        reply = await self._client._call("snapshot", {**_handle_payload(self.handle), "kind": kind.value})
        return SnapshotRef(
            backend=reply.get("backend", self.handle.backend),
            snapshot_id=reply["snapshot_id"],
            kind=SnapshotKind(reply.get("kind", kind.value)),
            resume_level=_resume_level(reply.get("resume_level")),
        )

    async def pause(self, mode: PauseMode = PauseMode.HIBERNATE) -> None:
        """Pause this sandbox with explicit semantics."""
        await self._client._call("pause", {**_handle_payload(self.handle), "mode": mode.value})

    async def resume(self) -> None:
        """Resume a paused sandbox."""
        await self._client._call("resume", _handle_payload(self.handle))

    async def release(self) -> None:
        """Destroy this sandbox and return everything it held.

        Idempotent: a retried release is normal, because the first may have timed
        out after the service already destroyed the sandbox.
        """
        if self._released:
            return
        await self._client._call("release", _handle_payload(self.handle))
        self._released = True

    async def __aenter__(self) -> Sandbox:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.release()


class SandboxClient:
    """A connection to one sandbox service."""

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        *,
        control_timeout_s: float = DEFAULT_CONTROL_TIMEOUT_S,
        agent_factory: Any = None,
    ) -> None:
        self._transport = Transport(endpoint, timeout_s=control_timeout_s)
        # Returns a runner for a sandbox that has its own agent, or None when it
        # has none and the owning node proxies instead. Injected so a test can
        # drive the data plane, and so a deployment whose agents speak another
        # protocol can supply its own.
        self._agent_factory = agent_factory or (lambda _agent: None)

    async def create(self, spec: SandboxSpec) -> Sandbox:
        """Provision one sandbox.

        Raises:
            SandboxCapabilityError: When no backend can serve the spec. Waiting
                will not add a capability, so this is a deployment fault.
            SandboxCapacityTimeout: When the fleet had no room in time. This
                clears as sandboxes are released.
        """
        reply = await self._call("create", {"spec": _spec_payload(spec)})
        return self._sandbox_from(reply)

    async def create_group(self, specs: Sequence[SandboxSpec]) -> list[Sandbox]:
        """Provision one group.

        Members are admitted independently, because a group is a completion unit
        rather than a scheduling unit: the algorithm needs its trajectories to
        finish together, not to start together. A failed group leaves nothing
        charged.
        """
        if not specs:
            raise ValueError("A sandbox group needs at least one member spec.")
        reply = await self._call("create_group", {"specs": [_spec_payload(spec) for spec in specs]})
        return [self._sandbox_from(member) for member in reply.get("members", [])]

    async def restore(self, snapshot: SnapshotRef, spec: SandboxSpec | None = None) -> Sandbox:
        """Create a sandbox from a capture."""
        payload: dict[str, Any] = {"snapshot": _snapshot_payload(snapshot)}
        if spec is not None:
            payload["spec"] = _spec_payload(spec)
        return self._sandbox_from(await self._call("restore", payload))

    async def delete_snapshot(self, snapshot: SnapshotRef) -> None:
        """Delete a capture."""
        await self._call("delete_snapshot", _snapshot_payload(snapshot))

    async def fleet(self) -> FleetReport:
        """Return what the monitor sees, for a metric hook."""
        reply = await self._call("fleet", {})
        return FleetReport(
            nodes=tuple(
                NodeReport(
                    node_id=node.get("node_id", ""),
                    backend=node.get("backend", ""),
                    live_sandboxes=int(node.get("live_sandboxes", 0)),
                    cpu_used_pct=float(node.get("cpu_used_pct", 0.0)),
                    mem_used_pct=float(node.get("mem_used_pct", 0.0)),
                    gpu_free=int(node.get("gpu_free", 0)),
                    draining=bool(node.get("draining", False)),
                )
                for node in reply.get("nodes", [])
            ),
            drained_nodes=int(reply.get("drained_nodes", 0)),
            reservations_open=int(reply.get("reservations_open", 0)),
            capacity_exhausted=int(reply.get("capacity_exhausted", 0)),
            no_candidate=int(reply.get("no_candidate", 0)),
            locality_hit_ratio=float(reply.get("locality_hit_ratio", 0.0)),
        )

    async def quota(self) -> QuotaReport:
        """Return the cross-backend ledger, for a metric hook."""
        reply = await self._call("quota", {})
        return QuotaReport(
            classes={
                name: ClassQuota(
                    guaranteed_share=float(entry.get("guaranteed_share", 0.0)),
                    max_share=float(entry.get("max_share", 0.0)),
                    granted_memory_mb=int(entry.get("granted_memory_mb", 0)),
                    headroom_memory_mb=int(entry.get("headroom_memory_mb", 0)),
                    queued=int(entry.get("queued", 0)),
                )
                for name, entry in (reply.get("classes") or {}).items()
            }
        )

    async def close(self) -> None:
        """Close the connection to the service."""
        await self._transport.close()

    async def __aenter__(self) -> SandboxClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    # -- Internals -------------------------------------------------------------

    async def _call(self, method: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        return await self._transport.call(method, payload)

    def _sandbox_from(self, reply: Mapping[str, Any]) -> Sandbox:
        handle = reply.get("handle") or {}
        agent = reply.get("agent") or {}
        capabilities = reply.get("capabilities") or {}
        return Sandbox(
            self,
            SandboxHandle(
                backend=handle.get("backend", ""),
                sandbox_id=handle.get("sandbox_id", ""),
                node_id=handle.get("node_id", ""),
            ),
            Capabilities(
                features=frozenset(Feature(name) for name in capabilities.get("features", [])),
                resume_level=_resume_level(capabilities.get("resume_level")),
                pause_modes=frozenset(PauseMode(name) for name in capabilities.get("pause_modes", [])),
            ),
            AgentEndpoint(
                address=agent.get("address", ""),
                headers=agent.get("headers") or {},
                callback_host_alias=agent.get("callback_host_alias"),
                callback_port=agent.get("callback_port"),
            ),
            warm_start=bool(reply.get("warm_start", False)),
        )

    async def _exec(
        self,
        handle: SandboxHandle,
        agent: AgentEndpoint,
        command: str,
        *,
        cwd: str | None,
        env: Mapping[str, str] | None,
        timeout_s: float | None,
        silence_timeout_s: float | None,
    ) -> ExecResult:
        """Run a command, direct to the sandbox where it has its own agent.

        A backend whose sandboxes run no agent of their own -- a bare container
        is the case -- reports an empty address, and its node proxies instead.
        Either way the traffic stays node-local and off the control plane.
        """
        payload = {
            **_handle_payload(handle),
            "command": command,
            "cwd": cwd,
            "env": dict(env or {}),
            "timeout_s": timeout_s,
            "silence_timeout_s": silence_timeout_s,
        }
        reply = await self._run(agent, "exec", payload, timeout_s=timeout_s)
        return ExecResult(
            exit_code=int(reply["exit_code"]),
            stdout=reply.get("stdout", ""),
            stderr=reply.get("stderr", ""),
            truncated=bool(reply.get("truncated", False)),
        )

    async def _read_bytes(self, handle: SandboxHandle, agent: AgentEndpoint, path: str) -> bytes:
        reply = await self._run(agent, "read_bytes", {**_handle_payload(handle), "path": path}, timeout_s=None)
        return base64.b64decode(reply["data"])

    async def _write_bytes(self, handle: SandboxHandle, agent: AgentEndpoint, path: str, data: bytes) -> None:
        await self._run(
            agent,
            "write_bytes",
            {**_handle_payload(handle), "path": path, "data": base64.b64encode(data).decode()},
            timeout_s=None,
        )

    async def _run(
        self,
        agent: AgentEndpoint,
        method: str,
        payload: Mapping[str, Any],
        *,
        timeout_s: float | None,
    ) -> dict[str, Any]:
        """Dispatch one data-plane call to the sandbox, or to the node proxying it.

        A provider sandbox runs its own agent, so the call goes straight there. A
        bare container runs none, so the owning node proxies -- still one hop and
        still node-local, with the control plane out of the path either way.

        A command's deadline is its own budget and is enforced where the command
        runs, so it travels with the call rather than being capped here.
        """
        runner = self._agent_factory(agent)
        if runner is not None:
            return await runner(method, payload, timeout_s=timeout_s)
        return await self._transport.call(method, payload, timeout_s=timeout_s or DEFAULT_CONTROL_TIMEOUT_S)


def _handle_payload(handle: SandboxHandle) -> dict[str, Any]:
    return {"backend": handle.backend, "sandbox_id": handle.sandbox_id, "node_id": handle.node_id}


def _snapshot_payload(snapshot: SnapshotRef) -> dict[str, Any]:
    return {"backend": snapshot.backend, "snapshot_id": snapshot.snapshot_id, "kind": snapshot.kind.value}


def _spec_payload(spec: SandboxSpec) -> dict[str, Any]:
    """Render a spec for the wire, omitting what the caller did not state.

    An omitted resource stays omitted rather than becoming zero: a dimension
    nobody named must not be compared against a limit.
    """
    payload: dict[str, Any] = {
        "source": {"kind": spec.source.kind, "reference": spec.source.reference},
        "resource_class": spec.resource_class,
    }
    resources = {
        name: value
        for name, value in (
            ("cpu_count", spec.resources.cpu_count),
            ("memory_mb", spec.resources.memory_mb),
            ("disk_mb", spec.resources.disk_mb),
            ("gpu_count", spec.resources.gpu_count),
        )
        if value is not None
    }
    if resources:
        payload["resources"] = resources
    for name, value in (
        ("workflow_id", spec.workflow_id),
        ("idempotency_key", spec.idempotency_key),
        ("workdir", spec.workdir),
        ("backend", spec.backend),
    ):
        if value:
            payload[name] = value
    if spec.env:
        payload["env"] = dict(spec.env)
    if spec.metadata:
        payload["metadata"] = dict(spec.metadata)
    if spec.exec_mode is not None:
        payload["exec_mode"] = spec.exec_mode.value
    if spec.required_features:
        payload["required_features"] = sorted(feature.value for feature in spec.required_features)
    if spec.required_resume_level is not None:
        payload["required_resume_level"] = spec.required_resume_level.value
    if spec.backend_options:
        payload["backend_options"] = {name: dict(values) for name, values in spec.backend_options.items()}
    return payload


def _resume_level(value: Any) -> ResumeLevel | None:
    if not value:
        return None
    return ResumeLevel(value)
