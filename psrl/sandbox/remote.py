"""The caller side of a remote sandbox node.

A caller holds the lease and the workflow reservation. The node holds the
container and its capacity. That split is the whole point: admission has to live
with the daemon it guards, and ownership has to live with the episode.

The backend presents the same name and the same session protocol as a local one,
so a recipe cannot tell a remote node from the worker's own daemon. That is what
makes the internal Docker backend external to the rest of PSRL rather than a
special case inside it.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from psrl.sandbox.core import (
    ExecResult,
    PauseMode,
    ResumeLevel,
    SandboxBackend,
    SandboxCapabilities,
    SandboxDiagnostics,
    SandboxExitReason,
    SandboxFeature,
    SandboxRef,
    SandboxSession,
    SandboxSourceKind,
    SandboxSpec,
    SandboxStatePolicy,
    SandboxStatus,
    SnapshotKind,
    SnapshotRef,
    rewrite_loopback_proxy,
)
from psrl.sandbox.node_agent import snapshot_to_payload, spec_to_payload

psrl_logger = logging.getLogger(__file__)

# Renewal cadence fallback, used only when the placement TTL cannot be read. Placement's own
# default is a minute, and renewing every third of it is the stated cadence.
_FALLBACK_RESERVATION_TTL_S = 60.0


class RemoteNodeError(RuntimeError):
    """
    Raised when a sandbox node could not serve a request.
    """


def _optional_int(value: Any) -> int | None:
    """
    Read a possibly absent quantity, keeping absence distinct from zero.
    """
    return None if value is None else int(value)


class NodeAgentTransport(Protocol):
    """
    The control channel to the placement service and the node agents.

    It is a protocol so the same caller code runs against an in-process transport
    in a test and a Ray transport in a cluster.
    """

    async def choose(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...

    async def register_node(self, node_id: str) -> None: ...

    async def cancel_reservation(self, reservation_id: str) -> None: ...

    async def renew_reservation(self, reservation_id: str) -> None: ...

    async def reservation_ttl_s(self) -> float: ...

    async def candidate_nodes(self, request: Mapping[str, Any], *, limit: int) -> list[str]: ...

    async def prefetch(self, node_id: str, references: Sequence[str], *, concurrency: int) -> int: ...

    async def release_reservation(self, reservation_id: str) -> None: ...

    async def acquire(
        self,
        node_id: str,
        spec: Mapping[str, Any],
        *,
        callback: str | None = None,
    ) -> Mapping[str, Any]: ...

    async def acquire_group(
        self,
        node_id: str,
        specs: Sequence[Mapping[str, Any]],
        *,
        callback: str | None = None,
    ) -> list[Mapping[str, Any]]: ...

    async def connect(self, node_id: str, backend: str, sandbox_id: str) -> Mapping[str, Any]: ...

    async def exec(
        self,
        node_id: str,
        backend: str,
        sandbox_id: str,
        command: str,
        *,
        cwd: str | None,
        env: Mapping[str, str] | None,
        timeout_s: float | None,
        silence_timeout_s: float | None,
    ) -> Mapping[str, Any]: ...

    async def read_bytes(self, node_id: str, backend: str, sandbox_id: str, path: str) -> bytes: ...

    async def write_bytes(self, node_id: str, backend: str, sandbox_id: str, path: str, data: bytes) -> None: ...

    async def status(self, node_id: str, backend: str, sandbox_id: str) -> str: ...

    async def resolve_callback_url(self, node_id: str, backend: str, sandbox_id: str, url: str) -> str: ...

    async def diagnostics(self, node_id: str, backend: str, sandbox_id: str) -> Mapping[str, Any]: ...

    async def pause(self, node_id: str, backend: str, sandbox_id: str, mode: str) -> None: ...

    async def resume(self, node_id: str, backend: str, sandbox_id: str) -> None: ...

    async def checkpoint(
        self,
        node_id: str,
        backend: str,
        sandbox_id: str,
        kind: str,
        policy: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    async def restore(
        self,
        node_id: str,
        snapshot: Mapping[str, Any],
        spec: Mapping[str, Any] | None,
        *,
        callback: str | None = None,
    ) -> Mapping[str, Any]: ...

    async def release(self, node_id: str, backend: str, sandbox_id: str) -> None: ...


@dataclass(frozen=True)
class RemoteSandbox:
    """
    One provisioned sandbox, as the caller remembers it.
    """

    node_id: str
    backend: str
    sandbox_id: str
    reservation_id: str | None = None
    features: frozenset[SandboxFeature] = frozenset()
    resume_level: ResumeLevel | None = None
    # The alias the hosting node uses to reach the worker that called it, when that
    # node's policy rewrites loopback URLs at all.
    callback_host_alias: str | None = None
    # The port the hosting node forwards to that worker's session server on, when the
    # caller asked for a forwarder. The sandbox reaches it through the alias.
    callback_port: int | None = None

    @property
    def capabilities(self) -> SandboxCapabilities:
        """
        Return the capabilities the hosting node reported for this sandbox.
        """
        return SandboxCapabilities(self.features, resume_level=self.resume_level)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> RemoteSandbox:
        """
        Rebuild a handle from a node's reply.
        """
        level = payload.get("resume_level")
        return cls(
            node_id=str(payload["node_id"]),
            backend=str(payload["backend"]),
            sandbox_id=str(payload["sandbox_id"]),
            reservation_id=payload.get("reservation_id"),
            features=frozenset(SandboxFeature(item) for item in payload.get("features") or ()),
            resume_level=ResumeLevel(str(level)) if level else None,
            callback_host_alias=payload.get("callback_host_alias"),
            callback_port=payload.get("callback_port"),
        )


class InProcessTransport:
    """A transport that calls a local placement service and local node agents.

    A single-node deployment is the same code path with one registry entry, and
    this is what lets that be verified without a cluster.
    """

    def __init__(self, placement, agents: Mapping[str, Any]) -> None:
        self.placement = placement
        self.agents = dict(agents)

    async def register_node(self, node_id: str) -> None:
        """
        Advertise one node to placement, from the node's own report.
        """
        from psrl.sandbox.placement import NodeCapabilities

        agent = self.agents[node_id]
        # The node keeps itself known, so the transport hands it a way to do that before it
        # becomes known. The cadence is the node's, and the mechanism is the transport's.
        agent.liveness_reporter = lambda: self.report_node(node_id)
        advertisement = await agent.advertise()
        for payload in advertisement["backends"]:
            self.placement.register(NodeCapabilities.from_advertisement(payload))
        # A registered node means a live cluster, so this is where the sweeper earns
        # its keep. A service driven from synchronous code enforces its TTL on call.
        self.placement.start_sweeper()

    async def report_node(self, node_id: str) -> None:
        """Re-announce a node placement has forgotten, or refresh one it still knows.

        Heartbeating a node an empty registry never heard of is ignored, so a node that
        outlives a placement restart has to notice and register again. Without that, a
        restart silently drains the whole fleet until the job ends.

        It is not part of the node-agent protocol because it reaches placement from wherever
        the agent runs, and each transport installs its own as the agent's liveness reporter.
        """
        if self.placement.has_node(node_id):
            self.placement.heartbeat(node_id)
            return
        await self.register_node(node_id)

    async def choose(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        from psrl.sandbox.placement import PlacementRequest

        decision = self.placement.choose(PlacementRequest(**request))
        return decision.as_dict()

    async def cancel_reservation(self, reservation_id: str) -> None:
        self.placement.cancel(reservation_id)

    async def renew_reservation(self, reservation_id: str) -> None:
        self.placement.renew(reservation_id)

    async def reservation_ttl_s(self) -> float:
        return self.placement.reservation_ttl_s()

    async def candidate_nodes(self, request: Mapping[str, Any], *, limit: int) -> list[str]:
        from psrl.sandbox.placement import PlacementRequest

        return self.placement.candidates(PlacementRequest(**request), limit=limit)

    async def prefetch(self, node_id: str, references: Sequence[str], *, concurrency: int) -> int:
        return await self.agents[node_id].prefetch(references, concurrency=concurrency)

    async def release_reservation(self, reservation_id: str) -> None:
        self.placement.release(reservation_id)
    async def acquire(
        self,
        node_id: str,
        spec: Mapping[str, Any],
        *,
        callback: str | None = None,
    ) -> Mapping[str, Any]:
        return await self.agents[node_id].acquire(spec, callback=callback)

    async def acquire_group(
        self,
        node_id: str,
        specs: Sequence[Mapping[str, Any]],
        *,
        callback: str | None = None,
    ) -> list[Mapping[str, Any]]:
        return await self.agents[node_id].acquire_group(specs, callback=callback)

    async def connect(self, node_id: str, backend: str, sandbox_id: str) -> Mapping[str, Any]:
        return await self.agents[node_id].connect(backend, sandbox_id)

    async def exec(self, node_id, backend, sandbox_id, command, *, cwd, env, timeout_s, silence_timeout_s):
        return await self.agents[node_id].exec(
            backend,
            sandbox_id,
            command,
            cwd=cwd,
            env=env,
            timeout_s=timeout_s,
            silence_timeout_s=silence_timeout_s,
        )

    async def read_bytes(self, node_id, backend, sandbox_id, path) -> bytes:
        return await self.agents[node_id].read_bytes(backend, sandbox_id, path)

    async def write_bytes(self, node_id, backend, sandbox_id, path, data) -> None:
        await self.agents[node_id].write_bytes(backend, sandbox_id, path, data)

    async def status(self, node_id, backend, sandbox_id) -> str:
        return await self.agents[node_id].status(backend, sandbox_id)

    async def resolve_callback_url(self, node_id, backend, sandbox_id, url) -> str:
        return await self.agents[node_id].resolve_callback_url(backend, sandbox_id, url)

    async def diagnostics(self, node_id, backend, sandbox_id) -> Mapping[str, Any]:
        return await self.agents[node_id].diagnostics(backend, sandbox_id)

    async def pause(self, node_id, backend, sandbox_id, mode) -> None:
        await self.agents[node_id].pause(backend, sandbox_id, mode)

    async def resume(self, node_id, backend, sandbox_id) -> None:
        await self.agents[node_id].resume(backend, sandbox_id)

    async def checkpoint(self, node_id, backend, sandbox_id, kind, policy) -> Mapping[str, Any]:
        return await self.agents[node_id].checkpoint(backend, sandbox_id, kind, policy)

    async def restore(self, node_id, snapshot, spec, *, callback=None) -> Mapping[str, Any]:
        return await self.agents[node_id].restore(snapshot, spec, callback=callback)

    async def release(self, node_id, backend, sandbox_id) -> None:
        await self.agents[node_id].release(backend, sandbox_id)


class NodeAgentTimeout(TimeoutError):
    """
    Raised when a node agent or the placement service did not answer in time.

    It says nothing about the sandbox, so a caller must not read it as a task result. It
    names the call, because the caller's response differs: a timed-out `exec` is a command
    the node may still be running, while a timed-out `renew_reservation` only means the next
    pass has to try again.
    """


class BoundedTransport:
    """Apply a per-call deadline to every call to a node agent or the placement service.

    A remote call has no default deadline, so a wedged agent or a partitioned node blocks
    its caller for the rest of the run. The failure is worse than it looks: the reservation
    renewer makes its own calls, so one call that never returns stops every renewal and
    placement then frees the reservations of a worker that is still running sandboxes.

    It wraps the transport rather than living inside one, so the deadline is one policy
    applied to every transport and a fake can exercise it without a cluster.
    """

    def __init__(self, transport: NodeAgentTransport, *, timeout_s: float = 60.0) -> None:
        if timeout_s <= 0:
            raise ValueError("Bounded transport timeout_s must be greater than zero.")
        self.transport = transport
        self.timeout_s = timeout_s

    async def _bounded(self, name: str, call) -> Any:
        try:
            return await asyncio.wait_for(call(), timeout=self.timeout_s)
        except asyncio.TimeoutError as exc:
            raise NodeAgentTimeout(
                f"Sandbox node call {name!r} did not answer within {self.timeout_s:g}s."
            ) from exc

    async def choose(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        return await self._bounded("choose", lambda: self.transport.choose(request))

    async def register_node(self, node_id: str) -> None:
        await self._bounded("register_node", lambda: self.transport.register_node(node_id))

    async def cancel_reservation(self, reservation_id: str) -> None:
        await self._bounded("cancel_reservation", lambda: self.transport.cancel_reservation(reservation_id))

    async def renew_reservation(self, reservation_id: str) -> None:
        await self._bounded("renew_reservation", lambda: self.transport.renew_reservation(reservation_id))

    async def reservation_ttl_s(self) -> float:
        return await self._bounded("reservation_ttl_s", self.transport.reservation_ttl_s)

    async def candidate_nodes(self, request: Mapping[str, Any], *, limit: int) -> list[str]:
        return await self._bounded("candidate_nodes", lambda: self.transport.candidate_nodes(request, limit=limit))

    async def prefetch(self, node_id: str, references: Sequence[str], *, concurrency: int) -> int:
        return await self._bounded(
            "prefetch", lambda: self.transport.prefetch(node_id, references, concurrency=concurrency)
        )

    async def release_reservation(self, reservation_id: str) -> None:
        await self._bounded("release_reservation", lambda: self.transport.release_reservation(reservation_id))

    async def acquire(
        self,
        node_id: str,
        spec: Mapping[str, Any],
        *,
        callback: str | None = None,
    ) -> Mapping[str, Any]:
        return await self._bounded("acquire", lambda: self.transport.acquire(node_id, spec, callback=callback))

    async def acquire_group(
        self,
        node_id: str,
        specs: Sequence[Mapping[str, Any]],
        *,
        callback: str | None = None,
    ) -> list[Mapping[str, Any]]:
        return await self._bounded(
            "acquire_group", lambda: self.transport.acquire_group(node_id, specs, callback=callback)
        )

    async def connect(self, node_id: str, backend: str, sandbox_id: str) -> Mapping[str, Any]:
        return await self._bounded("connect", lambda: self.transport.connect(node_id, backend, sandbox_id))

    async def exec(
        self,
        node_id: str,
        backend: str,
        sandbox_id: str,
        command: str,
        *,
        cwd: str | None,
        env: Mapping[str, str] | None,
        timeout_s: float | None,
        silence_timeout_s: float | None,
    ) -> Mapping[str, Any]:
        return await self._bounded(
            "exec",
            lambda: self.transport.exec(
                node_id,
                backend,
                sandbox_id,
                command,
                cwd=cwd,
                env=env,
                timeout_s=timeout_s,
                silence_timeout_s=silence_timeout_s,
            ),
        )

    async def read_bytes(self, node_id: str, backend: str, sandbox_id: str, path: str) -> bytes:
        return await self._bounded(
            "read_bytes", lambda: self.transport.read_bytes(node_id, backend, sandbox_id, path)
        )

    async def write_bytes(self, node_id: str, backend: str, sandbox_id: str, path: str, data: bytes) -> None:
        await self._bounded(
            "write_bytes", lambda: self.transport.write_bytes(node_id, backend, sandbox_id, path, data)
        )

    async def status(self, node_id: str, backend: str, sandbox_id: str) -> str:
        return await self._bounded("status", lambda: self.transport.status(node_id, backend, sandbox_id))

    async def resolve_callback_url(self, node_id: str, backend: str, sandbox_id: str, url: str) -> str:
        return await self._bounded(
            "resolve_callback_url", lambda: self.transport.resolve_callback_url(node_id, backend, sandbox_id, url)
        )

    async def diagnostics(self, node_id: str, backend: str, sandbox_id: str) -> Mapping[str, Any]:
        return await self._bounded(
            "diagnostics", lambda: self.transport.diagnostics(node_id, backend, sandbox_id)
        )

    async def pause(self, node_id: str, backend: str, sandbox_id: str, mode: str) -> None:
        await self._bounded("pause", lambda: self.transport.pause(node_id, backend, sandbox_id, mode))

    async def resume(self, node_id: str, backend: str, sandbox_id: str) -> None:
        await self._bounded("resume", lambda: self.transport.resume(node_id, backend, sandbox_id))

    async def checkpoint(
        self,
        node_id: str,
        backend: str,
        sandbox_id: str,
        kind: str,
        policy: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return await self._bounded(
            "checkpoint", lambda: self.transport.checkpoint(node_id, backend, sandbox_id, kind, policy)
        )

    async def restore(
        self,
        node_id: str,
        snapshot: Mapping[str, Any],
        spec: Mapping[str, Any] | None,
        *,
        callback: str | None = None,
    ) -> Mapping[str, Any]:
        return await self._bounded(
            "restore", lambda: self.transport.restore(node_id, snapshot, spec, callback=callback)
        )

    async def release(self, node_id: str, backend: str, sandbox_id: str) -> None:
        await self._bounded("release", lambda: self.transport.release(node_id, backend, sandbox_id))


class RemoteSandboxSession:
    """
    One sandbox living on another node, speaking the local session protocol.
    """

    def __init__(
        self,
        owner: RemoteSandboxBackend,
        transport: NodeAgentTransport,
        remote: RemoteSandbox,
        spec: SandboxSpec | None = None,
    ) -> None:
        self.owner = owner
        self.transport = transport
        self.remote = remote
        self._spec = spec
        self._terminated = False
        self._exit_reason = SandboxExitReason.UNKNOWN

    @property
    def ref(self) -> SandboxRef:
        return SandboxRef(self.remote.backend, self.remote.sandbox_id)

    @property
    def capabilities(self) -> SandboxCapabilities:
        return self.remote.capabilities

    @property
    def spec(self) -> SandboxSpec | None:
        return self._spec

    @property
    def exit_reason(self) -> SandboxExitReason:
        return self._exit_reason

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
        silence_timeout_s: float | None = None,
    ) -> ExecResult:
        payload = await self.transport.exec(
            self.remote.node_id,
            self.remote.backend,
            self.remote.sandbox_id,
            command,
            cwd=cwd,
            env=env,
            timeout_s=timeout_s,
            silence_timeout_s=silence_timeout_s,
        )
        return ExecResult(
            exit_code=int(payload["exit_code"]),
            stdout=str(payload["stdout"]),
            stderr=str(payload["stderr"]),
            truncated=bool(payload["truncated"]),
        )

    async def read_bytes(self, path: str) -> bytes:
        return await self.transport.read_bytes(self.remote.node_id, self.remote.backend, self.remote.sandbox_id, path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        await self.transport.write_bytes(
            self.remote.node_id, self.remote.backend, self.remote.sandbox_id, path, data
        )

    async def status(self) -> SandboxStatus:
        if self._terminated:
            return SandboxStatus.TERMINATED
        raw = await self.transport.status(self.remote.node_id, self.remote.backend, self.remote.sandbox_id)
        return _status_from(raw)

    async def diagnostics(self) -> SandboxDiagnostics:
        payload = await self.transport.diagnostics(
            self.remote.node_id, self.remote.backend, self.remote.sandbox_id
        )
        usage = payload.get("usage") or {}
        from psrl.sandbox.core import ResourceUsage

        return SandboxDiagnostics(
            ref=self.ref,
            status=_status_from(str(payload.get("status", "unknown"))),
            exit_reason=SandboxExitReason(str(payload.get("exit_reason", "unknown"))),
            log_tail=str(payload.get("log_tail", "")),
            inspect=dict(payload.get("inspect") or {}),
            usage=ResourceUsage(
                memory_bytes=_optional_int(usage.get("memory_bytes")),
                peak_memory_bytes=_optional_int(usage.get("peak_memory_bytes")),
                cpu_total_ns=_optional_int(usage.get("cpu_total_ns")),
            ),
        )

    async def pause(self, mode: PauseMode) -> None:
        await self.transport.pause(
            self.remote.node_id, self.remote.backend, self.remote.sandbox_id, mode.value
        )

    async def resume(self) -> None:
        await self.transport.resume(self.remote.node_id, self.remote.backend, self.remote.sandbox_id)

    async def snapshot(self, kind: SnapshotKind) -> SnapshotRef:
        from psrl.sandbox.node_agent import snapshot_from_payload

        policy = self._spec.state_policy if self._spec is not None else SandboxStatePolicy()
        payload = await self.transport.checkpoint(
            self.remote.node_id,
            self.remote.backend,
            self.remote.sandbox_id,
            kind.value,
            {
                "enabled": policy.enabled,
                "allow_secret_capture": policy.allow_secret_capture,
                "allow_external_side_effects": policy.allow_external_side_effects,
                "reseed_after_restore": policy.reseed_after_restore,
            },
        )
        return snapshot_from_payload(payload)

    def resolve_callback_url(self, url: str) -> str:
        """Translate a caller URL into one the sandbox can reach.

        The node reports the alias it uses for the worker it was reached from, and the
        port it forwards that worker's session server on, and the caller applies both.
        Deciding it on the node and applying it here keeps this a pure string operation:
        asking the node per URL would make a synchronous contract depend on a round trip.
        """
        if self.remote.callback_host_alias is None:
            return url
        return rewrite_loopback_proxy(url, self.remote.callback_host_alias, port=self.remote.callback_port)

    async def refresh_transport(self) -> None:
        """
        A remote session holds no provider connection of its own, so a reconnect is
        a no-op here and the node's own transport is refreshed by the node.
        """
        return None

    async def fork(self, count: int = 1) -> Sequence[SandboxSession]:
        """A remote node forks its own sandbox, so this is not a caller operation.

        The group path asks the node for a whole group rather than forking a handle,
        because a fork has to be one provider call on the node that holds the parent.
        """
        raise NotImplementedError(
            "A remote sandbox forks through the group path on its own node, not through a caller handle."
        )

    async def terminate(self) -> None:
        if self._terminated:
            return
        await self.transport.release(self.remote.node_id, self.remote.backend, self.remote.sandbox_id)
        if self.remote.reservation_id:
            await self.transport.release_reservation(self.remote.reservation_id)
        self._terminated = True
        self._exit_reason = SandboxExitReason.RELEASED
        # A finished sandbox must leave the worker's index, or a long run accumulates
        # one handle per sandbox it ever made and keeps renewing dead reservations.
        self.owner._forget_handle(self.remote.sandbox_id)


class RemoteSandboxBackend(SandboxBackend):
    """A `SandboxBackend` whose sandboxes live on other nodes.

    It presents the name of the backend its nodes run, so a spec that names that
    backend reaches a remote node by configuration rather than by a code change.
    """

    def __init__(
        self,
        transport: NodeAgentTransport,
        *,
        name: str = "docker",
        owner_id: str = "",
        required_label: str | None = None,
        capabilities: SandboxCapabilities | None = None,
        callback_target: str | None = None,
        renew_interval_s: float | None = None,
    ) -> None:
        self.transport = transport
        self._name = name
        self.owner_id = owner_id
        self.required_label = required_label
        self._declared = capabilities or SandboxCapabilities()
        # This worker's own session server, as a host:port a node can reach. The node
        # opens a forwarder to it, so a sandbox on any node resolves the same URL.
        self.callback_target = callback_target
        # A deployment that wants a different renewal cadence sets this. Unset derives it
        # from the placement TTL, which is the only value that has to be respected.
        if renew_interval_s is not None and renew_interval_s <= 0:
            raise ValueError("Remote sandbox renew_interval_s must be greater than zero when set.")
        self._renew_interval_s = renew_interval_s
        self._handles: dict[str, RemoteSandbox] = {}
        self._renew_task: asyncio.Task[None] | None = None
        # A reservation exists from the moment placement chooses a node, well before the
        # sandbox does, so it is renewed from that moment rather than from the create.
        self._pending_reservations: set[str] = set()
        self._known_ttl_s: float | None = None

    def _live_reservations(self) -> list[str]:
        """
        Return every reservation this worker is still responsible for.
        """
        held = {handle.reservation_id for handle in self._handles.values() if handle.reservation_id is not None}
        return sorted(held | self._pending_reservations)

    def start_reservation_renewer(self) -> None:
        """Keep telling placement that this worker still holds its reservations.

        Placement sweeps a reservation an owner stops renewing, because it holds a
        cache and not a ledger: it cannot tell a live long-lived sandbox from an
        abandoned reservation by age alone. Renewing well inside the TTL means one
        lost round trip does not cost the reservation.

        Started as soon as a reservation exists, because the wait for node admission can
        outlast the TTL by a wide margin and the reservation has to survive that wait.
        The loop stops once nothing is left to renew.
        """
        if self._renew_task is not None and not self._renew_task.done():
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # A caller with no loop has no reservations to renew either.
            return
        self._renew_task = asyncio.create_task(self._renew_loop())

    async def _reservation_ttl_s(self) -> float:
        """Return the placement TTL, reusing the last answer when the lookup fails.

        A failed lookup must not stop renewal. Falling back to the last known value keeps
        the cadence, where propagating the error would leave every reservation to expire
        because one RPC was slow.
        """
        try:
            self._known_ttl_s = float(await self.transport.reservation_ttl_s())
        except Exception:
            if self._known_ttl_s is None:
                self._known_ttl_s = _FALLBACK_RESERVATION_TTL_S
            psrl_logger.warning(
                "Sandbox placement TTL lookup failed, renewing on the last known %.1fs.",
                self._known_ttl_s,
                exc_info=True,
            )
        return self._known_ttl_s if self._known_ttl_s is not None else _FALLBACK_RESERVATION_TTL_S

    async def _renew_loop(self) -> None:
        while self._live_reservations():
            interval = self._renew_interval_s
            if interval is None:
                interval = max(1.0, await self._reservation_ttl_s() / 3.0)
            await asyncio.sleep(interval)
            await self.renew_reservations()

    async def renew_reservations(self) -> int:
        """Renew every reservation this worker is still responsible for.

        A renewal that fails is reported rather than raised, because the next pass retries
        and losing the loop would guarantee the reservations expire.
        """
        reservation_ids = self._live_reservations()
        if not reservation_ids:
            return 0
        results = await asyncio.gather(
            *(self.transport.renew_reservation(reservation_id) for reservation_id in reservation_ids),
            return_exceptions=True,
        )
        for reservation_id, result in zip(reservation_ids, results, strict=True):
            if isinstance(result, BaseException):
                psrl_logger.warning(
                    "Sandbox placement renewal failed for reservation %s.", reservation_id, exc_info=result
                )
        return len(reservation_ids)

    async def prefetch_images(
        self,
        references: Sequence[str],
        *,
        concurrency: int = 2,
        nodes: int = 1,
    ) -> int:
        """Warm a run's working set on the nodes a task is likely to land on.

        Bounded to the top ranked candidates for each reference, because warming the
        whole fleet would pay a pull per node for images most nodes will never serve.
        A node that cannot be reached or cannot materialize an image reports nothing,
        since a prefetch is an optimization and the task still runs either way.
        """
        if nodes < 1:
            raise ValueError("A remote prefetch must target at least one node.")
        warmed = 0
        for reference in references:
            request = {"backend": self._name, "image_references": [reference]}
            targets = await self.transport.candidate_nodes(request, limit=nodes)
            for node_id in targets:
                warmed += await self.transport.prefetch(node_id, [reference], concurrency=concurrency)
        return warmed

    def _forget_handle(self, sandbox_id: str) -> None:
        """
        Drop a finished sandbox from the worker's index.
        """
        self._handles.pop(sandbox_id, None)

    async def shutdown(self) -> None:
        """
        Stop renewing reservations, which releases them to the sweeper.
        """
        if self._renew_task is not None:
            self._renew_task.cancel()
            await asyncio.gather(self._renew_task, return_exceptions=True)
            self._renew_task = None
        self._handles.clear()

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> SandboxCapabilities:
        """
        Return the capabilities this deployment declared for its nodes.

        The declaration is configuration, not a union of whatever happens to be
        registered: a capability that only one node has cannot be promised to a
        caller, and a requirement that is sometimes met is worse than one that is
        refused.
        """
        return self._declared

    @property
    def uses_node_capacity(self) -> bool:
        """
        Return that capacity is charged on the node, not on this worker.
        """
        return False

    def _claim_reservation(self, reservation_id: str) -> None:
        """Start protecting a reservation that does not have a sandbox yet.

        Placement chose the node, so the slot is already charged. The wait for node
        admission can outlast the reservation TTL by a wide margin, and an un-renewed
        reservation is swept and re-granted to somebody else while this call is still
        waiting, so protection starts here rather than after the sandbox exists.
        """
        self._pending_reservations.add(reservation_id)
        self.start_reservation_renewer()

    def _settle_reservation(self, reservation_id: str) -> None:
        """
        Hand a reservation over to the handle index, or drop it when nothing was created.
        """
        self._pending_reservations.discard(reservation_id)

    async def _abandon_reservation(self, reservation_id: str) -> None:
        """
        Withdraw a reservation whose sandbox was never created.
        """
        self._settle_reservation(reservation_id)
        await asyncio.gather(self.transport.cancel_reservation(reservation_id), return_exceptions=True)

    async def create(self, spec: SandboxSpec) -> SandboxSession:
        """
        Pick a node, provision there, and hand back a session that speaks locally.
        """
        decision = await self.transport.choose(self._placement_request(spec))
        reservation_id = str(decision["reservation_id"])
        node_id = str(decision["node_id"])
        self._claim_reservation(reservation_id)
        try:
            payload = await self.transport.acquire(node_id, spec_to_payload(spec), callback=self.callback_target)
        except BaseException:
            # A provision that never happened must not leave the node reserved, or the
            # cluster starves one slot at a time.
            await self._abandon_reservation(reservation_id)
            raise
        # The running loop renews it from here. Renewing again would add a failure path to
        # a create whose sandbox exists, stranding a session the caller cannot release.
        self._settle_reservation(reservation_id)
        remote = self._record(payload, node_id=node_id, reservation_id=reservation_id)
        return RemoteSandboxSession(self, self.transport, remote, spec)

    async def acquire_group(self, specs: Sequence[SandboxSpec]) -> list[SandboxSession]:
        """Provision a whole group on one node, so a fork is a single provider call.

        The caller builds the member specs. The node decides whether it forks or
        creates, because only the node knows what its provider supports.
        """
        members = list(specs)
        if not members:
            raise ValueError("A remote sandbox group requires at least one member spec.")
        decision = await self.transport.choose(self._placement_request(members[0]))
        reservation_id = str(decision["reservation_id"])
        node_id = str(decision["node_id"])
        self._claim_reservation(reservation_id)
        try:
            payloads = await self.transport.acquire_group(
                node_id,
                [spec_to_payload(spec) for spec in members],
                callback=self.callback_target,
            )
        except BaseException:
            await self._abandon_reservation(reservation_id)
            raise
        self._settle_reservation(reservation_id)
        sessions = []
        for spec, payload in zip(members, payloads, strict=True):
            remote = self._record(payload, node_id=node_id, reservation_id=reservation_id)
            sessions.append(RemoteSandboxSession(self, self.transport, remote, spec))
        return sessions

    async def connect(self, sandbox_id: str) -> SandboxSession:
        """
        Re-adopt a sandbox this deployment already hosts.
        """
        known = self._handles.get(sandbox_id)
        if known is None:
            raise RemoteNodeError(
                f"Remote sandbox {sandbox_id!r} is not in this worker's handle index, so no node can be asked "
                "for it. Re-adoption after a worker restart needs the node to be named, which is a placement "
                "record the caller does not keep."
            )
        payload = await self.transport.connect(known.node_id, known.backend, sandbox_id)
        remote = self._record(payload, node_id=known.node_id, reservation_id=known.reservation_id)
        return RemoteSandboxSession(self, self.transport, remote)

    async def restore(self, snapshot: SnapshotRef, spec: SandboxSpec | None = None) -> SandboxSession:
        """
        Restore a snapshot on whichever node placement picks.
        """
        request = self._placement_request(spec, snapshot=snapshot)
        decision = await self.transport.choose(request)
        reservation_id = str(decision["reservation_id"])
        node_id = str(decision["node_id"])
        self._claim_reservation(reservation_id)
        try:
            payload = await self.transport.restore(
                node_id,
                snapshot_to_payload(snapshot),
                spec_to_payload(spec) if spec is not None else None,
                callback=self.callback_target,
            )
        except BaseException:
            await self._abandon_reservation(reservation_id)
            raise
        self._settle_reservation(reservation_id)
        remote = self._record(payload, node_id=node_id, reservation_id=reservation_id)
        return RemoteSandboxSession(self, self.transport, remote, spec)

    def restore_node(self, node_id: str) -> SandboxSpec | None:
        """
        Return nothing, kept explicit so a reader does not look for a hidden request.
        """
        return None

    def _record(self, payload: Mapping[str, Any], *, node_id: str, reservation_id: str) -> RemoteSandbox:
        """
        Remember a handle and index it by sandbox id.
        """
        enriched = dict(payload)
        enriched.setdefault("node_id", node_id)
        enriched["reservation_id"] = reservation_id
        declared = self._declared
        enriched.setdefault("features", sorted(feature.value for feature in declared.features))
        if declared.resume_level is not None:
            enriched.setdefault("resume_level", declared.resume_level.value)
        remote = RemoteSandbox.from_payload(enriched)
        self._handles[remote.sandbox_id] = remote
        return remote

    def _placement_request(self, spec: SandboxSpec | None, *, snapshot: SnapshotRef | None = None) -> dict[str, Any]:
        """Turn a spec, and optionally the snapshot it restores, into a placement request.

        A restore that has to land on another host is a resume requirement, and the
        level comes from the snapshot rather than from the caller, so a filesystem
        snapshot is never placed as a full-state resume.
        """
        features: set[SandboxFeature] = set(spec.required_features) if spec is not None else set()
        level = spec.required_resume_level if spec is not None else None
        if snapshot is not None:
            features.add(SandboxFeature.RESTORE)
            level = snapshot.resume_level or level
        references: set[str] = set()
        digests: set[str] = set()
        if spec is not None and spec.source.kind is SandboxSourceKind.IMAGE:
            # A source that already names a digest is exact, and one that names a tag is what a
            # caller usually has, so both travel and placement trusts the digest more.
            if "@" in spec.source.reference:
                digests.add(spec.source.reference)
            else:
                references.add(spec.source.reference)
        return {
            "backend": self._name,
            "required_features": [feature.value for feature in features],
            "required_resume_level": level.value if level else None,
            "requires_host_mount": bool(spec.mounts) if spec is not None else False,
            "gpu_count": int(spec.resources.gpu_count or 0) if spec is not None else 0,
            "required_label": self.required_label,
            "image_references": sorted(references),
            "image_digests": sorted(digests),
            "owner_id": self.owner_id,
        }

    def snapshot(self) -> dict[str, float]:
        """
        Return this backend's caller-side counters.
        """
        return {
            "remote/handles": float(len(self._handles)),
        }


def _status_from(value: str) -> SandboxStatus:
    """
    Map a node's reported status onto the portable one.
    """
    try:
        return SandboxStatus(value)
    except ValueError:
        return SandboxStatus.UNKNOWN
