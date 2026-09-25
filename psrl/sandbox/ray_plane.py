"""The Ray transport and the actor topology one job uses to place sandboxes.

The sandbox plane is transport-agnostic by design, and this is the only module that knows
it runs on Ray. It is deliberately **not** re-exported from `psrl.sandbox`, because that
package eagerly re-exports every symbol and Ray is an optional dependency: importing
`psrl.sandbox` must keep working where Ray is absent.

Two things are created per job.

- One `PlacementService`, which is the job's placement authority. It holds a cache and
  rebuilds from the node agents, so it is safe to lose and rebuild.
- One `SandboxNodeActor` per sandbox node, pinned to that node with hard affinity. Each
  owns the manager for its own node, because admission has to live with the daemon it
  guards: a caller that reached another node's daemon would create containers whose
  memory its own accounting never sees.

A worker then holds a `RayNodeAgentTransport` over those handles, which is what makes a
sandbox on another node look like a local one to everything above it.
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from psrl.sandbox.capacity import SandboxCapacityCoordinator
from psrl.sandbox.config import build_sandbox_manager, resolve_capacity
from psrl.sandbox.core import SandboxCapabilities
from psrl.sandbox.node_agent import SandboxNodeAgent
from psrl.sandbox.placement import NodeCapabilities, PlacementService, fleet_capabilities
from psrl.sandbox.remote import BoundedTransport, RemoteSandboxBackend

psrl_logger = logging.getLogger(__file__)


class SandboxNodeWireError(RuntimeError):
    """A node failure that could not cross the actor boundary as itself.

    A backend's transport failure is usually an aiohttp error, and those do not pickle, so
    Ray replaces one with an `UnserializableException` that hides what actually broke. The
    node reports this instead, carrying the original type and message.
    """


def _wire_safe(operation: str, node_id: str, error: BaseException) -> BaseException:
    """Return an error that can cross an actor boundary, wrapping one that cannot.

    The test is the wire rather than a list of exception types, because which libraries
    raise an error that cannot cross is not knowable in advance and a list would silently
    fall behind them.

    It round-trips instead of only dumping, because that is what actually fails: an error
    with more constructor arguments than its `args` carries dumps cleanly and then cannot be
    rebuilt. `aiohttp.UnixClientConnectorError` is exactly that, and it is what a Docker
    daemon that is not listening raises.

    A wrapper must not carry the original as its cause or its context either, since both are
    pickled with it, which would make the wrapper fail for the same reason.
    """
    try:
        pickle.loads(pickle.dumps(error))
    except Exception:
        wrapped = SandboxNodeWireError(
            f"Sandbox node {node_id!r} failed to {operation}: {type(error).__name__}: {error}"
        )
        wrapped.__cause__ = None
        wrapped.__context__ = None
        return wrapped
    return error


class SandboxNodeActor:
    """One sandbox node: its manager, its agent, and the daemon they guard.

    Ray constructs this inside the node it is pinned to, so the manager it builds is bound
    to that node's daemon and its capacity coordinator runs in the same process. That is
    why the coordinator is a plain object here rather than a Ray actor: one process owns
    both, and a second hop would only add latency to every admission.

    The methods below are the node-agent protocol, forwarded rather than reimplemented, so
    a node's behaviour has one definition and this class stays a wire adapter.
    """

    def __init__(
        self,
        sandbox_config: Any,
        *,
        node_id: str | None = None,
        labels: Sequence[str] = (),
        capacity_config: Mapping[str, Any] | None = None,
        owner_id: str | None = None,
        liveness_interval_s: float | None = None,
        placement: Any = None,
        capacity_coordinator: SandboxCapacityCoordinator | None = None,
    ) -> None:
        self.node_id = node_id or ray.get_runtime_context().get_node_id()
        self.placement = placement
        # Derived from the sandbox config this node already holds rather than passed
        # separately, so a node can never be built without an admission envelope.
        coordinator = capacity_coordinator
        if coordinator is None:
            coordinator = SandboxCapacityCoordinator(
                capacity_config if capacity_config is not None else resolve_capacity(sandbox_config)
            )
        self.capacity = coordinator
        self.manager = build_sandbox_manager(
            sandbox_config,
            capacity_coordinator=coordinator,
            owner_id=owner_id or f"node-{self.node_id}",
        )
        agent_kwargs: dict[str, Any] = {"node_id": self.node_id, "labels": labels}
        if liveness_interval_s is not None:
            agent_kwargs["liveness_interval_s"] = liveness_interval_s
        self.agent = SandboxNodeAgent(self.manager, **agent_kwargs)
        if placement is not None:
            # The reporter reaches placement from here, so this actor owns it. A node that
            # becomes known without a way to stay known is drained at its TTL.
            self.agent.liveness_reporter = self.report_node

    async def _call(self, operation: str, call: Any) -> Any:
        """Run one node operation and make sure its failure can cross the boundary.

        The failure is raised outside the handler on purpose. Python sets an exception's
        context to whatever was being handled when it was raised, and a context is pickled
        with the exception, so raising inside the handler would re-attach the very error
        this is replacing.
        """
        failure: BaseException | None = None
        try:
            return await call()
        except Exception as error:
            failure = _wire_safe(operation, self.node_id, error)
        raise failure

    async def report_node(self) -> None:
        """Re-announce this node to placement, or refresh it when placement still knows it."""
        if await self.placement.has_node.remote(self.node_id):
            await self.placement.heartbeat.remote(self.node_id)
            return
        await self.register()

    async def register(self) -> None:
        """
        Advertise this node to placement, from the node's own report.
        """
        advertisement = await self.agent.advertise()
        for payload in advertisement["backends"]:
            await self.placement.register.remote(NodeCapabilities.from_advertisement(payload))
        await self.placement.start_sweeper.remote()

    async def advertise(self) -> dict[str, Any]:
        return await self._call("advertise", lambda: self.agent.advertise())

    async def prefetch(self, references: Sequence[str], *, concurrency: int = 2) -> int:
        return await self._call("prefetch", lambda: self.agent.prefetch(references, concurrency=concurrency))

    async def acquire(self, spec_payload: Mapping[str, Any], *, callback: str | None = None) -> dict[str, Any]:
        return await self._call("acquire", lambda: self.agent.acquire(spec_payload, callback=callback))

    async def acquire_group(
        self, spec_payloads: Sequence[Mapping[str, Any]], *, callback: str | None = None
    ) -> list[dict[str, Any]]:
        return await self._call("acquire_group", lambda: self.agent.acquire_group(spec_payloads, callback=callback))

    async def connect(self, backend: str, sandbox_id: str) -> dict[str, Any]:
        return await self._call("connect", lambda: self.agent.connect(backend, sandbox_id))

    async def exec(self, backend, sandbox_id, command, *, cwd=None, env=None, timeout_s=None, silence_timeout_s=None):
        return await self._call(
            "exec",
            lambda: self.agent.exec(
                backend,
                sandbox_id,
                command,
                cwd=cwd,
                env=env,
                timeout_s=timeout_s,
                silence_timeout_s=silence_timeout_s,
            ),
        )

    async def read_bytes(self, backend: str, sandbox_id: str, path: str) -> bytes:
        return await self._call("read_bytes", lambda: self.agent.read_bytes(backend, sandbox_id, path))

    async def write_bytes(self, backend: str, sandbox_id: str, path: str, data: bytes) -> None:
        await self._call("write_bytes", lambda: self.agent.write_bytes(backend, sandbox_id, path, data))

    async def status(self, backend: str, sandbox_id: str) -> str:
        return await self._call("status", lambda: self.agent.status(backend, sandbox_id))

    async def resolve_callback_url(self, backend: str, sandbox_id: str, url: str) -> str:
        return await self._call(
            "resolve_callback_url", lambda: self.agent.resolve_callback_url(backend, sandbox_id, url)
        )

    async def diagnostics(self, backend: str, sandbox_id: str) -> dict[str, Any]:
        return await self._call("diagnostics", lambda: self.agent.diagnostics(backend, sandbox_id))

    async def pause(self, backend: str, sandbox_id: str, mode: str = "hibernate") -> None:
        await self._call("pause", lambda: self.agent.pause(backend, sandbox_id, mode))

    async def resume(self, backend: str, sandbox_id: str) -> None:
        await self._call("resume", lambda: self.agent.resume(backend, sandbox_id))

    async def checkpoint(self, backend: str, sandbox_id: str, kind: str, policy: Mapping[str, Any]) -> dict:
        return await self._call("checkpoint", lambda: self.agent.checkpoint(backend, sandbox_id, kind, policy))

    async def restore(
        self, snapshot: Mapping[str, Any], spec: Mapping[str, Any] | None, *, callback: str | None = None
    ) -> dict:
        return await self._call("restore", lambda: self.agent.restore(snapshot, spec, callback=callback))

    async def release(self, backend: str, sandbox_id: str) -> None:
        await self._call("release", lambda: self.agent.release(backend, sandbox_id))

    async def release_group(self, handles: Sequence[Mapping[str, Any]]) -> None:
        await self._call("release_group", lambda: self.agent.release_group(handles))

    async def busy(self, backend: str, sandbox_id: str) -> bool:
        return await self._call("busy", lambda: self.agent.busy(backend, sandbox_id))

    async def sweep(self) -> dict[str, Any]:
        return await self._call("sweep", lambda: self.agent.sweep())

    async def prune_snapshot_cache(self) -> list[str]:
        return await self._call("prune_snapshot_cache", lambda: self.agent.prune_snapshot_cache())

    async def snapshot(self) -> dict[str, float]:
        return await self._call("snapshot", lambda: self.agent.snapshot())

    async def shutdown(self) -> None:
        """
        Destroy what this node holds and close its backends.
        """
        await self.agent.shutdown()
        if self.capacity is not None:
            await self.capacity.shutdown()


class RayNodeAgentTransport:
    """A `NodeAgentTransport` over Ray actor handles.

    Every call is a Ray RPC, so the payloads are the mappings the node agent already speaks
    and no live object crosses a process boundary.
    """

    def __init__(self, placement: Any, agents: Mapping[str, Any]) -> None:
        self.placement = placement
        self.agents = dict(agents)

    def agent_for(self, node_id: str) -> Any:
        """
        Return one node's actor handle, naming the node when it is unknown.
        """
        handle = self.agents.get(node_id)
        if handle is None:
            raise KeyError(f"Sandbox node {node_id!r} is not part of this plane. Known nodes: {sorted(self.agents)}.")
        return handle

    async def register_node(self, node_id: str) -> None:
        await self.agent_for(node_id).register.remote()

    async def choose(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        # The decision crosses as a mapping, because that is what the transport protocol
        # carries and what a non-Ray caller already speaks.
        decision = await self.placement.choose.remote(dict(request))
        return decision.as_dict()

    async def cancel_reservation(self, reservation_id: str) -> None:
        await self.placement.cancel.remote(reservation_id)

    async def renew_reservation(self, reservation_id: str) -> None:
        await self.placement.renew.remote(reservation_id)

    async def reservation_ttl_s(self) -> float:
        return await self.placement.reservation_ttl_s.remote()

    async def candidate_nodes(self, request: Mapping[str, Any], *, limit: int) -> list[str]:
        return await self.placement.candidates.remote(dict(request), limit=limit)

    async def prefetch(self, node_id: str, references: Sequence[str], *, concurrency: int) -> int:
        return await self.agent_for(node_id).prefetch.remote(list(references), concurrency=concurrency)

    async def release_reservation(self, reservation_id: str) -> None:
        await self.placement.release.remote(reservation_id)

    async def acquire(
        self, node_id: str, spec: Mapping[str, Any], *, callback: str | None = None
    ) -> Mapping[str, Any]:
        return await self.agent_for(node_id).acquire.remote(dict(spec), callback=callback)

    async def acquire_group(
        self, node_id: str, specs: Sequence[Mapping[str, Any]], *, callback: str | None = None
    ) -> list[Mapping[str, Any]]:
        return await self.agent_for(node_id).acquire_group.remote([dict(spec) for spec in specs], callback=callback)

    async def connect(self, node_id: str, backend: str, sandbox_id: str) -> Mapping[str, Any]:
        return await self.agent_for(node_id).connect.remote(backend, sandbox_id)

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
        return await self.agent_for(node_id).exec.remote(
            backend,
            sandbox_id,
            command,
            cwd=cwd,
            env=dict(env) if env is not None else None,
            timeout_s=timeout_s,
            silence_timeout_s=silence_timeout_s,
        )

    async def read_bytes(self, node_id: str, backend: str, sandbox_id: str, path: str) -> bytes:
        return await self.agent_for(node_id).read_bytes.remote(backend, sandbox_id, path)

    async def write_bytes(self, node_id: str, backend: str, sandbox_id: str, path: str, data: bytes) -> None:
        await self.agent_for(node_id).write_bytes.remote(backend, sandbox_id, path, data)

    async def status(self, node_id: str, backend: str, sandbox_id: str) -> str:
        return await self.agent_for(node_id).status.remote(backend, sandbox_id)

    async def resolve_callback_url(self, node_id: str, backend: str, sandbox_id: str, url: str) -> str:
        return await self.agent_for(node_id).resolve_callback_url.remote(backend, sandbox_id, url)

    async def diagnostics(self, node_id: str, backend: str, sandbox_id: str) -> Mapping[str, Any]:
        return await self.agent_for(node_id).diagnostics.remote(backend, sandbox_id)

    async def pause(self, node_id: str, backend: str, sandbox_id: str, mode: str) -> None:
        await self.agent_for(node_id).pause.remote(backend, sandbox_id, mode)

    async def resume(self, node_id: str, backend: str, sandbox_id: str) -> None:
        await self.agent_for(node_id).resume.remote(backend, sandbox_id)

    async def checkpoint(
        self, node_id: str, backend: str, sandbox_id: str, kind: str, policy: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        return await self.agent_for(node_id).checkpoint.remote(backend, sandbox_id, kind, dict(policy))

    async def restore(
        self,
        node_id: str,
        snapshot: Mapping[str, Any],
        spec: Mapping[str, Any] | None,
        *,
        callback: str | None = None,
    ) -> Mapping[str, Any]:
        return await self.agent_for(node_id).restore.remote(
            dict(snapshot), dict(spec) if spec is not None else None, callback=callback
        )

    async def release(self, node_id: str, backend: str, sandbox_id: str) -> None:
        await self.agent_for(node_id).release.remote(backend, sandbox_id)


@dataclass(frozen=True)
class SandboxPlaneHandle:
    """What one worker needs from the plane, and nothing that would let it stop the plane.

    A worker builds its own caller-side backend from this, which is what makes a sandbox on
    another node look local to everything above it. It deliberately does not carry
    `SandboxPlane`, because shutting the plane down is the trainer's job.
    """

    placement: Any
    agents: Mapping[str, Any]
    node_ids: tuple[str, ...]
    capabilities: SandboxCapabilities
    backend_name: str
    rpc_timeout_s: float
    required_label: str | None = None
    callback_target: str | None = None

    def transport(self) -> Any:
        """
        Return a bounded transport over the plane's actors.
        """
        return BoundedTransport(RayNodeAgentTransport(self.placement, self.agents), timeout_s=self.rpc_timeout_s)

    def remote_backend(self, *, owner_id: str) -> RemoteSandboxBackend:
        """Build the caller-side backend that places sandboxes through this plane.

        The name is the backend the fleet runs rather than a caller-side label, because it is
        also the filter placement matches against what each node advertises.
        """
        return RemoteSandboxBackend(
            self.transport(),
            name=self.backend_name,
            owner_id=owner_id,
            required_label=self.required_label,
            capabilities=self.capabilities,
            callback_target=self.callback_target,
        )


class SandboxPlane:
    """The actors one job uses to place sandboxes, and the handles a worker needs.

    Created by the trainer, because the topology is a property of the deployment rather
    than of one worker: the placement service is shared by every worker, and one node agent
    owns each node's daemon.
    """

    def __init__(
        self,
        *,
        placement: Any,
        agents: Mapping[str, Any],
        node_ids: Sequence[str],
    ) -> None:
        self.placement = placement
        self.agents = dict(agents)
        self.node_ids = list(node_ids)
        self._capabilities: SandboxCapabilities | None = None

    def transport(self) -> RayNodeAgentTransport:
        """
        Return a transport over this plane's actors, for one worker to use.
        """
        return RayNodeAgentTransport(self.placement, self.agents)

    def handle(
        self,
        *,
        backend_name: str,
        rpc_timeout_s: float,
        required_label: str | None = None,
        callback_target: str | None = None,
    ) -> SandboxPlaneHandle:
        """Return what one worker needs, once the plane's capabilities are known."""
        return SandboxPlaneHandle(
            placement=self.placement,
            agents=dict(self.agents),
            node_ids=tuple(self.node_ids),
            capabilities=self.capabilities,
            backend_name=backend_name,
            rpc_timeout_s=rpc_timeout_s,
            required_label=required_label,
            callback_target=callback_target,
        )

    def register_nodes(self) -> dict[str, dict[str, Any]]:
        """Register every node and return each one's advertisement.

        Registering is a handshake, not a fire-and-forget: it also starts each node's
        liveness reporting and the placement sweeper, and it produces the advertisement the
        fleet's declared capabilities are derived from.

        It blocks rather than awaiting, because this object lives in the driver and the
        trainer that owns it is synchronous. `ray.get` is the idiom for that boundary.
        """
        advertisements: dict[str, dict[str, Any]] = {}
        for node_id in self.node_ids:
            ray.get(self.agents[node_id].register.remote())
            advertisements[node_id] = ray.get(self.agents[node_id].advertise.remote())
        self._capabilities = fleet_capabilities(list(advertisements.values()))
        return advertisements

    def placement_snapshot(self) -> Any:
        """
        Return the placement service's point-in-time view, for the trainer's metric hook.
        """
        return ray.get(self.placement.snapshot.remote())

    @property
    def capabilities(self) -> SandboxCapabilities:
        """
        Return the fleet's capabilities, which a caller declares and placement refines.
        """
        if self._capabilities is None:
            raise RuntimeError("The sandbox plane has no capabilities until its nodes are registered.")
        return self._capabilities

    def shutdown(self) -> None:
        """
        Stop every node and the placement service.
        """
        for node_id in self.node_ids:
            agent = self.agents.get(node_id)
            if agent is None:
                continue
            try:
                ray.get(agent.shutdown.remote())
            except Exception:
                # A node that cannot be reached is already gone, and the reclaimer owns
                # whatever it left behind.
                psrl_logger.warning("Sandbox node %s did not shut down cleanly.", node_id, exc_info=True)
        try:
            ray.get(self.placement.shutdown.remote())
        except Exception:
            # Stopping a placement that already died is not a failure worth raising: nothing
            # is left to stop, and the job is ending.
            psrl_logger.warning("The sandbox placement service did not shut down cleanly.", exc_info=True)


def build_sandbox_plane(
    sandbox_config: Any,
    node_ids: Sequence[str],
    *,
    node_ttl_s: float = 120.0,
    reservation_ttl_s: float = 60.0,
    sweep_interval_s: float = 10.0,
    heartbeat_interval_s: float = 30.0,
    max_concurrency_per_node: int = 8,
    owner_id: str | None = None,
    labels: Sequence[str] = (),
) -> SandboxPlane:
    """Create the placement service and one node agent per node, pinned to that node.

    A node agent is pinned with hard affinity, because a manager admitted against one node's
    envelope must run on that node. It takes no CPU, because it must not perturb GPU
    scheduling and it is idle except while a request is in flight.
    """
    if not node_ids:
        raise ValueError("A sandbox plane needs at least one node.")
    # The placement service holds a cache and rebuilds from the nodes, so losing it is
    # recoverable. A node whose report reaches the empty replacement announces itself again.
    placement = (
        ray.remote(PlacementService)
        .options(num_cpus=0, max_restarts=2)
        .remote(
            node_ttl_s=node_ttl_s,
            reservation_ttl_s=reservation_ttl_s,
            sweep_interval_s=sweep_interval_s,
        )
    )
    agents: dict[str, Any] = {}
    for node_id in node_ids:
        agents[node_id] = (
            ray.remote(SandboxNodeActor)
            .options(
                num_cpus=0,
                max_concurrency=max_concurrency_per_node,
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
            )
            .remote(
                sandbox_config,
                node_id=node_id,
                labels=list(labels),
                liveness_interval_s=heartbeat_interval_s,
                owner_id=owner_id,
                placement=placement,
            )
        )
    return SandboxPlane(placement=placement, agents=agents, node_ids=list(node_ids))
