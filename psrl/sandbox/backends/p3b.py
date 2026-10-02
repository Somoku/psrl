"""The p3b sandbox service as a PSRL backend.

PSRL has two ways to run a sandbox, and this is the one that does not run it
in-process. The in-process backends (`docker`, `agentenv`, `opensandbox`) each
drive a provider from the worker; this one hands the whole decision to the p3b
service and asks it for a sandbox.

What moves, and why it is worth a hop: placement, the cross-backend quota
ledger, and reclamation all become the service's, so every worker in a fleet
draws on one ledger rather than on its own node envelope. A worker that places
locally cannot know what the other workers are holding.

What does not move: commands. `create` returns the sandbox's own agent
endpoint, and `exec` goes straight there. An episode issues one create and
dozens of commands, so routing commands through the service would add a hop per
command without adding a decision.

The two type systems are deliberately isomorphic — both were generated against
`psrl/sandbox/api/v1/sandbox.proto` — so the conversions here are field renames
rather than semantic translation. Where a PSRL spec carries something the
service has no field for, this backend refuses rather than dropping it: a
sandbox that silently lost its mount or its egress policy would be a worse
failure than one that was never created.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping

from psrl.pysandbox import Sandbox as ServiceSandbox
from psrl.pysandbox import SandboxClient
from psrl.pysandbox import SandboxSpec as ServiceSpec
from psrl.pysandbox import (
    Resources as ServiceResources,
)
from psrl.pysandbox import (
    Source as ServiceSource,
)
from psrl.sandbox.core import (
    ExecResult,
    PauseMode,
    ResourceUsage,
    ResumeLevel,
    SandboxBackend,
    SandboxCapabilities,
    SandboxFeature,
    SandboxRef,
    SandboxSession,
    SandboxSourceKind,
    SandboxSpec,
    SandboxStatus,
    SnapshotKind,
    SnapshotRef,
)

psrl_logger = logging.getLogger(__file__)

# The service's default socket. A deployment that runs one service per node uses
# this; a shared control plane is reached by host:port instead.
DEFAULT_ENDPOINT = "unix:///run/sandboxd.sock"


class P3bSession(SandboxSession):
    """One sandbox held by the service.

    Commands go to the sandbox's agent endpoint, which `create` already
    resolved. Lifecycle goes back to the service.
    """

    def __init__(self, backend: P3bBackend, sandbox: ServiceSandbox, spec: SandboxSpec | None) -> None:
        self._backend = backend
        self._sandbox = sandbox
        self._spec = spec
        self._command_count = 0
        self._busy = False
        self._last_activity_at: float | None = None
        self._terminated = False
        self._exit_reason = SandboxStatus.UNKNOWN

    @property
    def ref(self) -> SandboxRef:
        return SandboxRef(backend=self._backend.name, sandbox_id=self._sandbox.sandbox_id)

    @property
    def capabilities(self) -> SandboxCapabilities:
        return _capabilities_from(self._sandbox.capabilities)

    @property
    def spec(self) -> SandboxSpec | None:
        return self._spec

    @property
    def command_count(self) -> int:
        return self._command_count

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def last_activity_at(self) -> float | None:
        return self._last_activity_at

    @property
    def exit_reason(self) -> SandboxStatus:
        return self._exit_reason

    @property
    def node_id(self) -> str:
        """Return the node the service placed this sandbox on.

        Not part of the session contract, but a rollout that reports where its
        sandboxes landed needs it, and only the service knows.
        """
        return self._sandbox.node_id

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
        silence_timeout_s: float | None = None,
    ) -> ExecResult:
        """Run one command in the sandbox.

        The SDK's exceptions are already PSRL's: both modules raise the same
        `SandboxCommandTimeout`, `SandboxOomError`, and `SandboxSessionLost`
        types, so a caller's attribution logic does not change with the backend.
        """
        self._busy = True
        try:
            result = await self._sandbox.exec(
                command,
                cwd=cwd,
                env=env,
                timeout_s=timeout_s,
                silence_timeout_s=silence_timeout_s,
            )
        finally:
            self._busy = False
            self._command_count += 1
            self._last_activity_at = asyncio.get_running_loop().time()
        return ExecResult(
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            truncated=result.truncated,
        )

    async def read_bytes(self, path: str) -> bytes:
        return await self._sandbox.read_bytes(path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        await self._sandbox.write_bytes(path, data)

    async def status(self) -> SandboxStatus:
        if self._terminated:
            return SandboxStatus.TERMINATED
        return SandboxStatus(await self._sandbox.status())

    async def terminate(self) -> None:
        """Release the sandbox back to the service.

        Idempotent, because a lease's teardown and a reaper's sweep can both
        reach a session, and the second one must not raise.
        """
        if self._terminated:
            return
        self._terminated = True
        self._exit_reason = SandboxStatus.RELEASED
        await self._sandbox.release()

    async def stats(self) -> ResourceUsage:
        """Report resource use.

        The service's reports describe a node and a class rather than one
        sandbox, so per-sandbox use is unknown here. Reported as unknown rather
        than zero: a metric that reads zero would be taken as a measurement.
        """
        return ResourceUsage()

    async def snapshot(self, kind: SnapshotKind = SnapshotKind.FILESYSTEM) -> SnapshotRef:
        """Capture the sandbox, through whichever backend the service placed it on."""
        ref = await self._sandbox.snapshot(kind)
        return SnapshotRef(
            backend=self._backend.name,
            snapshot_id=ref.snapshot_id,
            kind=SnapshotKind(ref.kind),
            resume_level=ResumeLevel(ref.resume_level) if ref.resume_level else None,
        )

    async def pause(self, mode: PauseMode = PauseMode.HIBERNATE) -> None:
        await self._sandbox.pause(mode)

    async def resume(self) -> None:
        await self._sandbox.resume()

    def resolve_callback_url(self, url: str) -> str:
        """Return the URL unchanged.

        A worker-loopback rewrite is a property of where the sandbox runs
        relative to the worker, and the service placed it — possibly on another
        machine, where no alias this worker could name would resolve. A
        deployment that needs the worker reachable from the sandbox gives the
        worker a routable address rather than a loopback one.
        """
        return url


class P3bBackend(SandboxBackend):
    """The p3b service as one PSRL backend.

    This backend holds no envelope of its own: `uses_node_capacity` is False
    because the service's ledger already bounds what may be admitted across the
    whole fleet, and charging a worker's node envelope as well would refuse a
    sandbox twice for the same memory.
    """

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        *,
        name: str = "p3b",
        resource_class: str = "rollout",
        control_timeout_s: float = 60.0,
    ) -> None:
        self._name = name
        self._endpoint = endpoint
        self._resource_class = resource_class
        self._client = SandboxClient(endpoint, control_timeout_s=control_timeout_s)
        self._owner_id = ""
        self._closed = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def uses_node_capacity(self) -> bool:
        """Return False: the service's ledger is the only admission that applies."""
        return False

    @property
    def capabilities(self) -> SandboxCapabilities:
        """Declare the union of what the service's backends can serve.

        A union rather than an intersection, because the service routes a spec
        to a backend that can meet it: a request for a full-state resume is
        servable by the fleet even though its Docker backend alone could not.
        The service refuses at admission anything no backend can serve, and that
        refusal names what was missing — so a capability claimed here and absent
        everywhere fails with a usable message rather than silently degrading.
        """
        return SandboxCapabilities(
            features=frozenset(
                {
                    SandboxFeature.FREEZE,
                    SandboxFeature.HIBERNATE,
                    SandboxFeature.FILESYSTEM_SNAPSHOT,
                    SandboxFeature.FULL_STATE_SNAPSHOT,
                    SandboxFeature.RESTORE,
                    SandboxFeature.NATIVE_FORK,
                    SandboxFeature.RESUME_ANYWHERE,
                    SandboxFeature.WARM_POOL,
                    SandboxFeature.IMAGE_ON_DEMAND,
                }
            ),
            resume_level=ResumeLevel.FULL_STATE,
        )

    def adopt_owner_id(self, owner_id: str) -> None:
        """Record the owner for this worker's sandboxes.

        The service labels and reclaims by its own owner, so this is carried as
        spec metadata for correlation rather than used for cleanup here.
        """
        if not self._owner_id:
            self._owner_id = owner_id

    async def create(self, spec: SandboxSpec) -> SandboxSession:
        """Ask the service for one sandbox."""
        self._require_open()
        sandbox = await self._client.create(self._to_service_spec(spec))
        return P3bSession(self, sandbox, spec)

    async def connect(self, sandbox_id: str) -> SandboxSession:
        """Reattach to a sandbox this worker did not create.

        Not offered: the service owns the handle, and a reattach needs the agent
        endpoint that only a create reply carries. A worker that lost its
        session asks for a new sandbox instead.
        """
        raise NotImplementedError(
            "The p3b service does not hand out a session for a sandbox a worker did not create; "
            "the agent endpoint only travels with a create reply."
        )

    async def restore(self, snapshot: SnapshotRef, spec: SandboxSpec | None = None) -> SandboxSession:
        """Start a sandbox from a snapshot the service holds."""
        self._require_open()
        service_spec = self._to_service_spec(spec) if spec is not None else None
        sandbox = await self._client.restore(_to_service_snapshot(snapshot), service_spec)
        return P3bSession(self, sandbox, spec)

    async def prepare(self, spec: SandboxSpec) -> None:
        """Warm whatever the service can warm for this spec.

        A no-op rather than an error: image pulls, template builds, and warm
        pools are the service's, and it does them when a create needs them. A
        caller that prepares is asking to pay the cost early, and there is no
        early to pay it at from here.
        """
        return None

    async def shutdown(self) -> None:
        """Close the connection to the service.

        The sandboxes outlive it: the service holds the lifetime and reclaims by
        its own sweep, so a worker exiting does not strand them.
        """
        if self._closed:
            return
        self._closed = True
        await self._client.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"Sandbox backend {self._name!r} is closed.")

    def _to_service_spec(self, spec: SandboxSpec) -> ServiceSpec:
        """Convert a PSRL spec into the service's.

        Anything the service has no field for is refused here rather than
        dropped. A sandbox that quietly lost its bind mount, its egress
        allowlist, or its injected credentials would run and produce a wrong
        result; one that was refused costs a configuration fix.
        """
        if spec.mounts:
            raise ValueError(
                "A p3b sandbox runs on a node the service chose, which may not be this worker's, "
                "so a host bind mount cannot be honoured. Use a volume the service can attach, "
                "or run the docker backend in-process."
            )
        if spec.volumes:
            raise ValueError("The p3b backend does not yet forward provider volumes.")
        if spec.egress is not None:
            raise ValueError("The p3b backend does not yet forward an egress policy.")
        if spec.credentials or spec.credential_bindings:
            raise ValueError("The p3b backend does not yet forward injected credentials.")
        if spec.assigned_gpus:
            raise ValueError(
                "A p3b sandbox is placed by the service, which assigns its own devices, "
                "so a worker cannot pin host GPU indices."
            )

        metadata = dict(spec.metadata)
        if self._owner_id:
            metadata.setdefault("psrl_owner_id", self._owner_id)

        return ServiceSpec(
            source=ServiceSource(
                kind=_source_kind(spec.source.kind),
                reference=spec.source.reference,
            ),
            resources=ServiceResources(
                cpu_count=spec.resources.cpu_count,
                memory_mb=spec.resources.memory_mb,
                disk_mb=spec.resources.disk_mb,
                gpu_count=spec.resources.gpu_count,
            ),
            resource_class=spec.resource_class or self._resource_class,
            workflow_id=spec.workflow_id,
            idempotency_key=spec.idempotency_key,
            env=dict(spec.env),
            metadata=metadata,
        )


def _source_kind(kind: SandboxSourceKind) -> str:
    """Return the service's name for a source kind."""
    return str(kind.value if hasattr(kind, "value") else kind)


def _to_service_snapshot(snapshot: SnapshotRef):
    """Convert a PSRL snapshot reference into the service's."""
    from psrl.pysandbox import SnapshotRef as ServiceSnapshotRef

    return ServiceSnapshotRef(
        backend=snapshot.backend,
        snapshot_id=snapshot.snapshot_id,
        kind=snapshot.kind,
    )


def _capabilities_from(declared) -> SandboxCapabilities:
    """Convert the service's capability report into PSRL's.

    The feature vocabularies are the same strings in both modules, so a feature
    the service names and PSRL does not is dropped rather than failing: a newer
    service must not break an older worker over a capability it never asked for.
    """
    features = set()
    for feature in getattr(declared, "features", ()) or ():
        try:
            features.add(SandboxFeature(str(feature)))
        except ValueError:
            psrl_logger.debug(f"The p3b service declared feature {feature!r}, which this worker does not know.")
    level = getattr(declared, "resume_level", None)
    return SandboxCapabilities(
        features=frozenset(features),
        resume_level=ResumeLevel(str(level)) if level else None,
    )
