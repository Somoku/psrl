"""The p3b sandbox service as a PSRL backend.

PSRL has two sandbox paths. The in-process path (`docker`, `agentenv`,
`opensandbox`) runs sandbox logic inside the worker and charges the worker's
node envelope. This path does neither: it delegates every placement and
lifetime decision to the p3b service, and PSRL becomes a pure client.

The separation is explicit by design: p3b is an independently deployable
service that PSRL calls rather than a library it hosts. Nothing in this file
may import from other `psrl.sandbox` modules — it imports only from the
standalone `sandboxd` SDK and from `psrl.sandbox.core`, which holds the
abstract interfaces. The rest of `psrl.sandbox` is the in-process path and
must not bleed through here.

What this backend is responsible for:
- Translating between PSRL's SandboxSpec and the service's, refusing anything
  the service cannot represent rather than silently dropping it
- Wrapping the service's sandbox handle in a SandboxSession so the rest of
  PSRL never knows which path it is on
- Reporting fleet and quota state through the service's own reports, so PSRL's
  metric hooks see a consistent view of a fleet the service manages

What this backend is not responsible for:
- Placement, admission, or capacity — the service owns its ledger
- Reclamation — the service's sweep handles it; a worker exiting does not
  strand sandboxes
- Node capacity charging — uses_node_capacity is False; the service's ledger
  is the only admission that applies
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any

from sandboxd import SandboxClient
from sandboxd import SandboxSpec as ServiceSpec
from sandboxd import (
    Resources as ServiceResources,
    SnapshotRef as ServiceSnapshotRef,
    Source as ServiceSource,
)
from sandboxd import Sandbox as ServiceSandbox
from sandboxd.errors import (
    SandboxCapabilityError,
    SandboxCapacityTimeout,
    SandboxError,
)

from psrl.sandbox.core import (
    ExecResult,
    PauseMode,
    ResourceUsage,
    ResumeLevel,
    SandboxBackend,
    SandboxCapabilities,
    SandboxExitReason,
    SandboxFeature,
    SandboxRef,
    SandboxSession,
    SandboxSourceKind,
    SandboxSpec,
    SandboxStatus,
    SnapshotKind,
    SnapshotRef,
)

psrl_logger = logging.getLogger(__name__)

# The service's default socket. A service per node uses this; a shared control
# plane is reached by host:port.
DEFAULT_ENDPOINT = "unix:///run/sandboxd.sock"


class P3bSession(SandboxSession):
    """One sandbox held by the p3b service.

    Commands go to the sandbox's agent endpoint, which `create` already
    resolved. Lifecycle goes back to the service. This class holds no
    admission state of its own — the service is the source of truth.
    """

    def __init__(
        self,
        backend: P3bBackend,
        sandbox: ServiceSandbox,
        spec: SandboxSpec | None,
    ) -> None:
        self._backend = backend
        self._sandbox = sandbox
        self._spec = spec
        self._command_count = 0
        self._busy = False
        self._last_activity_at: float | None = None
        self._terminated = False
        self._exit_reason = SandboxExitReason.UNKNOWN

    # -- SandboxSession identity -----------------------------------------------

    @property
    def ref(self) -> SandboxRef:
        return SandboxRef(backend=self._backend.name, sandbox_id=self._sandbox.sandbox_id)

    @property
    def capabilities(self) -> SandboxCapabilities:
        # The backend declares the fleet-level capabilities; per-sandbox
        # capabilities are not reported by the service yet.
        return self._backend.capabilities

    @property
    def spec(self) -> SandboxSpec | None:
        return self._spec

    # -- SandboxSession activity -----------------------------------------------

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
    def exit_reason(self) -> SandboxExitReason:
        return self._exit_reason

    # -- Node identity (not part of the SandboxSession contract) ---------------

    @property
    def node_id(self) -> str:
        """Return the node the service placed this sandbox on."""
        return self._sandbox.node_id

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
        """Run one command straight to the sandbox agent endpoint.

        The SDK raises the same SandboxCommandTimeout, SandboxOomError, and
        SandboxSessionLost types that PSRL already handles — no translation.
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

    # -- Control plane: back to the service ------------------------------------

    async def status(self) -> SandboxStatus:
        if self._terminated:
            return SandboxStatus.TERMINATED
        raw = await self._sandbox.status()
        return SandboxStatus(str(raw))

    async def terminate(self) -> None:
        """Release the sandbox back to the service.

        Idempotent: a lease teardown and the service's reclamation sweep can
        both reach a session, and the second must not raise.
        """
        if self._terminated:
            return
        self._terminated = True
        self._exit_reason = SandboxExitReason.RELEASED
        await self._sandbox.release()

    async def stats(self) -> ResourceUsage:
        """Return unknown usage.

        The service's fleet reports describe a node and a class, not one
        sandbox. Returning None fields rather than zeros: a metric that reads
        zero is indistinguishable from a sandbox that used nothing.
        """
        return ResourceUsage()

    async def snapshot(self, kind: SnapshotKind = SnapshotKind.FILESYSTEM) -> SnapshotRef:
        ref = await self._sandbox.snapshot(kind)
        return SnapshotRef(
            backend=self._backend.name,
            snapshot_id=ref.snapshot_id,
            kind=SnapshotKind(str(ref.kind)),
            resume_level=ResumeLevel(str(ref.resume_level)) if ref.resume_level else None,
        )

    async def pause(self, mode: PauseMode = PauseMode.HIBERNATE) -> None:
        await self._sandbox.pause(mode)

    async def resume(self) -> None:
        await self._sandbox.resume()

    def resolve_callback_url(self, url: str) -> str:
        """Return the URL unchanged.

        A worker-loopback rewrite is a property of where the sandbox runs
        relative to the worker. The service places the sandbox — possibly on
        another node — so no alias this worker names is guaranteed to resolve
        there. A deployment that needs the worker reachable from the sandbox
        gives the worker a routable address, not a loopback.
        """
        return url


class P3bBackend(SandboxBackend):
    """The p3b service as a PSRL backend.

    Selecting this backend switches PSRL from in-process sandbox management
    to the p3b service. The service holds the quota ledger, placement, and
    reclamation; PSRL is a client.

    Configuration in psrl_rollout.yaml:

        sandbox:
          default_backend: p3b
          backends:
            p3b:
              _target_: psrl.sandbox.backends.P3bBackend
              endpoint: unix:///run/sandboxd.sock
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
        """Return False: the service's ledger is the only admission that applies.

        Charging the worker's node envelope as well would refuse a sandbox
        twice for the same memory.
        """
        return False

    @property
    def capabilities(self) -> SandboxCapabilities:
        """Declare the union of what a fleet running all four backends can serve.

        A union rather than an intersection, because the service routes a spec
        to whichever backend can meet it. A spec no backend can serve is
        refused at admission, and that refusal names what was missing — so a
        capability claimed here and absent from every deployed backend fails
        with a usable message rather than silently degrading.

        Operators running a Docker-only fleet should not reduce this set here:
        the service's own admission is the gate. Reducing this set would only
        cause PSRL to refuse before asking the service.
        """
        return SandboxCapabilities(
            features=frozenset({
                SandboxFeature.FREEZE,
                SandboxFeature.HIBERNATE,
                SandboxFeature.FILESYSTEM_SNAPSHOT,
                SandboxFeature.FULL_STATE_SNAPSHOT,
                SandboxFeature.RESTORE,
                SandboxFeature.NATIVE_FORK,
                SandboxFeature.RESUME_ANYWHERE,
                SandboxFeature.WARM_POOL,
                SandboxFeature.IMAGE_ON_DEMAND,
            }),
            resume_level=ResumeLevel.FULL_STATE,
        )

    def adopt_owner_id(self, owner_id: str) -> None:
        """Record the owner for this worker's sandboxes.

        Carried as spec metadata for correlation. The service reclaims by its
        own owner labels, not by anything PSRL writes here.
        """
        if not self._owner_id:
            self._owner_id = owner_id

    async def create(self, spec: SandboxSpec) -> SandboxSession:
        self._require_open()
        sandbox = await self._client.create(_to_service_spec(spec, self._owner_id, self._resource_class))
        return P3bSession(self, sandbox, spec)

    async def connect(self, sandbox_id: str) -> SandboxSession:
        """Not supported: the agent endpoint only travels with a create reply.

        A worker that lost its session asks for a new sandbox. The service's
        reclamation sweep owns the lifetime of orphaned sandboxes.
        """
        raise NotImplementedError(
            "P3bBackend does not support connect(): the agent endpoint is only "
            "available in the create reply. Request a new sandbox instead."
        )

    async def restore(self, snapshot: SnapshotRef, spec: SandboxSpec | None = None) -> SandboxSession:
        self._require_open()
        service_spec = _to_service_spec(spec, self._owner_id, self._resource_class) if spec is not None else None
        sandbox = await self._client.restore(
            ServiceSnapshotRef(
                backend=snapshot.backend,
                snapshot_id=snapshot.snapshot_id,
                kind=str(snapshot.kind.value if hasattr(snapshot.kind, "value") else snapshot.kind),
            ),
            service_spec,
        )
        return P3bSession(self, sandbox, spec)

    async def prepare(self, spec: SandboxSpec) -> None:
        """No-op: the service manages its own warm pools and image pulls.

        A caller that prepares is asking to pay the cost early. There is no
        early to pay from here — the service does it on demand.
        """
        return None

    async def shutdown(self) -> None:
        """Close the connection to the service.

        Sandboxes outlive the connection: the service holds the lifetime and
        reclaims by its own sweep, so a worker exiting does not strand them.
        """
        if self._closed:
            return
        self._closed = True
        await self._client.close()

    # -- Metric passthroughs ---------------------------------------------------

    async def fleet_report(self) -> dict[str, Any]:
        """Return the service's fleet view for PSRL's metric hooks.

        Not part of the SandboxBackend contract, but callable by any code that
        knows it is talking to a P3bBackend and wants to report fleet state.
        Delegates directly to the SDK's FleetReport.as_metrics() for a flat
        metrics dict, and includes per-node state for debugging.
        """
        self._require_open()
        report = await self._client.fleet()
        result = report.as_metrics()
        result["nodes"] = [
            {
                "node_id": n.node_id,
                "backend": n.backend,
                "live_sandboxes": n.live_sandboxes,
                "cpu_used_pct": n.cpu_used_pct,
                "mem_used_pct": n.mem_used_pct,
                "draining": n.draining,
            }
            for n in (report.nodes or [])
        ]
        return result

    async def quota_report(self) -> dict[str, Any]:
        """Return the service's quota view for PSRL's metric hooks.

        Delegates to QuotaReport.as_metrics() for a flat metrics dict.
        """
        self._require_open()
        report = await self._client.quota()
        return report.as_metrics()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"Sandbox backend {self._name!r} is closed.")


# -- Module-level helpers (no PSRL sandbox state, pure conversion) -------------

def _to_service_spec(
    spec: SandboxSpec,
    owner_id: str,
    default_resource_class: str,
) -> ServiceSpec:
    """Convert a PSRL SandboxSpec into the service's.

    Anything the service has no field for is refused here rather than dropped.
    A sandbox that quietly lost a bind mount, an egress policy, or injected
    credentials would run and produce a wrong result; one that was refused
    costs a configuration fix.
    """
    if spec.mounts:
        raise ValueError(
            "A p3b sandbox is placed by the service, which may land on a node "
            "other than this worker's. A host bind mount cannot be honoured. "
            "Use a provider volume that the service can attach, or use the "
            "docker backend to run the sandbox on this worker's own node."
        )
    if spec.volumes:
        raise ValueError(
            "P3bBackend does not forward provider volumes to the service. "
            "Declare the volume in the service configuration instead."
        )
    if spec.egress is not None:
        raise ValueError(
            "P3bBackend does not forward an egress policy to the service. "
            "Declare the policy in the service configuration instead."
        )
    if spec.credentials or spec.credential_bindings:
        raise ValueError(
            "P3bBackend does not forward injected credentials to the service. "
            "Declare the credentials in the service configuration instead."
        )
    if spec.assigned_gpus:
        raise ValueError(
            "A p3b sandbox is placed by the service, which assigns its own "
            "GPU indices on the node it chose. A worker cannot pin host GPU "
            "indices for a remotely placed sandbox."
        )

    metadata = dict(spec.metadata)
    if owner_id:
        metadata.setdefault("psrl_owner_id", owner_id)

    return ServiceSpec(
        source=ServiceSource(
            kind=_source_kind_str(spec.source.kind),
            reference=spec.source.reference,
        ),
        resources=ServiceResources(
            cpu_count=spec.resources.cpu_count,
            memory_mb=spec.resources.memory_mb,
            disk_mb=spec.resources.disk_mb,
            gpu_count=spec.resources.gpu_count,
        ),
        resource_class=spec.resource_class or default_resource_class,
        workflow_id=spec.workflow_id,
        idempotency_key=spec.idempotency_key,
        env=dict(spec.env),
        metadata=metadata,
    )


def _source_kind_str(kind: SandboxSourceKind) -> str:
    return str(kind.value if hasattr(kind, "value") else kind)
