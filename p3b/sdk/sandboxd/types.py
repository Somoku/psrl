"""The portable types a caller states a request in.

These mirror `api/v1/sandbox.proto`, which is the single source
while the Python and Go control planes coexist. A field here that the proto does
not carry would be dropped silently at the boundary, so the drift test in
`tests/sandbox/test_proto_contract.py` checks the enums against the file.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any


class Feature(str, Enum):
    """Optional semantics a backend may implement.

    A caller states what it needs and admission refuses a backend that cannot
    provide it, rather than serving something weaker: a workspace restore must
    not be readable as proof that a live process survived a move.
    """

    FREEZE = "freeze"
    HIBERNATE = "hibernate"
    FILESYSTEM_SNAPSHOT = "filesystem_snapshot"
    FULL_STATE_SNAPSHOT = "full_state_snapshot"
    RESTORE = "restore"
    NATIVE_FORK = "native_fork"
    HOST_MOUNT = "host_mount"
    RESUME_ANYWHERE = "resume_anywhere"
    WARM_POOL = "warm_pool"
    IMAGE_ON_DEMAND = "image_on_demand"
    IMAGE_BLOCK_DELIVERY = "image_block_delivery"
    TEMPLATE_BUILD = "template_build"
    VOLUME = "volume"
    EGRESS_POLICY = "egress_policy"
    CREDENTIAL_INJECTION = "credential_injection"
    ISOLATION_RUNTIME = "isolation_runtime"


class ResumeLevel(str, Enum):
    """What survives a resume elsewhere.

    Ordered, so a requirement is one comparison. A filesystem resume restores the
    workspace and starts every process fresh; a full-state resume lets a harness
    continue instead of restart.
    """

    FILESYSTEM = "filesystem"
    FULL_STATE = "full_state"

    @property
    def rank(self) -> int:
        return {"filesystem": 1, "full_state": 2}[self.value]

    def satisfies(self, required: ResumeLevel) -> bool:
        """Return whether this level is at least as strong as a requirement."""
        return self.rank >= required.rank


class ExecMode(str, Enum):
    """How a backend runs commands.

    A harness that keeps shell state across turns needs a persistent shell. A
    grader that runs a single command is cheaper without one.
    """

    PERSISTENT = "persistent"
    ONE_SHOT = "one_shot"


class PauseMode(str, Enum):
    """Pause semantics. A freeze keeps the sandbox resident; a hibernation
    releases its compute."""

    FREEZE = "freeze"
    HIBERNATE = "hibernate"


class SnapshotKind(str, Enum):
    """What a capture includes."""

    FILESYSTEM = "filesystem"
    FULL_STATE = "full_state"


class SandboxStatus(str, Enum):
    """Backend-neutral sandbox state."""

    RUNNING = "running"
    PAUSED = "paused"
    EXITED = "exited"
    TERMINATED = "terminated"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Source:
    """Where a sandbox's filesystem comes from."""

    kind: str
    reference: str

    @classmethod
    def image(cls, reference: str) -> Source:
        """Create an image-backed source."""
        return cls("image", reference)

    @classmethod
    def template(cls, reference: str) -> Source:
        """Create a template-backed source."""
        return cls("template", reference)

    def __post_init__(self) -> None:
        if not self.reference.strip():
            raise ValueError("A sandbox source needs a reference.")
        if self.kind not in ("image", "template"):
            raise ValueError(f"A sandbox source is an image or a template, not {self.kind!r}.")


@dataclass(frozen=True)
class Resources:
    """A portable resource request.

    None means the caller states no requirement in that dimension, which is not
    the same as requesting nothing: a dimension nobody named is not compared
    against a limit.
    """

    cpu_count: float | None = None
    memory_mb: int | None = None
    disk_mb: int | None = None
    gpu_count: int | None = None

    def __post_init__(self) -> None:
        for name in ("cpu_count", "memory_mb", "disk_mb"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"Sandbox {name} must be greater than zero when stated.")
        if self.gpu_count is not None and self.gpu_count < 0:
            raise ValueError("Sandbox gpu_count must not be negative.")


@dataclass(frozen=True)
class SandboxSpec:
    """The portable creation request.

    `backend_options` carries tuning only one backend understands, keyed by
    backend name. Two rules keep it from eroding the portable contract: it never
    takes part in routing, so two specs differing only there are still
    comparable, and it is outside the spec's identity, so one environment tuned
    two ways is not two idempotent requests.
    """

    source: Source
    resources: Resources = field(default_factory=Resources)

    resource_class: str = "default"
    workflow_id: str | None = None
    idempotency_key: str | None = None

    env: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, str] = field(default_factory=dict)
    workdir: str | None = None
    exec_mode: ExecMode | None = None

    required_features: frozenset[Feature] = frozenset()
    required_resume_level: ResumeLevel | None = None
    backend: str | None = None

    backend_options: Mapping[str, Mapping[str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.resource_class.strip():
            raise ValueError("A sandbox resource_class cannot be empty; it names the quota share that pays.")
        if self.required_resume_level is not None and Feature.RESUME_ANYWHERE not in self.required_features:
            raise ValueError(
                "A required_resume_level needs RESUME_ANYWHERE in required_features, or the level would "
                "describe a resume the caller never asked for."
            )

    def with_options(self, backend: str, **options: str) -> SandboxSpec:
        """Return this spec with one backend's private tuning attached."""
        merged = {**self.backend_options, backend: {**self.backend_options.get(backend, {}), **options}}
        return replace(self, backend_options=merged)


@dataclass(frozen=True)
class Capabilities:
    """What a backend actually granted, which is not always what was declared."""

    features: frozenset[Feature] = frozenset()
    resume_level: ResumeLevel | None = None
    pause_modes: frozenset[PauseMode] = frozenset()

    def supports(self, feature: Feature) -> bool:
        """Return whether a feature is implemented with its declared semantics."""
        return feature in self.features


@dataclass(frozen=True)
class AgentEndpoint:
    """Where the SDK reaches a sandbox's data plane.

    Returned with the handle, so command traffic needs no further round trip
    through the control plane.

    Two shapes, because two kinds of backend exist. A provider sandbox runs its
    own agent and reports it in `address`. A bare container runs none, so the
    node that owns its daemon drives it instead and reports itself in
    `data_plane`. Either way the call leaves the control plane out of the path;
    what differs is only which host answers it.
    """

    address: str = ""
    # The owning node's own plane, for a sandbox with no in-sandbox agent. Kept
    # apart from `address` because they are not interchangeable: one speaks the
    # backend's agent protocol, the other speaks this service's node protocol.
    data_plane: str = ""
    headers: Mapping[str, str] = field(default_factory=dict)
    callback_host_alias: str | None = None
    callback_port: int | None = None


@dataclass(frozen=True)
class SandboxHandle:
    """A stable reference to one sandbox."""

    backend: str
    sandbox_id: str
    node_id: str = ""


@dataclass(frozen=True)
class ExecResult:
    """A finished command."""

    exit_code: int
    stdout: str
    stderr: str = ""
    # Set when the backend stopped retaining output at its diagnostic budget. The
    # command still ran to completion, so this reports a bounded answer rather
    # than a failure.
    truncated: bool = False


@dataclass(frozen=True)
class SnapshotRef:
    """An opaque reference to captured state."""

    backend: str
    snapshot_id: str
    kind: SnapshotKind = SnapshotKind.FILESYSTEM
    resume_level: ResumeLevel | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NodeReport:
    """One node as the fleet monitor last saw it."""

    node_id: str
    # Every runtime the node hosts, not one: a machine running a container
    # daemon beside a microVM satisfies different specs through each, and the
    # service reports them as a list for that reason.
    backends: tuple[str, ...] = ()
    live_sandboxes: int = 0
    cpu_used_pct: float = 0.0
    mem_used_pct: float = 0.0
    gpu_free: int = 0
    draining: bool = False


@dataclass(frozen=True)
class FleetReport:
    """What the monitor sees, for the caller's metric hook."""

    nodes: tuple[NodeReport, ...] = ()
    drained_nodes: int = 0
    reservations_open: int = 0
    capacity_exhausted: int = 0
    no_candidate: int = 0
    locality_hit_ratio: float = 0.0

    def as_metrics(self) -> dict[str, float]:
        """Flatten for a metrics sink, on the training step axis."""
        return {
            "sandbox/fleet/nodes": float(len(self.nodes)),
            "sandbox/fleet/drained_nodes": float(self.drained_nodes),
            "sandbox/fleet/reservations_open": float(self.reservations_open),
            "sandbox/fleet/capacity_exhausted": float(self.capacity_exhausted),
            "sandbox/fleet/no_candidate": float(self.no_candidate),
            "sandbox/image/locality_hit_ratio": self.locality_hit_ratio,
        }


@dataclass(frozen=True)
class ClassQuota:
    """One class's share of the fleet."""

    guaranteed_share: float = 0.0
    max_share: float = 0.0
    granted_memory_mb: int = 0
    headroom_memory_mb: int = 0
    queued: int = 0


@dataclass(frozen=True)
class QuotaReport:
    """The cross-backend ledger, for the caller's metric hook."""

    classes: Mapping[str, ClassQuota] = field(default_factory=dict)

    def as_metrics(self) -> dict[str, float]:
        """Flatten for a metrics sink."""
        metrics: dict[str, float] = {}
        for name, quota in self.classes.items():
            metrics[f"sandbox/quota/{name}/granted_memory_mb"] = float(quota.granted_memory_mb)
            metrics[f"sandbox/quota/{name}/headroom_memory_mb"] = float(quota.headroom_memory_mb)
            metrics[f"sandbox/quota/{name}/queued"] = float(quota.queued)
        return metrics
