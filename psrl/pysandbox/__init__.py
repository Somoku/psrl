"""The sandbox service's Python SDK.

This is the whole public surface. Its discipline is what keeps the service from
growing a second control plane: **the SDK serializes, remembers an endpoint, and
forwards a deadline. It makes no decisions.** No cached placement, no retry
policy of its own, no lifecycle state machine. Anything that decides lives in
the service, so the two cannot drift.

The split that makes it affordable: a create goes to the control plane and comes
back carrying the sandbox's own agent endpoint, and every command after that
goes straight to the sandbox. An episode issues one create and dozens of
commands, so a control plane in the command path would add a hop and a
serialization per command without making a decision.
"""

from psrl.pysandbox.client import Sandbox, SandboxClient
from psrl.pysandbox.errors import (
    SandboxCapabilityError,
    SandboxCapacityTimeout,
    SandboxCommandTimeout,
    SandboxError,
    SandboxOomError,
    SandboxSessionLost,
    SandboxSetupError,
    SandboxTransportError,
)
from psrl.pysandbox.types import (
    AgentEndpoint,
    Capabilities,
    ClassQuota,
    ExecMode,
    ExecResult,
    Feature,
    FleetReport,
    PauseMode,
    QuotaReport,
    Resources,
    ResumeLevel,
    SandboxHandle,
    SandboxSpec,
    SandboxStatus,
    SnapshotKind,
    SnapshotRef,
    Source,
)

__all__ = [
    "AgentEndpoint",
    "Capabilities",
    "ClassQuota",
    "ExecMode",
    "ExecResult",
    "Feature",
    "FleetReport",
    "PauseMode",
    "QuotaReport",
    "Resources",
    "ResumeLevel",
    "Sandbox",
    "SandboxCapabilityError",
    "SandboxCapacityTimeout",
    "SandboxClient",
    "SandboxCommandTimeout",
    "SandboxError",
    "SandboxHandle",
    "SandboxOomError",
    "SandboxSessionLost",
    "SandboxSetupError",
    "SandboxSpec",
    "SandboxStatus",
    "SandboxTransportError",
    "SnapshotKind",
    "SnapshotRef",
    "Source",
]
