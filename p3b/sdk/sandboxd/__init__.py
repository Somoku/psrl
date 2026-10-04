"""Python SDK for the sandboxd service.

The full public surface: SandboxClient, Sandbox, all types, and all errors.
No imports from PSRL — this package is self-contained.
"""

from sandboxd.client import Sandbox, SandboxClient
from sandboxd.errors import (
    SandboxCapabilityError,
    SandboxCapacityTimeout,
    SandboxCommandTimeout,
    SandboxError,
    SandboxOomError,
    SandboxSessionLost,
    SandboxSetupError,
    SandboxTransportError,
)
from sandboxd.types import (
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
