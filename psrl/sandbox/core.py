"""Backend-neutral sandbox contracts.

The contract has two halves, and keeping them apart is what lets a stateless
backend stay small.

- The **required protocol** is the data plane: run a command, move bytes, read
  status, destroy. Every backend implements it.
- The **optional protocol** is state: pause, resume, snapshot, fork, plus the
  supporting transport refresh and resource stats. A backend declares what it
  implements through `SandboxCapabilities`, and every optional method has a
  default that raises, so a backend without state operations writes none.

There is one session type rather than a stateful and a stateless one.
`SandboxStateProtocol` exists so a caller can be typed against the optional half
without a second class, and the manager is the only caller of it, which is what
keeps capability checks and safety policy in one place.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit, urlunsplit


class SandboxOomError(RuntimeError):
    """
    Raised when a sandbox's container was OOM-killed mid-command.

    The kernel kills the whole container, so the episode loses its sandbox and any
    partial work in it. Reported as its own type because it is a resource fault, not
    a model or task failure, and it must not be confused with a slow or hung episode.
    """


class SandboxTransportError(RuntimeError):
    """
    Raised when a backend's control channel failed rather than the command itself.

    A lost transport says nothing about the workload, so callers must not read it as a
    task result. Backends raise their own subclass so a session can recognize a transport
    fault without catching every `RuntimeError` its own code might raise.
    """


class SandboxCapacityTimeout(RuntimeError):
    """
    Raised when node capacity admission never granted a sandbox in time.

    The sandbox was never created, so no episode ever ran: this is a capacity
    planning fault rather than a task, model, or harness failure. It is reported
    as its own type so callers can classify it apart from a cancelled or failed
    episode, and so the wait is visible instead of surfacing as a bare
    cancellation.
    """


class SandboxCapabilityError(RuntimeError):
    """
    Raised when a request requires a capability the selected backend does not have.

    Admission fails fast rather than losing a guarantee quietly. A caller that needs
    a full-state resume must not be served a filesystem resume and told it is the
    same thing.
    """


class SandboxBusyError(RuntimeError):
    """
    Raised when a state operation finds a command in flight.

    A pause that found the sandbox busy must not wait for the command to finish,
    because the caller decided the sandbox was idle and would otherwise hold the
    idle pass open for the duration of a long command. The caller skips the sandbox
    and decides again on the next pass.
    """


class SandboxCommandTimeout(TimeoutError):
    """Raised when a command exceeded its deadline.

    `sandbox_preserved` says whether the sandbox survived. A backend that can stop
    the overrunning process keeps the sandbox and its filesystem, so a caller can
    continue the episode with a fresh shell. A backend that cannot stop it must
    destroy the sandbox, because a process left running would corrupt the next
    command's output. The two outcomes need different caller responses, so they must
    not arrive as the same exception.

    It subclasses `TimeoutError`, which is what the session contract already promises.
    """

    def __init__(self, message: str, *, sandbox_preserved: bool) -> None:
        super().__init__(message)
        self.sandbox_preserved = sandbox_preserved


class SandboxSetupError(RuntimeError):
    """
    Raised when a task's preparation step failed inside its sandbox.

    The environment is not what the task needs, so a captured version of it would be
    wrong for every later reuse. Reported as its own type because it is a task
    preparation fault rather than a sandbox or model fault.
    """


class SandboxProvisionError(RuntimeError):
    """
    Transfer ownership of a partially provisioned sandbox to its caller.

    The session may still consume resources and must be terminated before its
    capacity reservation can be returned.
    """

    def __init__(self, session: SandboxSession, cause: BaseException) -> None:
        super().__init__(f"Sandbox provisioning requires cleanup for {session.ref!r}: {cause!r}.")
        self.session = session


class SandboxSessionLostError(RuntimeError):
    """Raised when a sandbox stopped for a reason PSRL did not ask for.

    The container is gone, so the command that was running never completed and the work
    in it is unrecoverable. It carries its own type, and the exit diagnostics the runtime
    still knows, because it is an infrastructure fault rather than a model or harness
    failure: the rollout must be replaced, but it must not be counted as evidence that the
    harness is broken.

    Distinct from `SandboxOomError`, which is the sandbox hitting its own configured
    memory limit, and from `SandboxCommandTimeout`, which is a command that merely ran
    out of time inside a container that is still healthy.
    """

    def __init__(
        self,
        message: str,
        *,
        sandbox_id: str = "",
        exit_reason: SandboxExitReason | None = None,
        exit_code: int | None = None,
        state_error: str = "",
        finished_at: str = "",
    ) -> None:
        super().__init__(message)
        self.sandbox_id = sandbox_id
        # Reuses the backend-neutral vocabulary a post mortem already reads, so a caller
        # never has to parse a second set of reason strings.
        self.exit_reason = exit_reason or SandboxExitReason.UNKNOWN
        self.exit_code = exit_code
        self.state_error = state_error
        self.finished_at = finished_at


class SandboxFeature(str, Enum):
    """
    Optional semantic features exposed by a sandbox backend.
    """

    # Suspend processes while retaining the sandbox's allocated host resources.
    FREEZE = "freeze"
    # Persist a sandbox so its compute resources can be released and later reacquired.
    HIBERNATE = "hibernate"
    # Capture durable filesystem state without promising process or memory state.
    FILESYSTEM_SNAPSHOT = "filesystem_snapshot"
    # Capture filesystem, process, and memory state as one restorable checkpoint.
    FULL_STATE_SNAPSHOT = "full_state_snapshot"
    # Create a new sandbox from a previously captured snapshot.
    RESTORE = "restore"
    # Clone a live sandbox through a backend-native copy-on-write operation.
    NATIVE_FORK = "native_fork"
    # Bind a path on the local host into the sandbox.
    HOST_MOUNT = "host_mount"
    # Restore a snapshot on a host other than the one that captured it. The resume
    # level is separate because restoring a disk and a live process differ.
    RESUME_ANYWHERE = "resume_anywhere"
    # Hand out a sandbox that is already started, instead of creating one.
    WARM_POOL = "warm_pool"
    # Load image layers lazily, so a node never materializes the whole image.
    IMAGE_ON_DEMAND = "image_on_demand"
    # Deliver image blocks from a node cache and from peers rather than from the
    # origin. Block delivery is not layer laziness, so a request must select one.
    IMAGE_BLOCK_DELIVERY = "image_block_delivery"
    # Build a reusable sandbox template from a build context.
    TEMPLATE_BUILD = "template_build"
    # Create and mount provider-managed storage that outlives one sandbox.
    VOLUME = "volume"
    # Enforce a per-sandbox network allowlist.
    EGRESS_POLICY = "egress_policy"
    # Inject secrets without placing them in the image or the spec.
    CREDENTIAL_INJECTION = "credential_injection"
    # Select a stronger isolation boundary than a shared host kernel.
    ISOLATION_RUNTIME = "isolation_runtime"


class ResumeLevel(str, Enum):
    """
    What survives when a sandbox resumes somewhere else.

    Ordered from weakest to strongest, so a requirement is one comparison rather
    than a capability matrix. A filesystem resume restores the workspace and every
    process is new, which is enough for a setup step whose result is on disk. A
    full-state resume restores memory and running processes, which is what a
    harness needs to continue instead of restart.

    The distinction is the whole reason the level exists: a conformance run on a
    filesystem backend must not be readable as proof that a harness survives a
    resume.
    """

    FILESYSTEM = "filesystem"
    FULL_STATE = "full_state"

    @property
    def rank(self) -> int:
        """
        Return the strength of this level for ordering comparisons.
        """
        return _RESUME_LEVEL_RANK[self]

    def satisfies(self, required: ResumeLevel) -> bool:
        """
        Return whether this level is at least as strong as a requirement.
        """
        return self.rank >= required.rank


_RESUME_LEVEL_RANK = {ResumeLevel.FILESYSTEM: 0, ResumeLevel.FULL_STATE: 1}


class SandboxSourceKind(str, Enum):
    """
    Portable sandbox source kinds.
    """

    # OCI-style image containing the sandbox filesystem and startup metadata.
    IMAGE = "image"
    # Provider-managed template that may include prebuilt runtime configuration.
    TEMPLATE = "template"


class SnapshotKind(str, Enum):
    """
    State included in a snapshot.
    """

    # Filesystem contents only. Running process and memory state are excluded.
    FILESYSTEM = "filesystem"
    # Filesystem, running processes, and memory state when supported by the backend.
    FULL_STATE = "full_state"


class PauseMode(str, Enum):
    """
    Pause semantics requested by the caller.
    """

    # Stop process scheduling but keep the sandbox resident on the current host.
    FREEZE = "freeze"
    # Release compute resources while retaining enough state for a later resume.
    HIBERNATE = "hibernate"


class ExecMode(str, Enum):
    """
    How a backend runs commands for this sandbox.

    One exec interface with two strategies, not two interfaces. A harness that
    keeps shell state across turns needs a persistent shell. A grader that runs a
    single command is cheaper without one.
    """

    # Keep a shell alive across commands so state accumulates.
    PERSISTENT = "persistent"
    # Start a process per command.
    ONE_SHOT = "one_shot"


class SandboxStatus(str, Enum):
    """
    Backend-neutral session state.
    """

    # The sandbox accepts commands.
    RUNNING = "running"
    # The sandbox exists but cannot execute commands until resumed.
    PAUSED = "paused"
    # The sandbox workload exited but its runtime object still exists.
    EXITED = "exited"
    # The sandbox runtime object has been destroyed.
    TERMINATED = "terminated"
    # The provider cannot map its current state to a portable state.
    UNKNOWN = "unknown"


class SandboxExitReason(str, Enum):
    """
    Why a sandbox stopped existing.

    A post mortem starts here, so these must not collapse into one another. An
    operator acts differently on a sandbox killed for memory, one stopped because
    its caller finished, and one reclaimed because its owner died.
    """

    # The caller released it.
    RELEASED = "released"
    # The reclaimer destroyed it after it sat idle past its window.
    REAPED_IDLE = "reaped_idle"
    # The reclaimer destroyed it at its absolute lifetime.
    REAPED_LIFETIME = "reaped_lifetime"
    # The kernel killed it for memory.
    OOM = "oom"
    # A node reclaimer found it with no live owner.
    RECLAIMED_ORPHAN = "reclaimed_orphan"
    # It failed on its own.
    FAILED = "failed"
    # It is gone and no local record explains why.
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SandboxStatePolicy:
    """
    Safety requirements for full-state snapshots and branches.
    """

    enabled: bool = False
    allow_secret_capture: bool = False
    allow_external_side_effects: bool = False
    reseed_after_restore: bool = True


@dataclass(frozen=True)
class SandboxRef:
    """
    Stable reference to a backend-owned sandbox.
    """

    backend: str
    sandbox_id: str


@dataclass(frozen=True)
class SnapshotRef:
    """
    Opaque reference to backend-owned snapshot state.
    """

    backend: str
    snapshot_id: str
    kind: SnapshotKind
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # What a restore from this snapshot preserves. None means the backend did not
    # report a level, which a caller that needs one must treat as unsatisfied.
    resume_level: ResumeLevel | None = None


@dataclass(frozen=True)
class SandboxSource:
    """
    Source from which a sandbox is provisioned.
    """

    kind: SandboxSourceKind
    reference: str

    def __post_init__(self) -> None:
        if not self.reference.strip():
            raise ValueError("Sandbox source reference cannot be empty.")

    @classmethod
    def image(cls, image: str) -> SandboxSource:
        """
        Create an image-backed source.
        """
        return cls(SandboxSourceKind.IMAGE, image)

    @classmethod
    def template(cls, template: str) -> SandboxSource:
        """
        Create a provider-template-backed source.
        """
        return cls(SandboxSourceKind.TEMPLATE, template)


@dataclass(frozen=True)
class ResourceSpec:
    """
    Optional portable resource requests.
    """

    cpu_count: float | None = None
    memory_mb: int | None = None
    disk_mb: int | None = None
    # Accelerators, by count, for a workload that needs a device inside the sandbox.
    # A node admits them from the same envelope as CPU and memory.
    gpu_count: int | None = None

    def __post_init__(self) -> None:
        if self.cpu_count is not None and self.cpu_count <= 0:
            raise ValueError("Sandbox cpu_count must be greater than zero.")
        if self.memory_mb is not None and self.memory_mb <= 0:
            raise ValueError("Sandbox memory_mb must be greater than zero.")
        if self.disk_mb is not None and self.disk_mb <= 0:
            raise ValueError("Sandbox disk_mb must be greater than zero.")
        if self.gpu_count is not None and self.gpu_count < 0:
            raise ValueError("Sandbox gpu_count must not be negative.")


@dataclass(frozen=True)
class MountSpec:
    """
    Local host bind mount requested by a sandbox workload.
    """

    source: str
    target: str
    read_only: bool = False


@dataclass(frozen=True)
class VolumeSpec:
    """
    Provider-managed storage that outlives one sandbox.

    Distinct from `MountSpec` on purpose: a host bind mount names a path on this
    node, and provider storage does not. Collapsing them would let a portable spec
    depend on a node detail.
    """

    name: str
    target: str
    read_only: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Sandbox volume name cannot be empty.")
        if not self.target.strip():
            raise ValueError("Sandbox volume target cannot be empty.")


class EgressAction(str, Enum):
    """
    What a policy or a rule does with a matching destination.
    """

    ALLOW = "allow"
    DENY = "deny"


@dataclass(frozen=True)
class EgressRule:
    """
    One destination and what happens to it.
    """

    action: EgressAction
    # A hostname, a wildcard domain such as `*.example.com`, or a CIDR block. A
    # name-resolving provider accepts the first two, and the internal firewall all three.
    target: str
    # Ports this rule covers. A provider that derives the port from the scheme
    # refuses a rule that names one, because it cannot honour the request.
    ports: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.target.strip():
            raise ValueError("Sandbox egress rule target cannot be empty.")
        for port in self.ports:
            if not 0 < port < 65536:
                raise ValueError("Sandbox egress rule ports must be within 1..65535.")


@dataclass(frozen=True)
class EgressPolicy:
    """Per-sandbox outbound policy: one default, then ordered rules.

    A sandbox with an unbounded egress can reach a service it should not, which
    makes its own reward untrustworthy before it is a security problem. The policy
    is a training data integrity control for that reason.

    The default is deny, because a policy that has to deny explicitly is one a
    caller forgets, and the omission is invisible until a reward looks wrong.
    """

    default_action: EgressAction = EgressAction.DENY
    rules: tuple[EgressRule, ...] = ()

    @property
    def allow_all(self) -> bool:
        """
        Return whether anything outside the rules may be reached.
        """
        return self.default_action is EgressAction.ALLOW

    @property
    def allowed_targets(self) -> tuple[str, ...]:
        """
        Return the destinations an allow rule covers.
        """
        return tuple(rule.target for rule in self.rules if rule.action is EgressAction.ALLOW)

    @property
    def allowed_ports(self) -> tuple[int, ...]:
        """
        Return the ports an allow rule names, in order and without duplicates.
        """
        return tuple(dict.fromkeys(port for rule in self.rules for port in rule.ports))

    def allows(self, host: str) -> bool:
        """
        Return whether this policy lets the sandbox reach a host.
        """
        if self.default_action is EgressAction.ALLOW:
            return True
        return any(_host_matches(host, target) for target in self.allowed_targets)


def _host_matches(host: str, target: str) -> bool:
    """
    Return whether one allow rule target covers a host.
    """
    if target == host:
        return True
    if target.startswith("*."):
        return host.endswith(target[1:]) and host != target[2:]
    return False


@dataclass(frozen=True)
class CredentialRef:
    """
    A secret, named but never carried in the spec.

    The value is resolved from the environment at the point of use, so a spec can
    be logged, hashed, or compared without leaking a secret.
    """

    # Environment variable in this process that holds the value.
    source_env: str
    # Environment variable the sandbox sees. A backend with no broker puts the value here,
    # while a brokering provider injects it on the way out behind a placeholder.
    target_env: str
    # The name the credential is registered under when a provider brokers it.
    # Defaults to the sandbox-facing variable name.
    name: str | None = None

    def __post_init__(self) -> None:
        if not self.source_env.strip() or not self.target_env.strip():
            raise ValueError("Sandbox credential references require non-empty environment variable names.")
        if self.name is not None and not self.name.strip():
            raise ValueError("Sandbox credential name cannot be empty when set.")

    @property
    def vault_name(self) -> str:
        """
        Return the name a brokering provider registers this credential under.
        """
        return self.name or self.target_env


class CredentialAuthType(str, Enum):
    """
    How a binding attaches its credential to an outbound request.
    """

    BEARER = "bearer"
    BASIC = "basic"
    API_KEY = "apiKey"
    CUSTOM_HEADERS = "customHeaders"
    PASSTHROUGH = "passthrough"


@dataclass(frozen=True)
class CredentialSubstitution:
    """
    A literal placeholder an outbound request may carry instead of a secret.
    """

    credential: str
    placeholder: str
    # Where the replacement is allowed: `path`, `query`, or `body`.
    surfaces: tuple[str, ...] = ("body",)

    def __post_init__(self) -> None:
        if not self.credential.strip() or not self.placeholder.strip():
            raise ValueError("A credential substitution requires a credential name and a placeholder.")
        unknown = set(self.surfaces) - {"path", "query", "body"}
        if unknown:
            raise ValueError(f"Credential substitution surfaces {sorted(unknown)} are not path, query, or body.")


@dataclass(frozen=True)
class CredentialAuth:
    """
    How one binding's credential is rendered into the outbound request.
    """

    type: CredentialAuthType = CredentialAuthType.BEARER
    # Header the value is injected into, for `apiKey`.
    header_name: str | None = None
    # For `customHeaders`: pairs of header name and the credential that fills it.
    headers: tuple[tuple[str, str], ...] = ()
    substitutions: tuple[CredentialSubstitution, ...] = ()

    def __post_init__(self) -> None:
        if self.type is CredentialAuthType.API_KEY and not (self.header_name or "").strip():
            raise ValueError("An apiKey credential auth requires the header name to inject into.")
        if self.type is CredentialAuthType.CUSTOM_HEADERS and not self.headers:
            raise ValueError("A customHeaders credential auth requires at least one header.")
        if self.type is not CredentialAuthType.CUSTOM_HEADERS and self.headers:
            raise ValueError(f"Credential auth type {self.type.value!r} carries no header list.")
        if self.type is CredentialAuthType.PASSTHROUGH and not self.substitutions:
            raise ValueError("A passthrough credential auth only substitutes, so it requires a substitution.")

    def to_payload(self, credential: str) -> dict[str, Any]:
        """
        Render this rule for a provider, with the credential it applies to.
        """
        payload: dict[str, Any] = {"type": self.type.value}
        if self.type is CredentialAuthType.CUSTOM_HEADERS:
            payload["headers"] = [{"name": name, "credential": item} for name, item in self.headers]
        else:
            payload["credential"] = credential
        if self.type is CredentialAuthType.API_KEY and self.header_name:
            payload["name"] = self.header_name
        if self.substitutions:
            payload["substitutions"] = [
                {"credential": item.credential, "placeholder": item.placeholder, "in": list(item.surfaces)}
                for item in self.substitutions
            ]
        return payload

    def credential_names(self) -> tuple[str, ...]:
        """
        Return every credential this rule reads.
        """
        if self.type is CredentialAuthType.CUSTOM_HEADERS:
            names = [item for _, item in self.headers]
        else:
            names = []
        names.extend(item.credential for item in self.substitutions)
        return tuple(names)


@dataclass(frozen=True)
class CredentialBinding:
    """Where a brokered credential applies, and how it is attached.

    A broker matches an outbound request against this and injects the credential,
    which is what keeps the real value out of the sandbox's environment, command
    line, filesystem, and logs.
    """

    name: str
    # The host(s) this credential may be sent to. Required, because a binding that
    # matched every host would hand the credential to whatever the workload asked for.
    hosts: tuple[str, ...]
    # The credential this binding injects, by vault name.
    credential: str
    auth: CredentialAuth = field(default_factory=CredentialAuth)
    schemes: tuple[str, ...] = ("https",)
    methods: tuple[str, ...] = ("GET", "POST", "PUT", "PATCH", "DELETE")
    paths: tuple[str, ...] = ("/*",)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("A credential binding requires a name.")
        if not self.hosts:
            raise ValueError(
                "A credential binding requires at least one host, or the credential would be offered to any "
                "destination the workload chooses."
            )
        if not self.credential.strip():
            raise ValueError("A credential binding requires the credential it injects.")
        if not self.schemes:
            raise ValueError("A credential binding requires at least one scheme.")
        if not self.methods:
            raise ValueError("A credential binding requires at least one method.")

    def to_payload(self) -> dict[str, Any]:
        """
        Render this binding for a provider.
        """
        return {
            "name": self.name,
            "match": {
                "schemes": list(self.schemes),
                "hosts": list(self.hosts),
                "methods": list(self.methods),
                "paths": list(self.paths),
            },
            "auth": self.auth.to_payload(self.credential),
        }


@dataclass(frozen=True)
class SandboxSpec:
    """
    Portable sandbox creation request.

    `resource_class` selects admission priority. `workflow_id` reserves one
    sandbox phase per workflow, including requests still waiting for capacity.
    """

    source: SandboxSource
    resources: ResourceSpec = field(default_factory=ResourceSpec)
    workdir: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)
    env: Mapping[str, str] = field(default_factory=dict)
    mounts: tuple[MountSpec, ...] = ()
    volumes: tuple[VolumeSpec, ...] = ()
    policy_profile: str | None = None
    idle_timeout_s: float | None = None
    lifetime_timeout_s: float | None = None
    idempotency_key: str | None = None
    state_policy: SandboxStatePolicy = field(default_factory=SandboxStatePolicy)
    required_features: frozenset[SandboxFeature] = frozenset()
    required_resume_level: ResumeLevel | None = None
    resource_class: str = "default"
    workflow_id: str | None = None
    # An optional backend pin. A recipe names a provider here, and its requirements
    # are still validated against that backend rather than trusted.
    backend: str | None = None
    # None means the backend's own default, so a deployment declares its intent once
    # and a workload overrides it only when it actually needs the other strategy.
    exec_mode: ExecMode | None = None
    egress: EgressPolicy | None = None
    credentials: tuple[CredentialRef, ...] = ()
    # Where each credential applies, for a backend that brokers secrets at its egress
    # boundary. A brokerless backend injects into the environment and takes these as absent.
    credential_bindings: tuple[CredentialBinding, ...] = ()
    # Device indices granted by node admission, written by the manager and never by a
    # caller, which is why it is outside the request identity below.
    assigned_gpus: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (("idle_timeout_s", self.idle_timeout_s), ("lifetime_timeout_s", self.lifetime_timeout_s)):
            if value is not None and value <= 0:
                raise ValueError(f"Sandbox {name} must be greater than zero.")
        if self.idempotency_key is not None and not self.idempotency_key.strip():
            raise ValueError("Sandbox idempotency_key cannot be empty.")
        if not self.resource_class.strip():
            raise ValueError("Sandbox resource_class cannot be empty.")
        if self.workflow_id is not None and not self.workflow_id.strip():
            raise ValueError("Sandbox workflow_id cannot be empty when set.")
        if self.required_resume_level is not None and SandboxFeature.RESUME_ANYWHERE not in self.required_features:
            raise ValueError(
                "Sandbox required_resume_level requires RESUME_ANYWHERE in required_features, or the level "
                "would describe a resume the caller never asked for."
            )
        known = {credential.vault_name for credential in self.credentials}
        for binding in self.credential_bindings:
            named = {binding.credential, *binding.auth.credential_names()}
            missing = named - known
            if missing:
                raise ValueError(
                    f"Sandbox credential binding {binding.name!r} names {sorted(missing)}, which the spec's "
                    "credentials do not define."
                )

    def _identity(self) -> tuple:
        """
        Return the fields that make two specs the same request.
        """
        return (
            self.source,
            self.resources,
            self.workdir,
            tuple(sorted(self.metadata.items())),
            tuple(sorted(self.env.items())),
            self.mounts,
            self.volumes,
            self.policy_profile,
            self.idle_timeout_s,
            self.lifetime_timeout_s,
            self.idempotency_key,
            self.state_policy,
            frozenset(self.required_features),
            self.required_resume_level,
            self.resource_class,
            self.workflow_id,
            self.backend,
            self.exec_mode,
            self.egress,
            self.credentials,
            self.credential_bindings,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SandboxSpec):
            return NotImplemented
        return self._identity() == other._identity()

    def __hash__(self) -> int:
        return hash(self._identity())


@dataclass(frozen=True)
class ExecResult:
    """
    Completed command result.
    """

    exit_code: int
    stdout: str
    stderr: str
    # Set when the backend stopped retaining output at its diagnostic budget. The command
    # itself ran to completion, so this reports a bounded answer rather than a failure.
    truncated: bool = False


@dataclass(frozen=True)
class ResourceUsage:
    """Point-in-time resource usage reported by a backend.

    A field is None when the backend does not report that quantity, which is not the same
    as reporting a measured zero. A provider that exposes only a memory level and a CPU
    share leaves the peak and the cumulative time unknown, and reporting zero for them
    would be indistinguishable from a sandbox that used nothing.
    """

    memory_bytes: int | None = None
    peak_memory_bytes: int | None = None
    cpu_total_ns: int | None = None


@dataclass(frozen=True)
class SandboxDiagnostics:
    """
    Read-only evidence about a sandbox that may already be gone.

    Once a sandbox can land on any node, the operator no longer knows which node
    to inspect, and the container may already be destroyed. The node that hosted
    it is the only place that can answer.
    """

    ref: SandboxRef
    status: SandboxStatus = SandboxStatus.UNKNOWN
    exit_reason: SandboxExitReason = SandboxExitReason.UNKNOWN
    # Bounded, the same way an observation is bounded, so a diagnosis cannot pull a
    # whole log through a control channel.
    log_tail: str = ""
    # The container's own inspect payload, which is what distinguishes an OOM kill
    # from a non-zero exit.
    inspect: Mapping[str, Any] = field(default_factory=dict)
    usage: ResourceUsage = field(default_factory=ResourceUsage)

    def as_dict(self) -> dict[str, Any]:
        """
        Serialize the diagnosis for a control channel.
        """
        return {
            "backend": self.ref.backend,
            "sandbox_id": self.ref.sandbox_id,
            "status": self.status.value,
            "exit_reason": self.exit_reason.value,
            "log_tail": self.log_tail,
            "inspect": dict(self.inspect),
            "usage": {
                "memory_bytes": self.usage.memory_bytes,
                "peak_memory_bytes": self.usage.peak_memory_bytes,
                "cpu_total_ns": self.usage.cpu_total_ns,
            },
        }


@dataclass(frozen=True)
class SandboxCapabilities:
    """
    Semantic capabilities implemented by one backend or session.
    """

    features: frozenset[SandboxFeature] = frozenset()
    # Set only when RESUME_ANYWHERE is declared, so the two cannot contradict each
    # other. It states what a resume elsewhere preserves.
    resume_level: ResumeLevel | None = None

    def __post_init__(self) -> None:
        if self.resume_level is not None and SandboxFeature.RESUME_ANYWHERE not in self.features:
            raise ValueError(
                "Sandbox capabilities cannot declare a resume_level without RESUME_ANYWHERE, or a caller "
                "would read a level for a resume the backend does not offer."
            )

    def supports(self, feature: SandboxFeature) -> bool:
        """
        Return whether the feature is implemented with its declared semantics.
        """
        return feature in self.features

    def require(self, *features: SandboxFeature) -> None:
        """
        Raise when one or more required features are unavailable.
        """
        missing = [feature.value for feature in features if feature not in self.features]
        if missing:
            raise SandboxCapabilityError(f"Sandbox does not support: {', '.join(sorted(missing))}.")

    def require_resume_level(self, required: ResumeLevel) -> None:
        """
        Raise when this backend cannot resume elsewhere at the required strength.

        Admission rejects rather than downgrades, so a caller that needs a live
        process to survive a move never receives a workspace-only resume.
        """
        self.require(SandboxFeature.RESUME_ANYWHERE)
        level = self.resume_level
        if level is None:
            raise SandboxCapabilityError(
                "Sandbox does not report a resume level, so a cross node resume cannot be guaranteed."
            )
        if not level.satisfies(required):
            raise SandboxCapabilityError(
                f"Sandbox resumes at level {level.value!r}, which does not satisfy a required "
                f"{required.value!r} resume."
            )


@runtime_checkable
class SandboxStateProtocol(Protocol):
    """
    The optional half of the session contract.

    A session implements this when its backend declares the matching capability.
    The manager is the only caller, so capability gating and state safety policy
    cannot be bypassed by reaching a session directly.
    """

    async def pause(self, mode: PauseMode) -> None:
        """
        Pause the session with explicit semantics.
        """

    async def resume(self) -> None:
        """
        Resume a paused session.
        """

    async def snapshot(self, kind: SnapshotKind) -> SnapshotRef:
        """
        Capture session state.
        """

    async def fork(self, count: int) -> Sequence[SandboxSession]:
        """
        Create `count` backend-native branches of the session.
        """

    async def refresh_transport(self) -> None:
        """
        Drop stale provider connections after restoring or cloning state.
        """

    async def stats(self) -> ResourceUsage:
        """
        Return resource usage when supported.
        """


class SandboxSession(ABC):
    """
    One live execution environment.

    Command and file operations form the required data plane. State operations
    have default unsupported implementations and are enabled by capabilities,
    which keeps simple backends free of state-operation boilerplate.
    """

    @property
    @abstractmethod
    def ref(self) -> SandboxRef:
        """
        Return the stable backend reference.
        """

    @property
    @abstractmethod
    def capabilities(self) -> SandboxCapabilities:
        """
        Return features implemented by this session.
        """

    @abstractmethod
    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
        silence_timeout_s: float | None = None,
    ) -> ExecResult:
        """
        Execute a command and collect its result.

        Implementations may serialize commands per session, so callers must not rely on
        running two commands in one sandbox concurrently. `timeout_s` therefore bounds the
        whole operation including any wait for the session, not just the command itself.

        Args:
            command (str): Command line interpreted by the backend's shell.
            cwd (str | None): Working directory, defaulting to the session's own.
            env (Mapping[str, str] | None): Variables added for this command only.
            timeout_s (float | None): Deadline covering the queue wait, setup, streaming,
                and final status. None waits indefinitely.
            silence_timeout_s (float | None): Longest stretch with no output at all. A
                command that prints nothing for this long is stuck rather than slow, and
                the two need different answers. None uses the backend's own default.

        Raises:
            TimeoutError: When the deadline expires.
            SandboxOomError: When the container was OOM-killed while the command ran.
        """

    @abstractmethod
    async def read_bytes(self, path: str) -> bytes:
        """
        Read a file without assuming text encoding.
        """

    @abstractmethod
    async def write_bytes(self, path: str, data: bytes) -> None:
        """
        Write a complete file.
        """

    @abstractmethod
    async def status(self) -> SandboxStatus:
        """
        Return the current session status.
        """

    @abstractmethod
    async def terminate(self) -> None:
        """
        Idempotently destroy the session, raising if destruction is unconfirmed.
        """

    async def stats(self) -> ResourceUsage:
        """
        Return resource usage when supported.
        """
        return ResourceUsage()

    def resolve_callback_url(self, url: str) -> str:
        """
        Translate a worker URL into an equivalent URL reachable from this sandbox.
        """
        return url

    @property
    def spec(self) -> SandboxSpec | None:
        """
        Return the creation spec when this process created the session.
        """
        return None

    @property
    def command_count(self) -> int:
        """
        Return the number of user commands executed by this session object.
        """
        return 0

    @property
    def callback_host_alias(self) -> str | None:
        """Return the alias this sandbox reaches the caller's worker by, when it needs one.

        A caller on another node applies this rather than deciding it, because the
        alias depends on the hosting node's network policy.
        """
        return None

    @property
    def protected_env_names(self) -> frozenset[str]:
        """Return environment variable names whose values must not enter a snapshot.

        A credential is injected at creation rather than named on the spec, so the
        spec alone cannot tell the state policy that the sandbox holds a secret. A
        session that injects one has to say so, or a checkpoint would capture it
        with the workspace.
        """
        return frozenset()

    @property
    def busy(self) -> bool:
        """Return whether a command is executing right now.

        This is the in-flight half of the idle test. A backend that stamps a
        last-used time when a command *returns* leaves that stamp stale for the
        whole duration of a long command, so age alone cannot distinguish a busy
        sandbox from an idle one.
        """
        return False

    @property
    def last_activity_at(self) -> float | None:
        """
        Return the monotonic time of the last command boundary, start or return.
        """
        return None

    @property
    def exit_reason(self) -> SandboxExitReason:
        """
        Return why this session stopped existing, once something stopped it.
        """
        return SandboxExitReason.UNKNOWN

    async def diagnostics(self) -> SandboxDiagnostics:
        """
        Return read-only evidence about this session.
        """
        return SandboxDiagnostics(ref=self.ref, status=await self.status(), exit_reason=self.exit_reason)

    async def refresh_transport(self) -> None:
        """
        Drop stale provider connections after restoring VM state.
        """
        return None

    async def pause(self, mode: PauseMode) -> None:
        """
        Pause the session with explicit semantics.
        """
        raise NotImplementedError(f"Sandbox {self.ref} does not support {mode.value}.")

    async def resume(self) -> None:
        """
        Resume a paused session.
        """
        raise NotImplementedError(f"Sandbox {self.ref} does not support resume.")

    async def snapshot(self, kind: SnapshotKind) -> SnapshotRef:
        """
        Capture session state.
        """
        raise NotImplementedError(f"Sandbox {self.ref} does not support {kind.value} snapshots.")

    async def fork(self, count: int = 1) -> Sequence[SandboxSession]:
        """
        Create backend-native branches of the session.

        Args:
            count (int): Number of children to create. The group size is a config
                value, so a caller asks for the whole group in one call rather than
                discovering the size per prompt.

        Returns:
            Sequence[SandboxSession]: The children, in the order the backend made them.
        """
        raise NotImplementedError(f"Sandbox {self.ref} does not support native fork.")


def resolve_credential(credential: CredentialRef) -> str:
    """
    Resolve one secret from this process, never from the spec.

    Args:
        credential (CredentialRef): The reference naming the variable to read.

    Returns:
        str: The secret value.

    Raises:
        RuntimeError: When this process does not hold the variable. A sandbox that
            silently ran unauthenticated would produce a reward that looks valid.
    """
    value = os.environ.get(credential.source_env)
    if value is None:
        raise RuntimeError(
            f"Sandbox credential {credential.source_env!r} is not set in this process, so "
            f"{credential.target_env!r} cannot be injected."
        )
    return value


_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def rewrite_loopback_proxy(value: str, host_alias: str, *, port: int | None = None) -> str:
    """Replace a URL's loopback host with the alias its sandbox reaches the host by.

    A backend that fronts its sandbox with a host alias needs this for any URL the
    workload is handed, and the control plane needs it to resolve a cross-node callback,
    so it lives beside the contracts rather than inside one backend.

    Args:
        value (str): URL or `host:port` authority, with or without a scheme.
        host_alias (str): Alias the sandbox resolves to its host.
        port (int | None): Port to substitute, when the service is reached through a
            node-side forwarder listening on a port of its own.

    Returns:
        str: The rewritten value, or the original when its host is not loopback.
    """
    has_scheme = "://" in value
    parsed = urlsplit(value if has_scheme else f"//{value}")
    if parsed.hostname not in _LOOPBACK_HOSTS:
        return value
    try:
        resolved_port = port if port is not None else parsed.port
    except ValueError:
        return value
    suffix = f":{resolved_port}" if resolved_port is not None else ""
    userinfo, separator, _ = parsed.netloc.rpartition("@")
    authority = f"{userinfo}{separator}{host_alias}{suffix}"
    rewritten = urlunsplit((parsed.scheme, authority, parsed.path, parsed.query, parsed.fragment))
    return rewritten if has_scheme else rewritten.removeprefix("//")


def resolve_credentials(spec: SandboxSpec) -> dict[str, str]:
    """Resolve injected secrets from this process, never from the spec.

    A referenced variable that is absent fails the create rather than starting a
    sandbox without the credential: a task that silently runs unauthenticated
    produces a reward that looks valid and is not.

    Args:
        spec (SandboxSpec): The spec whose `credentials` name the variables.

    Returns:
        dict[str, str]: Target variable name to secret value, for the point of use.
    """
    return {credential.target_env: resolve_credential(credential) for credential in spec.credentials}


def require_batch_count(count: int) -> int:
    """
    Validate a fork batch count.

    Args:
        count (int): Requested child count.

    Returns:
        int: The validated count.
    """
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("Sandbox fork count must be a positive integer.")
    return count


class SandboxBackend(ABC):
    """
    Provisioning and reconnection boundary for one runtime backend.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the unique manager registration name.
        """

    @property
    @abstractmethod
    def capabilities(self) -> SandboxCapabilities:
        """
        Return capabilities available to newly created sessions.
        """

    @property
    def uses_node_capacity(self) -> bool:
        """
        Return whether sessions consume resources on the worker's node.
        """
        return False

    @abstractmethod
    async def create(self, spec: SandboxSpec) -> SandboxSession:
        """
        Create a session from a portable specification.

        Raises:
            SandboxProvisionError: When creation failed but a runtime object might still
                exist. Carrying the session transfers cleanup to the caller, which is the
                only way its resources can be released. Failures that allocated nothing
                must raise their own error instead.
        """

    @abstractmethod
    async def connect(self, sandbox_id: str) -> SandboxSession:
        """
        Connect to and, when necessary, resume a session.
        """

    async def prepare(self, spec: SandboxSpec) -> None:
        """
        Warm reusable artifacts without allocating a sandbox or capacity lease.

        Backends without artifact preparation may leave this as a no-op. A template
        build belongs here, so a build failure is an admission failure rather than a
        rollout failure.
        """
        return None

    async def restore(self, snapshot: SnapshotRef, spec: SandboxSpec | None = None) -> SandboxSession:
        """
        Create a session from a snapshot when supported.
        """
        raise NotImplementedError(f"Backend {self.name!r} does not support restore.")

    async def shutdown(self) -> None:
        """
        Release backend-level resources.
        """
        return None

    async def delete_snapshot(self, snapshot: SnapshotRef) -> None:
        """
        Delete a provider-owned snapshot when supported.
        """
        raise NotImplementedError(f"Backend {self.name!r} does not support snapshot deletion.")

    def metrics_snapshot(self) -> Any:
        """
        Return backend-local metrics without imposing an exporter.
        """
        from psrl.sandbox.metrics import SandboxMetricsSnapshot

        return SandboxMetricsSnapshot()
