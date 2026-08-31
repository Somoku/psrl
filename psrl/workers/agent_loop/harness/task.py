"""Task contract consumed by the generic harness agent loop."""

from dataclasses import dataclass
from typing import Generic, TypeVar

from psrl.sandbox import SandboxSpec, SnapshotKind

TaskStateT = TypeVar("TaskStateT")


@dataclass(frozen=True)
class HarnessTaskContext(Generic[TaskStateT]):
    """
    Store an immutable task description with opaque task-specific state.

    The context does not own generic runtime resources. `HarnessAgentLoop` owns
    the session, sandbox lease, harness process, and snapshot lifecycles. The
    opaque `state` may carry task resources released by `close_harness_task`.
    """

    state: TaskStateT
    prompt: str
    sandbox_spec: SandboxSpec
    backend: str | None = None
    clean_sandbox_spec: SandboxSpec | None = None
    collect_resource_metrics: bool = False


def clean_snapshot_compatible(task: HarnessTaskContext, kind: SnapshotKind | None = None) -> bool:
    """
    Return whether the rollout snapshot can seed the task's clean sandbox.

    A ``FILESYSTEM`` snapshot (e.g. a docker commit) is self-contained: the
    committed image captures the clean state regardless of the base image
    identity, so only the resources and mounts must match. A ``FULL_STATE``
    snapshot (microVM) additionally requires the exact same source so the
    restore can rebuild the VM configuration.

    Mounts: docker commit does not capture host bind mounts, so the committed
    image can only seed the grader when the rollout carries no content-bearing
    mounts the grader relies on. The per-sandbox harness tarball mounts are
    excluded — the grader never needs them.
    """
    clean_spec = task.clean_sandbox_spec
    if not (task.sandbox_spec.state_policy.enabled and clean_spec is not None and clean_spec.state_policy.enabled):
        return False
    if not _mounts_compatible(task.sandbox_spec.mounts, clean_spec.mounts):
        return False
    if task.sandbox_spec.resources != clean_spec.resources:
        return False
    if kind != SnapshotKind.FILESYSTEM and task.sandbox_spec.source != clean_spec.source:
        return False
    return True


# Harness tarball bind mounts (node / claude-code / codex) are per-sandbox and
# irrelevant to the grader; docker commit cannot capture them, and that is fine.
_HARNESS_TARBALL_TARGETS = frozenset({"/tmp/node22.tarball", "/tmp/claude-code.tgz", "/tmp/codex.tgz"})


def _mounts_compatible(rollout_mounts, clean_mounts) -> bool:
    """Whether the committed image can satisfy the grader's mount expectations."""
    effective_rollout = [m for m in rollout_mounts if m.target not in _HARNESS_TARBALL_TARGETS]
    return list(effective_rollout) == list(clean_mounts)
