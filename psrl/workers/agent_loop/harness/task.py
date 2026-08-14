"""Task contract consumed by the generic harness agent loop."""

from dataclasses import dataclass
from typing import Generic, TypeVar

from psrl.sandbox import SandboxSpec

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


def clean_snapshot_compatible(task: HarnessTaskContext) -> bool:
    """
    Return whether the rollout snapshot can seed the task's clean sandbox.
    """
    clean_spec = task.clean_sandbox_spec
    return (
        task.sandbox_spec.state_policy.enabled
        and clean_spec is not None
        and clean_spec.state_policy.enabled
        and task.sandbox_spec.source == clean_spec.source
        and task.sandbox_spec.resources == clean_spec.resources
    )
