"""Train-step boundary hooks for weight publication (implementation item C3).

The design asks for version publication to be driven by a train-step-end callback, so
that publishing happens "once the weights are in place and the NIC is idle" rather than
at an arbitrary point chosen by the caller.

**The callback the design names is not reachable from PSRL.** Megatron-Bridge's
``on_train_step_end`` (``training/callbacks.py``, fired at ``training/train.py``) exists
and fires after the optimizer step, exactly as documented -- but it belongs to
Bridge's own ``pretrain``/``train`` loop, and PSRL never enters it. PSRL uses
Megatron-Bridge for model *conversion* only (``AutoBridge`` in ``megatron_converter``);
the RL optimizer step runs through verl's ``train_mini_batch``. Registering a Bridge
callback would compile, run, and silently never fire.

So the step boundary is defined here, on the loop PSRL actually runs. The ordering
requirement is already met structurally -- ``update_actor`` calls ``push_model``
immediately after the optimizer step returns -- and what was missing is that the moment
was not *observable*, which is precisely what the design's criterion asks to verify.
This module makes it observable and gives the publication point a name, so a future
change that moves publication away from the step boundary is visible rather than silent.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

#: Hook signature: receives the boundary record, returns nothing.
StepEndHook = Callable[["StepBoundary"], None]


@dataclass(frozen=True)
class StepBoundary:
    """When a train step ended and what was published at that boundary."""

    #: Monotonic clock, for ordering against other trace events.
    ended_at: float
    #: Wall clock, so a boundary can be matched against external traces.
    wall_time: float
    step: int | None = None
    version: int | None = None

    def ordered_before(self, other: "StepBoundary") -> bool:
        return self.ended_at < other.ended_at


@dataclass
class StepBoundaryRecorder:
    """Records step boundaries and fans out to registered hooks.

    Bounded by construction: `history` keeps only the most recent `capacity` entries,
    because this lives in a long-running trainer and an unbounded list would be a slow
    leak rather than a visible failure.

    A hook that raises must not take the training step down with it -- publication is
    observability here, not the transfer itself -- so exceptions are collected and
    reported rather than propagated.
    """

    capacity: int = 64
    hooks: list[StepEndHook] = field(default_factory=list)
    history: list[StepBoundary] = field(default_factory=list)
    hook_errors: list[BaseException] = field(default_factory=list)

    def register(self, hook: StepEndHook) -> None:
        self.hooks.append(hook)

    def on_train_step_end(self, step: int | None = None, version: int | None = None) -> StepBoundary:
        """Record a step boundary and run every hook; return the recorded boundary."""
        boundary = StepBoundary(
            ended_at=time.monotonic(),
            wall_time=time.time(),
            step=step,
            version=version,
        )
        self.history.append(boundary)
        if len(self.history) > self.capacity:
            del self.history[: len(self.history) - self.capacity]

        for hook in self.hooks:
            try:
                hook(boundary)
            except BaseException as error:  # noqa: BLE001 - deliberately collected
                self.hook_errors.append(error)
        return boundary

    @property
    def last(self) -> StepBoundary | None:
        return self.history[-1] if self.history else None

    def versions(self) -> list[int]:
        """Published versions in boundary order, skipping boundaries that published none."""
        return [b.version for b in self.history if b.version is not None]

    def assert_versions_advance(self) -> None:
        """Raise unless published versions strictly increase.

        A repeated or regressing version means two different weight states were
        published under one name, which readers cannot detect: they would accept stale
        bytes as current. Checking it at the boundary catches it where the cause is,
        rather than as a correctness puzzle on the rollout side.
        """
        versions = self.versions()
        for earlier, later in zip(versions, versions[1:]):
            if later <= earlier:
                raise AssertionError(
                    f"published versions must strictly increase, got {earlier} then {later}; "
                    "two weight states would share one version"
                )
