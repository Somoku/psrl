"""Bounding how long a publish waits for readers of the previous version (item E1).

The optimizer writes parameter memory in place. A rollout reading version ``v`` while the
trainer writes ``v+1`` into the same bytes gets a tensor that is partly new and partly
old -- a policy that never existed, which is worse than staleness because no correction
term accounts for it. The fix is to drain readers of ``v`` before publishing ``v+1``.

**Why this cannot simply block.** Draining is unbounded: a slow reader stalls the trainer,
and the trainer's next forward pass is what produces ``v+2``. So the wait carries a budget,
and on expiry the publish proceeds while ``v`` is marked invalid. Readers still on ``v``
abandon and re-pull ``v+1`` -- strictly newer weights, which the algorithm is happy with.
Correctness is preserved by invalidation, not by waiting.

**Necessity depends on the backend, and the difference is not cosmetic** (design 3.8.0a):

* **Megatron, same precision.** Master params are separate, so the conflict window is the
  instant of parameter write-back. The barrier is an optimisation.
* **FSDP2 + `model_dtype=bf16`, same precision.** `build_optimizer(module.parameters())`
  means no master params: the *entire* optimizer step writes the bytes rollouts read.
  The barrier is **required**, and the design blocks enabling that configuration without
  it.
* **Any cast path.** The wire bytes are a separate quantised buffer, so rollouts never
  read parameter memory and the wait is structurally zero -- not "usually small".

`requires_version_barrier` encodes that as a function of the configuration rather than a
comment, so a new backend has to answer the question explicitly.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum


class PublishOutcome(Enum):
    """How a publish resolved."""

    #: Readers drained within budget; the previous version stays valid.
    DRAINED = "drained"
    #: Budget expired; published anyway and invalidated the previous version.
    INVALIDATED = "invalidated"
    #: Nothing was reading, so there was nothing to wait for.
    NO_READERS = "no_readers"


@dataclass(frozen=True)
class PublishReport:
    outcome: PublishOutcome
    waited_ms: float
    readers_remaining: int

    @property
    def previous_version_valid(self) -> bool:
        return self.outcome is not PublishOutcome.INVALIDATED


def requires_version_barrier(
    *,
    backend: str,
    train_dtype: str,
    rollout_dtype: str,
    colocate: bool,
) -> bool:
    """Whether publishing must drain readers before overwriting parameter memory.

    False for a cast path at any backend: the rollout reads a quantised buffer, not the
    parameters, so there is no shared memory to protect. False under colocate, where
    rollout is paused for the duration and no read can be in flight. True for FSDP2 at
    matching precision, where the whole optimizer step is the conflict window.

    Megatron at matching precision returns True as well, even though its window is only
    the write-back instant. Narrow is not zero, and the cost when nothing is reading is a
    counter check -- cheaper than reasoning about whether this particular step is safe.
    """
    if colocate:
        return False
    if train_dtype != rollout_dtype:
        return False
    return backend in ("fsdp2", "fsdp_hybrid", "megatron")


#: How many expired versions `invalidated_versions` keeps. A reader further behind than
#: this re-pulls the current version regardless, so the bound is about memory.
_INVALIDATED_HISTORY = 64


@dataclass
class VersionBarrier:
    """Drains readers of the previous version, within a budget, before publishing.

    ``count_readers`` returns how many readers are currently in flight for a version.
    Injected rather than reached for so the policy can be tested without a manager.

    The invalidation set is a signal, not bookkeeping. A version enters it only when the
    publish deadline expired with readers still in flight, and
    `PSManager.get_nixl_version_state` is what makes it reachable from the reader side.
    """

    count_readers: Callable[[int], int]
    budget_ms: float = 50.0
    poll_interval_s: float = 0.0005
    _invalidated: set[int] = field(default_factory=set)
    #: Insertion order, so pruning drops the oldest rather than an arbitrary entry.
    _invalidated_order: list[int] = field(default_factory=list)
    #: Reports in order, for assertions and for reporting wait time as a metric.
    history: list[PublishReport] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.budget_ms < 0:
            raise ValueError(f"budget_ms must be non-negative, got {self.budget_ms}")

    def is_invalidated(self, version: int) -> bool:
        return version in self._invalidated

    def invalidated_versions(self) -> frozenset[int]:
        """Every version whose bytes may have been overwritten under a live reader.

        Bounded to the most recent `_INVALIDATED_HISTORY` versions. A reader further
        behind than that re-pulls the current version anyway.
        """
        return frozenset(self._invalidated)

    def _record_invalidation(self, version: int) -> None:
        if version in self._invalidated:
            return
        self._invalidated.add(version)
        self._invalidated_order.append(version)
        while len(self._invalidated_order) > _INVALIDATED_HISTORY:
            self._invalidated.discard(self._invalidated_order.pop(0))

    def publish(self, previous_version: int) -> PublishReport:
        """Wait for readers of ``previous_version``, then allow the publish.

        Always returns -- the budget guarantees it. The caller publishes regardless; what
        the report decides is whether the previous version remains usable by readers that
        were mid-pull.
        """
        start = time.monotonic()
        remaining = self.count_readers(previous_version)
        if remaining == 0:
            report = PublishReport(PublishOutcome.NO_READERS, 0.0, 0)
            self.history.append(report)
            return report

        budget_s = self.budget_ms / 1000.0
        while remaining > 0 and (time.monotonic() - start) < budget_s:
            time.sleep(self.poll_interval_s)
            remaining = self.count_readers(previous_version)

        waited_ms = (time.monotonic() - start) * 1000.0
        if remaining > 0:
            # Publishing over live readers is safe only because they are told to abandon.
            self._record_invalidation(previous_version)
            outcome = PublishOutcome.INVALIDATED
        else:
            outcome = PublishOutcome.DRAINED

        report = PublishReport(outcome, waited_ms, remaining)
        self.history.append(report)
        return report

    @property
    def total_wait_ms(self) -> float:
        """Cumulative wait, which is the cost side of the trade-off being made."""
        return sum(report.waited_ms for report in self.history)
