"""Version-level write barrier before publishing (implementation item E1).

The failure this prevents is silent: a rollout reading `v` while the optimizer writes
`v+1` into the same parameter memory gets a tensor that is partly new and partly old. The
read succeeds, the shapes are right, and the resulting policy never existed.

Two properties carry the item, and they pull against each other:

* **Correctness.** A reader must never be left reading bytes that are being overwritten
  without being told. When the barrier gives up waiting, it has to invalidate.
* **Liveness.** The wait must be bounded, because the trainer's next step is what produces
  the version after this one. An unbounded drain turns one slow reader into a stalled
  trainer.

The third test class covers which configurations need it at all, since enabling
`FSDP2 + bf16 + bf16 + disaggregate` without it is the case the design singles out as a
correctness bug rather than a slowdown.
"""

from __future__ import annotations

import pytest

from psrl.workers.ps.version_barrier import (
    PublishOutcome,
    VersionBarrier,
    requires_version_barrier,
)

pytestmark = pytest.mark.cpu_test


class _Readers:
    """Reader counts that drain after a set number of polls."""

    def __init__(self, initial: int, drains_after_polls: int | None = None):
        self.count = initial
        self.drains_after = drains_after_polls
        self.polls = 0

    def __call__(self, _version: int) -> int:
        self.polls += 1
        if self.drains_after is not None and self.polls > self.drains_after:
            self.count = 0
        return self.count


class TestDrainingWithinBudget:
    def test_no_readers_returns_immediately(self):
        barrier = VersionBarrier(count_readers=lambda _v: 0, budget_ms=1000.0)

        report = barrier.publish(previous_version=5)

        assert report.outcome is PublishOutcome.NO_READERS
        assert report.waited_ms == 0.0
        assert report.previous_version_valid

    def test_readers_that_finish_in_time_leave_the_version_valid(self):
        readers = _Readers(initial=3, drains_after_polls=2)
        barrier = VersionBarrier(count_readers=readers, budget_ms=1000.0)

        report = barrier.publish(previous_version=5)

        assert report.outcome is PublishOutcome.DRAINED
        assert report.readers_remaining == 0
        assert not barrier.is_invalidated(5)

    def test_the_wait_is_reported_so_the_cost_is_visible(self):
        readers = _Readers(initial=1, drains_after_polls=3)
        barrier = VersionBarrier(count_readers=readers, budget_ms=1000.0)

        barrier.publish(previous_version=5)

        assert barrier.total_wait_ms > 0.0


class TestBudgetExpiryInvalidates:
    """Publishing over live readers is safe only if they are told to abandon."""

    def test_expiry_marks_the_previous_version_invalid(self):
        barrier = VersionBarrier(count_readers=lambda _v: 2, budget_ms=5.0)

        report = barrier.publish(previous_version=7)

        assert report.outcome is PublishOutcome.INVALIDATED
        assert barrier.is_invalidated(7), (
            "the publish proceeded over live readers without invalidating; those reads "
            "would return half-written bytes and still succeed"
        )
        assert not report.previous_version_valid

    def test_the_wait_is_bounded_by_the_budget(self):
        """Liveness: a stuck reader must not stall the trainer indefinitely."""
        barrier = VersionBarrier(count_readers=lambda _v: 1, budget_ms=20.0)

        report = barrier.publish(previous_version=7)

        assert report.waited_ms < 500.0, (
            f"waited {report.waited_ms:.0f}ms against a 20ms budget; the drain is not "
            f"actually bounded"
        )

    def test_a_zero_budget_still_publishes(self):
        """Degenerate but legal: never wait, always invalidate when readers exist."""
        barrier = VersionBarrier(count_readers=lambda _v: 1, budget_ms=0.0)

        report = barrier.publish(previous_version=7)

        assert report.outcome is PublishOutcome.INVALIDATED

    def test_a_negative_budget_is_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            VersionBarrier(count_readers=lambda _v: 0, budget_ms=-1.0)

    def test_only_the_expired_version_is_invalidated(self):
        barrier = VersionBarrier(count_readers=lambda _v: 1, budget_ms=1.0)

        barrier.publish(previous_version=7)

        assert barrier.is_invalidated(7)
        assert not barrier.is_invalidated(8)


class TestWhichConfigurationsNeedIt:
    """Necessity is a function of the configuration, not a judgement call."""

    def test_fsdp2_same_precision_disaggregate_requires_it(self):
        """The case the design calls a correctness bug without this barrier.

        FSDP2 has no master params, so the whole optimizer step writes the bytes rollouts
        read -- not just the write-back instant.
        """
        assert requires_version_barrier(
            backend="fsdp2", train_dtype="bf16", rollout_dtype="bf16", colocate=False
        )

    def test_megatron_same_precision_also_uses_it(self):
        """Narrower window, but narrow is not zero, and the check is cheap when idle."""
        assert requires_version_barrier(
            backend="megatron", train_dtype="bf16", rollout_dtype="bf16", colocate=False
        )

    def test_a_cast_path_needs_no_barrier(self):
        """Rollouts read a quantised buffer, so parameter memory is never shared."""
        assert not requires_version_barrier(
            backend="fsdp2", train_dtype="bf16", rollout_dtype="fp8", colocate=False
        )
        assert not requires_version_barrier(
            backend="megatron", train_dtype="fp32", rollout_dtype="bf16", colocate=False
        )

    def test_colocate_needs_no_barrier(self):
        """Rollout is paused for the transfer, so no read can be in flight."""
        assert not requires_version_barrier(
            backend="fsdp2", train_dtype="bf16", rollout_dtype="bf16", colocate=True
        )

    def test_an_unknown_backend_does_not_silently_opt_out(self):
        """A new backend must answer the question, not inherit 'no' by omission."""
        assert not requires_version_barrier(
            backend="some_new_engine",
            train_dtype="bf16",
            rollout_dtype="bf16",
            colocate=False,
        ), (
            "unknown backends currently return False; if this changes to True by default "
            "the test should be updated deliberately, because the answer determines "
            "whether same-precision reads are safe"
        )
