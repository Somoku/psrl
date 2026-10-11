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


class TestTheDecisionIsResolvedFromTheRealConfigKeys:
    """`PSManager` receives only `config.psrl`, so it cannot see the backend or the
    dtypes. The decision is therefore made in `validate_config`, which can -- and if it
    reads the wrong key it silently resolves to the wrong answer, because a `get` with a
    default never raises.

    The keys were got wrong on the first attempt: `actor_rollout_ref.model.dtype` does not
    exist, and the training dtype actually lives under `actor.megatron.dtype` with the
    backend. A defaulted lookup would have reported `bfloat16` for every run.
    """

    def _config(self, strategy, *, megatron_dtype="bfloat16", quantization=None,
                colocate=True, override="auto"):
        from omegaconf import OmegaConf

        return OmegaConf.create({
            "train_actor_rollout_ref": {"actor": {"strategy": strategy}},
            "actor_rollout_ref": {
                "actor": {"megatron": {"dtype": megatron_dtype}},
                "rollout": {"quantization": quantization},
            },
            "psrl": {
                "colocate_validate_and_train": colocate,
                "nixl": {"await_readers_before_publish": override},
            },
        })

    def _resolve(self, config):
        from psrl.utils.config import _resolve_version_barrier

        _resolve_version_barrier(config)
        return config.psrl.nixl.await_readers_before_publish

    @pytest.mark.parametrize(
        ("strategy", "colocate", "expected"),
        [
            # Required: no master parameters, so the whole step writes live weights.
            ("fsdp2", False, True),
            # Wanted: the window is only the write-back instant, but narrow is not zero.
            ("megatron", False, True),
            ("fsdp_hybrid", False, True),
            # Structurally safe: rollout is paused, so no read can be in flight.
            ("fsdp2", True, False),
            # Not a backend we gate on.
            ("fsdp", False, False),
        ],
    )
    def test_the_backend_and_placement_decide_it(self, strategy, colocate, expected):
        config = self._config(strategy, colocate=colocate)

        assert self._resolve(config) is expected

    def test_a_quantised_rollout_needs_no_barrier(self):
        """It reads a separate quantised buffer, so there is no shared memory to protect
        -- regardless of the training backend."""
        config = self._config("fsdp2", quantization="fp8", colocate=False)

        assert self._resolve(config) is False

    def test_an_explicit_setting_beats_the_derivation(self):
        """Which the E1 criterion depends on: turning the barrier *off* deliberately is
        the only way to observe the corruption it prevents."""
        assert self._resolve(self._config("fsdp2", colocate=False, override=False)) is False
        assert self._resolve(self._config("megatron", override=True)) is True

    def test_the_default_is_auto_rather_than_false(self):
        """A hard `False` default would be indistinguishable from an explicit opt-out, so
        `isinstance(..., bool)` -- the check that makes an explicit setting win -- would
        treat every run as opted out."""
        from pathlib import Path

        text = (
            Path(__file__).resolve().parents[2] / "psrl/trainer/config/psrl/nixl.yaml"
        ).read_text()

        assert "await_readers_before_publish: auto" in text

    def test_the_manager_reads_the_resolved_value_not_the_config(self):
        """It has no access to the backend, so deriving it there would raise or default."""
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[2] / "psrl/workers/ps/ps_manager.py"
        ).read_text()
        start = source.index("def push_model_state_dict_nixl")
        body = source[start : source.index("\n    def ", start + 10)]

        assert "await_readers_before_publish" in body
        assert "train_actor_rollout_ref" not in body, (
            "the manager cannot reach that key: it is constructed with config.psrl only"
        )


class TestTheReaderAnnouncesItsReads:
    """The registry is only as good as what populates it, and the reader side is where a
    count can get stuck: an announcement with no matching end makes every later publish
    wait out its whole budget, with nothing in the logs to explain it.

    Structural rather than behavioural because the surrounding method is a Ray actor body
    driving vLLM -- it cannot be constructed here. What is checkable in source is the part
    that actually went wrong in the earlier handshake work: whether the two calls line up.
    """

    def _source(self) -> str:
        from pathlib import Path

        return (
            Path(__file__).resolve().parents[2] / "psrl/workers/gen/vllm_async_server.py"
        ).read_text()

    def test_the_end_is_inside_the_finally_block(self):
        """The pull can raise -- a timeout, a dead source, a cancelled rollout. If the end
        is not in `finally`, that read stays in flight for the life of the process.

        Checked by indentation, not by the presence of the word `finally:`. This method
        has more than one -- `self._pull_commit_deferred = False` is in a `finally` of its
        own -- so a substring search passes with the `end_read` moved out of the block,
        which I confirmed it did before writing this version.
        """
        lines = self._source().splitlines()
        begin_at = next(i for i, line in enumerate(lines) if "begin_read.remote" in line)
        end_at = next(i for i, line in enumerate(lines) if "end_read.remote" in line)

        # The `try:` this read is guarded by, and the `finally:` that closes it.
        try_at = max(i for i in range(begin_at) if lines[i].strip() == "try:")
        finally_at = next(i for i in range(try_at, end_at) if lines[i].strip() == "finally:")

        body_indent = len(lines[finally_at]) - len(lines[finally_at].lstrip())
        end_indent = len(lines[end_at]) - len(lines[end_at].lstrip())

        assert end_indent > body_indent, (
            f"end_read at line {end_at + 1} is dedented to {end_indent} spaces, level with "
            f"or above the `finally:` at line {finally_at + 1} ({body_indent} spaces) -- so "
            f"it runs only when the pull succeeds, and an exception leaves the "
            f"read in flight forever"
        )

    def test_both_calls_name_the_same_reader(self):
        """`end_read` decrements by identity. A different id leaves the original entry in
        place, so the count never returns to zero."""
        source = self._source()
        start = source.index("await ps_manager_handle.begin_read.remote")
        end = source.index("await ps_manager_handle.end_read.remote") + 200

        assert "reader_id" in source[start:end]

    def test_the_version_reported_is_the_one_being_left_behind(self):
        """Which is what a publisher is about to overwrite. Reporting the *new* version is
        impossible here -- the pull assigns it -- and would also point the barrier at the
        wrong memory."""
        source = self._source()
        start = source.index("await ps_manager_handle.begin_read.remote")

        assert "curr_rollout_instance_model_version" in source[max(0, start - 500) : start]
