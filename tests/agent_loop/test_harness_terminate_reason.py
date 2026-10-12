"""Terminate-reason classification for the sandboxed harness loops."""

import asyncio
from types import SimpleNamespace

import pytest
from psrl.workers.agent_loop.loops.harness_agent_loop import HarnessAgentLoop, _raise_if_episode_is_over
from psrl.workers.agent_loop.loops.utils import TerminateReason
from psrl.workers.agent_loop.worker import _classify_rollout_failure

pytestmark = pytest.mark.cpu_test

_BUDGET = 65536
_MAX_TURNS = 80


def _reason(
    training_data: list[dict],
    *,
    grader_unavailable: bool = False,
    grader_capacity_timeout: bool = False,
    agent_timed_out: bool = False,
) -> TerminateReason:
    loop = SimpleNamespace(
        rollout_budget=_BUDGET,
        max_turns=_MAX_TURNS,
        grader_unavailable=grader_unavailable,
        grader_capacity_timeout=grader_capacity_timeout,
        _agent_timed_out=agent_timed_out,
    )
    return HarnessAgentLoop.get_harness_terminate_reason(loop, training_data)


def _item(prompt_tokens: int, response_tokens: int, num_turns: int) -> dict:
    return {
        "prompt_ids": list(range(prompt_tokens)),
        "response_ids": list(range(response_tokens)),
        "num_turns": num_turns,
    }


def test_finished_episode_may_exceed_response_length_without_fitting_budget() -> None:
    # A finished episode that exceeded `response_length` but still fits the
    # prompt plus response budget must not count as response-length limited.
    data = [_item(prompt_tokens=1000, response_tokens=40000, num_turns=10)]

    assert _reason(data) is TerminateReason.FINISHED


def test_response_at_exactly_response_length_is_finished() -> None:
    # The old `>= response_length` test mislabeled every episode that happened to
    # land exactly on the configured response length.
    data = [_item(prompt_tokens=1000, response_tokens=32768, num_turns=12)]

    assert _reason(data) is TerminateReason.FINISHED


def test_response_length_exceeded_reports_when_the_budget_overflows() -> None:
    data = [_item(prompt_tokens=30000, response_tokens=40000, num_turns=20)]

    assert _reason(data) is TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED


def test_budget_overflow_wins_over_the_turn_cap() -> None:
    data = [_item(prompt_tokens=_BUDGET, response_tokens=1, num_turns=_MAX_TURNS)]

    assert _reason(data) is TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED


def test_turn_cap_reports_when_the_budget_fits() -> None:
    data = [_item(prompt_tokens=1000, response_tokens=1000, num_turns=_MAX_TURNS)]

    assert _reason(data) is TerminateReason.MAX_TURNS_EXCEEDED


def test_an_unlimited_episode_finishes_instead_of_exceeding_turns() -> None:
    # With no turn cap the episode is bounded by the trajectory time budget, so a
    # high turn count is finished, not exhausted.
    loop = SimpleNamespace(
        rollout_budget=_BUDGET,
        max_turns=None,
        grader_unavailable=False,
        grader_capacity_timeout=False,
        _agent_timed_out=False,
    )
    data = [_item(prompt_tokens=1000, response_tokens=1000, num_turns=10_000)]

    assert HarnessAgentLoop.get_harness_terminate_reason(loop, data) is TerminateReason.FINISHED


def _limit_loop(kind: str, *, supports_turn_limit: bool) -> SimpleNamespace:
    return SimpleNamespace(
        harness_config=SimpleNamespace(kind=kind, supports_turn_limit=supports_turn_limit),
        _turn_limit_warned=False,
    )


@pytest.mark.parametrize("configured", [None, -1, 0, -5])
def test_a_non_positive_turn_budget_means_unlimited(configured) -> None:
    loop = _limit_loop("claude_code", supports_turn_limit=True)

    assert HarnessAgentLoop._resolve_turn_limit(loop, configured) is None


def test_a_positive_turn_budget_is_a_cap() -> None:
    loop = _limit_loop("claude_code", supports_turn_limit=True)

    assert HarnessAgentLoop._resolve_turn_limit(loop, 80) == 80
    # The supported harness does not warn about a cap it honours.
    assert loop._turn_limit_warned is False


def test_a_turn_cap_on_a_harness_that_ignores_it_warns_once(caplog) -> None:
    import logging

    loop = _limit_loop("codex", supports_turn_limit=False)

    with caplog.at_level(logging.WARNING):
        assert HarnessAgentLoop._resolve_turn_limit(loop, 80) == 80

    warnings = [r for r in caplog.records if "takes no turn cap" in r.getMessage()]
    assert len(warnings) == 1, "a cap the harness ignores must be reported"
    assert loop._turn_limit_warned is True

    # A second request does not repeat the warning.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        HarnessAgentLoop._resolve_turn_limit(loop, 80)
    assert not [r for r in caplog.records if "takes no turn cap" in r.getMessage()]


def test_an_unset_turn_budget_on_a_codex_loop_does_not_warn(caplog) -> None:
    # Unlimited is the intended default for codex, not a misconfiguration.
    import logging

    loop = _limit_loop("codex", supports_turn_limit=False)

    with caplog.at_level(logging.WARNING):
        assert HarnessAgentLoop._resolve_turn_limit(loop, None) is None

    assert not [r for r in caplog.records if "takes no turn cap" in r.getMessage()]


def test_any_trajectory_over_budget_marks_the_request_truncated() -> None:
    # A single TITO branch that overflows the budget forces the loop to cut that
    # branch, so the whole request is reported as truncated.
    data = [
        _item(prompt_tokens=1000, response_tokens=1000, num_turns=5),
        _item(prompt_tokens=40000, response_tokens=40000, num_turns=5),
    ]

    assert _reason(data) is TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED


def test_an_ungraded_episode_reports_verifier_error() -> None:
    # The trajectory is complete, but no grader ever scored it. Reporting a budget
    # reason would train the absence of a measurement, and reporting FINISHED would
    # train a reward that defaults to zero.
    data = [_item(prompt_tokens=1000, response_tokens=1000, num_turns=10)]

    assert _reason(data, grader_unavailable=True) is TerminateReason.VERIFIER_ERROR
    assert _reason(data, grader_unavailable=True) is not TerminateReason.FINISHED


def test_a_grader_that_was_never_admitted_reports_the_capacity_reason() -> None:
    # Also ungraded, but the cause is node capacity rather than a broken grader, and only
    # the narrower reason lets the manager's capacity breaker see grading starvation.
    # Both flags are set together by the task hook, so the narrower one must win.
    data = [_item(prompt_tokens=1000, response_tokens=1000, num_turns=10)]

    reason = _reason(data, grader_unavailable=True, grader_capacity_timeout=True)

    assert reason is TerminateReason.GRADER_CAPACITY_TIMEOUT
    # Ungraded keeps the trajectory and masks its reward; the capacity flag is what
    # additionally routes it to the breaker rather than to the harness-fault streak.
    assert reason.is_ungraded
    assert reason.is_successful
    assert reason.is_coordination_fault
    assert not reason.counts_toward_refill_breaker()


@pytest.mark.parametrize(
    ("action", "keeps"),
    [("discard", False), ("keep", True), (" keep ", True)],
)
def test_the_timeout_action_selects_whether_partial_work_is_kept(action: str, keeps: bool) -> None:
    loop = SimpleNamespace(trajectory_timeout_action=action)

    assert HarnessAgentLoop._resolve_timeout_action(loop) is keeps


def test_an_unset_timeout_action_discards_the_partial_trajectory() -> None:
    # The shipped default preserves the pre-existing behavior: the episode is
    # cancelled and retried, losing the partial work. An absent key and an empty
    # value both mean "use the default", the same way `grading_timeout: null` does.
    assert HarnessAgentLoop._resolve_timeout_action(SimpleNamespace()) is False
    assert HarnessAgentLoop._resolve_timeout_action(SimpleNamespace(trajectory_timeout_action=None)) is False
    assert HarnessAgentLoop._resolve_timeout_action(SimpleNamespace(trajectory_timeout_action="")) is False


@pytest.mark.parametrize("action", ["halt", "KEEP", "none"])
def test_an_unknown_timeout_action_is_refused(action: str) -> None:
    loop = SimpleNamespace(trajectory_timeout_action=action)

    with pytest.raises(ValueError, match="trajectory_timeout_action"):
        HarnessAgentLoop._resolve_timeout_action(loop)


def test_an_episode_whose_deadline_stopped_the_agent_is_a_kept_success() -> None:
    """`keep` stops the agent at the deadline and grades the partial trajectory.

    It must report `AGENT_TIMEOUT` -- a success kept for training -- rather than the
    `TRAJECTORY_TIMEOUT` that discards and retries, so the episode's own turns are not
    thrown away and its reward is not read as a failure.
    """
    data = [_item(prompt_tokens=1000, response_tokens=1000, num_turns=10)]

    reason = _reason(data, agent_timed_out=True)

    assert reason is TerminateReason.AGENT_TIMEOUT
    assert reason.is_successful
    assert reason.is_timeout is False
    assert reason.is_error is False
    assert not reason.needs_worker_retry()
    assert not reason.needs_manager_retry()
    # A deadline is an infrastructure cutoff, not budget exhaustion, so it must not be
    # gated by the truncation mask that penalises a genuinely over-budget episode.
    assert reason.is_budget_truncated is False


def test_a_deadline_stopped_agent_is_reported_before_a_turn_cap() -> None:
    # The episode was cut off by the wall clock, so whatever turn count it reached is
    # incidental and must not be reported as a turn-cap exhaustion.
    data = [_item(prompt_tokens=1000, response_tokens=1000, num_turns=_MAX_TURNS)]

    assert _reason(data, agent_timed_out=True) is TerminateReason.AGENT_TIMEOUT


def _agent_config(action: str) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            gen_actor_rollout_ref=SimpleNamespace(
                rollout=SimpleNamespace(
                    agent=SimpleNamespace(trajectory_timeout_action=action, traj_reward_mode="traj"),
                    multi_turn=SimpleNamespace(enable=True),
                )
            )
        )
    )


class _ConcreteHarnessLoop(HarnessAgentLoop):
    """A concrete subclass so the abstract `HarnessAgentLoop` can be instantiated."""

    async def prepare_harness_task(self, request):  # pragma: no cover - never called
        raise NotImplementedError


@pytest.mark.parametrize(("action", "owns_budget"), [("discard", False), ("keep", True)])
def test_keep_mode_makes_the_loop_own_the_episode_budget(monkeypatch, action, owns_budget) -> None:
    """`keep` moves the deadline into the loop, so the base must stop enforcing it.

    This is the wiring that lets the loop stop its own agent work at the deadline and
    carry on to collection and grading, instead of the base cancelling the episode. It
    is set from the same switch as `keeps_partial_on_timeout`, so the two cannot drift.
    """
    from psrl.workers.agent_loop.loops import session_agent_loop

    def _stub_init(self, context):
        # Skip the heavy framework wiring `SessionAgentLoop.__init__` performs; the
        # only value `HarnessAgentLoop.__init__` reads back is the strategy.
        self.trajectory_id_strategy = "auto"

    monkeypatch.setattr(session_agent_loop.SessionAgentLoop, "__init__", _stub_init)

    loop = _ConcreteHarnessLoop(
        context=_agent_config(action),
        harness={"kind": "codex", "executable": "codex"},
    )

    assert loop.keeps_partial_on_timeout is owns_budget
    assert loop.episode_budget_enforced_by_loop is owns_budget


@pytest.mark.asyncio
async def test_a_cancelled_episode_stops_before_it_collects_artifacts() -> None:
    """An aborted episode must not spend a sandbox collecting a result nobody reads.

    The manager clears the buffer entry when it aborts a group, so the phase after the
    harness run -- patch collection, grading, reading the TITO trajectory -- has nowhere
    to put its output. Worse, it runs against a sandbox whose teardown is already in
    flight. A cancellation delivered inside a shielded section is only recorded on the
    task, so the loop has to check for it rather than wait for the next await to raise.
    """
    observed: list[str] = []

    async def episode() -> None:
        try:
            await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            # Stands in for the cleanup paths that legitimately absorb a cancellation.
            observed.append("cancel absorbed")
        _raise_if_episode_is_over(None)
        observed.append("collected artifacts")

    task = asyncio.create_task(episode())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert observed == ["cancel absorbed"], "collection ran after the episode was cancelled"


def test_a_cancelled_episode_is_a_coordination_fault_not_a_rollout_error() -> None:
    # What `_raise_if_episode_is_over` raises has to land on the reason that keeps an abort out
    # of the breaker built to catch a broken harness.
    reason = _classify_rollout_failure(asyncio.CancelledError())

    assert reason is TerminateReason.ROLLOUT_CANCELLED
    assert reason.is_coordination_fault
    assert not reason.counts_toward_refill_breaker()
    assert not reason.needs_worker_retry()


def test_an_externally_released_lease_ends_the_episode_before_collection() -> None:
    """The lease can be released by something this episode never asked.

    An aborted group releases the lease of a sibling that is still running. That destroys
    the sandbox while this episode's own task was never cancelled, so `cancelling()` stays
    zero and collection would proceed against a sandbox that is already gone -- which is
    how a released lease reported itself as a bare "sandbox is terminated" from patch
    collection rather than as the coordination fault it is.
    """
    released = SimpleNamespace(released=True, ref="docker/abc")
    still_held = SimpleNamespace(released=False, ref="docker/def")

    with pytest.raises(asyncio.CancelledError):
        _raise_if_episode_is_over(released)

    # A lease this episode still owns must not be mistaken for a lost one.
    _raise_if_episode_is_over(still_held)
    _raise_if_episode_is_over(None)
