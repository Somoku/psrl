"""Terminate-reason classification for the sandboxed harness loops."""

from types import SimpleNamespace

import pytest
from psrl.workers.agent_loop.loops.harness_agent_loop import HarnessAgentLoop
from psrl.workers.agent_loop.loops.utils import TerminateReason

pytestmark = pytest.mark.cpu_test

_BUDGET = 65536
_MAX_TURNS = 80


def _reason(
    training_data: list[dict],
    *,
    grader_unavailable: bool = False,
    grader_capacity_timeout: bool = False,
) -> TerminateReason:
    loop = SimpleNamespace(
        rollout_budget=_BUDGET,
        max_turns=_MAX_TURNS,
        grader_unavailable=grader_unavailable,
        grader_capacity_timeout=grader_capacity_timeout,
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
