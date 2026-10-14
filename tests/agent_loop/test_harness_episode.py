"""The harness CLI's own episode report, and where its numbers are used.

Two things are pinned here. The stop label comes from the harness's own verdict
rather than from a count the framework reconstructed, and a trajectory's logged turn
count is that trajectory's own rather than a single episode-wide value broadcast to
all of them.
"""

from __future__ import annotations

import json
from types import MethodType, SimpleNamespace

import pytest
from psrl.utils.common.http_utils import SESSION_TURN_BUDGET_ERROR_CODE
from psrl.workers.agent_loop.harness import (
    CLAUDE_CODE_STREAM_JSON,
    CODEX_JSONL,
    STOP_REASON_MAX_TURNS,
    HarnessConfig,
    HarnessEpisodeSummary,
    HarnessResult,
    create_harness,
    episode_summary,
)
from psrl.workers.agent_loop.loops.base_agent_loop import AgentLoopBase
from psrl.workers.agent_loop.loops.harness_agent_loop import HarnessAgentLoop
from psrl.workers.agent_loop.loops.utils import TerminateReason
from tests.agent_loop.test_harness import FakeSandbox

pytestmark = pytest.mark.cpu_test


def _events(*objects: dict) -> bytes:
    return b"".join(json.dumps(obj).encode() + b"\n" for obj in objects)


def _result(*, num_turns: int, terminal_reason: str = "completed", subtype: str = "success") -> dict:
    return {"type": "result", "subtype": subtype, "is_error": False, "num_turns": num_turns,
            "terminal_reason": terminal_reason, "stop_reason": "end_turn"}


def test_a_finished_run_reports_its_turn_count() -> None:
    log = _events({"type": "system", "subtype": "init"}, {"type": "assistant"}, _result(num_turns=12))

    summary = episode_summary(CLAUDE_CODE_STREAM_JSON, log)

    assert summary == HarnessEpisodeSummary(stop_reason="completed", num_turns=12)


def test_the_turn_cap_is_reported_from_the_harness_own_verdict() -> None:
    log = _events(_result(num_turns=81, terminal_reason=STOP_REASON_MAX_TURNS, subtype="error_max_turns"))

    summary = episode_summary(CLAUDE_CODE_STREAM_JSON, log)

    assert summary is not None
    assert summary.stop_reason == STOP_REASON_MAX_TURNS
    # A capped run reports the turn it refused to start, so its count is one more than
    # the turns it completed.
    assert summary.num_turns == 80


def test_every_run_of_one_invocation_is_summed() -> None:
    """One `-p` invocation can run more than once in the same session.

    Each run counts from its own start, so the episode's count is their sum. Measured
    against the wire, that sum is the main-agent model calls: a subagent's calls reach
    the same endpoint but are not counted against the parent.
    """
    log = _events(
        {"type": "system", "subtype": "init"},
        {"type": "assistant", "parent_tool_use_id": None},
        {"type": "assistant", "parent_tool_use_id": "toolu_1"},
        _result(num_turns=2),
        {"type": "system", "subtype": "init"},
        {"type": "assistant"},
        _result(num_turns=1),
    )

    summary = episode_summary(CLAUDE_CODE_STREAM_JSON, log)

    assert summary == HarnessEpisodeSummary(stop_reason="completed", num_turns=3)


def test_a_cap_in_any_run_labels_the_episode() -> None:
    log = _events(_result(num_turns=5), _result(num_turns=9, terminal_reason=STOP_REASON_MAX_TURNS))

    summary = episode_summary(CLAUDE_CODE_STREAM_JSON, log)

    assert summary is not None
    assert summary.stop_reason == STOP_REASON_MAX_TURNS
    assert summary.num_turns == 5 + 8


def test_a_log_without_a_result_reports_nothing() -> None:
    # Absent is not zero: a harness that never reported cannot be read as reporting none.
    log = _events({"type": "system", "subtype": "init"}, {"type": "assistant"})

    assert episode_summary(CLAUDE_CODE_STREAM_JSON, log) is None


def test_unparsable_lines_are_not_part_of_the_report() -> None:
    log = (
        b"not json at all\n"
        b'{"type": "result", "subtype": "success", "num_turns": 7, "terminal_reason": "completed"}\n'
        b'{"truncated": \n'
        b"[1, 2, 3]\n"
    )

    summary = episode_summary(CLAUDE_CODE_STREAM_JSON, log)

    assert summary == HarnessEpisodeSummary(stop_reason="completed", num_turns=7)


def test_a_format_without_a_report_returns_nothing() -> None:
    log = _events({"type": "turn.completed"})

    assert episode_summary(CODEX_JSONL, log) is None
    assert episode_summary("plain_text", log) is None


def test_the_harness_dispatches_on_its_trajectory_format() -> None:
    harness = create_harness(HarnessConfig(kind="claude_code", executable="claude"), FakeSandbox())
    log = _events(_result(num_turns=4, terminal_reason=STOP_REASON_MAX_TURNS))

    summary = harness.episode_summary(log)

    assert summary is not None
    assert summary.stop_reason == STOP_REASON_MAX_TURNS


def _item(num_turns: int) -> dict:
    return {"prompt_ids": [1, 2, 3], "response_ids": [4], "num_turns": num_turns}


def _loop(**attributes) -> SimpleNamespace:
    """A stand-in loop whose bound helpers are the real ones.

    These tests call unbound loop methods, so the stand-in has to answer the methods those
    call on it.
    """
    loop = SimpleNamespace(**attributes)
    for helper in ("_episode_num_turns", "_session_main_turns"):
        setattr(loop, helper, MethodType(getattr(HarnessAgentLoop, helper), loop))
    return loop


def _stamped_loop(episode: HarnessEpisodeSummary | None, snapshot: dict | None = None) -> SimpleNamespace:
    return _loop(_episode_summary=episode, session_snapshot=snapshot or {})


def test_the_session_reports_the_harness_count_as_its_own_turn_count() -> None:
    """One session is one episode, so its samples carry the episode's count.

    A trajectory's own count is not the episode's: a rewrite cuts one run into several
    trajectories, and a subagent adds turns the CLI does not count against the parent.
    """
    loop = _stamped_loop(HarnessEpisodeSummary(stop_reason=None, num_turns=47))
    outputs = [SimpleNamespace(agent_reward_info={}, num_turns=0), SimpleNamespace(agent_reward_info={}, num_turns=0)]
    training_data = [_item(num_turns=3), _item(num_turns=41)]

    HarnessAgentLoop._stamp_reward_info(loop, outputs, training_data, {"num_turns": 47})

    assert [output.num_turns for output in outputs] == [47, 47]
    assert [output.agent_reward_info["num_turns"] for output in outputs] == [47, 47]
    # The trajectory's own count stays beside it, for that trajectory's log.
    assert [output.agent_reward_info["trajectory_num_turns"] for output in outputs] == [3, 41]


def test_a_harness_that_reports_nothing_falls_back_to_the_longest_trajectory() -> None:
    loop = _stamped_loop(None)
    training_data = [_item(num_turns=3), _item(num_turns=41)]

    assert HarnessAgentLoop._episode_num_turns(loop, training_data) == 41


def test_episode_diagnostics_report_the_harness_count() -> None:
    loop = _loop(
        _episode_summary=HarnessEpisodeSummary(stop_reason=None, num_turns=47),
        compaction_budget=None,
        harness_config=SimpleNamespace(kind="claude_code"),
    )

    info = HarnessAgentLoop._build_partial_reward_info(loop, [_item(3), _item(41)], HarnessResult(exit_code=0))

    assert info["num_turns"] == 47
    # A trajectory's count is not an episode-level diagnostic: stamping one here would
    # report it as every trajectory's own.
    assert "trajectory_num_turns" not in info


def test_the_summary_prints_the_local_count_and_the_episode_count_separately() -> None:
    loop = SimpleNamespace()
    out = SimpleNamespace(
        agent_reward_info={"patch": "", "num_turns": 47, "trajectory_num_turns": 12},
        extra_fields={},
        prompt_ids=[1, 2, 3],
        response_ids=[4],
        response_mask=[1],
        num_turns=47,
    )

    text = AgentLoopBase._build_summary_text(loop, out, TerminateReason.FINISHED)

    assert "turns: 12" in text
    assert "episode_turns: 47" in text


# --- Codex: what it can and cannot report ---------------------------------------------

BUDGET_CODE = SESSION_TURN_BUDGET_ERROR_CODE


def _codex_budget_error() -> dict:
    """The event Codex emits when the session router refuses its turn."""
    body = json.dumps({"error": {"code": "session_turn_budget_exhausted", "message": "budget spent"}})
    return {"type": "error", "message": body}


def test_codex_reports_a_spent_turn_budget_as_the_turn_cap() -> None:
    """The refusal travels back through the CLI's own output.

    The router's error body is echoed verbatim by the CLI, which is the only channel a
    Codex episode has for saying it was stopped by a cap.
    """
    log = _events({"type": "turn.started"}, _codex_budget_error())

    summary = episode_summary(CODEX_JSONL, log)

    assert summary is not None
    assert summary.stop_reason == STOP_REASON_MAX_TURNS
    # Codex counts turns, not model calls, so it has no count to report.
    assert summary.num_turns is None


def test_codex_reports_a_spent_turn_budget_from_a_failed_turn() -> None:
    log = _events(
        {"type": "error", "message": "stream disconnected"},
        {"type": "turn.failed", "error": {"message": json.dumps({"error": {"code": BUDGET_CODE}})}},
    )

    summary = episode_summary(CODEX_JSONL, log)

    assert summary is not None
    assert summary.stop_reason == STOP_REASON_MAX_TURNS


def test_a_codex_run_that_ended_for_another_reason_reports_nothing() -> None:
    """Only the code this framework emits is translated.

    Codex's own wording is not this module's vocabulary, and inventing a mapping for it
    would label an episode from a guess.
    """
    log = _events(
        {"type": "turn.completed", "usage": {"input_tokens": 1}},
        {"type": "error", "message": "stream disconnected before completion"},
        {"type": "turn.failed", "error": {"message": "stream disconnected before completion"}},
    )

    assert episode_summary(CODEX_JSONL, log) is None


def test_the_budget_code_inside_model_output_is_not_a_stop_reason() -> None:
    """The code is read from the CLI's error events, not from anywhere in the log.

    A log is mostly what the model produced; a conversation that happens to discuss this
    code must not be read as an episode the code stopped.
    """
    log = _events(
        {"type": "item.completed", "item": {"type": "agent_message", "text": f"the code {BUDGET_CODE} means..."}},
        {"type": "turn.completed", "usage": {}},
    )

    assert episode_summary(CODEX_JSONL, log) is None


# --- Which count an episode reports ---------------------------------------------------


def _router_snapshot(main_turns: int | None) -> dict:
    if main_turns is None:
        return {}
    return {
        "psrl_turn_accounting": {
            "max_turns": 8,
            "main_turns": main_turns,
            "subagent_turns": 2,
            "exhausted": False,
        }
    }


def _counting_loop(episode, snapshot) -> SimpleNamespace:
    loop = _loop(
        _episode_summary=episode,
        session_snapshot=snapshot,
        max_turns=None,
        harness_config=SimpleNamespace(kind="codex", cli_takes_turn_cap=False),
    )
    return loop


def test_the_harness_own_count_wins_over_the_session_accounting() -> None:
    loop = _counting_loop(HarnessEpisodeSummary(stop_reason=None, num_turns=90), _router_snapshot(12))
    training_data = [_item(3), _item(41)]

    assert HarnessAgentLoop._episode_num_turns(loop, training_data) == 90


def test_the_session_accounting_is_used_when_the_harness_reports_no_count() -> None:
    """A Codex session's model calls are counted from outside, excluding subagents."""
    loop = _counting_loop(None, _router_snapshot(12))
    training_data = [_item(3), _item(41)]

    assert HarnessAgentLoop._episode_num_turns(loop, training_data) == 12


def test_a_session_without_accounting_falls_back_to_the_longest_trajectory() -> None:
    loop = _counting_loop(None, _router_snapshot(None))
    training_data = [_item(3), _item(41)]

    assert HarnessAgentLoop._episode_num_turns(loop, training_data) == 41
