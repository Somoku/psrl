"""The harness CLI's own episode report, and where its numbers are used.

Two things are pinned here. The stop label comes from the harness's own verdict
rather than from a count the framework reconstructed, and a trajectory's logged turn
count is that trajectory's own rather than a single episode-wide value broadcast to
all of them.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
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


def test_every_trajectory_keeps_its_own_turn_count() -> None:
    """A rewrite or a subagent leaves one episode's trajectories at different lengths.

    A single value broadcast to all of them would be reported as each one's own count.
    """
    outputs = [SimpleNamespace(agent_reward_info={}), SimpleNamespace(agent_reward_info={})]
    training_data = [_item(num_turns=3), _item(num_turns=41)]

    HarnessAgentLoop._stamp_reward_info(outputs, training_data, {"episode_num_turns": 47})

    assert [output.agent_reward_info["num_turns"] for output in outputs] == [3, 41]
    assert outputs[0].agent_reward_info["episode_num_turns"] == 47


def test_episode_diagnostics_carry_the_harness_count_under_its_own_name() -> None:
    loop = SimpleNamespace(
        _episode_summary=HarnessEpisodeSummary(stop_reason=None, num_turns=47),
        compaction_budget=None,
        harness_config=SimpleNamespace(kind="claude_code"),
    )

    info = HarnessAgentLoop._build_partial_reward_info(loop, [_item(3), _item(41)], HarnessResult(exit_code=0))

    assert info["episode_num_turns"] == 47
    # The shared diagnostics must not claim a turn count: it would be stamped onto every
    # trajectory as if the episode's count were each trajectory's own.
    assert "num_turns" not in info


def test_the_summary_prints_the_local_count_and_the_episode_count_separately() -> None:
    loop = SimpleNamespace()
    out = SimpleNamespace(
        agent_reward_info={"patch": "", "num_turns": 12, "episode_num_turns": 47},
        extra_fields={},
        prompt_ids=[1, 2, 3],
        response_ids=[4],
        response_mask=[1],
        num_turns=12,
    )

    text = AgentLoopBase._build_summary_text(loop, out, TerminateReason.FINISHED)

    assert "turns: 12" in text
    assert "episode_turns: 47" in text
