"""The rollout deadline ladder: one user-facing knob, every internal deadline derived.

These numbers used to be configured independently, and a shipped recipe had the harness
allowed two hours while the manager declared an entry dead after one, so healthy long
episodes were abandoned. The ladder is now the only place they are decided.
"""

import pytest
from psrl.workers.agent_loop.harness import HarnessConfig
from psrl.workers.agent_loop.loops.harness_agent_loop import HarnessAgentLoop
from psrl.workers.agent_loop.timeouts import (
    DEFAULT_EPISODE_TIMEOUT_S,
    AgentLoopTimeouts,
    resolve_agent_loop_timeouts,
)

pytestmark = pytest.mark.cpu_test


def test_the_episode_budget_is_the_only_required_number() -> None:
    ladder = resolve_agent_loop_timeouts(3600, 900)

    assert ladder.episode_timeout_s == 3600
    assert ladder.admission_timeout_s == 900
    # Everything else is derived, and the derivations are ordered.
    assert ladder.setup_timeout_s > ladder.admission_timeout_s
    assert ladder.harness_exec_timeout_s > ladder.episode_timeout_s
    assert ladder.child_deadline_s > ladder.harness_exec_timeout_s
    assert ladder.entry_stall_timeout_s > ladder.heartbeat_interval_s


def test_a_null_episode_budget_selects_the_default_rather_than_no_limit() -> None:
    """A rollout with no bound cannot be told apart from a wedged one."""
    assert resolve_agent_loop_timeouts(None, None).episode_timeout_s == DEFAULT_EPISODE_TIMEOUT_S


def test_the_stall_threshold_is_a_silence_check_not_a_second_episode_budget() -> None:
    """A long episode must not look stalled: the threshold tracks the heartbeat, not the budget."""
    short = resolve_agent_loop_timeouts(600, None)
    long = resolve_agent_loop_timeouts(7200, None)

    assert short.entry_stall_timeout_s < long.entry_stall_timeout_s
    assert long.entry_stall_timeout_s < long.episode_timeout_s / 2
    assert long.heartbeat_interval_s >= 30


def test_the_harness_exec_budget_is_derived_from_the_episode_budget() -> None:
    ladder = resolve_agent_loop_timeouts(1800, None)
    config = HarnessConfig(kind="claude_code", executable="claude")

    resolved = HarnessAgentLoop.apply_timeout_ladder(config, ladder)

    assert resolved.time_budget_s == ladder.harness_exec_timeout_s
    assert resolved.time_budget_s > ladder.episode_timeout_s
    # The harness's own cap can no longer disagree with the episode budget.
    assert config.time_budget_s != resolved.time_budget_s


@pytest.mark.parametrize(
    ("episode", "admission", "stall", "expected"),
    [
        (0, None, None, "must be greater than zero"),
        (-5, None, None, "must be greater than zero"),
        (7200, None, 1, "shorter than"),
    ],
)
def test_contradictory_configurations_are_refused(
    episode: float,
    admission: float | None,
    stall: float | None,
    expected: str,
) -> None:
    with pytest.raises(ValueError, match=expected):
        resolve_agent_loop_timeouts(episode, admission, entry_stall_timeout_s=stall)


def test_the_watchdog_can_still_be_disabled_explicitly() -> None:
    assert resolve_agent_loop_timeouts(7200, None, entry_stall_timeout_s=0).entry_stall_timeout_s == 0


def test_the_ladder_describes_itself_for_the_startup_log() -> None:
    described = resolve_agent_loop_timeouts(7200, 1800).describe()

    assert "episode=7200s" in described
    assert "admission<=1800s" in described
    assert "stall=" in described


def test_an_admission_wait_that_outlasts_the_episode_is_refused() -> None:
    """A request that queues longer than it would work is a capacity fault, not a wait.

    This was a warning once, which meant a misconfigured node reported the inversion as a
    slow trickle of capacity timeouts hours into a run instead of at startup. Raising the
    deadline past the work it precedes cannot fix it either: a node that cannot admit
    inside one episode cannot admit inside two.
    """
    with pytest.raises(ValueError, match="longer waiting for a sandbox than working"):
        resolve_agent_loop_timeouts(600, 5000)

    # Equal is refused too: the queue would consume the entire budget before work began.
    with pytest.raises(ValueError, match="longer waiting for a sandbox than working"):
        resolve_agent_loop_timeouts(600, 600)

    # Just inside the budget stays legal, so the check bounds the ladder without
    # narrowing the range an operator can actually use.
    ladder = resolve_agent_loop_timeouts(600, 599)
    assert ladder.admission_timeout_s == 599
    assert ladder.child_deadline_s > 599


def test_zero_valued_fields_are_rejected_by_the_value_object() -> None:
    with pytest.raises(ValueError, match="heartbeat_interval_s"):
        AgentLoopTimeouts(
            episode_timeout_s=1,
            setup_allowance_s=1,
            admission_timeout_s=0,
            heartbeat_interval_s=0,
            entry_stall_timeout_s=1,
        )


def test_the_shipped_configs_resolve_to_a_consistent_ladder() -> None:
    """Guard the shipped pair against drifting apart again.

    The recipe that motivated this had the harness allowed two hours while the manager
    declared an entry dead after one, because each number was written by hand.
    """
    from pathlib import Path

    import yaml
    from omegaconf import OmegaConf
    from psrl.workers.agent_loop.timeouts import resolve_from_config

    rollout = yaml.safe_load(Path("psrl/trainer/config/rollout/psrl_rollout.yaml").read_text())
    agentic = yaml.safe_load(Path("psrl/trainer/config/psrl/agentic_rl.yaml").read_text())
    config = OmegaConf.create(
        {
            "gen_actor_rollout_ref": {"rollout": {"agent": rollout["agent"]}},
            "psrl": {"agentic_rl": agentic},
        }
    )

    ladder = resolve_from_config(config)

    assert ladder.episode_timeout_s == rollout["agent"]["trajectory_timeout"]
    assert ladder.admission_timeout_s == rollout["agent"]["sandbox"]["capacity"]["acquire_timeout_s"]
    # The stall threshold is derived, and it no longer has to cover a whole episode.
    assert ladder.entry_stall_timeout_s < ladder.episode_timeout_s
    assert ladder.harness_exec_timeout_s > ladder.episode_timeout_s
