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
    validate_lease_max_age_s,
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


def test_a_long_admission_wait_is_reported_without_failing_the_run() -> None:
    """Legal but worth saying out loud: queueing longer than working is rarely intended."""
    ladder = resolve_agent_loop_timeouts(600, 5000)

    assert ladder.admission_timeout_s == 5000
    assert ladder.child_deadline_s > 5000


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
    # The recovery cap has to outlive a healthy sandbox, and the shipped value does.
    validate_lease_max_age_s(rollout["agent"]["sandbox"]["capacity"]["lease_max_age_s"], ladder)


def test_a_recovery_cap_shorter_than_a_healthy_sandbox_is_refused() -> None:
    """The age cap is the only thing that reclaims a leaked lease, so it must not be eager.

    A lease is renewed by its live owner, so the cap firing is the only feedback: by the
    time it is too short, it has already taken capacity from a sandbox that was working.
    """
    ladder = resolve_agent_loop_timeouts(7200, 1800)

    validate_lease_max_age_s(ladder.child_deadline_s + 1, ladder)
    # Null disables the cap, which is a deliberate choice rather than a contradiction.
    validate_lease_max_age_s(None, ladder)

    with pytest.raises(ValueError, match="child deadline"):
        validate_lease_max_age_s(ladder.child_deadline_s, ladder)
