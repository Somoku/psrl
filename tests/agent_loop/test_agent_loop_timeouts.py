"""The rollout budgets: one number for agent work, a second for grading.

These were once a derived ladder, which was a way of keeping several nested deadlines
consistent. With sandbox admission and provisioning no longer charged to the episode
there is nothing left to nest: the episode budget is one number, grading is a second and
may be unset, and the rest is the sandbox service's own business.
"""

import pytest
from psrl.workers.agent_loop.timeouts import (
    DEFAULT_TRAJECTORY_TIMEOUT_S,
    LIFETIME_SLACK_S,
    resolve_rollout_timeouts,
)

pytestmark = pytest.mark.cpu_test


def test_the_trajectory_budget_is_the_episode_number() -> None:
    assert resolve_rollout_timeouts(3600, None).trajectory_timeout_s == 3600


def test_a_null_trajectory_budget_selects_the_default_rather_than_no_limit() -> None:
    """A rollout with no bound cannot be told apart from a wedged one."""
    assert resolve_rollout_timeouts(None, None).trajectory_timeout_s == DEFAULT_TRAJECTORY_TIMEOUT_S


def test_a_null_grading_budget_means_grading_is_not_timed() -> None:
    assert resolve_rollout_timeouts(3600, None).grading_timeout_s is None
    assert resolve_rollout_timeouts(3600, 900).grading_timeout_s == 900


@pytest.mark.parametrize("episode", [0, -5])
def test_a_non_positive_trajectory_budget_is_refused(episode: float) -> None:
    with pytest.raises(ValueError, match="trajectory_timeout"):
        resolve_rollout_timeouts(episode, None)


@pytest.mark.parametrize("grading", [0, -5])
def test_a_non_positive_grading_budget_is_refused(grading: float) -> None:
    with pytest.raises(ValueError, match="grading_timeout"):
        resolve_rollout_timeouts(3600, grading)


def test_the_budgets_describe_themselves_for_the_startup_log() -> None:
    described = resolve_rollout_timeouts(7200, 900).describe()

    assert "trajectory=7200s" in described
    assert "grading=900s" in described
    assert "unbounded" in resolve_rollout_timeouts(7200, None).describe()


def test_the_sandbox_lifetime_covers_agent_work_grading_and_margin() -> None:
    """The node reclaims a sandbox by this, so it must cover everything the sandbox is for."""
    timed = resolve_rollout_timeouts(3600, 900)
    assert timed.sandbox_lifetime_s == 3600 + 900 + LIFETIME_SLACK_S

    # Grading untimed means the sandbox has to outlive the episode and the margin.
    untimed = resolve_rollout_timeouts(3600, None)
    assert untimed.sandbox_lifetime_s == 3600 + LIFETIME_SLACK_S


def test_the_shipped_config_resolves_to_the_configured_budgets() -> None:
    """Guard the shipped recipe: the two roots read the keys an operator sets."""
    from pathlib import Path

    import yaml
    from omegaconf import OmegaConf
    from psrl.workers.agent_loop.timeouts import resolve_from_config

    rollout = yaml.safe_load(Path("psrl/trainer/config/rollout/psrl_rollout.yaml").read_text())
    config = OmegaConf.create({"gen_actor_rollout_ref": {"rollout": {"agent": rollout["agent"]}}})

    timeouts = resolve_from_config(config)

    assert timeouts.trajectory_timeout_s == rollout["agent"]["trajectory_timeout"]
    # The shipped default leaves grading untimed.
    assert rollout["agent"]["grading_timeout"] is None
    assert timeouts.grading_timeout_s is None
