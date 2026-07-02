# psrl/tests/config/test_reward_yaml_merge.py
import pytest

pytestmark = pytest.mark.cpu_test


def test_reward_yaml_merges_psrl_fields():
    """Verify that ppo_trainer config composes PSRL reward fields correctly."""
    import os

    from hydra import compose, initialize_config_dir

    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../psrl/trainer/config"))
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_trainer")

    # PSRL reward fields (from psrl's reward/reward.yaml)
    assert "managers" in cfg.reward
    assert isinstance(dict(cfg.reward.managers), dict)
    assert "active_managers" in cfg.reward
    assert isinstance(list(cfg.reward.active_managers), list)
    assert "dapo" in cfg.reward.active_managers
    assert "dapo" in cfg.reward.managers
    assert cfg.reward.launch_reward_fn_async is False

    # Verify the dapo manager has expected structure
    dapo_cfg = cfg.reward.managers.dapo
    assert dapo_cfg.reward_loop_type == "dapo"
    assert dapo_cfg.reward_manager.name == "dapo"
    assert dapo_cfg.reward_manager.source == "register"
    assert dapo_cfg.reward_kwargs.max_resp_len == 5120
    assert dapo_cfg.reward_kwargs.overlong_buffer_cfg.enable is True
