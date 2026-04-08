# psrl/tests/config/test_reward_yaml_merge.py
import pytest

pytestmark = pytest.mark.cpu_test


def test_reward_yaml_merges_psrl_fields():
    """Verify that ppo_trainer config composes both veRL base and PSRL incremental reward fields."""
    import os

    from hydra import compose, initialize_config_dir

    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../psrl/trainer/config"))
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_trainer")

    # veRL base fields (from verl's reward/reward.yaml via PSRLSearchPath)
    assert cfg.reward.num_workers == 8
    assert cfg.reward.reward_manager.name == "naive"
    assert cfg.reward.reward_model.enable is False

    # PSRL incremental fields (from psrl_reward.yaml, merged into same key)
    assert "reward_models" in cfg.reward
    assert isinstance(cfg.reward.reward_models, list)
    assert cfg.reward.reward_normalization.enable is False
    assert cfg.reward.reward_normalization.level == "batch"
    assert cfg.reward.launch_reward_fn_async is False
    assert cfg.reward.data_processor.path is None
