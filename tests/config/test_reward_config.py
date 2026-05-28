# psrl/tests/config/test_reward_config.py
import pytest

pytestmark = pytest.mark.cpu_test


def test_psrl_reward_config_is_subclass_of_verl():
    from psrl.workers.config.reward import PSRLRewardConfig
    from verl.workers.config.reward import RewardConfig

    assert issubclass(PSRLRewardConfig, RewardConfig)


def test_psrl_reward_config_defaults():
    from psrl.workers.config.reward import (
        PSRLRewardConfig,
    )

    cfg = PSRLRewardConfig()
    # veRL base fields
    assert cfg.num_workers == 8
    assert cfg.reward_manager.name == "naive"
    assert cfg.reward_model.enable is False
    # PSRL extension fields
    assert cfg.managers == {}
    assert cfg.active_managers == []
    assert cfg.reward_normalization.enable is False
    assert cfg.reward_normalization.level == "batch"
    assert cfg.launch_reward_fn_async is False
    assert cfg.data_processor.path is None


def test_psrl_reward_model_config_is_subclass_of_verl():
    from psrl.workers.config.reward import PSRLRewardModelConfig
    from verl.workers.config.reward import RewardModelConfig

    assert issubclass(PSRLRewardModelConfig, RewardModelConfig)


def test_psrl_reward_model_config_defaults():
    from psrl.workers.config.reward import PSRLRewardModelConfig

    cfg = PSRLRewardModelConfig()
    # veRL base fields
    assert cfg.enable is False
    assert cfg.n_gpus_per_node == 0
    # PSRL additions
    assert cfg.strategy == "fsdp"
    assert cfg.micro_batch_size_per_gpu is None
    assert cfg.use_dynamic_bsz is False


def test_single_reward_model_config_defaults():
    from psrl.workers.config.reward import SingleRewardModelConfig

    cfg = SingleRewardModelConfig()
    assert cfg.reward_model_name == "default"
    assert cfg.reward_loop_type == "naive"
    assert cfg.reward_fn == "default"
    assert cfg.reward_coef == 1.0
    assert cfg.reward_loop_kwargs == {}


def test_reward_data_processor_config_no_path():
    from psrl.workers.config.reward import RewardDataProcessorConfig

    cfg = RewardDataProcessorConfig()
    pre, post = cfg.get_process_fn()
    assert pre is None
    assert post is None


def test_new_dataclasses_exported_from_workers_config():
    """Verify that the new dataclasses are accessible from psrl.workers.config."""
    from psrl.workers.config import (  # noqa: F401
        PSRLRewardConfig,
        PSRLRewardModelConfig,
        RewardDataProcessorConfig,
        RewardNormalizationConfig,
        SingleRewardModelConfig,
    )


def test_multi_reward_model_config_has_managers():
    """Verify that MultiRewardModelConfig uses managers dict + active_managers list."""
    from psrl.workers.config.reward_model import MultiRewardModelConfig

    cfg = MultiRewardModelConfig()
    assert cfg.managers == {}
    assert cfg.active_managers == []
    assert cfg.launch_reward_fn_async is False


def test_resolve_active_managers():
    """Test resolve_active_managers function resolves names to configs."""
    from dataclasses import dataclass

    from psrl.workers.config.reward_model import resolve_active_managers

    @dataclass
    class FakeManagerConfig:
        reward_loop_type: str = "naive"

    @dataclass
    class FakeRewardConfig:
        active_managers: list = None
        managers: dict = None

        def __post_init__(self):
            if self.active_managers is None:
                self.active_managers = []
            if self.managers is None:
                self.managers = {}

    dapo_cfg = FakeManagerConfig(reward_loop_type="dapo")
    naive_cfg = FakeManagerConfig(reward_loop_type="naive")

    config = FakeRewardConfig(
        active_managers=["dapo", "naive"],
        managers={"dapo": dapo_cfg, "naive": naive_cfg},
    )
    resolved = resolve_active_managers(config)
    assert len(resolved) == 2
    assert resolved[0].reward_loop_type == "dapo"
    assert resolved[1].reward_loop_type == "naive"


def test_resolve_active_managers_error():
    """Test resolve_active_managers raises ValueError for unknown name."""
    from dataclasses import dataclass

    from psrl.workers.config.reward_model import resolve_active_managers

    @dataclass
    class FakeRewardConfig:
        active_managers: list = None
        managers: dict = None

        def __post_init__(self):
            if self.active_managers is None:
                self.active_managers = []
            if self.managers is None:
                self.managers = {}

    config = FakeRewardConfig(
        active_managers=["nonexistent"],
        managers={},
    )
    with pytest.raises(ValueError, match="nonexistent"):
        resolve_active_managers(config)
