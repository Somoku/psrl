import importlib.util
import os
import sys
from dataclasses import dataclass, field

from verl.base_config import BaseConfig
from verl.utils.profiler import ProfilerConfig
from verl.workers.config.model import HFModelConfig

from .rollout import PoolingConfig, RolloutConfig, SamplingConfig, ServerConfig

__all__ = [
    "SandboxFusionConfig",
    "RewardModelConfig",
    "SingleRewardModelConfig",
    "MultiRewardModelConfig",
]


def get_custome_process_fn(file_path, function_name):
    if not file_path:
        return None

    assert function_name is not None
    module_name = f"custom_reward_module_{function_name}"
    module = sys.modules.get(module_name, None)

    if module is None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Reward process function file '{file_path}' not found.")

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules[module_name] = module
            assert spec.loader is not None
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e

    if not hasattr(module, function_name):
        raise AttributeError(f"Reward preprocess function '{function_name}' not found in '{module.__file__}'.")

    print(f"using customized reward function '{function_name}' from '{module.__file__}'")
    raw_fn = getattr(module, function_name)
    return raw_fn


@dataclass
class SandboxFusionConfig(BaseConfig):
    """Configuration for cloud/local sandbox fusion.

    Args:
        url (str | None): Cloud/local function URL for sandbox execution.
        max_concurrent (int): Max concurrent requests allowed to sandbox.
        memory_limit_mb (int): Max memory limit for each sandbox process in MB.
    """

    url: str | None = None
    max_concurrent: int = 64
    memory_limit_mb: int = 1024


@dataclass
class RewardModelConfig(BaseConfig):
    _mutable_fields = BaseConfig._mutable_fields

    enable: bool = False
    model_type: str = "discriminative"
    name: str = "sglang"
    enable_resource_pool: bool = False
    n_gpus_per_node: int = 0
    nnodes: int = 0
    reward_manager: str = "naive"
    launch_reward_fn_async: bool = False

    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.5
    free_cache_engine: bool = True
    tensor_model_parallel_size: int = 2

    # for generative reward model
    sampling_config: SamplingConfig = field(default_factory=SamplingConfig)
    max_new_tokens: int = 4096

    engine_kwargs: dict = field(default_factory=dict)
    max_num_seqs: int = 1024

    sandbox_fusion: SandboxFusionConfig = field(default_factory=SandboxFusionConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    input_model_config: HFModelConfig = field(default_factory=HFModelConfig)
    model_config: HFModelConfig = field(default_factory=HFModelConfig)
    # Server configuration for sglang server mode
    server_config: ServerConfig = field(default_factory=ServerConfig)


@dataclass
class SingleRewardModelConfig(BaseConfig):
    """Configuration for a single reward model in the multi-reward model setup.

    Args:
        reward_loop_type (Optional[str]): Type of reward loop (naive, dapo, gen, None).
        reward_fn (Optional[str]): Reward function name (default, None).
        reward_model_name (Optional[str]): Name of the model.
        enable_resource_pool (bool): Whether to enable resource pool.
        n_gpus_per_node (int): Number of GPUs per node.
        num_replicas (int): Number of replicas.
        nnodes (int): Number of nodes.
        rollout_ngpus_per_instance_per_node (int): Number of GPUs per instance per node for rollout.
        rollout_nnodes_per_instance (int): Number of nodes per instance for rollout.
        model (HFModelConfig): Model configuration.
        rollout (RolloutConfig): Rollout configuration.
        sampling_config (SamplingConfig): Sampling configuration.
        pooling_config (PoolingConfig): Pooling configuration.
        sandbox_fusion (SandboxFusionConfig): Sandbox fusion configuration.
    """

    reward_loop_type: str | None = None
    reward_fn: list[str] | None = None
    reward_model_name: str | None = None
    enable_resource_pool: bool = False
    n_gpus_per_node: int = 0
    num_replicas: int = 1
    nnodes: int = 1
    rollout_ngpus_per_instance_per_node: int = 1
    rollout_nnodes_per_instance: int = 1
    model: HFModelConfig = field(default_factory=HFModelConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    sampling_config: SamplingConfig = field(default_factory=SamplingConfig)
    pooling_config: PoolingConfig = field(default_factory=PoolingConfig)
    sandbox_fusion: SandboxFusionConfig = field(default_factory=SandboxFusionConfig)


@dataclass
class MultiRewardModelConfig(BaseConfig):
    """Configuration for multi-reward model setup.

    Args:
        launch_reward_fn_async (bool): Whether to launch reward function asynchronously.
        reward_models (list[SingleRewardModelConfig]): List of reward model configurations.
        profiler (ProfilerConfig): Profiler configuration.
    """

    launch_reward_fn_async: bool = False
    reward_models: list[SingleRewardModelConfig] = field(default_factory=list)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
