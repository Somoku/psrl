import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from verl.base_config import BaseConfig
from verl.trainer.config import CheckpointConfig
from verl.utils.profiler import ProfilerConfig

from .model import HFModelConfig
from .optimizer import OptimizerConfig

__all__ = [
    "FSDPEngineConfig",
    "McoreEngineConfig",
    "TrainingWorkerConfig",
    "EngineConfig",
    "EngineRouterReplayConfig",
]


# TODO: rename to RouterReplayConfig after removing the legacy implementation
@dataclass
class EngineRouterReplayConfig(BaseConfig):
    """Configuration for router replay in MoE models.

    This configuration controls the routing behavior for Mixture of Experts (MoE) models,
    allowing for deterministic training through route recording and replay.

    Args:
        mode (str): Router replay mode. Options: 'disabled', 'R2', 'R3'.
            - 'disabled': No router replay functionality
            - 'R2': Use Router Replay routing strategy
            - 'R3': Use Rollout Router Replay routing strategy
        record_file (Optional[str]): File path to save recorded routing decisions.
            Required when mode is 'record', 'R2', or 'R3'.
        replay_file (Optional[str]): File path to load recorded routing decisions for replay.
            Required when mode is 'replay'.
    """

    mode: str = "disabled"
    record_file: str | None = None
    replay_file: str | None = None

    def __post_init__(self):
        """Validate router replay configuration."""
        valid_modes = ["disabled", "R2", "R3"]
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid router_replay mode: {self.mode}. Must be one of {valid_modes}")


@dataclass
class EngineConfig(BaseConfig):
    _mutable_fields = BaseConfig._mutable_fields | {
        "use_dynamic_bsz",
        "max_token_len_per_gpu",
        "micro_batch_size_per_gpu",
        "infer_max_token_len_per_gpu",
        "infer_micro_batch_size_per_gpu",
        "use_fused_kernels",
        "use_remove_padding",
    }

    # whether to offload param
    param_offload: bool = False
    # whether to offload optimizer
    optimizer_offload: bool = False
    # whether to offload grad
    grad_offload: bool = False
    # whether the engine is forward only (e.g., ref policy)
    forward_only: bool = False
    # the strategy (backend)
    strategy: str = None
    # model dtype
    dtype: str = "bfloat16"  # ["bfloat16", "float16"]
    # whether to use dynamic bsz
    use_dynamic_bsz: bool = True
    # for training
    max_token_len_per_gpu: int = None
    micro_batch_size_per_gpu: int = None
    # for inference
    infer_max_token_len_per_gpu: int = None
    infer_micro_batch_size_per_gpu: int = None
    # whether use fuse lm head kernel
    use_fused_kernels: bool = False
    # TODO (this may conflict with the one in model config)
    use_remove_padding: bool = True

    seed: int = 42

    full_determinism: bool = False
    router_replay: EngineRouterReplayConfig = field(default_factory=EngineRouterReplayConfig)

    def __post_init__(self):
        pass
        # TODO: turn on this check after we reorg config
        # if self.use_dynamic_bsz:
        #     assert self.max_token_len_per_gpu is not None
        # else:
        #     assert self.micro_batch_size_per_gpu is not None


@dataclass
class McoreEngineConfig(EngineConfig):
    """Configuration for Megatron parallelism.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        param_offload (bool): Whether to offload parameters to CPU.
        grad_offload (bool): Whether to offload gradients to CPU.
        optimizer_offload (bool): Whether to offload optimizer states to CPU.
        tensor_model_parallel_size (int): Tensor model parallel size.
        expert_model_parallel_size (int): Expert model parallel size for MoE models.
        expert_tensor_parallel_size (int | None): Expert tensor parallel size for MoE models.
        pipeline_model_parallel_size (int): Pipeline model parallel size.
        virtual_pipeline_model_parallel_size (int | None): Virtual pipeline model parallel size
            for interleaved scheduling.
        context_parallel_size (int): Context parallel size for long sequences.
        sequence_parallel (bool): Whether to enable sequence parallelism.
        use_distributed_optimizer (bool): Whether to use distributed optimizer.
        use_dist_checkpointing (bool): Whether to use distributed checkpointing.
        dist_checkpointing_path (str | None): Path for distributed checkpointing.
        seed (int): Random seed for reproducibility.
        override_ddp_config (dict[str, Any]): Override configuration for DDP.
        override_transformer_config (dict[str, Any]): Override configuration for transformer.
        use_mbridge (bool): Whether to use MBridge for communication.
        dtype (str): Mixed precision training param dtype, default "bfloat16"
    """

    # sequence_parallel is not listed as a frozen field for auto-correction purpose
    _mutable_fields = EngineConfig._mutable_fields | {"sequence_parallel"}
    # mcore parallelism
    tensor_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int | None = None
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: int | None = None
    context_parallel_size: int = 1
    sequence_parallel: bool = True
    use_distributed_optimizer: bool = True
    use_dist_checkpointing: bool = False
    dist_checkpointing_path: str | None = None
    dist_checkpointing_prefix: str = ""
    dist_ckpt_optim_fully_reshardable: bool = False
    distrib_optim_fully_reshardable_mem_efficient: bool = False
    override_ddp_config: dict[str, Any] = field(default_factory=dict)
    override_transformer_config: dict[str, Any] = field(default_factory=dict)
    override_mcore_model_config: dict[str, Any] = field(default_factory=dict)
    use_mbridge: bool = False
    vanilla_mbridge: bool = True
    strategy: str = "megatron"

    def __post_init__(self) -> None:
        super().__post_init__()
        """config validation logics go here"""
        assert self.strategy == "megatron"
        assert self.dtype in ["bfloat16", "float16"], f"dtype {self.dtype} not supported"
        if self.tensor_model_parallel_size == 1:
            warnings.warn("set sequence parallel to false as TP size is 1", stacklevel=2)
            self.sequence_parallel = False


@dataclass
class FSDPEngineConfig(EngineConfig):
    """Configuration for FSDP (Fully Sharded Data Parallel).

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        wrap_policy (Dict[str, Any]): Configuration for FSDP wrap policy.
        param_offload (bool): Whether to offload parameters to CPU, default False
        optimizer_offload (bool): Whether to offload optimizer states to CPU, default False
        offload_policy (bool): Whether to offload policy model parameters, default False
        reshard_after_forward (bool): Whether to reshard parameters after forward pass, default True
        fsdp_size (int): FSDP group size. -1 means use all available GPUs.
        forward_prefetch (bool): Whether to prefetch parameters for next forward pass, default False
        model_dtype (str): Model data type used to initialize the transformers model. default "fp32"
        use_orig_params (bool): Whether to use original parameters when initialize FSDP1, default False
        seed (int): Random seed for reproducibility.
        full_determinism (bool): If true, enable_full_determinism is called to ensure reproducible results
            in distributed training. Important: this will negatively impact performance, so only use it for
            debugging.
        mixed_precision (dict[str, Any] | None): Mixed precision configuration for FSDP, default None
        dtype (str): Mixed precision training param dtype, default "bfloat16"
    """

    # ulysses_sequence_parallel_size is mutable for backward compatibility
    _mutable_fields = EngineConfig._mutable_fields | {"ulysses_sequence_parallel_size"}

    # fsdp specific flags
    wrap_policy: dict[str, Any] = field(default_factory=dict)
    offload_policy: bool = False
    reshard_after_forward: bool = True
    fsdp_size: int = -1
    forward_prefetch: bool = False
    model_dtype: str = "fp32"
    use_orig_params: bool = False
    mixed_precision: dict[str, Any] | None = None
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    use_torch_compile: bool = True
    entropy_checkpointing: bool = False
    strategy: str = "fsdp"

    def __post_init__(self):
        super().__post_init__()
        assert self.strategy in ["fsdp", "fsdp2"], f"strategy {self.strategy} not supported"


@dataclass
class TrainingWorkerConfig(BaseConfig):
    model_type: str = None  # model type (language_model/value_model)
    model_config: HFModelConfig = None
    engine_config: EngineConfig = None
    optimizer_config: OptimizerConfig = None
    checkpoint_config: CheckpointConfig = None
    checkpoint_config: CheckpointConfig = None
    profiler_config: ProfilerConfig = None
    # automatically select engine and optimizer function.
    # This function takes model config and the device name as parameter.
    # Users can pass in a higher-order function to take more parameters
    auto_select_engine_optim_fn: Callable[["HFModelConfig", str], tuple["EngineConfig", "OptimizerConfig"]] = None
