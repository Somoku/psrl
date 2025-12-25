from dataclasses import dataclass, field

from omegaconf import MISSING
from verl.base_config import BaseConfig
from verl.utils.profiler import ProfilerConfig

__all__ = [
    "SamplingConfig",
    "MultiTurnConfig",
    "CustomAsyncServerConfig",
    "AgentLoopConfig",
    "TraceConfig",
    "ServerConfig",
    "PrometheusConfig",
    "RolloutConfig",
]


@dataclass
class SamplingConfig(BaseConfig):
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    do_sample: bool = True
    n: int = 1


@dataclass
class MultiTurnConfig(BaseConfig):
    _mutable_fields = {"max_turns"}

    enable: bool = False
    max_turns: int | None = None
    tool_config_path: str | None = None
    max_parallel_calls: int = 1
    max_tool_response_length: int = 256
    tool_response_truncate_side: str = "middle"
    use_inference_chat_template: bool = False
    tokenization_sanity_check_mode: str = "strict"
    format: str = "hermes"


@dataclass
class CustomAsyncServerConfig(BaseConfig):
    path: str | None = None
    name: str | None = None


@dataclass
class EnvironmentConfig(BaseConfig):
    name: str | None = MISSING
    step_timeout: float | None = None


@dataclass
class AgentLoopConfig(BaseConfig):
    num_workers: int = 8
    agent_loop_config_path: str | None = None
    route_strategy: str = "round_robin"
    custom_async_server: CustomAsyncServerConfig = field(default_factory=CustomAsyncServerConfig)
    trajectory_timeout: float | None = None
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    retry_limit: int = 1
    raise_on_error: bool = True
    gamma: float = 0.0
    reward_bonus_coeff: float = 0.0
    traj_reward_mode: str = "traj"


@dataclass
class TraceConfig(BaseConfig):
    backend: str | None = None
    token2text: bool = False


@dataclass
class ServerConfig(BaseConfig):
    """
    Configuration for SGLang server when running in server mode
    """

    timeout: float = 60.0
    max_attempts: int = 3
    retry_delay: float = 2.0
    max_connections: int = 1000
    max_start_wait_time: float = 300.0


@dataclass
class PrometheusConfig(BaseConfig):
    """
    Configuration for Prometheus server
    """

    # whether enable prometheus on server mode rollout
    enable: bool = False
    # Port number that Prometheus listens on, default is 9090
    port: int = 9090
    # Path to Prometheus configuration file
    file: str = "/tmp/ray/session_latest/metrics/prometheus/prometheus.yml"
    # Specify served_model_name to avoid displaying overly long model paths in Grafana
    served_model_name: str | None = None


@dataclass
class RolloutConfig(BaseConfig):
    _mutable_fields = {"max_model_len", "load_format"}

    name: str | None = MISSING
    mode: str = "sync"
    disable_attn: bool = False
    skip_tokenizer_init: bool = True

    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    do_sample: bool = True
    n: int = 1

    # Early termination threshold for multi-turn rollout in sglang.
    # Abort remaining requests when (1 - over_sample_rate) * total_requests are completed.
    over_sample_rate: float = 0.0

    prompt_length: int = 512
    response_length: int = 512

    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.5
    ignore_eos: bool = False
    enforce_eager: bool = True
    cudagraph_capture_sizes: list | None = None
    free_cache_engine: bool = True
    data_parallel_size: int = 1
    expert_parallel_size: int = 1
    tensor_model_parallel_size: int = 2
    pipeline_model_parallel_size: int = 1
    max_num_batched_tokens: int = 8192

    # TODO: enable train_kwargs
    # train_sampling_config: SamplingConfig = field(default_factory=SamplingConfig)

    val_kwargs: SamplingConfig = field(default_factory=SamplingConfig)

    max_model_len: int | None = None
    max_num_seqs: int = 1024

    # note that the logprob computation should belong to the actor
    log_prob_micro_batch_size: int | None = None
    log_prob_micro_batch_size_per_gpu: int | None = None
    log_prob_use_dynamic_bsz: bool = False
    log_prob_max_token_len_per_gpu: int = 16384

    disable_log_stats: bool = False

    multi_stage_wake_up: bool = False
    engine_kwargs: dict = field(default_factory=dict)

    calculate_log_probs: bool = False

    agent: AgentLoopConfig = field(default_factory=AgentLoopConfig)

    trace: TraceConfig = field(default_factory=TraceConfig)

    multi_turn: MultiTurnConfig = field(default_factory=MultiTurnConfig)

    # Server configuration for sglang server mode
    server: ServerConfig = field(default_factory=ServerConfig)

    # Use Prometheus to collect and monitor rollout statistics
    prometheus: PrometheusConfig = field(default_factory=PrometheusConfig)

    update_weights_bucket_megabytes: int = 512

    skip_rollout: bool = False

    skip_dump_dir: str = "/tmp/rollout_dump"

    profiler: ProfilerConfig | None = None

    enable_chunked_prefill: bool = True

    enable_prefix_caching: bool = True

    logprobs_mode: str = "raw_logprobs"

    load_format: str = "dummy"

    layered_summon: bool = False

    layer_name_map: dict = field(default_factory=dict)

    sglang_engine_mode: str = "local"

    limit_images: int | None = None

    skip_tokenizer_init: bool = False

    enable_rollout_routing_replay: bool = False
