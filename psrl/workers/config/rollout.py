# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rollout configuration — common classes re-exported from veRL, PSRL-unique classes defined here."""

from dataclasses import dataclass, field

from omegaconf import MISSING
from verl.base_config import BaseConfig

# Re-export common classes from veRL (keep veRL's RolloutConfig as the base)
from verl.workers.config.rollout import (
    CheckpointEngineConfig,
    CustomAsyncServerConfig,
    PrometheusConfig,
    SamplingConfig,
    ServerConfig,
    TraceConfig,
)
from verl.workers.config.rollout import (
    MultiTurnConfig as _VeRLMultiTurnConfig,
    RolloutConfig as _VeRLRolloutConfig,
    AgentLoopConfig as _VeRLAgentLoopConfig,
)


@dataclass
class PoolingConfig(BaseConfig):
    """Configuration for vLLM pooling models (e.g., reward/embedding models)."""

    # Whether to L2-normalize the pooling output.
    normalize: bool = False
    # Whether to apply an activation function (e.g., sigmoid) to the output.
    use_activation: bool = False


# PSRL-unique classes not present in veRL
@dataclass
class EnvironmentConfig(BaseConfig):
    name: str | None = MISSING
    step_timeout: float | None = None


@dataclass
class AgentDataConfig(BaseConfig):
    name: str | None = MISSING

@dataclass
class MultiTurnConfig(_VeRLMultiTurnConfig):
    _mutable_fields = {"max_turns"}

    enable: bool = False
    max_turns: int | None = None
    tool_config_path: str | None = None
    function_tool_path: str | None = None
    max_parallel_calls: int = 1
    max_tool_response_length: int = 256
    tool_response_truncate_side: str = "middle"
    use_inference_chat_template: bool = False
    tokenization_sanity_check_mode: str = "strict"
    format: str = "hermes"

@dataclass
class AgentLoopConfig(_VeRLAgentLoopConfig):
    """PSRL-specific AgentLoopConfig with environment and data sub-configs."""

    route_strategy: str = "round_robin"
    trajectory_timeout: float | None = None
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    data: AgentDataConfig = field(default_factory=AgentDataConfig)
    retry_limit: int = 1
    raise_on_error: bool = True
    gamma: float = 0.0
    reward_bonus_coeff: float = 0.0
    traj_reward_mode: str = "traj"
    default_agent_loop: str = "generate_only_agent"


@dataclass
class RolloutConfig(_VeRLRolloutConfig):
    """PSRL extension of veRL RolloutConfig.

    Adds:
    - pooling-model support (runner, task, reward_kwargs) for gen_dplb reward/embedding models
    - enable_weights_cpu_backup for TMS-style level-1 CPU sleep
    - PSRL-specific agent config (AgentLoopConfig with env/data sub-configs)
    """

    # Whether disable attention in vLLM.
    disable_attn: bool = False
    # vLLM runner type: 'generate' for autoregressive LLMs, 'pooling' for
    # embedding / reward / classification models.
    runner: str = "generate"
    # vLLM task type forwarded to the engine (e.g., 'generate', 'classify', 'embed').
    task: str = "generate"
    # Pooling configuration, effective only when runner == 'pooling'.
    reward_kwargs: PoolingConfig = field(default_factory=PoolingConfig)

    # (TMS-only) Whether to enable offloading weights (level-1 sleep) to CPU.
    enable_weights_cpu_backup: bool = False

    # Multi-turn config
    multi_turn: MultiTurnConfig = field(default_factory=MultiTurnConfig)

    # Override veRL's AgentLoopConfig with PSRL's richer variant (env + data sub-configs).
    agent: AgentLoopConfig = field(default_factory=AgentLoopConfig)


__all__ = [
    "SamplingConfig",
    "MultiTurnConfig",
    "CustomAsyncServerConfig",
    "AgentLoopConfig",
    "TraceConfig",
    "ServerConfig",
    "PrometheusConfig",
    "RolloutConfig",
    "PoolingConfig",
    "CheckpointEngineConfig",
    "EnvironmentConfig",
    "AgentDataConfig",
]
