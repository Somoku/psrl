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
    MultiTurnConfig,
    PrometheusConfig,
    SamplingConfig,
    ServerConfig,
    TraceConfig,
)
from verl.workers.config.rollout import (
    RolloutConfig as _VeRLRolloutConfig,
)


@dataclass
class PoolingConfig(BaseConfig):
    """Configuration for vLLM pooling models (e.g., reward/embedding models)."""

    # Whether to L2-normalize the pooling output.
    normalize: bool = False
    # Whether to apply an activation function (e.g., sigmoid) to the output.
    use_activation: bool = False


@dataclass
class RolloutConfig(_VeRLRolloutConfig):
    """PSRL extension of veRL RolloutConfig.

    Adds pooling-model support fields so that gen_dplb reward/embedding models
    can be configured to run vLLM in pooling mode instead of generative mode.
    """

    # vLLM runner type: 'generate' for autoregressive LLMs, 'pooling' for
    # embedding / reward / classification models.
    runner: str = "generate"
    # vLLM task type forwarded to the engine (e.g., 'generate', 'classify', 'embed').
    task: str = "generate"
    # Pooling configuration, effective only when runner == 'pooling'.
    reward_kwargs: PoolingConfig = field(default_factory=PoolingConfig)


# PSRL-unique classes not present in veRL
@dataclass
class EnvironmentConfig(BaseConfig):
    name: str | None = MISSING
    step_timeout: float | None = None


@dataclass
class AgentDataConfig(BaseConfig):
    name: str | None = MISSING


@dataclass
class AgentLoopConfig(BaseConfig):
    """PSRL-specific AgentLoopConfig with environment and data sub-configs."""

    num_workers: int = 8
    agent_loop_config_path: str | None = None
    route_strategy: str = "round_robin"
    custom_async_server: CustomAsyncServerConfig = field(default_factory=CustomAsyncServerConfig)
    trajectory_timeout: float | None = None
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    data: AgentDataConfig = field(default_factory=AgentDataConfig)
    retry_limit: int = 1
    raise_on_error: bool = True
    gamma: float = 0.0
    reward_bonus_coeff: float = 0.0
    traj_reward_mode: str = "traj"


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
