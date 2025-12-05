# Modified from verl/experimental/reward/reward_loop/registry.py
import asyncio
from collections.abc import Callable
from functools import partial
from typing import Any

from omegaconf import DictConfig
from verl.trainer.ppo.reward import get_custom_reward_fn

from psrl.utils.reward_score import default_compute_score_async
from psrl.workers.reward.reward_loop.base import RewardLoopManagerBase

__all__ = ["register", "get_reward_loop_manager_cls", "load_reward_loop_manager"]

REWARD_LOOP_MANAGER_REGISTRY: dict[str, type[RewardLoopManagerBase]] = {}


def register(
    name: str,
) -> Callable[[type[RewardLoopManagerBase]], type[RewardLoopManagerBase]]:
    """Decorator to register a reward loop manager class with a given name.

    Args:
        name: `(str)`
            The name of the reward loop manager.
    """

    def decorator(cls: type[RewardLoopManagerBase]) -> type[RewardLoopManagerBase]:
        if name in REWARD_LOOP_MANAGER_REGISTRY and REWARD_LOOP_MANAGER_REGISTRY[name] != cls:
            raise ValueError(
                f"reward loop manager {name} has already been registered: "
                f"{REWARD_LOOP_MANAGER_REGISTRY[name]} vs {cls}"
            )
        REWARD_LOOP_MANAGER_REGISTRY[name] = cls
        return cls

    return decorator


def get_reward_loop_manager_cls(name: str) -> type[RewardLoopManagerBase]:
    """Get the reward loop manager class with a given name.

    Args:
        name: `(str)`
            The name of the reward loop manager.

    Returns:
        `(type)`: The reward loop manager class.
    """
    if name not in REWARD_LOOP_MANAGER_REGISTRY:
        raise ValueError(f"Unknown reward loop manager: {name}")
    return REWARD_LOOP_MANAGER_REGISTRY[name]


def load_reward_loop_manager(
    config: DictConfig,
    input_tokenizer: Any,
    reward_model_router: Any,
    reward_model_tokenizer: Any,
    **reward_kwargs: Any,
) -> RewardLoopManagerBase:
    """Load the reward loop manager based on the configuration.

    Args:
        config: `(DictConfig)`
            The configuration for the reward loop manager.
        input_tokenizer: `(Any)`
            The tokenizer for the input.
        reward_model_router: `(Any)`
            The reward model router.
        reward_model_tokenizer: `(Any)`
            The tokenizer for the reward model.
        **reward_kwargs: `(Any)`
            Additional keyword arguments for the reward loop manager.
    Returns:
        `(RewardLoopManagerBase)`: The reward loop manager instance.
    """
    # Try to get a custom reward function based on the configuration
    # user defined reward manager can be registered in custom_reward_fn
    compute_score = get_custom_reward_fn(config)

    reward_loop_manager_name = config.reward_model.get("reward_manager", "naive")
    reward_loop_manager_cls = get_reward_loop_manager_cls(reward_loop_manager_name)

    if compute_score is None:
        sandbox_config = config.reward_model.get("sandbox_fusion")
        sandbox_url = sandbox_config.get("url") if sandbox_config else None
        memory_limit_mb = sandbox_config.get("memory_limit_mb", 1024)
        if sandbox_url:
            # Create an asyncio.Semaphore to control concurrent access to the sandbox
            # Note: asyncio.Semaphore must be created in the same event loop where it will be used
            # Therefore, we pass max_concurrent as a parameter and create the semaphore later
            max_concurrent = sandbox_config.get("max_concurrent", 64)
            _concurrent_semaphore = asyncio.Semaphore(max_concurrent)
            final_compute_score = partial(
                default_compute_score_async,
                sandbox_fusion_url=sandbox_url,
                concurrent_semaphore=_concurrent_semaphore,
                memory_limit_mb=memory_limit_mb,
            )
        else:
            final_compute_score = default_compute_score_async

    return reward_loop_manager_cls(
        config,
        input_tokenizer,
        final_compute_score,
        reward_model_router,
        reward_model_tokenizer,
        **reward_kwargs,
    )
