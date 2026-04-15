# Modified from verl/experimental/reward/reward_loop/registry.py
import asyncio
import inspect
from collections.abc import Callable
from functools import partial
from typing import Any, cast

from omegaconf import DictConfig
from verl.trainer.ppo.reward import _call_with_kwargs, _call_with_kwargs_async
from verl.workers.config.reward import RewardManagerConfig

from psrl.utils.reward_score import default_compute_score_async
from psrl.workers.reward.gen_reward_function import get_gen_reward_function_cls
from psrl.workers.reward.reward_loop.base import RawRewardFn, RewardManagerBase
from psrl.workers.reward.reward_model import RewardModelManager

__all__ = ["register", "get_reward_manager_cls", "load_reward_manager"]

REWARD_MANAGER: dict[str, type[RewardManagerBase]] = {}


def register(name: str) -> Callable[[type[RewardManagerBase]], type[RewardManagerBase]]:
    """Decorator to register a reward manager class with a given name.

    Args:
        name: `(str)`
            The name of the reward manager.
    """

    def decorator(cls: type[RewardManagerBase]) -> type[RewardManagerBase]:
        if name in REWARD_MANAGER and REWARD_MANAGER[name] != cls:
            raise ValueError(f"reward manager {name} has already been registered: {REWARD_MANAGER[name]} vs {cls}")
        REWARD_MANAGER[name] = cls
        return cls

    return decorator


def get_reward_manager_cls(name: str) -> type[RewardManagerBase]:
    """Get the reward manager class with a given name.

    Args:
        name: `(str)`
            The name of the reward manager.

    Returns:
        `(type)`: The reward manager class.
    """
    if name not in REWARD_MANAGER:
        raise ValueError(f"Unknown reward manager: {name}")
    return REWARD_MANAGER[name]


def get_custom_reward_fn(reward_fn_config: DictConfig) -> RawRewardFn | None:
    """Load and return a custom reward function from external file.

    Dynamically imports a reward function from a specified file path and wraps
    it with additional keyword arguments from the configuration.

    Args:
        reward_fn_config (DictConfig): Configuration dictionary containing custom_reward_function
                                      settings with 'path', 'name', and 'reward_kwargs' fields.

    Returns:
        callable or None: Wrapped reward function with merged kwargs, or None
                         if no custom reward function is configured.

    Raises:
        FileNotFoundError: If the specified reward function file doesn't exist.
        RuntimeError: If there's an error loading the module from file.
        AttributeError: If the specified function name isn't found in the module.
    """

    module_path = reward_fn_config.get("path")
    if not module_path:
        return None

    fn_name = reward_fn_config.get("name")
    assert fn_name is not None

    from verl.utils.import_utils import load_extern_object

    raw_fn = load_extern_object(module_path=module_path, object_name=fn_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))
    if not inspect.iscoroutinefunction(raw_fn):
        return partial(_call_with_kwargs, raw_fn, reward_kwargs)
    else:
        return partial(_call_with_kwargs_async, raw_fn, reward_kwargs)


def load_reward_manager(
    config: DictConfig,
    tokenizer: Any,
    reward_manager_cfg: RewardManagerConfig,
    reward_fn_config: DictConfig,
    reward_model_manager: RewardModelManager | None = None,
    **reward_kwargs: Any,
) -> RewardManagerBase:
    """Load the reward loop manager based on the configuration.

    Args:
        config: `(DictConfig)`
            The configuration for the reward loop manager.
        tokenizer: `(Any)`
            The tokenizer for the input.
        reward_model_router: `(Any)`
            The reward model router.
        reward_model_tokenizer: `(Any)`
            The tokenizer for the reward model.
        **reward_kwargs: `(Any)`
            Additional keyword arguments for the reward loop manager.
    Returns:
        `(RewardManagerBase)`: The reward loop manager instance.
    """
    reward_manager_cls: type[RewardManagerBase]
    if reward_manager_cfg.source == "register":
        reward_manager_cls = get_reward_manager_cls(reward_manager_cfg.name)
    elif reward_manager_cfg.source == "importlib":
        from verl.trainer.config.config import ModuleConfig
        from verl.utils.import_utils import load_extern_object

        module_cfg: ModuleConfig | None = reward_manager_cfg.module
        assert module_cfg is not None and module_cfg.path is not None, (
            f"Module path is required when {reward_manager_cfg.source=}, but got {module_cfg=}"
        )
        reward_manager_cls_name = reward_manager_cfg.name
        reward_manager_cls = cast(
            "type[RewardManagerBase]",
            load_extern_object(module_path=module_cfg.path, object_name=reward_manager_cls_name),
        )

    if reward_manager_cfg.source == "register" and reward_manager_cfg.name == "gen":
        gen_reward_function_cls = get_gen_reward_function_cls(reward_fn_config.name)
        return reward_manager_cls(
            config=config,
            tokenizer=tokenizer,
            reward_model_manager=reward_model_manager,
            reward_function=gen_reward_function_cls(),
            **reward_kwargs,
        )

    # Try to get a custom reward function based on the configuration
    # user defined reward manager can be registered in custom_reward_fn
    compute_score = get_custom_reward_fn(reward_fn_config)
    final_compute_score = compute_score

    default_compute_score_ = default_compute_score_async

    if compute_score is None:
        sandbox_config = reward_kwargs.get("sandbox_fusion", None)
        sandbox_url = sandbox_config.get("url") if sandbox_config else None
        memory_limit_mb = sandbox_config.get("memory_limit_mb", 1024) if sandbox_config else 1024
        if sandbox_url:
            # Create an asyncio.Semaphore to control concurrent access to the sandbox
            # Note: asyncio.Semaphore must be created in the same event loop where it will be used
            # Therefore, we pass max_concurrent as a parameter and create the semaphore later
            max_concurrent = reward_kwargs.get("max_concurrent", 64)
            _concurrent_semaphore = asyncio.Semaphore(max_concurrent)
            final_compute_score = partial(
                default_compute_score_,
                sandbox_fusion_url=sandbox_url,
                concurrent_semaphore=_concurrent_semaphore,
                memory_limit_mb=memory_limit_mb,
            )
        else:
            final_compute_score = default_compute_score_

    return reward_manager_cls(
        config=config,
        tokenizer=tokenizer,
        compute_score=final_compute_score,
        **reward_kwargs,
    )
