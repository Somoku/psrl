"""
Introspection helpers for vLLM modules used by the weight converters.

vLLM exposes tensor and expert parallel sizes differently per module. The
parallel linear layers carry `tp_size` directly. The fused MoE expert module
`RoutedExperts` keeps them in `moe_config.moe_parallel_config`. These helpers
give the converters one uniform accessor for both shapes.
"""

import torch.nn as nn
from vllm.model_executor.layers.fused_moe import RoutedExperts


def is_moe_experts_module(module: object) -> bool:
    """
    Return whether a module owns fused MoE expert weights.

    Args:
        module (object): The module to test.

    Returns:
        bool: True for `RoutedExperts` and its subclasses, which is where every
            vLLM MoE architecture registers `w13_weight` and `w2_weight`.
    """
    return isinstance(module, RoutedExperts)


def get_module_tp_size(module: nn.Module | None) -> int:
    """
    Return a module's tensor parallel size.

    Args:
        module (nn.Module | None): The module to inspect.

    Returns:
        int: The parallel size, or 1 when the module is unknown or unsharded.
    """
    if module is None:
        return 1
    tp_size = getattr(module, "tp_size", None)
    if tp_size is not None:
        return int(tp_size)
    if is_moe_experts_module(module):
        return int(module.moe_config.moe_parallel_config.tp_size)
    return 1


def get_module_ep_size(module: nn.Module | None) -> int:
    """
    Return a module's expert parallel size.

    Args:
        module (nn.Module | None): The module to inspect.

    Returns:
        int: The parallel size, or 1 when the module uses no expert parallelism.
    """
    if module is None:
        return 1
    ep_size = getattr(module, "ep_size", None)
    if ep_size is not None:
        return int(ep_size)
    if is_moe_experts_module(module):
        return int(module.moe_config.moe_parallel_config.ep_size)
    return 1
