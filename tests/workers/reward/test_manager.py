"""CPU tests for RewardModelManager interface."""

import importlib
import pathlib as _p
import sys
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.cpu_test

# Stub all heavy dependencies
_MOCKED = [
    "ray",
    "ray.actor",
    "ray.util",
    "ray.util.queue",
    "torch",
    "aiohttp",
    "grpc",
    "tensordict",
    "numpy",
    "verl",
    "verl.single_controller",
    "verl.single_controller.ray",
    "verl.utils",
    "verl.utils.device",
    "verl.utils.fs",
    "verl.workers",
    "verl.workers.config",
    "verl.workers.rollout",
    "verl.workers.rollout.replica",
    "verl.workers.rollout.vllm_rollout",
    "verl.workers.rollout.vllm_rollout.vllm_async_server",
    "verl.workers.rollout.utils",
    "psrl.utils.logger",
    "psrl.utils.common",
    "psrl.utils.common.http_utils",
    "psrl.workers.gen",
    "psrl.workers.gen.vllm_async_server",
    "psrl.workers.reward.reward_model.coordinator",
]
for _m in _MOCKED:
    if _m not in sys.modules:
        sys.modules[_m] = MagicMock()
sys.modules["ray"].remote = lambda cls=None, **kw: (cls if cls is not None else lambda c: c)
sys.modules["ray"].actor.ActorHandle = object
sys.modules["verl.workers.rollout.vllm_rollout.vllm_async_server"].vLLMHttpServer = object
sys.modules["verl.workers.rollout.vllm_rollout.vllm_async_server"].vLLMReplica = object


def _load(rel):
    path = _p.Path("/Users/linsh/Desktop/verl_align/psrl") / rel
    spec = importlib.util.spec_from_file_location(rel.replace("/", ".").removesuffix(".py"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_manager_mod = _load("psrl/workers/reward/reward_model/manager.py")
RewardModelManager = _manager_mod.RewardModelManager


def test_manager_requires_gateway_url():
    """RewardModelManager constructor must accept gateway_url parameter."""
    import inspect

    sig = inspect.signature(RewardModelManager.__init__)
    assert "gateway_url" in sig.parameters, "RewardModelManager.__init__ must have a 'gateway_url' parameter"
    assert "status_queues" not in sig.parameters, (
        "RewardModelManager.__init__ must NOT have 'status_queues' (old API)"
    )


def test_manager_has_get_gateway_url():
    """RewardModelManager must expose get_gateway_url()."""
    assert hasattr(RewardModelManager, "get_gateway_url")


def test_manager_has_get_reward_model_tokenizer():
    """RewardModelManager must expose get_reward_model_tokenizer()."""
    assert hasattr(RewardModelManager, "get_reward_model_tokenizer")
