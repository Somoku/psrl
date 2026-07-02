"""CPU tests for GenRewardManager endpoint dispatch and response parsers."""

import importlib.util
import pathlib as _p
import sys
import types as _types
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

pytestmark = pytest.mark.cpu_test

# Pre-import mocking
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
    "psrl.utils.dataset",
    "psrl.utils.dataset.utils",
    "psrl.workers.reward.reward_model.manager",
    "psrl.workers.reward.reward_loop",
    "psrl.workers.reward.reward_loop.base",
]
for _m in _MOCKED:
    if _m not in sys.modules:
        sys.modules[_m] = MagicMock()
sys.modules["ray"].remote = lambda cls=None, **kw: (cls if cls is not None else lambda c: c)
sys.modules["ray"].actor.ActorHandle = object
sys.modules["verl.workers.rollout.vllm_rollout.vllm_async_server"].vLLMHttpServer = object
sys.modules["verl.workers.rollout.vllm_rollout.vllm_async_server"].vLLMReplica = object

_grf_mod = _types.ModuleType("psrl.workers.reward.gen_reward_function")
_grf_pkg = _types.ModuleType("psrl.workers.reward")
sys.modules.setdefault("psrl.workers.reward", _grf_pkg)
sys.modules.setdefault("psrl.workers.reward.gen_reward_function", _grf_mod)


class _DefaultGenRewardFunction:
    using_sys_prompt = True

    def prompt_constructor(self, prompt_str, response_str):
        return f"Q: {prompt_str}\nA: {response_str}"

    def compute_score(self, **kwargs):
        return 1.0


_grf_mod.DefaultGenRewardFunction = _DefaultGenRewardFunction
_grf_mod.GenRewardFunctionBase = object

# Stub register decorator for reward_loop __init__
sys.modules["psrl.workers.reward.reward_loop"].register = lambda name: (lambda cls: cls)


# Stub RewardManagerBase
class _RewardManagerBase:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def _format_request_uid(self, uid):
        return "test-uid"


sys.modules["psrl.workers.reward.reward_loop.base"].RewardManagerBase = _RewardManagerBase

_gen_path = _p.Path("/Users/linsh/Desktop/verl_align/psrl") / "psrl/workers/reward/reward_loop/gen.py"
_spec = importlib.util.spec_from_file_location("psrl.workers.reward.reward_loop.gen", _gen_path)
_gen_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gen_mod)
GenRewardManager = _gen_mod.GenRewardManager


def _make_manager(runner: str, task: str):
    """Construct a GenRewardManager with a mock reward_model_manager."""
    cfg = OmegaConf.create(
        {
            "psrl": {"logging_path": "/tmp"},
            "gen_actor_rollout_ref": {"rollout": {"prompt_length": 512, "response_length": 512}},
        }
    )

    rm_mgr = MagicMock()
    rm_mgr.get_gateway_url.return_value = "http://127.0.0.1:8200"
    rm_mgr.get_reward_model_tokenizer.return_value = MagicMock()
    rm_mgr.reward_model_config = OmegaConf.create(
        {
            "rollout": {"runner": runner, "task": task, "response_length": 512},
            "sampling_config": {"temperature": 1.0, "top_p": -1},
        }
    )

    manager = GenRewardManager.__new__(GenRewardManager)
    manager.config = cfg
    manager.tokenizer = MagicMock()
    manager.reward_model_manager = rm_mgr
    manager.reward_model_tokenizer = MagicMock()
    manager.reward_function = _DefaultGenRewardFunction()
    manager._sampling_config = {"temperature": 1.0, "top_p": -1}
    manager._http_client = None
    manager._rm_response_length = 512
    manager.reward_kwargs = {}
    manager._smg_endpoint, manager._parse_response = manager._resolve_endpoint_and_parser(runner, task)
    return manager


def test_endpoint_dispatch_generate():
    mgr = _make_manager("generate", "generate")
    assert mgr._smg_endpoint == "/v1/completions"


def test_endpoint_dispatch_classify():
    mgr = _make_manager("pooling", "classify")
    assert mgr._smg_endpoint == "/v1/classify"


def test_endpoint_dispatch_embeddings():
    mgr = _make_manager("pooling", "embed")
    assert mgr._smg_endpoint == "/v1/embeddings"


def test_endpoint_dispatch_unknown_raises():
    mgr = GenRewardManager.__new__(GenRewardManager)
    with pytest.raises(ValueError, match="Unsupported"):
        mgr._resolve_endpoint_and_parser("unknown", "task")


def test_parse_completions_response():
    mgr = _make_manager("generate", "generate")
    data = {
        "choices": [{"text": "yes"}],
        "usage": {"completion_tokens": 1},
    }
    result = mgr._parse_completions_response(data, "uid-1")
    assert result["rm_output_str"] == "yes"
    assert result["rm_output_value"] is None
    assert result["rm_output_len"] == 1


def test_parse_classify_response_scalar():
    mgr = _make_manager("pooling", "classify")
    data = {"data": [{"embedding": [0.87]}]}
    result = mgr._parse_classify_response(data, "uid-2")
    assert abs(result["rm_output_value"] - 0.87) < 1e-6
    assert result["rm_output_str"] == ""


def test_parse_classify_response_empty():
    mgr = _make_manager("pooling", "classify")
    result = mgr._parse_classify_response({}, "uid-3")
    assert result["rm_output_value"] is None


def test_parse_embeddings_response():
    mgr = _make_manager("pooling", "embed")
    vec = [0.1, 0.2, 0.3]
    data = {"data": [{"embedding": vec}]}
    result = mgr._parse_embeddings_response(data, "uid-4")
    assert result["rm_output_value"] == vec


def test_build_request_payload_completions():
    mgr = _make_manager("generate", "generate")
    payload = mgr._build_request_payload([1, 2, 3])
    assert payload["prompt"] == [1, 2, 3]
    assert "max_tokens" in payload
    assert "temperature" in payload


def test_build_request_payload_classify():
    mgr = _make_manager("pooling", "classify")
    payload = mgr._build_request_payload([10, 20])
    assert payload == {"input": [10, 20]}
