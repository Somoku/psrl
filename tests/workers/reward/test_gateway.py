"""CPU tests for RewardModelGateway router args construction."""

import importlib.util
import pathlib
import sys
import types as _types
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

pytestmark = pytest.mark.cpu_test

# ---------------------------------------------------------------------------
# Pre-import mocking: inject stubs for ray and other heavy deps so that
# psrl.workers.reward.reward_model.gateway can be imported on a CPU-only
# machine that has no ray / torch installed.
# ---------------------------------------------------------------------------

_MOCKED_MODULES = [
    "ray",
    "ray.util",
    "ray.util.queue",
    "torch",
    "torch.distributed",
    "vllm",
    "vllm.engine",
    # Stub out psrl.utils.logger so we don't pull in verl/tensordict/torch
    "psrl.utils.logger",
    "psrl.utils.common",
    "psrl.utils.common.http_utils",
    "smg",
    "smg.launch_router",
]

for _mod in _MOCKED_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Make ray.remote a pass-through decorator
sys.modules["ray"].remote = lambda cls=None, **kwargs: (cls if cls is not None else lambda c: c)

# Provide a real find_available_port stub that returns a port integer
sys.modules["psrl.utils.common.http_utils"].find_available_port = lambda base_port=8000: base_port

# Provide DualOutputHandler stub
sys.modules["psrl.utils.logger"].DualOutputHandler = MagicMock(return_value=MagicMock())

# Return the argparse namespace from RouterArgs.from_cli_args so this CPU test can
# inspect the values without importing the Rust extension.
sys.modules["smg.launch_router"].RouterArgs.from_cli_args = lambda args, use_router_prefix=False: args

_rm_pkg = _types.ModuleType("psrl.workers.reward.reward_model")
_rm_pkg.RewardModelManager = MagicMock()
_rm_pkg.RewardModelReplica = MagicMock()
sys.modules["psrl.workers.reward.reward_model"] = _rm_pkg

_gateway_path = (
    pathlib.Path(__file__).parent.parent.parent.parent / "psrl" / "workers" / "reward" / "reward_model" / "gateway.py"
)
_spec = importlib.util.spec_from_file_location("psrl.workers.reward.reward_model.gateway", _gateway_path)
_gateway_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gateway_mod)
RewardModelGateway = _gateway_mod.RewardModelGateway


def _make_config():
    return OmegaConf.create(
        {
            "psrl": {
                "logging_path": "/tmp/psrl_test_logs",
            }
        }
    )


def test_gateway_init():
    """RewardModelGateway can be instantiated with config + model_name."""
    cfg = _make_config()
    gw = RewardModelGateway.__new__(RewardModelGateway)
    gw.__init__(cfg, "TestRM")
    assert gw.model_name == "TestRM"
    assert gw.smg_url is None


def test_gateway_router_args_policy():
    """_init_router_args sets policy to round_robin and disables PSRL routing."""
    cfg = _make_config()
    gw = RewardModelGateway.__new__(RewardModelGateway)
    gw.__init__(cfg, "TestRM")
    gw.smg_ip = "127.0.0.1"
    gw.smg_port = 8300

    args = gw._init_router_args()
    assert args.policy == "round_robin"
    assert args.enable_routing_loop is False
    assert args.worker_selection_strategy == "naive"


def test_gateway_shutdown_noop_when_not_started():
    """shutdown_router is a no-op when router was never started."""
    cfg = _make_config()
    gw = RewardModelGateway.__new__(RewardModelGateway)
    gw.__init__(cfg, "TestRM")
    gw.shutdown_router()  # must not raise
