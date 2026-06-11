"""CPU tests for RewardModelCoordinator."""

import asyncio
import importlib.util
import pathlib
import sys
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.cpu_test

# Pre-import mocking: stub heavy deps before importing the coordinator
_MOCKED_MODULES = [
    "ray",
    "ray.actor",
    "aiohttp",
    "torch",
    "numpy",
    "psrl.utils.common",
    "psrl.utils.common.http_utils",
    "psrl.utils.logger",
    "psrl.utils.elastic_rm",
    "psrl.utils.elastic_rm.diagnostics",
    "psrl.utils.server",
    "psrl.utils.server.command",
    "psrl.workers.gen.stats_collector",
    "psrl.workers.gen.utils",
    "psrl.workers.gen.zmq_queue",
    "psrl.workers.gen",
    "psrl.workers.gen.rollout_coordinator",
]
for _mod in _MOCKED_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Make ray.remote a pass-through decorator
sys.modules["ray"].remote = lambda cls=None, **kwargs: (cls if cls is not None else lambda c: c)
sys.modules["ray"].actor = sys.modules["ray.actor"]
sys.modules["ray.actor"].ActorHandle = object


# Create a real CommandExtension stub (not MagicMock) so inheritance works
class _CommandExtensionStub:
    def __init__(self):
        import queue

        self.command_queue = queue.Queue()

    def _complete_command(self, cmd_id, result):
        pass


sys.modules["psrl.utils.server.command"].CommandExtension = _CommandExtensionStub
sys.modules["psrl.utils.server.command"].Command = MagicMock
sys.modules["psrl.utils.server.command"].CommandType = MagicMock


# Create a real RolloutCoordinator stub so we can test inheritance
class _RolloutCoordinatorStub(_CommandExtensionStub):
    def __init__(self, config, ps_manager, rollout_router):
        super().__init__()
        self.config = config
        self.ps_manager = ps_manager
        self.rollout_router = rollout_router

    async def sync_model(self):
        return "base_sync"

    async def update_model_version(self, model_version):
        return "base_update"

    async def _greedy_sync_and_migrate_loop(self):
        return "base_greedy"

    async def _status_based_sync_and_migrate_loop(self):
        return "base_status"

    def get_status_sink_endpoint(self):
        return "tcp://127.0.0.1:28000"

    def add_worker(self, *args, **kwargs):
        pass

    async def start_busy_loop(self):
        pass


sys.modules["psrl.workers.gen.rollout_coordinator"].RolloutCoordinator = _RolloutCoordinatorStub

# Now stub the gen package __init__
sys.modules["psrl.workers.gen"].RolloutCoordinator = _RolloutCoordinatorStub

_coord_path = (
    pathlib.Path(__file__).parent.parent.parent.parent
    / "psrl"
    / "workers"
    / "reward"
    / "reward_model"
    / "coordinator.py"
)
_spec = importlib.util.spec_from_file_location("psrl.workers.reward.reward_model.coordinator", _coord_path)
_coord_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_coord_mod)
RewardModelCoordinator = _coord_mod.RewardModelCoordinator


def _make_config():
    from unittest.mock import MagicMock

    cfg = MagicMock()
    cfg.psrl.logging_path = "/tmp"
    return cfg


def _make_rm_config():
    rm_cfg = MagicMock()
    rm_cfg.reward_model_name = "TestRM"
    return rm_cfg


def test_coordinator_is_subclass():
    """RewardModelCoordinator must be a subclass of RolloutCoordinator."""
    assert issubclass(RewardModelCoordinator, _RolloutCoordinatorStub)


def test_sync_model_is_noop():
    """sync_model() must be a coroutine returning None (no PS interaction)."""
    coord = object.__new__(RewardModelCoordinator)
    # Initialize minimal state
    coord.config = _make_config()
    coord.rm_config = _make_rm_config()
    coord.reward_model_name = "TestRM"
    result = asyncio.get_event_loop().run_until_complete(coord.sync_model())
    assert result is None


def test_update_model_version_is_noop():
    """update_model_version() must be a coroutine returning None."""
    coord = object.__new__(RewardModelCoordinator)
    coord.config = _make_config()
    coord.rm_config = _make_rm_config()
    coord.reward_model_name = "TestRM"
    result = asyncio.get_event_loop().run_until_complete(coord.update_model_version(model_version=5))
    assert result is None
