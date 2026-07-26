"""CPU tests for RolloutCoordinator elastic hooks and RewardModelCoordinator overrides."""

import asyncio
import importlib
import importlib.util
import pathlib
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.cpu_test

# ── Heavy-dep stubs ──────────────────────────────────────────────────────────
_MOCKED = [
    "ray",
    "ray.actor",
    "ray.util",
    "ray.util.queue",
    "aiohttp",
    "torch",
    "numpy",
    "psrl.utils.logger",
    "psrl.utils.common",
    "psrl.utils.common.http_utils",
    "psrl.utils.elastic_rm",
    "psrl.utils.elastic_rm.diagnostics",
    "psrl.utils.server",
    "psrl.utils.server.command",
    "psrl.workers.gen.stats_collector",
    "psrl.workers.gen.utils",
    "psrl.workers.gen.zmq_queue",
]
for _m in _MOCKED:
    if _m not in sys.modules:
        sys.modules[_m] = MagicMock()

sys.modules["ray"].remote = lambda cls=None, **kw: (cls if cls is not None else lambda c: c)
sys.modules["ray"].actor.ActorHandle = object
sys.modules["ray.util"].get_node_ip_address = MagicMock(return_value="127.0.0.1")
sys.modules["psrl.workers.gen.utils"].RolloutInstanceId = tuple
sys.modules["psrl.workers.gen.utils"].DEFAULT_MAX_CONNECTIONS = 100
sys.modules["psrl.workers.gen.utils"].DEFAULT_TIMEOUT = 60.0


# Stub CommandExtension
class _CommandExtension:
    def __init__(self):
        import queue as _q

        self.command_queue = _q.Queue()

    def _complete_command(self, cmd_id, result):
        pass


sys.modules["psrl.utils.server.command"].CommandExtension = _CommandExtension
sys.modules["psrl.utils.server.command"].Command = MagicMock
sys.modules["psrl.utils.server.command"].CommandType = MagicMock

RolloutCoordinator = importlib.import_module("psrl.workers.gen.rollout_coordination").RolloutCoordinator
_COORDINATOR_PATH = (
    pathlib.Path(__file__).parent.parent.parent.parent
    / "psrl"
    / "workers"
    / "reward"
    / "reward_model"
    / "coordinator.py"
)
_COORDINATOR_SPEC = importlib.util.spec_from_file_location("reward_model_coordinator_under_test", _COORDINATOR_PATH)
_COORDINATOR_MODULE = importlib.util.module_from_spec(_COORDINATOR_SPEC)
_COORDINATOR_SPEC.loader.exec_module(_COORDINATOR_MODULE)
RewardModelCoordinator = _COORDINATOR_MODULE.RewardModelCoordinator


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_server_handle():
    """Build a server handle mock where .nixl_sleep.remote and .nixl_wake_up.remote are AsyncMock."""
    handle = MagicMock()
    handle.nixl_sleep.remote = AsyncMock(return_value=None)
    handle.nixl_wake_up.remote = AsyncMock(return_value=None)
    return handle


def _make_rollout_coord():
    """Build a minimal RolloutCoordinator stub instance (bypass __init__)."""
    coord = object.__new__(RolloutCoordinator)
    coord.instance_ids = {("wid-0", 0), ("wid-0", 1), ("wid-1", 0)}
    coord.server_handles = {
        "wid-0": _make_server_handle(),
        "wid-1": _make_server_handle(),
    }
    return coord


def _make_reward_coord():
    coord = object.__new__(RewardModelCoordinator)
    coord.instance_ids = {("rm-0", 0), ("rm-1", 0)}

    def _make_rm_handle():
        handle = MagicMock()
        handle.sleep.remote = AsyncMock(return_value=None)
        handle.wake_up.remote = AsyncMock(return_value=None)
        handle.nixl_sleep.remote = AsyncMock(return_value=None)
        handle.nixl_wake_up.remote = AsyncMock(return_value=None)
        return handle

    coord.server_handles = {
        "rm-0": _make_rm_handle(),
        "rm-1": _make_rm_handle(),
    }
    coord.rollout_gateway_url = ""
    coord.gateway_client = None
    return coord


# ── Task 1 tests ─────────────────────────────────────────────────────────────


def test_get_all_instance_ids_returns_sorted_tuples():
    coord = _make_rollout_coord()
    ids = coord.get_all_instance_ids()
    assert isinstance(ids, list)
    assert ids == sorted([("wid-0", 0), ("wid-0", 1), ("wid-1", 0)])
    for iid in ids:
        assert isinstance(iid, tuple) and len(iid) == 2


def test_get_sleep_level_rollout_returns_2():
    coord = _make_rollout_coord()
    assert coord._get_sleep_level() == 2


def test_do_sleep_instance_calls_nixl_sleep():
    coord = _make_rollout_coord()
    coord._get_sleep_level = lambda: 2
    asyncio.get_event_loop().run_until_complete(coord._do_sleep_instance("wid-0"))
    coord.server_handles["wid-0"].nixl_sleep.remote.assert_called_once_with(level=2)


def test_do_wake_up_instance_calls_nixl_wake_up():
    coord = _make_rollout_coord()
    asyncio.get_event_loop().run_until_complete(coord._do_wake_up_instance("wid-0"))
    coord.server_handles["wid-0"].nixl_wake_up.remote.assert_called_once_with()


# ── Task 2 tests ─────────────────────────────────────────────────────────────


def test_reward_get_sleep_level_returns_1():
    coord = _make_reward_coord()
    assert coord._get_sleep_level() == 1


def test_reward_do_sleep_instance_calls_plain_sleep():
    """RewardModelCoordinator.sleep uses server.sleep (not nixl_sleep)."""
    coord = _make_reward_coord()
    asyncio.get_event_loop().run_until_complete(coord._do_sleep_instance("rm-0"))
    coord.server_handles["rm-0"].sleep.remote.assert_called_once_with(level=1)
    coord.server_handles["rm-0"].nixl_sleep.remote.assert_not_called()


def test_reward_do_wake_up_instance_calls_plain_wake_up():
    """RewardModelCoordinator.wake_up uses server.wake_up (not nixl_wake_up)."""
    coord = _make_reward_coord()
    asyncio.get_event_loop().run_until_complete(coord._do_wake_up_instance("rm-0"))
    coord.server_handles["rm-0"].wake_up.remote.assert_called_once_with()
    coord.server_handles["rm-0"].nixl_wake_up.remote.assert_not_called()


def test_get_router_backlog_size_no_url_returns_0():
    """Without gateway URL, backlog size returns 0 immediately."""
    coord = _make_reward_coord()
    coord.rollout_gateway_url = ""
    result = asyncio.get_event_loop().run_until_complete(coord.get_router_backlog_size())
    assert result == 0


def test_get_router_backlog_size_sums_worker_loads():
    """get_router_backlog_size sums 'load' field from GET /workers response."""
    coord = _make_reward_coord()
    coord.rollout_gateway_url = "http://127.0.0.1:8300"

    # Mock _gateway_get_json to return a two-worker response
    async def _mock_get(path):
        return {"workers": [{"id": "w1", "load": 3}, {"id": "w2", "load": 5}]}

    coord._gateway_get_json = _mock_get
    result = asyncio.get_event_loop().run_until_complete(coord.get_router_backlog_size())
    assert result == 8
