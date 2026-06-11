"""CPU tests: ElasticExecutor uses RolloutInstanceId keys throughout."""

import enum as _enum
import importlib.util
import pathlib as _p
import sys
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.cpu_test

_MOCKED = [
    "ray",
    "ray.actor",
    "ray.util",
    "ray.util.queue",
    "torch",
    "psrl.utils.logger",
    "psrl.utils.common",
    "psrl.utils.common.memory_utils",
    "psrl.utils.server",
    "psrl.utils.server.command",
    "psrl.trainer.ppo.utils",
    "psrl.utils.elastic_rm.scaling_policy",
    "psrl.utils.elastic_rm.diagnostics",
    "psrl.workers.gen.utils",
]
for _m in _MOCKED:
    if _m not in sys.modules:
        sys.modules[_m] = MagicMock()
sys.modules["ray"].remote = lambda cls=None, **kw: (cls if cls is not None else lambda c: c)
sys.modules["psrl.workers.gen.utils"].RolloutInstanceId = tuple


class _InstanceStatus(_enum.Enum):
    ASLEEP = _enum.auto()
    AWAKEN = _enum.auto()


_policy_mod = sys.modules["psrl.utils.elastic_rm.scaling_policy"]
_policy_mod.InstanceStatus = _InstanceStatus


class _FakePolicy:
    min_awake_per_role = 0

    def decide(self, *a, **kw):
        return MagicMock(actions=[])


_policy_mod.ScalingPolicy = lambda **kw: _FakePolicy()
_policy_mod.InstanceSignal = MagicMock


def _load(rel):
    path = _p.Path("/Users/linsh/Desktop/verl_align/psrl") / rel
    spec = importlib.util.spec_from_file_location(rel.replace("/", ".").removesuffix(".py"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ee = _load("psrl/utils/elastic_rm/elastic_executor.py")
ElasticExecutor = _ee.ElasticExecutor


def _make_executor():
    role = MagicMock()
    model = "test_model"
    coord = MagicMock()
    ex = ElasticExecutor.__new__(ElasticExecutor)
    ex.coordinators = {role: {model: coord}}
    ex.roles = [(role, model)]
    ex.instances_status_flags = {}
    ex.instances_engine_stats = {}
    ex.instance_gpu_mappings = {}
    ex.gpu_to_instances = {}
    ex.scaling_policy = _FakePolicy()
    ex.elastic_rm_config = {}
    ex._post_scale_up_abort_waiting_ratio = 0.0
    ex._coordinator_command_timeout_s = None
    ex._coordinator_sync_timeout_s = 60.0
    ex._decision_execution_in_progress = False
    ex._next_decision_id = 1
    ex._decision_pending_action_counts = {}
    ex._execution_in_progress_stall_ticks = 0
    ex._decision_abandon_stall_ticks = 0
    ex.router_backlog_by_role = {}
    ex.trainer_waiting_hint = {}
    ex._last_monitor_instance_log_ms = 0.0
    ex._monitor_instance_log_interval_ms = 5000
    ex._enable_monitor_instance_log = False
    return ex, role, model


def test_register_instances_accepts_rollout_instance_id_list():
    """register_instances must accept list[RolloutInstanceId], not num_instances int."""
    ex, role, model = _make_executor()
    ids = [("wid-0", 0), ("wid-0", 1), ("wid-1", 0)]
    ex.register_instances(role, model, ids)
    assert ("wid-0", 0) in ex.instances_status_flags[role][model]
    assert ("wid-0", 1) in ex.instances_status_flags[role][model]
    assert ("wid-1", 0) in ex.instances_status_flags[role][model]


def test_register_instances_rejects_int():
    """register_instances must NOT accept a plain integer (old API)."""
    ex, role, model = _make_executor()
    with pytest.raises((TypeError, AttributeError)):
        ex.register_instances(role, model, 3)  # old API: int


def test_register_instance_gpu_mapping_uses_tuple_key():
    ex, role, model = _make_executor()
    ids = [("wid-0", 0)]
    ex.register_instances(role, model, ids)
    ex.register_instance_gpu_mapping(role, model, ("wid-0", 0), gpu_ids=[0], node_id="node1")
    mapping = ex.instance_gpu_mappings[role][model][("wid-0", 0)]
    assert mapping["node_id"] == "node1"
    assert mapping["gpu_ids"] == [0]


def test_initialize_instance_states_with_tuple_ids():
    ex, role, model = _make_executor()
    ids = [("wid-0", 0), ("wid-1", 0)]
    ex.register_instances(role, model, ids)
    awaken = [{"role_name": role, "model_name": model, "instance_id": ("wid-0", 0)}]
    ex.initialize_instance_states(awaken)
    assert ex.instances_status_flags[role][model][("wid-0", 0)] == _ee.InstanceStatus.AWAKEN
    assert ex.instances_status_flags[role][model][("wid-1", 0)] == _ee.InstanceStatus.ASLEEP


def test_sync_engine_status_stores_tuple_key():
    """Engine stats can be stored and retrieved with RolloutInstanceId tuple key."""
    ex, role, model = _make_executor()
    ids = [("wid-0", 0)]
    ex.register_instances(role, model, ids)
    role_stats = ex.instances_engine_stats.setdefault(role, {}).setdefault(model, {})
    snapshot = {"instance_id": ("wid-0", 0), "scheduler_stats": {}, "generation_throughput": 0.0}
    role_stats[("wid-0", 0)] = snapshot
    assert ("wid-0", 0) in ex.instances_engine_stats[role][model]
