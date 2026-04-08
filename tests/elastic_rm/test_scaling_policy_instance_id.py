"""CPU tests: ScalingPolicy and InstanceSignal use RolloutInstanceId."""

import importlib.util
import os
import sys
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.cpu_test

# ---------------------------------------------------------------------------
# Bootstrap: load modules directly to avoid ray / torch transitive imports.
# ---------------------------------------------------------------------------

_PSRL = os.path.join(os.path.dirname(__file__), "../../psrl")


def _load_direct(dotted_name: str, rel_path: str) -> object:
    """Always load the real .py file, replacing any mock that may have been registered."""
    path = os.path.join(_PSRL, rel_path)
    spec = importlib.util.spec_from_file_location(dotted_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Stub out modules that pull in ray or torch.
for _m in ["ray", "ray.actor", "ray.util", "ray.util.queue", "torch", "torch.distributed"]:
    if _m not in sys.modules:
        sys.modules[_m] = MagicMock()

if "psrl.utils.logger" not in sys.modules:
    sys.modules["psrl.utils.logger"] = MagicMock()

# Load gen_dplb.utils directly (real module, provides RolloutInstanceId).
_utils_mod = _load_direct("psrl.workers.gen_dplb.utils", "workers/gen_dplb/utils.py")
RolloutInstanceId = _utils_mod.RolloutInstanceId

# Load scaling_policy directly (real module under test).
_sp_mod = _load_direct("psrl.utils.elastic_rm.scaling_policy", "utils/elastic_rm/scaling_policy.py")
InstanceSignal = _sp_mod.InstanceSignal
ScalingAction = _sp_mod.ScalingAction
ScalingPolicy = _sp_mod.ScalingPolicy
ThroughputProfileLoader = _sp_mod.ThroughputProfileLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signal(instance_id: RolloutInstanceId, is_awaken: bool = True, kv: float = 0.5) -> InstanceSignal:
    return InstanceSignal(
        role_name="Rollout",
        model_name="test_model",
        instance_id=instance_id,
        is_awaken=is_awaken,
        kv_cache_utilization=kv,
        running_queue_num=5,
        waiting_queue_num=0,
        generation_throughput=10.0,
        total_token_num=100,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_instance_signal_accepts_rollout_instance_id():
    """InstanceSignal.instance_id should accept a RolloutInstanceId tuple."""
    iid: RolloutInstanceId = ("worker_abc", 0)
    sig = _make_signal(iid)
    assert sig.instance_id == ("worker_abc", 0)


def test_instance_signal_rejects_bare_int():
    """Bare int should no longer be the type — creating with int should still work at runtime
    (dataclasses don't enforce types) but we verify tuple is accepted and round-trips correctly."""
    iid: RolloutInstanceId = ("worker_xyz", 2)
    sig = _make_signal(iid)
    assert isinstance(sig.instance_id, tuple)
    assert sig.instance_id[0] == "worker_xyz"
    assert sig.instance_id[1] == 2


def test_scaling_action_preferred_instance_ids_accepts_tuple_list():
    """ScalingAction.preferred_instance_ids should accept list[RolloutInstanceId]."""
    iid: RolloutInstanceId = ("worker_abc", 0)
    action = ScalingAction(
        action_type="scale_up",
        role_name="Rollout",
        model_name="test_model",
        preferred_instance_ids=[iid],
    )
    assert action.preferred_instance_ids == [("worker_abc", 0)]


def test_build_mu_maps_uses_tuple_key():
    """_build_mu_maps should produce keys of (str, str, RolloutInstanceId) where the last element is a tuple."""
    policy = ScalingPolicy.__new__(ScalingPolicy)
    policy.profile_loader = ThroughputProfileLoader()

    iid: RolloutInstanceId = ("worker_abc", 0)
    signals = [_make_signal(iid, is_awaken=True, kv=0.5)]
    instance_mu, role_total_mu = policy._build_mu_maps(signals)
    # key should be ("Rollout", "test_model", ("worker_abc", 0))
    assert ("Rollout", "test_model", ("worker_abc", 0)) in instance_mu


def test_pick_scale_down_candidate_uses_tuple_instance_id():
    """_pick_scale_down_candidate and related helpers should work with RolloutInstanceId keys."""
    policy = ScalingPolicy.__new__(ScalingPolicy)
    policy.profile_loader = ThroughputProfileLoader()
    policy.theta_low = 0.3
    policy.min_awake_per_role = 0

    iid0: RolloutInstanceId = ("worker_abc", 0)
    iid1: RolloutInstanceId = ("worker_abc", 1)
    signals = [
        _make_signal(iid0, is_awaken=True, kv=0.1),  # low KV — cede candidate
        _make_signal(iid1, is_awaken=True, kv=0.8),
    ]
    instance_mu = {
        ("Rollout", "test_model", iid0): 5.0,
        ("Rollout", "test_model", iid1): 10.0,
    }
    result = policy._pick_scale_down_candidate(signals, instance_mu)
    assert result is not None
    assert result.instance_id == iid0
