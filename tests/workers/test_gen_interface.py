"""CPU tests for GenInterface Optional ps_manager_handle."""

import sys
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.cpu_test

# ---------------------------------------------------------------------------
# Pre-import mocking: inject stubs for every heavy dependency so that
# psrl.workers.gen_dplb.vllm_async_server can be imported on a CPU-only
# machine that has no ray / torch / vllm installed.
# ---------------------------------------------------------------------------

_MOCKED_MODULES = [
    "ray",
    "ray.actor",
    "aiohttp",
    "grpc",
    "grpc_reflection",
    "grpc_reflection.v1alpha",
    "grpc_reflection.v1alpha.reflection",
    "smg_grpc_proto",
    "smg_grpc_proto.vllm_engine_pb2",
    "smg_grpc_proto.vllm_engine_pb2_grpc",
    "smg_grpc_servicer",
    "smg_grpc_servicer.vllm",
    "smg_grpc_servicer.vllm.preemption",
    "smg_grpc_servicer.vllm.servicer",
    "torch",
    "torch.distributed",
    "torch.distributed.tensor",
    "torch.multiprocessing",
    "torch.multiprocessing.reductions",
    "verl",
    "verl.single_controller",
    "verl.single_controller.ray",
    "verl.utils",
    "verl.utils.device",
    "verl.utils.memory_utils",
    "verl.utils.net_utils",
    "verl.utils.profiler",
    "verl.workers",
    "verl.workers.config",
    "verl.workers.rollout",
    "verl.workers.rollout.replica",
    "verl.workers.rollout.utils",
    "verl.workers.rollout.vllm_rollout",
    "verl.workers.rollout.vllm_rollout.utils",
    "verl.workers.rollout.vllm_rollout.vllm_async_server",
    "vllm",
    "vllm.engine",
    "vllm.engine.arg_utils",
    "vllm.entrypoints",
    "vllm.entrypoints.openai",
    "vllm.entrypoints.openai.parser",
    "vllm.entrypoints.openai.parser.harmony_utils",
    "vllm.inputs",
    "vllm.lora",
    "vllm.lora.request",
    "vllm.outputs",
    "vllm.pooling_params",
    "vllm.usage",
    "vllm.usage.usage_lib",
    "vllm.utils",
    "vllm.utils.argparse_utils",
    "vllm.v1",
    "vllm.v1.engine",
    "vllm.v1.engine.async_llm",
    # psrl sub-packages that need heavy deps
    "psrl.utils",
    "psrl.utils.common",
    "psrl.utils.common.http_utils",
    "psrl.utils.logger",
    "psrl.utils.ray",
    "psrl.workers.gen_dplb.stats_collector",
    "psrl.workers.gen_dplb.zmq_queue",
    "psrl.workers.ps",
    "psrl.workers.ps.request_status_tracker",
    # rollout_coordinator pulls in many deps via psrl.utils
    "psrl.workers.gen_dplb.rollout_coordinator",
]

_injected: list[str] = []
for _mod in _MOCKED_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
        _injected.append(_mod)

# Ensure ray.actor.ActorHandle exists as a type
sys.modules["ray"].actor = sys.modules["ray.actor"]
sys.modules["ray.actor"].ActorHandle = object

# Ensure vllm_async_server mock exports the classes we're *replacing*
sys.modules["verl.workers.rollout.vllm_rollout.vllm_async_server"].vLLMHttpServer = object
sys.modules["verl.workers.rollout.vllm_rollout.vllm_async_server"].vLLMReplica = object

# Override gen_dplb __init__ so importing the package doesn't pull in
# RolloutCoordinator (which needs ray/torch at module level).
import psrl.workers.gen_dplb as _gen_dplb_pkg  # noqa: E402

sys.modules["psrl.workers.gen_dplb.rollout_coordinator"] = MagicMock()
_gen_dplb_pkg.RolloutCoordinator = MagicMock()

# Now import GenInterface — the module may already be cached; force a fresh load.
_target = "psrl.workers.gen_dplb.vllm_async_server"
sys.modules.pop(_target, None)

from psrl.workers.gen_dplb.vllm_async_server import GenInterface  # noqa: E402

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_gen_interface_with_ps_manager():
    """GenInterface still works when ps_manager_handle is provided."""
    iface = GenInterface(
        role="rollout",
        rollout_replica_idx=0,
        ps_manager_handle=object(),  # any truthy object
        status_endpoint="tcp://127.0.0.1:28000",
    )
    assert iface.ps_manager_handle is not None
    assert iface.role == "rollout"


def test_gen_interface_without_ps_manager():
    """GenInterface can be constructed with ps_manager_handle=None (reward model path)."""
    iface = GenInterface(
        role="reward_model_MyRM",
        rollout_replica_idx=2,
        ps_manager_handle=None,
        status_endpoint="tcp://127.0.0.1:28001",
    )
    assert iface.ps_manager_handle is None
    assert iface.rollout_replica_idx == 2
