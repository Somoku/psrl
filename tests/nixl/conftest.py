# tests/nixl/conftest.py
"""
Conftest for nixl tests.

CPU-safe (pytestmark = cpu_test): test_nixl_sharding.py, test_comm_planner.py,
test_comm_plan_invariants.py, test_shard_algebra_properties.py, test_coord_judge.py,
test_xfer_tag.py, test_join_admission.py, test_meta_server_sharding.py,
test_post_wait_split.py, test_config_defenses.py.

Single-node GPU tests, collected normally but skipping without CUDA:
test_colocate_ipc_gpu.py, test_fp8_wire_gpu.py, test_cost_model.py.

test_descriptor_correctness_gpu.py also runs under pytest but pays ~120s once for NIXL
agent setup; see conftest_agents.py for why.

test_nixl_e2e.py, test_nixl_meta_server_comm.py, test_send_recv.py,
test_send_recv_model.py require a full nixl + GPU + multi-node environment
and are designed to be run as scripts (not via pytest in CI).

test_fsdp2_converter_gpu.py self-launches via torchrun; pytest drives it but the real
work happens in child processes.
"""

import os
from pathlib import Path

# Exclude the script-style / multi-node tests from pytest collection.
collect_ignore = [
    "test_nixl_e2e.py",
    "test_nixl_meta_server_comm.py",
    "test_send_recv.py",
    "test_send_recv_model.py",
]

#: Where the tests that read real checkpoints look for them.
#:
#: Several tests ground their numbers in real weight maps rather than synthetic key lists
#: -- MoE grouping and the cost model both do. That means a filesystem path, and a
#: hardcoded one ties the suite to whichever volume happened to hold the models when it
#: was written. `PSRL_MODEL_ROOT` lets a checkout point at its own copy; the tests skip
#: when a model is absent, so an unset variable degrades to fewer tests, not failures.
#:
#: Only metadata is needed for the large models -- `model.safetensors.index.json` lists
#: every key, so a weight map costs a megabyte instead of sixty gigabytes. Models without
#: an index are read through `safetensors`, which does need the real file.
#:
#: The fallback is `models/` at the repo root, which is gitignored and is normally a
#: symlink to wherever the checkpoints actually live.
MODEL_ROOT = Path(
    os.environ.get(
        "PSRL_MODEL_ROOT",
        str(Path(__file__).resolve().parents[2] / "models"),
    )
)
