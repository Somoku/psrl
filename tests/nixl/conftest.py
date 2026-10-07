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

# Exclude the script-style / multi-node tests from pytest collection.
collect_ignore = [
    "test_nixl_e2e.py",
    "test_nixl_meta_server_comm.py",
    "test_send_recv.py",
    "test_send_recv_model.py",
]
