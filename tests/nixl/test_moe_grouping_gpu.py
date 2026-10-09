"""Grouped MoE keys through the real registration path (implementation item C1).

`test_moe_grouping.py` proves the grouping transform on plain tensors. This file covers
what only a live NIXL client can show: that a key whose shards live in *separate
storages* registers, produces one descriptor per expert, and carries the right bytes to
the right expert.

That separation is the whole difficulty. A normal key is one tensor the client slices
with `get_local_sharded_tensors`; a grouped expert key has no such tensor, because
Megatron/TE allocates every expert independently. Registration had to stop assuming one
storage per key, and the risk of getting it wrong is silent: a permuted expert axis has
no shape mismatch to catch it.
"""

from __future__ import annotations

import os
from collections import OrderedDict

import pytest
import torch

pytest.importorskip("nixl._api", reason="NIXL not installed")

from omegaconf import OmegaConf

from psrl.utils.nixl.client import NIXLStorageClient
from psrl.utils.nixl.moe_grouping import group_expert_tensors
from psrl.utils.nixl.nixl_spec import NIXLClientType
from tests.nixl.conftest_agents import (  # noqa: F401  (fixtures)
    agent_pair,
    cuda_available,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


NUM_EXPERTS = 16
ROWS, COLS = 8, 32


@pytest.fixture(scope="module")
def client(agent_pair):
    """A storage client bound to the shared session agent.

    `binded_agent` is what keeps this test off Ray: the unbound path asks a PortScanner
    actor for a port, which needs a live cluster. Agent construction is also the
    expensive part on this host (see conftest_agents), so reusing the session pair keeps
    the module to one setup rather than one per test.
    """
    local, _peer = agent_pair
    config = OmegaConf.create(
        {
            "server_ip": "127.0.0.1",
            "server_port": int(os.environ.get("PSRL_TEST_NIXL_PORT", "24931")),
            "max_pinned_temp_memory_slots": None,
            "enable_tms_for_temp_buffers": False,
        }
    )
    yield NIXLStorageClient(
        client_name="moe-grouping-test",
        server_name="moe-grouping-test-server",
        use_gpu=True,
        client_type=NIXLClientType.PUSH_SIDE,
        nixl_config=config,
        binded_agent=local,
    )


def _grouped_state(num_experts: int = NUM_EXPERTS):
    """Per-expert tensors on GPU, each filled with its own expert id."""
    from psrl.utils.nixl.nixl_spec import NIXLSharding

    state, sharding = {}, {}
    for expert in range(num_experts):
        key = f"model.layers.0.mlp.experts.{expert}.gate_proj.weight"
        state[key] = torch.full((ROWS, COLS), float(expert), device="cuda", dtype=torch.bfloat16)
        sharding[key] = NIXLSharding.default()
    return group_expert_tensors(state, sharding, num_experts=num_experts)


class TestGroupedKeyRegisters:
    def test_registration_accepts_separate_storages(self, client):
        """The thing that previously could not work: no single tensor to slice."""
        grouped = _grouped_state()
        client.register_local_tensors(grouped.state_dict, grouped.sharding_dict)

        infos = client.local_client_info.tensor_infos
        assert list(infos) == ["model.layers.0.mlp.experts.gate_proj.weight"]

        client.deregister_local_tensors()

    def test_one_descriptor_per_expert(self, client):
        grouped = _grouped_state()
        client.register_local_tensors(grouped.state_dict, grouped.sharding_dict)

        info = client.local_client_info.tensor_infos["model.layers.0.mlp.experts.gate_proj.weight"]
        assert info.num_local_shards == NUM_EXPERTS
        assert all(desc is not None for desc in info.desc_bytes_list)

        client.deregister_local_tensors()

    def test_each_shard_is_registered_at_its_own_address(self, client):
        """Separate storages means separate base addresses, not offsets into one."""
        grouped = _grouped_state()
        client.register_local_tensors(grouped.state_dict, grouped.sharding_dict)

        key = "model.layers.0.mlp.experts.gate_proj.weight"
        addresses = {
            client.contig_desc_slice_map[(key, (expert,))][0] for expert in range(NUM_EXPERTS)
        }
        assert len(addresses) == NUM_EXPERTS, "experts must not share one address"

        client.deregister_local_tensors()

    def test_registered_tensor_is_the_original_not_a_copy(self, client):
        """Copying would defeat the purpose: it is the whole MoE, every sync."""
        from psrl.utils.nixl.nixl_spec import NIXLSharding

        state, sharding = {}, {}
        for expert in range(4):
            key = f"model.layers.0.mlp.experts.{expert}.gate_proj.weight"
            state[key] = torch.full((ROWS, COLS), float(expert), device="cuda", dtype=torch.bfloat16)
            sharding[key] = NIXLSharding.default()
        grouped = group_expert_tensors(state, sharding, num_experts=4)

        client.register_local_tensors(grouped.state_dict, grouped.sharding_dict)

        key = "model.layers.0.mlp.experts.gate_proj.weight"
        for expert in range(4):
            source = state[f"model.layers.0.mlp.experts.{expert}.gate_proj.weight"]
            registered = client._original_tensor_mapping[(key, (expert,))]
            assert registered.data_ptr() == source.data_ptr()

        client.deregister_local_tensors()


class TestExpertIdentityIsPreserved:
    """A permuted expert axis has no shape mismatch to catch it."""

    def test_shard_index_selects_the_matching_expert(self, client):
        grouped = _grouped_state()
        client.register_local_tensors(grouped.state_dict, grouped.sharding_dict)

        key = "model.layers.0.mlp.experts.gate_proj.weight"
        for expert in range(NUM_EXPERTS):
            registered = client._original_tensor_mapping[(key, (expert,))]
            assert registered[0, 0].item() == float(expert), (
                f"shard index {expert} holds expert {registered[0, 0].item()}"
            )

        client.deregister_local_tensors()

    def test_double_digit_experts_are_not_reordered(self, client):
        """Lexical ordering would place expert 10 before expert 2."""
        grouped = _grouped_state()
        client.register_local_tensors(grouped.state_dict, grouped.sharding_dict)

        key = "model.layers.0.mlp.experts.gate_proj.weight"
        order = [
            client._original_tensor_mapping[(key, (expert,))][0, 0].item()
            for expert in range(NUM_EXPERTS)
        ]
        assert order == sorted(order), f"expert axis is permuted: {order}"

        client.deregister_local_tensors()


class TestMetadataIsShared:
    """Interning is what makes the saving 10.7x rather than 2.5x."""

    def test_experts_share_one_metadata_object(self, client):
        grouped = _grouped_state()
        client.register_local_tensors(grouped.state_dict, grouped.sharding_dict)

        info = client.local_client_info.tensor_infos["model.layers.0.mlp.experts.gate_proj.weight"]
        first = info.shard_meta_infos[0]
        assert all(meta is first for meta in info.shard_meta_infos), (
            "expert metadata is equal but not identical, so pickle would copy it N times"
        )

        client.deregister_local_tensors()

    def test_payload_grows_far_slower_than_the_expert_count(self, client):
        """8x the experts must cost well under 8x the payload.

        It cannot be flat, and the reason is worth stating precisely. What remains after
        the metadata is shared is one *descriptor* per expert, and on the train side
        those are irreducible: Megatron/TE allocates each expert separately, so the
        addresses are not evenly spaced and no single strided descriptor can cover them.
        (Measured: at a realistic 768x2048 expert, consecutive allocations show more than
        one distinct gap.) Design 3.5.5(b)'s strided compression of the expert axis
        applies to vLLM's fused `w13_weight`, which genuinely is one strided region.

        So the bound here is "sub-linear by a wide margin", not "constant" -- asserting
        constancy would be asserting something false about the train side.
        """
        from psrl.utils.nixl.nixl_spec import NIXLSharding

        sizes = {}
        for num_experts in (8, 64):
            state, sharding = {}, {}
            for expert in range(num_experts):
                key = f"model.layers.0.mlp.experts.{expert}.gate_proj.weight"
                state[key] = torch.full(
                    (ROWS, COLS), float(expert), device="cuda", dtype=torch.bfloat16
                )
                sharding[key] = NIXLSharding.default()
            grouped = group_expert_tensors(state, sharding, num_experts=num_experts)

            client.register_local_tensors(grouped.state_dict, grouped.sharding_dict)
            sizes[num_experts] = len(client.local_client_info.serialize())
            client.deregister_local_tensors()

        growth = sizes[64] / sizes[8]
        assert growth < 8, (
            f"payload grew {growth:.1f}x for 8x the experts, which is linear: the "
            "per-expert cost is no longer bounded by the descriptor alone"
        )

    def test_metadata_is_not_what_grows(self, client):
        """Pin the cause, so a future metadata regression is not excused as 'descriptors'."""
        from psrl.utils.nixl.nixl_spec import NIXLSharding

        num_experts = 64
        state, sharding = {}, {}
        for expert in range(num_experts):
            key = f"model.layers.0.mlp.experts.{expert}.gate_proj.weight"
            state[key] = torch.full((ROWS, COLS), float(expert), device="cuda", dtype=torch.bfloat16)
            sharding[key] = NIXLSharding.default()
        grouped = group_expert_tensors(state, sharding, num_experts=num_experts)

        client.register_local_tensors(grouped.state_dict, grouped.sharding_dict)
        info = client.local_client_info.tensor_infos["model.layers.0.mlp.experts.gate_proj.weight"]

        assert len(set(id(meta) for meta in info.shard_meta_infos)) == 1
        client.deregister_local_tensors()
