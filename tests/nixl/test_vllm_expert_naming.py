"""vLLM 0.30 expert keys must land in the canonical namespace (C1 regression).

`c2b0f81` bumped vLLM to 0.30, which nests FusedMoE weights under `routed_experts`. The
gen side then emitted `experts.routed_experts.<i>.*` while the train side emitted
`experts.<i>.*`, so grouping skipped the gen keys and the planner found no source for
them. These tests pin the name mapping and that grouping recognises its output.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.cpu_test

vllm_converter = pytest.importorskip("psrl.utils.converter.vllm_converter")


def test_routed_experts_segment_is_removed():
    name = "model.layers.3.mlp.experts.routed_experts.17.gate_proj.weight"
    assert vllm_converter.canonical_vllm_name(name) == "model.layers.3.mlp.experts.17.gate_proj.weight"


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.mlp.experts.5.down_proj.weight",
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.self_attn.q_proj.weight",
    ],
)
def test_other_names_are_unchanged(name):
    assert vllm_converter.canonical_vllm_name(name) == name


def test_grouping_recognises_the_canonical_name():
    from psrl.utils.nixl.moe_grouping import group_expert_keys

    keys = [
        vllm_converter.canonical_vllm_name(f"model.layers.0.mlp.experts.routed_experts.{i}.up_proj.weight")
        for i in range(4)
    ]
    groups, passthrough = group_expert_keys(keys)
    assert not passthrough and len(groups) == 1
