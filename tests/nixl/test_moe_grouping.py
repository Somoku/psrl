"""Canonical MoE expert grouping (implementation item C1).

Verified against the real checkpoints under /jizhicfs/johnnyslin/models rather than
synthetic key lists, because the premise being tested is a property of actual
models: that expert keys dominate the namespace and collapse by one to two orders of
magnitude.

The correctness risk is ordering. The expert axis becomes shard dimension 0, so the
position of each expert in the grouped key *is* its shard index. Sorting expert ids
lexically would put expert 10 before expert 2 and permute the experts -- delivering
every byte to the wrong expert, with no size or shape mismatch to catch it.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import pytest

from psrl.utils.nixl.moe_grouping import (
    group_expert_keys,
    grouped_key_count,
    grouping_report,
    parse_expert_key,
)

pytestmark = pytest.mark.cpu_test

MODEL_ROOT = Path("/jizhicfs/johnnyslin/models")


def _hf_keys(model: str) -> list[str]:
    model_dir = MODEL_ROOT / model
    if not model_dir.exists():
        pytest.skip(f"{model} not present")
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        return list(json.loads(index.read_text())["weight_map"])
    from safetensors import safe_open

    with safe_open(str(model_dir / "model.safetensors"), "pt") as handle:
        return list(handle.keys())


def _synthetic(layers: int, experts: int, projections=("gate_proj", "up_proj", "down_proj")):
    keys = []
    for layer in range(layers):
        for expert in range(experts):
            for projection in projections:
                keys.append(f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight")
        keys.append(f"model.layers.{layer}.self_attn.q_proj.weight")
    return keys


class TestKeyParsing:
    def test_expert_key_is_split_into_its_parts(self):
        parsed = parse_expert_key("model.layers.3.mlp.experts.57.gate_proj.weight")
        assert parsed == ("model.layers.3.mlp", 57, "gate_proj.weight")

    @pytest.mark.parametrize(
        "key",
        [
            "model.layers.3.self_attn.q_proj.weight",
            "model.embed_tokens.weight",
            "model.layers.3.mlp.gate.weight",  # the router, not an expert
            "model.norm.weight",
        ],
    )
    def test_non_expert_keys_are_left_alone(self, key):
        assert parse_expert_key(key) is None

    def test_multi_digit_expert_ids_parse(self):
        assert parse_expert_key("p.experts.127.w.weight") == ("p", 127, "w.weight")


class TestGrouping:
    def test_one_group_per_layer_and_projection(self):
        keys = _synthetic(layers=2, experts=8)
        groups, passthrough = group_expert_keys(keys)

        assert len(groups) == 2 * 3
        assert len(passthrough) == 2
        assert all(group.num_experts == 8 for group in groups)

    def test_grouped_key_drops_the_expert_id(self):
        groups, _ = group_expert_keys(_synthetic(layers=1, experts=4))
        names = {group.grouped_key for group in groups}
        assert names == {
            "model.layers.0.mlp.experts.gate_proj.weight",
            "model.layers.0.mlp.experts.up_proj.weight",
            "model.layers.0.mlp.experts.down_proj.weight",
        }

    def test_layers_are_never_merged(self):
        """The prefix carries the layer, so two layers must stay separate groups."""
        groups, _ = group_expert_keys(_synthetic(layers=3, experts=4))
        prefixes = {group.prefix for group in groups}
        assert len(prefixes) == 3

    def test_projections_are_never_merged(self):
        groups, _ = group_expert_keys(_synthetic(layers=1, experts=4))
        assert len({group.projection for group in groups}) == 3

    def test_duplicate_expert_id_is_rejected(self):
        keys = ["p.experts.0.w.weight", "p.experts.0.w.weight"]
        with pytest.raises(ValueError, match="duplicate expert ids"):
            group_expert_keys(keys)


class TestExpertOrdering:
    """The shard index is the position in this list, so order is correctness."""

    def test_expert_ids_sort_numerically_not_lexically(self):
        keys = [f"p.experts.{i}.w.weight" for i in (0, 1, 2, 10, 11, 20, 100)]
        groups, _ = group_expert_keys(keys)

        assert groups[0].expert_ids == (0, 1, 2, 10, 11, 20, 100), (
            "lexical ordering would give (0, 1, 10, 100, 11, 2, 20) and permute experts"
        )

    def test_shard_index_matches_expert_id_position(self):
        keys = [f"p.experts.{i}.w.weight" for i in range(128)]
        groups, _ = group_expert_keys(keys)
        group = groups[0]

        for expert_id in (0, 1, 2, 10, 42, 127):
            assert group.shard_index_of(expert_id) == expert_id

    def test_ordering_is_independent_of_input_order(self):
        """Keys arrive in whatever order the checkpoint index lists them."""
        ordered = [f"p.experts.{i}.w.weight" for i in range(16)]
        shuffled = ordered[7:] + ordered[:7]

        from_ordered = group_expert_keys(ordered)[0][0]
        from_shuffled = group_expert_keys(shuffled)[0][0]

        assert from_ordered.expert_ids == from_shuffled.expert_ids
        assert from_ordered.member_keys == from_shuffled.member_keys

    def test_member_keys_align_with_expert_ids(self):
        groups, _ = group_expert_keys([f"p.experts.{i}.w.weight" for i in (5, 0, 12)])
        group = groups[0]
        for position, expert_id in enumerate(group.expert_ids):
            assert f".experts.{expert_id}." in group.member_keys[position]


class TestShardMesh:
    def test_expert_axis_becomes_dimension_zero(self):
        groups, _ = group_expert_keys([f"p.experts.{i}.w.weight" for i in range(64)])
        assert groups[0].shard_mesh() == OrderedDict([(0, 64)])

    def test_inner_sharding_is_pushed_out_by_one_dimension(self):
        """ETP != 1 makes the grouped key two-dimensional (design 3.5.5)."""
        groups, _ = group_expert_keys([f"p.experts.{i}.w.weight" for i in range(8)])
        mesh = groups[0].shard_mesh(inner_mesh=OrderedDict([(0, 4)]))
        assert mesh == OrderedDict([(0, 8), (1, 4)])

    def test_two_dimensional_mesh_survives_finest_mesh_merging(self):
        """The case design 3.5.5 flags for verification against lcm."""
        from psrl.utils.nixl.nixl_spec import NIXLSharding

        groups, _ = group_expert_keys([f"p.experts.{i}.w.weight" for i in range(8)])
        train = groups[0].shard_mesh(inner_mesh=OrderedDict([(0, 2)]))
        gen = groups[0].shard_mesh(inner_mesh=OrderedDict([(0, 4)]))

        finest = NIXLSharding.find_finest_shard_mesh([train, gen])

        assert finest[0] == 8, "the expert axis is identical and must not grow"
        assert finest[1] == 4, "the inner axis must be the lcm of 2 and 4"


class TestRealCheckpoints:
    """The premise, measured rather than assumed."""

    def test_qwen3_30b_collapses_by_more_than_an_order_of_magnitude(self):
        keys = _hf_keys("Qwen3-30B-A3B-Instruct-2507")
        report = grouping_report(keys)

        assert report["keys_before"] == 18867
        assert report["keys_after"] == 579
        assert report["reduction"] > 30
        assert report["expert_groups"] == 48 * 3, "48 layers x 3 projections"

    def test_moonlight_collapses_too(self):
        keys = _hf_keys("Moonlight-16B-A3B")
        report = grouping_report(keys)
        assert report["reduction"] > 10
        assert report["keys_after"] < 600

    def test_dense_model_is_untouched(self):
        """Grouping must be a no-op where there are no experts."""
        keys = _hf_keys("Qwen2.5-0.5B-Instruct")
        groups, passthrough = group_expert_keys(keys)

        assert groups == []
        assert len(passthrough) == len(keys)
        assert grouped_key_count(keys) == len(keys)

    def test_every_expert_key_is_accounted_for(self):
        """No key may be dropped: a missing expert is a silently wrong model."""
        keys = _hf_keys("Qwen3-30B-A3B-Instruct-2507")
        groups, passthrough = group_expert_keys(keys)

        covered = {key for group in groups for key in group.member_keys}
        assert len(covered) + len(passthrough) == len(keys)
        assert covered.isdisjoint(passthrough)
        assert covered | set(passthrough) == set(keys)

    def test_expert_count_matches_the_model_config(self):
        model_dir = MODEL_ROOT / "Qwen3-30B-A3B-Instruct-2507"
        if not model_dir.exists():
            pytest.skip("model not present")
        config = json.loads((model_dir / "config.json").read_text())

        groups, _ = group_expert_keys(_hf_keys("Qwen3-30B-A3B-Instruct-2507"))

        assert all(group.num_experts == config["num_experts"] for group in groups)

    def test_grouping_is_reversible(self):
        """Every grouped key must be able to name its sources, for debugging."""
        keys = _hf_keys("Qwen3-30B-A3B-Instruct-2507")
        groups, _ = group_expert_keys(keys)

        for group in groups[:5]:
            for position, expert_id in enumerate(group.expert_ids):
                expected = f"{group.prefix}.experts.{expert_id}.{group.projection}"
                assert group.member_keys[position] == expected


class TestMetadataImpact:
    """Why this is a blocking item rather than an optimisation."""

    def test_payload_scales_with_the_grouped_count(self):
        keys = _hf_keys("Qwen3-30B-A3B-Instruct-2507")
        bytes_per_key = 400  # design 3.5.5's estimate for a serialised tensor_info

        before = len(keys) * bytes_per_key
        after = grouped_key_count(keys) * bytes_per_key

        assert before > 7 * 1024 * 1024, "the ungrouped payload is megabytes per client"
        assert after < 256 * 1024, "the grouped payload must fit in hundreds of KB"

    def test_kimi_k2_scale_projection(self):
        """384 experts x 61 layers x 3 projections, the figure design 3.5.5 derives."""
        keys = _synthetic(layers=61, experts=384)
        report = grouping_report(keys)

        assert report["keys_before"] == 61 * 384 * 3 + 61
        assert report["reduction"] > 100
