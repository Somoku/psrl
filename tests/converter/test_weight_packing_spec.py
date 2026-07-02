"""Tests for WeightLayoutSpec dataclass and SupportsWeightLayoutSpec Protocol."""

import pytest
from vllm.model_executor.models.interfaces import SupportsWeightLayoutSpec, WeightLayoutSpec

pytestmark = pytest.mark.cpu_test


class TestWeightLayoutSpec:
    def test_basic_construction(self):
        spec = WeightLayoutSpec(
            stacked_params=[("qkv_proj", "q_proj", "q")],
            packing_metadata={"num_heads": 32},
        )
        assert spec.stacked_params == [("qkv_proj", "q_proj", "q")]
        assert spec.packing_metadata == {"num_heads": 32}
        assert spec.expert_params is None
        assert spec.extra_stacked_params is None

    def test_empty_spec(self):
        spec = WeightLayoutSpec(stacked_params=[], packing_metadata={})
        assert spec.stacked_params == []
        assert spec.packing_metadata == {}

    def test_moe_spec(self):
        expert_params = [("experts.w13_", "experts.0.gate_proj.", 0, "w1")]
        spec = WeightLayoutSpec(
            stacked_params=[("gate_up_proj", "gate_proj", 0)],
            packing_metadata={"num_experts": 8},
            expert_params=expert_params,
        )
        assert spec.expert_params == expert_params

    def test_extra_stacked_params(self):
        spec = WeightLayoutSpec(
            stacked_params=[("qkv_proj", "q_proj", "q")],
            packing_metadata={},
            extra_stacked_params=[("layers.0.mlp.w13.weight", "layers.0.mlp.w1.weight", 0)],
        )
        assert spec.extra_stacked_params is not None
        assert len(spec.extra_stacked_params) == 1

    def test_shard_id_types(self):
        """stacked_params shard_id can be str or int."""
        spec = WeightLayoutSpec(
            stacked_params=[
                ("qkv_proj", "q_proj", "q"),  # str shard_id
                ("gate_up_proj", "gate_proj", 0),  # int shard_id
            ],
            packing_metadata={},
        )
        assert spec.stacked_params[0][2] == "q"
        assert spec.stacked_params[1][2] == 0


class TestSupportsWeightLayoutSpecProtocol:
    def test_isinstance_check_with_method(self):
        """isinstance check passes for any object with get_weight_layout_spec."""

        class FakeModel:
            def get_weight_layout_spec(self):
                return WeightLayoutSpec(stacked_params=[], packing_metadata={})

        model = FakeModel()
        assert isinstance(model, SupportsWeightLayoutSpec)

    def test_isinstance_check_without_method(self):
        """isinstance check fails for objects without the method."""

        class PlainModel:
            pass

        assert not isinstance(PlainModel(), SupportsWeightLayoutSpec)

    def test_supports_weight_layout_spec_classvar(self):
        """supports_weight_layout_spec ClassVar is True on the Protocol."""
        assert SupportsWeightLayoutSpec.supports_weight_layout_spec is True
