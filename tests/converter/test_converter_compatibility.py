"""Compatibility test: SupportsWeightLayoutSpec fast path must produce the same
fused_mappings as the old ParameterMapping path for the same model."""
import pytest
from psrl.utils.converter.model_mappings import MappingType, ParameterMapping
from psrl.utils.converter.vllm_converter import VllmConverter
from vllm.model_executor.models.interfaces import WeightLayoutSpec, SupportsWeightLayoutSpec
from unittest.mock import MagicMock

pytestmark = pytest.mark.cpu_test


def make_mock_config(**kwargs):
    config = MagicMock()
    defaults = {
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "hidden_size": 16,
        "intermediate_size": 32,
    }
    defaults.update(kwargs)
    for k, v in defaults.items():
        setattr(config, k, v)
    type(config).__getattr__ = lambda self, name: defaults.get(name, MagicMock())
    return config


class OldStyleQwen2Mapping(ParameterMapping):
    """Equivalent of the deleted VllmQwen2ParameterMapping from vllm_modeling.py."""
    def __init__(self):
        self.config = make_mock_config()

    def get_mappings(self):
        return [
            ("qkv_proj", "q_proj", MappingType.QKV_SPLIT, 0),
            ("qkv_proj", "k_proj", MappingType.QKV_SPLIT, 1),
            ("qkv_proj", "v_proj", MappingType.QKV_SPLIT, 2),
            ("gate_up_proj", "gate_proj", MappingType.GATE_UP_PROJ_SPLIT, 0),
            ("gate_up_proj", "up_proj", MappingType.GATE_UP_PROJ_SPLIT, 1),
        ]

    def get_model_info(self):
        return {
            "num_heads": self.config.num_attention_heads,
            "num_kv_heads": getattr(self.config, "num_key_value_heads",
                                    self.config.num_attention_heads),
            "head_size": self.config.hidden_size // self.config.num_attention_heads,
            "intermediate_size": self.config.intermediate_size,
        }


class NewStyleQwen2Model(SupportsWeightLayoutSpec):
    """New-style model implementing SupportsWeightLayoutSpec."""
    supports_weight_layout_spec = True

    def __init__(self, config):
        self.config = config

    def get_weight_layout_spec(self) -> WeightLayoutSpec:
        return WeightLayoutSpec(
            stacked_params=[
                ("qkv_proj", "q_proj", "q"),
                ("qkv_proj", "k_proj", "k"),
                ("qkv_proj", "v_proj", "v"),
                ("gate_up_proj", "gate_proj", 0),
                ("gate_up_proj", "up_proj", 1),
            ],
            packing_metadata={
                "num_heads": self.config.num_attention_heads,
                "num_kv_heads": getattr(self.config, "num_key_value_heads",
                                        self.config.num_attention_heads),
                "head_size": self.config.hidden_size // self.config.num_attention_heads,
                "intermediate_size": self.config.intermediate_size,
            },
        )


class TestConverterCompatibility:
    def test_qkv_mapping_types_equivalent(self):
        """New spec path must produce QKV_SPLIT for qkv_proj, same as old ParameterMapping."""
        config = make_mock_config()
        old_mapping = OldStyleQwen2Mapping()
        new_model = NewStyleQwen2Model(config)

        old_conv = VllmConverter(parameter_mapping=old_mapping, tp_rank=0)
        old_fused, old_info = old_conv._build_from_parameter_mapping()

        new_conv = VllmConverter(parameter_mapping=None, tp_rank=0)
        new_fused, new_info = new_conv._build_from_spec(new_model.get_weight_layout_spec())

        assert old_fused["qkv_proj"].mapping_type == MappingType.QKV_SPLIT
        assert new_fused["qkv_proj"].mapping_type == MappingType.QKV_SPLIT
        assert old_fused["gate_up_proj"].mapping_type == MappingType.GATE_UP_PROJ_SPLIT
        assert new_fused["gate_up_proj"].mapping_type == MappingType.GATE_UP_PROJ_SPLIT

    def test_packing_metadata_equivalent_to_model_info(self):
        """spec.packing_metadata must have the same keys and values as get_model_info()."""
        config = make_mock_config()
        old_mapping = OldStyleQwen2Mapping()
        new_model = NewStyleQwen2Model(config)

        _, old_info = VllmConverter(parameter_mapping=old_mapping, tp_rank=0)._build_from_parameter_mapping()
        _, new_info = VllmConverter(parameter_mapping=None, tp_rank=0)._build_from_spec(new_model.get_weight_layout_spec())

        for key in old_info:
            assert key in new_info, f"Key '{key}' in old model_info but missing from packing_metadata"
            assert old_info[key] == new_info[key], \
                f"Mismatch for '{key}': old={old_info[key]}, new={new_info[key]}"

    def test_hf_component_names_identical(self):
        """The HF output names (q_proj, k_proj, ...) must be the same from both paths."""
        config = make_mock_config()
        old_mapping = OldStyleQwen2Mapping()
        new_model = NewStyleQwen2Model(config)

        old_fused, _ = VllmConverter(parameter_mapping=old_mapping, tp_rank=0)._build_from_parameter_mapping()
        new_fused, _ = VllmConverter(parameter_mapping=None, tp_rank=0)._build_from_spec(new_model.get_weight_layout_spec())

        old_qkv_hf_names = {e[0] for e in old_fused["qkv_proj"].mappings}
        new_qkv_hf_names = {e[0] for e in new_fused["qkv_proj"].mappings}
        assert old_qkv_hf_names == new_qkv_hf_names

    def test_moe_w13_w2_shard_ids_correct(self):
        """Expert shard_id encoding: gate (w1) = 2*eid, up (w3) = 2*eid+1, down (w2) = eid."""
        NUM_EXPERTS = 3
        expert_params = []
        for eid in range(NUM_EXPERTS):
            expert_params += [
                ("experts.w13_", f"experts.{eid}.gate_proj.", eid, "w1"),
                ("experts.w2_",  f"experts.{eid}.down_proj.",  eid, "w2"),
                ("experts.w13_", f"experts.{eid}.up_proj.",    eid, "w3"),
            ]
        conv = VllmConverter(parameter_mapping=None, tp_rank=0)
        result = conv._build_fused_mappings([], [], expert_params)
        w13_entries = result["experts.w13_"].mappings
        w2_entries  = result["experts.w2_"].mappings
        w13_shard_ids = [e[1] for e in w13_entries]
        w2_shard_ids  = [e[1] for e in w2_entries]
        expected_w13 = [0, 1, 2, 3, 4, 5]  # interleaved gate/up
        assert w13_shard_ids == expected_w13
        # down (w2): 0, 1, 2
        assert w2_shard_ids == [0, 1, 2]
