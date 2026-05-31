import torch

from psrl.utils.converter.base_converter import BaseConverter
from psrl.utils.converter.model_mappings import (
    ParameterMapping,
    slice_attn_conv1d,
    slice_qwen3_5_in_proj_qkv,
    reshape_visual_block_qkv
)
from psrl.utils.nixl.nixl_spec import NIXLSharding


class HFConverter(BaseConverter):
    """Convert HuggingFace model to a unified format and generate sharding info."""

    def __init__(self, parameter_mapping: ParameterMapping):
        """
        Args:
            parameter_mapping (ParameterMapping): Parameter mapping instance carrying
                model_info (num_heads, num_kv_heads, head_size). Use HFParameterMapping
                to enable Q/K/V 3D reshaping for NIXL compatibility with Megatron workers.
        """
        super().__init__(parameter_mapping)

    def convert_state_and_sharding_dict(self, model) -> tuple[dict[str, torch.Tensor], dict[str, NIXLSharding]]:
        """
        Convert HuggingFace model to unified state dict and sharding info.

        Args:
            model: The HuggingFace model instance.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, NIXLSharding]]: A pair of
                (converted_state_dict, sharding_dict).
        """
        hf_state_dict = model.state_dict()

        converted_state_dict = {}
        sharding_dict = {}
        for param_name, param in hf_state_dict.items():
            assert isinstance(param, torch.Tensor), f"Expected Tensor for {param_name}, got {type(param)}."
            sharding = NIXLSharding.default()
            # NOTE(lhy): PS holds full (non-TP-sharded) weights, so num_heads_local = global num_heads.
            # maybe_reshape_qkv_to_3d is a no-op when model_info does not contain num_heads.
            param, sharding = self.maybe_reshape_qkv_to_3d(param_name, param, sharding)
            param_dict = maybe_convert_to_smaller_parts(self.model_info, param_name, param)
            for new_param_name, new_param in param_dict.items():
                converted_state_dict[new_param_name] = new_param
                sharding_dict[new_param_name] = sharding
        return converted_state_dict, sharding_dict


def maybe_convert_to_smaller_parts(model_info, param_name, param):
    """
    Convert certain fused parameters to multiple smaller parameters for correct TP sharding.
    """
    if "linear_attn.conv1d.weight" in param_name:
        new_param_names = [param_name + "_q", param_name + "_k", param_name + "_v"]
        new_params = slice_attn_conv1d(
            fused_param=param,
            num_k_heads=model_info["linear_num_key_heads"],
            num_v_heads=model_info["linear_num_value_heads"],
            k_head_size=model_info["linear_key_head_dim"],
            v_head_size=model_info["linear_value_head_dim"],
            tp_size=1,
        )
        return dict(zip(new_param_names, new_params))
    if "linear_attn.in_proj_qkv.weight" in param_name:
        new_param_names = [param_name + "_q", param_name + "_k", param_name + "_v"]
        new_params = slice_qwen3_5_in_proj_qkv(
            fused_param=param,
            key_dim=model_info["linear_key_dim"],
            value_dim=model_info["linear_value_dim"],
            tp_size=1,
        )
        return dict(zip(new_param_names, new_params))
    if "visual.blocks" in param_name and "qkv" in param_name:
        param = reshape_visual_block_qkv(param)
    if "mlp.experts.gate_up_proj" in param_name:
        param_dict = {}
        num_experts = model_info["num_experts"]
        name_prefix = param_name.rsplit('.', 1)[0]
        for i in range(num_experts):
            gate_name = f"{name_prefix}.{i}.gate_proj.weight"
            up_name = f"{name_prefix}.{i}.up_proj.weight"
            gate_param, up_param = param[i].chunk(2, dim=0)
            param_dict[gate_name] = gate_param
            param_dict[up_name] = up_param
        return param_dict
    if "mlp.experts.down_proj" in param_name:
        param_dict = {}
        num_experts = model_info["num_experts"]
        name_prefix = param_name.rsplit('.', 1)[0]
        for i in range(num_experts):
            down_name = f"{name_prefix}.{i}.down_proj.weight"
            param_dict[down_name] = param[i]
        return param_dict
    return {param_name: param}


def convert_hf_inplace(
    parameter_mapping: ParameterMapping,
    model,
) -> tuple[dict[str, torch.Tensor], dict[str, NIXLSharding]]:
    """
    Convert HuggingFace model to unified state dict and sharding info.

    Args:
        model: The HuggingFace model instance.
        parameter_mapping (ParameterMapping): Parameter mapping instance carrying
            model_info. Use HFParameterMapping to enable Q/K/V 3D reshaping for
            NIXL shape compatibility between PS and Megatron train workers.

    Returns:
        tuple[dict[str, torch.Tensor], dict[str, NIXLSharding]]: A pair of
            (converted_state_dict, sharding_dict).
    """
    converter = HFConverter(parameter_mapping)
    return converter.convert_state_and_sharding_dict(model)
