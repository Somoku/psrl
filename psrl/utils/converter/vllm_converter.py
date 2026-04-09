from collections import OrderedDict

import torch
from torch.nn import Parameter
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    set_weight_attrs,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.qwen3_5 import Qwen3_5GatedDeltaNet

from psrl.utils.converter.base_converter import BaseConverter
from psrl.utils.converter.model_mappings import (
    MappingType,
    ParameterMapping,
    slice_in_proj_ba,
    slice_in_proj_qkvz,
    slice_qwen3_5_in_proj_qkv,
    slice_fused_moe_w2_weight,
    slice_fused_moe_w13_weight,
    slice_gate_up_proj,
    slice_qkv_proj,
    slice_attn_conv1d,
    reshape_visual_block_qkv,
)
from psrl.utils.nixl.nixl_spec import NIXLSharding


def enable_sharded_weight_attrs(params: dict[str, Parameter]):
    for name, param in params.items():
        set_weight_attrs(param, {"is_sharded_weight": True})
    return params


class VllmConverter(BaseConverter):
    """Convert vLLM model to a unified format (i.e., HuggingFace) and generate sharding info."""

    def __init__(self, parameter_mapping: ParameterMapping, tp_rank: int | None = 1):
        super().__init__(parameter_mapping)
        self.parameter_mapping = parameter_mapping
        self.tp_rank = tp_rank
        self.mappings = parameter_mapping.get_mappings()
        self.fused_mappings: dict[str, tuple[MappingType, list[tuple[str, int]]]] = {}
        for vllm_name, hf_name, mapping_type, shard_id in self.mappings:
            if vllm_name not in self.fused_mappings:
                self.fused_mappings[vllm_name] = (mapping_type, [])
            else:
                assert mapping_type != MappingType.DIRECT, f"Mapping type should not be DIRECT for {vllm_name}"
                assert mapping_type == self.fused_mappings[vllm_name][0], (
                    f"Mapping type for {vllm_name} must be the same, "
                    f"but got {mapping_type} and {self.fused_mappings[vllm_name][0]}"
                )
            self.fused_mappings[vllm_name][1].append((hf_name, shard_id))

    def convert_state_and_sharding_dict(self, model) -> tuple[dict[str, torch.Tensor], dict[str, NIXLSharding]]:
        """
        Convert vLLM model to unified state dict and generate sharding info.
        Args:
            model: The vLLM model instance
        Returns:
            (converted_state_dict, sharding_dict)
        """
        converted_state_dict = {}
        sharding_dict = {}

        # Workaround: for lm_head, we do not care if it shares the weight with wte
        lm_head_module = None
        lm_head_module_prefix = None
        if hasattr(model, "lm_head"):
            lm_head_module = model.lm_head
            lm_head_module_prefix = "lm_head"
        if hasattr(model, "language_model") and hasattr(model.language_model, "lm_head"):
            lm_head_module = model.language_model.lm_head
            lm_head_module_prefix = "lm_head"

        seen_module_prefixes = set()
        for module_prefix, module in model.named_modules():
            if module_prefix.startswith("visual"):
                module_prefix = f"model.{module_prefix}"
            if module_prefix.startswith("language_model"):
                module_prefix = f"model.language_model.{module_prefix[21:]}"
            seen_module_prefixes.add(module_prefix)
            for param_name, param in module.named_parameters(recurse=False):
                full_name = f"{module_prefix}.{param_name}" if module_prefix else param_name
                if full_name.startswith("."):
                    full_name = full_name[1:]
                new_params = self.convert_parameter(full_name, param, module)
                sharding = self.get_sharding_for_param(module, param_name, full_name)
                for new_param_name, new_param in new_params.items():
                    new_param, sharding_for_param = self.maybe_reshape_qkv_to_3d(new_param_name, new_param, sharding)
                    converted_state_dict[new_param_name] = new_param
                    sharding_dict[new_param_name] = sharding_for_param

        # Handle lm_head separately
        if lm_head_module is not None and (lm_head_module_prefix not in seen_module_prefixes):
            module = lm_head_module
            module_prefix = lm_head_module_prefix
            for param_name, param in module.named_parameters(recurse=False):
                full_name = f"{module_prefix}.{param_name}" if module_prefix else param_name
                new_params = self.convert_parameter(full_name, param, module)
                sharding = self.get_sharding_for_param(module, param_name, full_name)
                for new_param_name, new_param in new_params.items():
                    new_param, sharding_for_param = self.maybe_reshape_qkv_to_3d(new_param_name, new_param, sharding)
                    converted_state_dict[new_param_name] = new_param
                    sharding_dict[new_param_name] = sharding_for_param

        return converted_state_dict, sharding_dict

    def convert_parameter(self, full_name: str, param: Parameter, module) -> dict:
        """
        Convert the parameter, may need to split inplace
        if it matches a split mapping type (e.g., qkv_proj, gate_up_proj).
        """
        tp_size = getattr(module, "tp_size", 1)
        for vllm_name in self.fused_mappings:
            if vllm_name in full_name:
                mapping_type, mappings = self.fused_mappings[vllm_name]
                if mapping_type == MappingType.DIRECT:
                    assert len(mappings) == 1, (
                        f"Mapping type is DIRECT for {vllm_name}, but got {len(mappings)} mappings"
                    )
                    new_param = param
                    new_param_name = full_name.replace(vllm_name, mappings[0][0])
                    return {new_param_name: new_param}
                elif mapping_type == MappingType.QKV_SPLIT:
                    try:
                        sliced_params = slice_qkv_proj(
                            fused_param=param,
                            num_heads=self.model_info["num_heads"],
                            num_kv_heads=self.model_info["num_kv_heads"],
                            head_size=self.model_info["head_size"],
                            tp_size=tp_size,
                        )
                    except Exception as e:
                        raise ValueError(f"Failed to slice qkv parameter {full_name}: {e}") from e
                    out = {}
                    for hf_name, shard_id in mappings:
                        assert shard_id < len(sliced_params), f"Shard id {shard_id} is out of range for {vllm_name}"
                        new_param = sliced_params[shard_id]
                        new_param_name = full_name.replace(vllm_name, hf_name)
                        out[new_param_name] = new_param
                    return out
                elif mapping_type == MappingType.GATE_UP_PROJ_SPLIT:
                    intermediate_size = self.model_info["intermediate_size"]
                    try:
                        sliced_params = slice_gate_up_proj(
                            fused_param=param,
                            output_sizes=[intermediate_size, intermediate_size],
                            tp_size=tp_size,
                        )
                    except Exception as e:
                        raise ValueError(f"Failed to slice gate up proj parameter {full_name}: {e}") from e
                    out = {}
                    for hf_name, shard_id in mappings:
                        assert shard_id < len(sliced_params), f"Shard id {shard_id} is out of range for {vllm_name}"
                        new_param = sliced_params[shard_id]
                        new_param_name = full_name.replace(vllm_name, hf_name)
                        out[new_param_name] = new_param
                    return out
                elif mapping_type == MappingType.IN_PROJ_QKVZ_SPLIT:
                    key_dim = self.model_info.get("linear_key_dim")
                    value_dim = self.model_info.get("linear_value_dim")
                    if key_dim is None or value_dim is None:
                        raise ValueError(
                            "Qwen3.5 linear attention dims are missing in model_info; "
                            f"got linear_key_dim={key_dim} and linear_value_dim={value_dim}."
                        )
                    try:
                        sliced_params = slice_in_proj_qkvz(
                            fused_param=param,
                            key_dim=key_dim,
                            value_dim=value_dim,
                            tp_size=tp_size,
                        )
                    except Exception as e:
                        raise ValueError(f"Failed to slice in_proj_qkvz parameter {full_name}: {e}") from e
                    out = {}
                    for hf_name, shard_id in mappings:
                        assert shard_id < len(sliced_params), f"Shard id {shard_id} is out of range for {vllm_name}"
                        new_param = sliced_params[shard_id]
                        new_param_name = full_name.replace(vllm_name, hf_name)
                        if shard_id == 0:
                            qkv_names = [new_param_name + "_q", new_param_name + "_k", new_param_name + "_v"]
                            qkv_params = slice_qwen3_5_in_proj_qkv(
                                fused_param=new_param, 
                                key_dim=key_dim,
                                value_dim=value_dim,
                                tp_size=tp_size,
                            )
                            out.update(dict(zip(qkv_names, qkv_params)))
                            continue
                        out[new_param_name] = new_param
                    return out
                elif mapping_type == MappingType.IN_PROJ_BA_SPLIT:
                    num_v_heads = self.model_info.get("linear_num_value_heads")
                    if num_v_heads is None:
                        raise ValueError(
                            "Qwen3.5 linear attention num_v_heads is missing in model_info; "
                            f"got linear_num_value_heads={num_v_heads}."
                        )
                    try:
                        sliced_params = slice_in_proj_ba(
                            fused_param=param,
                            num_v_heads=num_v_heads,
                            tp_size=tp_size,
                        )
                    except Exception as e:
                        raise ValueError(f"Failed to slice in_proj_ba parameter {full_name}: {e}") from e
                    out = {}
                    for hf_name, shard_id in mappings:
                        assert shard_id < len(sliced_params), f"Shard id {shard_id} is out of range for {vllm_name}"
                        new_param = sliced_params[shard_id]
                        new_param_name = full_name.replace(vllm_name, hf_name)
                        out[new_param_name] = new_param
                    return out
                elif mapping_type == MappingType.FUSED_MOE_W13_SPLIT:
                    try:
                        sliced_params = slice_fused_moe_w13_weight(
                            fused_param=param,
                        )
                    except Exception as e:
                        raise ValueError(f"Failed to slice w13_weight parameter {full_name}: {e}") from e
                    out = {}
                    ep_size = getattr(module, "ep_size", 1)
                    # NOTE(zym) Though module has attribute "ep_rank", the value is incorrect,
                    # and now we only have dp=1, so we use tp_rank as ep_rank
                    ep_rank = (
                        self.tp_rank if ep_size > 1 else 0
                    )  # considering the case where not enable_expert_parallel
                    num_experts = self.model_info["num_experts"]
                    num_experts_per_ep_rank = num_experts // ep_size
                    local_experts_start_id = ep_rank * num_experts_per_ep_rank
                    local_experts_end_id = local_experts_start_id + num_experts_per_ep_rank
                    local_shard_ids = list(range(local_experts_start_id * 2, local_experts_end_id * 2))
                    for hf_name, shard_id in mappings:
                        if shard_id not in local_shard_ids:
                            continue
                        slice_idx = shard_id - local_experts_start_id * 2
                        assert slice_idx < len(sliced_params), f"Slice idx {slice_idx} is out of range for {vllm_name}"
                        new_param = sliced_params[slice_idx]
                        new_param_name = full_name.replace(vllm_name, hf_name)
                        out[new_param_name] = new_param
                    return out
                elif mapping_type == MappingType.FUSED_MOE_W2_SPLIT:
                    try:
                        sliced_params = slice_fused_moe_w2_weight(
                            fused_param=param,
                        )
                    except Exception as e:
                        raise ValueError(f"Failed to slice w2_weight parameter {full_name}: {e}") from e
                    out = {}
                    ep_size = getattr(module, "ep_size", 1)
                    ep_rank = self.tp_rank if ep_size > 1 else 0
                    num_experts = self.model_info["num_experts"]
                    num_experts_per_ep_rank = num_experts // ep_size
                    local_experts_start_id = ep_rank * num_experts_per_ep_rank
                    local_experts_end_id = local_experts_start_id + num_experts_per_ep_rank
                    local_shard_ids = list(range(local_experts_start_id, local_experts_end_id))
                    for hf_name, shard_id in mappings:
                        if shard_id not in local_shard_ids:
                            continue
                        slice_idx = shard_id - local_experts_start_id
                        assert slice_idx < len(sliced_params), f"Slice idx {slice_idx} is out of range for {vllm_name}"
                        new_param = sliced_params[slice_idx]
                        new_param_name = full_name.replace(vllm_name, hf_name)
                        out[new_param_name] = new_param
                    return out
                else:
                    raise ValueError(f"Unsupported mapping type: {mapping_type}")
        
        if "linear_attn.conv1d.weight" in full_name:
            new_param_names = [full_name + "_q", full_name + "_k", full_name + "_v"]
            new_params = slice_attn_conv1d(
                fused_param=param,
                num_k_heads=self.model_info["linear_num_key_heads"],
                num_v_heads=self.model_info["linear_num_value_heads"],
                k_head_size=self.model_info["linear_key_head_dim"],
                v_head_size=self.model_info["linear_value_head_dim"],
                tp_size=tp_size,
            )
            return dict(zip(new_param_names, new_params))

        if "visual.blocks" in full_name and "qkv" in full_name:
            param = reshape_visual_block_qkv(param)

        # Default: No conversion needed
        return {full_name: param}

    def get_sharding_for_param(self, module, param_name, full_name=None) -> NIXLSharding:
        """
        Generate sharding info for a parameter given its module and tp_rank.
        Returns a NIXLSharding object.
        """
        tp_size = getattr(module, "tp_size", 1)
        if tp_size > 1:
            assert tp_size > self.tp_rank, (
                f"Tensor parallel size ({tp_size}) must be "
                f"greater than tensor parallel rank ({self.tp_rank}), "
                f"please check the tensor parallel size and rank."
            )
            shard_indices = [(self.tp_rank,)] if self.tp_rank is not None else [(0,)]
            if isinstance(
                module,
                (
                    ColumnParallelLinear,
                    MergedColumnParallelLinear,
                    QKVParallelLinear,
                    VocabParallelEmbedding,
                ),
            ):
                if "visual.blocks" in full_name and "qkv" in full_name:
                    shard_dim = 1
                else:
                    shard_dim = 0
            elif isinstance(module, RowParallelLinear):
                if param_name == "bias":
                    # NOTE(zym) bias doesn't need to be sharded
                    tp_size = 1
                    shard_indices = [(0,)]
                    shard_dim = 0
                else:
                    shard_dim = 1
            elif isinstance(module, FusedMoE):
                if "w13" in param_name:
                    shard_dim = 0
                else:
                    assert "w2" in param_name, f"FusedMoE param can only be w13 and w2, but get {param_name}"
                    shard_dim = 1
            elif isinstance(module, ReplicatedLinear):
                # qwen2_moe  mlp.gate.weight
                # NOTE(zym): ReplicatedLinear layer doesn't use tp, but it still has tp_size
                # which is equal to get_tensor_model_parallel_world_size().
                # Refer to vllm/vllm/model_executor/layers/linear.py
                tp_size = 1
                shard_indices = [(0,)]
                shard_dim = 0
            elif isinstance(module, Qwen3_5GatedDeltaNet):
                # NOTE(zym): For param dt_bias and A_log
                shard_dim = 0
            else:
                raise ValueError(f"Unsupported module type for sharding: {type(module)}")
        else:
            shard_indices = [(0,)]
            shard_dim = 0
        kwargs = {
            "shard_mesh": OrderedDict([(shard_dim, tp_size)]),
            "shard_indices": shard_indices,
        }
        return NIXLSharding(**kwargs)


def convert_vllm_inplace(
    parameter_mapping: ParameterMapping,
    model,
    tp_rank: int = 0,
) -> tuple[dict[str, torch.Tensor], dict[str, NIXLSharding]]:
    """
    Convenience function to convert vLLM model to unified state dict and sharding info.
    Args:
        parameter_mapping: Parameter mapping instance for the specific model
        model: The vLLM model instance
        tp_rank: tensor parallel rank
    Returns:
        (converted_state_dict, sharding_dict)
    """
    converter = VllmConverter(parameter_mapping, tp_rank=tp_rank)
    return converter.convert_state_and_sharding_dict(model)
