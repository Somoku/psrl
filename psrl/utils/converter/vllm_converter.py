from collections import OrderedDict
from dataclasses import dataclass, field

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
from vllm.model_executor.models.interfaces import SupportsWeightLayoutSpec

from psrl.utils.converter.base_converter import BaseConverter
from psrl.utils.converter.model_mappings import (
    MappingType,
    ParameterMapping,
    slice_fused_moe_w2_weight,
    slice_fused_moe_w13_weight,
    slice_gate_up_proj,
    slice_qkv_proj,
)
from psrl.utils.nixl.nixl_spec import NIXLSharding

# QKV string shard_id → integer index
# vLLM uses "q"/"k"/"v" in stacked_params_mapping, but convert_parameter()
# uses integer index: assert shard_id < len(sliced_params); sliced_params[shard_id]
QKV_SHARD_ID_MAP: dict[str, int] = {"q": 0, "k": 1, "v": 2}


@dataclass
class ParamMappingEntry:
    mapping_type: MappingType
    mappings: list = field(default_factory=list)
    is_full_path: bool = False


def enable_sharded_weight_attrs(params: dict[str, Parameter]):
    for name, param in params.items():
        set_weight_attrs(param, {"is_sharded_weight": True})
    return params


class VllmConverter(BaseConverter):
    """Convert vLLM model to a unified format (i.e., HuggingFace) and generate sharding info."""

    def __init__(self, parameter_mapping: ParameterMapping | None, tp_rank: int | None = 1):
        super().__init__(parameter_mapping)
        self.parameter_mapping = parameter_mapping
        self.tp_rank = tp_rank

    def _build_fused_mappings(
        self,
        stacked_params: list,
        extra_stacked_params: list,
        expert_params: list,
    ) -> dict:
        fused_mappings: dict[str, ParamMappingEntry] = {}

        for packed_suffix, hf_suffix, shard_id in stacked_params:
            if packed_suffix not in fused_mappings:
                mapping_type = MappingType.QKV_SPLIT if isinstance(shard_id, str) else MappingType.GATE_UP_PROJ_SPLIT
                fused_mappings[packed_suffix] = ParamMappingEntry(mapping_type=mapping_type, is_full_path=False)
            int_shard_id = QKV_SHARD_ID_MAP[shard_id] if isinstance(shard_id, str) else shard_id
            fused_mappings[packed_suffix].mappings.append((hf_suffix, int_shard_id))

        for full_packed, full_hf, shard_id in extra_stacked_params:
            if full_packed not in fused_mappings:
                fused_mappings[full_packed] = ParamMappingEntry(
                    mapping_type=MappingType.GATE_UP_PROJ_SPLIT, is_full_path=True
                )
            int_shard_id = QKV_SHARD_ID_MAP[shard_id] if isinstance(shard_id, str) else shard_id
            fused_mappings[full_packed].mappings.append((full_hf, int_shard_id))

        # "w1" (gate_proj), "w2" (down_proj), "w3" (up_proj)
        w13_by_suffix: dict[str, list] = {}
        w2_by_suffix: dict[str, list] = {}
        for packed_suffix, hf_suffix, expert_id, shard_id in expert_params:
            if shard_id in ("w1", "w3"):
                int_shard_id = expert_id * 2 if shard_id == "w1" else expert_id * 2 + 1
                w13_by_suffix.setdefault(packed_suffix, []).append((hf_suffix, int_shard_id))
            else:
                w2_by_suffix.setdefault(packed_suffix, []).append((hf_suffix, expert_id))

        for suffix, entries in w13_by_suffix.items():
            fused_mappings[suffix] = ParamMappingEntry(
                mapping_type=MappingType.FUSED_MOE_W13_SPLIT, mappings=entries, is_full_path=False
            )
        for suffix, entries in w2_by_suffix.items():
            fused_mappings[suffix] = ParamMappingEntry(
                mapping_type=MappingType.FUSED_MOE_W2_SPLIT, mappings=entries, is_full_path=False
            )

        return fused_mappings

    @staticmethod
    def _spec_has_content(spec) -> bool:
        return bool(spec.stacked_params or spec.expert_params or spec.extra_stacked_params)

    def _build_from_spec(self, spec) -> tuple[dict, dict]:
        """Build fused_mappings and model_info from a non-empty WeightLayoutSpec."""
        fused = self._build_fused_mappings(
            spec.stacked_params,
            spec.extra_stacked_params or [],
            spec.expert_params or [],
        )
        return fused, spec.packing_metadata

    def _build_from_parameter_mapping(self) -> tuple[dict, dict]:
        fused_mappings: dict = {}
        for vllm_name, hf_name, mapping_type, shard_id in self.parameter_mapping.get_mappings():
            if vllm_name not in fused_mappings:
                fused_mappings[vllm_name] = ParamMappingEntry(mapping_type=mapping_type, is_full_path=False)
            else:
                assert mapping_type != MappingType.DIRECT, f"Mapping type should not be DIRECT for {vllm_name}"
                assert mapping_type == fused_mappings[vllm_name].mapping_type, (
                    f"Mapping type for {vllm_name} must be the same, "
                    f"but got {mapping_type} and {fused_mappings[vllm_name].mapping_type}"
                )
            fused_mappings[vllm_name].mappings.append((hf_name, shard_id))
        model_info = self.parameter_mapping.get_model_info()
        return fused_mappings, model_info

    def convert_state_and_sharding_dict(self, model) -> tuple[dict[str, torch.Tensor], dict[str, NIXLSharding]]:
        """
        Convert vLLM model to unified state dict and generate sharding info.
        Args:
            model: The vLLM model instance
        Returns:
            (converted_state_dict, sharding_dict)
        """
        if isinstance(model, SupportsWeightLayoutSpec):
            top_spec = model.get_weight_layout_spec()
            if self._spec_has_content(top_spec):
                # Pattern A/B/E: top-level spec is non-empty — use it directly.
                fused_mappings, model_info = self._build_from_spec(top_spec)
            else:
                # Pattern C/D: top-level spec is empty.
                # Walk sub-modules to find the first non-empty spec (Pattern D: e.g. KimiK25
                # wrapping DeepseekV2). If none is found, all weights are already in HF format
                # and passthrough is correct (Pattern C: e.g. Mamba, Qwen3).
                sub_spec = next(
                    (
                        spec
                        for _, m in model.named_modules()
                        if isinstance(m, SupportsWeightLayoutSpec)
                        and self._spec_has_content(spec := m.get_weight_layout_spec())
                    ),
                    None,
                )
                fused_mappings, model_info = self._build_from_spec(sub_spec) if sub_spec else ({}, {})
            # Sync self.model_info so that maybe_reshape_qkv_to_3d (which reads
            # self.model_info) sees the packing_metadata from the spec.
            self.model_info = model_info
        elif self.parameter_mapping is not None:
            fused_mappings, model_info = self._build_from_parameter_mapping()
        else:
            raise ValueError(
                f"{type(model).__name__} does not implement SupportsWeightLayoutSpec "
                "and no parameter_mapping was provided. Either implement "
                "get_weight_layout_spec() in the model, or pass parameter_mapping= "
                "to convert_vllm_inplace()."
            )

        converted_state_dict = {}
        sharding_dict = {}

        # Workaround: for lm_head, we do not care if it shares the weight with wte
        lm_head_module = None
        lm_head_module_prefix = None
        if hasattr(model, "lm_head"):
            lm_head_module = model.lm_head
            lm_head_module_prefix = "lm_head"

        seen_module_prefixes = set()
        for module_prefix, module in model.named_modules():
            seen_module_prefixes.add(module_prefix)
            for param_name, param in module.named_parameters(recurse=False):
                full_name = f"{module_prefix}.{param_name}" if module_prefix else param_name
                new_params = self.convert_parameter(full_name, param, module, fused_mappings, model_info)
                sharding = self.get_sharding_for_param(module, param_name)
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
                new_params = self.convert_parameter(full_name, param, module, fused_mappings, model_info)
                sharding = self.get_sharding_for_param(module, param_name)
                for new_param_name, new_param in new_params.items():
                    new_param, sharding_for_param = self.maybe_reshape_qkv_to_3d(new_param_name, new_param, sharding)
                    converted_state_dict[new_param_name] = new_param
                    sharding_dict[new_param_name] = sharding_for_param

        return converted_state_dict, sharding_dict

    def convert_parameter(
        self,
        full_name: str,
        param: Parameter,
        module,
        fused_mappings: dict,
        model_info: dict,
    ) -> dict:
        """
        Convert the parameter, may need to split inplace
        if it matches a split mapping type (e.g., qkv_proj, gate_up_proj).
        """
        tp_size = getattr(module, "tp_size", 1)
        for vllm_name, entry in fused_mappings.items():
            mapping_type = entry.mapping_type
            mappings = entry.mappings
            is_full_path = entry.is_full_path
            matched = (full_name == vllm_name) if is_full_path else (vllm_name in full_name)
            if not matched:
                continue
            if mapping_type == MappingType.DIRECT:
                assert len(mappings) == 1, f"Mapping type is DIRECT for {vllm_name}, but got {len(mappings)} mappings"
                new_param = param
                hf_name = mappings[0][0]
                new_param_name = hf_name if is_full_path else full_name.replace(vllm_name, hf_name)
                return {new_param_name: new_param}
            elif mapping_type == MappingType.QKV_SPLIT:
                try:
                    sliced_params = slice_qkv_proj(
                        fused_param=param,
                        num_heads=model_info["num_heads"],
                        num_kv_heads=model_info["num_kv_heads"],
                        head_size=model_info["head_size"],
                        tp_size=tp_size,
                    )
                except Exception as e:
                    raise ValueError(f"Failed to slice qkv parameter {full_name}: {e}") from e
                out = {}
                for hf_name, shard_id in mappings:
                    assert shard_id < len(sliced_params), f"Shard id {shard_id} is out of range for {vllm_name}"
                    new_param = sliced_params[shard_id]
                    new_param_name = hf_name if is_full_path else full_name.replace(vllm_name, hf_name)
                    out[new_param_name] = new_param
                return out
            elif mapping_type == MappingType.GATE_UP_PROJ_SPLIT:
                intermediate_size = model_info["intermediate_size"]
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
                    new_param_name = hf_name if is_full_path else full_name.replace(vllm_name, hf_name)
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
                ep_rank = self.tp_rank if ep_size > 1 else 0  # considering the case where not enable_expert_parallel
                num_experts = model_info["num_experts"]
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
                    new_param_name = hf_name if is_full_path else full_name.replace(vllm_name, hf_name)
                    out[new_param_name] = new_param
                return out
            elif mapping_type == MappingType.FUSED_MOE_W2_SPLIT:
                try:
                    sliced_params = slice_fused_moe_w2_weight(
                        fused_param=param,
                    )
                except Exception as e:
                    raise ValueError(f"Failed to slice w13_weight parameter {full_name}: {e}") from e
                out = {}
                ep_size = getattr(module, "ep_size", 1)
                ep_rank = self.tp_rank if ep_size > 1 else 0
                num_experts = model_info["num_experts"]
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
                    new_param_name = hf_name if is_full_path else full_name.replace(vllm_name, hf_name)
                    out[new_param_name] = new_param
                return out
            else:
                raise ValueError(f"Unsupported mapping type: {mapping_type}")
        # Default: No conversion needed
        return {full_name: param}

    def get_sharding_for_param(self, module, param_name) -> NIXLSharding:
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
                shard_dim = 0
            elif isinstance(module, RowParallelLinear):
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
    model,
    tp_rank: int = 0,
    *,
    parameter_mapping: ParameterMapping | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, NIXLSharding]]:
    """
    Convert a vLLM model to unified HF-format state dict with NIXL sharding info.

    For models implementing SupportsWeightLayoutSpec, no parameter_mapping is needed.
    For custom models not yet supporting the interface, pass parameter_mapping as
    a keyword argument.
    """
    converter = VllmConverter(parameter_mapping=parameter_mapping, tp_rank=tp_rank)
    return converter.convert_state_and_sharding_dict(model)
