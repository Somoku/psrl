import torch
from torch.nn import Parameter
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
from vllm.model_executor.layers.linear import set_weight_attrs
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear, MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

from psrl.utils.state_dict.model_mappings import ParameterMapping
from psrl.utils.state_dict.base_converter import BaseConverter
from psrl.utils.nixl.nixl_spec import NIXLSharding


def enable_sharded_weight_attrs(params: dict[str, Parameter]):
    for name, param in params.items():
        set_weight_attrs(param, {"is_sharded_weight": True})
    return params


def slice_gate_up_proj(
    fused: torch.Tensor,
    output_sizes: List[int],
    tp_size: int = 1,
    output_dim: int = 0
) -> Dict[str, torch.Tensor]:
    """
    Split a fused gate_up_proj tensor into two shards:
      - "gate_proj" (shard index 0)
      - "up_proj"   (shard index 1)

    Parameters
    ----------
    fused : torch.Tensor
        The fused tensor of shape [..., sum(output_sizes), ...].
    output_sizes : List[int]
        List of two sizes [gate_size, up_size].
    output_dim : int, optional
        Dimension along which to split (default is 0).

    Returns
    -------
    Dict[str, torch.Tensor]
        A mapping from shard name to the corresponding tensor view:
        {
          "gate_proj": view of shape [..., gate_size, ...],
          "up_proj": view of shape [..., up_size, ...],
        }
    """
    assert len(output_sizes) == 2, "Expected exactly two shards for gate_up_proj"
    assert all([output_size % tp_size == 0 for output_size in output_sizes]), \
        "Output sizes must be divisible by tensor parallel size"
    gate_size, up_size = [output_size // tp_size for output_size in output_sizes]

    # Create views for gate and up projections without copying data
    gate_view = fused.narrow(output_dim, 0, gate_size)
    up_view = fused.narrow(output_dim, gate_size, up_size)
    return {"gate_proj": gate_view, "up_proj": up_view}


def slice_qkv_proj(
    fused: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    tp_size: int = 1,
    output_dim: int = 0
) -> Dict[str, torch.Tensor]:
    """
    Split a fused qkv_proj tensor into three shards:
      - "q": query heads
      - "k": key heads
      - "v": value heads

    Parameters
    ----------
    fused : torch.Tensor
        The fused tensor of shape [..., total_qkv, ...].
    num_heads : int
        Number of query heads.
    num_kv_heads : int
        Number of key/value heads.
    head_size : int
        Dimensionality of each head.
    output_dim : int, optional
        Dimension along which to split (default is 0).

    Returns
    -------
    Dict[str, torch.Tensor]
        A mapping from shard name to the corresponding tensor view:
        {
          "q": view of shape [..., num_heads * head_size, ...],
          "k": view of shape [..., num_kv_heads * head_size, ...],
          "v": view of shape [..., num_kv_heads * head_size, ...],
        }
    """
    # Compute offset and length for each of Q, K, and V segments
    assert num_heads % tp_size == 0, f"Number of heads must be divisible by tensor parallel size, but got num_heads = {num_heads} and tp_size = {tp_size}"
    assert num_kv_heads % tp_size == 0, f"Number of KV heads must be divisible by tensor parallel size, but got num_kv_heads = {num_kv_heads} and tp_size = {tp_size}"
    q_len = num_heads // tp_size * head_size
    k_len = num_kv_heads // tp_size * head_size
    v_len = num_kv_heads // tp_size * head_size

    offsets = {
        "q": (0, q_len),
        "k": (q_len, k_len),
        "v": (q_len + k_len, v_len),
    }

    slices: Dict[str, torch.Tensor] = {}
    for shard_name, (offset, size) in offsets.items():
        # Create a view for each shard without copying data
        slices[shard_name] = fused.narrow(output_dim, offset, size)
    return slices


class VllmConverter(BaseConverter):
    """Convert vLLM model to a unified format (i.e., HuggingFace) and generate sharding info."""
    
    def __init__(self, parameter_mapping: ParameterMapping, tp_rank: Optional[int] = None):
        self.parameter_mapping = parameter_mapping
        self.tp_rank = tp_rank
        self.model_info = parameter_mapping.get_model_info()
        self.mappings = parameter_mapping.get_mappings()
        # Create reverse mapping for efficient lookup
        self.reverse_mappings = {}
        for vllm_name, hf_name, shard_id in self.mappings:
            if vllm_name not in self.reverse_mappings:
                self.reverse_mappings[vllm_name] = []
            self.reverse_mappings[vllm_name].append((hf_name, shard_id))
        
    def convert_state_and_sharding_dict(self, model) -> Tuple[Dict[str, torch.Tensor], Dict[str, NIXLSharding]]:
        """
        Convert vLLM model to unified state dict and generate sharding info.
        Args:
            model: The vLLM model instance
        Returns:
            (converted_state_dict, sharding)
        """
        converted_state_dict = {}
        sharding_dict = {}
        
        # Workaround: for lm_head, we do not care if it shares the weight with wte
        lm_head_module = None
        lm_head_module_prefix = None
        if hasattr(model, 'lm_head'):
            lm_head_module = model.lm_head
            lm_head_module_prefix = "lm_head"

        seen_module_prefixes = set()
        for module_prefix, module in model.named_modules():
            seen_module_prefixes.add(module_prefix)
            for param_name, param in module.named_parameters(recurse=False):
                full_name = f"{module_prefix}.{param_name}" if module_prefix else param_name
                if full_name.startswith("."):
                    full_name = full_name[1:]
                # Split parameter if needed
                split_params = self.split_parameter_if_needed(full_name, param, module)
                for new_param_name, new_param in split_params.items():
                    converted_state_dict[new_param_name] = new_param
                    sharding_dict[new_param_name] = self.get_sharding_for_param(module)
        
        # Handle lm_head separately
        if lm_head_module is not None and (lm_head_module_prefix not in seen_module_prefixes):
            module = lm_head_module
            module_prefix = lm_head_module_prefix
            for param_name, param in module.named_parameters(recurse=False):
                full_name = f"{module_prefix}.{param_name}" if module_prefix else param_name
                split_params = self.split_parameter_if_needed(full_name, param, module)
                for new_param_name, new_param in split_params.items():
                    converted_state_dict[new_param_name] = new_param
                    sharding_dict[new_param_name] = self.get_sharding_for_param(module)
        return converted_state_dict, sharding_dict

    def split_parameter_if_needed(self, full_name: str, param: Parameter, module) -> dict:
        """
        Split the parameter if it matches a known split pattern (e.g., qkv_proj, gate_up_proj).
        Returns a dict of {new_param_name: new_param}.
        If no split is needed, returns {full_name: param}.
        """
        tp_size = getattr(module, "tp_size", 1)
        for vllm_name in self.reverse_mappings:
            if vllm_name in full_name:
                mappings = self.reverse_mappings[vllm_name]
                if vllm_name == "qkv_proj":
                    slices = slice_qkv_proj(
                        fused=param.data,
                        num_heads=self.model_info["num_heads"],
                        num_kv_heads=self.model_info["num_kv_heads"],
                        head_size=self.model_info["head_size"],
                        tp_size=tp_size,
                        output_dim=0
                    )
                    out = {}
                    for hf_name, shard_id in mappings:
                        if shard_id in slices:
                            slice_data = slices[shard_id]
                            new_param = Parameter(slice_data, requires_grad=param.requires_grad)
                            for attr_name, attr_value in param.__dict__.items():
                                if attr_name not in ['data', '_grad', '_grad_fn', '_backward_hooks']:
                                    setattr(new_param, attr_name, attr_value)
                            new_param_name = full_name.replace(vllm_name, hf_name)
                            out[new_param_name] = new_param
                    return out
                elif vllm_name == "gate_up_proj":
                    intermediate_size = self.model_info["intermediate_size"]
                    slices = slice_gate_up_proj(
                        fused=param.data,
                        output_sizes=[intermediate_size, intermediate_size],
                        tp_size=tp_size,
                        output_dim=0
                    )
                    out = {}
                    for hf_name, shard_id in mappings:
                        if shard_id == 0 and "gate_proj" in slices:
                            slice_data = slices["gate_proj"]
                        elif shard_id == 1 and "up_proj" in slices:
                            slice_data = slices["up_proj"]
                        else:
                            continue
                        new_param = Parameter(slice_data, requires_grad=param.requires_grad)
                        for attr_name, attr_value in param.__dict__.items():
                            if attr_name not in ['data', '_grad', '_grad_fn', '_backward_hooks']:
                                setattr(new_param, attr_name, attr_value)
                        new_param_name = full_name.replace(vllm_name, hf_name)
                        out[new_param_name] = new_param
                    return out
                # Extend here for other split types if needed
        # No split needed
        return {full_name: param}

    def get_sharding_for_param(self, module) -> NIXLSharding:
        """
        Generate sharding info for a parameter given its module and tp_rank.
        Returns a NIXLSharding object.
        """
        tp_size = getattr(module, "tp_size", 1)
        if tp_size > 1:
            shard_indices = [(self.tp_rank,)] if self.tp_rank is not None else [(0,)]
            if isinstance(module, (ColumnParallelLinear, MergedColumnParallelLinear, QKVParallelLinear, VocabParallelEmbedding)):
                shard_dim = 0
            elif isinstance(module, RowParallelLinear):
                shard_dim = 1
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


def convert_vllm_inplace(parameter_mapping: ParameterMapping, model, tp_rank: int = 0) -> Tuple[Dict[str, torch.Tensor], Dict[str, NIXLSharding]]:
    """
    Convenience function to convert vLLM model to unified state dict and sharding info.
    Args:
        parameter_mapping: Parameter mapping instance for the specific model
        model: The vLLM model instance
        tp_rank: tensor parallel rank
    Returns:
        (converted_state_dict, sharding)
    """
    converter = VllmConverter(parameter_mapping, tp_rank=tp_rank)
    return converter.convert_state_and_sharding_dict(model)
