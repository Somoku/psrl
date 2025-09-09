import torch
from torch.nn import Parameter
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Type, Union
from enum import Enum


def make_slice_parameter(
    slice_data: torch.Tensor,
    param: Parameter
) -> Parameter:
    """
    Make a slice parameter from a slice data tensor and a parameter.
    This is inplace operation.
    """
    new_param = Parameter(slice_data, requires_grad=param.requires_grad)
    for attr_name, attr_value in param.__dict__.items():
        if attr_name not in ['data', '_grad', '_grad_fn', '_backward_hooks']:
            setattr(new_param, attr_name, attr_value)
    return new_param


def slice_gate_up_proj(
    fused_param: Parameter,
    output_sizes: List[int],
    tp_size: int = 1,
    output_dim: int = 0
) -> List[Parameter]:
    """
    Split a fused gate_up_proj parameter into two shards:
      - gate_proj_param (shard index 0)
      - up_proj_param   (shard index 1)

    Args
    ----------
    fused_param : Parameter
        The fused parameter of shape [..., sum(output_sizes), ...].
    output_sizes : List[int]
        List of two sizes [gate_size, up_size].
    tp_size : int, optional
        Tensor parallel size (default is 1).
    output_dim : int, optional
        Dimension along which to split (default is 0).

    Returns
    -------
    List[Parameter]
        A list of two parameters:
        [
          gate_proj_param, # view of shape [..., gate_size, ...]
          up_proj_param, # view of shape [..., up_size, ...]
        ]
    """
    assert len(output_sizes) == 2, "Expected exactly two shards for gate_up_proj"
    assert all([output_size % tp_size == 0 for output_size in output_sizes]), \
        "Output sizes must be divisible by tensor parallel size"
    gate_size, up_size = [output_size // tp_size for output_size in output_sizes]

    # Create views for gate and up projections without copying data
    assert fused_param.data.shape[output_dim] == (gate_size + up_size), f"Dim {output_dim} of fused parameter shape {fused_param.data.shape} must match the sum of gate and up sizes {[gate_size, up_size]}"
    gate_data = fused_param.data.narrow(output_dim, 0, gate_size)
    up_data = fused_param.data.narrow(output_dim, gate_size, up_size)
    return [make_slice_parameter(gate_data, fused_param), make_slice_parameter(up_data, fused_param)]


def slice_qkv_proj(
    fused_param: Parameter,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    tp_size: int = 1,
    output_dim: int = 0
) -> List[Parameter]:
    """
    Split a fused qkv_proj parameter into three shards:
      - q_param: query heads
      - k_param: key heads
      - v_param: value heads

    Args
    ----------
    fused_param : Parameter
        The fused parameter of shape [..., total_qkv, ...].
    num_heads : int
        Number of query heads.
    num_kv_heads : int
        Number of key/value heads.
    head_size : int
        Dimensionality of each head.
    tp_size : int, optional
        Tensor parallel size (default is 1).
    output_dim : int, optional
        Dimension along which to split (default is 0).

    Returns
    -------
    List[Parameter]
        A list of three parameters:
        [
          q_param, # view of shape [..., num_heads * head_size, ...]
          k_param, # view of shape [..., num_kv_heads * head_size, ...]
          v_param, # view of shape [..., num_kv_heads * head_size, ...]
        ]
    """
    # Compute offset and length for each of Q, K, and V segments
    assert num_heads % tp_size == 0, f"Number of heads must be divisible by tensor parallel size, but got num_heads = {num_heads} and tp_size = {tp_size}"
    assert num_kv_heads % tp_size == 0, f"Number of KV heads must be divisible by tensor parallel size, but got num_kv_heads = {num_kv_heads} and tp_size = {tp_size}"
    q_len = num_heads // tp_size * head_size
    k_len = num_kv_heads // tp_size * head_size
    v_len = num_kv_heads // tp_size * head_size

    assert fused_param.data.shape[output_dim] == (q_len + k_len + v_len), f"Dim {output_dim} of fused parameter shape {fused_param.data.shape} must match the sum of qkv lengths {[q_len, k_len, v_len]}"
    offset_and_sizes = [
        (0, q_len),
        (q_len, k_len),
        (q_len + k_len, v_len),
    ]

    qkv_params: List[Parameter] = []
    for offset, size in offset_and_sizes:
        # Create a view for each shard without copying data
        data = fused_param.data.narrow(output_dim, offset, size)
        qkv_params.append(make_slice_parameter(data, fused_param))
    return qkv_params


class MappingType(Enum):
    """Enum for mapping prototypes."""
    DIRECT = "direct"
    QKV_SPLIT = "qkv_split"
    GATE_UP_PROJ_SPLIT = "gate_up_proj_split"


class ParameterMapping(ABC):
    """Abstract base class for parameter mappings."""
    
    @abstractmethod
    def get_mappings(self) -> List[Tuple[str, str, MappingType, int]]:
        """Return list of (original_param_name, hf_param_name, mapping_prototype, shard_id) mappings."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return model-specific information needed for parameter splitting."""
        pass


class ModelRegistry:
    """Registry for different model parameter mappings based on model class types."""
    
    def __init__(self):
        self._mappings: Dict[Union[Type, str], Type[ParameterMapping]] = {}
        self._reverse_mappings: Dict[Type[ParameterMapping], List[Union[Type, str]]] = {}
    
    def register(self, model_classes: Union[Type, str, List[Union[Type, str]]], mapping_class: Type[ParameterMapping]):
        """Register a parameter mapping for one or more model classes or class names."""
        if not isinstance(model_classes, list):
            model_classes = [model_classes]
        
        # Register the mapping for each model class
        for model_class in model_classes:
            self._mappings[model_class] = mapping_class
        
        # Store reverse mapping for cleanup
        if mapping_class not in self._reverse_mappings:
            self._reverse_mappings[mapping_class] = []
        self._reverse_mappings[mapping_class].extend(model_classes)
    
    def create_mapping(self, model_class: Union[Type, str], model_path: str) -> ParameterMapping:
        """Create a parameter mapping instance for a model class or class name."""
        if model_class not in self._mappings:
            supported_classes = list(self._mappings.keys())
            supported_names = [str(c) for c in supported_classes]
            raise ValueError(f"Unsupported model class: {model_class}. Supported classes: {supported_names}")
        
        mapping_class = self._mappings[model_class]
        return mapping_class(model_path)
    
    def get_supported_models(self) -> List[Union[Type, str]]:
        """Get list of supported model classes."""
        return list(self._mappings.keys())
    
    def unregister_mapping(self, mapping_class: Type[ParameterMapping]):
        """Unregister a parameter mapping and all its associated model classes."""
        if mapping_class in self._reverse_mappings:
            for model_class in self._reverse_mappings[mapping_class]:
                if model_class in self._mappings:
                    del self._mappings[model_class]
            del self._reverse_mappings[mapping_class]


def register_model(model_classes: Union[Type, str, List[Union[Type, str]]]):
    """Decorator to register a model parameter mapping for one or more model classes."""
    def decorator(mapping_class: Type[ParameterMapping]):
        model_registry.register(model_classes, mapping_class)
        return mapping_class
    return decorator


# Factory function for creating parameter mappings
def create_parameter_mapping(model_class: Union[Type, str], model_path: str) -> ParameterMapping:
    """Create parameter mapping for a specific model class or class name."""
    return model_registry.create_mapping(model_class, model_path) 


# Global registry instance
model_registry = ModelRegistry()