from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import torch
from torch.nn import Parameter


def make_slice_parameter(slice_data: torch.Tensor, param: Parameter) -> Parameter:
    """
    Make a slice parameter from a slice data tensor and a parameter.
    This is inplace operation.
    """
    new_param = Parameter(slice_data, requires_grad=param.requires_grad)
    for attr_name, attr_value in param.__dict__.items():
        if attr_name not in ["data", "_grad", "_grad_fn", "_backward_hooks"]:
            setattr(new_param, attr_name, attr_value)
    return new_param


# Suffix patterns that identify Q, K, V projection weights in HF/FSDP state dicts.
_QKV_WEIGHT_SUFFIXES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "q_proj.bias",
    "k_proj.bias",
    "v_proj.bias",
)


def is_qkv_weight(param_name: str) -> bool:
    """
    Return True if param_name ends with a Q/K/V projection weight suffix.

    Args:
        param_name (str): Fully-qualified parameter name from a state dict.

    Returns:
        bool: True if the parameter is a Q, K, or V projection weight or bias.
    """
    return any(param_name.endswith(suffix) for suffix in _QKV_WEIGHT_SUFFIXES)


def reshape_qkv_to_3d(
    param: Parameter,
    num_heads_local: int,
    num_kv_heads_local: int,
    head_size: int,
) -> Parameter:
    """
    Reshape a 2D Q/K/V weight tensor into the 3D group-interleaved layout used by
    slice_qkv_proj_megatron, so that all three sides (Megatron/PS/vLLM) register
    the same shape with NIXL and can_xfer_to passes.

    Args:
        param (Parameter): 2D tensor of shape (rows, hidden) where rows is either
            num_heads_local * head_size (for Q) or num_kv_heads_local * head_size (for K/V).
        num_heads_local (int): Number of Q heads on this TP rank.
        num_kv_heads_local (int): Number of KV heads on this TP rank.
        head_size (int): Dimension of each head.

    Returns:
        Parameter: View of shape (num_groups_local, heads_per_group * head_size, hidden)
            where num_groups_local = gcd(num_heads_local, num_kv_heads_local).
    """
    import math

    num_groups_local = math.gcd(num_heads_local, num_kv_heads_local)
    rows = param.shape[0]
    hidden = param.shape[1] if param.ndim == 2 else 1
    assert rows % num_groups_local == 0, (
        f"rows={rows} is not divisible by num_groups_local={num_groups_local}."
    )
    new_shape = (num_groups_local, rows // num_groups_local, hidden)
    # NOTE(lhy): param.data is always contiguous (it is directly from HF state_dict or vLLM
    # fused split), so reshape produces a view, not a copy.
    reshaped_data = param.data.reshape(new_shape)
    return make_slice_parameter(reshaped_data, param)


def slice_gate_up_proj(
    fused_param: Parameter,
    output_sizes: list[int],
    tp_size: int = 1,
    output_dim: int = 0,
) -> list[Parameter]:
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
    assert all([output_size % tp_size == 0 for output_size in output_sizes]), (
        "Output sizes must be divisible by tensor parallel size"
    )
    gate_size, up_size = [output_size // tp_size for output_size in output_sizes]

    # Create views for gate and up projections without copying data
    assert fused_param.data.shape[output_dim] == (gate_size + up_size), (
        f"Dim {output_dim} of fused parameter shape {fused_param.data.shape} "
        f"must match the sum of gate and up sizes {[gate_size, up_size]}"
    )
    gate_data = fused_param.data.narrow(output_dim, 0, gate_size)
    up_data = fused_param.data.narrow(output_dim, gate_size, up_size)
    return [
        make_slice_parameter(gate_data, fused_param),
        make_slice_parameter(up_data, fused_param),
    ]


def slice_qkv_proj(
    fused_param: Parameter,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    tp_size: int = 1,
    output_dim: int = 0,
) -> list[Parameter]:
    """
    Split a fused qkv_proj parameter into three 2D shards:
      - q_param: query heads
      - k_param: key heads
      - v_param: value heads

    Args:
        fused_param (Parameter): The fused parameter of shape (total_qkv, hidden).
        num_heads (int): Number of query heads (global, across all TP ranks).
        num_kv_heads (int): Number of key/value heads (global, across all TP ranks).
        head_size (int): Dimensionality of each head.
        tp_size (int): Tensor parallel size (default is 1).
        output_dim (int): Dimension along which to split (default is 0).

    Returns:
        list[Parameter]: A list of three 2D parameters sharing storage with fused_param:
            [
              q_param, # shape (num_heads // tp_size * head_size, hidden)
              k_param, # shape (num_kv_heads // tp_size * head_size, hidden)
              v_param, # shape (num_kv_heads // tp_size * head_size, hidden)
            ]
    """
    assert num_heads % tp_size == 0, (
        "Number of heads must be divisible by tensor parallel size, "
        f"but got num_heads = {num_heads} and tp_size = {tp_size}."
    )
    assert num_kv_heads % tp_size == 0, (
        "Number of KV heads must be divisible by tensor parallel size, "
        f"but got num_kv_heads = {num_kv_heads} and tp_size = {tp_size}."
    )
    num_heads_local = num_heads // tp_size
    num_kv_heads_local = num_kv_heads // tp_size
    q_len = num_heads_local * head_size
    k_len = num_kv_heads_local * head_size
    v_len = num_kv_heads_local * head_size

    assert fused_param.data.shape[output_dim] == (q_len + k_len + v_len), (
        f"Dim {output_dim} of fused parameter shape {fused_param.data.shape} "
        f"must match the sum of qkv lengths {[q_len, k_len, v_len]}."
    )
    offset_and_sizes = [
        (0, q_len),
        (q_len, k_len),
        (q_len + k_len, v_len),
    ]

    qkv_params: list[Parameter] = []
    for offset, size in offset_and_sizes:
        data = fused_param.data.narrow(output_dim, offset, size)
        qkv_params.append(make_slice_parameter(data, fused_param))
    return qkv_params


def slice_qkv_proj_megatron(
    fused_param: Parameter,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    tp_size: int = 1,
    output_dim: int = 1,
) -> list[Parameter]:
    """
    Split a fused qkv_proj parameter into three shards according to Megatron-style:
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
        A list of three parameters, each a 3D non-contiguous view sharing storage with fused_param:
        [
          q_param, # view of shape (num_groups, q_heads_per_group * head_size, hidden)
          k_param, # view of shape (num_groups, kv_heads_per_group * head_size, hidden)
          v_param, # view of shape (num_groups, kv_heads_per_group * head_size, hidden)
        ]
        where num_groups = gcd(num_heads, num_kv_heads) // tp_size.
        These views share storage with fused_param, so NIXL pull/push correctly operate
        on the actual fused linear_qkv.weight used by Megatron forward.
    """
    import math

    num_split_heads = math.gcd(num_heads, num_kv_heads)
    assert num_split_heads % tp_size == 0, (
        "Number of split heads must be divisible by tensor parallel size, "
        f"but got num_split_heads = {num_split_heads} and tp_size = {tp_size}"
    )

    q_len = head_size * num_heads // num_split_heads
    k_len = head_size * num_kv_heads // num_split_heads
    v_len = head_size * num_kv_heads // num_split_heads

    fused_param = fused_param.reshape(num_split_heads // tp_size, q_len + k_len + v_len, -1)
    assert fused_param.data.shape[output_dim] == (q_len + k_len + v_len), (
        f"Dim {output_dim} of fused parameter shape {fused_param.data.shape} must "
        f"match the sum of qkv lengths {[q_len, k_len, v_len]}"
    )
    offset_and_sizes = [
        (0, q_len),
        (q_len, k_len),
        (q_len + k_len, v_len),
    ]

    qkv_params: list[Parameter] = []
    for offset, size in offset_and_sizes:
        # NOTE(lhy): Keep as a 3D non-contiguous view to share storage with fused_param.
        # Do NOT reshape to 2D here: reshape() on a non-contiguous tensor triggers an implicit
        # copy, producing an independent tensor whose storage is separate from fused_param.
        # NIXL pull would then write to the copy and never update the actual
        # linear_qkv.weight used by Megatron forward, causing training_ppl to explode.
        # The 3D non-contiguous path in NIXL handles this via a temp buffer and
        # original_tensor.data.copy_() after transfer, correctly updating fused_param.
        data = fused_param.data.narrow(output_dim, offset, size)
        qkv_params.append(make_slice_parameter(data, fused_param))
    return qkv_params


def slice_fused_moe_w13_weight(
    fused_param: Parameter,
) -> list[Parameter]:
    expert_params: list[Parameter] = []
    expert_num = fused_param.shape[0]
    shard_size = fused_param.shape[1] // 2
    for expert_id in range(expert_num):
        expert = fused_param.data[expert_id]
        gate = expert.narrow(0, 0, shard_size)
        up = expert.narrow(0, shard_size, shard_size)
        expert_params.append(make_slice_parameter(gate, fused_param))
        expert_params.append(make_slice_parameter(up, fused_param))
    return expert_params


def slice_fused_moe_w2_weight(
    fused_param: Parameter,
) -> list[Parameter]:
    expert_params: list[Parameter] = []
    expert_num = fused_param.shape[0]
    for expert_id in range(expert_num):
        down = fused_param.data[expert_id]
        expert_params.append(make_slice_parameter(down, fused_param))
    return expert_params


class MappingType(Enum):
    """Enum for mapping prototypes."""

    DIRECT = "direct"
    QKV_SPLIT = "qkv_split"
    GATE_UP_PROJ_SPLIT = "gate_up_proj_split"
    FUSED_MOE_W13_SPLIT = "fused_moe_w13_split"
    FUSED_MOE_W2_SPLIT = "fused_moe_w2_split"


class ParameterMapping(ABC):
    """Abstract base class for parameter mappings."""

    def __init__(self, config):
        """
        Args:
            config: HuggingFace model config (e.g. from AutoConfig.from_pretrained).
        """
        self.config = config

    @abstractmethod
    def get_mappings(self) -> list[tuple[str, str, MappingType, int]]:
        """Return list of (original_param_name, hf_param_name, mapping_prototype, shard_id) mappings."""
        pass

    def get_model_info(self) -> dict[str, Any]:
        """
        Return model-specific information needed for parameter splitting.

        Subclasses that need extra fields (e.g. num_experts) should call
        super().get_model_info() and extend the returned dict.
        """
        head_size = getattr(self.config, "head_dim", None)
        if head_size is None:
            head_size = self.config.hidden_size // self.config.num_attention_heads
        return {
            "num_heads": self.config.num_attention_heads,
            "num_kv_heads": getattr(self.config, "num_key_value_heads", self.config.num_attention_heads),
            "head_size": head_size,
            "intermediate_size": self.config.intermediate_size,
        }


class ModelRegistry:
    """Registry for different model parameter mappings based on model class types."""

    def __init__(self):
        self._mappings: dict[type | str, type[ParameterMapping]] = {}
        self._reverse_mappings: dict[type[ParameterMapping], list[type | str]] = {}

    def register(
        self,
        model_classes: type | str | list[type | str],
        mapping_class: type[ParameterMapping],
    ):
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

    def create_mapping(self, model_class: type | str, config) -> ParameterMapping:
        """
        Create a parameter mapping instance for a model class or class name.

        Args:
            model_class (type | str): The model class or its string name.
            config: HuggingFace model config (e.g. from AutoConfig.from_pretrained).

        Returns:
            ParameterMapping: An instantiated parameter mapping for the given model.
        """
        if model_class not in self._mappings:
            supported_classes = list(self._mappings.keys())
            supported_names = [str(c) for c in supported_classes]
            raise ValueError(f"Unsupported model class: {model_class}. Supported classes: {supported_names}")

        mapping_class = self._mappings[model_class]
        return mapping_class(config)

    def get_supported_models(self) -> list[type | str]:
        """Get list of supported model classes."""
        return list(self._mappings.keys())

    def unregister_mapping(self, mapping_class: type[ParameterMapping]):
        """Unregister a parameter mapping and all its associated model classes."""
        if mapping_class in self._reverse_mappings:
            for model_class in self._reverse_mappings[mapping_class]:
                if model_class in self._mappings:
                    del self._mappings[model_class]
            del self._reverse_mappings[mapping_class]


def register_model(model_classes: type | str | list[type | str]):
    """Decorator to register a model parameter mapping for one or more model classes."""

    def decorator(mapping_class: type[ParameterMapping]):
        model_registry.register(model_classes, mapping_class)
        return mapping_class

    return decorator


# Factory function for creating parameter mappings
def create_parameter_mapping(model_class: type | str, config) -> ParameterMapping:
    """
    Create parameter mapping for a specific model class or class name.

    Args:
        model_class (type | str): The model class or its string name.
        config: HuggingFace model config (e.g. from AutoConfig.from_pretrained).

    Returns:
        ParameterMapping: An instantiated parameter mapping for the given model.
    """
    return model_registry.create_mapping(model_class, config)


# Global registry instance
model_registry = ModelRegistry()
