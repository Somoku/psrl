import math
from abc import ABC, abstractmethod
from collections import OrderedDict

import torch
from torch.nn import Parameter

from psrl.utils.converter.model_mappings import (
    ParameterMapping,
    is_qkv_weight,
    is_q_weight,
    make_slice_parameter,
    reshape_qkv_to_3d,
    reshape_q_to_5d,
)
from psrl.utils.nixl.nixl_spec import NIXLSharding


class BaseConverter(ABC):
    """Base class for state dict and sharding converter."""

    def __init__(self, parameter_mapping: ParameterMapping):
        """
        Initialize the converter with model info from the given parameter mapping.

        Args:
            parameter_mapping (ParameterMapping): Parameter mapping for the model.
                model_info is extracted here and shared across all converter methods.
        """
        self.model_info = parameter_mapping.get_model_info()

    def maybe_reshape_qkv_to_3d(
        self,
        param_name: str,
        param: Parameter,
        sharding: NIXLSharding,
    ) -> tuple[Parameter, NIXLSharding]:
        """
        Conditionally reshape a 2D Q/K/V weight to 3D group-interleaved layout and
        update the sharding descriptor to match the new tensor shape.

        Returns unchanged (param, sharding) if self.model_info does not contain
        num_heads, param_name is not a Q/K/V weight, or param is not 2-dimensional.

        Local head counts are derived from the sharding descriptor:
          - dim=0 sharding: num_heads_local = num_heads // ws (may be 0 for fine-grained FSDP)
          - dim=1 sharding: all heads present on this rank; use global counts

        Reshape rule: (rows, H) -> (G_eff, rows // G_eff, H)
        where G_eff = gcd(num_heads_local, num_kv_heads_local).

        Sharding update — three cases based on ws and G_global = gcd(num_heads, num_kv_heads):

          Case A — shard_dim_2d == 1 (hidden dim sharded):
            Hidden shifts from index 1 -> 2 after reshape.
            New sharding: shard_mesh={2: ws}, shard_indices=[(rank,)].

          Case B — shard_dim_2d == 0, ws <= G_global (coarse head sharding):
            Each rank holds whole head groups; shard_mesh={0: ws} stays valid in 3D.
            Sharding unchanged.

          Case C — shard_dim_2d == 0, ws > G_global (fine-grained, spills into dim=1):
            FSDP slices inside a group; spill into dim=1 of the 3D tensor.
            3D shape: (1, rows, H) (G_eff = 1 since num_heads_local may be 0).
            New sharding: shard_mesh={0: G_global, 1: ws // G_global},
                          shard_indices=[(rank // steps, rank % steps)]
            where steps = ws // G_global. Requires ws % G_global == 0.

        Args:
            param_name (str): Fully-qualified parameter name.
            param (Parameter): The 2D parameter tensor.
            sharding (NIXLSharding): Existing sharding descriptor for this param.

        Returns:
            tuple[Parameter, NIXLSharding]: Reshaped param and updated sharding.
        """
        num_heads = self.model_info.get("num_heads")
        if num_heads is None or not is_qkv_weight(param_name) or param.ndim != 2:
            return param, sharding

        num_kv_heads = self.model_info["num_kv_heads"]
        head_size = self.model_info["head_size"]
        G_global = math.gcd(num_heads, num_kv_heads)
        shard_dim_2d = next(iter(sharding.shard_mesh.keys()))
        ws = next(iter(sharding.shard_mesh.values()))
        rank = sharding.shard_indices[0][0]
        rows, H = param.shape

        attn_output_gate = self.model_info.get("attn_output_gate", False)

        # Derive local head counts from the sharding descriptor.
        if shard_dim_2d == 0:
            num_heads_local = num_heads // ws
            num_kv_heads_local = num_kv_heads // ws
        else:
            # dim=1 (hidden sharded): all heads are present on this rank.
            num_heads_local = num_heads
            num_kv_heads_local = num_kv_heads

        if shard_dim_2d == 1:
            # Case A: hidden dim sharded; hidden moves from dim=1 to dim=2 after reshape.
            reshaped = reshape_qkv_to_3d(
                make_slice_parameter(param.data, param),
                num_heads_local=num_heads_local,
                num_kv_heads_local=num_kv_heads_local,
                head_size=head_size,
            )
            new_sharding = NIXLSharding(
                shard_mesh=OrderedDict([(2, ws)]),
                shard_indices=[(rank,)],
            )
            if attn_output_gate and is_q_weight(param_name):
                reshaped = reshape_q_to_5d(
                    reshaped,
                    num_heads_local=num_heads_local,
                    head_size=head_size,
                )
                new_sharding = NIXLSharding(
                    shard_mesh=OrderedDict([(4, ws)]),
                    shard_indices=[(rank,)],
                )

        elif ws <= G_global:
            # Case B: coarse head sharding, shard_mesh={0: ws} stays valid in 3D.
            reshaped = reshape_qkv_to_3d(
                make_slice_parameter(param.data, param),
                num_heads_local=num_heads_local,
                num_kv_heads_local=num_kv_heads_local,
                head_size=head_size,
            )
            new_sharding = sharding
            if attn_output_gate and is_q_weight(param_name):
                reshaped = reshape_q_to_5d(
                    reshaped,
                    num_heads_local=num_heads_local,
                    head_size=head_size,
                )
        else:
            # Case C: fine-grained sharding spills from dim=0 into dim=1.
            assert ws % G_global == 0, (
                f"FSDP world size ({ws}) is not divisible by G_global={G_global} for {param_name}."
            )
            steps = ws // G_global
            reshaped = make_slice_parameter(param.data.reshape(1, rows, H), param)
            new_sharding = NIXLSharding(
                shard_mesh=OrderedDict([(0, G_global), (1, steps)]),
                shard_indices=[(rank // steps, rank % steps)],
            )
            if attn_output_gate and is_q_weight(param_name):
                raise NotImplementedError(
                    "attn_output_gate is not supported for fine-grained sharding (Case C)"
                )


        return reshaped, new_sharding

    @abstractmethod
    def convert_state_and_sharding_dict(self, model) -> tuple[dict[str, torch.Tensor], dict[str, NIXLSharding]]:
        pass
