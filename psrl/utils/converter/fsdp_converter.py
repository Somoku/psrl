import warnings
from collections import OrderedDict

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import StateDictType
from torch.distributed.tensor import DTensor
from verl.utils.fsdp_utils import fsdp_version

from psrl.utils.converter.base_converter import BaseConverter
from psrl.utils.converter.model_mappings import ParameterMapping
from psrl.utils.nixl.nixl_spec import NIXLSharding


class FSDPConverter(BaseConverter):
    """Converter for FSDP/FSDP2 model."""

    def __init__(self, fsdp_strategy: str, parameter_mapping: ParameterMapping):
        """
        Args:
            fsdp_strategy (str): FSDP strategy, either 'fsdp' or 'fsdp2'.
            parameter_mapping (ParameterMapping): Parameter mapping instance carrying
                model_info (num_heads, num_kv_heads, head_size). Use FSDPParameterMapping
                to enable Q/K/V 3D reshaping for NIXL shape compatibility between
                FSDP train workers and the PS.
        """
        super().__init__(parameter_mapping)
        self.fsdp_strategy = fsdp_strategy

    def convert_state_and_sharding_dict(self, model) -> tuple[dict[str, torch.Tensor], dict[str, NIXLSharding]]:
        """
        Convert FSDP/FSDP2 model to unified state dict and sharding info.

        Args:
            model: The FSDP/FSDP2 model instance.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, NIXLSharding]]: A pair of
                (converted_state_dict, sharding_dict).
        """
        # Determine the FSDP strategy and convert accordingly.
        # fsdp_state_dict will be (name, DTensor) pairs.
        if self.fsdp_strategy == "fsdp":
            warnings.warn(
                "FSDP strategy is deprecated beacause it cannot "
                "guarantee the in-place representation of the state dict.",
                stacklevel=2,
            )
            with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
                fsdp_state_dict = model.state_dict()
        elif self.fsdp_strategy == "fsdp2":
            fsdp_state_dict = model.state_dict()
        else:
            raise ValueError(f"Unsupported FSDP strategy: {self.fsdp_strategy}")

        # Convert the FSDP state dict to a unified format.
        converted_state_dict = {}
        sharding_dict = {}
        for param_name, param in fsdp_state_dict.items():
            assert isinstance(param, DTensor), f"Expected DTensor for {param_name}, got {type(param)}."
            local_param = param.to_local()
            # Compute sharding from original 2D DTensor placements before any reshape.
            sharding = self.get_sharding_for_param(param_name, param)
            # NOTE(lhy): Reshape Q/K/V local shards to 3D to match slice_qkv_proj_megatron layout
            # and update sharding to reflect the new tensor shape.
            local_param, sharding = self.maybe_reshape_qkv_to_3d(param_name, local_param, sharding)
            converted_state_dict[param_name] = local_param
            sharding_dict[param_name] = sharding
        return converted_state_dict, sharding_dict

    def get_sharding_for_param(self, param_name: str, param: DTensor) -> NIXLSharding:
        """
        Generate sharding info for a parameter.
        Returns a NIXLSharding object.
        """
        # FSDP
        if len(param.placements) == 1:
            assert param.device_mesh and param.device_mesh.ndim == 1, (
                f"Expected 1 dim device mesh for {param_name}, got {param.device_mesh}"
            )
            if param.placements[0].is_shard(dim=0):
                shard_dim = 0
            elif param.placements[0].is_shard(dim=1):
                shard_dim = 1
            else:
                raise ValueError(f"Unexpected shard_dim for {param_name}, got {param.placements}")
            kwargs = {
                "shard_mesh": OrderedDict([(shard_dim, param.device_mesh.size())]),
                "shard_indices": [(param.device_mesh.get_rank(),)],
            }
        # HSDP
        else:
            assert len(param.placements) == 2 and param.placements[0].is_replicate(), (
                f"Expected two shards (first replicate, second shard on dim 0) for {param_name} "
                f"when using hybrid FSDP, got {param.placements}"
            )
            assert param.device_mesh and param.device_mesh.ndim == 2, (
                f"Expected 2 dim device mesh for {param_name}, got {param.device_mesh}"
            )
            if param.placements[1].is_shard(dim=0):
                shard_dim = 0
            elif param.placements[1].is_shard(dim=1):
                shard_dim = 1
            else:
                raise ValueError(f"Unexpected shard_dim for {param_name}, got {param.placements}")
            kwargs = {
                "shard_mesh": OrderedDict([(shard_dim, param.device_mesh.size(mesh_dim=1))]),
                "shard_indices": [(param.device_mesh.get_local_rank(mesh_dim=1),)],
            }
        return NIXLSharding(**kwargs)


def convert_fsdp_inplace(
    parameter_mapping: ParameterMapping,
    model,
    fsdp_strategy: str = "fsdp2",
) -> tuple[dict[str, torch.Tensor], dict[str, NIXLSharding]]:
    """
    Convert FSDP/FSDP2 model to unified state dict and sharding info.

    Args:
        fsdp_strategy (str): FSDP strategy, either 'fsdp' or 'fsdp2'.
        model: The FSDP/FSDP2 model instance.
        parameter_mapping (ParameterMapping): Parameter mapping instance carrying
            model_info. Use FSDPParameterMapping to enable Q/K/V 3D reshaping for
            NIXL shape compatibility between FSDP train workers and the PS.

    Returns:
        tuple[dict[str, torch.Tensor], dict[str, NIXLSharding]]: A pair of
            (converted_state_dict, sharding_dict).
    """
    if fsdp_strategy == "fsdp":
        assert fsdp_version(model) == 1, (
            f"FSDP version 1 is expected for 'fsdp' strategy, but got {fsdp_version(model)}"
        )
    elif fsdp_strategy == "fsdp2":
        assert fsdp_version(model) == 2, (
            f"FSDP version 2 is expected for 'fsdp2' strategy, but got {fsdp_version(model)}"
        )
    else:
        raise ValueError(f"Unsupported FSDP strategy: {fsdp_strategy}")
    converter = FSDPConverter(fsdp_strategy, parameter_mapping)
    return converter.convert_state_and_sharding_dict(model)
