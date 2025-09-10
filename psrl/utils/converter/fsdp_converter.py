import torch
import warnings
from typing import Dict, Tuple
from collections import OrderedDict
from torch.distributed.tensor import DTensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy, StateDictType
from verl.utils.fsdp_utils import fsdp_version

from psrl.utils.nixl.nixl_spec import NIXLSharding
from psrl.utils.converter.base_converter import BaseConverter


class FSDPConverter(BaseConverter):
    """Converter for FSDP/FSDP2 model"""
    
    def __init__(self, fsdp_strategy):
        self.fsdp_strategy = fsdp_strategy

    def convert_state_and_sharding_dict(self, model) -> Tuple[Dict[str, torch.Tensor], Dict[str, NIXLSharding]]:
        """
        Convert FSDP/FSDP2 model to unified state dict and sharding info.
        Args:
            model: The FSDP/FSDP2 model instance
        Returns:
            (converted_state_dict, sharding_dict)
        """
        # Determine the FSDP strategy and convert accordingly
        # fsdp_state_dict will be (name, DTensor) pairs
        if self.fsdp_strategy == "fsdp":
            warnings.warn("FSDP strategy is deprecated beacause it cannot guarantee the in-place representation of the state dict.")
            with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
                fsdp_state_dict = model.state_dict()
        elif self.fsdp_strategy == "fsdp2":
            fsdp_state_dict = model.state_dict()
        else:
            raise ValueError(f"Unsupported FSDP strategy: {self.fsdp_strategy}")

        # Convert the FSDP state dict to a unified format
        converted_state_dict = {}
        sharding_dict = {}
        for param_name, param in fsdp_state_dict.items():
            assert isinstance(param, DTensor), f"Expected DTensor for {param_name}, got {type(param)}"
            converted_state_dict[param_name] = param.to_local()
            sharding_dict[param_name] = self.get_sharding_for_param(param_name, param)
        return converted_state_dict, sharding_dict
            
    def get_sharding_for_param(self, param_name: str, param: DTensor) -> NIXLSharding:
        """
        Generate sharding info for a parameter.
        Returns a NIXLSharding object.
        """
        assert len(param.placements) == 1 and param.placements[0].is_shard(0), \
            f"Expected single shard on dim 0 for {param_name}, got {param.placements}"
        assert param.device_mesh and param.device_mesh.ndim == 1, \
            f"Expected 1 dim device mesh for {param_name}, got {param.device_mesh}"
        kwargs = {
            "shard_mesh": OrderedDict([(0, param.device_mesh.size())]),
            "shard_indices": [(param.device_mesh.get_rank(),)]
        }
        return NIXLSharding(**kwargs)

def convert_fsdp_inplace(fsdp_strategy: str, model) -> Tuple[Dict[str, torch.Tensor], Dict[str, NIXLSharding]]:
    """
    Convenience function to convert FSDP/FSDP2 model to unified state dict and sharding info.
    Args:
        model: The FSDP/FSDP2 model instance
    Returns:
        (converted_state_dict, sharding_dict)
    """
    if fsdp_strategy == "fsdp":
        assert fsdp_version(model) == 1, "FSDP version 1 is expected for 'fsdp' strategy."
    elif fsdp_strategy == "fsdp2":
        assert fsdp_version(model) == 2, "FSDP version 2 is expected for 'fsdp2' strategy."
    else:
        raise ValueError(f"Unsupported FSDP strategy: {fsdp_strategy}")
    converter = FSDPConverter(fsdp_strategy)
    return converter.convert_state_and_sharding_dict(model)