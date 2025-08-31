import torch
from abc import ABC, abstractmethod
from typing import Dict, Tuple

from psrl.utils.nixl.nixl_spec import NIXLSharding


class BaseConverter(ABC):
    """Base class for state dict and sharding converter"""
    
    @abstractmethod
    def convert_state_and_sharding_dict(self, model) -> Tuple[Dict[str, torch.Tensor], Dict[str, NIXLSharding]]:
        pass