from abc import ABC, abstractmethod

import torch

from psrl.utils.nixl.nixl_spec import NIXLSharding


class BaseConverter(ABC):
    """Base class for state dict and sharding converter"""

    @abstractmethod
    def convert_state_and_sharding_dict(self, model) -> tuple[dict[str, torch.Tensor], dict[str, NIXLSharding]]:
        pass
