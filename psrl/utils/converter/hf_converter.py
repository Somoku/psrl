import torch

from psrl.utils.converter.base_converter import BaseConverter
from psrl.utils.nixl.nixl_spec import NIXLSharding


class HFConverter(BaseConverter):
    """Converter for HuggingFace model"""

    def __init__(self):
        pass

    def convert_state_and_sharding_dict(self, model) -> tuple[dict[str, torch.Tensor], dict[str, NIXLSharding]]:
        """
        Convert HuggingFace model to unified state dict and sharding info.
        Args:
            model: The HuggingFace model instance
        Returns:
            (converted_state_dict, sharding_dict)
        """
        # Extract the state dict from the model
        hf_state_dict = model.state_dict()

        # Convert the state dict to a unified format
        converted_state_dict = {}
        sharding_dict = {}
        for param_name, param in hf_state_dict.items():
            assert isinstance(param, torch.Tensor), f"Expected Tensor for {param_name}, got {type(param)}"
            converted_state_dict[param_name] = param
            sharding_dict[param_name] = NIXLSharding.default()
        return converted_state_dict, sharding_dict


def convert_hf_inplace(
    model,
) -> tuple[dict[str, torch.Tensor], dict[str, NIXLSharding]]:
    """
    Convenience function to convert HuggingFace model to unified state dict and sharding info.
    Args:
        model: The HuggingFace model instance
    Returns:
        (converted_state_dict, sharding_dict)
    """
    converter = HFConverter()
    return converter.convert_state_and_sharding_dict(model)
