from transformers import AutoConfig
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Type, Union
from torch.nn import Parameter


class ParameterMapping(ABC):
    """Abstract base class for parameter mappings."""
    
    @abstractmethod
    def get_mappings(self) -> List[Tuple[str, str, Any]]:
        """Return list of (vllm_param_name, hf_param_name, shard_id) mappings."""
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