# vLLM to Unified (HuggingFace) State Dict Converter

This module provides functionality to convert vLLM model state dicts to unified (HuggingFace) format. The key feature is **inplace conversion**, which means no new memory is allocated, and parameter splitting is achieved through tensor views.

## Core Features

- **Inplace Conversion**: Uses tensor views instead of data copying to save memory
- **Modular Design**: Supports multiple models and is easy to extend
- **Auto Configuration**: Automatically retrieves model information from HuggingFace config

## Supported Models

Currently supports the following models:
- Qwen2
- Llama
- Mistral  
- Phi
- Gemma

## Usage

```python
from psrl.utils.state_dict import convert_vllm_to_hf_inplace, create_parameter_mapping_from_class

# 1. Get the model class
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
model_class = Qwen2ForCausalLM

# 2. Create parameter mapping
model_path = "/path/to/your/model"
param_mapping = create_parameter_mapping_from_class(model_class, model_path)

# 3. Get vLLM model state dict
vllm_state_dict = model.state_dict()

# 4. Convert to HuggingFace format
hf_state_dict = convert_vllm_to_hf_inplace(model_path, param_mapping, vllm_state_dict)
```

## Parameter Mapping

The converter primarily handles two types of parameter merging:

### 1. QKV Parameter Merging (QKVParallelLinear)
vLLM merges query, key, and value projections into a single parameter:
- `qkv_proj.weight` → `q_proj.weight`, `k_proj.weight`, `v_proj.weight`

### 2. Gate-Up Parameter Merging (MergedColumnParallelLinear)  
vLLM merges gate and up projections into a single parameter:
- `gate_up_proj.weight` → `gate_proj.weight`, `up_proj.weight`

## Adding New Model Support

To add support for a new model:

1. Create a parameter mapping class:

```python
from psrl.utils.state_dict import register_model, ParameterMapping

@register_model(YourModelClass)  # Replace with actual model class
class YourModelParameterMapping(ParameterMapping):
    def __init__(self, config_path: str):
        self.config = AutoConfig.from_pretrained(config_path)
    
    def get_mappings(self):
        return [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"), 
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
    
    def get_model_info(self):
        return {
            "num_heads": self.config.num_attention_heads,
            "num_kv_heads": getattr(self.config, 'num_key_value_heads', self.config.num_attention_heads),
            "head_size": self.config.hidden_size // self.config.num_attention_heads,
            "intermediate_size": self.config.intermediate_size,
        }
```

2. Use the new model:

```python
param_mapping = create_parameter_mapping_from_class(YourModelClass, model_path)
```

## Technical Details

### Inplace Conversion Implementation

The converter uses PyTorch's `tensor.narrow()` method to create tensor views instead of copying data:

```python
# Create views instead of copying
gate_view = fused.narrow(output_dim, 0, gate_size)
up_view = fused.narrow(output_dim, gate_size, up_size)
```

### Memory Efficiency

- Original parameters are not modified
- New parameters reference original data through views
- All parameters are marked as `is_sharded_weight: True`

### Auto Configuration

The converter automatically retrieves from HuggingFace config:
- `num_attention_heads`
- `num_key_value_heads` 
- `hidden_size`
- `intermediate_size`

## Important Notes

1. **Model Path**: Must point to a valid HuggingFace model path for config loading
2. **Memory Sharing**: Converted parameters share memory with original parameters; modifying one affects the other
3. **Model Compatibility**: Ensure vLLM model and HuggingFace model architectures are consistent