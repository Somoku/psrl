from transformers import AutoConfig
from psrl.utils.state_dict.model_mappings import ParameterMapping, register_model
import warnings

# Qwen2
vllm_qwen2_classes = []
try:
    from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM, Qwen2Model
    vllm_qwen2_classes = [Qwen2ForCausalLM, Qwen2Model]
except ImportError as e:
    warnings.warn(f"Could not import Qwen2 classes: {e}")

@register_model(["VllmQwen2ForCausalLM", "VllmQwen2Model", "VllmQwen2ForSequenceClassification"] + vllm_qwen2_classes)
class VllmQwen2ParameterMapping(ParameterMapping):
    """Parameter mapping for Qwen2 model."""
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

# Llama
vllm_llama_classes = []
try:
    from vllm.model_executor.models.llama import LlamaForCausalLM, LlamaModel
    vllm_llama_classes = [LlamaForCausalLM, LlamaModel]
except ImportError as e:
    warnings.warn(f"Could not import Llama classes: {e}")

@register_model(["VllmLlamaForCausalLM", "VllmLlamaModel", "VllmLLaMAForCausalLM", "VllmInternLMForCausalLM", "VllmInternLM3ForCausalLM", "VllmXverseForCausalLM"] + vllm_llama_classes)
class VllmLlamaParameterMapping(ParameterMapping):
    """Parameter mapping for Llama model."""
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

# Mistral (implemented as LlamaForCausalLM in vLLM)
@register_model(["VllmMistralForCausalLM", "VllmMistralModel", *(vllm_llama_classes if vllm_llama_classes else [])])
class VllmMistralParameterMapping(ParameterMapping):
    """Parameter mapping for Mistral model (uses Llama implementation)."""
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

# Phi
vllm_phi_classes = []
try:
    from vllm.model_executor.models.phi import PhiForCausalLM
    vllm_phi_classes = [PhiForCausalLM]
except ImportError as e:
    warnings.warn(f"Could not import Phi classes: {e}")

@register_model(["VllmPhiForCausalLM"] + vllm_phi_classes)
class VllmPhiParameterMapping(ParameterMapping):
    """Parameter mapping for Phi model."""
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

# Gemma
vllm_gemma_classes = []
try:
    from vllm.model_executor.models.gemma import GemmaForCausalLM
    vllm_gemma_classes = [GemmaForCausalLM]
except ImportError as e:
    warnings.warn(f"Could not import Gemma classes: {e}")

@register_model(["VllmGemmaForCausalLM"] + vllm_gemma_classes)
class VllmGemmaParameterMapping(ParameterMapping):
    """Parameter mapping for Gemma model."""
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
