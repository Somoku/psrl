from transformers import AutoConfig

from psrl.utils.converter.model_mappings import ParameterMapping, register_model


# Megatron (all models)
# NOTE(lhy): the name transformation is done by mbridge
# That's why it is a class already been *bridged*, and it is not used for name transformation
@register_model(["Megatron"])
class BridgedMegatronParameterMapping(ParameterMapping):
    """Parameter mapping for Megatron model after mbridge."""

    def __init__(self, config_path: str):
        self.config = AutoConfig.from_pretrained(config_path)
        # NOTE(lhy): this is a hack to ensure the lm_head can be transformed separately
        # Otherwise we need to handle the complex logic of sharding weight for lm_head and embedding layer
        self.original_tie_word_embeddings = getattr(self.config, "tie_word_embeddings", False)

    def disable_tie_word_embeddings(self):
        self.config.tie_word_embeddings = False

    def get_mappings(self):
        raise ValueError(
            "BridgedMegatronParameterMapping is not used for name transformation, please use mbrige instead"
        )

    def get_model_info(self):
        # NOTE(zym): Some models such as qwen3moe directly provide head_dim,
        #  which isn't equal to hidden_size // num_attention_heads
        return {
            "num_heads": self.config.num_attention_heads,
            "num_kv_heads": getattr(self.config, "num_key_value_heads", self.config.num_attention_heads),
            "head_size": getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads),
            "intermediate_size": self.config.intermediate_size,
            "moe_intermediate_size": getattr(self.config, "moe_intermediate_size", self.config.intermediate_size),
            "shared_expert_intermediate_size": getattr(self.config, "shared_expert_intermediate_size", None),
        }
