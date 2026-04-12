from psrl.utils.converter.model_mappings import ParameterMapping, register_model


# Megatron (all models)
# NOTE(lhy): the name transformation is done by mbridge
# That's why it is a class already been *bridged*, and it is not used for name transformation
@register_model(["Megatron"])
class BridgedMegatronParameterMapping(ParameterMapping):
    """Parameter mapping for Megatron model after mbridge."""

    def __init__(self, config):
        super().__init__(config)
        # NOTE(lhy): this is a hack to ensure the lm_head can be transformed separately
        # Otherwise we need to handle the complex logic of sharding weight for lm_head and embedding layer
        self.original_tie_word_embeddings = getattr(self.config, "tie_word_embeddings", False)

    def disable_tie_word_embeddings(self):
        self.config.tie_word_embeddings = False

    def get_mappings(self):
        raise ValueError(
            "BridgedMegatronParameterMapping is not used for name transformation, please use mbrige instead"
        )
