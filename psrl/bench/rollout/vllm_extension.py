from vllm.v1.core.kv_cache_utils import estimate_max_model_len


class vLLMWorkerExtension:
    def estimate_max_model_len(self):
        assert hasattr(self, "available_kv_cache_memory_bytes"), "available_kv_cache_memory_bytes must be set"
        assert hasattr(self, "vllm_config"), "vllm_config must be set"
        kv_cache_spec = self.get_kv_cache_spec()
        assert kv_cache_spec is not None, "kv_cache_spec must not be None"
        # It use the binary search to estimate the max model length
        actual_max_model_len = self.vllm_config.model_config.max_model_len
        # Set the max model length to the upper limit of the estimation
        self.vllm_config.model_config.max_model_len = self.vllm_config.additional_config.get(
            "max_model_len_used_in_estimation",
            self.vllm_config.model_config.max_model_len * 8192,
        )
        estimated_max_model_len = estimate_max_model_len(
            self.vllm_config, kv_cache_spec, self.available_kv_cache_memory_bytes
        )
        # Restore the actual max model length
        self.vllm_config.model_config.max_model_len = actual_max_model_len
        return estimated_max_model_len
