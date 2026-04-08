PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "false",
        "NCCL_DEBUG": "VERSION",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "VLLM_DISABLE_COMPILE_CACHE": "1",  # NOTE: workaround for vllm compile cache issue, see https://github.com/vllm-project/vllm/issues/18851
        "VLLM_SKIP_P2P_CHECK": "1",  # Skip P2P check for init speedup in vLLM
        "VERL_DATAPROTO_SERIALIZATION_METHOD": "numpy",
        "PSRL_LOGGING_PATH": "",
        "PSRL_LOGGING_LEVEL": "INFO",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_CUMEM_ENABLE": "0",
    },
}
