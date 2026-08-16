import os

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env as get_verl_ppo_ray_runtime_env

PSRL_RAY_ENV_DEFAULTS = {
    "TOKENIZERS_PARALLELISM": "false",
    "NCCL_DEBUG": "VERSION",
    "VLLM_LOGGING_LEVEL": "WARN",
    "VLLM_SKIP_P2P_CHECK": "1",  # Skip P2P check for init speedup in vLLM
    "VERL_DATAPROTO_SERIALIZATION_METHOD": "numpy",
    "PSRL_LOGGING_LEVEL": "INFO",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_CUMEM_ENABLE": "0",
}


def get_ppo_ray_runtime_env(config=None):
    """Build the veRL runtime environment with PSRL-specific defaults.

    veRL owns platform and engine-sensitive policy, including ROCm visibility,
    Blackwell workarounds, determinism forwarding, and the conditional
    ``CUDA_DEVICE_MAX_CONNECTIONS`` setting. PSRL only overlays defaults needed
    by its SMG/NIXL data path. An environment value supplied by the launcher
    always takes precedence.

    Args:
        config: Optional resolved or unresolved training configuration.

    Returns:
        dict: Ray runtime environment suitable for ``ray.init``.
    """
    runtime_env = get_verl_ppo_ray_runtime_env(config)
    runtime_env.setdefault("env_vars", {})

    for key, value in PSRL_RAY_ENV_DEFAULTS.items():
        if os.environ.get(key) is None:
            runtime_env["env_vars"][key] = value

    for key in ("PYTHONPATH", "PYTHONPYCACHEPREFIX"):
        val = os.environ.get(key)
        if val:
            runtime_env["env_vars"][key] = val

    return runtime_env
