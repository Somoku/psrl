from dataclasses import dataclass

from psrl.utils.kv_cache.types import KVCacheBackend


@dataclass
class LMCacheConfig:
    """
    PSRL-level configuration for LMCache KV cache offloading.

    Read from `psrl.lmcache` in the Hydra YAML and translated into vLLM
    engine arguments at `PSRL_vLLMRollout` init time.
    """

    # Whether to enable LMCache KV cache offloading.
    enable: bool = False

    # Offloading backend: "cpu", "disk", or "remote".
    backend: str = "cpu"

    # Total KV cache offloading buffer size in GiB (summed across all TP ranks).
    offload_size_gb: float = 10.0

    # LMCache token chunk size for hashing.
    chunk_size: int = 256

    # Path to a full LMCache YAML config file (overrides individual settings).
    config_file: str | None = None

    # Disk backend settings.
    disk_path: str | None = None
    max_disk_size_gb: float = 50.0

    # Remote backend settings (Phase 2).
    remote_url: str | None = None

    # Whether to clear the LMCache KV cache on model weight updates.
    clear_on_weight_update: bool = True

    def get_backend_enum(self) -> KVCacheBackend:
        """
        Convert the string backend to `KVCacheBackend` enum.
        """
        return KVCacheBackend(self.backend)

    def to_engine_kwargs(self) -> dict:
        """
        Translate this config into vLLM engine kwargs for `AsyncEngineArgs`.

        Returns:
            dict: Key-value pairs to merge into `llm_kwargs`.
        """
        if not self.enable:
            return {}

        return {
            "kv_offloading_backend": "lmcache",
            "kv_offloading_size": self.offload_size_gb,
        }

    def to_env_vars(self) -> dict[str, str]:
        """
        Translate this config into environment variables for LMCache.

        Must be called before vLLM engine initialization so that LMCache
        reads these via `LMCACHE_*` env vars.

        Returns:
            dict[str, str]: Env var name to value pairs.
        """
        if not self.enable:
            return {}

        env_vars: dict[str, str] = {}
        env_vars["LMCACHE_CHUNK_SIZE"] = str(self.chunk_size)

        if self.config_file:
            env_vars["LMCACHE_CONFIG_FILE"] = self.config_file

        if self.backend == "disk" and self.disk_path:
            env_vars["LMCACHE_LOCAL_DISK"] = self.disk_path
            env_vars["LMCACHE_MAX_LOCAL_DISK_SIZE"] = str(self.max_disk_size_gb)

        if self.backend == "remote" and self.remote_url:
            env_vars["LMCACHE_REMOTE_URL"] = self.remote_url

        return env_vars
