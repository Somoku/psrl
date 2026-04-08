from dataclasses import dataclass, field

from psrl.utils.kv_cache.types import KVCacheBackend


@dataclass
class LMCacheConfig:
    """
    PSRL-level configuration for LMCache KV cache offloading.

    Read from `psrl.lmcache` in the Hydra YAML and translated into vLLM
    engine arguments at `PSRL_vLLMRollout` init time.

    Configuration pathway
    ---------------------
    Two mechanisms feed settings into LMCache:

    1. **vLLM engine kwargs** (`to_engine_kwargs()`):
       `kv_offloading_backend` and `kv_offloading_size` are consumed by vLLM's
       `VllmConfig._update_kv_transfer_from_offloading()`, which:
       - Sets `kv_transfer_config.kv_connector = "LMCacheConnectorV1"`
       - Computes `max_local_cpu_size = kv_offloading_size / num_kv_ranks` and
         injects it via `kv_connector_extra_config["lmcache.max_local_cpu_size"]`
       This ensures each TP rank gets the correct per-rank budget.
       Do NOT set `LMCACHE_MAX_LOCAL_CPU_SIZE` via env var — it would apply the
       total budget to every rank instead of the per-rank share.

    2. **LMCache environment variables** (`to_env_vars()`):
       All `LMCACHE_*` env vars are read by LMCache's config system before the
       vLLM extra config is applied. The extra config has higher priority and will
       override any conflicting env var values.
    """

    # --- Core ---

    # Whether to enable LMCache KV cache offloading.
    enable: bool = False

    # Offloading backend: "cpu", "disk", or "remote".
    backend: str = "cpu"

    # Total KV cache offloading buffer size in GiB (summed across all TP ranks).
    # vLLM divides this by the number of KV ranks (TP × PP) to obtain the
    # per-rank budget passed to LMCache as max_local_cpu_size / max_local_disk_size.
    offload_size_gb: float = 10.0

    # LMCache token chunk size (in tokens) for hash-based KV indexing.
    # Must be a multiple of the model's block size.
    chunk_size: int = 256

    # Path to a full LMCache YAML config file.
    # When set, this overrides all individual fields below (except chunk_size).
    config_file: str | None = None

    # Whether to clear the LMCache KV cache on model weight updates from PS.
    clear_on_weight_update: bool = True

    # --- CPU backend ---

    # GiB of CPU memory to reserve and never use for KV offloading.
    # Useful when other processes compete for pinned CPU memory.
    reserve_local_cpu_size: float = 0.0

    # --- Disk backend ---

    # Local filesystem path for disk-backed KV storage (required when backend="disk").
    disk_path: str | None = None

    # Maximum disk usage in GiB when using the disk backend.
    max_disk_size_gb: float = 50.0

    # --- Remote backend (Phase 2) ---

    # Remote LMCache server URL (e.g., "redis://host:6379") when backend="remote".
    remote_url: str | None = None

    # --- P2P transfer (for `kv_transfer_trajectory`) ---

    # Enable LMCache P2P backend + Controller for cross-instance KV transfer.
    enable_p2p: bool = False

    # Per-instance identifier passed to LMCache (must be unique per vLLM instance
    # in the cluster).
    lmcache_instance_id: str = "psrl_instance_0"

    # Transport channel for P2P transfer: "nixl" (RDMA/IB) or "tcp".
    p2p_transfer_channel: str = "nixl"

    # Base port for the LMCache Controller subprocess.
    # `find_available_port()` picks the actual port at runtime.
    controller_base_port: int = 9000

    # --- GPU pin budget ---

    # Maximum number of GPU KV blocks that PSRL may hold pinned simultaneously.
    # When this limit is exceeded, the oldest-pinned trajectory is unpinned
    # (PSRL-side LRU). 0 means no limit.
    gpu_pin_block_budget: int = 0

    # --- Cache behaviour ---

    # Whether to also cache KV entries produced during the decode phase.
    # Disabled by default — enabling increases memory usage but can benefit
    # multi-turn scenarios where the same decode prefix is reused.
    save_decode_cache: bool = False

    # Whether to persist a chunk even when it is not yet fully filled.
    # Useful when prompts are shorter than `chunk_size`.
    save_unfull_chunk: bool = False

    # Eviction policy for the local CPU/disk cache.
    # Supported values: "LRU" (least-recently-used) or "FIFO".
    cache_policy: str = "LRU"

    # Whether to retrieve KV cache entries asynchronously (overlapped with
    # prefill computation).  Can reduce effective TTFT for cache-hit requests.
    enable_async_loading: bool = False

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
            # LMCacheConnectorV1 does not support vLLM's Hybrid Memory Allocator (HMA).
            # HMA must be disabled whenever LMCache is used as the KV offloading backend.
            "disable_hybrid_kv_cache_manager": True,
        }

    def to_env_vars(self) -> dict[str, str]:
        """
        Translate this config into environment variables for LMCache.

        Must be called before vLLM engine initialization so that LMCache reads
        these via `LMCACHE_*` env vars during its config bootstrap.

        Note on max_local_cpu_size / max_local_disk_size
        -------------------------------------------------
        These are intentionally NOT set here.  vLLM's
        `VllmConfig._update_kv_transfer_from_offloading()` computes the correct
        *per-rank* value from `kv_offloading_size / num_kv_ranks` and injects it
        via `kv_connector_extra_config` (higher priority than env vars).  Setting
        the env var would apply the full budget to every rank, overcounting memory.

        Returns:
            dict[str, str]: Env var name to value pairs.
        """
        if not self.enable:
            return {}

        env_vars: dict[str, str] = {}

        # Activate LMCache v1 experimental config system.
        env_vars["LMCACHE_USE_EXPERIMENTAL"] = "True"

        # Chunk size (env var is read before vLLM extra config kicks in).
        env_vars["LMCACHE_CHUNK_SIZE"] = str(self.chunk_size)

        # Full config file override — takes precedence over all other env vars.
        if self.config_file:
            env_vars["LMCACHE_CONFIG_FILE"] = self.config_file

        # CPU backend.
        if self.backend == "cpu":
            env_vars["LMCACHE_LOCAL_CPU"] = "True"
            # reserve_local_cpu_size: keep some CPU memory free for other processes.
            if self.reserve_local_cpu_size > 0.0:
                env_vars["LMCACHE_RESERVE_LOCAL_CPU_SIZE"] = str(
                    self.reserve_local_cpu_size
                )

        # Disk backend.
        if self.backend == "disk" and self.disk_path:
            env_vars["LMCACHE_LOCAL_DISK"] = self.disk_path
            env_vars["LMCACHE_MAX_LOCAL_DISK_SIZE"] = str(self.max_disk_size_gb)

        # Remote backend (Phase 2).
        if self.backend == "remote" and self.remote_url:
            env_vars["LMCACHE_REMOTE_URL"] = self.remote_url

        # Cache behaviour flags.
        if self.save_decode_cache:
            env_vars["LMCACHE_SAVE_DECODE_CACHE"] = "True"
        if self.save_unfull_chunk:
            env_vars["LMCACHE_SAVE_UNFULL_CHUNK"] = "True"
        if self.cache_policy != "LRU":
            env_vars["LMCACHE_CACHE_POLICY"] = self.cache_policy
        if self.enable_async_loading:
            env_vars["LMCACHE_ENABLE_ASYNC_LOADING"] = "True"

        return env_vars
