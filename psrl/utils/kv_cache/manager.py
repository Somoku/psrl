import logging
import os

from psrl.utils.kv_cache.config import LMCacheConfig
from psrl.utils.kv_cache.types import KVCacheBackend, KVCacheStatus

psrl_logger = logging.getLogger(__file__)


class KVCacheManager:
    """
    KV cache manager for PSRL.

    Orchestrates LMCache configuration injection into vLLM and provides
    an abstraction layer for trajectory-level KV cache operations.

    Phase 1: Config injection + cache clear on weight update.
    Phase 2: Explicit evict/pin/transfer per trajectory via vLLM worker RPCs.
    """

    def __init__(self, config: LMCacheConfig):
        """
        Initialize the KV cache manager.

        Args:
            config (LMCacheConfig): The resolved LMCache configuration.
        """
        self._config = config

    @property
    def enabled(self) -> bool:
        """
        Whether LMCache offloading is enabled.
        """
        return self._config.enable

    @property
    def should_clear_on_weight_update(self) -> bool:
        """
        Whether to clear the LMCache KV cache on model weight updates.
        """
        return self._config.enable and self._config.clear_on_weight_update

    def get_status(self) -> KVCacheStatus:
        """
        Get the current KV cache offloading status.

        Returns:
            KVCacheStatus: Snapshot of the current offloading state.
        """
        if not self.enabled:
            return KVCacheStatus(enabled=False)

        return KVCacheStatus(
            enabled=True,
            backend=self._config.get_backend_enum(),
            offload_size_gb=self._config.offload_size_gb,
        )

    def apply_env_vars(self) -> None:
        """
        Set LMCache environment variables before vLLM engine initialization.

        Must be called before `AsyncEngineArgs` / `AsyncLLM` creation.
        """
        env_vars = self._config.to_env_vars()
        for key, value in env_vars.items():
            os.environ[key] = value
            psrl_logger.debug(f"Set {key}={value!r} for LMCache.")

    def get_engine_kwargs(self) -> dict:
        """
        Get vLLM engine kwargs for LMCache integration.

        Returns:
            dict: Key-value pairs to merge into vLLM engine arguments.
        """
        return self._config.to_engine_kwargs()

    # --- Phase 2 stubs: trajectory-level operations ---

    async def evict_trajectory(self, trajectory_id: str) -> bool:
        """
        Evict all KV cache entries for a specific trajectory.

        Phase 2: Will issue an RPC to vLLM worker extension to call
        `LMCacheEngine.clear()` with the token hash prefix for this trajectory.

        Args:
            trajectory_id (str): Unique identifier for the trajectory.

        Returns:
            bool: True if eviction was performed, False if not yet implemented.
        """
        # TODO(claude): Implement via vLLM worker extension RPC in Phase 2.
        psrl_logger.debug(
            f"evict_trajectory called for {trajectory_id!r}, "
            "but explicit eviction is not yet implemented."
        )
        return False

    async def pin_trajectory(self, trajectory_id: str) -> bool:
        """
        Pin a trajectory's KV cache to prevent LRU eviction.

        Args:
            trajectory_id (str): Unique identifier for the trajectory.

        Returns:
            bool: True if pinning was performed, False if not yet implemented.
        """
        # TODO(claude): Implement via LMCache cache policy extension in Phase 2.
        psrl_logger.debug(
            f"pin_trajectory called for {trajectory_id!r}, "
            "but explicit pinning is not yet implemented."
        )
        return False

    async def transfer_trajectory(
        self,
        trajectory_id: str,
        target_instance: str,
    ) -> bool:
        """
        Transfer a trajectory's KV cache to another GPU instance.

        Phase 2: Will use LMCache P2P or remote backend.

        Args:
            trajectory_id (str): Unique identifier for the trajectory.
            target_instance (str): Target instance identifier.

        Returns:
            bool: True if transfer was initiated, False if not yet implemented.
        """
        # TODO(claude): Implement via LMCache P2P or remote backend in Phase 2.
        psrl_logger.debug(
            f"transfer_trajectory called for {trajectory_id!r} -> {target_instance!r}, "
            "but cross-instance transfer is not yet implemented."
        )
        return False
