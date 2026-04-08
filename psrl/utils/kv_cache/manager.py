import asyncio
import logging
import subprocess
import time
from collections import deque

from psrl.utils.common.http_utils import find_available_port, get_host_info, post
from psrl.utils.kv_cache.config import LMCacheConfig
from psrl.utils.kv_cache.types import KVCacheBackend, KVCacheStatus, TrajectoryCacheInfo

psrl_logger = logging.getLogger(__file__)


class KVCacheManager:
    """
    KV cache manager for PSRL.

    Stateless with respect to trajectory identity — all public methods accept
    `tokens: list[int]` directly.  Trajectory-to-token mapping is maintained
    by `RolloutRouter`.

    Responsibilities:
    - Orchestrate KV cache queries and operations via `collective_rpc`.
    - Enforce a configurable GPU pin block budget (PSRL-side LRU eviction).
    - Manage the LMCache Controller subprocess lifecycle for cross-instance
      P2P transfer.
    """

    def __init__(self, config: LMCacheConfig) -> None:
        """
        Initialize the KV cache manager.

        Args:
            config (LMCacheConfig): The resolved LMCache configuration.
        """
        self._config = config
        self._gpu_pin_budget: int = config.gpu_pin_block_budget
        self._pinned_gpu_blocks: int = 0
        self._gpu_pinned_order: deque[list[int]] = deque()

        self._inference_engine = None
        self._controller_proc: subprocess.Popen | None = None
        self._controller_url: str | None = None

        self._log_init_status()

    # --- Initialization helpers ---

    def _log_init_status(self) -> None:
        """
        Log LMCache initialization status and parameters at INFO level.
        """
        if not self._config.enable:
            psrl_logger.info("[LMCache] KV cache offloading is DISABLED.")
            return

        psrl_logger.info(
            "[LMCache] KV cache offloading is ENABLED with the following parameters:"
        )
        psrl_logger.info(f"  backend                = {self._config.backend!r}")
        psrl_logger.info(f"  offload_size_gb        = {self._config.offload_size_gb}")
        psrl_logger.info(f"  chunk_size             = {self._config.chunk_size}")
        psrl_logger.info(f"  cache_policy           = {self._config.cache_policy!r}")
        psrl_logger.info(f"  save_decode_cache      = {self._config.save_decode_cache}")
        psrl_logger.info(f"  save_unfull_chunk      = {self._config.save_unfull_chunk}")
        psrl_logger.info(
            f"  enable_async_loading   = {self._config.enable_async_loading}"
        )
        psrl_logger.info(
            f"  clear_on_weight_update = {self._config.clear_on_weight_update}"
        )
        psrl_logger.info(
            f"  gpu_pin_block_budget   = {self._config.gpu_pin_block_budget}"
        )
        if self._config.enable_p2p:
            psrl_logger.info(
                f"  enable_p2p             = {self._config.enable_p2p}"
            )
            psrl_logger.info(
                f"  lmcache_instance_id    = {self._config.lmcache_instance_id!r}"
            )
        if self._config.config_file:
            psrl_logger.info(f"  config_file            = {self._config.config_file!r}")
        self._verify_lmcache_importable()

    def _verify_lmcache_importable(self) -> None:
        """
        Verify that the lmcache package is importable and log its version.
        """
        try:
            import lmcache  # type: ignore[import-untyped]

            version = getattr(lmcache, "__version__", "unknown")
            psrl_logger.info(
                f"[LMCache] lmcache package is importable, version={version!r}."
            )
        except ImportError:
            psrl_logger.error(
                "[LMCache] lmcache package is NOT importable! "
                "KV cache offloading will NOT work. "
                "Run `bash scripts/install_lmcache.sh` to install it."
            )

    # --- Engine attachment ---

    def attach_engine(self, inference_engine) -> None:
        """
        Attach the vLLM AsyncLLM engine after it has been initialised.

        Must be called before any async KV cache operation.

        Args:
            inference_engine: The `AsyncLLM` (or compatible) engine object whose
                `collective_rpc` method dispatches to vLLM worker processes.
        """
        self._inference_engine = inference_engine
        psrl_logger.info("[LMCache] KVCacheManager: Engine attached.")

    @property
    def is_attached(self) -> bool:
        """Whether the inference engine has been attached via `attach_engine`."""
        return self._inference_engine is not None

    # --- Legacy Phase 1 helpers (still used for engine init) ---

    @property
    def enabled(self) -> bool:
        """Whether LMCache offloading is enabled."""
        return self._config.enable

    @property
    def should_clear_on_weight_update(self) -> bool:
        """Whether to clear the LMCache KV cache on model weight updates from PS."""
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
        import os

        env_vars = self._config.to_env_vars()
        if not env_vars:
            psrl_logger.info(
                "[LMCache] No environment variables to set (LMCache disabled)."
            )
            return
        psrl_logger.info("[LMCache] Setting environment variables for LMCache...")
        for key, value in env_vars.items():
            os.environ[key] = value
            psrl_logger.info(f"  {key}={value!r}")

    def get_engine_kwargs(self) -> dict:
        """
        Get vLLM engine kwargs for LMCache integration.

        Returns:
            dict: Key-value pairs to merge into vLLM engine arguments.
        """
        kwargs = self._config.to_engine_kwargs()
        if kwargs:
            psrl_logger.info(f"[LMCache] Injecting engine kwargs into vLLM: {kwargs}.")
        else:
            psrl_logger.info(
                "[LMCache] No engine kwargs to inject (LMCache disabled)."
            )
        return kwargs

    # --- Private helpers ---

    def _assert_engine(self) -> None:
        assert self._inference_engine is not None, (
            "KVCacheManager inference engine is not attached. "
            "Call attach_engine() after the rollout is initialised."
        )

    async def _rpc(self, method: str, args: tuple) -> object:
        """
        Call a `lmcache_*` method on all vLLM workers via `collective_rpc`.

        `collective_rpc` returns a list with one result per TP rank. We take
        the first element (rank-0), which is canonical for aggregate results.

        Args:
            method (str): The `vllm_extension.py` method name to call.
            args (tuple): Positional arguments forwarded to the worker method.

        Returns:
            object: The result from rank-0 worker.
        """
        results = await self._inference_engine.collective_rpc(method, args=args)
        # `collective_rpc` returns a list[result]; take rank-0 value.
        return results[0] if isinstance(results, list) else results

    # --- Public KV cache operations ---

    async def get_cache_info(self, tokens: list[int]) -> TrajectoryCacheInfo:
        """
        Query GPU prefix-cache and LMCache backend usage for a token sequence.

        Operations act only on the cached prefix — the longest contiguous run
        of chunks/blocks that exist in the target store.

        Args:
            tokens (list[int]): Full token sequence for the trajectory.

        Returns:
            TrajectoryCacheInfo: Snapshot of cache usage for this token sequence.
        """
        self._assert_engine()
        assert tokens, "tokens must be a non-empty list."
        raw: dict = await self._rpc("lmcache_get_cache_info", (tokens,))
        return TrajectoryCacheInfo(**raw)

    async def pin(self, tokens: list[int], targets: list[str]) -> bool:
        """
        Pin the cached prefix of a trajectory to prevent LRU eviction.

        Supported targets: `"gpu"` (vLLM block pool) and `"backend"` (LMCache).
        GPU pinning is subject to `gpu_pin_block_budget`; if the budget is
        exceeded, the oldest-pinned entry is unpinned first (PSRL-side LRU).

        Args:
            tokens (list[int]): Full token sequence for the trajectory.
            targets (list[str]): Subset of `["gpu", "backend"]`.

        Returns:
            bool: True if all requested pin operations succeeded.
        """
        self._assert_engine()
        assert tokens, "tokens must be a non-empty list."
        assert targets, "targets must be a non-empty list."
        assert all(t in ("gpu", "backend") for t in targets), (
            f"Invalid pin targets: {targets!r}. Must be a subset of ['gpu', 'backend']."
        )
        ok = True
        if "gpu" in targets:
            ok = ok and await self._pin_gpu(tokens)
        if "backend" in targets:
            pinned: int = await self._rpc("lmcache_pin_backend", (tokens,))
            ok = ok and pinned >= 0
        return ok

    async def unpin(self, tokens: list[int], targets: list[str]) -> bool:
        """
        Unpin the cached prefix of a trajectory, allowing LRU eviction.

        Args:
            tokens (list[int]): Full token sequence for the trajectory.
            targets (list[str]): Subset of `["gpu", "backend"]`.

        Returns:
            bool: True if all unpin operations completed without error.
        """
        self._assert_engine()
        assert tokens, "tokens must be a non-empty list."
        assert targets, "targets must be a non-empty list."
        assert all(t in ("gpu", "backend") for t in targets), (
            f"Invalid unpin targets: {targets!r}. Must be a subset of ['gpu', 'backend']."
        )
        ok = True
        if "gpu" in targets:
            ok = ok and await self._unpin_gpu(tokens)
        if "backend" in targets:
            await self._rpc("lmcache_unpin_backend", (tokens,))
        return ok

    async def clear_from_backend(self, tokens: list[int]) -> int:
        """
        Remove the cached prefix chunks for a trajectory from the LMCache backend.

        Args:
            tokens (list[int]): Full token sequence for the trajectory.

        Returns:
            int: Number of LMCache chunks removed.
        """
        self._assert_engine()
        assert tokens, "tokens must be a non-empty list."
        n: int = await self._rpc("lmcache_clear_from_backend", (tokens,))
        psrl_logger.debug(f"[LMCache] Cleared {n} chunks from backend.")
        return n

    async def transfer(
        self,
        tokens: list[int],
        src: tuple[str, str],
        dst: tuple[str, str],
        copy: bool = False,
    ) -> bool:
        """
        Transfer the cached prefix of a trajectory to another (instance, backend).

        Uses the LMCache Controller HTTP API (`/move`). The Controller subprocess
        is started on demand if not already running.  Both intra-machine (same host,
        different GPU) and inter-machine (via NIXL RDMA or TCP) transfers are
        supported through the same interface.

        Args:
            tokens (list[int]): Full token sequence for the trajectory.
            src (tuple[str, str]): `(lmcache_instance_id, backend_location)` of
                the source, e.g. `("psrl_instance_0", "LocalCPUBackend")`.
            dst (tuple[str, str]): `(lmcache_instance_id, backend_location)` of
                the destination.
            copy (bool): If True, keep the data at `src` as well.

        Returns:
            bool: True if the Controller acknowledged the move request.
        """
        self._assert_engine()
        assert tokens, "tokens must be a non-empty list."
        if not self._config.enable_p2p:
            psrl_logger.warning(
                "[LMCache] transfer() called but enable_p2p is False. "
                "Set LMCacheConfig.enable_p2p=True to enable cross-instance transfer."
            )
            return False

        controller_url = await asyncio.get_event_loop().run_in_executor(
            None, self._ensure_controller_running
        )

        payload = {
            "old_position": list(src),
            "new_position": list(dst),
            "tokens": tokens,
            "copy": copy,
        }
        try:
            resp = await post(f"{controller_url}/move", payload)
            psrl_logger.debug(f"[LMCache] Controller move ACK: {resp!r}.")
            return True
        except Exception as e:
            psrl_logger.error(f"[LMCache] Controller /move request failed: {e}.")
            return False

    # --- GPU pin budget internals ---

    async def _pin_gpu(self, tokens: list[int]) -> bool:
        """
        Pin GPU prefix-cache blocks for `tokens`, enforcing the budget.

        If `_gpu_pin_budget > 0` and the new blocks would exceed the budget,
        the oldest-pinned token sequence is unpinned first (PSRL-side LRU).
        If a single trajectory exceeds the entire budget after eviction, the
        pin is skipped and a warning is logged.

        Args:
            tokens (list[int]): Full token sequence for the trajectory.

        Returns:
            bool: True if the pin succeeded, False if budget cannot accommodate.
        """
        # Determine how many blocks this trajectory uses.
        info = await self.get_cache_info(tokens)
        new_blocks = info.gpu_cached_blocks

        if self._gpu_pin_budget > 0:
            # Evict oldest-pinned trajectories to make room.
            while (
                self._pinned_gpu_blocks + new_blocks > self._gpu_pin_budget
                and self._gpu_pinned_order
            ):
                oldest_tokens = self._gpu_pinned_order.popleft()
                # Use the actual freed count returned by lmcache_unpin_gpu rather
                # than re-querying get_cache_info.  The cache-info query was
                # intended to handle vLLM reclaiming blocks, but PSRL-pinned
                # blocks have ref_cnt > 0 and cannot be reclaimed by vLLM, so
                # the query is both unnecessary and misleading: it counts *all*
                # GPU-cached blocks for the trajectory (including blocks added
                # during inference after the original pin), which causes
                # _pinned_gpu_blocks to be over-decremented and the budget
                # constraint to become ineffective.
                freed: int = await self._rpc("lmcache_unpin_gpu", (oldest_tokens,))
                self._pinned_gpu_blocks = max(0, self._pinned_gpu_blocks - freed)
                psrl_logger.debug(
                    f"[LMCache] GPU pin budget: evicted oldest trajectory "
                    f"({freed} blocks freed, budget={self._gpu_pin_budget})."
                )

            # After eviction, if the trajectory still exceeds the budget, skip the pin.
            if self._pinned_gpu_blocks + new_blocks > self._gpu_pin_budget:
                psrl_logger.warning(
                    f"[LMCache] GPU pin budget exceeded: trajectory needs {new_blocks} blocks "
                    f"but only {self._gpu_pin_budget - self._pinned_gpu_blocks} available "
                    f"(budget={self._gpu_pin_budget}). Skipping GPU pin."
                )
                return False

        pinned: int = await self._rpc("lmcache_pin_gpu", (tokens,))
        self._pinned_gpu_blocks += pinned
        self._gpu_pinned_order.append(tokens)
        psrl_logger.debug(
            f"[LMCache] GPU pin: {pinned} blocks pinned, "
            f"total={self._pinned_gpu_blocks}, budget={self._gpu_pin_budget}."
        )
        return True

    async def _unpin_gpu(self, tokens: list[int]) -> bool:
        """
        Unpin GPU prefix-cache blocks for `tokens`.

        Args:
            tokens (list[int]): Full token sequence for the trajectory.

        Returns:
            bool: True if the unpin succeeded.
        """
        freed: int = await self._rpc("lmcache_unpin_gpu", (tokens,))
        self._pinned_gpu_blocks = max(0, self._pinned_gpu_blocks - freed)
        # Remove from order tracking (`deque` does not support arbitrary removal,
        # so rebuild without the evicted entry).
        self._gpu_pinned_order = deque(
            t for t in self._gpu_pinned_order if t != tokens
        )
        psrl_logger.debug(
            f"[LMCache] GPU unpin: {freed} blocks freed, "
            f"total={self._pinned_gpu_blocks}."
        )
        return True

    # --- Controller lifecycle ---

    def _ensure_controller_running(self) -> str:
        """
        Start the LMCache controller subprocess if not already running.

        Polls `GET /health` up to 30 times (1 s interval) before raising.

        Returns:
            str: Base URL of the controller, e.g. `"http://127.0.0.1:9042"`.

        Raises:
            RuntimeError: If the controller does not become healthy within 30 s.
        """
        if self._controller_proc is not None and self._controller_url is not None:
            return self._controller_url

        import requests as _requests

        _, host = get_host_info()
        port = find_available_port(self._config.controller_base_port)
        cmd = [
            "lmcache_controller",
            "--host", host,
            "--port", str(port),
        ]
        psrl_logger.info(f"[LMCache] Starting controller: {' '.join(cmd)}.")
        self._controller_proc = subprocess.Popen(cmd)
        self._controller_url = f"http://{host}:{port}"

        # Poll until healthy.
        health_url = f"{self._controller_url}/health"
        for attempt in range(30):
            try:
                resp = _requests.get(health_url, timeout=1)
                if resp.status_code == 200:
                    psrl_logger.info(
                        f"[LMCache] Controller healthy at {self._controller_url} "
                        f"(attempt {attempt + 1})."
                    )
                    return self._controller_url
            except Exception:
                pass
            time.sleep(1)

        self._controller_proc.kill()
        self._controller_proc = None
        self._controller_url = None
        raise RuntimeError(
            f"LMCache controller did not become healthy at {health_url} "
            "within 30 s."
        )

    def __del__(self) -> None:
        if self._controller_proc is not None:
            try:
                self._controller_proc.terminate()
                self._controller_proc.wait(timeout=5)
            except Exception:
                pass
