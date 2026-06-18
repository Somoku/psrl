import asyncio
import logging
import uuid
from collections import deque

import msgspec

from psrl.utils.common.http_utils import post
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
        # NOTE(lhy): Controller URL is no longer started locally per instance.
        # It is started once by RolloutCoordinator and broadcast to all GenWorkers
        # via set_controller_url(). Until that call, transfer() will assert-fail.
        self._controller_url: str | None = None

        # Direct transfer bypass: peer registry maps lmcache_instance_id → per-rank
        # peer_init_url list, indexed by global rank (list[rank] = that rank's NIXL
        # endpoint). Populated by set_peer_registry() after P2P init. When available,
        # transfer_direct() sends MoveWorkerMsg directly to the local LMCacheWorker via
        # ZMQ, bypassing the Controller HTTP round-trip. KV is sharded per rank (TP heads,
        # PP layers), so each source rank must target the destination's same-rank endpoint.
        self._peer_registry: dict[str, list[str]] = {}
        # This worker's own global rank within its rollout instance, used to index the
        # destination's per-rank peer_init_url list for same-rank pairing. Set in
        # set_peer_registry() (constant after init).
        self._my_rank: int = 0
        # Worker ZMQ URL for direct command dispatch (ip:port of local LMCacheWorker REP socket).
        self._worker_zmq_url: str | None = None
        # Async ZMQ socket for direct transfer (created lazily on first use).
        self._direct_zmq_socket = None
        self._direct_zmq_context = None
        # Lock to serialize ZMQ REQ send/recv pairs (REQ pattern requires strict alternation).
        self._direct_zmq_lock = asyncio.Lock()

        # Push-based GPU prefix cache snapshot (updated by GenWorker background task).
        # The EngineCore pushes the hash set to a queue every ~100ms; the GenWorker
        # consumes it and stores it here. This allows the router to query GPU prefix
        # cache hit counts via a fast GenWorker RPC (no EngineCore blocking).
        self._gpu_cache_hash_set: set | None = None
        self._gpu_cache_block_size: int = 128
        self._gpu_cache_total_blocks: int = 0

        # Push-based LMCache backend snapshot (updated by GenWorker background task).
        # Stores the set of chunk_hash values from hot_cache.keys().
        self._lmcache_chunk_hash_set: set | None = None
        self._lmcache_chunk_size: int = 256
        self._lmcache_total_bytes: int = 0
        self._lmcache_chunk_bytes: int = 0

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

    def set_instance_id(self, instance_id: int) -> None:
        """
        Set the per-instance LMCache identifier.

        Must be called once by `PSRL_GenWorker` after construction, passing the
        numeric instance id assigned to this worker group.  Sets
        `lmcache_instance_id` to `"psrl_instance_{instance_id}"`, which the
        LMCache Controller uses to identify KV transfer sources and destinations.

        Args:
            instance_id (int): Numeric identifier of this rollout instance.
        """
        self._config.lmcache_instance_id = f"psrl_instance_{instance_id}"
        psrl_logger.info(
            f"[LMCache] Instance ID set to {self._config.lmcache_instance_id!r}."
        )

    def set_controller_url(self, controller_url: str) -> None:
        """
        Set the shared LMCache Controller URL.

        Called by `PSRL_GenWorker.set_lmcache_controller_url()` after
        `RolloutCoordinator.init_lmcache_p2p()` broadcasts the URL of the
        single shared Controller subprocess to all GenWorker instances.

        Must be called before `transfer()` is used.

        Args:
            controller_url (str): Base URL of the shared Controller, e.g.
                `"http://10.0.0.1:9042"`.
        """
        self._controller_url = controller_url
        psrl_logger.info(
            f"[LMCache] Controller URL set to {self._controller_url!r}."
        )

    def set_peer_registry(
        self,
        registry: dict[str, list[str]],
        worker_zmq_url: str | None = None,
        my_rank: int = 0,
    ) -> None:
        """
        Set the peer registry and local worker ZMQ URL for direct transfer bypass.

        Maps each LMCache instance_id to its per-rank list of peer_init_url (NIXL
        endpoints), indexed by global rank. When populated along with `worker_zmq_url`,
        `transfer_direct()` sends MoveWorkerMsg directly to the local LMCacheWorker via
        ZMQ, bypassing the Controller HTTP round-trip entirely. Because KV is sharded
        per rank (TP heads, PP layers), each source rank targets the destination's
        same-rank endpoint via `my_rank`.

        Called by `PSRL_GenWorker.kv_set_peer_registry()` after
        `RolloutCoordinator._broadcast_peer_registry()` completes.

        Args:
            registry (dict[str, list[str]]): Maps lmcache_instance_id (e.g.
                "psrl_instance_0") to a rank-sorted list of peer_init_url
                (e.g. ["10.0.0.1:18200", "10.0.0.1:18201"]).
            worker_zmq_url (str | None): ZMQ REP URL of the local LMCacheWorker
                (e.g. "10.0.0.1:18100"). If None, direct transfer falls back to
                Controller HTTP path.
            my_rank (int): This worker's own global rank within its rollout instance,
                used to index the destination's per-rank endpoint list.
        """
        self._peer_registry = registry
        self._worker_zmq_url = worker_zmq_url
        self._my_rank = my_rank
        # Reset ZMQ socket so it reconnects with new URL on next use.
        if self._direct_zmq_socket is not None:
            try:
                self._direct_zmq_socket.close(linger=0)
            except Exception:
                pass
            self._direct_zmq_socket = None
        psrl_logger.info(
            f"[LMCache] Peer registry set with {len(registry)} entries, "
            f"worker_zmq_url={worker_zmq_url!r}, my_rank={my_rank}."
        )

    # --- Push-based GPU prefix cache snapshot methods ---

    def update_gpu_cache_snapshot(self, hash_snapshot: dict) -> None:
        """
        Update the local GPU prefix cache hash set from a pushed EngineCore snapshot.

        Called by the GenWorker background task that consumes the
        `kv_cache_hash_queue`. After this call, `get_gpu_cache_info_local(tokens)`
        can compute prefix-cache hits locally without any EngineCore RPC.

        Args:
            hash_snapshot (dict): Must contain keys `"hash_set"` (set of
                `BlockHashWithGroupId`), `"block_size"` (int), and
                `"total_blocks"` (int).
        """
        self._gpu_cache_hash_set = hash_snapshot["hash_set"]
        self._gpu_cache_block_size = hash_snapshot["block_size"]
        self._gpu_cache_total_blocks = hash_snapshot["total_blocks"]

    def get_gpu_cache_info_local(self, tokens: list[int]) -> dict:
        """
        Compute GPU prefix cache hit for `tokens` using the locally-cached hash set.

        Walks the hash chain (same algorithm as `RolloutScheduler._psrl_iter_gpu_prefix_blocks`)
        but checks against the periodically-pushed hash snapshot stored in this manager
        rather than accessing `block_pool` in the EngineCore process.

        No EngineCore RPC — runs entirely in the GenWorker process.

        Args:
            tokens (list[int]): Full token sequence for the trajectory.

        Returns:
            dict: Dict with keys `gpu_cached_blocks`, `gpu_cached_tokens`,
                `gpu_total_blocks`, `gpu_usage_pct`.
        """
        if self._gpu_cache_hash_set is None or not tokens:
            return {
                "gpu_cached_blocks": 0,
                "gpu_cached_tokens": 0,
                "gpu_total_blocks": self._gpu_cache_total_blocks,
                "gpu_usage_pct": 0.0,
            }

        from vllm.v1.core.kv_cache_utils import (
            hash_block_tokens,
            init_none_hash,
            make_block_hash_with_group_id,
        )

        hash_fn = self._get_caching_hash_fn()
        init_none_hash(hash_fn)

        prev_hash = None
        cached_blocks = 0
        block_size = self._gpu_cache_block_size
        num_full_blocks = len(tokens) // block_size
        for block_idx in range(num_full_blocks):
            start = block_idx * block_size
            end = start + block_size
            chunk = tokens[start:end]
            block_hash = hash_block_tokens(hash_fn, prev_hash, chunk, None)
            prev_hash = block_hash
            key = make_block_hash_with_group_id(block_hash, 0)
            if key not in self._gpu_cache_hash_set:
                break
            cached_blocks += 1

        gpu_cached_tokens = cached_blocks * block_size
        total = self._gpu_cache_total_blocks
        return {
            "gpu_cached_blocks": cached_blocks,
            "gpu_cached_tokens": gpu_cached_tokens,
            "gpu_total_blocks": total,
            "gpu_usage_pct": cached_blocks / total if total > 0 else 0.0,
        }

    def _get_caching_hash_fn(self):
        """
        Cache and return the token-hashing function used by the block pool.

        Reads prefix_caching_hash_algo from the attached engine's vllm_config
        to ensure consistency with the EngineCore's hash chain.
        """
        if not hasattr(self, "_caching_hash_fn"):
            from vllm.utils.hashing import get_hash_fn_by_name

            hash_algo = self._inference_engine.vllm_config.cache_config.prefix_caching_hash_algo
            self._caching_hash_fn = get_hash_fn_by_name(hash_algo)
        return self._caching_hash_fn

    # --- Push-based LMCache backend snapshot methods ---

    def update_lmcache_backend_snapshot(self, snapshot: dict) -> None:
        """
        Update the local LMCache backend hash set from a pushed Worker snapshot.

        Called by the GenWorker background task that consumes the
        `kv_cache_hash_queue`. After this call, `get_lmcache_cache_info_local()`
        can check LMCache backend hits locally without any collective_rpc.

        Args:
            snapshot (dict): Must contain keys `"chunk_hash_set"` (set of int),
                `"chunk_size"` (int), `"total_bytes"` (int), `"chunk_bytes"` (int),
                and `"none_hash"` (bytes).
        """
        self._lmcache_chunk_hash_set = snapshot["chunk_hash_set"]
        # msgspec deserializes set→list through zmq; convert back for O(1) lookup.
        if not isinstance(self._lmcache_chunk_hash_set, set):
            self._lmcache_chunk_hash_set = set(self._lmcache_chunk_hash_set)
        self._lmcache_chunk_size = snapshot["chunk_size"]
        self._lmcache_total_bytes = snapshot["total_bytes"]
        self._lmcache_chunk_bytes = snapshot["chunk_bytes"]
        # NONE_HASH from the EngineCore process (serializable via zmq).
        if "none_hash" in snapshot:
            self._lmcache_none_hash = snapshot["none_hash"]
        # Hash algorithm name for loading the correct hash function.
        # LMCache uses "builtin" by default, vLLM uses "sha256" — they differ!
        if "hash_algo" in snapshot:
            new_algo = snapshot["hash_algo"]
            if getattr(self, "_lmcache_hash_algo", None) != new_algo:
                self._lmcache_hash_algo = new_algo
                # Invalidate cached hash fn so it gets re-loaded with new algo.
                if hasattr(self, "_lmcache_hash_fn_cached"):
                    del self._lmcache_hash_fn_cached

    def get_lmcache_cache_info_local(self, tokens: list[int]) -> dict:
        """
        Compute LMCache backend cache hit for `tokens` using the local snapshot.

        Replicates ChunkedTokenDatabase._prefix_hash logic: chunks tokens into
        chunk_size blocks, computes cumulative prefix hashes, and checks each
        hash against the pushed snapshot set. Breaks at first miss (prefix match).

        No collective_rpc — runs entirely in the GenWorker process.

        Args:
            tokens (list[int]): Full token sequence for the trajectory.

        Returns:
            dict: Dict with keys matching `lmcache_get_backend_cache_info` output.
        """
        chunk_size = self._lmcache_chunk_size
        total_bytes = self._lmcache_total_bytes
        chunk_bytes = self._lmcache_chunk_bytes

        if self._lmcache_chunk_hash_set is None or not tokens:
            return {
                "total_tokens": len(tokens) if tokens else 0,
                "lmcache_cached_chunks": 0,
                "lmcache_cached_tokens": 0,
                "lmcache_bytes": 0,
                "lmcache_total_bytes": total_bytes,
                "lmcache_usage_pct": 0.0,
                "gpu_pinned": False,
                "backend_pinned": False,
            }

        hash_fn = self._get_lmcache_hash_fn()
        none_hash = self._lmcache_none_hash

        # Replicate ChunkedTokenDatabase._prefix_hash prefix-matching logic.
        prefix_hash = none_hash
        cached_chunks = 0
        num_full_chunks = len(tokens) // chunk_size
        for i in range(num_full_chunks):
            start = i * chunk_size
            end = start + chunk_size
            tokens_tuple = tuple(tokens[start:end])
            prefix_hash = hash_fn((prefix_hash, tokens_tuple, ()))
            if prefix_hash not in self._lmcache_chunk_hash_set:
                break
            cached_chunks += 1

        # Diagnostic: log first-chunk miss (rate-limited).
        if cached_chunks == 0 and num_full_chunks > 0:
            if not hasattr(self, "_lmcache_miss_log_count"):
                self._lmcache_miss_log_count = 0
            self._lmcache_miss_log_count += 1
            if self._lmcache_miss_log_count <= 5:
                first_hash = hash_fn((none_hash, tuple(tokens[:chunk_size]), ()))
                import sys
                print(
                    f"[LMCache DIAG] First-chunk miss #{self._lmcache_miss_log_count}: "
                    f"tokens[:5]={tokens[:5]}, "
                    f"first_hash={first_hash}, "
                    f"hash_set_size={len(self._lmcache_chunk_hash_set)}, "
                    f"none_hash={none_hash}, num_full_chunks={num_full_chunks}",
                    file=sys.stderr, flush=True
                )

        cached_tokens = cached_chunks * chunk_size
        cached_bytes = cached_chunks * chunk_bytes
        usage_pct = cached_bytes / total_bytes if total_bytes > 0 else 0.0

        return {
            "total_tokens": len(tokens),
            "lmcache_cached_chunks": cached_chunks,
            "lmcache_cached_tokens": cached_tokens,
            "lmcache_bytes": cached_bytes,
            "lmcache_total_bytes": total_bytes,
            "lmcache_usage_pct": usage_pct,
            "gpu_pinned": False,
            "backend_pinned": False,
        }

    def _get_lmcache_hash_fn(self):
        """
        Return the hash function used by LMCache's ChunkedTokenDatabase.

        IMPORTANT: LMCache uses `pre_caching_hash_algorithm` (default "builtin")
        which differs from vLLM's `prefix_caching_hash_algo` (default "sha256").
        "builtin" means Python's built-in `hash()` function. Other values
        (e.g., "sha256_cbor") are loaded via vLLM's `get_hash_fn_by_name`.
        """
        if not hasattr(self, "_lmcache_hash_fn_cached"):
            algo = getattr(self, "_lmcache_hash_algo", "builtin")
            if algo == "builtin":
                self._lmcache_hash_fn_cached = hash
            else:
                from vllm.utils.hashing import get_hash_fn_by_name
                self._lmcache_hash_fn_cached = get_hash_fn_by_name(algo)
        return self._lmcache_hash_fn_cached


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

        NOTE: `get_cache_info()` no longer uses this for `lmcache_get_backend_cache_info`
        — that query is now served locally via the push-based snapshot. This method is
        still used by `pin()`, `unpin()`, `clear_from_backend()`, and `transfer()`.

        Args:
            method (str): The `vllm_extension.py` method name to call.
            args (tuple): Positional arguments forwarded to the worker method.

        Returns:
            object: The result from rank-0 worker.
        """
        results = await self._inference_engine.collective_rpc(method, args=args)
        # `collective_rpc` returns a list[result]; take rank-0 value.
        return results[0] if isinstance(results, list) else results

    async def _utility(self, method: str, *args) -> object:
        """
        Call a `psrl_*` method on the vLLM `EngineCore` via `call_utility_async`.

        Unlike `_rpc` (which dispatches to Worker processes via `collective_rpc`),
        this routes directly to the `EngineCore` object in the engine-core process.
        `EngineCore.scheduler` is the `RolloutScheduler` instance, so any method
        defined on `RolloutScheduler` (and therefore on `EngineCore` via attribute
        lookup) is reachable here.

        Use this for GPU block-pool operations (`psrl_pin_gpu`,
        `psrl_pin_gpu`, `psrl_unpin_gpu`) which must run in the same process
        as `block_pool` — the EngineCore process.  This is safe for TP>1 because
        the state is never copied across process boundaries.

        Args:
            method (str): The `RolloutScheduler` method name to call on EngineCore.
            *args: Positional arguments forwarded to the method.

        Returns:
            object: The return value of the called method.
        """
        return await self._inference_engine.engine_core.call_utility_async(method, *args)

    # --- Public KV cache operations ---

    async def get_cache_info(self, tokens: list[int]) -> TrajectoryCacheInfo:
        """
        Query GPU prefix-cache and LMCache backend usage for a token sequence.

        Both GPU and LMCache statistics are computed locally using pushed
        snapshots. No EngineCore RPC or collective_rpc involved — runs
        entirely in the GenWorker process with O(n_chunks) set lookups.

        Args:
            tokens (list[int]): Full token sequence for the trajectory.

        Returns:
            TrajectoryCacheInfo: Snapshot of cache usage for this token sequence.
        """
        self._assert_engine()
        assert tokens, "tokens must be a non-empty list."
        # GPU side: local hash snapshot (push-based, no EngineCore RPC).
        gpu_info = self.get_gpu_cache_info_local(tokens)
        # LMCache backend side: local snapshot (push-based, no collective_rpc).
        # If snapshot hasn't arrived yet (startup), returns zeros — safe because
        # the router treats 0 cached tokens as "skip migration" anyway.
        backend_info = self.get_lmcache_cache_info_local(tokens)
        return TrajectoryCacheInfo(**gpu_info, **backend_info)

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
        must have been started by `RolloutCoordinator.init_lmcache_p2p()` and its
        URL must have been injected via `set_controller_url()`.  Both intra-machine
        (same host, different GPU via UCX shared memory) and inter-machine (via NIXL
        RDMA or TCP) transfers are supported through the same interface.

        Args:
            tokens (list[int]): Full token sequence for the trajectory.
            src (tuple[str, str]): Source `(lmcache_instance_id, backend_location)` of
                the source, e.g. `("psrl_instance_0", "LocalCPUBackend")`.
            dst (tuple[str, str]): Destination `(lmcache_instance_id, backend_location)` of
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
        assert self._controller_url is not None, (
            "KVCacheManager controller URL is not set. "
            "Call set_controller_url() after RolloutCoordinator.init_lmcache_p2p() completes."
        )

        payload = {
            "old_position": list(src),
            "new_position": list(dst),
            "tokens": tokens,
            "copy": copy,
        }
        try:
            resp = await post(f"{self._controller_url}/move", payload, max_retries=1)
            psrl_logger.debug(f"[LMCache] Controller move ACK: {resp!r}.")
            self._last_transfer_error = None
            return True
        except Exception as e:
            # Include response body for HTTPStatusError (e.g., 500 from Controller).
            detail = ""
            if hasattr(e, "response") and hasattr(e.response, "text"):
                detail = f" Response: {e.response.text[:500]}"
            self._last_transfer_error = f"{type(e).__name__}: {e}{detail}"
            msg = f"[LMCache] Controller /move request failed: {e}.{detail}"
            psrl_logger.error(msg)
            # Also print to ensure visibility (manager.py logger may lack handlers).
            import sys
            print(msg, file=sys.stderr, flush=True)
            return False

    async def transfer_direct(
        self,
        tokens: list[int],
        src: tuple[str, str],
        dst: tuple[str, str],
        copy: bool = False,
    ) -> bool:
        """
        Transfer KV cache by sending MoveWorkerMsg directly to local LMCacheWorker.

        Bypasses the centralized Controller HTTP endpoint by constructing the
        same MoveWorkerMsg that the Controller's executor would send, and
        dispatching it directly to the local LMCacheWorker's ZMQ socket.

        This eliminates:
        - HTTP JSON serialization of large token lists
        - Controller single-process GIL bottleneck under burst
        - Controller event loop contention

        Each GenWorker sends to its OWN Worker, so N instances under burst have
        zero contention (fully parallel). Falls back to Controller path on failure.

        Args:
            tokens (list[int]): Full token sequence for the trajectory.
            src (tuple[str, str]): Source `(lmcache_instance_id, backend_location)`.
            dst (tuple[str, str]): Destination `(lmcache_instance_id, backend_location)`.
            copy (bool): If True, keep the data at `src` as well.

        Returns:
            bool: True if the transfer succeeded.
        """
        self._assert_engine()
        assert tokens, "tokens must be a non-empty list."
        if not self._config.enable_p2p:
            psrl_logger.warning(
                "[LMCache] transfer_direct() called but enable_p2p is False."
            )
            return False

        # Prerequisites must be met — no silent fallback.
        assert self._peer_registry, (
            "[LMCache] transfer_direct() called but peer_registry is empty. "
            "Call set_peer_registry() after P2P init."
        )
        assert self._worker_zmq_url, (
            "[LMCache] transfer_direct() called but worker_zmq_url is not set. "
            "Call set_peer_registry(registry, worker_zmq_url) after P2P init."
        )

        dst_instance_id = dst[0]
        dst_urls = self._peer_registry.get(dst_instance_id)
        # Same-rank pairing: source rank r moves its KV shard to the destination's
        # rank r endpoint. This is correct only for homogeneous layouts where src and
        # dst share TP and PP (so equal world_size and matching (tp, pp) per rank).
        # Heterogeneous layouts would need head/layer re-sharding, which LMCache cannot
        # do; in that case the destination simply re-prefills, so we skip gracefully.
        if not dst_urls or self._my_rank >= len(dst_urls) or not dst_urls[self._my_rank]:
            psrl_logger.warning(
                f"[LMCache] No same-rank peer_init_url for {dst_instance_id!r} "
                f"rank {self._my_rank} (dst has {0 if not dst_urls else len(dst_urls)} "
                "ranks). Likely heterogeneous TP/PP layout; skipping direct transfer, "
                "destination will re-prefill."
            )
            return False
        dst_peer_init_url = dst_urls[self._my_rank]

        num_tokens = await self._send_move_worker_msg(
            tokens=tokens,
            old_position=src[1],  # backend location string
            new_position=(dst_peer_init_url, dst[1]),
            copy=copy,
        )
        if num_tokens > 0:
            psrl_logger.debug(
                f"[LMCache] Direct transfer succeeded: {num_tokens} tokens "
                f"moved from {src!r} to {dst!r}."
            )
        else:
            psrl_logger.info(
                f"[LMCache] Direct transfer returned 0 tokens for "
                f"{src!r} → {dst!r}. Source may not have cached data."
            )
        return num_tokens > 0

    async def _send_move_worker_msg(
        self,
        tokens: list[int],
        old_position: str,
        new_position: tuple[str, str],
        copy: bool,
    ) -> int:
        """
        Construct and send MoveWorkerMsg directly to the local LMCacheWorker via ZMQ.

        Replicates what LMCacheClusterExecutor.move() does (executor.py:281-350)
        but without the Controller intermediary. Uses async ZMQ for non-blocking I/O.

        Args:
            tokens: Full token sequence.
            old_position: Source backend location (e.g. "LocalCPUBackend").
            new_position: Tuple of (dst_peer_init_url, dst_backend_location).
            copy: Whether to keep data at source.

        Returns:
            int: Number of tokens transferred.
        """
        from lmcache.v1.cache_controller.message import (
            MoveWorkerMsg,
            MoveWorkerRetMsg,
            Msg,
        )

        worker_event_id = f"DirectMove_{uuid.uuid4().hex[:8]}"
        msg = MoveWorkerMsg(
            worker_event_id=worker_event_id,
            old_position=old_position,
            new_position=new_position,
            tokens=tokens,
            copy=copy,
        )

        serialized_msg = msgspec.msgpack.encode(msg)
        # ZMQ REQ socket requires strict send→recv alternation; lock serializes concurrent calls.
        async with self._direct_zmq_lock:
            socket = self._get_or_create_zmq_socket()
            try:
                await socket.send(serialized_msg)
                serialized_resp = await socket.recv()
            except Exception:
                # A REQ socket that fails mid send→recv (e.g. RCVTIMEO fires) is
                # stuck in a bad EFSM state: every subsequent send() raises until
                # the socket is rebuilt. Reset it here so the next call reconnects,
                # otherwise one timeout cascades into a burst of transfer failures.
                self._reset_zmq_socket()
                raise
        resp = msgspec.msgpack.decode(serialized_resp, type=Msg)

        if hasattr(resp, "num_tokens"):
            return resp.num_tokens
        else:
            psrl_logger.warning(
                f"[LMCache] Unexpected response type from Worker: {type(resp).__name__}"
            )
            return 0

    def _get_or_create_zmq_socket(self):
        """
        Get or lazily create an async ZMQ REQ socket connected to the local Worker.

        Returns:
            zmq.asyncio.Socket: Connected ZMQ REQ socket.
        """
        if self._direct_zmq_socket is None:
            import zmq
            import zmq.asyncio

            assert self._worker_zmq_url is not None
            self._direct_zmq_context = zmq.asyncio.Context()
            self._direct_zmq_socket = self._direct_zmq_context.socket(zmq.REQ)
            self._direct_zmq_socket.connect(f"tcp://{self._worker_zmq_url}")
            # Set send/recv timeout to avoid hanging indefinitely.
            self._direct_zmq_socket.setsockopt(zmq.SNDTIMEO, 10000)  # 10s
            self._direct_zmq_socket.setsockopt(zmq.RCVTIMEO, 30000)  # 30s
            psrl_logger.info(
                f"[LMCache] Direct ZMQ socket connected to {self._worker_zmq_url}."
            )
        return self._direct_zmq_socket

    def _reset_zmq_socket(self) -> None:
        """
        Close and discard the direct ZMQ socket so the next call rebuilds it.

        Called after a send/recv failure on the REQ socket. A REQ socket that
        raised mid send→recv is stuck in a bad EFSM state and cannot be reused;
        dropping it here lets `_get_or_create_zmq_socket` reconnect cleanly.

        Caller must hold `_direct_zmq_lock`.
        """
        if self._direct_zmq_socket is not None:
            try:
                self._direct_zmq_socket.close(linger=0)
            except Exception:
                pass
            self._direct_zmq_socket = None
            psrl_logger.warning(
                "[LMCache] Direct ZMQ socket reset after transfer failure; "
                "will reconnect on next transfer."
            )

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
                freed: int = await self._utility("psrl_unpin_gpu", oldest_tokens)
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

        pinned: int = await self._utility("psrl_pin_gpu", tokens)
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
        freed: int = await self._utility("psrl_unpin_gpu", tokens)
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

