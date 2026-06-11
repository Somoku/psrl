import logging
import os

from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.v1.core.kv_cache_utils import hash_block_tokens, init_none_hash, make_block_hash_with_group_id
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.metrics.perf import PerfStats
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class RolloutScheduler(Scheduler):
    def make_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None = None,
        kv_connector_stats: KVConnectorStats | None = None,
        cudagraph_stats: CUDAGraphStat | None = None,
        perf_stats: PerfStats | None = None,
    ) -> SchedulerStats | None:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        connector_prefix_cache_stats: PrefixCacheStats | None = None
        if self.connector_prefix_cache_stats is not None:
            connector_prefix_cache_stats = self.connector_prefix_cache_stats
            self.connector_prefix_cache_stats = PrefixCacheStats()
        eviction_events = self.kv_metrics_collector.drain_events() if self.kv_metrics_collector is not None else []
        spec_stats = spec_decoding_stats
        connector_stats_payload = kv_connector_stats.data if kv_connector_stats else None
        req_id_to_prompt_token_num = {req_id: req.num_prompt_tokens for req_id, req in self.requests.items()}
        req_id_to_response_token_num = {req_id: req.num_output_tokens for req_id, req in self.requests.items()}
        # Snapshot the GPU prefix cache hash table for push-based KV cache info.
        # This is sent to the main process via SchedulerStats IPC, then consumed
        # by StatCollector.record() and pushed to GenWorker's KVCacheManager.
        block_pool = self.kv_cache_manager.block_pool
        gpu_prefix_cache_snapshot = {
            "hash_set": set(block_pool.cached_block_hash_to_block._cache.keys()),
            "block_size": block_pool.hash_block_size,
            "total_blocks": block_pool.num_gpu_blocks,
        }
        # Snapshot the LMCache backend chunk hash set for push-based cache info.
        # For TP=1 (UniProcExecutor), the Worker and EngineCore share the same
        # process, so get_kv_transfer_group() returns the active LMCache connector.
        lmcache_backend_snapshot = None
        try:
            from vllm.distributed.kv_transfer import has_kv_transfer_group, get_kv_transfer_group
            if has_kv_transfer_group():
                connector = get_kv_transfer_group()
                if hasattr(connector, "_lmcache_engine") and connector._lmcache_engine is not None:
                    lmcache_engine = connector._lmcache_engine.lmcache_engine
                    if lmcache_engine is not None:
                        backend = lmcache_engine.storage_manager.local_cpu_backend
                        hot_cache = backend.hot_cache
                        chunk_size = getattr(lmcache_engine.token_database, "chunk_size", 256)
                        # Snapshot: set of chunk_hash integers from all cached keys.
                        # Must hold cpu_lock to avoid OrderedDict mutation during iteration.
                        chunk_hash_set = set()
                        with backend.cpu_lock:
                            for key in hot_cache.keys():
                                chunk_hash_set.add(key.chunk_hash)
                        # Metadata for local query.
                        allocator = backend.memory_allocator
                        total_bytes = int(getattr(allocator, "total_size", 0))
                        try:
                            chunk_bytes = backend.get_full_chunk_size_bytes()
                        except Exception:
                            chunk_bytes = 0
                        lmcache_backend_snapshot = {
                            "chunk_hash_set": chunk_hash_set,
                            "chunk_size": chunk_size,
                            "total_bytes": total_bytes,
                            "chunk_bytes": chunk_bytes,
                        }
                        # Include NONE_HASH and hash_algo for GenWorker local computation.
                        # NOTE: hash_fn (function object) is NOT serializable through
                        # zmq/msgspec, so we pass the algorithm name and none_hash value.
                        from lmcache.v1.token_database import NONE_HASH as _LM_NONE_HASH
                        lmcache_backend_snapshot["none_hash"] = _LM_NONE_HASH
                        # Pass the hash algorithm name so GenWorker loads the SAME function.
                        # LMCache uses config.pre_caching_hash_algorithm (default: "builtin"),
                        # which may differ from vLLM's prefix_caching_hash_algo (default: "sha256").
                        lmcache_backend_snapshot["hash_algo"] = getattr(
                            lmcache_engine.token_database, "_hash_algo_name",
                            getattr(lmcache_engine.token_database.config, "pre_caching_hash_algorithm", "builtin")
                            if lmcache_engine.token_database.config is not None else "builtin"
                        )
                        # Hash consistency probe: GenWorker uses this to verify cross-process
                        # hash() determinism. If mismatch → PYTHONHASHSEED not set properly.
                        _hash_fn = lmcache_engine.token_database.hash_func
                        lmcache_backend_snapshot["hash_probe"] = _hash_fn(
                            (_LM_NONE_HASH, (0, 1, 2, 3), ())
                        )
        except Exception as exc:
            # Non-fatal: if snapshot fails, router falls back to collective_rpc.
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "LMCache backend snapshot failed: %r", exc, exc_info=True
            )
        # NOTE(lhy): we need to patch the original vllm SchedulerStats to add:
        # 1. `req_id_to_prompt_token_num` field. This is a dictionary of request ID to the number of prompt tokens.
        # 2. `req_id_to_response_token_num` field. This is a dictionary of request ID to the number of response tokens.
        # 3. `gpu_prefix_cache_snapshot` field. Push-based GPU prefix cache hash snapshot.
        # 4. `lmcache_backend_snapshot` field. Push-based LMCache backend chunk hash snapshot.
        return SchedulerStats(
            req_id_to_prompt_token_num=req_id_to_prompt_token_num,
            req_id_to_response_token_num=req_id_to_response_token_num,
            gpu_prefix_cache_snapshot=gpu_prefix_cache_snapshot,
            lmcache_backend_snapshot=lmcache_backend_snapshot,
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            num_skipped_waiting_reqs=len(self.skipped_waiting),
            kv_cache_usage=self.kv_cache_manager.usage,
            prefix_cache_stats=prefix_cache_stats,
            connector_prefix_cache_stats=connector_prefix_cache_stats,
            kv_cache_eviction_events=eviction_events,
            spec_decoding_stats=spec_stats,
            kv_connector_stats=connector_stats_payload,
            cudagraph_stats=cudagraph_stats,
            perf_stats=perf_stats,
        )

    # --- PSRL GPU block pool helpers (called via EngineCore.call_utility_async) ---
    # These methods run in the EngineCore process where `self.kv_cache_manager.block_pool`
    # is live and mutable.  They are intentionally NOT routed through `collective_rpc`
    # (which dispatches to Worker processes) because `block_pool` state must only be
    # mutated from a single process.  `KVCacheManager` in the PSRL coordinator calls
    # these via `engine_core.call_utility_async("psrl_pin_gpu/psrl_unpin_gpu", tokens)`.
    #
    # NOTE(lhy): GPU prefix cache *read* queries (hit counts for routing) no longer
    # go through call_utility_async — they use the push-based hash snapshot in
    # KVCacheManager.get_gpu_cache_info_local(). Only pin/unpin (which mutate
    # block_pool ref_cnt) still require EngineCore RPC.

    def _psrl_get_caching_hash_fn(self):
        """
        Return the token-hashing function used by the block pool.

        Reads prefix_caching_hash_algo from self.vllm_config (available on the Scheduler).
        """
        from vllm.utils.hashing import get_hash_fn_by_name

        hash_algo = self.vllm_config.cache_config.prefix_caching_hash_algo
        return get_hash_fn_by_name(hash_algo)

    def _psrl_iter_gpu_prefix_blocks(self, tokens: list[int]):
        """
        Yield GPU `KVCacheBlock` objects forming the longest contiguous cached prefix.

        Walks the prefix-hash chain on `block_pool.cached_block_hash_to_block`,
        stopping at the first miss.

        Args:
            tokens (list[int]): Full token sequence.

        Yields:
            KVCacheBlock: Blocks in prefix order.
        """
        block_pool = self.kv_cache_manager.block_pool
        block_size = block_pool.hash_block_size
        hash_fn = self._psrl_get_caching_hash_fn()
        # `NONE_HASH` in `kv_cache_utils` must be initialised before calling
        # `hash_block_tokens`.  `init_none_hash` is idempotent once called.
        init_none_hash(hash_fn)

        prev_hash = None
        num_full_blocks = len(tokens) // block_size
        for block_idx in range(num_full_blocks):
            start = block_idx * block_size
            end = start + block_size
            chunk = tokens[start:end]
            block_hash = hash_block_tokens(hash_fn, prev_hash, chunk, None)
            prev_hash = block_hash
            # `kv_cache_group_id=0` for standard (non-MLA) models.
            key = make_block_hash_with_group_id(block_hash, 0)
            block = block_pool.cached_block_hash_to_block.get_one_block(key)
            if block is None:
                return  # prefix break
            yield block

    def psrl_pin_gpu(self, tokens: list[int]) -> int:
        """
        Pin GPU prefix-cache blocks for `tokens` by incrementing `ref_cnt`.

        Only pins blocks with `ref_cnt == 0` (free queue).  Tracks pinned block
        IDs in `_psrl_pinned_block_ids` so `psrl_unpin_gpu` cannot decrement
        `ref_cnt` for blocks held by active vLLM requests.

        Args:
            tokens (list[int]): Full token sequence for the trajectory.

        Returns:
            int: Number of blocks newly pinned.
        """
        assert tokens, "tokens must be a non-empty list."
        if not hasattr(self, "_psrl_pinned_block_ids"):
            self._psrl_pinned_block_ids: set[int] = set()

        block_pool = self.kv_cache_manager.block_pool
        pinned = 0
        blocks_to_touch = []
        for block in self._psrl_iter_gpu_prefix_blocks(tokens):
            if block.ref_cnt == 0 and block.block_id not in self._psrl_pinned_block_ids:
                blocks_to_touch.append(block)
                self._psrl_pinned_block_ids.add(block.block_id)
                pinned += 1
        if blocks_to_touch:
            # NOTE(claude): `block_pool.touch()` expects a tuple of per-group block sequences.
            # For standard (non-MLA) models there is one KV cache group, so we pass all
            # blocks as a single-element tuple.
            block_pool.touch((blocks_to_touch,))
            for block in blocks_to_touch:
                assert block.ref_cnt > 0, (
                    f"Block {block.block_id} ref_cnt is {block.ref_cnt} after touch(). Expected > 0."
                )
        psrl_logger.debug(
            f"[LMCache] GPU pin (scheduler): {pinned} blocks pinned for token sequence "
            f"of length {len(tokens)}."
        )
        return pinned

    def psrl_unpin_gpu(self, tokens: list[int]) -> int:
        """
        Unpin GPU prefix-cache blocks for `tokens` by decrementing `ref_cnt`.

        Only decrements `ref_cnt` for blocks that PSRL itself pinned (tracked in
        `_psrl_pinned_block_ids`), preventing interference with active requests.

        Args:
            tokens (list[int]): Full token sequence for the trajectory.

        Returns:
            int: Number of blocks unpinned.
        """
        assert tokens, "tokens must be a non-empty list."
        if not hasattr(self, "_psrl_pinned_block_ids"):
            self._psrl_pinned_block_ids: set[int] = set()

        block_pool = self.kv_cache_manager.block_pool
        freed = 0
        for block in self._psrl_iter_gpu_prefix_blocks(tokens):
            if block.block_id in self._psrl_pinned_block_ids:
                assert block.ref_cnt > 0, (
                    f"Block {block.block_id} ref_cnt is {block.ref_cnt} before free_blocks(). "
                    "Cannot unpin a block with ref_cnt <= 0."
                )
                block_pool.free_blocks([block])
                self._psrl_pinned_block_ids.discard(block.block_id)
                freed += 1
        psrl_logger.debug(
            f"[LMCache] GPU unpin (scheduler): {freed} blocks released for token sequence "
            f"of length {len(tokens)}."
        )
        return freed

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        """Preempt a request and put it back to the waiting queue.

        NOTE: The request should be popped from the running queue outside of this
        method.
        """
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"
        )
        self.kv_cache_manager.free(request)
        self.encoder_cache_manager.free(request)
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        if request.spec_token_ids:
            request.spec_token_ids = []
        request.num_preemptions += 1
        if self.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)

        # NOTE(claude): Record QUEUED immediately after PREEMPTED to mark the
        # moment the request re-enters the waiting queue. This gives
        # `_split_into_segments` the QUEUED event it needs to open a new
        # segment for the eventual re-schedule, so scheduler_wait_s for the
        # resume segment is measured from this point rather than from the
        # original submission.
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED, timestamp)

        # NOTE(claude): Save the current output token count as the baseline for
        # the next scheduling cycle. The FIRST_TOKEN event fires when
        # `num_output_tokens` first exceeds this baseline, correctly identifying
        # the prefill→decode boundary after a preemption even though
        # `_output_token_ids` is not cleared on preemption.
        request._psrl_cycle_output_token_baseline = request.num_output_tokens

        # Put the request back to the waiting queue.
        self.waiting.prepend_request(request)
        # Notify external gateway if threshold is configured and waiting queue
        # is already congested — local re-queuing would only worsen the load.
        threshold = self.scheduler_config.preemption_notification_threshold
        if self.log_stats and threshold is not None and len(self.waiting) > threshold:
            self.preemption_req_ids.append(request.request_id)