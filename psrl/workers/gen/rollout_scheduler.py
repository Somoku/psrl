import logging
import os
import time

from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import hash_block_tokens, init_none_hash, make_block_hash_with_group_id
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.utils import record_function_or_nullcontext

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class RolloutScheduler(Scheduler):
    def make_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None = None,
        kv_connector_stats: KVConnectorStats | None = None,
    ) -> SchedulerStats | None:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        connector_prefix_cache_stats = self._make_connector_prefix_cache_stats()
        eviction_events = self.kv_metrics_collector.drain_events() if self.kv_metrics_collector is not None else []
        spec_stats = spec_decoding_stats
        connector_stats_payload = kv_connector_stats.data if kv_connector_stats else None
        req_id_to_prompt_token_num = {req_id: req.num_prompt_tokens for req_id, req in self.requests.items()}
        req_id_to_response_token_num = {req_id: req.num_output_tokens for req_id, req in self.requests.items()}
        # NOTE(lhy): we need to patch the original vllm SchedulerStats to add:
        # 1. `need_to_abort_reqs` field. This is a set of request IDs that need to be aborted.
        # 2. `req_id_to_prompt_token_num` field. This is a dictionary of request ID to the number of prompt tokens.
        # 3. `req_id_to_response_token_num` field. This is a dictionary of request ID to the number of response tokens.
        return SchedulerStats(
            need_to_abort_reqs=self.need_to_abort_reqs,
            req_id_to_prompt_token_num=req_id_to_prompt_token_num,
            req_id_to_response_token_num=req_id_to_response_token_num,
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            kv_cache_usage=self.kv_cache_manager.usage,
            prefix_cache_stats=prefix_cache_stats,
            connector_prefix_cache_stats=connector_prefix_cache_stats,
            kv_cache_eviction_events=eviction_events,
            spec_decoding_stats=spec_stats,
            kv_connector_stats=connector_stats_payload,
        )

    # --- PSRL GPU block pool helpers (called via EngineCore.call_utility_async) ---
    # These methods run in the EngineCore process where `self.kv_cache_manager.block_pool`
    # is live and mutable.  They are intentionally NOT routed through `collective_rpc`
    # (which dispatches to Worker processes) because `block_pool` state must only be
    # mutated from a single process.  `KVCacheManager` in the PSRL coordinator calls
    # these via `engine_core.call_utility_async("psrl_get_gpu_cache_info", tokens)`.

    def _psrl_get_caching_hash_fn(self):
        """
        Return the token-hashing function used by the block pool.

        Prefers `get_caching_hash_fn` from the LMCache integration utils when
        available; falls back to SHA-256 for environments without the full vLLM
        LMCache stack (e.g., unit tests).

        Returns:
            callable: A function that takes token data and returns a hash digest.
        """
        try:
            from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration.utils import (
                get_caching_hash_fn,
            )
            return get_caching_hash_fn()
        except ImportError:
            import hashlib
            return lambda data: hashlib.sha256(str(data).encode()).digest()
        except Exception as e:
            import hashlib
            psrl_logger.warning(
                f"[LMCache] Failed to load get_caching_hash_fn, falling back to SHA-256: {e!r}."
            )
            return lambda data: hashlib.sha256(str(data).encode()).digest()

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

    def psrl_get_gpu_cache_info(self, tokens: list[int]) -> dict:
        """
        Return GPU prefix-cache statistics for `tokens`.

        Called via `EngineCore.call_utility_async` from `KVCacheManager`.
        Only covers the GPU side; the LMCache backend side is queried separately
        on the Worker via `collective_rpc`.

        Args:
            tokens (list[int]): Full token sequence for the trajectory.

        Returns:
            dict: Dict with keys `gpu_cached_blocks`, `gpu_cached_tokens`,
                `gpu_total_blocks`, `gpu_usage_pct`.
        """
        assert tokens, "tokens must be a non-empty list."
        block_pool = self.kv_cache_manager.block_pool
        gpu_blocks = list(self._psrl_iter_gpu_prefix_blocks(tokens))
        gpu_cached_blocks = len(gpu_blocks)
        block_size = block_pool.hash_block_size
        gpu_cached_tokens = gpu_cached_blocks * block_size
        gpu_total_blocks = block_pool.num_gpu_blocks
        gpu_usage_pct = gpu_cached_blocks / gpu_total_blocks if gpu_total_blocks > 0 else 0.0
        return {
            "gpu_cached_blocks": gpu_cached_blocks,
            "gpu_cached_tokens": gpu_cached_tokens,
            "gpu_total_blocks": gpu_total_blocks,
            "gpu_usage_pct": gpu_usage_pct,
        }

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

    def _preempt_request(
        self,
        request: Request,
        timestamp: float,
    ) -> None:
        """Preempt a request and put it back to the waiting queue.

        NOTE: The request should be popped from the running queue outside of this
        method.
        """
        assert request.status == RequestStatus.RUNNING, "Only running requests can be preempted"
        self.kv_cache_manager.free(request)
        self.encoder_cache_manager.free(request)
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
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

        # NOTE(lhy): We examine the number of waiting requests to
        # determine whether to abort the preempted request.
        # Once aborted, the preempted request will be put back to
        # the rollout router to be scheduled again.
        max_num_waiting_reqs_after_preemption = self.vllm_config.additional_config.get(
            "max_num_waiting_reqs_after_preemption", 0
        )
        if len(self.waiting) > max_num_waiting_reqs_after_preemption:
            # NOTE(lhy): the `need_to_abort_reqs` is set and put
            # into the scheduler stats. Afterwards inside vllm
            # rollout, the abortion will be performed.
            print(
                f"Preempted request {request.request_id} is "
                f"aborted because of "
                f"max_num_waiting_reqs_after_preemption is "
                f"{max_num_waiting_reqs_after_preemption}"
            )
            self.need_to_abort_reqs.append(request.request_id)

        # Put the request back to the waiting queue.
        self.waiting.prepend_request(request)

    # NOTE(lhy): The most of the code is copied from the vLLM 10.0.2 scheduler.
    # We refactor the logic of preemption to allow preempted sequences to directly returned as aborted.
    # So that we can allow rollout migration of the waiting requests between different instances.
    def schedule(self) -> SchedulerOutput:
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []
        self.need_to_abort_reqs: list[str] = list()

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            if (
                request.num_output_placeholders > 0
                # This is (num_computed_tokens + 1) - (num_output_placeholders - 1).
                # Since output placeholders are also included in the computed tokens
                # count, we subtract (num_output_placeholders - 1) to remove any draft
                # tokens, so that we can be sure no further steps are needed even if
                # they are all rejected.
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                # Async scheduling: Avoid scheduling an extra step when we are sure that
                # the previous step has reached request.max_tokens. We don't schedule
                # partial draft tokens since this prevents uniform decode optimizations.
                req_index += 1
                continue

            num_new_tokens = (
                request.num_tokens_with_spec + request.num_output_placeholders - request.num_computed_tokens
            )
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens)

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            external_load_encoder_input: list[int] = []
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                    external_load_encoder_input,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                    shift_computed_tokens=1 if self.use_eagle else 0,
                )

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            # Schedule newly needed KV blocks for the request.
            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )

                    if new_blocks is not None:
                        # The request can be scheduled.
                        break

                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            scheduled_running_reqs.remove(preempted_req)
                            token_budget += num_scheduled_tokens[preempted_req.request_id]
                            req_to_new_blocks.pop(preempted_req.request_id)
                            num_scheduled_tokens.pop(preempted_req.request_id)
                            scheduled_spec_decode_tokens.pop(preempted_req.request_id, None)
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(preempted_req.request_id, None)
                            if preempted_encoder_inputs:
                                # Restore encoder compute budget if the preempted
                                # request had encoder inputs scheduled in this step.
                                num_tokens_to_restore = sum(
                                    preempted_req.get_num_encoder_tokens(i) for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_tokens_to_restore
                            req_index -= 1
                    else:
                        preempted_req = self.running.pop()

                    self._preempt_request(preempted_req, scheduled_timestamp)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt. Cannot schedule this request.
                        break

            if new_blocks is None:
                # Cannot schedule this request.
                break

            # Schedule the request.
            scheduled_running_reqs.append(request)
            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = num_new_tokens + request.num_computed_tokens - request.num_tokens
                if num_scheduled_spec_tokens > 0:
                    # Trim spec_token_ids list to num_scheduled_spec_tokens.
                    del request.spec_token_ids[num_scheduled_spec_tokens:]
                    scheduled_spec_decode_tokens[request.request_id] = request.spec_token_ids
                # New spec tokens will be set in `update_draft_token_ids` before the
                # next step when applicable.
                request.spec_token_ids = []

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = encoder_inputs_to_schedule
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_compute_budget = new_encoder_compute_budget
            if external_load_encoder_input:
                for i in external_load_encoder_input:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)

        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # Use a temporary RequestQueue to collect requests that need to be
        # skipped and put back at the head of the waiting queue later
        skipped_waiting_requests = create_request_queue(self.policy)

        # Next, schedule the WAITING requests.
        if not preempted_reqs:
            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request = self.waiting.peek_request()

                # KVTransfer: skip request if still waiting for remote kvs.
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    is_ready = self._update_waiting_for_remote_kv(request)
                    if is_ready:
                        request.status = RequestStatus.WAITING
                    else:
                        psrl_logger.debug(f"{request.request_id} is still in WAITING_FOR_REMOTE_KVS state.")
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Skip request if the structured output request is still waiting
                # for FSM compilation.
                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Check that adding the request still respects the max_loras
                # constraint.
                if (
                    self.lora_config
                    and request.lora_request
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id not in scheduled_loras
                    )
                ):
                    # Scheduling would exceed max_loras, skip.
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False

                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    # Get locally-cached tokens.
                    new_computed_blocks, num_new_local_computed_tokens = self.kv_cache_manager.get_computed_blocks(
                        request
                    )

                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None:
                        ext_tokens, load_kv_async = self.connector.get_num_new_matched_tokens(
                            request, num_new_local_computed_tokens
                        )

                        if ext_tokens is None:
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            self.waiting.pop_request()
                            skipped_waiting_requests.prepend_request(request)
                            continue

                        request.num_external_computed_tokens = ext_tokens
                        num_external_computed_tokens = ext_tokens

                    # Total computed tokens (local + external).
                    num_computed_tokens = num_new_local_computed_tokens + num_external_computed_tokens
                else:
                    # KVTransfer: WAITING reqs have num_computed_tokens > 0
                    # after async KV recvs are completed.
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                encoder_inputs_to_schedule = None
                external_load_encoder_input = []
                new_encoder_compute_budget = encoder_compute_budget

                if load_kv_async:
                    # KVTransfer: loading remote KV, do not allocate for new work.
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                else:
                    # Number of tokens to be scheduled.
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    if not self.scheduler_config.enable_chunked_prefill and num_new_tokens > token_budget:
                        # If chunked_prefill is disabled,
                        # we can stop the scheduling here.
                        break

                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0

                    # Schedule encoder inputs.
                    if request.has_encoder_inputs:
                        (
                            encoder_inputs_to_schedule,
                            num_new_tokens,
                            new_encoder_compute_budget,
                            external_load_encoder_input,
                        ) = self._try_schedule_encoder_inputs(
                            request,
                            num_computed_tokens,
                            num_new_tokens,
                            encoder_compute_budget,
                            shift_computed_tokens=1 if self.use_eagle else 0,
                        )
                        if num_new_tokens == 0:
                            # The request cannot be scheduled.
                            break

                # Handles an edge case when P/D Disaggregation
                # is used with Spec Decoding where an
                # extra block gets allocated which
                # creates a mismatch between the number
                # of local and remote blocks.
                effective_lookahead_tokens = 0 if request.num_computed_tokens == 0 else self.num_lookahead_tokens

                # Determine if we need to allocate cross-attention blocks.
                if self.is_encoder_decoder and request.has_encoder_inputs:
                    # TODO(russellb): For Whisper, we know that the input is
                    # always padded to the maximum length. If we support other
                    # encoder-decoder models, this will need to be updated if we
                    # want to only allocate what is needed.
                    num_encoder_tokens = self.scheduler_config.max_num_encoder_input_tokens
                else:
                    num_encoder_tokens = 0

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens + num_external_computed_tokens,
                    num_new_local_computed_tokens,
                    new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    delay_cache_blocks=load_kv_async,
                    num_encoder_tokens=num_encoder_tokens,
                )

                if new_blocks is None:
                    # The request cannot be scheduled.
                    break

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        new_computed_blocks + new_blocks,
                        num_external_computed_tokens,
                    )

                # Request was already popped from self.waiting
                # unless it was re-added above due to new_blocks being None.
                request = self.waiting.pop_request()
                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    skipped_waiting_requests.prepend_request(request)
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    continue

                self._update_connector_prefix_cache_stats(request)

                req_index += 1
                self.running.append(request)
                if self.log_stats:
                    request.record_event(EngineCoreEventType.SCHEDULED, scheduled_timestamp)
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_blocks[request.request_id] = self.kv_cache_manager.get_blocks(request.request_id)
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                # Count the number of prefix cached tokens.
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens
                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = encoder_inputs_to_schedule
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_compute_budget = new_encoder_compute_budget
                # Allocate for external load encoder cache
                if external_load_encoder_input:
                    for i in external_load_encoder_input:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)

        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens

        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(scheduled_running_reqs) <= len(self.running)

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
            if self.running:
                any_request = self.running[0]
                num_common_prefix_blocks = self.kv_cache_manager.get_num_common_prefix_blocks(any_request.request_id)

        # Construct the scheduler output.
        if self.use_v2_model_runner:
            scheduled_new_reqs = scheduled_new_reqs + scheduled_resumed_reqs
            scheduled_resumed_reqs = []
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    req._all_token_ids,
                )
                for req in scheduled_new_reqs
            ]
        else:
            new_reqs_data = [
                NewRequestData.from_request(req, req_to_new_blocks[req.request_id].get_block_ids())
                for req in scheduled_new_reqs
            ]

        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
            )

        # Record the request ids that were scheduled in this step.
        self.prev_step_scheduled_req_ids.clear()
        self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            preempted_req_ids={req.request_id for req in preempted_reqs},
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
        )

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self.connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        # Build the connector meta for ECConnector
        if self.ec_connector is not None:
            ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(scheduler_output)
            scheduler_output.ec_connector_metadata = ec_meta

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        return scheduler_output
