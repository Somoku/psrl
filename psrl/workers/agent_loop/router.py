import asyncio
import logging
import os
import time

import numpy as np
import ray
from omegaconf import DictConfig
from tensordict import TensorDict
from verl import DataProto
from vllm.sampling_params import RequestOutputKind

from psrl.utils.kv_cache.types import TrajectoryCacheInfo
from psrl.utils.logger import DualOutputHandler, EventType, deprecated, log_dual_events
from psrl.utils.ray import AsyncBusyPollingRayLock
from psrl.utils.rollout.rollout_trace import rollout_trace_op
from psrl.workers.agent_loop.request_queue import (
    MultiPriorityRequestQueue,
    PriorityRequestQueue,
    RequestSortIndicator,
)
from psrl.workers.agent_loop.route_strategy import (
    RouteStrategyBase,
    get_route_strategy_class,
)
from psrl.workers.gen.stats_collector import EngineStats
from psrl.workers.ps.request_status_tracker import PSRL_RequestStatus

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


def _to_token_list(tokens) -> list:
    """Convert tokens to a Python list, handling both numpy arrays, tensors, and plain lists."""
    if isinstance(tokens, list):
        return tokens
    if hasattr(tokens, "tolist"):
        return tokens.tolist()
    return list(tokens)


@ray.remote(concurrency_groups={"control": 1})
class RolloutRouter:
    def __init__(
        self,
        config: DictConfig,
        ps_manager_handle,
        tokenizer,
        rollout_wg_list,
    ):
        """Initialize the rollout router.
        Managing rollout requests across multiple worker groups.
        Handles request routing, load balancing, and consolidation of generation results.

        Args:
            config (DictConfig): Configuration containing rollout settings.
            ps_manager_handle: Handle to the parameter server manager.
            rollout_wg_list: List of rollout worker groups.
        """
        self.config = config
        self.staleness = self.config.psrl.staleness
        self.n_rollout_instances = self.config.psrl.deployment.n_rollout_instances
        self.n_validate_instances = (
            self.config.psrl.deployment.n_validate_instances if self.config.psrl.colocate_validate_and_train else 0
        )

        # TODO(linsh): currently we only support balance strategy on rollout instances
        # we may extend it to validate instances in the future with dynamic routing strategy
        if self.config.psrl.redundant_rollout.enable:
            self.rollout_n = self.config.psrl.redundant_rollout.redundant_rollout_n
            self.alg_rollout_n = self.config.psrl.redundant_rollout.alg_rollout_n
            self.balanced_concurrent_seqs_per_instance = (
                self.config.psrl.redundant_rollout.redundant_global_batch_size
                * self.rollout_n
                // self.n_rollout_instances
            )
        else:
            self.rollout_n = self.config.gen_actor_rollout_ref.rollout.n
            self.alg_rollout_n = self.rollout_n
            self.balanced_concurrent_seqs_per_instance = (
                self.config.psrl.staleness_buffer_entries * self.rollout_n // self.n_rollout_instances
            )

        self.val_rollout_n = self.config.train_actor_rollout_ref.rollout.val_kwargs.n
        self.ps_manager_handle = ps_manager_handle
        self.tokenizer = tokenizer
        self.rollout_wg_list = rollout_wg_list
        self.rollout_wg_size = len(rollout_wg_list)
        assert self.rollout_wg_size == self.n_rollout_instances + self.n_validate_instances, (
            "Rollout worker group size must match the number of deployment instances"
        )

        # Build logger
        self.log_prefix = "RolloutRouter"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))

        # Routing related attributes
        if self.config.psrl.routing_strategy.enable_multi_priority_queue:
            self.requests_to_route = MultiPriorityRequestQueue(
                self.staleness,
                request_sort_indicator=RequestSortIndicator(self.config.psrl.routing_strategy.request_sort_indicator),
            )
        else:
            self.requests_to_route = PriorityRequestQueue(
                self.staleness,
                request_sort_indicator=RequestSortIndicator(self.config.psrl.routing_strategy.request_sort_indicator),
            )
        self._is_routing = False
        self._pause_routing = False
        self.scheduler_task = None  # Will be created in async context
        # Track the inflight request ids for each instance (i.e., request that is being generated
        # and is not yet completed or queued in the priority queue): {instance_id: [request_id, ...]}
        self.instance_to_inflight_request_ids = {i: [] for i in range(self.rollout_wg_size)}
        # Track the instance id for each incomplete request (i.e., request that is not completed yet):
        # {request_id: instance_id}
        self.incomplete_request_to_instance = {}
        # Trajectory-level instance mapping — records the last instance each request
        # was routed to.  Survives individual turn completion (unlike incomplete_request_to_instance
        # which is cleared per-turn); used for KV cache migration and routing decisions
        # between turns.  Cleared only when the full trajectory ends (kv_unregister).
        self.request_to_last_routed_instance: dict[int, int] = {}
        # Trajectory-level token mapping — auto-maintained by generate_async / _route_single_request.
        self.request_to_tokens: dict[int, list[int]] = {}
        self.request_futures = {}  # Track request futures: {request_id: Future}
        # Track the version after synchronization for each instance: {instance_id: ps_model_version}
        self.instance_to_version_after_sync = {i: 0 for i in range(self.rollout_wg_size)}
        # Track the instance ids that are currently paused (not available for routing)
        self.currently_paused_instance_ids = set()
        # Track requests in sticky session: {request_id: bool}
        self.sticky_session_requests = {}

        # Build logger
        self.log_prefix = "RolloutRouter"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))
        psrl_logger.info("Initialized RolloutRouter")

    async def _inject_kv_hit_scores(
        self,
        request_id: int,
        candidates: list[int],
        route_kwargs: dict,
    ) -> None:
        """Query KV cache hit scores for all candidates and store in `route_kwargs`.

        Only runs when `self.config.psrl.routing_strategy.method` is
        `"kv_cache_aware"`.  Each candidate is queried concurrently via
        `execute_rank_zero_async("kv_get_cache_info", tokens)`.  Instances that
        time out or raise are assigned score 0 (graceful degradation).

        Mutates `route_kwargs` in-place by adding the key `"kv_hit_scores"`.
        """
        if self.config.psrl.routing_strategy.method != "kv_cache_aware":
            return
        assert request_id in self.request_to_tokens, (
            f"[KV] uid={request_id} not found in request_to_tokens. "
            "Expected START registration in generate_async before routing."
        )
        tokens: list[int] = self.request_to_tokens[request_id]
        timeout: float = self.config.psrl.routing_strategy.kv_query_timeout_ms / 1000
        raw_results = await asyncio.gather(
            *[
                asyncio.wait_for(
                    self.rollout_wg_list[i].execute_rank_zero_async(
                        "kv_get_cache_info", tokens
                    ),
                    timeout=timeout,
                )
                for i in candidates
            ],
            return_exceptions=True,
        )
        kv_hit_scores: dict[int, int] = {}
        for i, result in zip(candidates, raw_results):
            if isinstance(result, Exception):
                psrl_logger.warning(
                    f"[KV]: Cache info RPC failed for instance {i!r}: {result!r}."
                )
                kv_hit_scores[i] = 0
            else:
                info = TrajectoryCacheInfo(**result)
                kv_hit_scores[i] = max(info.gpu_cached_tokens, info.lmcache_cached_tokens)
        route_kwargs["kv_hit_scores"] = kv_hit_scores

    def init_route_strategy(self, **kwargs):
        """Initialize the route strategy for the router.

        Args:
            **kwargs: Keyword arguments for the route strategy.
        """
        if (
            self.config.psrl.routing_strategy.method == "request_num_balance"
            or self.config.psrl.routing_strategy.method == "throughput_optimal"
        ):
            status_required = (
                "Status collection must be enabled when using request num "
                "balance or throughput optimal routing strategy"
            )
            assert self.config.psrl.status_collection.enable, status_required
        n_instances = self.rollout_wg_size
        if self.config.psrl.deployment.heterogeneous_rollout.enable:
            het_config = self.config.psrl.deployment.heterogeneous_rollout
            tp_sizes = het_config.tensor_model_parallel_size_per_instance
            pp_sizes = het_config.pipeline_model_parallel_size_per_instance
            instance_to_tp_pp = {i: f"TP{tp_sizes[i]}_PP{pp_sizes[i]}" for i in range(n_instances)}
        else:
            rollout_config = self.config.gen_actor_rollout_ref.rollout
            instance_to_tp_pp = {
                i: f"TP{rollout_config.tensor_model_parallel_size}_PP{rollout_config.pipeline_model_parallel_size}"
                for i in range(n_instances)
            }
        strategy_kwargs = {
            "logging_interval_in_ms": self.config.psrl.routing_strategy.logging_interval_in_ms,
            "cost_model_path": self.config.psrl.routing_strategy.cost_model_path,
            "instance_to_tp_pp": instance_to_tp_pp,
            "max_num_waiting_reqs_after_preemption": (
                self.config.psrl.routing_strategy.max_num_waiting_reqs_after_preemption
            ),
            "balanced_concurrent_seqs_per_instance": self.balanced_concurrent_seqs_per_instance,
            "max_concurrent_seqs_per_instance": (self.config.psrl.routing_strategy.max_concurrent_seqs_per_instance),
            "delta_throughput_threshold": self.config.psrl.routing_strategy.delta_throughput_threshold,
            "max_prompt_length": self.config.data.max_prompt_length,
            "request_budget": self.config.psrl.routing_strategy.request_budget,
            "snapshot_staleness_threshold_in_ms": self.config.psrl.routing_strategy.snapshot_staleness_threshold_in_ms,
            "logger": psrl_logger,
            **kwargs,
        }
        try:
            route_strategy_class = get_route_strategy_class(self.config.psrl.routing_strategy.method)
            self.route_strategy: RouteStrategyBase = route_strategy_class(n_instances, strategy_kwargs)
            psrl_logger.info(f"Initialized route strategy: {self.config.psrl.routing_strategy.method}")
        except Exception as e:
            psrl_logger.warning(f"Route strategy error: {e}")
            psrl_logger.warning("Falling back to 'round_robin' strategy")
            from psrl.workers.agent_loop.route_strategy import RoundRobinRouteStrategy

            self.route_strategy: RouteStrategyBase = RoundRobinRouteStrategy(n_instances, strategy_kwargs)

    @ray.method(concurrency_group="control")
    async def update_instance_status(self, instance_to_engine_status: dict[int, EngineStats], **kwargs):
        """Update the instance status with latest information from coordinator.

        Args:
            instance_to_engine_status (dict[int, EngineStats]): Latest engine status information.
            **kwargs: Keyword arguments for the update.
        """
        # NOTE(lhy): This method is called by RolloutCoordinator
        # Each agent loop worker contains a RolloutRouter, which shares the same engine status
        # Note that the instance_to_engine_status may be stale and some instances may be absent at beginning

        # Filter out the stale engine status (has a large bias from current engine status)
        filtered_instance_ids = []
        for instance_id, engine_status in instance_to_engine_status.items():
            if self.route_strategy.is_staled(instance_id, engine_status):
                # psrl_logger.warning(f"Instance {instance_id} collected engine status is stale, skipping")
                continue
            filtered_instance_ids.append(instance_id)
        self.route_strategy.update_instance_to_engine_status(
            {instance_id: instance_to_engine_status[instance_id] for instance_id in filtered_instance_ids}
        )

    @ray.method(concurrency_group="control")
    async def update_currently_syncing_instances(self, instance_ids: list[int], ps_model_version: int):
        """Update the currently syncing instances.

        Args:
            instance_ids (List[int]): The instance IDs to update.
            ps_model_version (int): The version of the PS model to update.
        """
        for instance_id in instance_ids:
            self.instance_to_version_after_sync[instance_id] = ps_model_version
        psrl_logger.info(f"Updated currently syncing instances: {instance_ids} to version {ps_model_version}")

    @ray.method(concurrency_group="control")
    async def enter_sticky_session(self, request_id: int):
        """Mark a request as entering sticky session.

        Args:
            request_id (int): The request ID to mark.
        """
        self.sticky_session_requests[request_id] = True

    @ray.method(concurrency_group="control")
    async def exit_sticky_session(self, request_id: int):
        """Mark a request as exiting sticky session.

        Args:
            request_id (int): The request ID to unmark.
        """
        self.sticky_session_requests.pop(request_id, None)

    @ray.method(concurrency_group="control")
    async def pause_instances(self, instance_ids: list[int]):
        """Notify the router about paused instances.

        Args:
            instance_ids (List[int]): List of instance IDs that are paused.
        """
        for instance_id in instance_ids:
            self.currently_paused_instance_ids.add(instance_id)

    @ray.method(concurrency_group="control")
    async def resume_instances(self, instance_ids: list[int]):
        """Notify the router about resumed instances.

        Args:
            instance_ids (List[int]): List of instance IDs that are resumed.
        """
        for instance_id in instance_ids:
            self.currently_paused_instance_ids.discard(instance_id)

    async def _choose_new_rollout_instance(self, request: DataProto) -> int:
        """Select the best rollout instance for handling the generation request.

        Args:
            request (DataProto): The request to be routed.

        Returns:
            int: Index of the selected rollout instance.
        """
        # Ensure the whole routing process is atomic from the PS manager side
        # psrl_logger.info(f"Choosing new rollout instance for request {request.non_tensor_batch['uid'][0]}")
        request_id = request.non_tensor_batch["uid"][0]
        is_validate = request.meta_info.get("validate", False)
        rollout_n = self.val_rollout_n if is_validate else self.rollout_n
        assert "version_tag" in request.non_tensor_batch and request.non_tensor_batch["version_tag"][0] is not None, (
            "Request must have 'version_tag' for routing and it must not be None"
        )
        needed_model_version = request.non_tensor_batch["version_tag"][0]

        # 1. Filter the rollout instances that are not paused and can tolerate the needed staleness of the request
        # This guarantees that the gen worker will have no ahead-of-time version tag when generating
        if self.config.psrl.fuse_rollout_with_validate:
            if is_validate:
                # Val requests can go to any instance (rollout + validate)
                available_instance_ids = set(range(self.rollout_wg_size))
            else:
                # Train/rollout requests can only go to rollout instances, never to
                # validate instances.  This prevents train partial-rollout requests
                # from being stranded when validate instances are paused during
                # switch_to_trainer_mode.
                available_instance_ids = set(range(self.n_rollout_instances))
        else:
            # If not fusing rollout with validate, separate the instance IDs for rollout and validate
            available_instance_ids = set(
                range(self.rollout_wg_size - self.n_validate_instances)
                if not is_validate
                else range(self.rollout_wg_size - self.n_validate_instances, self.rollout_wg_size)
            )
        available_instance_ids = available_instance_ids - self.currently_paused_instance_ids
        candidates = [
            i
            for i, version in self.instance_to_version_after_sync.items()
            if i in available_instance_ids and version >= needed_model_version
        ]
        psrl_logger.debug(
            f"Routing candidates of request {request_id} is {candidates}, where "
            f"available instance: {available_instance_ids}, "
            f"instance_to_version: {self.instance_to_version_after_sync}"
        )

        # 2. If forbidden global migration and the request is a partial rollout request,
        # only consider the specific instance for routing
        if "rollout_instance_id" in request.non_tensor_batch and not self.config.psrl.sync_and_mig_strategy.mig.enable:
            old_instance_id = request.non_tensor_batch["rollout_instance_id"][0]
            assert old_instance_id in candidates, f"Old rollout instance {old_instance_id} is not in the candidates"
            candidates = [old_instance_id]

        # 2.5. If request is in sticky session, keep the existing instance
        if self.sticky_session_requests.get(request_id, False) and "rollout_instance_id" in request.non_tensor_batch:
            old_instance_id = request.non_tensor_batch["rollout_instance_id"][0]
            assert old_instance_id in candidates, f"Sticky session instance {old_instance_id} is not in the candidates"
            candidates = [old_instance_id]

        # 3. If forbidden group sampling on multiple instances, only consider the
        # instance that other requests in the same group are already routed to
        enable_multi_instance_group = self.config.psrl.routing_strategy.enable_group_sampling_on_multi_instances
        if not enable_multi_instance_group:
            group_request_instance_ids = [
                instance_id
                for incomplete_request_id, instance_id in self.incomplete_request_to_instance.items()
                if incomplete_request_id // rollout_n == request_id // rollout_n
            ]
            if len(group_request_instance_ids) > 0:
                first_instance = group_request_instance_ids[0]
                assert all(instance_id == first_instance for instance_id in group_request_instance_ids), (
                    f"All requests in the same group must be routed to "
                    f"the same instance, but found different instances: "
                    f"{group_request_instance_ids}"
                )
                group_instance = group_request_instance_ids[0]
                assert group_instance in candidates, (
                    f"Group request instance {group_instance} of request {request_id} is not in the candidates "
                    f"{candidates}, instance versions: {self.instance_to_version_after_sync}, "
                    f"needed model version: {needed_model_version}"
                )
                candidates = [group_instance]

        # 4. Filter the rollout instances that can reserve the request for the current instance model version
        # This is only used when the needed model version is -1 (i.e. new request)
        if needed_model_version == -1:
            # Snapshot versions before await to avoid race with concurrent sync updates.
            version_snapshot_for_filter = {
                candidate: self.instance_to_version_after_sync[candidate]
                for candidate in candidates
            }
            all_candidate_model_versions = list(set(version_snapshot_for_filter.values()))
            can_reserve_results = await self.ps_manager_handle.can_reserve_request.remote(
                request_id, all_candidate_model_versions, is_validate=is_validate
            )
            candidates = [
                candidate
                for candidate in candidates
                if can_reserve_results[
                    all_candidate_model_versions.index(version_snapshot_for_filter[candidate])
                ]
            ]

        # 5. Provide the indicator list to sort candidates for the route strategy
        candidate_indicator_list = []
        if self.config.psrl.routing_strategy.candidate_sort_indicator == "version":
            for candidate in candidates:
                version = self.instance_to_version_after_sync[candidate]
                # New request: sort by version in descending order
                # Existing request: sort by version in ascending order
                if needed_model_version == -1:
                    version_indicator = -version
                else:
                    version_indicator = version
                candidate_indicator_list.append(version_indicator)
        elif self.config.psrl.routing_strategy.candidate_sort_indicator == "reserve_capability":
            # Use the (reserve_indicator, version) pair as the final indicator.
            # Snapshot the per-candidate versions BEFORE the await so that a
            # concurrent update_currently_syncing_instances cannot invalidate the
            # index lookup.
            candidate_version_snapshot = {
                candidate: self.instance_to_version_after_sync[candidate]
                for candidate in candidates
            }
            all_candidate_model_versions = list(set(candidate_version_snapshot.values()))
            indicator_results = await self.ps_manager_handle.get_reserve_indicator.remote(
                request_id, all_candidate_model_versions, is_validate=is_validate
            )
            for candidate in candidates:
                version = candidate_version_snapshot[candidate]
                if needed_model_version == -1:
                    version_indicator = -version
                else:
                    version_indicator = version
                reserve_indicator = indicator_results[all_candidate_model_versions.index(version)]
                candidate_indicator_list.append((reserve_indicator, version_indicator))
        else:
            raise ValueError(
                f"Invalid candidate sort indicator: {self.config.psrl.routing_strategy.candidate_sort_indicator}"
            )
        route_kwargs = {"candidate_indicator_list": candidate_indicator_list}

        # 6. KV hit score query (only active when method == "kv_cache_aware").
        await self._inject_kv_hit_scores(request_id, candidates, route_kwargs)

        # 7. Strategy-based routing.
        chosen_rollout_instance = self.route_strategy.route(request, candidates=candidates, route_kwargs=route_kwargs)

        # 8. If not None, the request is routed to the chosen rollout instance.
        if chosen_rollout_instance is not None:
            # Allocate the version tag and reserve the request for the chosen
            # rollout instance if the request is not routed before
            not_routed_before = "rollout_instance_id" not in request.non_tensor_batch
            if not_routed_before:
                needed_model_version = self.instance_to_version_after_sync[chosen_rollout_instance]
                request.non_tensor_batch["version_tag"] = np.array([needed_model_version], dtype=int)
                await self.ps_manager_handle.reserve_rollout_instance_requests.remote(
                    rollout_instance_ids=chosen_rollout_instance,
                    request_ids=request_id,
                    model_versions=needed_model_version,
                    is_validate=is_validate,
                )
            # Otherwise, the request is already reserved
            # Only need to update the request instance id
            else:
                await self.ps_manager_handle.update_request_instance_id.remote(
                    request_id=request_id,
                    new_instance_id=chosen_rollout_instance,
                    is_validate=is_validate,
                )
        else:
            pass

        return chosen_rollout_instance

    # TODO(lhy): move this back to router again
    # since log_prob no need to transfered to vllm rollout engine many times (partial rollout)
    @deprecated("It is moved to the `post_process_outputs` inside vllm rollout now")
    def _consolidate_responses(
        self,
        prompts: DataProto,
        vllm_outputs,
    ) -> DataProto:
        """Consolidate VLLM generation outputs with input prompts.

        Args:
            prompts (DataProto): Original input prompts.
            vllm_outputs: Generation outputs from VLLM engine.

        Returns:
            DataProto: Consolidated response data.
        """
        if not isinstance(vllm_outputs, list):
            vllm_outputs = [vllm_outputs]
        assert len(vllm_outputs) == len(prompts), "Mismatched batch size between prompts and VLLM outputs."

        batch_size = len(prompts)
        non_tensor_batch = prompts.non_tensor_batch

        response_ids_list = []
        response_len_list = []
        interrupted_list = []
        all_log_prob_list = []

        for i in range(batch_size):
            vllm_output = vllm_outputs[i]
            assert len(vllm_output.outputs) == 1, "RolloutRouter only supports single request generation."

            response_ids = vllm_output.outputs[0].token_ids
            response_len = len(response_ids)
            interrupted = vllm_output.outputs[0].finish_reason == "abort"

            response_ids_list.append(response_ids)
            response_len_list.append(response_len)
            interrupted_list.append(interrupted)

            log_prob_list = []
            # if inference logprobs is required, we need to collect the log probabilities
            if (
                self.config.psrl.log_prob.enable_rollout_engine_log_prob
                and hasattr(vllm_output.outputs[0], "logprobs")
                and vllm_output.outputs[0].logprobs is not None
            ):
                if self.config.psrl.partial_rollout.interrupt_as_prompt:
                    curr_response_len = non_tensor_batch.get("response_unpadded_len", 0)
                    # Collect log probs only when the request finished normally
                    # The response log probs are collected in two parts:
                    # 1. The log probs of the accumulated response tokens (in current prompt tokens)
                    # 2. The log probs of the current response tokens
                    if not interrupted and curr_response_len > 0:
                        # partial response log probs from prompt log probs
                        prompt_token_ids = vllm_output.prompt_token_ids
                        for i, logprob in enumerate(vllm_output.prompt_logprobs[-curr_response_len:]):
                            log_prob_list.append(logprob[prompt_token_ids[i - curr_response_len]].logprob)
                        # new response log probs from decode log probs
                        for i, logprob in enumerate(vllm_output.outputs[0].logprobs):
                            log_prob_list.append(logprob[response_ids[i]].logprob)
                else:
                    # Response log probs from decode log probs
                    for i, logprob in enumerate(vllm_output.outputs[0].logprobs):
                        log_prob_list.append(logprob[response_ids[i]].logprob)
            all_log_prob_list.append(log_prob_list)

        # Consolidate batch results
        assert all(isinstance(ids, (list, tuple)) for ids in response_ids_list), (
            f"response_ids_list elements must be lists, got types: "
            f"{[type(ids) for ids in response_ids_list]}"
        )
        if "raw_response_ids" in non_tensor_batch:
            raw_response_ids = non_tensor_batch.pop("raw_response_ids")
            assert isinstance(raw_response_ids, np.ndarray), (
                f"raw_response_ids should be np.ndarray, got {type(raw_response_ids)}"
            )
            raw_response_ids = np.fromiter(raw_response_ids.tolist(), dtype=object)
        else:
            raw_response_ids = np.fromiter(([] for _ in range(batch_size)), dtype=object)

        raw_response_ids = raw_response_ids + np.fromiter(response_ids_list, dtype=object)
        assert isinstance(raw_response_ids, np.ndarray) and raw_response_ids.dtype == object, (
            f"raw_response_ids after concat should be np.ndarray(dtype=object), "
            f"got type={type(raw_response_ids)}, dtype={getattr(raw_response_ids, 'dtype', None)}"
        )
        non_tensor_batch["raw_response_ids"] = raw_response_ids

        if "response_unpadded_len" in non_tensor_batch:
            curr_response_unpadded_len = non_tensor_batch["response_unpadded_len"]
        else:
            curr_response_unpadded_len = [0] * batch_size
        response_unpadded_len = [curr_response_unpadded_len[i] + response_len_list[i] for i in range(batch_size)]
        non_tensor_batch["response_unpadded_len"] = np.array(response_unpadded_len, dtype=int)
        non_tensor_batch["interrupted"] = np.array(interrupted_list, dtype=bool)

        # Update rollout_log_probs
        if self.config.psrl.log_prob.enable_rollout_engine_log_prob:
            if "rollout_log_probs" in non_tensor_batch:
                curr_rollout_log_probs = non_tensor_batch.pop("rollout_log_probs")
                curr_rollout_log_probs = np.fromiter(curr_rollout_log_probs.tolist(), dtype=object)
            else:
                curr_rollout_log_probs = np.fromiter(([] for _ in range(batch_size)), dtype=object)
            curr_rollout_log_probs = curr_rollout_log_probs + np.fromiter(all_log_prob_list, dtype=object)
            non_tensor_batch["rollout_log_probs"] = curr_rollout_log_probs

        batch = TensorDict(
            {
                "input_ids": prompts.batch["input_ids"],
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)

    @rollout_trace_op
    async def generate_async(
        self,
        request: DataProto,
    ) -> DataProto:
        """Asynchronously generate response for a single request.

        Args:
            request (DataProto): Single generation request.

        Returns:
            DataProto or None: Generated result or None if request is invalid.
        """
        assert len(request) == 1, "RolloutRouter only supports single request generation."
        if self.scheduler_task is None:
            if self.config.psrl.routing_strategy.enable_multi_priority_queue:
                task_coro = self._multi_priority_queue_routing_loop()
                self.scheduler_task = asyncio.create_task(task_coro)
            else:
                task_coro = self._single_priority_queue_routing_loop()
                self.scheduler_task = asyncio.create_task(task_coro)
            # To avoid silent error in async tasks
            self.scheduler_task.add_done_callback(lambda f: f.result())
            psrl_logger.info("Started routing loop")

        request_id = request.non_tensor_batch["uid"][0]
        # Register current token sequence for KV-aware routing (START of turn).
        raw_prompt_ids = request.non_tensor_batch.get("raw_prompt_ids")
        self.request_to_tokens[request_id] = _to_token_list(raw_prompt_ids[0])
        
        is_validate = request.meta_info.get("validate", False)
        update_status_success = await self.ps_manager_handle.update_request_status.remote(
            [request_id],
            PSRL_RequestStatus.ROLLOUT_ROUTING,
            is_validate=is_validate,
        )
        if not update_status_success[0]:
            # Means the request is aborted
            psrl_logger.warning("generate_async: request %s aborted at ROLLOUT_ROUTING.", request_id)
            return None

        # Create a future to track this request's completion
        result_future = asyncio.Future()
        # Store the future in a way that the scheduler can access it
        self.request_futures[request_id] = result_future
        # Add request to priority queue
        self.requests_to_route.put(request)
        # psrl_logger.info(f"Adding request {request_id} to priority queue")
        # Wait for the request to be processed
        with log_dual_events(
            f"Routing request {request_id} and waiting for it to be processed",
            psrl_logger,
            level=logging.DEBUG,
            event_type=EventType.GEN,
        ):
            result = await result_future
        # Clean up the future
        self.request_futures.pop(request_id)
        if result is None:
            psrl_logger.warning("generate_async: request %s got None from GenWorker.", request_id)
        return result

    def _set_result(self, request_id: int, result: DataProto | None):
        """Set the result for a request.

        Args:
            request_id: The id of the request.
            result: The result to set.
        """
        assert request_id in self.request_futures, f"Request {request_id} should be in request futures"
        assert not self.request_futures[request_id].done(), f"Request {request_id} should not be done"
        self.request_futures[request_id].set_result(result)
        self.incomplete_request_to_instance.pop(request_id, None)
        self.sticky_session_requests.pop(request_id, None)

    def _set_exception(self, request_id: int, exc: BaseException):
        """Propagate a generation exception to the awaiting `generate_async` caller.

        Used when the underlying rollout raises (e.g. ``VLLMValidationError``):
        the awaiting caller will see ``await result_future`` re-raise ``exc``
        instead of receiving a sentinel value, so the agent loop can map it
        to ``TerminateReason.ROLLOUT_ERROR`` and clean up.
        """
        assert request_id in self.request_futures, f"Request {request_id} should be in request futures"
        assert not self.request_futures[request_id].done(), f"Request {request_id} should not be done"
        self.request_futures[request_id].set_exception(exc)
        self.incomplete_request_to_instance.pop(request_id, None)
        self.sticky_session_requests.pop(request_id, None)

    def _resolve_uid(self, uid: int) -> tuple[int | None, list[int] | None]:
        """
        Resolve the instance_id and token sequence for a trajectory uid.

        Checks `incomplete_request_to_instance` first (turn in-flight), then
        `request_to_last_routed_instance` (between turns).

        Args:
            uid (int): Trajectory unique identifier.

        Returns:
            tuple[int | None, list[int] | None]: `(instance_id, tokens)`, or
                `(None, None)` if uid is not found in either registry.
        """
        # NOTE(claude): Cannot use `get(uid) or get(uid)` because instance_id=0
        # is falsy in Python, which would incorrectly skip instance 0.
        if uid in self.incomplete_request_to_instance:
            instance_id = self.incomplete_request_to_instance[uid]
        elif uid in self.request_to_last_routed_instance:
            instance_id = self.request_to_last_routed_instance[uid]
        else:
            return None, None
        tokens = self.request_to_tokens.get(uid)
        return instance_id, tokens

    def is_routing(self) -> bool:
        """Check if the router is currently routing requests."""
        return self._is_routing

    @ray.method(concurrency_group="control")
    async def pause_routing(self):
        """Interrupt the routing."""
        self._pause_routing = True
        # NOTE(lhy): asyncio lock cannot be shared across concurrency groups (each group has its
        # own event loop). Poll _is_routing instead because a plain bool read/write is safe across threads
        # under CPython's GIL.
        while self._is_routing:
            await asyncio.sleep(self.config.psrl.routing_strategy.check_interval_in_ms / 1000)
        psrl_logger.info("Pausing routing")

    @ray.method(concurrency_group="control")
    async def resume_routing(self):
        """Resume the routing."""
        self._pause_routing = False
        psrl_logger.info("Resuming routing")

    async def _single_priority_queue_routing_loop(self):
        """Continuous routing loop for a single priority queue.

        This loop processes requests from the single priority queue.
        """
        psrl_logger.info("Started single priority queue routing loop")
        while True:
            self._is_routing = False
            async with (
                AsyncBusyPollingRayLock(self.ps_manager_handle),
            ):
                while not self.requests_to_route.empty() and not self._pause_routing:
                    self._is_routing = True
                    request = self.requests_to_route.pop()
                    assert request is not None, "Request should not be None in priority queue"
                    request_id = request.non_tensor_batch["uid"][0]
                    assert request_id in self.request_futures, f"Request {request_id} should be in request futures"
                    if await self.ps_manager_handle.check_aborted_requests.remote(request_id, remove=True):
                        self._set_result(request_id, None)
                        continue
                    # time_begin = time.time()
                    new_instance_id = await self._choose_new_rollout_instance(request)
                    # time_end = time.time()
                    # psrl_logger.info(f"Choosing rollout instance for request {request_id} to {new_instance_id} in {time_end - time_begin} seconds") # noqa: E501
                    if new_instance_id is None:
                        # new_instance_id is None indicates that we cannot find a suitable rollout instance
                        # for the request due to the current engine status (e.g.,
                        # version staleness, instance overload).
                        # Need to wait for engine status update to try again:
                        # 1. The overall engine status could be updated by the
                        #    coordinator periodically.
                        # 2. The engine status of the specific instance could be
                        #    updated after one request is added/completed.
                        self.requests_to_route.put(request)
                        break
                    self.incomplete_request_to_instance[request_id] = new_instance_id
                    self.request_to_last_routed_instance[request_id] = new_instance_id
                    task_coro = self._route_single_request(request, new_instance_id)
                    task = asyncio.create_task(task_coro)
                    # To avoid silent error in async tasks
                    task.add_done_callback(lambda f: f.result())
            self._is_routing = False
            sleep_time = self.config.psrl.routing_strategy.check_interval_in_ms / 1000
            await asyncio.sleep(sleep_time)

    async def _multi_priority_queue_routing_loop(self):
        """Continuous routing loop for multiple priority queues.

        This loop processes requests from the multiple priority queues.
        """
        psrl_logger.info("Started multi priority queue routing loop")
        while True:
            # Process all requests in the multiple priority queues
            self._is_routing = False
            route_num = 0
            begin_time = time.time()
            async with (
                AsyncBusyPollingRayLock(self.ps_manager_handle),
            ):
                self.requests_to_route.remove_empty_queues()
                remain_requests = []
                for queue_id, request_queue in self.requests_to_route.iter_queues():
                    if len(remain_requests) != 0:
                        # Method 1: If the last queue still has requests, we will not process the other queues
                        break
                        # Method 2: Try to process the other queues
                        # remain_requests.clear()
                    while not request_queue.empty() and not self._pause_routing:
                        request = request_queue.pop()
                        assert request is not None, "Request should not be None in priority queue"
                        request_id = request.non_tensor_batch["uid"][0]
                        assert request_id in self.request_futures, f"Request {request_id} should be in request futures"
                        if await self.ps_manager_handle.check_aborted_requests.remote(request_id, remove=True):
                            self._set_result(request_id, None)
                            continue
                        new_instance_id = await self._choose_new_rollout_instance(request)
                        if new_instance_id is None:
                            # new_instance_id is None indicates that we cannot find a suitable rollout instance
                            # for the request due to the current engine status (e.g.,
                            # version staleness, instance overload).
                            # Need to wait for engine status update to try again:
                            # 1. The overall engine status could be updated by the
                            #    coordinator periodically.
                            # 2. The engine status of the specific instance could be
                            #    updated after one request is added/completed.
                            remain_requests.append(request)
                            break
                        self.incomplete_request_to_instance[request_id] = new_instance_id
                        self.request_to_last_routed_instance[request_id] = new_instance_id
                        # Create a task to process this request
                        task_coro = self._route_single_request(request, new_instance_id)
                        task = asyncio.create_task(task_coro)
                        # To avoid silent error in async tasks
                        task.add_done_callback(lambda f: f.result())
                        route_num += 1
                    for request in remain_requests:
                        request_queue.put(request)
            self._is_routing = False
            sleep_time = self.config.psrl.routing_strategy.check_interval_in_ms / 1000
            if route_num > 0:
                psrl_logger.debug(
                    f"Routing {route_num} requests in multi priority queue "
                    f"routing loop, time cost: {time.time() - begin_time} seconds"
                )
            await asyncio.sleep(sleep_time)

    def _profiling_accumulate_and_overwrite(
        self,
        consolidated_output: DataProto,
        request: DataProto,
        reroute_trigger: str,
    ) -> None:
        """
        Accumulate profiling records from the current engine call and overwrite
        the transient RESUME trigger.

        Called when a request is rerouted (preempt_resume or partial_rollout_resume).
        Records are stored in non_tensor_batch for persistence across reroutes.

        Args:
            consolidated_output (DataProto): Output from the current engine call.
            request (DataProto): The original request (contains accumulated records
                from previous reroutes, if any).
            reroute_trigger (str): The trigger to assign
                ("preempt_resume" or "partial_rollout_resume").
        """
        enable_profiling = self.config.psrl.profile.trajectory.enable
        if not enable_profiling:
            return

        ntb = consolidated_output.non_tensor_batch

        # Get current call's records
        curr_prefill = ntb.get("profiling_prefill_records", [None])[0]
        curr_decode = ntb.get("profiling_decode_records", [None])[0]
        if curr_prefill is None or len(curr_prefill) == 0:
            return

        # Determine trigger for the first record of this engine call.
        # If this is the first engine call (no accumulated records yet), use INITIAL.
        # Otherwise use the reroute trigger.
        accumulated_prefill = request.non_tensor_batch.get(
            "_profiling_accumulated_prefill_records", [None]
        )[0]
        if accumulated_prefill is None or len(accumulated_prefill) == 0:
            curr_prefill[0]["trigger"] = "initial"
        else:
            curr_prefill[0]["trigger"] = reroute_trigger

        # Compute router_wait_s for rerouted calls.
        reroute_submit_ts = float(
            request.non_tensor_batch.get("_profiling_reroute_submit_ts", [0.0])[0]
        )
        gen_start_ts = float(ntb.get("profiling_generation_start_wall_ts", [0.0])[0])
        if reroute_submit_ts > 0 and gen_start_ts > 0:
            curr_prefill[0]["router_wait_s"] = max(gen_start_ts - reroute_submit_ts, 0.0)

        # Accumulate into consolidated_output's non_tensor_batch (persists across reroutes).
        if accumulated_prefill is None:
            accumulated_prefill = []
        accumulated_prefill = list(accumulated_prefill) + list(curr_prefill)

        accumulated_decode = request.non_tensor_batch.get(
            "_profiling_accumulated_decode_records", [None]
        )[0]
        if accumulated_decode is None:
            accumulated_decode = []
        if curr_decode is not None and len(curr_decode) > 0:
            accumulated_decode = list(accumulated_decode) + list(curr_decode)

        consolidated_output.non_tensor_batch["_profiling_accumulated_prefill_records"] = np.array(
            [accumulated_prefill], dtype=object
        )
        consolidated_output.non_tensor_batch["_profiling_accumulated_decode_records"] = np.array(
            [accumulated_decode], dtype=object
        )

        # Save first generation start ts if this is the first call.
        first_start_ts = float(
            request.non_tensor_batch.get("_profiling_first_gen_start_ts", [0.0])[0]
        )
        if first_start_ts == 0.0:
            first_start_ts = gen_start_ts
        consolidated_output.non_tensor_batch["_profiling_first_gen_start_ts"] = np.array(
            [first_start_ts], dtype=float
        )

        # Record reroute submit timestamp for the next call's router_wait computation.
        consolidated_output.non_tensor_batch["_profiling_reroute_submit_ts"] = np.array(
            [time.time()], dtype=float
        )

        # Record the reroute trigger so _profiling_finalize knows what to use.
        consolidated_output.non_tensor_batch["_profiling_last_reroute_trigger"] = np.array(
            [reroute_trigger], dtype=object
        )

    def _profiling_finalize(
        self,
        consolidated_output: DataProto,
        request: DataProto,
    ) -> None:
        """
        Finalize profiling data when a model turn completes.

        Merges any accumulated records from previous reroutes with the final
        engine call's records and overwrites the RESUME trigger on the first record.

        Args:
            consolidated_output (DataProto): Output from the final engine call.
            request (DataProto): The original request.
        """
        enable_profiling = self.config.psrl.profile.trajectory.enable
        if not enable_profiling:
            return

        ntb = consolidated_output.non_tensor_batch

        # Get final call's records.
        curr_prefill = ntb.get("profiling_prefill_records", [None])[0]
        curr_decode = ntb.get("profiling_decode_records", [None])[0]

        # Get accumulated records from previous reroutes.
        accumulated_prefill = request.non_tensor_batch.get(
            "_profiling_accumulated_prefill_records", [None]
        )[0]
        accumulated_decode = request.non_tensor_batch.get(
            "_profiling_accumulated_decode_records", [None]
        )[0]

        if accumulated_prefill is not None and len(accumulated_prefill) > 0:
            # This is a rerouted request — merge accumulated + current.
            if curr_prefill is not None and len(curr_prefill) > 0:
                # Use the stored trigger from the last reroute.
                last_reroute_trigger = str(
                    request.non_tensor_batch.get(
                        "_profiling_last_reroute_trigger", ["preempt_resume"]
                    )[0]
                )
                curr_prefill[0]["trigger"] = last_reroute_trigger
                # Compute router_wait for this final rerouted call.
                reroute_submit_ts = float(
                    request.non_tensor_batch.get("_profiling_reroute_submit_ts", [0.0])[0]
                )
                gen_start_ts = float(ntb.get("profiling_generation_start_wall_ts", [0.0])[0])
                if reroute_submit_ts > 0 and gen_start_ts > 0:
                    curr_prefill[0]["router_wait_s"] = max(gen_start_ts - reroute_submit_ts, 0.0)

                all_prefill = list(accumulated_prefill) + list(curr_prefill)
            else:
                all_prefill = list(accumulated_prefill)

            all_decode = list(accumulated_decode) if accumulated_decode is not None and len(accumulated_decode) > 0 else []
            if curr_decode is not None and len(curr_decode) > 0:
                all_decode = all_decode + list(curr_decode)

            # Use the first generation start ts from the first call.
            first_start_ts = float(
                request.non_tensor_batch.get("_profiling_first_gen_start_ts", [0.0])[0]
            )

            ntb["profiling_prefill_records"] = np.array([all_prefill], dtype=object)
            ntb["profiling_decode_records"] = np.array([all_decode], dtype=object)
            if first_start_ts > 0:
                ntb["profiling_generation_start_wall_ts"] = np.array(
                    [first_start_ts], dtype=float
                )
        else:
            # First (and only) engine call — just overwrite RESUME to INITIAL.
            if curr_prefill is not None and len(curr_prefill) > 0:
                curr_prefill[0]["trigger"] = "initial"

        # Clean up internal accumulation fields.
        for key in [
            "_profiling_accumulated_prefill_records",
            "_profiling_accumulated_decode_records",
            "_profiling_first_gen_start_ts",
            "_profiling_reroute_submit_ts",
            "_profiling_last_reroute_trigger",
        ]:
            ntb.pop(key, None)
            request.non_tensor_batch.pop(key, None)

    async def _maybe_migrate_kv(self, request: DataProto, new_instance_id: int) -> None:
        """
        Attempt to migrate accumulated KV cache from the previous instance to the new one.

        When a request is re-routed to a different instance (typically due to partial
        rollout interruption or load-balancing migration), this method transfers the
        prefix KV cache from the old instance's LMCache CPU backend to the new
        instance's CPU backend via the shared LMCache Controller (`/move`).  The new
        instance's LMCache then automatically loads the CPU-cached KV into GPU when
        the request is next scheduled, skipping re-prefill.

        Flow:
            1. Guard: return early if kv_migration is disabled, same instance, or no
               previous instance recorded (fresh request + partial_rollout_only=True).
            2. Query old instance for cached token count; skip if nothing cached.
            3. Pin CPU cache on old instance to prevent LRU eviction during transfer.
            4. Fire the Controller /move (A_CPU --NIXL/UCX--> B_CPU).
            5. Unpin old instance CPU cache immediately after transfer, in a
               `finally` block so the pin is always released on success or failure.

        In "async" transfer_mode the transfer runs in the background while the
        request is dispatched immediately; pair with `lmcache.enable_async_loading`
        to overlap CPU-to-GPU retrieval with prefill computation on instance B.

        Args:
            request (DataProto): The request being routed (single-element batch).
            new_instance_id (int): The instance the request will be dispatched to.
        """
        kv_migration_cfg = self.config.psrl.routing_strategy.get("kv_migration")
        if kv_migration_cfg is None or not kv_migration_cfg.get("enable", False):
            return

        request_id = request.non_tensor_batch["uid"][0]

        # Determine the old instance: prefer the rollout_instance_id already written
        # into the request (set on first routing and updated on each re-route), then
        # fall back to request_to_last_routed_instance (persists across turns).
        if "rollout_instance_id" in request.non_tensor_batch:
            old_instance_id = int(request.non_tensor_batch["rollout_instance_id"][0])
        else:
            old_instance_id = self.request_to_last_routed_instance.get(request_id)

        partial_rollout_only: bool = kv_migration_cfg.get("partial_rollout_only", True)
        if old_instance_id is None:
            # No previous instance recorded yet (fresh request).
            if partial_rollout_only:
                return
            # When partial_rollout_only=False, there is nothing to migrate for fresh
            # requests anyway (no KV has been produced yet), so always return.
            return

        if old_instance_id == new_instance_id:
            return  # Same instance: no migration needed.

        tokens = self.request_to_tokens.get(request_id)
        if not tokens:
            return

        # --- Step 1: Check whether old instance has anything cached. ---
        try:
            cache_info_raw = await asyncio.wait_for(
                self.rollout_wg_list[old_instance_id].execute_rank_zero_async(
                    "kv_get_cache_info", tokens
                ),
                timeout=5.0,
            )
        except Exception as e:
            psrl_logger.warning(
                f"[KVMigration] Cache info query failed for uid={request_id!r} "
                f"on instance {old_instance_id!r}: {e!r}. Skipping migration."
            )
            return

        if cache_info_raw is None or cache_info_raw.get("lmcache_cached_tokens", 0) == 0:
            psrl_logger.debug(
                f"[KVMigration] No LMCache CPU data for uid={request_id!r} "
                f"on instance {old_instance_id!r}. Skipping migration."
            )
            return

        # --- Step 2: Pin src CPU to prevent LRU eviction during transfer. ---
        try:
            await asyncio.wait_for(
                self.rollout_wg_list[old_instance_id].execute_rank_zero_async(
                    "kv_pin", tokens, ["backend"]
                ),
                timeout=5.0,
            )
        except Exception as e:
            psrl_logger.warning(
                f"[KVMigration] CPU pin failed for uid={request_id!r} "
                f"on instance {old_instance_id!r}: {e!r}. Skipping migration."
            )
            return

        src_id = f"psrl_instance_{old_instance_id}"
        dst_id = f"psrl_instance_{new_instance_id}"
        transfer_mode: str = kv_migration_cfg.get("transfer_mode", "async")
        timeout_s: float = kv_migration_cfg.get("transfer_timeout_ms", 5000) / 1000

        # --- Step 3: Transfer (pin → transfer → unpin in finally). ---
        async def _transfer_and_unpin() -> None:
            try:
                success = await self.rollout_wg_list[old_instance_id].execute_rank_zero_async(
                    "kv_transfer",
                    tokens,
                    (src_id, "LocalCPUBackend"),
                    (dst_id, "LocalCPUBackend"),
                    False,  # copy=False → move semantics
                )
                if success:
                    psrl_logger.debug(
                        f"[KVMigration] Transfer succeeded for uid={request_id!r}: "
                        f"instance {old_instance_id!r} -> {new_instance_id!r}."
                    )
                else:
                    psrl_logger.warning(
                        f"[KVMigration] Transfer returned False for uid={request_id!r}. "
                        f"Instance {new_instance_id!r} will re-prefill."
                    )
            except Exception as e:
                psrl_logger.warning(
                    f"[KVMigration] Transfer failed for uid={request_id!r}: {e!r}. "
                    f"Instance {new_instance_id!r} will re-prefill."
                )
            finally:
                # Unpin immediately after transfer completes (or fails).
                # /move success: B already has the data in its own CPU backend, A no longer needs the pin.
                # /move failure: No migration occurred; release the pin so A's LRU can run freely.
                try:
                    await asyncio.wait_for(
                        self.rollout_wg_list[old_instance_id].execute_rank_zero_async(
                            "kv_unpin", tokens, ["backend"]
                        ),
                        timeout=5.0,
                    )
                except Exception as e:
                    psrl_logger.warning(
                        f"[KVMigration] Post-transfer unpin failed for uid={request_id!r}: {e!r}."
                    )

        if transfer_mode == "sync":
            try:
                await asyncio.wait_for(_transfer_and_unpin(), timeout=timeout_s)
            except asyncio.TimeoutError:
                psrl_logger.warning(
                    f"[KVMigration] Sync transfer timed out after {timeout_s}s "
                    f"for uid={request_id!r}. Dispatch proceeds; instance {new_instance_id!r} "
                    "will re-prefill."
                )
        else:
            # async: fire-and-forget — dispatch immediately, transfer runs in background.
            task = asyncio.create_task(_transfer_and_unpin())
            # Surface any unhandled exception to the log rather than silencing it.
            task.add_done_callback(lambda f: f.result() if not f.cancelled() else None)

    async def _route_single_request(self, request: DataProto, new_instance_id: int):
        """Route a single request to a rollout instance.

        Args:
            request (DataProto): The request to process.
            new_instance_id (int): The new rollout instance id that the request
                will be routed to.
        """
        request_id = request.non_tensor_batch["uid"][0]
        try:
            await self._route_single_request_impl(request, new_instance_id)
        except BaseException as exc:
            # Last-resort safety net: if anything inside the routing path raises
            # an unhandled exception, the awaiting `result_future` would otherwise
            # never resolve, and the caller would block until its own timeout
            # (e.g. 600s in the agent loop). Surface the exception to the caller
            # so it can clean up immediately instead of hanging.
            psrl_logger.error(
                f"_route_single_request crashed for request {request_id} on "
                f"instance {new_instance_id}: {exc!r}. Propagating to caller.",
                exc_info=True,
            )
            if (
                request_id in self.request_futures
                and not self.request_futures[request_id].done()
            ):
                self._set_exception(request_id, exc)
            if not isinstance(exc, Exception):
                # Re-raise CancelledError / KeyboardInterrupt / SystemExit etc.
                raise

    async def _route_single_request_impl(self, request: DataProto, new_instance_id: int):
        """Implementation body of `_route_single_request`; see that method for docs."""
        request_id = request.non_tensor_batch["uid"][0]
        is_validate = request.meta_info.get("validate", False)
        rollout_n = self.val_rollout_n if is_validate else self.rollout_n
        request.non_tensor_batch["rollout_instance_id"] = np.array([new_instance_id], dtype=int)
        assert "version_tag" in request.non_tensor_batch, "Request must have 'version_tag' for routing"
        needed_model_version = request.non_tensor_batch["version_tag"][0]
        version_tag_error = (
            "The version tag should not be -1 (new request that is not "
            "allocated a version tag yet when enabled dynamic version tag) "
            "after routing"
        )
        assert needed_model_version != -1, version_tag_error
        # Update request status
        # psrl_logger.info(
        #     f"Updating request {request_id} status to "
        #     f"ROLLOUT_DISPATCHED with version tag {needed_model_version}"
        # )
        update_status_success = await self.ps_manager_handle.update_request_status.remote(
            [request_id],
            PSRL_RequestStatus.ROLLOUT_DISPATCHED,
            model_version=request.non_tensor_batch["version_tag"].tolist(),
            is_validate=request.meta_info.get("validate", False),
        )
        # psrl_logger.info(
        #     f"Update request {request_id} status to "
        #     f"ROLLOUT_DISPATCHED success: {update_status_success[0]}"
        # )

        if update_status_success[0]:
            # Change engine status
            self.route_strategy.push_request(request, new_instance_id)
            # Add request to inflight request ids for the instance
            self.instance_to_inflight_request_ids[new_instance_id].append(request_id)

            # Set sampling params
            rollout_config = self.config.gen_actor_rollout_ref.rollout
            sampling_params = dict(
                n=1,
                logprobs=0,  # can be set to 0 and let actor to recompute
                temperature=rollout_config.temperature,
                top_p=rollout_config.top_p,
                repetition_penalty=rollout_config.get("repetition_penalty", 1.0),
                output_kind=RequestOutputKind.CUMULATIVE,
                detokenize=False,
            )

            # override sampling params for validation
            if request.meta_info.get("validate", False):
                val_config = self.config.train_actor_rollout_ref.rollout.val_kwargs
                sampling_params["top_k"] = val_config.top_k
                sampling_params["top_p"] = val_config.top_p
                sampling_params["temperature"] = val_config.temperature

            # Attempt KV cache migration from the previous instance to this one,
            # so the new instance can skip re-prefill for the accumulated prefix.
            await self._maybe_migrate_kv(request, new_instance_id)

            # Generate response
            # psrl_logger.info(f"Generating response for request {request_id} on instance {new_instance_id}")
            try:
                consolidated_output, update_status = await self.rollout_wg_list[new_instance_id].execute_rank_zero_async(
                    "generate_async", request, sampling_params
                )
            except Exception as gen_exc:
                # Propagate the exception to the awaiting generate_async() so the
                # agent loop maps it to TerminateReason.ROLLOUT_ERROR. We do NOT
                # collapse it into _set_result(..., None) -- that channel now
                # exclusively means "intentionally aborted".
                psrl_logger.error(
                    f"Generation failed for request {request_id} on instance {new_instance_id}: {gen_exc}. "
                    "Propagating exception to the awaiting caller as ROLLOUT_ERROR."
                )
                self.route_strategy.pop_request(request, new_instance_id)
                self.instance_to_inflight_request_ids[new_instance_id].remove(request_id)
                if request_id in self.request_futures and not self.request_futures[request_id].done():
                    self._set_exception(request_id, gen_exc)
                return

            # Change engine status
            self.route_strategy.pop_request(request, new_instance_id)
            # Remove request from inflight request ids for the instance
            self.instance_to_inflight_request_ids[new_instance_id].remove(request_id)

            need_reroute = False
            # Check if request was interrupted and needs to be requeued
            if update_status == PSRL_RequestStatus.ROLLOUT_INTERRUPTED_BY_SCHEDULER:
                psrl_logger.debug(
                    f"Request {request_id} on instance {new_instance_id} was interrupted "
                    "by scheduler (most likely due to kv cache full and preemption), requeueing"
                )
                # Put back in priority queue for partial rollout
                # Ensure that the consolidated output has the rollout instance id recorded
                consolidated_output.non_tensor_batch["rollout_instance_id"] = np.array([new_instance_id], dtype=int)
                self._profiling_accumulate_and_overwrite(
                    consolidated_output, request, "preempt_resume"
                )
                self.requests_to_route.put(consolidated_output)
                # No result to set since the request is not completed
                need_reroute = True
            elif update_status == PSRL_RequestStatus.ROLLOUT_INTERRUPTED:
                psrl_logger.debug(
                    f"Request {request_id} on instance {new_instance_id} was interrupted "
                    "(due to model synchronization when enabled partial rollout), requeueing"
                )
                # Put back in priority queue for partial rollout
                # Ensure that the consolidated output has the rollout instance id recorded
                consolidated_output.non_tensor_batch["rollout_instance_id"] = np.array([new_instance_id], dtype=int)
                self._profiling_accumulate_and_overwrite(
                    consolidated_output, request, "partial_rollout_resume"
                )
                self.requests_to_route.put(consolidated_output)
                # No result to set since the request is not completed
                need_reroute = True
            elif update_status == PSRL_RequestStatus.ROLLOUT_COMPLETED:
                response_len = consolidated_output.non_tensor_batch["response_unpadded_len"][0]
                parent_prompt_id = request_id // rollout_n
                psrl_logger.debug(
                    f"Request {request_id} on instance {new_instance_id} of "
                    f"parent prompt {parent_prompt_id} completed successfully, "
                    f"length is {response_len}"
                )
                self._profiling_finalize(consolidated_output, request)
                result = consolidated_output
            else:
                # Means the request is aborted
                assert update_status is None, "The update status should be None if the request is aborted"
                # psrl_logger.info(f"Request {request_id} on instance {new_instance_id} is aborted")
                result = None
        else:
            # Means the request is aborted before reaching execute_rank_zero_async,
            # so consolidated_output was never produced.
            # psrl_logger.info(f"Request {request_id} is aborted")
            result = None
            consolidated_output = None
            need_reroute = False

        # Update token sequence with prompt + response for next turn's KV routing (END of turn).
        if consolidated_output is not None:
            raw_prompt_ids = consolidated_output.non_tensor_batch.get("raw_prompt_ids")
            raw_response_ids = consolidated_output.non_tensor_batch.get("raw_response_ids")
            prompt_tokens = _to_token_list(raw_prompt_ids[0])
            response_tokens = _to_token_list(raw_response_ids[0])
            self.request_to_tokens[request_id] = prompt_tokens + response_tokens

        # Set the result for the request
        if not need_reroute:
            self._set_result(request_id, result)

    @ray.method(concurrency_group="control")
    async def check_should_migrate(self) -> list[int]:
        """Check which instances should be interrupted to migrate to others due to starvation.

        Returns:
            List[int]: The instance IDs that should be interrupted to migrate.
        """
        # psrl_logger.info("Checking which instances should be interrupted to migrate to others due to starvation")
        instance_to_status = self.route_strategy.instance_to_engine_status
        filtered_instance_ids = []
        # Filter instances that can be routed to (version is not aborted)
        # but no requests in the priority queue can be routed to it.
        # In this case, the instance is starving
        # and we may migrate requests from other instances to it.
        async with AsyncBusyPollingRayLock(self.ps_manager_handle):
            for instance_id in range(self.rollout_wg_size):
                if instance_to_status[instance_id].get_waiting_queue_size() != 0:
                    continue
                instance_version = self.instance_to_version_after_sync[instance_id]
                if await self.ps_manager_handle.check_aborted_model_versions.remote(instance_version):
                    continue

                def version_filter(request, instance_version=instance_version):
                    assert "version_tag" in request.non_tensor_batch, (
                        "Request must have 'version_tag' for checking version"
                    )
                    request_version = request.non_tensor_batch["version_tag"][0]
                    return request_version <= instance_version

                filtered_requests = self.requests_to_route.filter_by_condition(version_filter)
                filtered_request_ids = [request.non_tensor_batch["uid"][0] for request in filtered_requests]

                if len(filtered_request_ids) > 0:
                    is_aborted = await self.ps_manager_handle.check_aborted_requests.remote(
                        filtered_request_ids, remove=False
                    )
                    # Keep `is_validate` aligned with `filtered_request_ids` (same indices as `filtered_requests`).
                    kept_indices = [i for i, aborted in enumerate(is_aborted) if not aborted]
                    filtered_request_ids = [filtered_request_ids[i] for i in kept_indices]
                    is_validate_list = [
                        filtered_requests[i].meta_info.get("validate", False) for i in kept_indices
                    ]
                    if len(filtered_request_ids) > 0:
                        can_reserve = await self.ps_manager_handle.can_reserve_request.remote(
                            filtered_request_ids,
                            [instance_version],
                            without_new_reserve_entry=False,
                            is_validate=is_validate_list,
                        )
                        filtered_request_ids = [
                            request_id
                            for i, request_id in enumerate(filtered_request_ids)
                            if can_reserve[i] == [True]
                        ]
                if len(filtered_request_ids) == 0:
                    filtered_instance_ids.append(instance_id)

        candidate_migrate_instance_ids = []  # (instance_id, ratio)
        for starved_instance_id in filtered_instance_ids:
            for instance_id in range(self.rollout_wg_size):
                if instance_id == starved_instance_id:
                    continue
                if (
                    self.instance_to_version_after_sync[instance_id]
                    > self.instance_to_version_after_sync[starved_instance_id]
                ):
                    continue

                if self.config.psrl.sync_and_mig_strategy.mig.indicator == "request_num":
                    request_num = instance_to_status[instance_id].get_waiting_and_running_queue_size()
                    starved_request_num = instance_to_status[starved_instance_id].get_waiting_and_running_queue_size()
                    if starved_request_num == 0:
                        ratio = float("inf") if request_num > 0 else 1
                    else:
                        ratio = request_num / starved_request_num
                elif self.config.psrl.sync_and_mig_strategy.mig.indicator == "throughput":
                    throughput = instance_to_status[instance_id].get_generation_throughput()
                    starved_throughput = instance_to_status[starved_instance_id].get_generation_throughput()
                    if starved_throughput == 0:
                        ratio = float("inf") if throughput > 0 else 1
                    else:
                        ratio = throughput / starved_throughput
                elif self.config.psrl.sync_and_mig_strategy.mig.indicator == "kv_cache":
                    kv_cache_utilization = instance_to_status[instance_id].get_kv_cache_utilization()
                    starved_kv_cache_utilization = instance_to_status[starved_instance_id].get_kv_cache_utilization()
                    if starved_kv_cache_utilization == 0:
                        ratio = float("inf") if kv_cache_utilization > 0 else 1
                    else:
                        ratio = kv_cache_utilization / starved_kv_cache_utilization
                else:
                    raise ValueError(
                        f"Unknown migrate indicator: {self.config.psrl.sync_and_mig_strategy.mig.indicator}"
                    )

                if ratio > self.config.psrl.sync_and_mig_strategy.mig.threshold:
                    # psrl_logger.info(
                    #     f"Instance {instance_id} (version {self.instance_to_version_after_sync[instance_id]}) "
                    #     f"has a ratio of {ratio} for migrating to instance {starved_instance_id} "
                    #     f"(version {self.instance_to_version_after_sync[starved_instance_id]})"
                    # )
                    candidate_migrate_instance_ids.append((instance_id, ratio))

        # We choose the instance with the highest ratio to migrate
        # TODO(lhy): support multiple instances to migrate and finer-grained migration strategy
        # Currently, we only support one instance to migrate,
        # and all the requests on the instance will be interrupted and looped back to the router.
        if len(candidate_migrate_instance_ids) > 0:
            candidate_migrate_instance_ids.sort(key=lambda x: x[1], reverse=True)
            migrate_instance_id = candidate_migrate_instance_ids[0][0]
            if self.config.psrl.sync_and_mig_strategy.mig.stop_indicator == "request_num":
                request_num = instance_to_status[migrate_instance_id].get_waiting_and_running_queue_size()
                if request_num < self.config.psrl.sync_and_mig_strategy.mig.stop_threshold:
                    return []
            elif self.config.psrl.sync_and_mig_strategy.mig.stop_indicator == "throughput":
                throughput = instance_to_status[migrate_instance_id].get_generation_throughput()
                if throughput < self.config.psrl.sync_and_mig_strategy.mig.stop_threshold:
                    return []
            elif self.config.psrl.sync_and_mig_strategy.mig.stop_indicator == "kv_cache":
                kv_cache_utilization = instance_to_status[migrate_instance_id].get_kv_cache_utilization()
                if kv_cache_utilization < self.config.psrl.sync_and_mig_strategy.mig.stop_threshold:
                    return []
            else:
                raise ValueError(
                    f"Unknown stop indicator: {self.config.psrl.sync_and_mig_strategy.mig.stop_indicator}"
                )
            return [migrate_instance_id]
        return []

    @ray.method(concurrency_group="control")
    async def check_should_sync(self, instance_id: int, ps_model_version: int) -> bool:
        """Check if the instance should synchronize with PS.

        Args:
            instance_id (int): The instance ID to synchronize with.
            ps_model_version (int): The version of the PS model to synchronize with.

        Returns:
            bool: True if the instance should synchronize with PS, False otherwise.
        """
        # psrl_logger.info(f"Checking if instance {instance_id} should synchronize with PS")
        # If there are requests in the waiting queue
        # we will not attempt to synchronize with PS since the instance is still busy.
        instance_status = self.route_strategy.instance_to_engine_status[instance_id]
        if instance_status.get_waiting_queue_size() > 0:
            return False

        # Check if there are any requests that still can be routed to the instance
        # In this case, we will not attempt to synchronize with PS
        async with AsyncBusyPollingRayLock(self.ps_manager_handle):
            # 1. Check if there are any requests version satisfies the condition before synchronization
            get_version_remote = self.ps_manager_handle.get_rollout_instance_model_version.remote
            current_instance_version = await get_version_remote(instance_id)
            if await self.ps_manager_handle.check_aborted_model_versions.remote(current_instance_version):
                filtered_request_ids = []
            else:

                def version_filter(request):
                    assert "version_tag" in request.non_tensor_batch, (
                        "Request must have 'version_tag' for checking version"
                    )
                    request_version = request.non_tensor_batch["version_tag"][0]
                    return request_version <= current_instance_version

                filtered_requests = self.requests_to_route.filter_by_condition(version_filter)
                filtered_request_ids = [request.non_tensor_batch["uid"][0] for request in filtered_requests]

            # 2. Check if there are any requests
            # that can be RESERVED for the instance but no need to reserve new entry
            # before synchronization
            if len(filtered_request_ids) > 0:
                is_aborted = await self.ps_manager_handle.check_aborted_requests.remote(
                    filtered_request_ids, remove=False
                )
                filtered_request_ids = [
                    request_id for i, request_id in enumerate(filtered_request_ids) if not is_aborted[i]
                ]
                can_reserve_without_new_reserve_entry = await self.ps_manager_handle.can_reserve_request.remote(
                    filtered_request_ids,
                    [current_instance_version],
                    without_new_reserve_entry=True,
                )
                filtered_request_ids = [
                    request_id
                    for i, request_id in enumerate(filtered_request_ids)
                    if can_reserve_without_new_reserve_entry[i] == [True]
                ]

        # If there are requests that can still be routed to
        # the instance before synchronization without new reserve entry
        # we will not attempt to synchronize with PS
        if len(filtered_request_ids) > 0 and self.config.psrl.sync_and_mig_strategy.sync.check_req_before_sync:
            return False

        # 3. Check indicator to determine whether to synchronize with PS
        if self.config.psrl.sync_and_mig_strategy.sync.indicator == "request_num":
            # Check whether request num is above threshold
            request_num = instance_status.get_waiting_and_running_queue_size()
            psrl_logger.debug(
                f"Instance {instance_id} (version {self.instance_to_version_after_sync[instance_id]}) "
                f"request_num: {request_num}, "
                f"threshold: {self.config.psrl.sync_and_mig_strategy.sync.threshold}"
            )
            if request_num > self.config.psrl.sync_and_mig_strategy.sync.threshold:
                return False
        elif self.config.psrl.sync_and_mig_strategy.sync.indicator == "throughput":
            # Check whether throughput is above threshold
            throughput = self.route_strategy.instance_to_engine_status[instance_id].get_generation_throughput()
            psrl_logger.debug(
                f"Instance {instance_id} (version {self.instance_to_version_after_sync[instance_id]}) "
                f"throughput: {throughput}, "
                f"threshold: {self.config.psrl.sync_and_mig_strategy.sync.threshold}"
            )
            if throughput > self.config.psrl.sync_and_mig_strategy.sync.threshold:
                return False
        elif self.config.psrl.sync_and_mig_strategy.sync.indicator == "kv_cache":
            # Check whether KV Cache is above threshold
            kv_cache_utilization = instance_status.get_kv_cache_utilization()
            psrl_logger.debug(
                f"Instance {instance_id} (version {self.instance_to_version_after_sync[instance_id]}) "
                f"kv_cache_utilization: {kv_cache_utilization}, "
                f"threshold: {self.config.psrl.sync_and_mig_strategy.sync.threshold}"
            )
            if kv_cache_utilization > self.config.psrl.sync_and_mig_strategy.sync.threshold:
                return False
        elif self.config.psrl.sync_and_mig_strategy.sync.indicator == "hypothesis_test":
            # TODO(lhy): Implement hypothesis test after refactor
            # We attempt to synchronize with PS and check if there is any
            # benefit from synchronization
            raise NotImplementedError("Hypothesis test is not implemented")
            """
            def filter_func(request):
                version = request.non_tensor_batch.get(
                    "version_tag", [ps_model_version + 1]
                )[0]
                min_version_limit = request.non_tensor_batch.get(
                    "min_version_limit",
                    [ps_model_version + 1 + self.staleness]
                )[0]
                return (
                    version <= ps_model_version
                    or min_version_limit <= ps_model_version + self.staleness
                )
            
            new_filtered_requests = (
                self.requests_to_route.filter_by_condition(filter_func)
            )
            # Requests may be able to be routed to the instance {instance_id}
            # after synchronization, checking routing benefit...
            for request in new_filtered_requests:
                routing_benefit = (
                    self.route_strategy.calculate_routing_benefit(
                        request, instance_id
                    )
                )
                if routing_benefit > 0:
                    return True
            # No requests will benefit from routing to the instance
            # {instance_id} after synchronization
            """
        else:
            raise ValueError(f"Unknown sync indicator: {self.config.psrl.sync_and_mig_strategy.sync.indicator}")

        return True

    @ray.method(concurrency_group="control")
    async def wait_interrupted_partial_requests_loop_back(self, instance_ids: list[int]):
        """Wait for the interrupted partial requests to be looped back in the priority queue.

        Args:
            instance_ids (List[int]): The instance IDs to wait for.
        """
        finished_instance_ids = set()
        psrl_logger.info("Waiting for the interrupted partial requests to be looped back in the priority queue")
        while True:
            for instance_id in instance_ids:
                if (
                    instance_id not in finished_instance_ids
                    and len(self.instance_to_inflight_request_ids[instance_id]) == 0
                ):
                    psrl_logger.info(f"All requests on instance {instance_id} are looped back")
                    finished_instance_ids.add(instance_id)
            if len(finished_instance_ids) == len(instance_ids):
                break
            await asyncio.sleep(0)
        psrl_logger.info("The interrupted partial requests are looped back in the priority queue")

    # --- Trajectory KV cache management methods ---

    @ray.method(concurrency_group="control")
    async def kv_unregister(self, uid: int) -> None:
        """
        Remove trajectory token and instance records when a trajectory ends.

        Called by `AgentLoopWorker` when the trajectory is fully complete.

        Args:
            uid (int): Trajectory unique identifier.
        """
        self.request_to_last_routed_instance.pop(uid, None)
        self.request_to_tokens.pop(uid, None)
        psrl_logger.debug(f"[KV] Unregistered uid={uid}.")

    @ray.method(concurrency_group="control")
    async def kv_get_cache_info(self, uid: int) -> dict | None:
        """
        Query GPU prefix-cache and LMCache backend usage for a trajectory.

        Args:
            uid (int): Trajectory unique identifier.

        Returns:
            dict | None: `TrajectoryCacheInfo` as a dict, or None if uid not found.
        """
        instance_id, tokens = self._resolve_uid(uid)
        assert instance_id is not None, f"kv_get_cache_info: uid={uid} not found in any registry."
        assert tokens is not None, (
            f"kv_get_cache_info: uid={uid} found on instance {instance_id} "
            f"but has no registered tokens in request_to_tokens."
        )
        assert 0 <= instance_id < len(self.rollout_wg_list), (
            f"kv_get_cache_info: instance_id={instance_id} out of range "
            f"[0, {len(self.rollout_wg_list)})."
        )
        return await self.rollout_wg_list[instance_id].execute_rank_zero_async(
            "kv_get_cache_info", tokens
        )

    @ray.method(concurrency_group="control")
    async def kv_pin(self, uid: int, targets: list[str]) -> bool:
        """
        Pin the cached prefix blocks/chunks for a trajectory.

        Args:
            uid (int): Trajectory unique identifier.
            targets (list[str]): Subset of `["gpu", "backend"]`.

        Returns:
            bool: True if all requested pins succeeded, False if uid not found.
        """
        instance_id, tokens = self._resolve_uid(uid)
        assert instance_id is not None, f"kv_pin: uid={uid} not found in any registry."
        assert tokens is not None, (
            f"kv_pin: uid={uid} found on instance {instance_id} "
            f"but has no registered tokens in request_to_tokens."
        )
        assert 0 <= instance_id < len(self.rollout_wg_list), (
            f"kv_pin: instance_id={instance_id} out of range "
            f"[0, {len(self.rollout_wg_list)})."
        )
        return await self.rollout_wg_list[instance_id].execute_rank_zero_async(
            "kv_pin", tokens, targets
        )

    @ray.method(concurrency_group="control")
    async def kv_unpin(self, uid: int, targets: list[str]) -> bool:
        """
        Unpin the cached prefix blocks/chunks for a trajectory.

        Args:
            uid (int): Trajectory unique identifier.
            targets (list[str]): Subset of `["gpu", "backend"]`.

        Returns:
            bool: True if all unpins succeeded, False if uid not found.
        """
        instance_id, tokens = self._resolve_uid(uid)
        assert instance_id is not None, f"kv_unpin: uid={uid} not found in any registry."
        assert tokens is not None, (
            f"kv_unpin: uid={uid} found on instance {instance_id} "
            f"but has no registered tokens in request_to_tokens."
        )
        assert 0 <= instance_id < len(self.rollout_wg_list), (
            f"kv_unpin: instance_id={instance_id} out of range "
            f"[0, {len(self.rollout_wg_list)})."
        )
        return await self.rollout_wg_list[instance_id].execute_rank_zero_async(
            "kv_unpin", tokens, targets
        )

    @ray.method(concurrency_group="control")
    async def kv_clear_from_backend(self, uid: int) -> int:
        """
        Remove the cached prefix chunks for a trajectory from the LMCache backend.

        Args:
            uid (int): Trajectory unique identifier.

        Returns:
            int: Number of chunks removed, or 0 if uid not found.
        """
        instance_id, tokens = self._resolve_uid(uid)
        assert instance_id is not None, f"kv_clear_from_backend: uid={uid} not found in any registry."
        assert tokens is not None, (
            f"kv_clear_from_backend: uid={uid} found on instance {instance_id} "
            f"but has no registered tokens in request_to_tokens."
        )
        assert 0 <= instance_id < len(self.rollout_wg_list), (
            f"kv_clear_from_backend: instance_id={instance_id} out of range "
            f"[0, {len(self.rollout_wg_list)})."
        )
        return await self.rollout_wg_list[instance_id].execute_rank_zero_async(
            "kv_clear_from_backend", tokens
        )

    @ray.method(concurrency_group="control")
    async def kv_transfer(
        self,
        uid: int,
        src: tuple[str, str],
        dst: tuple[str, str],
        copy: bool = False,
    ) -> bool:
        """
        Transfer the cached prefix of a trajectory to another (instance, backend).

        Args:
            uid (int): Trajectory unique identifier.
            src (tuple[str, str]): Source `(lmcache_instance_id, backend_location)`.
            dst (tuple[str, str]): Destination `(lmcache_instance_id, backend_location)`.
            copy (bool): If True, keep data at source as well.

        Returns:
            bool: True if transfer was initiated, False if uid not found.
        """
        instance_id, tokens = self._resolve_uid(uid)
        assert instance_id is not None, f"kv_transfer: uid={uid} not found in any registry."
        assert tokens is not None, (
            f"kv_transfer: uid={uid} found on instance {instance_id} "
            f"but has no registered tokens in request_to_tokens."
        )
        assert 0 <= instance_id < len(self.rollout_wg_list), (
            f"kv_transfer: instance_id={instance_id} out of range "
            f"[0, {len(self.rollout_wg_list)})."
        )
        return await self.rollout_wg_list[instance_id].execute_rank_zero_async(
            "kv_transfer", tokens, src, dst, copy
        )

