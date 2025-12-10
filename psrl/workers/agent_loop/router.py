import os
import logging
import numpy as np
import asyncio
from tensordict import TensorDict
from omegaconf import DictConfig
from typing import List, Optional, Dict, Tuple, Any, Callable, Iterator

import ray

from verl import DataProto

from psrl.workers.gen.stats_collector import EngineStats
from psrl.workers.ps.request_status_tracker import PSRL_RequestStatus
from psrl.utils.ray import BusyPollingRayLock, AsyncBusyPollingRayLock
from psrl.utils.logger import DualOutputHandler, EventType, log_dual_events, deprecated
from psrl.workers.agent_loop.route_strategy import (
    RouteStrategyBase, 
    get_route_strategy_class,
)
from psrl.workers.agent_loop.request_queue import PriorityRequestQueue, MultiPriorityRequestQueue


psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class RolloutRouter:
    def __init__(
        self,
        config: DictConfig,
        ps_manager_handle,
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
        if self.config.psrl.redundant_rollout.enable:
            self.rollout_n = self.config.psrl.redundant_rollout.redundant_rollout_n
            self.alg_rollout_n = self.config.psrl.redundant_rollout.alg_rollout_n
            self.balanced_concurrent_seqs_per_instance = self.config.psrl.redundant_rollout.redundant_global_batch_size * self.rollout_n // self.config.psrl.deployment.n_rollout_instances
        else:
            self.rollout_n = self.config.gen_actor_rollout_ref.rollout.n
            self.alg_rollout_n = self.rollout_n
            self.balanced_concurrent_seqs_per_instance = self.config.psrl.staleness_buffer_entries * self.rollout_n // self.config.psrl.deployment.n_rollout_instances
        self.ps_manager_handle = ps_manager_handle
        self.rollout_wg_list = rollout_wg_list
        self.rollout_wg_size = len(rollout_wg_list)
        assert self.rollout_wg_size == self.config.psrl.deployment.n_rollout_instances, "Rollout worker group size must match the number of deployment instances"
        
        # Build logger
        self.log_prefix = f"RolloutRouter"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))
        
        # Routing related attributes
        if self.config.psrl.routing_strategy.enable_multi_priority_queue:
            self.requests_to_route = MultiPriorityRequestQueue(self.staleness, short_request_first=self.config.psrl.routing_strategy.short_request_first)
        else:
            self.requests_to_route = PriorityRequestQueue(self.staleness, short_request_first=self.config.psrl.routing_strategy.short_request_first)
        self.routing_lock = asyncio.Lock()
        self.routing_status_update_event = asyncio.Event()
        self._is_routing = False
        self._interrupt_routing = False
        self.scheduler_task = None  # Will be created in async context
        self.instance_to_inflight_request_ids = {i: [] for i in range(self.rollout_wg_size)} # Track the inflight request ids for each instance (i.e., request that is being generated and is not yet completed or queued in the priority queue): {instance_id: [request_id, ...]}
        self.incomplete_request_to_instance = {} # Track the instance id for each incomplete request (i.e., request that is not completed yet): {request_id: instance_id}
        self.request_futures = {}  # Track request futures: {request_id: Future}
        self.instance_to_version_after_sync = {i: 0 for i in range(self.rollout_wg_size)} # Track the version after synchronization for each instance: {instance_id: ps_model_version}
        
        # Build logger
        self.log_prefix = f"RolloutRouter"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))
        psrl_logger.info(f"Initialized RolloutRouter")

    def init_route_strategy(self, **kwargs):
        """Initialize the route strategy for the router.
        
        Args:
            **kwargs: Keyword arguments for the route strategy.
        """
        if self.config.psrl.routing_strategy.method == "request_num_balance" or self.config.psrl.routing_strategy.method == "throughput_optimal":
            assert self.config.psrl.status_collection.enable, "Status collection must be enabled when using request num balance or throughput optimal routing strategy"
        n_instances = self.rollout_wg_size
        if self.config.psrl.deployment.heterogeneous_rollout.enable:
            instance_to_tp_pp = {
                i: f"TP{self.config.psrl.deployment.heterogeneous_rollout.tensor_model_parallel_size_per_instance[i]}_PP{self.config.psrl.deployment.heterogeneous_rollout.pipeline_model_parallel_size_per_instance[i]}"
                for i in range(n_instances)
            }
        else:
            instance_to_tp_pp = {
                i: f"TP{self.config.gen_actor_rollout_ref.rollout.tensor_model_parallel_size}_PP{self.config.gen_actor_rollout_ref.rollout.pipeline_model_parallel_size}"
                for i in range(n_instances)
            }
        strategy_kwargs = {
            "sort_candidate_by_indicator": self.config.psrl.routing_strategy.sort_candidate_by_indicator,
            "logging_interval_in_ms": self.config.psrl.routing_strategy.logging_interval_in_ms,
            "cost_model_path": self.config.psrl.routing_strategy.cost_model_path,
            "instance_to_tp_pp": instance_to_tp_pp,
            "max_num_waiting_reqs_after_preemption": self.config.psrl.routing_strategy.max_num_waiting_reqs_after_preemption,
            "balanced_concurrent_seqs_per_instance": self.balanced_concurrent_seqs_per_instance,
            "max_concurrent_seqs_per_instance": self.config.psrl.routing_strategy.max_concurrent_seqs_per_instance,
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
            psrl_logger.warning(f"Falling back to 'round_robin' strategy")
            from psrl.workers.agent_loop.route_strategy import RoundRobinRouteStrategy
            self.route_strategy: RouteStrategyBase = RoundRobinRouteStrategy(n_instances, strategy_kwargs)

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
        self.route_strategy.update_instance_to_engine_status({instance_id: instance_to_engine_status[instance_id] for instance_id in filtered_instance_ids})
        
        # Notify the scheduler that status has been updated
        if len(filtered_instance_ids) > 0:
            self.routing_status_update_event.set()
            
    async def update_currently_syncing_instances(self, instance_ids: List[int], ps_model_version: int):
        """Update the currently syncing instances.
        
        Args:
            instance_ids (List[int]): The instance IDs to update.
            ps_model_version (int): The version of the PS model to update.
        """
        for instance_id in instance_ids:
            self.instance_to_version_after_sync[instance_id] = ps_model_version

    def _choose_new_rollout_instance(self, request: DataProto) -> int:
        """Select the best rollout instance for handling the generation request.
        
        Args:
            request (DataProto): The request to be routed.

        Returns:
            int: Index of the selected rollout instance.
        """
        # Ensure the whole routing process is atomic from the PS manager side
        # psrl_logger.info(f"Choosing new rollout instance for request {request.non_tensor_batch['uid'][0]}")
        request_id = request.non_tensor_batch["uid"][0]
        if "version_tag" in request.non_tensor_batch:
            needed_model_version = request.non_tensor_batch["version_tag"][0]
        else:
            assert "min_version_limit" in request.non_tensor_batch, "Request must have either 'version_tag' or 'min_version_limit'"
            needed_model_version = request.non_tensor_batch["min_version_limit"][0] - self.staleness
        
        # 1. Filter the rollout instances that can tolerate the needed staleness of the request
        # This guarantees that the gen worker will have no ahead-of-time version tag when generating
        candidates = [i for i, version in self.instance_to_version_after_sync.items() if version >= needed_model_version]
        # psrl_logger.info(f"Candidates for request {request_id}: {candidates}")
        
        # 2. If forbidden global migration and the request is a partial rollout request, only consider the specific instance for routing
        if "rollout_instance_id" in request.non_tensor_batch and not self.config.psrl.sync_and_mig_strategy.mig.enable:
            old_instance_id = request.non_tensor_batch["rollout_instance_id"][0]
            assert old_instance_id in candidates, f"Old rollout instance {old_instance_id} is not in the candidates"
            candidates = [old_instance_id]
            
        # 3. If forbidden group sampling on multiple instances, only consider the instance that other requests in the same group are already routed to
        if not self.config.psrl.routing_strategy.enable_group_sampling_on_multi_instances:
            group_request_instance_ids = [instance_id for incomplete_request_id, instance_id in self.incomplete_request_to_instance.items() if incomplete_request_id // self.rollout_n == request_id // self.rollout_n]
            if len(group_request_instance_ids) > 0:
                assert all(instance_id == group_request_instance_ids[0] for instance_id in group_request_instance_ids), f"All requests in the same group must be routed to the same instance, but found different instances: {group_request_instance_ids}"
                assert group_request_instance_ids[0] in candidates, f"Group request instance {group_request_instance_ids[0]} of request {request_id} is not in the candidates {candidates}, instance versions: {self.instance_to_version_after_sync}, needed model version: {needed_model_version}"
                candidates = [group_request_instance_ids[0]]
        
        # 4. Filter the rollout instances that can reserve the request for the current instance model version
        # This is only used when dynamic version tag is enabled and the needed model version is -1 (i.e. new request)
        if self.config.psrl.routing_strategy.enable_dynamic_version_tag and needed_model_version == -1:
            all_candidate_model_versions = list(set([self.instance_to_version_after_sync[candidate] for candidate in candidates]))
            # psrl_logger.info(f"Trying to reserve request {request_id} with candidate model versions: {all_candidate_model_versions}")
            can_reserve_results = ray.get(self.ps_manager_handle.can_reserve_request.remote(request_id, all_candidate_model_versions))
            # psrl_logger.info(f"Request {request_id} all candidate model versions: {all_candidate_model_versions}, can reserve results: {can_reserve_results}")
            candidates = [candidate for candidate in candidates if can_reserve_results[all_candidate_model_versions.index(self.instance_to_version_after_sync[candidate])]]
        
        route_kwargs = {
            "instance_to_version_after_sync": self.instance_to_version_after_sync,
        }
        # 5. If the candidates are sorted by indicator, we need to provide the indicator list for the route strategy
        if self.config.psrl.routing_strategy.sort_candidate_by_indicator:
            if not self.config.psrl.routing_strategy.enable_dynamic_version_tag:
                # Use the version after synchronization as the indicator
                candidate_indicator_list = [self.instance_to_version_after_sync[candidate] for candidate in candidates]
            else:
                # Use the (reserve_indicator, version) pair as the final indicator
                all_candidate_model_versions = list(set([self.instance_to_version_after_sync[candidate] for candidate in candidates]))
                # psrl_logger.info(f"Getting reserve indicator for request {request_id} with candidate model versions: {all_candidate_model_versions}")
                indicator_results = ray.get(self.ps_manager_handle.get_reserve_indicator.remote(request_id, all_candidate_model_versions))
                # psrl_logger.info(f"Request {request_id} all candidate model versions: {all_candidate_model_versions}, indicator results: {indicator_results}")
                candidate_indicator_list = [(indicator_results[all_candidate_model_versions.index(self.instance_to_version_after_sync[candidate])], self.instance_to_version_after_sync[candidate]) for candidate in candidates]
            route_kwargs["candidate_indicator_list"] = candidate_indicator_list
        
        # 6. Strategy-based routing
        chosen_rollout_instance = self.route_strategy.route(
            request, 
            candidates=candidates,
            route_kwargs=route_kwargs
        )
        
        # 7. If not None, the request is routed to the chosen rollout instance
        if chosen_rollout_instance is not None:
            # Allocate the version tag and reserve the request for the chosen rollout instance if the request is not routed before and dynamic version tag is enabled
            if not "rollout_instance_id" in request.non_tensor_batch and self.config.psrl.routing_strategy.enable_dynamic_version_tag:
                needed_model_version = self.instance_to_version_after_sync[chosen_rollout_instance]
                request.non_tensor_batch["version_tag"] = np.array([needed_model_version], dtype=int)
                # psrl_logger.info(f"Reserving request {request_id} for rollout instance {chosen_rollout_instance} with version tag {needed_model_version}")
                ray.get(self.ps_manager_handle.reserve_rollout_instance_requests.remote(
                    rollout_instance_ids=chosen_rollout_instance,
                    request_ids=request_id,
                    model_versions=needed_model_version,
                ))
                # psrl_logger.info(f"Request {request_id} is reserved for rollout instance {chosen_rollout_instance} with version tag {needed_model_version}")
            # Otherwise, the request is already reserved, only need to update the request instance id
            else:
                # psrl_logger.info(f"Updating request {request_id} instance id to {chosen_rollout_instance}")
                ray.get(self.ps_manager_handle.update_request_instance_id.remote(
                    request_id=request_id,
                    new_instance_id=chosen_rollout_instance,
                ))
                # psrl_logger.info(f"Request {request_id} instance id updated to {chosen_rollout_instance}")
        
        return chosen_rollout_instance

    # Only used in batch gen mode
    def _get_retry_version_and_instance(
        self,
        sample_id: int,
        min_version_limit: int,
    ) -> (int, int):
        """Get version and rollout instance for a retried sample.
        
        Args:
            sample_id (int): Unique identifier of the sample.
            min_version_limit (int): Maximum allowed version limit.
            
        Returns:
            tuple: (version_tag, rollout_instance_id) for the sample.
        """
        filtered_rollout_instance_to_version = {
            instance_id: version 
            for instance_id, version in self.instance_to_version_after_sync.items() 
            if min_version_limit - self.staleness <= version <= min_version_limit
        }
        
        if not filtered_rollout_instance_to_version:
            raise AssertionError(
                f"No available rollout instance meets the version requirement for "
                f"sample {sample_id} with min_version_limit {min_version_limit} and staleness {self.staleness}. "
                f"All instance versions: {self.instance_to_version_after_sync}"
            )
        
        # Choose a rollout instance from available candidates
        candidates = list(filtered_rollout_instance_to_version.keys())
        chosen_instance_id = candidates[sample_id % len(candidates)]
        chosen_version = filtered_rollout_instance_to_version[chosen_instance_id]

        return chosen_version, chosen_instance_id

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
                self.config.psrl.log_prob.enable_rollout_engine_log_prob and
                hasattr(vllm_output.outputs[0], 'logprobs') and
                vllm_output.outputs[0].logprobs is not None
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
        if "raw_response_ids" in non_tensor_batch:
            raw_response_ids = non_tensor_batch["raw_response_ids"]
        else:
            raw_response_ids = np.fromiter(([] for _ in range(batch_size)), dtype=object)

        raw_response_ids += np.fromiter(response_ids_list, dtype=object)
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
                curr_rollout_log_probs = non_tensor_batch["rollout_log_probs"]
            else:
                curr_rollout_log_probs = np.fromiter(([] for _ in range(batch_size)), dtype=object)
            curr_rollout_log_probs += np.fromiter(all_log_prob_list, dtype=object)
            non_tensor_batch["rollout_log_probs"] = curr_rollout_log_probs

        batch = TensorDict(
            {
                "input_ids": prompts.batch["input_ids"],
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)

    def generate(
        self,
        requests: DataProto,
    ):
        """Synchronously generate responses for a batch of requests.
        
        Args:
            requests (DataProto): Batch of generation requests.
            
        Returns:
            DataProto or None: Generated results or None if no valid requests.
        """
        assert not self.config.psrl.routing_strategy.enable_dynamic_version_tag, "Dynamic version tag is not supported in batch mode"
        request_ids = requests.non_tensor_batch.get("uid", None)
        
        if "min_version_limit" in requests.non_tensor_batch:
            # Indicate that these requests are retry requests
            min_version_limit = requests.non_tensor_batch["min_version_limit"][0]
            assert all(v == min_version_limit for v in requests.non_tensor_batch["min_version_limit"]), "All requests in the batch must have the same min_version_limit."
            
            requests.non_tensor_batch.pop("min_version_limit")
            
            # Group requests by sample_id and assign versions
            sample_to_requests = {}
            for i, uid in enumerate(request_ids):
                sample_id = uid // self.rollout_n
                if sample_id not in sample_to_requests:
                    sample_to_requests[sample_id] = []
                sample_to_requests[sample_id].append(i)
            
            # Assign version and rollout instance for each sample_id
            requests_list = []
            for sample_id, request_indices in sample_to_requests.items():
                needed_model_version, rollout_instance_id = self._get_retry_version_and_instance(
                    sample_id, min_version_limit
                )
                # Create a sub-batch for this sample_id
                sample_requests = requests.select_idxs(request_indices)
                sample_requests.non_tensor_batch["rollout_instance_id"] = np.array([rollout_instance_id] * len(request_indices), dtype=int)
                sample_requests.non_tensor_batch["version_tag"] = np.array([needed_model_version] * len(request_indices), dtype=int) # Refactor the version tag (previouly is assigned by uid in agent loop manager)
                requests_list.append(sample_requests) 
            requests = DataProto.concat(requests_list)
            
        update_status_success = ray.get(self.ps_manager_handle.update_request_status.remote(
            request_ids.tolist(),
            PSRL_RequestStatus.ROLLOUT_DISPATCHED,
            model_version = requests.non_tensor_batch.get("version_tag", np.array([-1], dtype=int)).tolist(),
        ))
        filtered_request_idxs = [i for i, success in enumerate(update_status_success) if success]
        if filtered_request_idxs:
            requests = requests.select_idxs(filtered_request_idxs)
            request_ids = requests.non_tensor_batch["uid"]
            # evenly dispatch to rollout instances
            futures = []
            with log_dual_events(f"Dispatching {len(requests)} requests to {self.rollout_wg_size} rollout instances evenly and generate in batch", psrl_logger, event_type=EventType.GEN):
                filtered_requests_list = []
                if "rollout_instance_id" in requests.non_tensor_batch:
                    rollout_instance_ids = set(requests.non_tensor_batch["rollout_instance_id"].tolist())
                    for instance_id in rollout_instance_ids:
                        filtered_requests = requests.select_idxs(
                            [i for i, rid in enumerate(requests.non_tensor_batch["rollout_instance_id"]) if rid == instance_id]
                        )
                        filtered_requests_list.append(filtered_requests)
                else:
                    filtered_requests_list = requests.chunk(self.rollout_wg_size)
                    for i, filtered_requests in enumerate(filtered_requests_list):
                        filtered_requests.non_tensor_batch["rollout_instance_id"] = np.array([i] * len(filtered_requests), dtype=int)
                # Reserve data in staleness buffer
                for filtered_requests in filtered_requests_list:
                    ray.get(self.ps_manager_handle.reserve_rollout_instance_requests.remote(
                        rollout_instance_ids=filtered_requests.non_tensor_batch["rollout_instance_id"].tolist(),
                        request_ids=filtered_requests.non_tensor_batch["uid"].tolist(),
                        model_versions=filtered_requests.non_tensor_batch["version_tag"].tolist(),
                    ))

                for i, filtered_requests in enumerate(filtered_requests_list):
                    psrl_logger.debug(f"Dispatching requests to rollout instance {i} with request ids: {filtered_requests.non_tensor_batch['uid']}")
                    futures.append(
                        self.rollout_wg_list[i].execute_all_async("generate", filtered_requests)[0]
                    )
                rollout_results = ray.get(futures)
                
            # Process results as needed
            with log_dual_events(f"Concatenating results from {self.rollout_wg_size} rollout instances", psrl_logger, event_type=EventType.OTHER):
                results = []
                for i in range(self.rollout_wg_size):
                    consolidated_outputs, update_statuses = rollout_results[i]
                    if consolidated_outputs is None:
                        continue
                    psrl_logger.debug(f"Consolidated outputs from rollout instance {i} have request ids: {consolidated_outputs.non_tensor_batch['uid']}")
                    assert (
                        update_statuses is not None and
                        all(update_status == PSRL_RequestStatus.RUNNING for update_status in update_statuses)
                    ), "Interruption is not implemented in batching mode"
                    results.append(consolidated_outputs)
                return DataProto.concat(results)
            
        return None

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
        assert "rollout_instance_id" not in request.non_tensor_batch, "Rollout instance ID should not be provided in the original request"
        if self.scheduler_task is None:
            if self.config.psrl.routing_strategy.enable_multi_priority_queue:
                self.scheduler_task = asyncio.create_task(self._multi_priority_queue_routing_loop())
            else:
                self.scheduler_task = asyncio.create_task(self._single_priority_queue_routing_loop())
            self.scheduler_task.add_done_callback(lambda f: f.result()) # To avoid silent error in async tasks
            psrl_logger.info("Started routing loop")
        
        # Create a future to track this request's completion
        request_id = request.non_tensor_batch["uid"][0]
        result_future = asyncio.Future()
        # Store the future in a way that the scheduler can access it
        self.request_futures[request_id] = result_future
        if self.config.psrl.routing_strategy.enable_dynamic_version_tag:
            # 1. Put in the request queue and route to the instance (the version tag is -1 since we don't know the version tag yet)
            # 2. Allocate the version tag and reserve data in staleness buffer
            if "version_tag" in request.non_tensor_batch:
                assert request.non_tensor_batch["version_tag"] == -1, "The version tag should be -1 for dynamic version tag"
        else:
            # 1. Reserve data in staleness buffer (the instance id is -1 since we don't know the instance id yet)
            # 2. Put in the request queue and route to the instance
            model_version = request.non_tensor_batch["version_tag"][0] if "version_tag" in request.non_tensor_batch else request.non_tensor_batch["min_version_limit"][0] - self.staleness
            await self.ps_manager_handle.reserve_rollout_instance_requests.remote(
                rollout_instance_ids=-1,
                request_ids=request_id,
                model_versions=model_version,
            )
        # Add request to priority queue
        self.requests_to_route.put(request)
        psrl_logger.info(f"Adding request {request_id} to priority queue")
        self.routing_status_update_event.set()
        # Wait for the request to be processed
        with log_dual_events(f"Routing request {request_id} and waiting for it to be processed", psrl_logger, level=logging.DEBUG, event_type=EventType.GEN):
            result = await result_future
        # Clean up the future
        self.request_futures.pop(request_id)
        return result
    
    def is_routing(self) -> bool:
        """Check if the router is currently routing requests."""
        return self._is_routing
    
    async def interrupt_routing(self):
        """Interrupt the routing.
        """
        with log_dual_events(f"Interrupt routing", psrl_logger, event_type=EventType.OTHER):
            async with self.routing_lock:
                self._interrupt_routing = True
        
    async def resume_routing(self):
        """Resume the routing.
        """
        with log_dual_events(f"Resume routing", psrl_logger, event_type=EventType.OTHER):
            async with self.routing_lock:
                self._interrupt_routing = False
            self.routing_status_update_event.set()
            
    async def _single_priority_queue_routing_loop(self):
        """Continuous routing loop for a single priority queue.

        This loop processes requests from the single priority queue.
        """
        # psrl_logger.info("Started single priority queue routing loop")
        while True:
            # Process all requests in the priority queue
            self._is_routing = False
            is_stuck = True
            async with self.routing_lock, AsyncBusyPollingRayLock(self.ps_manager_handle):
                while not self.requests_to_route.empty() and not self._interrupt_routing:
                    self._is_routing = True
                    request = self.requests_to_route.pop()
                    assert request is not None, "Request should not be None in priority queue"
                    request_id = request.non_tensor_batch["uid"][0]
                    assert request_id in self.request_futures, f"Request {request_id} should be in request futures"
                    old_instance_id = request.non_tensor_batch["rollout_instance_id"][0] if "rollout_instance_id" in request.non_tensor_batch else None
                    new_instance_id = self._choose_new_rollout_instance(request)
                    # psrl_logger.info(f"Choosing rollout instance for request {request_id} to {instance_id}")
                    if new_instance_id is None:
                        # Indicate that we cannot find a suitable rollout instance for the request due to the current engine status (e.g., version staleness, instance overload).
                        # Need to wait for engine status update to try again:
                        # 1. The overall engine status could be updated by the coordinator periodically.
                        # 2. The engine status of the specific instance could be updated after one request is added/completed.
                        self.requests_to_route.put(request)
                        break
                    is_stuck = False
                    self.incomplete_request_to_instance[request_id] = new_instance_id
                    # Create a task to process this request
                    task = asyncio.create_task(self._route_single_request(request, old_instance_id, new_instance_id))
                    task.add_done_callback(lambda f: f.result()) # To avoid silent error in async tasks
            '''
            if is_stuck:
                self._is_routing = False
                self.routing_status_update_event.clear()
                await self.routing_status_update_event.wait()
            else:
                await asyncio.sleep(0)
            '''
            self._is_routing = False
            await asyncio.sleep(self.config.psrl.routing_strategy.check_interval_in_ms / 1000)
    
    async def _multi_priority_queue_routing_loop(self):
        """Continuous routing loop for multiple priority queues.
        
        This loop processes requests from the multiple priority queues.
        """
        psrl_logger.info("Started multi priority queue routing loop")
        while True:
            # Process all requests in the multiple priority queues
            self._is_routing = False
            is_stuck = True
            # psrl_logger.info("Trying to acquire lock")
            async with self.routing_lock, AsyncBusyPollingRayLock(self.ps_manager_handle):
                # psrl_logger.info("Acquired lock")
                self.requests_to_route.remove_empty_queues()
                remain_requests = []
                for queue_id, request_queue in self.requests_to_route.iter_queues():
                    if len(remain_requests) != 0:
                        # Method 1: If the last queue still has requests, we will not process the other queues
                        break
                        # Method 2: Try to process the other queues
                        # remain_requests.clear()
                    # psrl_logger.info(f"Processing requests in priority queue {queue_id}, there are {request_queue.size()} requests in the queue")
                    while not request_queue.empty() and not self._interrupt_routing:
                        request = request_queue.pop()
                        assert request is not None, "Request should not be None in priority queue"
                        request_id = request.non_tensor_batch["uid"][0]
                        # psrl_logger.info(f"Processing request {request_id} in priority queue {queue_id}")
                        assert request_id in self.request_futures, f"Request {request_id} should be in request futures"
                        old_instance_id = request.non_tensor_batch["rollout_instance_id"][0] if "rollout_instance_id" in request.non_tensor_batch else None
                        new_instance_id = self._choose_new_rollout_instance(request)
                        if new_instance_id is None:
                            remain_requests.append(request)
                            continue
                        is_stuck = False
                        self.incomplete_request_to_instance[request_id] = new_instance_id
                        # Create a task to process this request
                        task = asyncio.create_task(self._route_single_request(request, old_instance_id, new_instance_id))
                        task.add_done_callback(lambda f: f.result()) # To avoid silent error in async tasks
                    # psrl_logger.info(f"There are {len(remain_requests)} requests left in priority queue {queue_id}, putting them back to the queue")
                    for request in remain_requests:
                        request_queue.put(request)
            '''
            if is_stuck:
                self._is_routing = False
                self.routing_status_update_event.clear()
                # psrl_logger.info("Routing is stuck, waiting for routing status update event")
                await self.routing_status_update_event.wait()
                # psrl_logger.info("Routing is resumed")
            else:
                await asyncio.sleep(0)
            '''
            self._is_routing = False
            await asyncio.sleep(self.config.psrl.routing_strategy.check_interval_in_ms / 1000)
    
    async def _route_single_request(self, request: DataProto, old_instance_id: Optional[int], new_instance_id: int):
        """Route a single request to a rollout instance.
        
        Args:
            request (DataProto): The request to process.
            old_instance_id (Optional[int]): The old rollout instance id that the request is routed to, None if not exists.
            new_instance_id (int): The new rollout instance id that the request will be routed to.
        """  
        # Update request non-tensor batch
        # psrl_logger.info(f"Routing single request {request.non_tensor_batch['uid'][0]} to rollout instance {new_instance_id}")
        request_id = request.non_tensor_batch["uid"][0]
        request.non_tensor_batch["rollout_instance_id"] = np.array([new_instance_id], dtype=int)
        if "version_tag" in request.non_tensor_batch:
            needed_model_version = request.non_tensor_batch["version_tag"][0]
            assert needed_model_version != -1, "The version tag should not be -1 (new request that is not allocated a version tag yet when enabled dynamic version tag) after routing"
        else:
            # Indicate that it is a retry request
            assert "min_version_limit" in request.non_tensor_batch, "min_version_limit is required for routing if version_tag is not provided"
            needed_model_version = self.instance_to_version_after_sync[new_instance_id]
            request.non_tensor_batch["version_tag"] = np.array([needed_model_version], dtype=int)
        # psrl_logger.info(f"Routing single request {request_id} with version tag {needed_model_version} to rollout instance {new_instance_id}")  
        
        # Update request status
        # psrl_logger.info(f"Updating request {request_id} status to ROLLOUT_DISPATCHED with version tag {needed_model_version}")
        update_status_success = await self.ps_manager_handle.update_request_status.remote(
            [request_id],
            PSRL_RequestStatus.ROLLOUT_DISPATCHED,
            model_version=request.non_tensor_batch["version_tag"].tolist(),
        )
        # psrl_logger.info(f"Update request {request_id} status to ROLLOUT_DISPATCHED success: {update_status_success[0]}")
        
        if not update_status_success[0]:
            raise ValueError(f"Failed to update request {request_id} status to ROLLOUT_DISPATCHED")
            # return
            
        # Change engine status    
        self.route_strategy.push_request(request, new_instance_id)
        # Add request to inflight request ids for the instance
        self.instance_to_inflight_request_ids[new_instance_id].append(request_id)
        
        # Generate response
        # psrl_logger.info(f"Generating response for request {request_id} on instance {new_instance_id}")
        consolidated_output, update_status = await self.rollout_wg_list[new_instance_id].execute_rank_zero_async("generate_async", request)
            
        # Change engine status 
        self.route_strategy.pop_request(request, new_instance_id)
        # Remove request from inflight request ids for the instance
        self.instance_to_inflight_request_ids[new_instance_id].remove(request_id)
        self.routing_status_update_event.set()
        
        # Check if request was interrupted and needs to be requeued
        if update_status == PSRL_RequestStatus.ROLLOUT_INTERRUPTED_BY_SCHEDULER:
            psrl_logger.info(f"Request {request_id} on instance {new_instance_id} was interrupted by scheduler (most likely due to kv cache full and preemption), requeueing")
            # Put back in priority queue for partial rollout
            # Ensure that the consolidated output has the rollout instance id recorded
            consolidated_output.non_tensor_batch["rollout_instance_id"] = np.array([new_instance_id], dtype=int)
            self.requests_to_route.put(consolidated_output)
            return
        elif update_status == PSRL_RequestStatus.ROLLOUT_INTERRUPTED:
            psrl_logger.info(f"Request {request_id} on instance {new_instance_id} was interrupted (due to model synchronization or rollout migration when enabled partial rollout), requeueing")
            # Put back in priority queue for partial rollout
            # Ensure that the consolidated output has the rollout instance id recorded
            consolidated_output.non_tensor_batch["rollout_instance_id"] = np.array([new_instance_id], dtype=int)
            self.requests_to_route.put(consolidated_output)
            return
        elif update_status == PSRL_RequestStatus.RUNNING:
            psrl_logger.info(f"Request {request_id} on instance {new_instance_id} of parent prompt {request_id // self.rollout_n} completed successfully, length is {consolidated_output.non_tensor_batch['response_unpadded_len'][0]}")
            result = consolidated_output
        else:
            # TODO(lhy): support abortion
            raise ValueError(f"Unexpected update status for request {request_id}: {update_status}")
    
        # Set the result for any waiting futures
        assert request_id in self.request_futures, f"Request {request_id} should be in request futures"
        assert not self.request_futures[request_id].done(), f"Request {request_id} should not be done"
        self.incomplete_request_to_instance.pop(request_id)
        self.request_futures[request_id].set_result(result)
    
    async def check_should_migrate(self) -> List[int]:
        """Check which instances should be interrupted to migrate to others due to starvation.
        
        Returns:
            List[int]: The instance IDs that should be interrupted to migrate.
        """
        # psrl_logger.info("Checking which instances should be interrupted to migrate to others due to starvation")
        filtered_instance_ids = []
        for instance_id in range(self.rollout_wg_size):
            if self.route_strategy.instance_to_engine_status[instance_id].get_waiting_queue_size() != 0:
                continue
            instance_version = self.instance_to_version_after_sync[instance_id]
            filtered_requests = self.requests_to_route.filter_by_condition(
                lambda request: request.non_tensor_batch.get("version_tag", [instance_version + 1])[0] <= instance_version
                or request.non_tensor_batch.get("min_version_limit", [instance_version + 1 + self.staleness])[0] <= instance_version + self.staleness,
            )
            if self.config.psrl.routing_strategy.enable_dynamic_version_tag:
                filtered_request_ids = [request.non_tensor_batch["uid"][0] for request in filtered_requests]
                can_reserve = await self.ps_manager_handle.can_reserve_request.remote(filtered_request_ids, [instance_version], without_new_reserve_entry=False)
                filtered_requests = [request for i, request in enumerate(filtered_requests) if can_reserve[i] == [True]]
            if len(filtered_requests) == 0:
                filtered_instance_ids.append(instance_id)
        
        candidate_migrate_instance_ids = [] # (instance_id, ratio)
        for starved_instance_id in filtered_instance_ids:
            for instance_id in range(self.rollout_wg_size):
                if instance_id == starved_instance_id:
                    continue
                if self.instance_to_version_after_sync[instance_id] > self.instance_to_version_after_sync[starved_instance_id]:
                    continue
                
                if self.config.psrl.sync_and_mig_strategy.mig.indicator == "request_num":
                    request_num = self.route_strategy.instance_to_engine_status[instance_id].get_waiting_and_running_queue_size()
                    starved_request_num = self.route_strategy.instance_to_engine_status[starved_instance_id].get_waiting_and_running_queue_size()
                    if starved_request_num == 0:
                        ratio = float('inf') if request_num > 0 else 1
                    else:
                        ratio = request_num / starved_request_num
                elif self.config.psrl.sync_and_mig_strategy.mig.indicator == "throughput":
                    throughput = self.route_strategy.instance_to_engine_status[instance_id].get_generation_throughput()
                    starved_throughput = self.route_strategy.instance_to_engine_status[starved_instance_id].get_generation_throughput()
                    if starved_throughput == 0:
                        ratio = float('inf') if throughput > 0 else 1
                    else:
                        ratio = throughput / starved_throughput
                elif self.config.psrl.sync_and_mig_strategy.mig.indicator == "kv_cache":
                    kv_cache_utilization = self.route_strategy.instance_to_engine_status[instance_id].get_kv_cache_utilization()
                    starved_kv_cache_utilization = self.route_strategy.instance_to_engine_status[starved_instance_id].get_kv_cache_utilization()
                    if starved_kv_cache_utilization == 0:
                        ratio = float('inf') if kv_cache_utilization > 0 else 1
                    else:
                        ratio = kv_cache_utilization / starved_kv_cache_utilization
                else:
                    raise ValueError(f"Unknown migrate indicator: {self.config.psrl.sync_and_mig_strategy.mig.indicator}")
                
                if ratio > self.config.psrl.sync_and_mig_strategy.mig.threshold:
                    psrl_logger.info(f"Instance {instance_id} (version {self.instance_to_version_after_sync[instance_id]}) has a ratio of {ratio} for migrating to instance {starved_instance_id} (version {self.instance_to_version_after_sync[starved_instance_id]})")
                    candidate_migrate_instance_ids.append((instance_id, ratio))
    
        # We choose the instance with the highest ratio to migrate
        # TODO(lhy): support multiple instances to migrate and finer-grained migration strategy
        # Currently, we only support one instance to migrate, and all the requests on the instance will be interrupted and looped back to the router.    
        if len(candidate_migrate_instance_ids) > 0:
            candidate_migrate_instance_ids.sort(key=lambda x: x[1], reverse=True)
            migrate_instance_id = candidate_migrate_instance_ids[0][0]
            if self.config.psrl.sync_and_mig_strategy.mig.stop_indicator == "request_num":
                request_num = self.route_strategy.instance_to_engine_status[migrate_instance_id].get_waiting_and_running_queue_size()
                if request_num < self.config.psrl.sync_and_mig_strategy.mig.stop_threshold:
                    return []
            elif self.config.psrl.sync_and_mig_strategy.mig.stop_indicator == "throughput":
                throughput = self.route_strategy.instance_to_engine_status[migrate_instance_id].get_generation_throughput()
                if throughput < self.config.psrl.sync_and_mig_strategy.mig.stop_threshold:
                    return []
            elif self.config.psrl.sync_and_mig_strategy.mig.stop_indicator == "kv_cache":
                kv_cache_utilization = self.route_strategy.instance_to_engine_status[migrate_instance_id].get_kv_cache_utilization()
                if kv_cache_utilization < self.config.psrl.sync_and_mig_strategy.mig.stop_threshold:
                    return []
            else:
                raise ValueError(f"Unknown stop indicator: {self.config.psrl.sync_and_mig_strategy.mig.stop_indicator}")
            return [migrate_instance_id]
        return []
    
    async def check_should_sync(self, instance_id: int, ps_model_version: int) -> bool:
        """Check if the instance should synchronize with PS.
        
        Args:
            instance_id (int): The instance ID to synchronize with.
            ps_model_version (int): The version of the PS model to synchronize with.
        
        Returns:
            bool: True if the instance should synchronize with PS, False otherwise.
        """
        # psrl_logger.info(f"Checking if the instance {instance_id} (version {self.instance_to_version_after_sync[instance_id]}) should synchronize with PS (PS model version: {ps_model_version})")
        # If there are requests in the waiting queue, we will not attempt to synchronize with PS since the instance is still busy.
        if self.route_strategy.instance_to_engine_status[instance_id].get_waiting_queue_size() > 0:
            return False
        
        # 1. Check if there are any requests version satisfies the condition before synchronization
        current_instance_version = await self.ps_manager_handle.get_rollout_instance_model_version.remote(instance_id)
        filtered_requests = self.requests_to_route.filter_by_condition(
            lambda request: request.non_tensor_batch.get("version_tag", [current_instance_version + 1])[0] <= current_instance_version
            or request.non_tensor_batch.get("min_version_limit", [current_instance_version + 1 + self.staleness])[0] <= current_instance_version + self.staleness,
        )
        # 2. If enabled dynamic version tag, check if there are any requests that can be RESERVED for the instance but no need to reserve new entry before synchronization
        if self.config.psrl.routing_strategy.enable_dynamic_version_tag:
            filtered_request_ids = [request.non_tensor_batch["uid"][0] for request in filtered_requests]
            can_reserve_without_new_reserve_entry = await self.ps_manager_handle.can_reserve_request.remote(filtered_request_ids, [current_instance_version], without_new_reserve_entry=True)
            filtered_requests = [request for i, request in enumerate(filtered_requests) if can_reserve_without_new_reserve_entry[i] == [True]]
        
        # If there are requests that can still be routed to the instance before synchronization, we will not attempt to synchronize with PS
        if len(filtered_requests) > 0:
            # psrl_logger.info(f"{len(filtered_requests)} requests can still be routed to the instance {instance_id} before synchronization")
            return False
        
        # 3. Check indicator to determine whether to synchronize with PS
        if self.config.psrl.sync_and_mig_strategy.sync.indicator == "request_num":
            # Check whether request num is above threshold
            request_num = self.route_strategy.instance_to_engine_status[instance_id].get_waiting_and_running_queue_size()
            psrl_logger.debug(f"Instance {instance_id} (version {self.instance_to_version_after_sync[instance_id]}) "
                              f"request_num: {request_num}, threshold: {self.config.psrl.sync_and_mig_strategy.sync.threshold}")
            if request_num > self.config.psrl.sync_and_mig_strategy.sync.threshold:
                return False
        elif self.config.psrl.sync_and_mig_strategy.sync.indicator == "throughput":
            # Check whether throughput is above threshold
            throughput = self.route_strategy.instance_to_engine_status[instance_id].get_generation_throughput()
            psrl_logger.debug(f"Instance {instance_id} (version {self.instance_to_version_after_sync[instance_id]}) "
                              f"throughput: {throughput}, threshold: {self.config.psrl.sync_and_mig_strategy.sync.threshold}")
            if throughput > self.config.psrl.sync_and_mig_strategy.sync.threshold:
                return False
        elif self.config.psrl.sync_and_mig_strategy.sync.indicator == "kv_cache":
            # Check whether KV Cache is above threshold
            kv_cache_utilization = self.route_strategy.instance_to_engine_status[instance_id].get_kv_cache_utilization()
            psrl_logger.debug(f"Instance {instance_id} (version {self.instance_to_version_after_sync[instance_id]}) "
                              f"kv_cache_utilization: {kv_cache_utilization}, threshold: {self.config.psrl.sync_and_mig_strategy.sync.threshold}")
            if kv_cache_utilization > self.config.psrl.sync_and_mig_strategy.sync.threshold:
                return False
        elif self.config.psrl.sync_and_mig_strategy.sync.indicator == "hypothesis_test":
            # TODO(lhy): Implement hypothesis test after refactor
            # We attempt to synchronize with PS and check if there is any benefit from synchronization
            raise NotImplementedError("Hypothesis test is not implemented")
            '''
            new_filtered_requests = self.requests_to_route.filter_by_condition(
                lambda request: request.non_tensor_batch.get("version_tag", [ps_model_version + 1])[0] <= ps_model_version
                or request.non_tensor_batch.get("min_version_limit", [ps_model_version + 1 + self.staleness])[0] <= ps_model_version + self.staleness,
            )
            # psrl_logger.info(f"{len(new_filtered_requests)} requests may be able to be routed to the instance {instance_id} after synchronization, checking routing benefit...")
            for request in new_filtered_requests:
                if self.route_strategy.calculate_routing_benefit(request, instance_id) > 0:
                    return True
            # psrl_logger.info(f"No requests will benefit from routing to the instance {instance_id} after synchronization")
            '''   
        else:
            raise ValueError(f"Unknown sync indicator: {self.config.psrl.sync_and_mig_strategy.sync.indicator}")
        
        return True
    
    async def wait_interrupted_partial_requests_loop_back(self, instance_ids: List[int]):
        """Wait for the interrupted partial requests to be looped back in the priority queue.
        
        Args:
            instance_ids (List[int]): The instance IDs to wait for.
        """
        finished_instance_ids = set()
        psrl_logger.info("Waiting for the interrupted partial requests to be looped back in the priority queue")
        while True:
            for instance_id in instance_ids:
                if instance_id not in finished_instance_ids and len(self.instance_to_inflight_request_ids[instance_id]) == 0:
                    finished_instance_ids.add(instance_id)
            if len(finished_instance_ids) == len(instance_ids):
                break
            psrl_logger.info(f"Waiting for the interrupted partial requests to be looped back in the priority queue, instance to inflight requests: {self.instance_to_inflight_request_ids}")
            await asyncio.sleep(0.1)
        psrl_logger.info(f"The interrupted partial requests are looped back in the priority queue")
       