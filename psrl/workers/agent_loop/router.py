import os
import logging
import numpy as np
from tensordict import TensorDict
from omegaconf import DictConfig
from typing import List, Optional, Dict
from cachetools import LRUCache

import ray

from verl import DataProto

from psrl.workers.gen.stats_collector import EngineStats
from psrl.workers.ps.request_status_tracker import PSRL_RequestStatus
from psrl.utils.logger import DualOutputHandler, EventType, log_dual_events, deprecated
from psrl.workers.agent_loop.route_strategy import (
    RouteStrategyBase, 
    RequestNumBalanceRouteStrategy,
    get_route_strategy_class,
)

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
        else:
            self.rollout_n = self.config.gen_actor_rollout_ref.rollout.n
            self.alg_rollout_n = self.rollout_n
        self.ps_manager_handle = ps_manager_handle
        self.rollout_wg_list = rollout_wg_list
        self.rollout_wg_size = len(rollout_wg_list)

        self.rank_0_is_model_owner = self.config.gen_actor_rollout_ref.rollout.mode == "psrl_async"
        
        # Initialize route strategy
        self._init_route_strategy()
        
        # Engine status tracking
        # NOTE: The engine status will be updated by RolloutCoordinator
        # and can be accessed via get_engine_status() method from agent loop worker
        self.latest_instance_to_engine_status = Dict[int, EngineStats] # {instance_id: engine_stats}
        self.status_changed = False
        
        # Cache for sample_id to version_tag mapping with LRU eviction
        # Key: sample_id (uid // self.rollout_n), Value: version_tag
        cache_size = getattr(self.config.psrl, 'sample_version_cache_size', 64)
        self.sample_version_cache = LRUCache(maxsize=cache_size)
        psrl_logger.info(f"Initialized sample version cache with max size: {cache_size}")
        
        # Build logger
        self.log_prefix = f"RolloutRouter"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))

    def _init_route_strategy(self):
        """Initialize the routing strategy based on configuration."""
        route_strategy_name = getattr(
            self.config.gen_actor_rollout_ref.rollout.agent, 
            'route_strategy', 
            'round_robin'
        )
        
        try:
            route_strategy_class = get_route_strategy_class(route_strategy_name)
            self.route_strategy: RouteStrategyBase = route_strategy_class(self.rollout_wg_size)
            psrl_logger.info(f"Initialized route strategy: {route_strategy_name}")
        except Exception as e:
            psrl_logger.warning(f"Route strategy error: {e}")
            psrl_logger.warning(f"Falling back to 'round_robin' strategy")
            from psrl.workers.agent_loop.route_strategy import RoundRobinRouteStrategy
            self.route_strategy: RouteStrategyBase = RoundRobinRouteStrategy(self.rollout_wg_size)

    def update_instance_to_engine_status(self, instance_to_engine_status: dict[int, EngineStats]):
        """Update the engine status with latest information from coordinator.
        
        Args:
            instance_to_engine_status (dict[int, EngineStats]): Latest engine status information.
        """
        # NOTE(lhy): This method is called by RolloutCoordinator
        # Each agent loop worker contains a RolloutRouter, which shares the same engine status
        self.latest_instance_to_engine_status = instance_to_engine_status
        self.status_changed = True

    def _choose_gen_worker(self, request: DataProto, candidates: Optional[List[int]] = None) -> int:
        """Select the best worker for handling the generation request.
        
        Args:
            request (DataProto): The request to be routed.
            candidates (Optional[List[int]]): List of candidate worker indices if any.

        Returns:
            int: Index of the selected worker.
        """
        if isinstance(self.route_strategy, RequestNumBalanceRouteStrategy):
            assert self.config.psrl.status_collection.enable, "RequestNumBalanceRouteStrategy requires status collection to be enabled."
            if self.status_changed:
                self.route_strategy.update_instance_request_counts(
                    {instance_id: engine_stats.get_waiting_and_running_queue_size() for instance_id, engine_stats in self.latest_instance_to_engine_status.items()}
                )
                self.status_changed = False
            chosen_worker = self.route_strategy.route(request, candidates)
            if chosen_worker not in self.latest_instance_to_engine_status:
                self.latest_instance_to_engine_status[chosen_worker] = EngineStats(
                    instance_id=chosen_worker,
                    model_version=-1, # -1 means not initialized
                    snapshot={},
                )
            self.latest_instance_to_engine_status[chosen_worker].increment_waiting_and_running_queue_size()
        else:
            chosen_worker = self.route_strategy.route(request, candidates)
        return chosen_worker

    def _get_cached_version_and_worker(
        self,
        sample_id: int,
        max_version_limit: int,
        all_rollout_instance_to_version: dict,
    ) -> (int, int):
        """Get cached version and worker for a sample, or determine new ones.
        
        Args:
            sample_id (int): Unique identifier of the sample.
            max_version_limit (int): Maximum allowed version limit.
            all_rollout_instance_to_version (dict): Available instance to version mapping.
            
        Returns:
            tuple: (version_tag, gen_worker_idx) for the sample.
        """
        # Check if we have a cached version for this sample_id
        cached_version = self.sample_version_cache.get(sample_id)
        if cached_version is not None:
            psrl_logger.debug(f"Cache hit for sample_id {sample_id}: version {cached_version}")
            filtered_rollout_instance_to_version = {
                instance_id: version 
                for instance_id, version in all_rollout_instance_to_version.items() 
                if version == cached_version
            }
        else:
            psrl_logger.debug(f"Cache miss for sample_id {sample_id}")
            filtered_rollout_instance_to_version = {
                instance_id: version 
                for instance_id, version in all_rollout_instance_to_version.items() 
                if max_version_limit - self.staleness <= version <= max_version_limit
            }
        
        if not filtered_rollout_instance_to_version:
            raise AssertionError(
                f"No available rollout instance meets the version requirement for "
                f"sample {sample_id} with max_version_limit {max_version_limit} and staleness {self.staleness}. "
                f"All instance versions: {all_rollout_instance_to_version} and "
                f"{'cached_version = ' + str(cached_version) if cached_version is not None else 'no cached version'}"
            )
        
        # Choose a worker from available candidates
        candidates = list(filtered_rollout_instance_to_version.keys())
        chosen_worker_idx = candidates[sample_id % len(candidates)]
        chosen_version = filtered_rollout_instance_to_version[chosen_worker_idx]
        
        # Cache the result using LRU cache
        self.sample_version_cache[sample_id] = chosen_version

        return chosen_version, chosen_worker_idx

    @deprecated("It is moved to the `post_process_outputs_lite` inside vllm rollout now")
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
        request_ids = requests.non_tensor_batch.get("uid", None)
        if "max_version_limit" in requests.non_tensor_batch:
            # Indicate that these requests are retry requests
            max_version_limit = requests.non_tensor_batch["max_version_limit"][0]
            assert all(v == max_version_limit for v in requests.non_tensor_batch["max_version_limit"]), "All requests in the batch must have the same max_version_limit."
            
            requests.non_tensor_batch.pop("max_version_limit")
            
            all_rollout_instance_to_version = ray.get(self.ps_manager_handle.get_all_rollout_instance_model_versions.remote())
            
            # Group requests by sample_id and assign versions using cache
            sample_to_requests = {}
            for i, uid in enumerate(request_ids):
                sample_id = uid // self.rollout_n
                if sample_id not in sample_to_requests:
                    sample_to_requests[sample_id] = []
                sample_to_requests[sample_id].append(i)
            
            # Assign version and worker for each sample_id
            requests_list = []
            for sample_id, request_indices in sample_to_requests.items():
                needed_model_version, gen_worker_idx = self._get_cached_version_and_worker(
                    sample_id, max_version_limit, all_rollout_instance_to_version
                )
                # Create a sub-batch for this sample_id
                sample_requests = requests.select_idxs(request_indices)
                sample_requests.non_tensor_batch["rollout_instance_id"] = np.array([gen_worker_idx] * len(request_indices), dtype=int)
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
            # evenly dispatch to rollout workers
            futures = []
            with log_dual_events(f"Dispatching {len(requests)} requests to {self.rollout_wg_size} rollout workers evenly and generate in batch", psrl_logger, event_type=EventType.GEN):
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
                    if self.rollout_n > 1:
                        parent_ids = np.unique(filtered_requests.non_tensor_batch["parent_id"]).tolist()
                        ray.get(self.ps_manager_handle.reserve_rollout_instance_requests.remote(
                            rollout_instance_ids=filtered_requests.non_tensor_batch["rollout_instance_id"].tolist(),
                            request_ids=filtered_requests.non_tensor_batch["uid"].tolist(),
                            model_versions=filtered_requests.non_tensor_batch["version_tag"].tolist(),
                            parent_ids=parent_ids,
                        ))
                    else:
                        ray.get(self.ps_manager_handle.reserve_rollout_instance_requests.remote(
                            rollout_instance_ids=filtered_requests.non_tensor_batch["rollout_instance_id"].tolist(),
                            request_ids=filtered_requests.non_tensor_batch["uid"].tolist(),
                            model_versions=filtered_requests.non_tensor_batch["version_tag"].tolist(),
                        ))

                for i, filtered_requests in enumerate(filtered_requests_list):
                    psrl_logger.debug(f"Dispatching requests to rollout worker {i} with request ids: {filtered_requests.non_tensor_batch['uid']}")
                    futures.append(
                        self.rollout_wg_list[i].execute_all_async("generate", filtered_requests)[0]
                    )
                rollout_results = ray.get(futures)
            # Process results as needed
            with log_dual_events(f"Concatenating results from {self.rollout_wg_size} rollout workers", psrl_logger, event_type=EventType.OTHER):
                results = []
                for i in range(self.rollout_wg_size):
                    consolidated_outputs, filtered_request_idxs, update_statuses = rollout_results[i]
                    if consolidated_outputs is None:
                        continue
                    psrl_logger.debug(f"Consolidated outputs from rollout worker {i} have request ids: {consolidated_outputs.non_tensor_batch['uid']}")
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

        request_ids = request.non_tensor_batch.get("uid", None)
        if "max_version_limit" in request.non_tensor_batch:
            # Indicate that this request is retry
            max_version_limit = request.non_tensor_batch.pop("max_version_limit")[0]
            all_rollout_instance_to_version = await self.ps_manager_handle.get_all_rollout_instance_model_versions.remote()
            
            # Use cache to get version and worker for this sample
            sample_id = request_ids[0] // self.rollout_n
            needed_model_version, gen_worker_idx = self._get_cached_version_and_worker(
                sample_id, max_version_limit, all_rollout_instance_to_version
            )
            
            request.non_tensor_batch["rollout_instance_id"] = np.array([gen_worker_idx], dtype=int)
            request.non_tensor_batch["version_tag"] = np.array([needed_model_version], dtype=int) # Refactor the version tag (previouly is assigned by uid in agent loop manager)
        update_status_success = await self.ps_manager_handle.update_request_status.remote(
            request_ids.tolist(),
            PSRL_RequestStatus.ROLLOUT_DISPATCHED,
            model_version=request.non_tensor_batch.get("version_tag", np.array([-1], dtype=int)).tolist(),
        )
        if update_status_success[0]:
            request_id = request.non_tensor_batch["uid"][0]
            if "rollout_instance_id" in request.non_tensor_batch:
                gen_worker_idx = request.non_tensor_batch["rollout_instance_id"][0]
            else:
                gen_worker_idx = self._choose_gen_worker(request)
            needed_model_version = request.non_tensor_batch["version_tag"][0]
            request.non_tensor_batch["rollout_instance_id"] = np.array([gen_worker_idx], dtype=int)
            # Reserve data in staleness buffer
            if self.rollout_n > 1:
                await self.ps_manager_handle.reserve_rollout_instance_requests.remote(
                    rollout_instance_ids=gen_worker_idx,
                    request_ids=request_id,
                    model_versions=needed_model_version,
                    parent_ids=request.non_tensor_batch["parent_id"][0],
                )
            else:
                await self.ps_manager_handle.reserve_rollout_instance_requests.remote(
                    rollout_instance_ids=gen_worker_idx,
                    request_ids=request_id,
                    model_versions=needed_model_version,
                )
            # NOTE(linsh): we use push/pop task to manage the lifecycle of the request in case of interruption
            if self.rank_0_is_model_owner:
                await self.rollout_wg_list[gen_worker_idx].execute_rank_zero_async("push_task", request_id, needed_model_version)
            else:
                await self.rollout_wg_list[gen_worker_idx].execute_all_async("push_task", request_id, needed_model_version)
            
            continue_generation = True
            while continue_generation:
                if self.rank_0_is_model_owner:
                    consolidated_output, update_status = await self.rollout_wg_list[gen_worker_idx].execute_rank_zero_async("generate_async", request)
                else:
                    consolidated_output_list, update_status_list = await self.rollout_wg_list[gen_worker_idx].execute_all_async("generate_async", request)
                    consolidated_output = consolidated_output_list[0]
                    update_status = update_status_list[0]
                if consolidated_output is None:
                    if self.rank_0_is_model_owner:
                        await self.rollout_wg_list[gen_worker_idx].execute_rank_zero_async("pop_task", request_id, needed_model_version)
                    else:
                        await self.rollout_wg_list[gen_worker_idx].execute_all_async("pop_task", request_id, needed_model_version)
                    return None
                
                request = consolidated_output
                if update_status == PSRL_RequestStatus.RUNNING:
                    break
                elif update_status == PSRL_RequestStatus.ROLLOUT_INTERRUPTED:
                    continue
    
            psrl_logger.debug(f"Generation completed for request {request_id} on gen worker {gen_worker_idx} with model version {needed_model_version}")
            if self.rank_0_is_model_owner:
                await self.rollout_wg_list[gen_worker_idx].execute_rank_zero_async("pop_task", request_id, needed_model_version)
            else:
                await self.rollout_wg_list[gen_worker_idx].execute_all_async("pop_task", request_id, needed_model_version)

            return request
        return None
