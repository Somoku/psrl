import os
import logging
import numpy as np
import heapq
import asyncio
import threading
from tensordict import TensorDict
from omegaconf import DictConfig
from typing import List, Optional, Dict, Tuple, Any
from cachetools import LRUCache

import ray

from verl import DataProto

from psrl.workers.gen.stats_collector import EngineStats
from psrl.workers.ps.request_status_tracker import PSRL_RequestStatus
from psrl.utils.logger import DualOutputHandler, EventType, log_dual_events, deprecated
from psrl.workers.agent_loop.route_strategy import (
    RouteStrategyBase, 
    RequestNumBalanceRouteStrategy,
    ThroughputBalanceRouteStrategy,
    get_route_strategy_class,
)


psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class PriorityRequestQueue:
    """A thread-safe priority queue for routing requests based on version tags."""
    
    def __init__(self, staleness: int):
        """Initialize the priority queue.
        
        Args:
            staleness (int): The staleness tolerance for version comparison.
        """
        self._queue = []
        self._staleness = staleness
        self._counter = 0  # To ensure FIFO for same priority items
    
    def _get_priority(self, request: DataProto) -> int:
        """Get the priority value for a request.
        
        Args:
            request (DataProto): The request to get priority for.
            
        Returns:
            int: Priority value (lower is higher priority).
        """
        assert len(request) == 1, "Request must be a single request"
        if "version_tag" in request.non_tensor_batch:
            return request.non_tensor_batch["version_tag"][0]
        elif "max_version_limit" in request.non_tensor_batch:
            return request.non_tensor_batch["max_version_limit"][0] - self._staleness
        else:
            raise AssertionError("Request must have either 'version_tag' or 'max_version_limit'")
    
    def put(self, request: DataProto) -> None:
        """Put a request into the priority queue.
        
        Args:
            request (DataProto): The request to enqueue.
        """
        priority = self._get_priority(request)
        # Use counter to maintain FIFO order for items with same priority
        heapq.heappush(self._queue, (priority, self._counter, request))
        self._counter += 1
    
    def pop(self) -> Optional[DataProto]:
        """Pop the highest priority request from the queue.
        
        Returns:
            Optional[DataProto]: The highest priority request, or None if queue is empty.
        """
        if self._queue:
            _, _, request = heapq.heappop(self._queue)
            return request
        return None
    
    def peek(self) -> Optional[DataProto]:
        """Peek at the highest priority request without removing it.
        
        Returns:
            Optional[DataProto]: The highest priority request, or None if queue is empty.
        """
        if self._queue:
            _, _, request = self._queue[0]
            return request
        return None
    
    def empty(self) -> bool:
        """Check if the queue is empty.
        
        Returns:
            bool: True if queue is empty, False otherwise.
        """
        return len(self._queue) == 0
    
    def size(self) -> int:
        """Get the current size of the queue.
        
        Returns:
            int: Number of items in the queue.
        """
        return len(self._queue)


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
        assert self.rollout_wg_size == self.config.psrl.deployment.n_rollout_instances, "Rollout worker group size must match the number of deployment instances"
        
        # Build logger
        self.log_prefix = f"RolloutRouter"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))
        
        # Routing related attributes
        self.priority_queue = PriorityRequestQueue(self.staleness)
        self.route_strategy_event = asyncio.Event()  # Will be set when route strategy can be updated
        self.currently_syncing_instance_ids = set() # Track the instance ids that are currently being synchronized with PS
        self.scheduler_task = None  # Will be created in async context
        self.request_futures = {}  # Track request futures: {request_id: Future}
        
        # Build logger
        self.log_prefix = f"RolloutRouter"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))
        psrl_logger.info(f"Initialized RolloutRouter")

    def init_route_strategy(self, **kwargs):
        """Initialize the route strategy for the router.
        
        Args:
            **kwargs: Keyword arguments for the route strategy.
        """
        if self.config.psrl.routing_strategy.method == "request_num_balance" or self.config.psrl.routing_strategy.method == "throuput_balance":
            assert self.config.psrl.status_collection.enable, "Status collection must be enabled when using request num balance or throughput balance routing strategy"
        n_instances = self.rollout_wg_size
        strategy_kwargs = {
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

    def update_instance_status(self, instance_to_engine_status: dict[int, EngineStats], currently_syncing_instance_ids: set[int]):
        """Update the instance status with latest information from coordinator.
        
        Args:
            instance_to_engine_status (dict[int, EngineStats]): Latest engine status information.
            currently_syncing_instance_ids (set[int]): The instance ids that are currently being synchronized with PS.
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
        self.currently_syncing_instance_ids = currently_syncing_instance_ids
        
        # Notify the scheduler that status has been updated
        self.route_strategy_event.set()
        
        # psrl_logger.info(f"Updated instance status: {self.route_strategy.instance_to_engine_status}, currently syncing instance ids: {self.currently_syncing_instance_ids}")

    def _choose_new_rollout_instance(self, request: DataProto) -> int:
        """Select the best rollout instance for handling the generation request.
        
        Args:
            request (DataProto): The request to be routed.

        Returns:
            int: Index of the selected rollout instance.
        """
        if "version_tag" in request.non_tensor_batch:
            needed_model_version = request.non_tensor_batch["version_tag"][0]
        else:
            assert "max_version_limit" in request.non_tensor_batch, "Request must have either 'version_tag' or 'max_version_limit'"
            needed_model_version = request.non_tensor_batch["max_version_limit"][0] - self.staleness
        
        # We need to filter the rollout instances that can tolerate the needed staleness of the request
        # This guarantees that the gen worker will have no ahead-of-time version tag when generating
        all_rollout_instance_to_version = ray.get(self.ps_manager_handle.get_all_rollout_instance_model_versions.remote())
        '''
        all_rollout_instance_to_version = [
            (i, self.route_strategy.instance_to_engine_status[i].model_version)
            for i in range(self.rollout_wg_size)
        ]
        '''
        # psrl_logger.info(f"All rollout instance to version: {all_rollout_instance_to_version}")
        # Filter the rollout instances that can tolerate the needed staleness of the request
        candidates = [i for i, version in all_rollout_instance_to_version.items() if version >= needed_model_version]
        if "rollout_instance_id" in request.non_tensor_batch and not self.config.psrl.routing_strategy.enable_global_migration:
            old_instance_id = request.non_tensor_batch["rollout_instance_id"][0]
            assert old_instance_id in candidates, f"Old rollout instance {old_instance_id} is not in the candidates"
            candidates = [old_instance_id]
        chosen_rollout_instance = self.route_strategy.route(request, candidates)
        return chosen_rollout_instance

    # Only used in batch gen mode
    def _get_retry_version_and_instance(
        self,
        sample_id: int,
        max_version_limit: int,
        all_rollout_instance_to_version: dict,
    ) -> (int, int):
        """Get version and rollout instance for a retried sample.
        
        Args:
            sample_id (int): Unique identifier of the sample.
            max_version_limit (int): Maximum allowed version limit.
            all_rollout_instance_to_version (dict): Available instance to version mapping.
            
        Returns:
            tuple: (version_tag, rollout_instance_id) for the sample.
        """
        filtered_rollout_instance_to_version = {
            instance_id: version 
            for instance_id, version in all_rollout_instance_to_version.items() 
            if max_version_limit - self.staleness <= version <= max_version_limit
        }
        
        if not filtered_rollout_instance_to_version:
            raise AssertionError(
                f"No available rollout instance meets the version requirement for "
                f"sample {sample_id} with max_version_limit {max_version_limit} and staleness {self.staleness}. "
                f"All instance versions: {all_rollout_instance_to_version}"
            )
        
        # Choose a rollout instance from available candidates
        candidates = list(filtered_rollout_instance_to_version.keys())
        chosen_instance_id = candidates[sample_id % len(candidates)]
        chosen_version = filtered_rollout_instance_to_version[chosen_instance_id]

        return chosen_version, chosen_instance_id

    # TODO(lhy): move this back to router again
    # since log_prob no need to transfered to vllm rollout engine many times (partial rollout)
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
                    sample_id, max_version_limit, all_rollout_instance_to_version
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
                    psrl_logger.debug(f"Dispatching requests to rollout instance {i} with request ids: {filtered_requests.non_tensor_batch['uid']}")
                    futures.append(
                        self.rollout_wg_list[i].execute_all_async("generate", filtered_requests)[0]
                    )
                rollout_results = ray.get(futures)
                
            # Process results as needed
            with log_dual_events(f"Concatenating results from {self.rollout_wg_size} rollout instances", psrl_logger, event_type=EventType.OTHER):
                results = []
                for i in range(self.rollout_wg_size):
                    consolidated_outputs, filtered_request_idxs, update_statuses = rollout_results[i]
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
            self.scheduler_task = asyncio.create_task(self._routing_loop())
            self.scheduler_task.add_done_callback(lambda f: f.result()) # To avoid silent error in async tasks
            psrl_logger.info("Started routing loop")
        
        # Create a future to track this request's completion
        request_id = request.non_tensor_batch["uid"][0]
        result_future = asyncio.Future()
        # Store the future in a way that the scheduler can access it
        self.request_futures[request_id] = result_future
        # Add request to priority queue
        self.priority_queue.put(request)
        # Wait for the request to be processed
        with log_dual_events(f"Routing request {request_id} and waiting for it to be processed", psrl_logger, level=logging.DEBUG, event_type=EventType.GEN):
            result = await result_future
        # Clean up the future
        self.request_futures.pop(request_id)
        return result
            
    async def _routing_loop(self):
        """Continuous routing loop.
        
        This loop processes requests from the priority queue
        """
        while True:
            # Process all requests in the priority queue
            # If some instances are currently being synchronized with PS, wait for them to finish
            # Otherwise, we may planning on a intermediate version that is not yet synchronized with PS
            while len(self.currently_syncing_instance_ids) == 0 and not self.priority_queue.empty():
                request = self.priority_queue.peek()
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
                    # 2. The engine status of the specific instance could be updated after one request is generated.
                    self.route_strategy_event.clear()
                    await self.route_strategy_event.wait()
                    break
                self.priority_queue.pop()
                # Create a task to process this request
                if old_instance_id is not None and new_instance_id != old_instance_id and self.config.psrl.routing_strategy.enable_global_migration:
                    psrl_logger.info(f"Migrating request {request_id} from rollout instance {old_instance_id} to {new_instance_id}")
                task = asyncio.create_task(self._route_single_request(request, old_instance_id, new_instance_id))
                task.add_done_callback(lambda f: f.result()) # To avoid silent error in async tasks
            await asyncio.sleep(0)
    
    async def _route_single_request(self, request: DataProto, old_instance_id: Optional[int], new_instance_id: int):
        """Route a single request to a rollout instance.
        
        Args:
            request (DataProto): The request to process.
            old_instance_id (Optional[int]): The old rollout instance id that the request is routed to, None if not exists.
            new_instance_id (int): The new rollout instance id that the request will be routed to.
        """  
        # Update request non-tensor batch
        request_id = request.non_tensor_batch["uid"][0]
        request.non_tensor_batch["rollout_instance_id"] = np.array([new_instance_id], dtype=int)
        if "version_tag" in request.non_tensor_batch:
            needed_model_version = request.non_tensor_batch["version_tag"][0]
        else:
            # Indicate that it is a retry request
            assert "max_version_limit" in request.non_tensor_batch, "max_version_limit is required for routing if version_tag is not provided"
            all_rollout_instance_to_version = await self.ps_manager_handle.get_all_rollout_instance_model_versions.remote()
            needed_model_version = all_rollout_instance_to_version[new_instance_id]
            request.non_tensor_batch["version_tag"] = np.array([needed_model_version], dtype=int)
           
        # Update request status
        update_status_success = await self.ps_manager_handle.update_request_status.remote(
            [request_id],
            PSRL_RequestStatus.ROLLOUT_DISPATCHED,
            model_version=request.non_tensor_batch["version_tag"].tolist(),
        )
        
        if not update_status_success[0]:
            return
        
        # psrl_logger.info(f"Routing request {request_id} to rollout instance {instance_id} with needed model version {needed_model_version}")
        # New request, need to reserve data in staleness buffer
        if old_instance_id is None:
            # Reserve data in staleness buffer
            if self.rollout_n > 1:
                await self.ps_manager_handle.reserve_rollout_instance_requests.remote(
                    rollout_instance_ids=new_instance_id,
                    request_ids=request_id,
                    model_versions=needed_model_version,
                    parent_ids=request.non_tensor_batch["parent_id"][0],
                )
            else:
                await self.ps_manager_handle.reserve_rollout_instance_requests.remote(
                    rollout_instance_ids=new_instance_id,
                    request_ids=request_id,
                    model_versions=needed_model_version,
                )
        # Partial rollout, need to update request instance id (already reserved data in staleness buffer)
        else:
            # Update request instance id
            await self.ps_manager_handle.update_request_instance_id.remote(
                request_id=request_id,
                new_instance_id=new_instance_id,
            )
            
        # Change engine status    
        self.route_strategy.push_request(request, new_instance_id)
        
        # Generate response
        consolidated_output, update_status = await self.rollout_wg_list[new_instance_id].execute_rank_zero_async("generate_async", request)
            
        # Change engine status 
        self.route_strategy.pop_request(request, new_instance_id)
        # Notify the scheduler that new request may be routed
        self.route_strategy_event.set()
            
        # Check if request was interrupted and needs to be requeued
        if update_status == PSRL_RequestStatus.ROLLOUT_INTERRUPTED:
            psrl_logger.info(f"Request {request_id} was interrupted, requeueing")
            # Put back in priority queue for partial rollout
            # Ensure that the consolidated output has the rollout instance id recorded
            consolidated_output.non_tensor_batch["rollout_instance_id"] = np.array([new_instance_id], dtype=int)
            self.priority_queue.put(consolidated_output)
            return
        elif update_status == PSRL_RequestStatus.RUNNING:
            # psrl_logger.info(f"Request {request_id} completed successfully")
            result = consolidated_output
        else:
            raise ValueError(f"Unexpected update status for request {request_id}: {update_status}")
    
        # Set the result for any waiting futures
        assert request_id in self.request_futures, f"Request {request_id} should be in request futures"
        assert not self.request_futures[request_id].done(), f"Request {request_id} should not be done"
        self.request_futures[request_id].set_result(result)
    
    