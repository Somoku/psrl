import asyncio
import logging
import os
import time

import numpy as np
import ray
from omegaconf import DictConfig
from tensordict import TensorDict
from verl import DataProto

from psrl.utils.logger import DualOutputHandler, EventType, deprecated, log_dual_events
from psrl.utils.ray import AsyncBusyPollingRayLock
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
            self.balanced_concurrent_seqs_per_instance = (
                self.config.psrl.redundant_rollout.redundant_global_batch_size
                * self.rollout_n
                // self.config.psrl.deployment.n_rollout_instances
            )
        else:
            self.rollout_n = self.config.gen_actor_rollout_ref.rollout.n
            self.alg_rollout_n = self.rollout_n
            self.balanced_concurrent_seqs_per_instance = (
                self.config.psrl.staleness_buffer_entries
                * self.rollout_n
                // self.config.psrl.deployment.n_rollout_instances
            )
        self.ps_manager_handle = ps_manager_handle
        self.rollout_wg_list = rollout_wg_list
        self.rollout_wg_size = len(rollout_wg_list)
        assert self.rollout_wg_size == self.config.psrl.deployment.n_rollout_instances, (
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
        self.routing_lock = asyncio.Lock()
        self.routing_status_update_event = asyncio.Event()
        self._is_routing = False
        self._interrupt_routing = False
        self.scheduler_task = None  # Will be created in async context
        # Track the inflight request ids for each instance (i.e., request that is being generated
        # and is not yet completed or queued in the priority queue): {instance_id: [request_id, ...]}
        self.instance_to_inflight_request_ids = {i: [] for i in range(self.rollout_wg_size)}
        # Track the instance id for each incomplete request (i.e., request that is not completed yet):
        # {request_id: instance_id}
        self.incomplete_request_to_instance = {}
        self.request_futures = {}  # Track request futures: {request_id: Future}
        # Track the version after synchronization for each instance: {instance_id: ps_model_version}
        self.instance_to_version_after_sync = {i: 0 for i in range(self.rollout_wg_size)}

        # Build logger
        self.log_prefix = "RolloutRouter"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))
        psrl_logger.info("Initialized RolloutRouter")

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

        # Notify the scheduler that status has been updated
        if len(filtered_instance_ids) > 0:
            self.routing_status_update_event.set()

    async def update_currently_syncing_instances(self, instance_ids: list[int], ps_model_version: int):
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
        assert "version_tag" in request.non_tensor_batch, (
            "Request must have 'version_tag' for routing"
        )
        needed_model_version = request.non_tensor_batch["version_tag"][0]

        # 1. Filter the rollout instances that can tolerate the needed staleness of the request
        # This guarantees that the gen worker will have no ahead-of-time version tag when generating
        candidates = [
            i for i, version in self.instance_to_version_after_sync.items() if version >= needed_model_version
        ]
        # psrl_logger.info(f"Candidates for request {request_id}: {candidates}")

        # 2. If forbidden global migration and the request is a partial rollout request,
        # only consider the specific instance for routing
        if "rollout_instance_id" in request.non_tensor_batch and not self.config.psrl.sync_and_mig_strategy.mig.enable:
            old_instance_id = request.non_tensor_batch["rollout_instance_id"][0]
            assert old_instance_id in candidates, f"Old rollout instance {old_instance_id} is not in the candidates"
            candidates = [old_instance_id]

        # 3. If forbidden group sampling on multiple instances, only consider the
        # instance that other requests in the same group are already routed to
        enable_multi_instance_group = self.config.psrl.routing_strategy.enable_group_sampling_on_multi_instances
        if not enable_multi_instance_group:
            group_request_instance_ids = [
                instance_id
                for incomplete_request_id, instance_id in self.incomplete_request_to_instance.items()
                if incomplete_request_id // self.rollout_n == request_id // self.rollout_n
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
            all_candidate_model_versions = list(
                set([self.instance_to_version_after_sync[candidate] for candidate in candidates])
            )
            can_reserve_results = ray.get(
                self.ps_manager_handle.can_reserve_request.remote(request_id, all_candidate_model_versions)
            )
            candidates = [
                candidate
                for candidate in candidates
                if can_reserve_results[
                    all_candidate_model_versions.index(self.instance_to_version_after_sync[candidate])
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
            # Use the (reserve_indicator, version) pair as the final indicator
            all_candidate_model_versions = list(
                set([self.instance_to_version_after_sync[candidate] for candidate in candidates])
            )
            indicator_results = ray.get(
                self.ps_manager_handle.get_reserve_indicator.remote(request_id, all_candidate_model_versions)
            )
            for candidate in candidates:
                version = self.instance_to_version_after_sync[candidate]
                if needed_model_version == -1:
                    version_indicator = -version
                else:
                    version_indicator = version
                reserve_indicator = indicator_results[all_candidate_model_versions.index(version)]
                candidate_indicator_list.append((reserve_indicator, version_indicator))
        else:
            raise ValueError(f"Invalid candidate sort indicator: {self.config.psrl.routing_strategy.candidate_sort_indicator}")
        route_kwargs = {"candidate_indicator_list": candidate_indicator_list}

        # 6. Strategy-based routing
        chosen_rollout_instance = self.route_strategy.route(request, candidates=candidates, route_kwargs=route_kwargs)
        # psrl_logger.info(
        #     f"Chosen rollout instance for request {request_id} "
        #     f"among candidates {candidates}: {chosen_rollout_instance}"
        # )

        # 7. If not None, the request is routed to the chosen rollout instance
        if chosen_rollout_instance is not None:
            # Allocate the version tag and reserve the request for the chosen
            # rollout instance if the request is not routed before
            not_routed_before = "rollout_instance_id" not in request.non_tensor_batch
            if not_routed_before:
                needed_model_version = self.instance_to_version_after_sync[chosen_rollout_instance]
                request.non_tensor_batch["version_tag"] = np.array([needed_model_version], dtype=int)
                # psrl_logger.info(
                #     f"Reserving request {request_id} for rollout instance "
                #     f"{chosen_rollout_instance} with version tag {needed_model_version}"
                # )
                ray.get(
                    self.ps_manager_handle.reserve_rollout_instance_requests.remote(
                        rollout_instance_ids=chosen_rollout_instance,
                        request_ids=request_id,
                        model_versions=needed_model_version,
                    )
                )
            # Otherwise, the request is already reserved
            # Only need to update the request instance id
            else:
                # psrl_logger.info(f"Updating request {request_id} to rollout instance {chosen_rollout_instance}")
                ray.get(
                    self.ps_manager_handle.update_request_instance_id.remote(
                        request_id=request_id,
                        new_instance_id=chosen_rollout_instance,
                    )
                )
        else:
            # psrl_logger.info(f"Request {request_id} is not routed to any rollout instance")
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
        assert "rollout_instance_id" not in request.non_tensor_batch, (
            "Rollout instance ID should not be provided in the original request"
        )
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
        update_status_success = await self.ps_manager_handle.update_request_status.remote(
            [request_id],
            PSRL_RequestStatus.ROLLOUT_ROUTING,
        )
        if not update_status_success[0]:
            # Means the request is aborted
            return None

        # Create a future to track this request's completion
        result_future = asyncio.Future()
        # Store the future in a way that the scheduler can access it
        self.request_futures[request_id] = result_future
        # Add request to priority queue
        self.requests_to_route.put(request)
        # psrl_logger.info(f"Adding request {request_id} to priority queue")
        self.routing_status_update_event.set()
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
        if request_id in self.incomplete_request_to_instance:
            self.incomplete_request_to_instance.pop(request_id)

    def is_routing(self) -> bool:
        """Check if the router is currently routing requests."""
        return self._is_routing

    async def interrupt_routing(self):
        """Interrupt the routing."""
        async with self.routing_lock:
            psrl_logger.info("Interrupting routing")
            self._interrupt_routing = True

    async def resume_routing(self):
        """Resume the routing."""
        async with self.routing_lock:
            psrl_logger.info("Resuming routing")
            self._interrupt_routing = False
        self.routing_status_update_event.set()

    async def _single_priority_queue_routing_loop(self):
        """Continuous routing loop for a single priority queue.

        This loop processes requests from the single priority queue.
        """
        psrl_logger.info("Started single priority queue routing loop")
        while True:
            # Process all requests in the priority queue
            self._is_routing = False
            # psrl_logger.info("Trying to acquire lock")
            async with (
                self.routing_lock,
                AsyncBusyPollingRayLock(self.ps_manager_handle),
            ):
                # psrl_logger.info("Acquired lock")
                while not self.requests_to_route.empty() and not self._interrupt_routing:
                    self._is_routing = True
                    request = self.requests_to_route.pop()
                    assert request is not None, "Request should not be None in priority queue"
                    request_id = request.non_tensor_batch["uid"][0]
                    assert request_id in self.request_futures, f"Request {request_id} should be in request futures"
                    # psrl_logger.info(f"Checking if request {request_id} is aborted")
                    if ray.get(self.ps_manager_handle.check_aborted_requests.remote(request_id, remove=True)):
                        # Indicate that the request is aborted
                        # psrl_logger.info(f"Request {request_id} in single priority queue is aborted")
                        self._set_result(request_id, None)
                        continue
                    has_instance_id = "rollout_instance_id" in request.non_tensor_batch
                    old_instance_id = request.non_tensor_batch["rollout_instance_id"][0] if has_instance_id else None
                    new_instance_id = self._choose_new_rollout_instance(request)
                    # psrl_logger.info(f"Choosing rollout instance for request {request_id} to {new_instance_id}")
                    if new_instance_id is None:
                        # Indicate that we cannot find a suitable rollout instance
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
                    # Create a task to process this request
                    task_coro = self._route_single_request(request, old_instance_id, new_instance_id)
                    task = asyncio.create_task(task_coro)
                    # To avoid silent error in async tasks
                    task.add_done_callback(lambda f: f.result())
            """
            if is_stuck:
                self._is_routing = False
                self.routing_status_update_event.clear()
                await self.routing_status_update_event.wait()
            else:
                await asyncio.sleep(0)
            """
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
            is_stuck = True
            # psrl_logger.info("Trying to acquire lock")
            async with (
                self.routing_lock,
                AsyncBusyPollingRayLock(self.ps_manager_handle),
            ):
                # psrl_logger.info("Acquired lock")
                self.requests_to_route.remove_empty_queues()
                remain_requests = []
                for queue_id, request_queue in self.requests_to_route.iter_queues():
                    if len(remain_requests) != 0:
                        # Method 1: If the last queue still has requests, we will not process the other queues
                        break
                        # Method 2: Try to process the other queues
                        # remain_requests.clear()
                    # psrl_logger.info(
                    #     f"Processing requests in priority queue {queue_id}, "
                    #     f"there are {request_queue.size()} requests in the queue"
                    # )
                    while not request_queue.empty() and not self._interrupt_routing:
                        request = request_queue.pop()
                        assert request is not None, "Request should not be None in priority queue"
                        request_id = request.non_tensor_batch["uid"][0]
                        # psrl_logger.info(f"Processing request {request_id} in priority queue {queue_id}")
                        assert request_id in self.request_futures, f"Request {request_id} should be in request futures"
                        if ray.get(self.ps_manager_handle.check_aborted_requests.remote(request_id, remove=True)):
                            # Indicate that the request is aborted
                            # psrl_logger.info(f"Request {request_id} in multi priority queue is aborted")
                            self._set_result(request_id, None)
                            continue
                        has_instance_id = "rollout_instance_id" in request.non_tensor_batch
                        old_instance_id = (
                            request.non_tensor_batch["rollout_instance_id"][0] if has_instance_id else None
                        )
                        new_instance_id = self._choose_new_rollout_instance(request)
                        if new_instance_id is None:
                            remain_requests.append(request)
                            continue
                        self.incomplete_request_to_instance[request_id] = new_instance_id
                        # Create a task to process this request
                        task_coro = self._route_single_request(request, old_instance_id, new_instance_id)
                        task = asyncio.create_task(task_coro)
                        # To avoid silent error in async tasks
                        task.add_done_callback(lambda f: f.result())
                        route_num += 1
                    # psrl_logger.info(f"There are {len(remain_requests)} requests left in priority queue {queue_id}, putting them back to the queue")
                    for request in remain_requests:
                        request_queue.put(request)
            """
            if is_stuck:
                self._is_routing = False
                self.routing_status_update_event.clear()
                # psrl_logger.info("Routing is stuck, waiting for routing status update event")
                await self.routing_status_update_event.wait()
                # psrl_logger.info("Routing is resumed")
            else:
                await asyncio.sleep(0)
            """
            self._is_routing = False
            sleep_time = self.config.psrl.routing_strategy.check_interval_in_ms / 1000
            if route_num > 0:
                psrl_logger.info(f"Routing {route_num} requests in multi priority queue routing loop, time cost: {time.time() - begin_time} seconds")
            await asyncio.sleep(sleep_time)

    async def _route_single_request(self, request: DataProto, old_instance_id: int | None, new_instance_id: int):
        """Route a single request to a rollout instance.

        Args:
            request (DataProto): The request to process.
            old_instance_id (Optional[int]): The old rollout instance id that
                the request is routed to, None if not exists.
            new_instance_id (int): The new rollout instance id that the request
                will be routed to.
        """
        # Update request non-tensor batch
        # psrl_logger.info(
        #     f"Routing single request {request.non_tensor_batch['uid'][0]} "
        #     f"to rollout instance {new_instance_id}"
        # )
        request_id = request.non_tensor_batch["uid"][0]
        request.non_tensor_batch["rollout_instance_id"] = np.array([new_instance_id], dtype=int)
        assert "version_tag" in request.non_tensor_batch, (
            "Request must have 'version_tag' for routing"
        )
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

            # Generate response
            # psrl_logger.info(f"Generating response for request {request_id} on instance {new_instance_id}")
            consolidated_output, update_status = await self.rollout_wg_list[new_instance_id].execute_rank_zero_async(
                "generate_async", request
            )

            # Change engine status
            self.route_strategy.pop_request(request, new_instance_id)
            # Remove request from inflight request ids for the instance
            self.instance_to_inflight_request_ids[new_instance_id].remove(request_id)
            self.routing_status_update_event.set()

            # Check if request was interrupted and needs to be requeued
            if update_status == PSRL_RequestStatus.ROLLOUT_INTERRUPTED_BY_SCHEDULER:
                psrl_logger.debug(
                    f"Request {request_id} on instance {new_instance_id} was interrupted "
                    "by scheduler (most likely due to kv cache full and preemption), requeueing"
                )
                # Put back in priority queue for partial rollout
                # Ensure that the consolidated output has the rollout instance id recorded
                consolidated_output.non_tensor_batch["rollout_instance_id"] = np.array([new_instance_id], dtype=int)
                self.requests_to_route.put(consolidated_output)
                # No result to set since the request is not completed
                return
            elif update_status == PSRL_RequestStatus.ROLLOUT_INTERRUPTED:
                psrl_logger.debug(
                    f"Request {request_id} on instance {new_instance_id} was interrupted "
                    "(due to model synchronization when enabled partial rollout), requeueing"
                )
                # Put back in priority queue for partial rollout
                # Ensure that the consolidated output has the rollout instance id recorded
                consolidated_output.non_tensor_batch["rollout_instance_id"] = np.array([new_instance_id], dtype=int)
                self.requests_to_route.put(consolidated_output)
                # No result to set since the request is not completed
                return
            elif update_status == PSRL_RequestStatus.ROLLOUT_COMPLETED:
                response_len = consolidated_output.non_tensor_batch["response_unpadded_len"][0]
                parent_prompt_id = request_id // self.rollout_n
                psrl_logger.debug(
                    f"Request {request_id} on instance {new_instance_id} of "
                    f"parent prompt {parent_prompt_id} completed successfully, "
                    f"length is {response_len}"
                )
                result = consolidated_output
            else:
                # Means the request is aborted
                assert update_status is None, "The update status should be None if the request is aborted"
                # psrl_logger.info(f"Request {request_id} on instance {new_instance_id} is aborted")
                result = None
        else:
            # Means the request is aborted
            # psrl_logger.info(f"Request {request_id} is aborted")
            result = None

        # Set the result for the request
        self._set_result(request_id, result)

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
                    filtered_request_ids = [
                        request_id for i, request_id in enumerate(filtered_request_ids) if not is_aborted[i]
                    ]
                    can_reserve = await self.ps_manager_handle.can_reserve_request.remote(
                        filtered_request_ids,
                        [instance_version],
                        without_new_reserve_entry=False,
                    )
                    filtered_request_ids = [
                        request_id for i, request_id in enumerate(filtered_request_ids) if can_reserve[i] == [True]
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

        # If there are requests that can still be routed to the instance before synchronization without new reserve entry
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
                    finished_instance_ids.add(instance_id)
            if len(finished_instance_ids) == len(instance_ids):
                break
            await asyncio.sleep(0)
        psrl_logger.info("The interrupted partial requests are looped back in the priority queue")
