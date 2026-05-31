import asyncio
import logging
import os
import time

import ray
from omegaconf import DictConfig
from vllm.sampling_params import RequestOutputKind

from psrl.utils.elastic_rm.diagnostics import log_elastic_rm_backlog_diag
from psrl.utils.logger import DualOutputHandler, EventType, log_dual_events
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
from psrl.workers.gen_dplb.stats_collector import EngineStats
from psrl.workers.gen_dplb.utils import RolloutInstanceId, TokenInput, TokenOutput
from psrl.workers.ps.request_status_tracker import PSRL_RequestStatus

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@ray.remote(concurrency_groups={"control": 1})
class RolloutRouter:
    def __init__(
        self,
        config: DictConfig,
        ps_manager_handle,
        tokenizer,
    ):
        """Initialize the rollout router.
        Managing rollout requests across multiple worker groups.
        Handles request routing, load balancing, and consolidation of generation results.

        Args:
            config (DictConfig): Configuration containing rollout settings.
            ps_manager_handle: Handle to the parameter server manager.
        """
        self.config = config
        self.staleness = self.config.psrl.staleness
        self.server_handles = {}
        self.n_rollout_instances = 0
        self.instance_ids = set()
        self.val_instance_ids = set()

        if self.config.psrl.redundant_rollout.enable:
            self.rollout_n = self.config.psrl.redundant_rollout.redundant_rollout_n
            self.alg_rollout_n = self.config.psrl.redundant_rollout.alg_rollout_n
        else:
            self.rollout_n = self.config.gen_actor_rollout_ref.rollout.n
            self.alg_rollout_n = self.rollout_n

        self.val_rollout_n = self.config.train_actor_rollout_ref.rollout.val_kwargs.n
        self.ps_manager_handle = ps_manager_handle
        self.tokenizer = tokenizer

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
        self.instance_to_inflight_request_ids = {}

        # Track the instance id for each incomplete request (i.e., request that is not completed yet):
        # {request_id: instance_id}
        self.incomplete_request_to_instance = {}

        self.request_futures = {}  # Track request futures: {request_id: Future}

        # Track the version after synchronization for each instance: {instance_id: ps_model_version}
        self.instance_to_version_after_sync: dict[RolloutInstanceId, int] = {}

        # Track the instance ids that are currently paused (not available for routing)
        self.currently_paused_instance_ids = set()

        # Track requests in sticky session: {request_id: bool}
        self.sticky_session_requests = {}

        self.instance_to_tp_pp = {}

        self._init_route_strategy()

        # NOTE(linsh): Store the partial request outputs
        # TODO(linsh): optimize and extend to more fields after
        # combining TransferQueue
        self.partial_request_output_store = {}

        # Build logger
        self.log_prefix = "RolloutRouter"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))
        psrl_logger.info("Initialized RolloutRouter")

    def _init_route_strategy(self, **kwargs):
        """Initialize the route strategy for the router."""
        if (
            self.config.psrl.routing_strategy.method == "request_num_balance"
            or self.config.psrl.routing_strategy.method == "throughput_optimal"
        ):
            assert self.config.psrl.status_collection.enable, (
                "Status collection must be enabled when using request num "
                "balance or throughput optimal routing strategy"
            )
        strategy_kwargs = {
            "logging_interval_in_ms": self.config.psrl.routing_strategy.logging_interval_in_ms,
            "cost_model_path": self.config.psrl.routing_strategy.cost_model_path,
            "max_num_waiting_reqs_after_preemption": (
                self.config.psrl.routing_strategy.max_num_waiting_reqs_after_preemption
            ),
            "max_concurrent_seqs_per_instance": (self.config.psrl.routing_strategy.max_concurrent_seqs_per_instance),
            "delta_throughput_threshold": self.config.psrl.routing_strategy.delta_throughput_threshold,
            "max_prompt_length": self.config.data.max_prompt_length,
            "request_budget": self.config.psrl.routing_strategy.request_budget,
            "snapshot_staleness_threshold_in_ms": self.config.psrl.routing_strategy.snapshot_staleness_threshold_in_ms,
            "logger": psrl_logger,
            **kwargs,
        }

        # Init without instance num first, will be updated during `add_worker`
        try:
            route_strategy_class = get_route_strategy_class(self.config.psrl.routing_strategy.method)
            self.route_strategy: RouteStrategyBase = route_strategy_class(strategy_kwargs)
            psrl_logger.info(f"Initialized route strategy: {self.config.psrl.routing_strategy.method}")
        except Exception as e:
            psrl_logger.warning(f"Route strategy error: {e}")
            psrl_logger.warning("Falling back to 'round_robin' strategy")
            from psrl.workers.agent_loop.route_strategy import RoundRobinRouteStrategy

            self.route_strategy: RouteStrategyBase = RoundRobinRouteStrategy(strategy_kwargs)

    def is_routing(self) -> bool:
        """Check if the router is currently routing requests."""
        return self._is_routing

    @ray.method(concurrency_group="control")
    def get_pending_request_count(self) -> int:
        """Return current number of requests waiting in router queue."""
        t0 = time.monotonic()
        log_elastic_rm_backlog_diag(psrl_logger, "stage=RolloutRouter_enter")
        n = int(self.requests_to_route.size())
        log_elastic_rm_backlog_diag(
            psrl_logger,
            "stage=RolloutRouter_exit pending=%d body_s=%.6f",
            n,
            time.monotonic() - t0,
        )
        return n

    @ray.method(concurrency_group="control")
    async def pause_routing(self):
        self._pause_routing = True
        # NOTE: asyncio lock cannot be shared across Ray concurrency groups (each group has its
        # own event loop). Poll _is_routing instead because a plain bool read/write is safe
        # across threads under CPython's GIL.
        while self._is_routing:
            await asyncio.sleep(self.config.psrl.routing_strategy.check_interval_in_ms / 1000)
        psrl_logger.info("Pausing routing")

    @ray.method(concurrency_group="control")
    async def resume_routing(self):
        self._pause_routing = False
        psrl_logger.info("Resuming routing")

    @ray.method(concurrency_group="control")
    def add_worker(
        self,
        server_handle,
        replica_id: str,
        data_parallel_size: int,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        is_validate: bool = False,
        **kwargs,
    ):
        """Add a rollout replica server to the router.

        Args:
            server_handle: Handle to the rollout replica server.
            replica_id (str): ID of the rollout replica.
            data_parallel_size (int): Number of instances in the replica.
            tensor_parallel_size (int): Tensor parallel size of the instances.
            pipeline_parallel_size (int): Pipeline parallel size of the instances.
        """
        if kwargs.get("max_model_len", None) is None:
            psrl_logger.warning(
                "max_model_len not provided, it is recommended to pass it "
                "for better performance, fetching from server..."
            )
            # TODO(linsh): support multi dp ranks and get max of them
            max_model_len = ray.get(server_handle.estimate_max_model_len.remote())
            kwargs["max_model_len"] = max_model_len

        self.server_handles[replica_id] = server_handle
        self.instance_to_tp_pp.update(
            {(replica_id, i): f"TP{tensor_parallel_size}_PP{pipeline_parallel_size}"}
            for i in range(data_parallel_size)
        )
        new_instance_ids = [(replica_id, i) for i in range(data_parallel_size)]
        self.instance_ids.update(new_instance_ids)
        if is_validate:
            self.val_instance_ids.update(new_instance_ids)
        for new_instance_id in new_instance_ids:
            self.instance_to_inflight_request_ids.setdefault(new_instance_id, [])
        self.n_rollout_instances += data_parallel_size

        if self.config.psrl.redundant_rollout.enable:
            balanced_concurrent_seqs_per_instance = (
                self.config.psrl.redundant_rollout.redundant_global_batch_size
                * self.rollout_n
                // self.n_rollout_instances
            )
        else:
            balanced_concurrent_seqs_per_instance = (
                self.config.psrl.staleness_buffer_entries * self.rollout_n // self.n_rollout_instances
            )
        self.route_strategy.add_worker((replica_id, data_parallel_size), **kwargs)
        self.route_strategy.update_config(
            balanced_concurrent_seqs_per_instance=balanced_concurrent_seqs_per_instance,
            instance_to_tp_pp=self.instance_to_tp_pp,
        )

        return replica_id

    @ray.method(concurrency_group="control")
    async def update_instance_status(self, instance_to_engine_status: dict[RolloutInstanceId, EngineStats]):
        """Update the instance status with latest information from coordinator.

        Args:
            instance_to_engine_status (dict[RolloutInstanceId, EngineStats]): Latest engine status information.
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

        filtered_stats = {instance_id: instance_to_engine_status[instance_id] for instance_id in filtered_instance_ids}
        self.route_strategy.update_instance_to_engine_status(filtered_stats)

    @ray.method(concurrency_group="control")
    async def update_currently_syncing_instances(self, instance_ids: list[RolloutInstanceId], ps_model_version: int):
        """Update the currently syncing instances.

        Args:
            instance_ids (List[RolloutInstanceId]): The instance IDs to update.
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
    async def pause_instances(self, instance_ids: list[RolloutInstanceId]):
        """Notify the router about paused instances.

        Args:
            instance_ids (List[RolloutInstanceId]): List of instance IDs that are paused.
        """
        for instance_id in instance_ids:
            self.currently_paused_instance_ids.add(instance_id)

    @ray.method(concurrency_group="control")
    async def resume_instances(self, instance_ids: list[RolloutInstanceId]):
        """Notify the router about resumed instances.

        Args:
            instance_ids (List[RolloutInstanceId]): List of instance IDs that are resumed.
        """
        for instance_id in instance_ids:
            self.currently_paused_instance_ids.discard(instance_id)

    async def _choose_new_rollout_instance(
        self,
        request: TokenInput,
    ) -> RolloutInstanceId | None:
        assert request.version_tag is not None, "Request must have a valid version_tag (not None)"
        request_id = request.request_id
        version_tag = request.version_tag
        rollout_instance_id = request.rollout_instance_id
        is_validate = request.is_validate
        # 1. Filter the rollout instances that are not paused and can tolerate the needed staleness of the request
        # This guarantees that the gen worker will have no ahead-of-time version tag when generating
        if self.config.psrl.fuse_rollout_with_validate:
            available_instance_ids = self.instance_ids
        else:
            # If not fusing rollout with validate, separate the instance IDs for rollout and validate
            available_instance_ids = self.val_instance_ids
        available_instance_ids = available_instance_ids - self.currently_paused_instance_ids
        candidates = [
            instance_id
            for instance_id, version in self.instance_to_version_after_sync.items()
            if instance_id in available_instance_ids and version >= version_tag
        ]
        psrl_logger.debug(
            f"Routing candidates of request {request_id} is {candidates}, where "
            f"available instance: {available_instance_ids}, "
            f"instance_to_version: {self.instance_to_version_after_sync}"
        )

        # 2. If forbidden global migration and the request is a partial rollout request,
        # only consider the specific instance for routing.
        # If request is in sticky session, keep the existing instance.
        if rollout_instance_id is not None and (
            not self.config.psrl.sync_and_mig_strategy.mig.enable
            or self.sticky_session_requests.get(request_id, False)
        ):
            # TODO: implement similar skip logic for Rust gateway
            if rollout_instance_id in candidates:
                candidates = [rollout_instance_id]
            else:
                # Elastic scale/sleep may invalidate historical rollout_instance_id.
                # Degrade gracefully to current candidate set instead of crashing router loop.
                psrl_logger.warning(
                    (
                        "Old rollout instance %s is not in candidates for request %s; "
                        "fallback to normal routing. candidates=%s"
                    ),
                    rollout_instance_id,
                    request_id,
                    candidates,
                )

        # 3. If forbidden group sampling on multiple instances, only consider the
        # instance that other requests in the same group are already routed to
        rollout_n = self.val_rollout_n if is_validate else self.rollout_n
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
                    f"needed model version: {version_tag}"
                )
                candidates = [group_instance]

        # 4. Filter the rollout instances that can reserve the request for the current instance model version
        # This is only used when the needed model version is -1 (i.e. new request)
        if version_tag == -1:
            all_candidate_model_versions = list(
                set([self.instance_to_version_after_sync[candidate] for candidate in candidates])
            )
            can_reserve_results = await self.ps_manager_handle.can_reserve_request.remote(
                request_id, all_candidate_model_versions, is_validate=is_validate
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
                if version_tag == -1:
                    version_indicator = -version
                else:
                    version_indicator = version
                candidate_indicator_list.append(version_indicator)
        elif self.config.psrl.routing_strategy.candidate_sort_indicator == "reserve_capability":
            # Use the (reserve_indicator, version) pair as the final indicator
            all_candidate_model_versions = list(
                set([self.instance_to_version_after_sync[candidate] for candidate in candidates])
            )
            indicator_results = await self.ps_manager_handle.get_reserve_indicator.remote(
                request_id, all_candidate_model_versions, is_validate=is_validate
            )
            for candidate in candidates:
                version = self.instance_to_version_after_sync[candidate]
                if version_tag == -1:
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

        # 6. Strategy-based routing
        chosen_rollout_instance = self.route_strategy.route(request, candidates=candidates, route_kwargs=route_kwargs)

        # 7. If not None, the request is routed to the chosen rollout instance
        if chosen_rollout_instance is not None:
            # Allocate the version tag and reserve the request for the chosen
            # rollout instance if the request is not routed before
            if rollout_instance_id is None:
                needed_model_version = self.instance_to_version_after_sync[chosen_rollout_instance]
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

    @rollout_trace_op
    async def route_generate(
        self,
        prompt_ids: list[int],
        request_id: int,
        prompt_id: int,
        version_tag: int | None = None,
        rollout_instance_id: RolloutInstanceId | None = None,
        cu_response_len: int = 0,
        is_validate: bool = False,
        stop_token_ids: list[int] | None = None,
    ) -> TokenOutput:
        """Asynchronously generate response for a single request.

        Args:
            prompt_ids (list[int]): Input prompt token IDs.
            request_id (int): Unique identifier for the request.
            version_tag (int | None): Model version tag for the request.
            rollout_instance_id (RolloutInstanceId | None): Specific rollout instance ID for partial rollout.
            cu_response_len (int): Current response length for continuation requests.
            is_validate (bool): Whether the request is for validation.
        Returns:
            TokenOutput or None: Generated result or None if request is invalid.
        """
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

        update_status_success = await self.ps_manager_handle.update_request_status.remote(
            [request_id],
            PSRL_RequestStatus.ROLLOUT_ROUTING,
            is_validate=is_validate,
        )
        if not update_status_success:
            # Means the request is aborted
            return None

        # Create a future to track this request's completion
        result_future = asyncio.Future()
        # Store the future in a way that the scheduler can access it
        self.request_futures[request_id] = result_future
        # Add request to priority queue
        request = TokenInput(
            input_ids=prompt_ids,
            request_id=request_id,
            prompt_id=prompt_id,
            version_tag=version_tag,
            rollout_instance_id=rollout_instance_id,
            cu_response_len=cu_response_len,
            is_validate=is_validate,
            stop_token_ids=stop_token_ids,
        )
        self.requests_to_route.put(request)
        psrl_logger.info(f"Adding request {request_id} to priority queue")
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

    def _set_result(self, request_id: int, result: TokenOutput | None):
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
                    # Used field in `requests_to_route`: request_id, rollout_instance_id, version_tag, is_validate,
                    request = self.requests_to_route.pop()
                    assert request is not None, "Request should not be None in priority queue"
                    request_id = request.request_id
                    assert request_id in self.request_futures, f"Request {request_id} should be in request futures"
                    if await self.ps_manager_handle.check_aborted_requests.remote(request_id, remove=True):
                        self._set_result(request_id, None)
                        continue
                    new_instance_id = await self._choose_new_rollout_instance(request)
                    # psrl_logger.info(f"Choosing rollout instance for request {request_id} to {new_instance_id}")
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
                    # Create a task to process this request
                    task_coro = self._route_single_request(request, new_instance_id)
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
            # psrl_logger.info("Trying to acquire lock")
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
                        request_id = request.request_id
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

    async def _route_single_request(
        self,
        request: TokenInput,
        instance_id: RolloutInstanceId,
    ):
        is_validate = request.is_validate
        version_tag = request.version_tag
        rollout_n = self.val_rollout_n if is_validate else self.rollout_n
        assert version_tag != -1, (
            "The version tag should not be -1 (new request that is not "
            "allocated a version tag yet when enabled dynamic version tag) "
            "after routing"
        )

        # Update request status
        update_status_success = await self.ps_manager_handle.update_request_status.remote(
            [request.request_id],
            PSRL_RequestStatus.ROLLOUT_DISPATCHED,
            model_version=version_tag,
            is_validate=is_validate,
        )

        result = None
        if update_status_success:
            # Change engine status
            self.route_strategy.push_request(request, instance_id)
            # Add request to inflight request ids for the instance
            self.instance_to_inflight_request_ids[instance_id].append(request.request_id)

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
            if is_validate:
                val_config = self.config.train_actor_rollout_ref.rollout.val_kwargs
                sampling_params["top_k"] = val_config.top_k
                sampling_params["top_p"] = val_config.top_p
                sampling_params["temperature"] = val_config.temperature
            if request.stop_token_ids:
                sampling_params["stop_token_ids"] = list(
                    set((sampling_params.get("stop_token_ids") or []) + request.stop_token_ids)
                )

            # Generate response
            replica_id, data_parallel_rank = instance_id
            output: TokenOutput = await self.server_handles[replica_id].generate.remote(
                request.input_ids,
                sampling_params,
                request.request_id,
                data_parallel_rank=data_parallel_rank,
                version_tag=version_tag,
                is_validate=is_validate,
            )

            # Change engine status
            self.route_strategy.pop_request(request, instance_id)
            # Remove request from inflight request ids for the instance
            self.instance_to_inflight_request_ids[instance_id].remove(request.request_id)

            # Check if request was interrupted and needs to be requeued
            update_status = output.update_status
            if update_status == PSRL_RequestStatus.ROLLOUT_INTERRUPTED:
                psrl_logger.debug(
                    f"Request {request.request_id} on instance {instance_id} was interrupted, requeueing"
                )
                # Put back in priority queue for partial rollout
                # Ensure that the consolidated output has the rollout instance id recorded
                output.rollout_instance_id = instance_id
                self._store_partial_rollout_output(request, output)
                updated_input = self._update_request_input(request, output)
                self.requests_to_route.put(updated_input)
                # No result to set since the request is not completed
                return
            elif update_status == PSRL_RequestStatus.ROLLOUT_COMPLETED:
                response_len = len(output.token_ids)
                parent_prompt_id = request.request_id // rollout_n
                psrl_logger.debug(
                    f"Request {request.request_id} on instance {instance_id} of "
                    f"parent prompt {parent_prompt_id} completed successfully, "
                    f"length is {response_len}"
                )
                result = self._consolidate_request_output(output, request)
            else:
                # Means the request is aborted
                assert update_status is None, "The update status should be None if the request is aborted"
                # psrl_logger.info(f"Request {request.request_id} on instance {new_instance_id} is aborted")
                result = None

        # Set the result for the request
        self._set_result(request.request_id, result)

    def _update_request_input(self, request: TokenInput, output: TokenOutput) -> TokenInput:
        """Update the request input based on the output for partial rollout.

        Args:
            request (TokenInput): The original request input.
            output (TokenOutput): The output from the rollout engine.
        Returns:
            TokenInput: The updated request input for partial rollout.
        """
        # Update the input ids by appending the generated token ids
        new_input_ids = request.input_ids + output.token_ids
        # Update the current unpadded response length
        new_cu_response_len = request.cu_response_len + len(output.token_ids)
        # Create a new TokenInput with updated fields
        updated_request = TokenInput(
            input_ids=new_input_ids,
            request_id=request.request_id,
            prompt_id=request.prompt_id,
            version_tag=request.version_tag,
            rollout_instance_id=output.rollout_instance_id,
            cu_response_len=new_cu_response_len,
            is_validate=request.is_validate,
            stop_token_ids=request.stop_token_ids,
        )
        return updated_request

    def _store_partial_rollout_output(
        self,
        request: TokenInput,
        output: TokenOutput,
    ):
        """Store the partial rollout output for consolidation later.

        Args:
            request (TokenInput): The original request input.
            output (TokenOutput): The output from the rollout engine.
        """
        request_id = request.request_id
        if request_id not in self.partial_request_output_store:
            self.partial_request_output_store[request_id] = {
                "log_probs": [],
            }
        self.partial_request_output_store[request_id]["log_probs"].extend(output.log_probs)

    def _consolidate_request_output(self, output: TokenOutput, request: TokenInput) -> TokenOutput:
        """Consolidate the request output for partial rollout.

        Args:
            output (TokenOutput): The output from the rollout engine.
            request (TokenInput): The original request input.
        Returns:
            TokenOutput: The consolidated output for the request.
        """
        request_id = request.request_id
        if request_id in self.partial_request_output_store:
            stored_log_probs = self.partial_request_output_store[request_id]["log_probs"]
            # Combine stored log probs with current output log probs
            combined_log_probs = stored_log_probs + output.log_probs
            output.log_probs = combined_log_probs
            # Remove from store after consolidation
            del self.partial_request_output_store[request_id]
        return output

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
            for instance_id in self.instance_ids:
                if instance_to_status[instance_id].get_waiting_queue_size() != 0:
                    continue
                instance_version = self.instance_to_version_after_sync[instance_id]
                if await self.ps_manager_handle.check_aborted_model_versions.remote(instance_version):
                    continue

                def version_filter(request, instance_version=instance_version):
                    request_version = request.version_tag
                    return request_version <= instance_version

                filtered_requests = self.requests_to_route.filter_by_condition(version_filter)
                filtered_request_ids = [request.request_id for request in filtered_requests]
                if len(filtered_request_ids) > 0:
                    is_aborted = await self.ps_manager_handle.check_aborted_requests.remote(
                        filtered_request_ids, remove=False
                    )
                    filtered_request_ids = [
                        request_id for i, request_id in enumerate(filtered_request_ids) if not is_aborted[i]
                    ]
                    is_validate_list = [request.is_validate for request in filtered_requests]
                    can_reserve = await self.ps_manager_handle.can_reserve_request.remote(
                        filtered_request_ids,
                        [instance_version],
                        without_new_reserve_entry=False,
                        is_validate=is_validate_list,
                    )
                    filtered_request_ids = [
                        request_id for i, request_id in enumerate(filtered_request_ids) if can_reserve[i] == [True]
                    ]
                if len(filtered_request_ids) == 0:
                    filtered_instance_ids.append(instance_id)

        candidate_migrate_instance_ids = []  # (instance_id, ratio)
        for starved_instance_id in filtered_instance_ids:
            for instance_id in self.instance_ids:
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
    async def check_should_sync(self, instance_id: RolloutInstanceId) -> bool:
        """Check if the instance should synchronize with PS.

        Args:
            instance_id (RolloutInstanceId): The instance ID to synchronize with.

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
                    request_version = request.version_tag
                    return request_version <= current_instance_version

                filtered_requests = self.requests_to_route.filter_by_condition(version_filter)
                filtered_request_ids = [request.request_id for request in filtered_requests]
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

        # If there are requests that can still be routed to the instance
        # before synchronization without new reserve entry
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
                version = request.version_tag
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
    async def wait_interrupted_partial_requests_loop_back(self, instance_ids: list[RolloutInstanceId]):
        """Wait for the interrupted partial requests to be looped back in the priority queue.

        Args:
            instance_ids (List[RolloutInstanceId]): The instance IDs to wait for.
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
