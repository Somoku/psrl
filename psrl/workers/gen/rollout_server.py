import os
import logging
import time
import enum
import heapq
import multiprocessing
import numpy as np
from multiprocessing import Process, Event, Manager
from enum import Enum
from typing import Union, List, Any
from dataclasses import dataclass
from collections import defaultdict

import ray

from verl import DataProto

from psrl.utils.server.command import CommandType, Command, CommandExtension
from psrl.workers.ps.staleness_controller import EntryInfo
from psrl.workers.request_manager.request_status_manager import RequestStatus, RequestStatusManager

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))

@ray.remote
class RolloutServer(CommandExtension):
    def __init__(
        self,
        config,
        rollout_wg_list,
        rollout_router_cls,
        data_queue,
        rollout_queue,
        replay_buffer,
        ps_handle,
        request_status_manager,
    ):
        """
        Initialize the rollout server with the given configuration, worker group list and communication handles.
        
        Args:
            config: Configuration object containing server settings.
            rollout_wg_list: List of rollout worker groups to handle requests.
            rollout_router_cls: Class used for routing requests to different instances.
            data_queue: Queue for incoming data requests.
            rollout_queue: Queue for outgoing rollout requests.
            replay_buffer: Buffer for storing replay data.
            ps_handle: Handle to the parameter server for model version management.
            request_status_manager: Manager for tracking request statuses.
        """
        super().__init__()

        self.config = config
        self.staleness = self.config.psrl.staleness
        if self.config.psrl.rollout_test.redundant_rollout.enable:
            self.rollout_n = self.config.psrl.rollout_test.redundant_rollout.redundant_rollout_n
            self.alg_rollout_n = self.config.psrl.rollout_test.redundant_rollout.alg_rollout_n
        else:
            self.rollout_n = self.config.gen_actor_rollout_ref.rollout.n
            self.alg_rollout_n = self.rollout_n
        
        self.rank_0_is_model_owner = (
            self.config.gen_actor_rollout_ref.rollout.tensor_model_parallel_size *
                self.config.gen_actor_rollout_ref.rollout.pipeline_model_parallel_size > 1 and
            self.config.gen_actor_rollout_ref.rollout.mode == "psrl_async"
        )

        # Rollout instance and router initialization
        self.rollout_wg_list = rollout_wg_list
        self.rollout_wg_num = len(rollout_wg_list)
        self.rollout_router = rollout_router_cls(self.rollout_wg_num)
        
        # Queues and buffers for data transfer between workers
        self.data_queue = data_queue
        self.rollout_queue = rollout_queue
        self.replay_buffer = replay_buffer

        # Server state management
        self.server_running = False
        self.rollout_paused = False
        self.skipping_data_queue = False

        # Background event handler
        self._processes = []
        
        # Instance tracking
        self.instance_running_status: dict[int, bool] = defaultdict(lambda: False)  # Track if an instance is running
        self.instance_to_version: dict[int, int] = {}  # Track the model version of each instance
        
        self._request_counter = 0 # For version tag setting
        self._abort_request_ids = set() # Request IDs to be aborted if the instance is interrupted
        
        # Request status manager for tracking request statuses
        self.request_status_manager = request_status_manager
        
        # Parameter server handle for model version management
        self.ps_handle = ps_handle

    def start_server(self):
        """
        Start the rollout server and begin processing requests.
        
        This method initializes the server, starts the background event handler,
        and begins the busy loop of backend rollout workers for generating sequences.
        """
        if self.server_running:
            return
        
        self.server_running = True
        # Initialize the busy loop of rollout workers
        futures = []
        for i in range(self.config.psrl.deployment.n_rollout_instances):
            if self.rank_0_is_model_owner:
                futures.append(self.rollout_wg_list[i].execute_rank_zero_async(
                    "busy_loop_generate_sequences",
                    rollout_queue=self.rollout_queue,
                    replay_buffer=self.replay_buffer
                ))
            else:
                futures.extend(self.rollout_wg_list[i].execute_all_async(
                    "busy_loop_generate_sequences",
                    rollout_queue=self.rollout_queue,
                    replay_buffer=self.replay_buffer
                ))
            self.instance_running_status[i] = True
        ray.get(futures)

        # Start the background event handler process
        event_handler = Process(
            target=self._background_event_handler,
            name="rollout_event_process",
            daemon=True,
        )

        event_handler.start()
        self._processes = [event_handler]
    
    def shutdown_server(self):
        """Shutdown the rollout server gracefully."""
        if not self.server_running:
            return
        
        psrl_logger.info("Shutdown rollout server...")
        self.server_running = False

        # Stop the background event handler
        for process in self._processes:
            if process.is_alive():
                process.join(timeout=10)
                if process.is_alive():
                    # Terminate the process if it did not exit gracefully
                    process.terminate()
        self._processes = []
        
        # Clean manager resources
        if hasattr(self, '_manager'):
            self._manager.shutdown()

        psrl_logger.info("Rollout server shutdown.")

    def stop_server(self):
        """Stop the rollout server and interrupt all workers if it is running."""
        if not self.rollout_paused:
            psrl_logger.info(f"Waiting for {self.rollout_wg_num} workers to stop...")
            self.exec_command(Command(CommandType.STOP))
            psrl_logger.info(f"Rollout server stopped.")

    def resume_server(self):
        """Resume the rollout server if it was paused."""
        if self.rollout_paused:
            psrl_logger.info("Resuming rollout server...")
            self.exec_command(Command(CommandType.RESUME))

    def dispatch_requests(self, data: DataProto):
        """
        Dispatch requests to different rollout instances based on the rollout router strategy.
        
        This method processes the incoming data, checks and updates the global request status,
        and dispatches the requests to the appropriate instances.
        
        Args:
            data (DataProto): The data containing requests to be dispatched.
        """
        assert "version_tag" in data.non_tensor_batch, \
            "DataProto must contain 'version_tag' in non_tensor_batch for dispatching requests."
        
        # Update requests status from PENDING to DISPATCHED
        batch_size = len(data)
        request_ids = list(data.non_tensor_batch["uid"])
        update_status_success = ray.get(self.request_status_manager.update_status.remote(request_ids, RequestStatus.DISPATCHED))
        dispatch_request_idxs = [i for i, success in enumerate(update_status_success) if success]
        if not dispatch_request_idxs:
            return
        
        # Filter aborted requests
        dispatch_data = data.select_idxs(dispatch_request_idxs)
        
        # Rollout router dispatches requests to different instances based on the strategy.
        dispatch_plan = self.rollout_router.dispatch(dispatch_data, self.instance_running_status)
        for instance_id, requests in dispatch_plan.items():
            if requests is None:
                continue

            version_tags = requests.non_tensor_batch["version_tag"]
            for i, version_tag in enumerate(version_tags):
                request_id = requests.non_tensor_batch["uid"][i]
                ray.get(self.request_status_manager.update_request_info.remote(
                    EntryInfo(
                        rollout_instance_id=instance_id,
                        request_id=request_id,
                        model_version=version_tag,
                    )
                ))
            psrl_logger.info(f"{len(requests)} requets are scheduled to instance {instance_id}")

            # Add requests to the rollout worker group for processing
            if self.rank_0_is_model_owner:
                self.rollout_wg_list[instance_id].execute_rank_zero_async("add_request", requests)
            else:
                self.rollout_wg_list[instance_id].execute_all_async("add_request", requests)

    def set_rollout_instance_model_version(self, rollout_instance_id: int, version_tag: int):
        """
        Set the model version for a specific rollout instance.
        
        Args:
            rollout_instance_id (int): The ID of the rollout instance.
            version_tag (int): The model version tag to set for the instance.
        """
        self.instance_to_version[rollout_instance_id] = version_tag

    def set_version_tag(self, request):
        """
        Set the version tag for the request based on the current staleness and request counter.
        
        NOTE: Currently it's a naive greedy implementation that increments the version tag
        for each request. This may not be optimal in a real-world scenario.
        """
        if self.config.psrl.rollout_test.redundant_rollout.enable:
            buffer_size = self.config.psrl.rollout_test.redundant_rollout.redundant_global_batch_size * self.rollout_n
        else:
            buffer_size = self.config.psrl.staleness_buffer_entries * self.rollout_n
        version_tag = max(self._request_counter - self.staleness * buffer_size, 0) // buffer_size
        self._request_counter += 1
        return version_tag

    def check_interrupt_ability(self, instance_id: int, curr_ps_model_version: int, staleness: int):
        """
        Check if the instance can be interrupted based on the current model version and staleness.
        
        This method checks if the instance can be interrupted without violating the staleness constraints:
        (1). The rest child requests of a parent prompt should be sufficient for Group Sampling.
        (2). The number of valid requests (in group) for each impacted version tag should be sufficient.
        If the instance can be interrupted, it returns True, otherwise returns False.
        
        Args:
            instance_id (int): The ID of the instance to check.
            curr_ps_model_version (int): The current model version in the parameter server.
            staleness (int): The staleness threshold for interrupting requests.
        
        Returns:
            bool: True if the instance can be interrupted, False otherwise.
        """
        instance_version = self.instance_to_version.get(instance_id, None)
        # Able to interrupt if no requests will be aborted after model update
        if instance_version is None or curr_ps_model_version - instance_version <= staleness:
            return True

        # Collect request IDs of the instance and check if they are stale
        instance_request_ids = ray.get(self.request_status_manager.get_dispatched_requests_of_instance.remote(instance_id))
        abort_version_to_requests: dict[int, set] = defaultdict(set)
        abort_parent_to_request_num: dict[int, int] = defaultdict(int)
        abort_request_ids = set()
        for request_id in instance_request_ids:
            parent_id = request_id // self.rollout_n
            version_tag = ray.get(self.request_status_manager.get_request_version_tag.remote(request_id))
            # Check if the request is stale
            if curr_ps_model_version - version_tag > staleness:
                abort_version_to_requests[version_tag].add(request_id)
                abort_parent_to_request_num[parent_id] += 1
                abort_request_ids.add(request_id)

        # Check if the number of rest child requests is sufficient for Group Sampling
        for parent_id, request_num in abort_parent_to_request_num.items():
            if len(ray.get(self.request_status_manager.get_recorded_child_requests.remote(parent_id))) - request_num < self.alg_rollout_n:
                return False

        # Check if the number of requests for each impacted version tag is sufficient
        # If we abort requests of version V, the requests of version [V, V + staleness] should be sufficient
        impacted_version_tags = set()
        version_to_requests = {}
        for version_tag in abort_version_to_requests.keys():
            for v in range(version_tag, version_tag + staleness + 1):
                impacted_version_tags.add(v)
                for src_v in range(max(0, v - staleness), v + 1):
                    if src_v not in version_to_requests:
                        version_to_requests[src_v] = ray.get(self.request_status_manager.get_requests_of_version.remote(src_v))
        
        for version_tag in impacted_version_tags:
            version_to_requests[version_tag] -= abort_version_to_requests.get(version_tag, set())
            # Classify requests by their parent IDs to ensure we can check the number of valid requests
            parent_to_child_request_num = defaultdict(int)
            for request_id in version_to_requests[version_tag]:
                parent_id = request_id // self.rollout_n
                parent_to_child_request_num[parent_id] += 1
            valid_group_request_num = np.sum([1 for child_request_num in parent_to_child_request_num.values() if child_request_num >= self.alg_rollout_n])
            if valid_group_request_num < self.config.psrl.staleness_buffer_entries:
                return False
        
        # If all checks passed, the instance can be interrupted and requests can be aborted
        self._abort_request_ids.update(abort_request_ids)
        return True

    def _background_event_handler(self):
        while self.server_running:
            # Command processing
            if not self.command_queue.empty():
                # Get command from the queue
                psrl_logger.debug("Command queue is not empty, try to get commands")
                command = self.command_queue.get()

                assert isinstance(command, Command), f"Expected Command, got {type(command)}"

                # Unpack command attributes
                command_type = command.type
                command_id = command.get_kwargs()["id"]
                command_args = command.get_args()

                psrl_logger.debug(f"Command: type = {command_type}, kwargs = {command.get_kwargs()}, args = {command_args}")
                
                result = None
                # Process the command based on its type
                if command_type == CommandType.STOP:
                    psrl_logger.debug("Begin to interrupt data queue processing")
                    self.rollout_paused = True
                    futures = []
                    for i in range(self.config.psrl.deployment.n_rollout_instances):
                        if self.rank_0_is_model_owner:
                            futures.append(self.rollout_wg_list[i].execute_rank_zero_async("interrupt_generation"))
                        else:
                            futures.append(self.rollout_wg_list[i].execute_all_async("interrupt_generation"))
                    interrupted_request_nums = ray.get(futures)
                    result = np.sum(interrupted_request_nums)
                    psrl_logger.debug(f"Received STOP command, interrupted {result} requests")
                elif command_type == CommandType.RESUME:
                    psrl_logger.debug("Begin to resume data queue processing")
                    instance_ids = command_args.get("instance_ids", None)
                    if instance_ids is None:
                        instance_ids = range(self.config.psrl.deployment.n_rollout_instances)
                    
                    for instance_id in instance_ids:
                        if self.rank_0_is_model_owner:
                            self.rollout_wg_list[instance_id].execute_rank_zero_async("resume_generation")
                        else:
                            self.rollout_wg_list[instance_id].execute_all_async("resume_generation")
                    self.rollout_paused = False
                    psrl_logger.debug("Resumed data queue processing")
                elif command_type == CommandType.SHUTDOWN:
                    psrl_logger.debug("Received SHUTDOWN command, shutting down all instances...")
                    self.server_running = False
                    for i in range(self.config.psrl.deployment.n_rollout_instances):
                        if self.rank_0_is_model_owner:
                            self.rollout_wg_list[i].execute_rank_zero_async("shutdown_generate")
                        else:
                            self.rollout_wg_list[i].execute_all_async("shutdown_generate")
                    psrl_logger.debug("All instances have been shut down.")
                elif command_type == CommandType.SYNC:
                    # Interrupt the instance, pull the model weights from PS and resume generation.
                    instance_id = command_args.get("instance_id", None)
                    curr_ps_model_version = command_args.get("curr_ps_model_version", None)
                    if instance_id is None or curr_ps_model_version is None:
                        raise ValueError("SYNC command must contain 'instance_id' and 'curr_ps_model_version' in args.")
                    psrl_logger.debug(f"Received SYNC command, interrupt the instance {instance_id} and synchronize model weights...")
                    
                    # Interrupt the instance (including requests in request queue and running tasks)
                    psrl_logger.debug(f"SYNC command for instance {instance_id}: stopping generation...")
                    future = None
                    if self.rank_0_is_model_owner:
                        future = self.rollout_wg_list[instance_id].execute_rank_zero_async("interrupt_generation", self.replay_buffer)
                    else:
                        future = self.rollout_wg_list[instance_id].execute_all_async("interrupt_generation", self.replay_buffer)
                    interrupted_request_num = ray.get(future)
                    self.instance_running_status[instance_id] = False
                    psrl_logger.debug(f"SYNC command for instance {instance_id}: interrupted {interrupted_request_num} requests")
                    
                    # Pull model
                    psrl_logger.debug(f"SYNC command for instance {instance_id}: pulling model...")
                    if self.rank_0_is_model_owner:
                        future = self.rollout_wg_list[instance_id].execute_rank_zero_async("pull_model")
                    else:
                        future = self.rollout_wg_list[instance_id].execute_all_async("pull_model")
                    ray.get(future)
                    psrl_logger.debug(f"SYNC command for instance {instance_id}: pulled model")
                    
                    # Resume generation
                    psrl_logger.debug(f"SYNC command for instance {instance_id}: resuming generation...")
                    if self.rank_0_is_model_owner:
                        self.rollout_wg_list[instance_id].execute_rank_zero_async("resume_generation")
                    else:
                        self.rollout_wg_list[instance_id].execute_all_async("resume_generation")
                    self.instance_running_status[instance_id] = True
                    psrl_logger.debug(f"SYNC command for instance {instance_id}: resumed generation")
                elif command_type == CommandType.ABORT:
                    assert "parent_ids" in command_args or "uids" in command_args, \
                        "Abort command must contain either 'parent_ids' or 'uids' in args."
                    parent_ids = command_args.get("parent_ids", None)
                    uids = command_args.get("uids", None)
                    if parent_ids is None and uids is None:
                        raise ValueError("Abort command must contain either 'parent_ids' or 'uids' in args.")

                    psrl_logger.debug(f"Received ABORT command with parent_ids: {parent_ids}, uids: {uids}")

                    if not isinstance(parent_ids, (list, type(None))):
                        parent_ids = [parent_ids]
                    if not isinstance(uids, (list, type(None))):
                        uids = [uids]

                    # Collect all requests to be aborted
                    abort_request_uids = set()
                    # Step 1. Get child requests from parent_ids
                    if parent_ids is not None:
                        parent_ids = set(parent_ids) # Ensure uniqueness
                        abort_request_uids.update(ray.get(
                            self.request_status_manager.get_recorded_child_requests.remote(list(parent_ids))
                        ))
                    # Step 2. Get requests from uids
                    if uids is not None:
                        uids = set(uids)
                        abort_request_uids.update(uids)

                    # Get classified abort requests from request tracker (in instance_id)
                    abort_map_from_instance_to_requests: dict[int, set[int]] = ray.get(
                        self.request_status_manager.classify_requests_in_instance.remote(abort_request_uids)
                    )
                    
                    futures = []
                    for i in range(self.config.psrl.deployment.n_rollout_instances):
                        # Abort requests in each instance
                        abort_requests = abort_map_from_instance_to_requests.get(i, set())
                        if not abort_requests:
                            continue

                        if self.rank_0_is_model_owner:
                            futures.append(self.rollout_wg_list[i].execute_rank_zero_async("interrupt_requests", abort_requests))
                        else:
                            futures.append(self.rollout_wg_list[i].execute_all_async("interrupt_requests", abort_requests))
                    if not futures:
                        interrupted_request_num = 0
                    else:
                        interrupted_request_nums = ray.get(futures)
                        interrupted_request_num = np.sum(interrupted_request_nums)
                    
                    result = interrupted_request_num
                    psrl_logger.debug(f"Received ABORT command, interrupted {interrupted_request_num} requests")
                elif command_type == CommandType.CHECK_AND_SYNC:
                    if not self.config.psrl.rollout_test.partial_rollout.enable:
                        raise ValueError("CHECK_AND_SYNC command is only available when partial rollout is enabled.")
                    # This command is used to check whether the instance can be interrupted
                    # and sync the model version with the parameter server.
                    
                    buffer_id = command_args.get("buffer_id", None)
                    curr_ps_model_version = command_args.get("curr_ps_model_version", None)
                    if buffer_id is None or curr_ps_model_version is None:
                        raise ValueError("CHECK command must contain 'buffer_id' and 'curr_ps_model_version' in args.")

                    # Get the workload of each instance (waiting and running queue size)
                    instance_to_request_num = {}
                    futures = []
                    for instance_id in range(self.config.psrl.deployment.n_rollout_instances):
                        if self.rank_0_is_model_owner:
                            waiting_and_running_queue_size_ref = self.rollout_wg_list[instance_id].execute_rank_zero_async("waiting_and_running_queue_size")
                        else:
                            waiting_and_running_queue_size_ref = self.rollout_wg_list[instance_id].execute_all_async("waiting_and_running_queue_size")
                        instance_to_request_num[instance_id] = waiting_and_running_queue_size_ref
                        futures.append(waiting_and_running_queue_size_ref)
                    
                    # Get the actual results and update the dictionary
                    results = ray.get(futures)
                    for instance_id, result in zip(range(self.config.psrl.deployment.n_rollout_instances), results):
                        instance_to_request_num[instance_id] = result

                    # Check if the instance can be interrupted based on the current model version
                    for instance_id, request_num in instance_to_request_num.items():
                        # Interrupt the instance if the workload is less than the threshold
                        if request_num > self.config.psrl.rollout_test.partial_rollout.threshould:
                            continue

                        if (
                            self.config.psrl.rollout_test.partial_rollout.interrupt_as_prompt or
                            self.check_interrupt_ability(instance_id, curr_ps_model_version, self.staleness)
                        ):
                            psrl_logger.debug(f"RolloutServer: Instance {instance_id} can be interrupted, current model version: {curr_ps_model_version}")
 
                            # Notify the request status manager to abort requests
                            ray.get(self.request_status_manager.abort_requests.remote(self._abort_request_ids))
                            self._abort_request_ids.clear()

                            # Add SYNC command to the command queue to interrupt the instance
                            # This will stop the instance, pull the model weights from PS, and resume generation.
                            self.exec_command(Command(
                                type=CommandType.SYNC,
                                instance_id=instance_id,
                                curr_ps_model_version=curr_ps_model_version,
                            ), blocking=False)
                else:
                    raise ValueError(f"Unknown command type: {command_type}")

                # Post process the command result
                self._complete_command(command_id, result)
            
            # Data processing
            # Process requests in the replay buffer first
            if not self.replay_buffer.empty() and not self.rollout_paused:
                replay_data = self.replay_buffer.get_nowait()
                assert replay_data is not None, "Replay buffer data should not be None."
                
                non_tensor_batch_keys = replay_data.non_tensor_batch.keys()
                assert "version_tag" and "rollout_instance_id" in non_tensor_batch_keys, \
                    "Replay buffer data must contain 'version_tag' and 'rollout_instance_id' in non_tensor_batch."
                
                if self.config.psrl.rollout_test.partial_rollout.interrupt_as_prompt:
                    # If `interrupt_as_prompt` is enabled, we update the version tag for each request
                    # to the corresponding rollout instance's model version.
                    # This is to ensure that the requests are processed as a prompt with the instance's model version.
                    version_tags = []
                    for request in replay_data.chunk(len(replay_data)):
                        rollout_instance_id = request.non_tensor_batch["rollout_instance_id"][0]
                        if rollout_instance_id not in self.instance_to_version:
                            raise ValueError(f"Rollout instance {rollout_instance_id} is not registered.")
                        version_tag = self.instance_to_version[rollout_instance_id]
                        version_tags.append(version_tag)
                    replay_data.non_tensor_batch["version_tag"] = np.array(version_tags)
                
                self.dispatch_requests(replay_data)
                # NOTE: replay buffer is prior to data queue, so we skip the data queue processing
                continue

            # Process requests in the data queue
            if not self.data_queue.empty() and not self.rollout_paused and not self.skipping_data_queue:
                data = self.data_queue.get_nowait()
                
                # Receive END signal to stop processing data queue
                if data is None:
                    psrl_logger.info(f"RolloutServer: Received `None` data, skipping scheduling.")
                    for i in range(self.config.psrl.deployment.n_rollout_instances):
                        if self.rank_0_is_model_owner:
                            self.rollout_wg_list[i].execute_rank_zero_async("add_request", None)
                        else:
                            self.rollout_wg_list[i].execute_all_async("add_request", None)
                    self.skipping_data_queue = True
                    continue
                
                # Set version tag for each request in the data
                batch_size = len(data)
                request_list = data.chunk(chunks=batch_size)
                for request in request_list:
                    version_tag = self.set_version_tag(request)
                    request.non_tensor_batch["version_tag"] = np.array([version_tag])
                
                self.dispatch_requests(data)

        psrl_logger.info("Background event handler of rollout server has finished.")
