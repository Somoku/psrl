import os
import logging
import time
import enum
import heapq
import threading
import numpy as np
from threading import Thread
from enum import Enum
from typing import Union, List, Any
from dataclasses import dataclass
from collections import defaultdict

import ray

from verl import DataProto

from psrl.utils.server.command import CommandType, Command, CommandExtension
from psrl.workers.ps.staleness_controller import EntryInfo
from psrl.workers.request_manager.request_status_manager import RequestStatus, RequestStatusManager
from psrl.utils.logger import log_dual_events, EventType, DualOutputHandler

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

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
        ps_manager_handle,
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
            ps_manager_handle: Handle to the parameter server for model version management.
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
        self._threads = []
        
        # Instance tracking
        self.instance_running_status: dict[int, bool] = defaultdict(lambda: False)  # Track if an instance is running
        self.instance_to_version: dict[int, int] = {}  # Track the model version of each instance
        
        self._request_counter = 0 # For version tag setting
        self._abort_request_ids = set() # Request IDs to be aborted if the instance is interrupted
        
        # Request status manager for tracking request statuses
        self.request_status_manager = request_status_manager
        
        # Parameter server handle for model version management
        self.ps_manager_handle = ps_manager_handle
        
        # Build logger
        self.log_prefix = "RolloutServer"
        psrl_logger.addHandler(DualOutputHandler(self.log_prefix))

    def start_server(self):
        """
        Start the rollout server and begin processing requests.
        
        This method initializes the server, starts the background event handler,
        and begins the busy loop of backend rollout workers for generating sequences.
        """
        if self.server_running:
            psrl_logger.debug("Server already running, ignoring start_server call")
            return
        
        self.server_running = True
        # Initialize the busy loop of rollout workers
        futures = []
        for i in range(self.config.psrl.deployment.n_rollout_instances):
            psrl_logger.debug(f"Initializing busy loop for rollout instance {i}")
            if self.rank_0_is_model_owner:
                psrl_logger.debug(f"Starting rank_zero busy_loop_generate_sequences for instance {i}")
                futures.append(self.rollout_wg_list[i].execute_rank_zero_async(
                    "busy_loop_generate_sequences",
                    rollout_queue=self.rollout_queue,
                    replay_buffer=self.replay_buffer
                ))
            else:
                psrl_logger.debug(f"Starting all ranks busy_loop_generate_sequences for instance {i}")
                futures.extend(self.rollout_wg_list[i].execute_all_async(
                    "busy_loop_generate_sequences",
                    rollout_queue=self.rollout_queue,
                    replay_buffer=self.replay_buffer
                ))
            self.instance_running_status[i] = True
            psrl_logger.debug(f"Marked instance {i} as running")
        psrl_logger.debug(f"Waiting for {len(futures)} futures to complete")
        ray.get(futures)
        psrl_logger.debug("All busy loops initialized")

        # Start the background event handler thread
        psrl_logger.debug("Starting background event handler thread")
        event_handler = Thread(
            target=self._background_event_handler,
            name="rollout_event_thread",
            daemon=True,
        )

        event_handler.start()
        self._threads = [event_handler]
    
    def shutdown_server(self):
        """Shutdown the rollout server gracefully."""
        if not self.server_running:
            return
        
        psrl_logger.info("Shutdown rollout server...")
        self.server_running = False

        # Stop the background event handler
        for thread in self._threads:
            if thread.is_alive():
                thread.join(timeout=10)
        self._threads = []

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
        psrl_logger.debug(f"Dispatching requests from DataProto with {len(data)} entries")
        
        # Update requests status from PENDING to DISPATCHED
        batch_size = len(data)
        request_ids = data.non_tensor_batch["uid"]
        version_tags = data.non_tensor_batch["version_tag"]
        update_status_success = ray.get(self.request_status_manager.update_status.remote(
            request_ids.tolist(),
            RequestStatus.DISPATCHED,
            model_version=version_tags.tolist(),
        ))
        dispatch_request_idxs = [i for i, success in enumerate(update_status_success) if success]
        
        psrl_logger.debug(f"Successfully updated status for {len(dispatch_request_idxs)}/{batch_size} requests")
        if not dispatch_request_idxs:
            psrl_logger.debug("No requests to dispatch after status update, returning")
            return
        
        # Filter aborted requests
        dispatch_data = data.select_idxs(dispatch_request_idxs)
        psrl_logger.debug(f"Filtered to {len(dispatch_data)} valid requests for dispatch")
        
        # Rollout router dispatches requests to different instances based on the strategy.
        psrl_logger.debug(f"Using {self.rollout_router.__class__.__name__} to dispatch requests")
        dispatch_plan = self.rollout_router.route(dispatch_data, self.instance_running_status)
        psrl_logger.debug(f"Dispatch plan: {[f'instance {k}: {len(v) if v is not None else 0} requests' for k, v in dispatch_plan.items()]}")
        
        for instance_id, requests in dispatch_plan.items():
            if requests is None:
                psrl_logger.debug(f"No requests to dispatch to instance {instance_id}")
                continue

            version_tags = requests.non_tensor_batch["version_tag"]
            psrl_logger.debug(f"Dispatching to instance {instance_id}: {len(requests)} requests with version tags {version_tags[:5]}{'...' if len(version_tags) > 5 else ''}")
            
            for i, version_tag in enumerate(version_tags):
                request_id = requests.non_tensor_batch["uid"][i]
                ray.get(self.request_status_manager.update_request_info.remote(
                    EntryInfo(
                        rollout_instance_id=instance_id,
                        request_id=request_id,
                        model_version=version_tag,
                    )
                ))
                psrl_logger.debug(f"Updated request info for request {request_id}: instance={instance_id}, version={version_tag}")
            
            psrl_logger.info(f"{len(requests)} requets are scheduled to instance {instance_id}")

            # Add requests to the rollout worker group for processing
            psrl_logger.debug(f"Sending {len(requests)} requests to instance {instance_id} worker group")
            if self.rank_0_is_model_owner:
                self.rollout_wg_list[instance_id].execute_rank_zero_async("add_request", requests)
                psrl_logger.debug(f"Requests sent to rank 0 of instance {instance_id}")
            else:
                self.rollout_wg_list[instance_id].execute_all_async("add_request", requests)
                psrl_logger.debug(f"Requests sent to all ranks of instance {instance_id}")

    def set_rollout_instance_model_version(self, rollout_instance_id: int, version_tag: int):
        """
        Set the model version for a specific rollout instance.
        
        Args:
            rollout_instance_id (int): The ID of the rollout instance.
            version_tag (int): The model version tag to set for the instance.
        """
        psrl_logger.debug(f"Setting model version for instance {rollout_instance_id} to {version_tag}")
        old_version = self.instance_to_version.get(rollout_instance_id, None)
        self.instance_to_version[rollout_instance_id] = version_tag
        psrl_logger.debug(f"Updated instance {rollout_instance_id} model version: {old_version} -> {version_tag}")

    def set_version_tag(self, request):
        """
        Set the version tag for the request based on the current staleness and request counter.
        
        NOTE: Currently it's a naive greedy implementation that increments the version tag
        for each request. This may not be optimal in a real-world scenario.
        """
        psrl_logger.debug(f"Setting version tag for request with counter {self._request_counter}")
        if self.config.psrl.rollout_test.redundant_rollout.enable:
            buffer_size = self.config.psrl.rollout_test.redundant_rollout.redundant_global_batch_size * self.rollout_n
            psrl_logger.debug(f"Using redundant rollout buffer_size: {buffer_size}")
        else:
            buffer_size = self.config.psrl.staleness_buffer_entries * self.rollout_n
            psrl_logger.debug(f"Using standard buffer_size: {buffer_size}")
            
        version_tag = max(self._request_counter - self.staleness * buffer_size, 0) // buffer_size
        psrl_logger.debug(f"Calculated version_tag={version_tag} with request_counter={self._request_counter}, staleness={self.staleness}, buffer_size={buffer_size}")
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
        instance_version = self.instance_to_version.get(instance_id, 0)
        psrl_logger.debug(f"Checking interrupt ability for instance {instance_id}: "
                          f"current PS model version={curr_ps_model_version}, instance version={instance_version}, staleness={staleness}")
        if curr_ps_model_version <= instance_version:
            return False
        
        # Able to interrupt if no requests will be aborted after model update
        if curr_ps_model_version - instance_version <= staleness:
            psrl_logger.debug(f"Instance {instance_id} can be safely interrupted: version diff within staleness threshold")
            return True

        # Collect request IDs of the instance and check if they are stale
        instance_requests = ray.get(self.request_status_manager.get_dispatched_requests_of_instance.remote(instance_id))

        abort_version_to_requests: dict[int, set] = defaultdict(set)
        abort_parent_to_request_num: dict[int, int] = defaultdict(int)
        abort_request_ids = set()

        for request in instance_requests:
            request_id = request.request_id
            parent_id = request_id // self.rollout_n
            version_tag = request.model_version
            
            # Check if the request is stale
            if curr_ps_model_version - version_tag > staleness:
                psrl_logger.debug(f"Request {request_id} is stale: version diff {curr_ps_model_version - version_tag} > staleness {staleness}")
                abort_version_to_requests[version_tag].add(request_id)
                abort_parent_to_request_num[parent_id] += 1
                abort_request_ids.add(request_id)

        # Check if the number of rest child requests is sufficient for Group Sampling
        psrl_logger.debug(f"Checking if remaining child requests are sufficient for Group Sampling")
        for parent_id, request_num in abort_parent_to_request_num.items():
            total_child_requests = len(ray.get(self.request_status_manager.get_recorded_child_requests.remote(parent_id)))
            remaining_requests = total_child_requests - request_num
            psrl_logger.debug(f"Parent {parent_id}: total={total_child_requests}, to_abort={request_num}, remaining={remaining_requests}, required={self.alg_rollout_n}")
            
            if remaining_requests < self.alg_rollout_n:
                psrl_logger.debug(f"Cannot interrupt: parent {parent_id} would have insufficient remaining requests ({remaining_requests} < {self.alg_rollout_n})")
                return False

        # Check if the number of requests for each impacted version tag is sufficient
        # If we abort requests of version V, the requests of version [V, V + staleness] should be sufficient
        psrl_logger.debug(f"Checking if impacted version tags have sufficient requests after abort")
        impacted_version_tags = set()
        version_to_requests = {}
        
        for version_tag in abort_version_to_requests.keys():
            psrl_logger.debug(f"Analyzing impact of aborting version {version_tag} requests")
            for v in range(version_tag, version_tag + staleness + 1):
                impacted_version_tags.add(v)
                psrl_logger.debug(f"Version {v} is impacted by aborting version {version_tag}")
                for src_v in range(max(0, v - staleness), v + 1):
                    if src_v not in version_to_requests:
                        version_to_requests[src_v] = ray.get(self.request_status_manager.get_requests_ids_of_version.remote(src_v))
        
        for version_tag in impacted_version_tags:
            original_count = len(version_to_requests[version_tag])
            version_to_requests[version_tag] -= abort_version_to_requests.get(version_tag, set())
            remaining_count = len(version_to_requests[version_tag])
            psrl_logger.debug(f"Version {version_tag}: original={original_count}, after abort={remaining_count}")
            
            # Classify requests by their parent IDs to ensure we can check the number of valid requests
            parent_to_child_request_num = defaultdict(int)
            for request_id in version_to_requests[version_tag]:
                parent_id = request_id // self.rollout_n
                parent_to_child_request_num[parent_id] += 1
            
            valid_groups = [parent_id for parent_id, child_request_num in parent_to_child_request_num.items() 
                            if child_request_num >= self.alg_rollout_n]
            valid_group_request_num = len(valid_groups)
            psrl_logger.debug(f"Version {version_tag}: found {valid_group_request_num} valid groups after abort")
            
            if valid_group_request_num < self.config.psrl.staleness_buffer_entries:
                psrl_logger.debug(f"Cannot interrupt: version {version_tag} would have insufficient valid groups " +
                                 f"({valid_group_request_num} < {self.config.psrl.staleness_buffer_entries})")
                return False
        
        # If all checks passed, the instance can be interrupted and requests can be aborted
        psrl_logger.debug(f"All checks passed, instance {instance_id} can be interrupted")
        psrl_logger.debug(f"Adding {len(abort_request_ids)} requests to abort list")
        self._abort_request_ids.update(abort_request_ids)
        return True

    def _background_event_handler(self):
        psrl_logger.debug("Starting background event handler thread in RolloutServer")
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
                        psrl_logger.debug("No specific instances provided, resuming all instances")
                        instance_ids = range(self.config.psrl.deployment.n_rollout_instances)
                    else:
                        psrl_logger.debug(f"Resuming specific instances: {instance_ids}")
                    
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
                        psrl_logger.debug(f"Shutting down instance {i}")
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
                    psrl_logger.debug(f"Received SYNC command for instance {instance_id} with PS model version {curr_ps_model_version}")
                    
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
                    assert "instance_to_uids" in command_args, \
                        "Abort command must contain 'instance_to_uids' in args."
                    instance_to_uids = command_args.get("instance_to_uids", None)
                    if instance_to_uids is None:
                        raise ValueError("Abort command must contain 'instance_to_uids' in args.")

                    psrl_logger.debug(f"Received ABORT command with instance_to_uids: {instance_to_uids}")
                    futures = []
                    for instance_id, uids in instance_to_uids.items():
                        if not uids:
                            continue
                        if not isinstance(uids, (list, set)):
                            uids = [uids]
                        abort_requests = set(uids)  # Ensure uniqueness
                        if self.rank_0_is_model_owner:
                            futures.append(self.rollout_wg_list[instance_id].execute_rank_zero_async("interrupt_requests", abort_requests))
                        else:
                            futures.append(self.rollout_wg_list[instance_id].execute_all_async("interrupt_requests", abort_requests))
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
                    
                    curr_ps_model_version = command_args.get("curr_ps_model_version", None)
                    if curr_ps_model_version is None:
                        raise ValueError("CHECK command must contain 'curr_ps_model_version' in args.")

                    psrl_logger.debug(f"Received CHECK_AND_SYNC command for PS model version {curr_ps_model_version}")

                    # Get the workload of each instance (waiting and running queue size)
                    instance_to_request_num = {}
                    futures = []
                    instance_ids = []
                    for instance_id in range(self.config.psrl.deployment.n_rollout_instances):
                        if self.instance_to_version.get(instance_id, 0) == curr_ps_model_version:
                            psrl_logger.debug(f"RolloutServer: Instance {instance_id} is already up-to-date with model version {curr_ps_model_version}, skipping interruption...")
                            continue
                        instance_ids.append(instance_id)
                        if self.rank_0_is_model_owner:
                            waiting_and_running_queue_size_ref = self.rollout_wg_list[instance_id].execute_rank_zero_async("waiting_and_running_queue_size")
                        else:
                            waiting_and_running_queue_size_ref = self.rollout_wg_list[instance_id].execute_all_async("waiting_and_running_queue_size")
                        instance_to_request_num[instance_id] = waiting_and_running_queue_size_ref
                        futures.append(waiting_and_running_queue_size_ref)
                    
                    # Get the actual results and update the dictionary
                    results = ray.get(futures)
                    for instance_id, result in zip(instance_ids, results):
                        instance_to_request_num[instance_id] = result
                        psrl_logger.debug(f"Instance {instance_id} workload: {result} requests")

                    # Check if the instance can be interrupted based on the current model version
                    for instance_id, request_num in instance_to_request_num.items():
                        # Interrupt the instance if the workload is less than the threshold
                        if request_num > self.config.psrl.rollout_test.partial_rollout.threshold:
                            continue

                        psrl_logger.debug(f"Instance {instance_id} workload ({request_num}) is below threshold ({self.config.psrl.rollout_test.partial_rollout.threshold})")
                        interrupt_as_prompt = self.config.psrl.rollout_test.partial_rollout.interrupt_as_prompt
                        can_interrupt = self.check_interrupt_ability(instance_id, curr_ps_model_version, self.staleness)

                        if interrupt_as_prompt or can_interrupt:
                            psrl_logger.debug(f"RolloutServer: Instance {instance_id} can be interrupted, current model version: {curr_ps_model_version}")
 
                            # Notify the request status manager to abort requests
                            if self._abort_request_ids:
                                psrl_logger.debug(f"Aborting {len(self._abort_request_ids)} requests: {list(self._abort_request_ids)[:5]}{'...' if len(self._abort_request_ids) > 5 else ''}")
                                ray.get(self.request_status_manager.abort_requests.remote(list(self._abort_request_ids)))
                                self._abort_request_ids.clear()
                            else:
                                psrl_logger.debug("No requests to abort")

                            # Add SYNC command to the command queue to interrupt the instance
                            # This will stop the instance, pull the model weights from PS, and resume generation.
                            psrl_logger.debug(f"Queueing SYNC command for instance {instance_id}")
                            '''
                            self.exec_command(Command(
                                type=CommandType.SYNC,
                                instance_id=instance_id,
                                curr_ps_model_version=curr_ps_model_version,
                            ), blocking=False)
                            '''
                            # NOTE: Move the SYNC command to the command queue for debugging
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
                else:
                    raise ValueError(f"Unknown command type: {command_type}")

                # Post process the command result
                self._complete_command(command_id, result)
                # NOTE: process all commands before continuing to data processing
                continue
            
            # Data processing
            # Process requests in the replay buffer first
            if not self.replay_buffer.empty() and not self.rollout_paused:
                psrl_logger.debug("Replay buffer is not empty, processing replayed requests first")
                replay_data = self.replay_buffer.get_nowait()
                assert replay_data is not None, "Replay buffer data should not be None."
                
                non_tensor_batch_keys = replay_data.non_tensor_batch.keys()
                psrl_logger.debug(f"Replay data non_tensor_batch keys: {non_tensor_batch_keys}")
                assert "version_tag" and "rollout_instance_id" in non_tensor_batch_keys, \
                    "Replay buffer data must contain 'version_tag' and 'rollout_instance_id' in non_tensor_batch."
                
                if self.config.psrl.rollout_test.partial_rollout.interrupt_as_prompt:
                    # If `interrupt_as_prompt` is enabled, we update the version tag for each request
                    # to the corresponding rollout instance's model version.
                    # This is to ensure that the requests are processed as a prompt with the instance's model version.
                    psrl_logger.debug("interrupt_as_prompt is enabled, updating version tags for replayed requests")
                    version_tags = []
                    for request in replay_data.chunk(len(replay_data)):
                        rollout_instance_id = request.non_tensor_batch["rollout_instance_id"][0]
                        version_tag = self.instance_to_version.get(rollout_instance_id, 0)
                        version_tags.append(version_tag)
                        psrl_logger.debug(f"Updated version tag for request from instance {rollout_instance_id} to {version_tag}")
                    replay_data.non_tensor_batch["version_tag"] = np.array(version_tags)
                
                psrl_logger.debug(f"Dispatching {len(replay_data)} replayed requests")
                self.dispatch_requests(replay_data)
                # NOTE: replay buffer is prior to data queue, so we skip the data queue processing
                continue

            # Process requests in the data queue
            if not self.data_queue.empty() and not self.rollout_paused and not self.skipping_data_queue:
                psrl_logger.debug("Data queue is not empty, processing new requests")
                data = self.data_queue.get_nowait()
                
                # Receive END signal to stop processing data queue
                if data is None:
                    psrl_logger.info(f"RolloutServer: Received `None` data, skipping scheduling.")
                    for i in range(self.config.psrl.deployment.n_rollout_instances):
                        psrl_logger.debug(f"Sending None request to instance {i} to signal end of data")
                        if self.rank_0_is_model_owner:
                            self.rollout_wg_list[i].execute_rank_zero_async("add_request", None)
                        else:
                            self.rollout_wg_list[i].execute_all_async("add_request", None)
                    self.skipping_data_queue = True
                    psrl_logger.debug("Set skipping_data_queue to True")
                    continue
                
                # Set version tag for each request in the data
                batch_size = len(data)
                psrl_logger.debug(f"Processing batch of {batch_size} requests")
                request_list = data.chunk(chunks=batch_size)
                for request in request_list:
                    version_tag = self.set_version_tag(request)
                    request.non_tensor_batch["version_tag"] = np.array([version_tag])
                    psrl_logger.debug(f"Set version_tag={version_tag} for request")
                data = DataProto.concat(request_list)
                
                self.dispatch_requests(data)

        psrl_logger.info("Background event handler of rollout server has finished.")
