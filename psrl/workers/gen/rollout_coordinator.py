import os
import logging
import numpy as np
from threading import Thread
from collections import defaultdict

import ray

from psrl.utils.server.command import CommandType, Command, CommandExtension
from psrl.utils.logger import DualOutputHandler

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

@ray.remote
class RolloutCoordinator(CommandExtension):
    def __init__(
        self,
        config,
        ps_manager_handle,
        rollout_wg_list,
        agent_loop_workers,
    ):
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
        
        self.ps_manager_handle = ps_manager_handle
        self.rollout_wg_list = rollout_wg_list
        self.rollout_wg_size = len(rollout_wg_list)
        self.agent_loop_workers = agent_loop_workers
        
        # Background event handler
        self._threads = []
        self.background_running = False
        
        # Instance tracking
        self.instance_running_status: dict[int, bool] = defaultdict(lambda: False)  # Track if an instance is running
        self.instance_to_version: dict[int, int] = {}  # Track the model version of each instance
        
        # Engine status tracking
        self.instance_engine_status: dict[int, dict] = {}  # Track the latest engine status of each instance
        self.engine_status_sync_interval = getattr(
            self.config.psrl.rollout_test, 'engine_status_sync_interval', 30.0
        )  # Interval to sync engine status to agent loop workers
        self._engine_status_sync_thread = None
        
        self._abort_request_ids = set() # Request IDs to be aborted if the instance is interrupted
        
        # Build logger
        self.log_prefix = "RolloutCoordinator"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))

    def start_busy_loop(self):
        if self.background_running:
            return

        # Initialize the busy loop of rollout workers
        futures = []
        for i in range(self.config.psrl.deployment.n_rollout_instances):
            if self.rank_0_is_model_owner:
                psrl_logger.debug(f"Starting rank_zero busy_loop_generate_sequences for instance {i}")
                futures.append(self.rollout_wg_list[i].execute_rank_zero_async("start_busy_loop"))
            else:
                psrl_logger.debug(f"Starting all ranks busy_loop_generate_sequences for instance {i}")
                futures.extend(self.rollout_wg_list[i].execute_all_async("start_busy_loop"))
            self.instance_running_status[i] = True
        ray.get(futures)

        # Start the background command handler thread
        event_handler = Thread(
            target=self._command_handler_loop,
            name="command_handler_thread",
            daemon=True,
        )
        
        event_handler.start()
        self._threads.append(event_handler)

        # Start the engine status sync thread
        self._engine_status_sync_thread = Thread(
            target=self._engine_status_sync_loop,
            name="engine_status_sync_thread",
            daemon=True,
        )
        
        self._engine_status_sync_thread.start()
        self._threads.append(self._engine_status_sync_thread)

        self.background_running = True
    
    def stop_busy_loop(self):
        if not self.background_running:
            return
        
        # Stop the background command handler thread
        self.background_running = False
        
        for thread in self._threads:
            thread.join()
        
        self._threads.clear()
        
        # Stop the busy loop of rollout workers
        futures = []
        for i in range(self.config.psrl.deployment.n_rollout_instances):
            if self.rank_0_is_model_owner:
                psrl_logger.debug(f"Stopping rank_zero busy_loop_generate_sequences for instance {i}")
                futures.append(self.rollout_wg_list[i].execute_rank_zero_async("stop_busy_loop"))
            else:
                psrl_logger.debug(f"Stopping all ranks busy_loop_generate_sequences for instance {i}")
                futures.extend(self.rollout_wg_list[i].execute_all_async("stop_busy_loop"))
        ray.get(futures)
    
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
        instance_requests = ray.get(self.ps_manager_handle.get_dispatched_requests_of_instance.remote(instance_id))

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
            total_child_requests = len(ray.get(self.ps_manager_handle.get_recorded_child_requests.remote(parent_id)))
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
                        version_to_requests[src_v] = ray.get(self.ps_manager_handle.get_requests_ids_of_version.remote(src_v))
        
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
    
    def _command_handler_loop(self):
        while self.background_running:
            # Command processing
            if not self.command_queue.empty():
                command = self.command_queue.get()
                
                assert isinstance(command, Command), f"Expected Command type, got {type(command)}"
                
                # Unpack command attributes
                command_type = command.command_type
                command_id = command.get_kwargs()["id"]
                command_args = command.get_args()
                psrl_logger.debug(f"Receive command: type = {command_type}, kwargs = {command.get_kwargs()}, args = {command_args}")
                
                result = None
                # Process the command based on its type
                if command_type == CommandType.ABORT:
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
                    
                    # TODO: replace with status from gen workers
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
                                ray.get(self.ps_manager_handle.abort_requests.remote(list(self._abort_request_ids)))
                                self._abort_request_ids.clear()
                            else:
                                psrl_logger.debug("No requests to abort")

                            # Add SYNC command to the command queue to interrupt the instance
                            # This will stop the instance, pull the model weights from PS, and resume generation.
                            psrl_logger.debug(f"Queueing SYNC command for instance {instance_id}")
                            
                            # Create a SYNC command to interrupt the instance and pull the model
                            self.exec_command(Command(
                                type=CommandType.SYNC,
                                instance_id=instance_id,
                                curr_ps_model_version=curr_ps_model_version,
                            ), blocking=False)
                            
                            # Execute the SYNC command immediately
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
                            '''
                elif command_type == CommandType.SYNC:
                    # Interrupt the instance, pull the model weights from PS and resume generation.
                    instance_id = command_args.get("instance_id", None)
                    curr_ps_model_version = command_args.get("curr_ps_model_version", None)
                    if instance_id is None or curr_ps_model_version is None:
                        raise ValueError("SYNC command must contain 'instance_id' and 'curr_ps_model_version' in args.")
                    psrl_logger.debug(f"Received SYNC command for instance {instance_id} with PS model version {curr_ps_model_version}")
                    
                    # Interrupt the instance
                    future = None
                    if self.rank_0_is_model_owner:
                        future = self.rollout_wg_list[instance_id].execute_rank_zero_async("interrupt_generation")
                    else:
                        future = self.rollout_wg_list[instance_id].execute_all_async("interrupt_generation")
                    interrupted_request_num = ray.get(future)
                    self.instance_running_status[instance_id] = False
                    
                    # Pull model
                    if self.rank_0_is_model_owner:
                        future = self.rollout_wg_list[instance_id].execute_rank_zero_async("pull_model")
                    else:
                        future = self.rollout_wg_list[instance_id].execute_all_async("pull_model")
                    ray.get(future)
                    
                    # Resume generation
                    if self.rank_0_is_model_owner:
                        self.rollout_wg_list[instance_id].execute_rank_zero_async("resume_generation")
                    else:
                        self.rollout_wg_list[instance_id].execute_all_async("resume_generation")
                    self.instance_running_status[instance_id] = True
                elif command_type == CommandType.ENGINE_STATUS:
                    # Handle engine status updates from rollout workers
                    engine_status = command_args.get("engine_status", None)
                    if engine_status is None:
                        raise ValueError("ENGINE_STATUS command must contain 'engine_status' in args.")
                    
                    instance_id = engine_status.get("instance_id")
                    if instance_id is not None:
                        self.instance_engine_status[instance_id] = engine_status
                        psrl_logger.debug(f"Updated engine status for instance {instance_id}: {engine_status}")
                    else:
                        psrl_logger.warning("Received engine status without instance_id")
                    
                    result = "status_updated"
                else:
                    raise ValueError(f"Unknown command type: {command_type}")

                # Post process the command result
                self._complete_command(command_id, result)
        
        psrl_logger.info("Background command handler of rollout coordinator has finished.")
    
    def _engine_status_sync_loop(self):
        """Background loop to sync engine status to agent loop workers periodically."""
        import time
        
        psrl_logger.info("Starting engine status sync loop")
        
        while self.background_running:
            try:
                # Check if we have any engine status to sync
                if self.instance_engine_status:
                    # Prepare consolidated engine status
                    consolidated_status = {
                        "timestamp": time.time(),
                        "instance_engine_status": dict(self.instance_engine_status),
                        "instance_running_status": dict(self.instance_running_status),
                        "instance_to_version": dict(self.instance_to_version),
                    }
                    
                    psrl_logger.debug(f"Syncing engine status to {len(self.agent_loop_workers)} agent loop workers")
                    
                    # Send to all agent loop workers
                    futures = []
                    for agent_worker in self.agent_loop_workers:
                        futures.append(agent_worker.update_engine_status.remote(consolidated_status))
                    
                    # Wait for all updates to complete
                    ray.get(futures)
                    psrl_logger.debug("Engine status sync completed")
                else:
                    psrl_logger.debug("No engine status to sync")
                
            except Exception as e:
                psrl_logger.error(f"Error in engine status sync loop: {e}")
            
            # Sleep until next sync interval
            time.sleep(self.engine_status_sync_interval)
        
        psrl_logger.info("Engine status sync loop stopped.")
