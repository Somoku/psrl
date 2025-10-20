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
        status_queue,
    ):
        """
        Initialize the RolloutCoordinator.
        Coordinates and manages rollout instances for PSRL.
    
        This class handles:
        - Registering and tracking rollout instances
        - Managing model version synchronization across instances
        - Handling command execution (abort, sync, check_and_sync)
        - Collecting and distributing engine status information
        - Coordinating interruption and resumption of generation tasks
        
        Args:
            config: Configuration object containing PSRL settings
            ps_manager_handle: Handle to the parameter server manager
            rollout_wg_list: List of rollout worker groups
            agent_loop_workers: List of agent loop worker handles
            status_queue: Queue for receiving status updates from engines
        """
        super().__init__()
        
        self.config = config
        self.staleness = self.config.psrl.staleness
        if self.config.psrl.redundant_rollout.enable:
            self.rollout_n = self.config.psrl.redundant_rollout.redundant_rollout_n
            self.alg_rollout_n = self.config.psrl.redundant_rollout.alg_rollout_n
        else:
            self.rollout_n = self.config.gen_actor_rollout_ref.rollout.n
            self.alg_rollout_n = self.rollout_n
        
        self.rank_0_is_model_owner = self.config.gen_actor_rollout_ref.rollout.mode == "psrl_async"
        
        self.ps_manager_handle = ps_manager_handle
        self.rollout_wg_list = rollout_wg_list
        self.rollout_wg_size = len(rollout_wg_list)
        self.agent_loop_workers = agent_loop_workers
        self.ps_model_version = 0  # Current model version in the parameter server
        self.check_partial_rollout = False # Whether to check for partial rollout in the loop

        # Stats collection
        self.status_queue = status_queue
        self.stats_changed = False
        
        # Background event handler
        self._threads = []
        self.background_running = False
        
        # Instance tracking
        self.instance_running_status: dict[int, bool] = defaultdict(lambda: False)  # Track if an instance is running
        self.instance_to_version: dict[int, int] = {}  # Track the model version of each instance
        
        # Engine status tracking
        self.instance_engine_status: dict[int, dict] = {}  # Track the latest engine status of each instance
        # Interval to sync engine status to agent loop workers
        self.engine_status_sync_interval_in_ms = self.config.psrl.status_collection.engine_status_sync_interval_in_ms
        self._engine_status_sync_thread = None
        
        self._abort_request_ids = set() # Request IDs to be aborted if the instance is interrupted
        
        # Build logger
        self.log_prefix = "RolloutCoordinator"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))
        
    def world_size(self):
        return sum([rollout_wg.world_size for rollout_wg in self.rollout_wg_list])

    def init_nixl_client(self):
        futures = []
        for i in range(self.config.psrl.deployment.n_rollout_instances):
            if self.rank_0_is_model_owner:
                futures.append(self.rollout_wg_list[i].execute_rank_zero_async("init_nixl_client"))
            else:
                futures.extend(self.rollout_wg_list[i].execute_all_async("init_nixl_client"))
        ray.get(futures)
        
    def nixl_protocol(self):
        futures = []
        for i in range(self.config.psrl.deployment.n_rollout_instances):
            if self.rank_0_is_model_owner:
                futures.append(self.rollout_wg_list[i].execute_rank_zero_async("nixl_protocol"))
            else:
                futures.extend(self.rollout_wg_list[i].execute_all_async("nixl_protocol"))
        ray.get(futures)

    def start_busy_loop(self):
        """
        Start the background event loops for command handling and status synchronization.
        
        This method:
        1. Registers all rollout instances with their respective worker groups
        2. Starts a background thread for handling commands (abort, sync, etc.)
        3. Optionally starts a thread for syncing engine status to agent workers
        """
        if self.background_running:
            return
        
        self.background_running = True

        # Register rollout instances
        futures = []
        for i in range(self.config.psrl.deployment.n_rollout_instances):
            if self.rank_0_is_model_owner:
                futures.append(self.rollout_wg_list[i].execute_rank_zero_async("register_rollout_instance"))
            else:
                futures.extend(self.rollout_wg_list[i].execute_all_async("register_rollout_instance"))
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

        if self.config.psrl.status_collection.enable:
            # Start the engine status sync thread
            self._engine_status_sync_thread = Thread(
                target=self._engine_status_sync_loop,
                name="engine_status_sync_thread",
                daemon=True,
            )
            
            self._engine_status_sync_thread.start()
            self._threads.append(self._engine_status_sync_thread)
    
    def stop_busy_loop(self):
        """
        Stop all background threads and clean up resources.
        
        This method gracefully shuts down:
        - Command handler thread
        - Engine status sync thread
        - Any other background threads
        """
        if not self.background_running:
            return
        
        # Stop the background command handler thread
        self.background_running = False
        
        for thread in self._threads:
            # NOTE(linsh): engine status sync thread maybe stuck in ray.get()
            thread.join(timeout=60)
        
        self._threads.clear()
    
    def set_rollout_instance_model_version(self, rollout_instance_id: int, version_tag: int):
        """
        Set the model version for a specific rollout instance.
        
        Args:
            rollout_instance_id (int): The ID of the rollout instance.
            version_tag (int): The model version tag to set for the instance.
        """
        old_version = self.instance_to_version.get(rollout_instance_id, None)
        self.instance_to_version[rollout_instance_id] = version_tag
        psrl_logger.debug(f"Updated instance {rollout_instance_id} model version: {old_version} -> {version_tag}")
    
    def _command_handler_loop(self):
        """
        Background loop for processing commands from the command queue.
        
        This method continuously processes different types of commands:
        - ABORT: Interrupt specific requests on instances
        - SYNC: Interrupt instance, pull new model weights, and resume generation
        
        The loop runs until background_running is set to False.
        """
        while self.background_running:
            # Command processing
            if not self.command_queue.empty():
                command = self.command_queue.get()
                
                assert isinstance(command, Command), f"Expected Command type, got {type(command)}"
                
                # Unpack command attributes
                command_type = command.type
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
                    
                    psrl_logger.info(f"Received ABORT command with instance_to_uids: {instance_to_uids}")
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
                            warnings.warn(f"Interrupt requests on instance {instance_id} in SPMD-style may cause undefined behavior, need to check the behavior")
                            futures.append(self.rollout_wg_list[instance_id].execute_all_async("interrupt_requests", abort_requests)[0])
                    if not futures:
                        interrupted_request_num = 0
                    else:
                        interrupted_request_nums = ray.get(futures)
                        interrupted_request_num = np.sum(interrupted_request_nums)
                    
                    result = interrupted_request_num
                    psrl_logger.info(f"Received ABORT command, interrupted {interrupted_request_num} requests")
                elif command_type == CommandType.SYNC:
                    # Interrupt the instance, pull the model weights from PS and resume generation.
                    instance_id = command_args.get("instance_id", None)
                    curr_ps_model_version = command_args.get("curr_ps_model_version", None)
                    if instance_id is None or curr_ps_model_version is None:
                        raise ValueError("SYNC command must contain 'instance_id' and 'curr_ps_model_version' in args.")
                    psrl_logger.info(f"Received SYNC command for instance {instance_id} with PS model version {curr_ps_model_version}")
                    assert self.config.gen_actor_rollout_ref.rollout.mode == "psrl_async", \
                        "SYNC command is only supported in 'psrl_async' rollout mode."
                    
                    # Sync with PS (interrupt, pull model, and resume generation)
                    self.instance_running_status[instance_id] = False
                    future = None
                    if self.rank_0_is_model_owner:
                        future = self.rollout_wg_list[instance_id].execute_rank_zero_async("sync_with_ps")
                    else:
                        raise ValueError("SYNC command in SPMD-style is not supported yet.")
                    interrupted_request_num = ray.get(future)
                    psrl_logger.info(f"Synced with PS on instance {instance_id}, interrupted {interrupted_request_num} requests")
                    self.instance_running_status[instance_id] = True
                else:
                    raise ValueError(f"Unknown command type: {command_type}")

                # Post process the command result
                self._complete_command(command_id, result)
        
        psrl_logger.info("Background command handler of rollout coordinator has finished.")
    
    def _engine_status_sync_loop(self):
        """
        Background loop to collect engine status and sync to agent loop workers periodically.
        
        This method:
        1. Receives status updates from the status queue (engine waiting/running request counts)
        2. Consolidates status information from all instances
        3. Periodically broadcasts consolidated status to all agent loop workers
        4. Ensures agent workers have up-to-date information for decision making
        """
        import time
        
        psrl_logger.info("Starting engine status sync loop")

        last_publish_time = 0
        while self.background_running:
            elapsed = int(time.time() * 1000) - last_publish_time
            wait_for = (self.engine_status_sync_interval_in_ms if self.stats_changed else 4000)
            try:
                recv_stats = self.status_queue.get(timeout=max(0, wait_for - elapsed))
            except ray.util.queue.Empty:
                recv_stats = None
            if not recv_stats:
                # Timeout - publish current stats to agent workers
                consolidated_stats = {
                    "timestamp": time.time(),
                    "instance_engine_status": dict(self.instance_engine_status),
                    "instance_running_status": dict(self.instance_running_status),
                    "instance_to_version": dict(self.instance_to_version),
                }
                    
                # Send to all agent loop workers
                futures = []
                for agent_worker in self.agent_loop_workers:
                    futures.append(agent_worker.update_engine_status.remote(consolidated_stats))
                
                # Wait for all updates to complete
                ray.get(futures)

                last_publish_time = int(time.time() * 1000)
                self.stats_changed = False
                continue
            
            engine_index, request_counts = recv_stats
            running_queue_size = request_counts[0]
            waiting_queue_size = request_counts[1]
            waiting_and_running_queue_size = running_queue_size + waiting_queue_size
            self.instance_engine_status[engine_index] = {
                "waiting_and_running_queue_size": waiting_and_running_queue_size,
                "running_queue_size": running_queue_size,
                "waiting_queue_size": waiting_queue_size,
            }
            psrl_logger.debug(f"Updated engine status for instance {engine_index}: {self.instance_engine_status[engine_index]}")
            self.stats_changed = True
            
            # partial rollout check
            if self.config.psrl.partial_rollout.enable and self.check_partial_rollout:
                continue_to_check = False
                for instance_id, status in self.instance_engine_status.items():
                    # Check whether instance is running
                    if not self.instance_running_status.get(instance_id, False):
                        continue
                    # Check whether instance version lags behind PS version
                    if self.instance_to_version.get(instance_id, 0) == self.ps_model_version:
                        continue
                    # Check whether current instance workload is below threshold
                    # Currently we consider running queue size as the workload metric
                    running_queue_size = status.get("running_queue_size", 0)
                    psrl_logger.info(f"Instance {instance_id} (version {self.instance_to_version.get(instance_id, 0)}) "
                                     f"workload: {running_queue_size}, "
                                     f"threshold: {self.config.psrl.partial_rollout.threshold}")
                    if running_queue_size > self.config.psrl.partial_rollout.threshold:
                        continue_to_check = True
                        continue
                    # Check whether instance workload would increase after update
                    # NOTE(linsh): this step relies on static version tag assignment
                    inc_request_num = ray.get(self.rollout_wg_list[instance_id].execute_rank_zero_async("get_workload_after_update_to", self.ps_model_version))
                    if inc_request_num > 0:
                        psrl_logger.info(f"Instance {instance_id} workload after update would be {inc_request_num + running_queue_size}")
                        continue_to_check = True
                        continue

                    # Add SYNC command to the command queue to interrupt the instance
                    # This will stop the instance, pull the model weights from PS, and resume generation.
                    psrl_logger.info(f"Queueing SYNC command for instance {instance_id}")
                    self.exec_command(Command(
                        type=CommandType.SYNC,
                        instance_id=instance_id,
                        curr_ps_model_version=self.ps_model_version,
                    ), blocking=False)
                    self.instance_to_version[instance_id] = self.ps_model_version
                self.check_partial_rollout = continue_to_check

        psrl_logger.info("Engine status sync loop stopped.")

    def set_ps_model_version(self, version: int):
        """
        Set the current PS model version.
        
        This method updates the internal PS model version and notifies any waiters
        that are waiting for this version or earlier.
        
        Args:
            version (int): The new PS model version to set.
        """
        self.ps_model_version = version
        if self.config.psrl.partial_rollout.enable:
            self.check_partial_rollout = True
        psrl_logger.debug(f"Set PS model version to {version}")
