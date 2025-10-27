import os
import logging
import asyncio
import numpy as np
from collections import defaultdict

import ray

from psrl.workers.gen.stats_collector import EngineStats
from psrl.utils.server.command import CommandType, Command, CommandExtension
from psrl.utils.logger import log_data_protocol, log_single_event, log_dual_events, EventType, DualOutputHandler

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
        self.check_partial_rollout = False # Whether to check for partial rollout in the loop

        # Stats collection
        self.status_queue = status_queue
        
        # Background event handler
        self.running_loop = None
        self.command_handler_task = None
        self.engine_status_sync_task = None
        self.stop_command_handler = False
        self.stop_engine_status_sync = False
        
        # Asyncio event loop order control
        self._is_init_model = asyncio.Event()
        self._is_init_nixl_client = asyncio.Event()
        
        # Version tracking
        self.instance_to_latest_stale_model_version: dict[int, int] = {}  # The latest stale model version of each instance
        self.instance_to_model_version: dict[int, int] = {}  # Track the model version of each instance
        self.ps_model_version = 0  # Current model version in the parameter server
        
        # Engine status tracking
        self.instance_to_engine_status: dict[int, EngineStats] = {}  # Track the latest engine stats of each instance
        
        # Build logger
        self.log_prefix = "RolloutCoordinator"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))
        
    def world_size(self):
        return sum([rollout_wg.world_size for rollout_wg in self.rollout_wg_list])

    async def init_model(self):
        futures = []
        for i in range(self.config.psrl.deployment.n_rollout_instances):
            if self.rank_0_is_model_owner:
                futures.append(self.rollout_wg_list[i].execute_rank_zero_async("init_model"))
            else:
                futures.extend(self.rollout_wg_list[i].execute_all_async("init_model"))
        await asyncio.gather(*futures)
        self._is_init_model.set()
         
    async def init_routing_strategy(self):
        await self._is_init_model.wait()
        futures = []
        for i in range(self.config.psrl.deployment.n_rollout_instances):
            if self.rank_0_is_model_owner:
                futures.append(self.rollout_wg_list[i].execute_rank_zero_async("estimate_max_model_len"))
            else:
                futures.extend(self.rollout_wg_list[i].execute_all_async("estimate_max_model_len"))
        max_model_lens = await asyncio.gather(*futures)
        psrl_logger.info(f"Max model lens: {max_model_lens}")
        self.max_model_len = max(max_model_lens)
        # TODO(lhy): use the max model len to budget the request number for each instance

    async def init_nixl_client(self):
        await self._is_init_model.wait()
        futures = []
        for i in range(self.config.psrl.deployment.n_rollout_instances):
            if self.rank_0_is_model_owner:
                futures.append(self.rollout_wg_list[i].execute_rank_zero_async("init_nixl_client"))
            else:
                futures.extend(self.rollout_wg_list[i].execute_all_async("init_nixl_client"))
        await asyncio.gather(*futures)
        self._is_init_nixl_client.set()
        
    async def nixl_protocol(self):
        await self._is_init_nixl_client.wait()
        futures = []
        for i in range(self.config.psrl.deployment.n_rollout_instances):
            if self.rank_0_is_model_owner:
                futures.append(self.rollout_wg_list[i].execute_rank_zero_async("nixl_protocol"))
            else:
                futures.extend(self.rollout_wg_list[i].execute_all_async("nixl_protocol"))
        await asyncio.gather(*futures)

    async def start_busy_loop(self):
        """
        Start the background event loops for command handling and status synchronization.
        
        This method:
        1. Registers all rollout instances with their respective worker groups
        2. Starts a background task for handling commands (abort, sync, etc.)
        3. Optionally starts a task for syncing engine status to agent workers
        """
        await self._is_init_model.wait()
        
        if self.command_handler_task is not None and not self.command_handler_task.done():
            return
        
        # Register rollout instances
        futures = []
        for i in range(self.config.psrl.deployment.n_rollout_instances):
            if self.rank_0_is_model_owner:
                futures.append(self.rollout_wg_list[i].execute_rank_zero_async("register_rollout_instance"))
            else:
                futures.extend(self.rollout_wg_list[i].execute_all_async("register_rollout_instance"))
        await asyncio.gather(*futures, return_exceptions=True)

        # Start the background tasks
        self.running_loop = asyncio.get_running_loop()
        self.command_handler_task = self.running_loop.create_task(self._command_handler_loop())
        self.command_handler_task.add_done_callback(lambda f: f.result())  # To avoid silent error in async tasks

        if self.config.psrl.status_collection.enable:
            # Start the engine status sync task
            self.engine_status_sync_task = self.running_loop.create_task(self._engine_status_sync_loop())
            self.engine_status_sync_task.add_done_callback(lambda f: f.result())  # To avoid silent error in async tasks
    
    async def stop_busy_loop(self):
        """
        Stop all background tasks and clean up resources.
        
        This method gracefully shuts down:
        - Command handler task
        - Engine status sync task
        """
        if self.command_handler_task is None or self.command_handler_task.done():
            return
        
        # Stop the background tasks
        self.stop_command_handler = True
        self.stop_engine_status_sync = True
        
        tasks_to_wait = [self.command_handler_task]
        if self.engine_status_sync_task is not None:
            tasks_to_wait.append(self.engine_status_sync_task)
        
        # Wait for tasks to finish with timeout
        await asyncio.gather(*tasks_to_wait, return_exceptions=True)
    
    async def _command_handler_loop(self):
        """
        Background loop for processing commands from the command queue.
        
        This method continuously processes different types of commands:
        - ABORT: Interrupt specific requests on instances
        - SYNC: Interrupt instance, pull new model weights, and resume generation
        
        The loop runs until stop_command_handler is set to True.
        """
        while not self.stop_command_handler:
            # Command processing
            if not self.command_queue.empty():
                command = self.command_queue.get_nowait()
                
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
                        interrupted_request_nums = await asyncio.gather(*futures)
                        interrupted_request_num = np.sum(interrupted_request_nums)
                    
                    result = interrupted_request_num
                    psrl_logger.info(f"Received ABORT command, interrupted {interrupted_request_num} requests")
                    
                    # Post process the command result
                    self._complete_command(command_id, result)
                elif command_type == CommandType.SYNC:
                    # Interrupt the instance, pull the model weights from PS and resume generation.
                    instance_ids = command_args.get("instance_ids", None)
                    curr_ps_model_version = command_args.get("curr_ps_model_version", None)
                    if not isinstance(instance_ids, list):
                        instance_ids = [instance_ids]
                    if instance_ids is None or curr_ps_model_version is None:
                        raise ValueError("SYNC command must contain 'instance_ids' and 'curr_ps_model_version' in args.")
                    psrl_logger.info(f"Received SYNC command for instances {instance_ids} with PS model version {curr_ps_model_version}")
                    assert self.config.gen_actor_rollout_ref.rollout.mode == "psrl_async", \
                        "SYNC command is only supported in 'psrl_async' rollout mode."
                    
                    # Sync with PS (interrupt, pull model, and resume generation)
                    futures = []
                    for instance_id in instance_ids:
                        future = None
                        if self.rank_0_is_model_owner:
                            future = self.rollout_wg_list[instance_id].execute_rank_zero_async("sync_with_ps", curr_ps_model_version)
                        else:
                            raise ValueError("SYNC command in SPMD-style is not supported yet.")
                        futures.append(future)
                    # Post process the command result
                    # NOTE(linsh): it's not necessary for engine status sync loop to wait for pulling from PS,
                    # so we simply mark the command as complete after.
                    self._complete_command(command_id, result)
                    interrupted_request_nums = await asyncio.gather(*futures)
                    for i, instance_id in enumerate(instance_ids):
                        psrl_logger.info(f"Synced with PS on instance {instance_id}, interrupted {interrupted_request_nums[i]} requests")
                else:
                    raise ValueError(f"Unknown command type: {command_type}")
            
            await asyncio.sleep(0)
        
        psrl_logger.info("Background command handler of rollout coordinator has finished.")
    
    async def _engine_status_sync_loop(self):
        """
        Background loop to collect engine status and sync to agent loop workers periodically.
        
        This method:
        1. Receives status updates from the status queue (engine waiting/running request counts)
        2. Consolidates status information from all instances
        3. Periodically broadcasts consolidated status to all agent loop workers
        4. Ensures agent workers have up-to-date information for decision making
        """        
        psrl_logger.info("Starting engine status sync loop")

        while not self.stop_engine_status_sync:
            # Wait for the next status update
            recv_stats = await self.status_queue.get_async(block=True)
            self.instance_to_engine_status[recv_stats.instance_id] = recv_stats
            psrl_logger.debug(f"Updated engine status for instance {recv_stats.instance_id}: {self.instance_to_engine_status[recv_stats.instance_id]}")
            
            # Send to all agent loop workers
            futures = []
            for agent_worker in self.agent_loop_workers:
                futures.append(agent_worker.update_instance_to_engine_status.remote(self.instance_to_engine_status))
            # Wait for all updates to complete using asyncio
            await asyncio.gather(*futures)
            
            # partial rollout check
            if self.config.psrl.partial_rollout.enable and self.check_partial_rollout:
                continue_to_check = False
                sync_instance_ids = []
                for instance_id, engine_stats in self.instance_to_engine_status.items():
                    # Check whether the received stats is stale
                    if engine_stats.model_version == self.instance_to_latest_stale_model_version.get(instance_id, -1):
                        continue
                    # Check whether instance version lags behind PS version
                    if self.instance_to_model_version.get(instance_id, 0) == self.ps_model_version:
                        continue
                    # Check whether current instance workload is below threshold
                    # Currently we consider running queue size as the workload metric
                    running_queue_size = engine_stats.snapshot["scheduler_stats"]["num_running_reqs"]
                    psrl_logger.debug(f"Instance {instance_id} (version {self.instance_to_model_version.get(instance_id, 0)}) "
                                      f"workload: {running_queue_size}, "
                                      f"threshold: {self.config.psrl.partial_rollout.threshold}")
                    if running_queue_size > self.config.psrl.partial_rollout.threshold or running_queue_size == 0:
                        psrl_logger.debug(f"Instance {instance_id} (version {self.instance_to_model_version.get(instance_id, 0)}) workload {running_queue_size} is above threshold {self.config.psrl.partial_rollout.threshold}, will not synchronize with PS (version {self.ps_model_version}) this time")
                        continue_to_check = True
                        continue
                    # Check whether instance workload would increase after update
                    # NOTE(linsh): this step relies on static version tag assignment
                    inc_request_num = await self.rollout_wg_list[instance_id].execute_rank_zero_async("get_workload_after_update_to", self.ps_model_version)
                    psrl_logger.debug(f"Instance {instance_id} workload after update would be {inc_request_num + running_queue_size}")
                    if inc_request_num == 0:
                        continue_to_check = True
                        continue

                    sync_instance_ids.append(instance_id)

                # Add batching SYNC command to the command queue to interrupt the instance
                # This will stop the instance, pull the model weights from PS, and resume generation.
                # But this will not block the current loop.
                if sync_instance_ids:
                    # NOTE(lhy): we don't need to update the instance version here because the version is updated in the `sync_with_ps` method of the GenWorker
                    # when calling `pull_model` or `pull_model_async` from the GenWorker, the ps manager will update the instance version.
                    # However, we need to update the latest stale model version here to avoid stale stats being handled after the synchronization.
                    with log_dual_events(f"Synchronize rollout instances {sync_instance_ids} with PS (model pull is non-blocking)", psrl_logger, level=logging.INFO, event_type=EventType.OTHER):
                        for instance_id in sync_instance_ids:
                            self.instance_to_latest_stale_model_version[instance_id] = self.instance_to_model_version.get(instance_id, 0)    
                        await self.exec_command(Command(
                            type=CommandType.SYNC,
                            instance_ids=sync_instance_ids,
                            curr_ps_model_version=self.ps_model_version,
                        ), blocking=True)

                self.check_partial_rollout = continue_to_check

            await asyncio.sleep(0)

        psrl_logger.info("Engine status sync loop stopped.")

    # This is called by the PS manager to update the PS model version after pushing
    def set_ps_model_version(self, version: int):
        """
        Set the current PS model version.
        
        This method updates the internal PS model version.
        
        Args:
            version (int): The new PS model version to set.
        """
        self.ps_model_version = version
        if self.config.psrl.partial_rollout.enable:
            self.check_partial_rollout = True
        psrl_logger.debug(f"Set PS model version to {version}")
     
    # This is called by the PS manager to update the rollout instance model version after pulling
    def set_rollout_instance_model_version(self, rollout_instance_id: int, version_tag: int):
        """
        Set the model version for a specific rollout instance.
        
        Args:
            rollout_instance_id (int): The ID of the rollout instance.
            version_tag (int): The model version tag to set for the instance.
        """
        old_version = self.instance_to_model_version.get(rollout_instance_id, None)
        self.instance_to_model_version[rollout_instance_id] = version_tag
        psrl_logger.debug(f"Updated instance {rollout_instance_id} model version: {old_version} -> {version_tag}")
