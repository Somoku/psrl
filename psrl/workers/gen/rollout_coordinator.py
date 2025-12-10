import os
import logging
import asyncio
import warnings
import ray
import numpy as np
from typing import Set, List

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
        rollout_wg_list,
        agent_loop_workers,
        status_queues,
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
            rollout_wg_list: List of rollout worker groups
            agent_loop_workers: List of agent loop worker handles
            status_queues: Queues for receiving status updates from different rollout instances
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
        
        self.rollout_wg_list = rollout_wg_list
        self.rollout_wg_size = len(rollout_wg_list)
        assert self.rollout_wg_size == self.config.psrl.deployment.n_rollout_instances, \
            "The number of rollout worker groups must be the same as the number of rollout instances."
        self.agent_loop_workers = agent_loop_workers

        # Stats collection
        self.status_queues = status_queues
        assert len(self.status_queues) == self.config.psrl.deployment.n_rollout_instances, \
            "The number of status queues must be the same as the number of rollout instances."
        
        # Background event handler
        self.running_loop = None
        self.command_handler_task = None
        self.sync_task = None
        self.process_status_queue_tasks = []
        self.broadcast_status_to_router_task = None
        self.stop_command_handler = False
        self.stop_sync_and_migrate = False
        self.stop_process_status_queue = [False] * self.config.psrl.deployment.n_rollout_instances
        self.stop_broadcast_status_to_router = False
        
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
        # Register rollout instances after initializing the model
        for i in range(self.config.psrl.deployment.n_rollout_instances):
            if self.rank_0_is_model_owner:
                futures.append(self.rollout_wg_list[i].execute_rank_zero_async("register_rollout_instance"))
            else:
                futures.extend(self.rollout_wg_list[i].execute_all_async("register_rollout_instance"))
        await asyncio.gather(*futures)
        self._is_init_model.set()
         
    async def init_route_strategy(self):
        await self._is_init_model.wait()
        futures = []
        for i in range(self.config.psrl.deployment.n_rollout_instances):
            if self.rank_0_is_model_owner:
                futures.append(self.rollout_wg_list[i].execute_rank_zero_async("estimate_max_model_len"))
            else:
                futures.extend(self.rollout_wg_list[i].execute_all_async("estimate_max_model_len"))
        max_model_lens = await asyncio.gather(*futures)
        psrl_logger.info(f"Max model lens: {max_model_lens}")
        instance_to_max_model_len = {i: max(max_model_lens[i]) for i in range(self.config.psrl.deployment.n_rollout_instances)}
        # Use the max model len to budget the kv cache size for each instance
        futures = []
        for agent_worker in self.agent_loop_workers:
            futures.append(agent_worker.init_route_strategy.remote(
                instance_to_max_model_len=instance_to_max_model_len,
            ))
        await asyncio.gather(*futures)

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
        1. Starts a background task for handling commands (abort, sync, etc.).
        2. Optionally starts tasks for processing status queues of each rollout instance.
        3. Starts a task to broadcast the engine status to the agent loop workers (i.e., router).
        4. Starts a task to synchronize rollout instances with PS.
        """
        await self._is_init_model.wait()
        
        if self.command_handler_task is not None and not self.command_handler_task.done():
            return

        # Start the background tasks
        self.running_loop = asyncio.get_running_loop()
        self.command_handler_task = self.running_loop.create_task(self._command_handler_loop())
        self.command_handler_task.add_done_callback(lambda f: f.result())  # To avoid silent error in async tasks

        if self.config.psrl.gen_mode == "stream":
            # Start the status collection tasks
            if self.config.psrl.status_collection.enable:
                for instance_id in range(self.config.psrl.deployment.n_rollout_instances):
                    self.process_status_queue_tasks.append(self.running_loop.create_task(self._process_status_queue(instance_id)))
                    self.process_status_queue_tasks[instance_id].add_done_callback(lambda f: f.result())  # To avoid silent error in async tasks
            # Start the task to broadcast the engine status to the router
            self.broadcast_status_to_router_task = self.running_loop.create_task(self._broadcast_status_to_router())
            self.broadcast_status_to_router_task.add_done_callback(lambda f: f.result())  # To avoid silent error in async tasks
            # Start the model synchronization and rollout migration loop
            if self.config.psrl.sync_and_mig_strategy.method == "greedy":
                self.sync_task = self.running_loop.create_task(self._greedy_sync_and_migrate_loop())
                self.sync_task.add_done_callback(lambda f: f.result())  # To avoid silent error in async tasks
            elif self.config.psrl.sync_and_mig_strategy.method == "status_based":
                assert self.config.psrl.status_collection.enable, "Status-based sync strategy is only supported when status collection is enabled"
                self.sync_task = self.running_loop.create_task(self._status_based_sync_and_migrate_loop())
                self.sync_task.add_done_callback(lambda f: f.result())  # To avoid silent error in async tasks
            else:
                raise NotImplementedError(f"Sync strategy {self.config.psrl.sync_and_mig_strategy.method} is not supported")
            # Check if rollout migration is enabled
            if self.config.psrl.sync_and_mig_strategy.mig.enable:
                assert self.config.psrl.status_collection.enable, "Rollout migration is only supported when status collection is enabled"
                assert self.config.psrl.partial_rollout.enable, "Rollout migration is only supported when partial rollout is enabled"
    
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
        self.stop_sync_and_migrate = True
        self.stop_process_status_queue = [True] * self.config.psrl.deployment.n_rollout_instances
        self.stop_broadcast_status_to_router = True
        
        tasks_to_wait = [self.command_handler_task]
        if self.config.psrl.gen_mode == "stream":
            tasks_to_wait.append(self.sync_task)
            if self.process_status_queue_tasks:
                tasks_to_wait.extend(self.process_status_queue_tasks)
            tasks_to_wait.append(self.broadcast_status_to_router_task)
            
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
                    instance_to_uids = command_args.get("instance_to_uids", None)
                    instance_ids = command_args.get("instance_ids", None)
                    if instance_to_uids is None and instance_ids is None:
                        raise ValueError("ABORT command must contain 'instance_to_uids' or 'instance_ids' in args.")
                    
                    psrl_logger.info(f"Received ABORT command with instance_to_uids: {instance_to_uids} and instance_ids: {instance_ids}")
                    futures = []
                    
                    if instance_to_uids is not None:
                        for instance_id, uids in instance_to_uids.items():
                            if not uids:
                                continue
                            if not isinstance(uids, (list, set)):
                                uids = [uids]
                            abort_requests = set(uids)  # Ensure uniqueness
                            if self.rank_0_is_model_owner:
                                futures.append(self.rollout_wg_list[instance_id].execute_rank_zero_async("interrupt_requests", abort_requests))
                            else:
                                raise ValueError("ABORT command in SPMD-style is not supported yet.")
                    if instance_ids is not None:
                        for instance_id in instance_ids:
                            if self.rank_0_is_model_owner:
                                futures.append(self.rollout_wg_list[instance_id].execute_rank_zero_async("interrupt_requests", None))
                            else:
                                raise ValueError("ABORT command in SPMD-style is not supported yet.")
                    
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
                    wait_model_sync = command_args.get("wait_model_sync", False)
                    if not isinstance(instance_ids, list):
                        instance_ids = [instance_ids]
                    if instance_ids is None or curr_ps_model_version is None:
                        raise ValueError("SYNC command must contain 'instance_ids' and 'curr_ps_model_version' in args.")
                    psrl_logger.info(f"Received SYNC command for instances {instance_ids} with PS model version {curr_ps_model_version}")
                    assert self.config.gen_actor_rollout_ref.rollout.mode == "psrl_async", \
                        "SYNC command is only supported in 'psrl_async' rollout mode."
                    
                    # Sync with PS (interrupt, pull model, and resume generation)
                    interrupt_futures = []
                    sync_futures = []
                    
                    for instance_id in instance_ids:
                        if self.rank_0_is_model_owner:
                            interrupt_future = self.rollout_wg_list[instance_id].execute_rank_zero_async("interrupt_generation")
                        else:
                            raise ValueError("SYNC command in SPMD-style is not supported yet.")
                        interrupt_futures.append(interrupt_future)
                    interrupted_request_nums = await asyncio.gather(*interrupt_futures)  
                    for i, instance_id in enumerate(instance_ids):
                        psrl_logger.info(f"Syncing with PS on instance {instance_id}, interrupted {interrupted_request_nums[i]} requests") 
                    
                    for instance_id in instance_ids:
                        if self.rank_0_is_model_owner:
                            sync_future = self.rollout_wg_list[instance_id].execute_rank_zero_async("sync_with_ps", curr_ps_model_version)
                        else:
                            raise ValueError("SYNC command in SPMD-style is not supported yet.")
                        sync_futures.append(sync_future)
                        
                    # Post process the command result
                    if wait_model_sync:
                        await asyncio.gather(*sync_futures)
                        self._complete_command(command_id, interrupted_request_nums)
                    else:
                        # NOTE(linsh): sometimes it's not necessary for the caller to wait for pulling from PS
                        self._complete_command(command_id, interrupted_request_nums)
                        await asyncio.gather(*sync_futures)  # Wait for the sync to complete
                else:
                    raise ValueError(f"Unknown command type: {command_type}")
            
            await asyncio.sleep(0)
        
        psrl_logger.info("Background command handler of rollout coordinator has finished.")
    
    async def _process_status_queue(self, instance_id: int):
        psrl_logger.info(f"Starting to process status queue for instance {instance_id}")
        while not self.stop_process_status_queue[instance_id]:
            # TODO(lhy): add timeout handling for future fault tolerance of rollout instances
            recv_stats = await self.status_queues[instance_id].get_async(block=True, timeout=None)
            self.instance_to_engine_status[instance_id] = recv_stats
            psrl_logger.debug(f"Updated engine status for instance {recv_stats.instance_id}: {self.instance_to_engine_status[instance_id]}")
            
    async def _broadcast_status_to_router(self):
        """
        Broadcast the engine status to the router.
        """
        while not self.stop_broadcast_status_to_router:
            # Broadcast the engine status to the router every coordinator sync interval
            await asyncio.sleep(self.config.psrl.status_collection.coordinator_sync_interval_in_ms / 1000) 
            futures = []
            # Send to all agent loop workers to update the instance status
            # TODO(lhy): change it to a global router
            for agent_worker in self.agent_loop_workers:
                futures.append(agent_worker.update_instance_status.remote(self.instance_to_engine_status))
            await asyncio.gather(*futures)
            
    async def _is_routing(self) -> bool:
        """Check if any agent loop worker is currently routing requests (i.e., the router is currently routing requests).
        
        Returns:
            bool: True if any agent loop worker is currently routing requests, False otherwise.
        """
        is_routing = await asyncio.gather(*[agent_worker.is_routing.remote() for agent_worker in self.agent_loop_workers])
        # psrl_logger.info(f"Is routing: {is_routing}")
        return any(is_routing)
    
    async def _greedy_sync_and_migrate_loop(self):
        """
        Background loop to synchronize with PS based on the greedy algorithm.
        
        This method:
        1. Greedily synchronize with PS for rollout that lags behind PS version.
        2. Check whether the instance has no active tasks if forbid partial rollout.
        3. Check if any instance is starving and do migration if necessary.
        """
        psrl_logger.info("Starting greedy model synchronization and rollout migration loop")

        while not self.stop_sync_and_migrate:
            # Sleep for a period of time
            await asyncio.sleep(self.config.psrl.sync_and_mig_strategy.check_interval_in_ms / 1000) 
            
            have_syncing_instance = False
            sync_instance_ids = []
            for instance_id in range(self.config.psrl.deployment.n_rollout_instances):
                # Check whether engine status is stale (the instance is currently being synchronized with PS)
                if self.instance_to_model_version.get(instance_id, 0) <= self.instance_to_latest_stale_model_version.get(instance_id, -1):
                    have_syncing_instance = True
                    continue
                # Check whether instance version lags behind PS version
                if self.instance_to_model_version.get(instance_id, 0) == self.ps_model_version:
                    continue
                # Check whether current instance workload is empty if forbid partial rollout
                if not self.config.psrl.partial_rollout.enable:
                    if not await self.check_no_activate_tasks(instance_id):
                        continue
                # Add the instance to the sync list
                sync_instance_ids.append(instance_id)
                
            if sync_instance_ids:
                await self.sync_with_ps(sync_instance_ids)
            elif not have_syncing_instance and self.config.psrl.sync_and_mig_strategy.mig.enable:
                # No instance is syncing with PS, check if migration is needed
                await self.check_and_migrate()

        psrl_logger.info("Greedy model synchronization and rollout migration loop stopped.")
    
    async def _status_based_sync_and_migrate_loop(self):
        """
        Background loop to collect engine status and decide whether to synchronize with PS based on the engine status.
        
        This method:
        1. Analyze the instance status (engine waiting & running request counts, etc.).
        2. Decide whether to synchronize with PS for each instance.
        """        
        psrl_logger.info("Starting status based model synchronization and rollout migration loop")

        while not self.stop_sync_and_migrate:
            # Sleep for a period of time and analyze the instance status
            await asyncio.sleep(self.config.psrl.sync_and_mig_strategy.check_interval_in_ms / 1000) 
            
            have_syncing_instance = False
            sync_instance_ids = []
            for instance_id, engine_stats in self.instance_to_engine_status.items():
                # Check whether engine status is stale (the instance is currently being synchronized with PS)
                if engine_stats.model_version <= self.instance_to_latest_stale_model_version.get(instance_id, -1):
                    have_syncing_instance = True
                    continue
                # We do not synchronize with PS if the router is currently routing requests
                if (await self._is_routing()):
                    # psrl_logger.info(f"Skipping synchronization with PS for instance {instance_id} because the router is currently routing requests")
                    continue
                # Check whether instance version lags behind PS version
                if self.instance_to_model_version.get(instance_id, 0) == self.ps_model_version:
                    continue
                # Check whether current instance workload is empty (forbid partial rollout) or satisfies the partial rollout policy
                if self.config.psrl.partial_rollout.enable:
                    if not (await self.check_should_sync(instance_id)):
                        continue
                else:
                    if engine_stats.get_waiting_and_running_queue_size() > 0:
                        continue
                # Add the instance to the sync list
                sync_instance_ids.append(instance_id)
                # NOTE(lhy): currently, we only synchronize with PS for one instance at a time
                # But the model pulling time can be overlapped
                break

            if sync_instance_ids:
                await self.sync_with_ps(sync_instance_ids)
            elif not have_syncing_instance and self.config.psrl.sync_and_mig_strategy.mig.enable:
                # No instance is syncing with PS, check if migration is needed
                await self.check_and_migrate()

        psrl_logger.info("Status based model synchronization and rollout migration loop stopped.")
    
    # ------- FUNCTIONS FOR MODEL SYNCING -------

    # This is called by the PS manager to update the PS model version after pushing
    def set_ps_model_version(self, version: int):
        """
        Set the current PS model version.
        
        This method updates the internal PS model version.
        
        Args:
            version (int): The new PS model version to set.
        """
        self.ps_model_version = version
        psrl_logger.info(f"Updated PS model version to {version}")
     
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
        psrl_logger.info(f"Updated rollout instance {rollout_instance_id} model version: {old_version} -> {version_tag}")

    async def sync_with_ps(self, instance_ids: List[int], wait_model_sync: bool = False, wait_interrupted_partial_requests_loop_back: bool = True):
        """
        Synchronize with PS for the given instance IDs.
        """
        # Add batching SYNC command to the command queue to interrupt the instance
        # This will stop the instance, pull the model weights from PS, and resume generation.
        # But this will not block the current loop.
        # NOTE(lhy): we don't need to update the instance version here because the version is updated in the `sync_with_ps` method of the GenWorker
        # when calling `pull_model` or `pull_model_async` from the GenWorker, the ps manager will update the instance version.
        # However, we need to update the latest stale model version here to avoid stale stats being handled after the synchronization.
        with log_dual_events(f"Synchronize rollout instances {instance_ids} with PS (model pull is {'non-blocking' if not wait_model_sync else 'blocking'} for the coordinator)", psrl_logger, level=logging.INFO, event_type=EventType.OTHER):
            for instance_id in instance_ids:
                self.instance_to_latest_stale_model_version[instance_id] = self.instance_to_model_version.get(instance_id, 0)   
            await self.agent_loop_workers[0].interrupt_routing.remote()
            psrl_logger.info(f"Interrupted routing for synchronization")
            await self.agent_loop_workers[0].update_currently_syncing_instances.remote(instance_ids, self.ps_model_version) 
            await self.exec_command(Command(
                type=CommandType.SYNC,
                instance_ids=instance_ids,
                curr_ps_model_version=self.ps_model_version,
                wait_model_sync=wait_model_sync,
            ), blocking=True)
            if wait_interrupted_partial_requests_loop_back and self.config.psrl.partial_rollout.enable:
                await self.agent_loop_workers[0].wait_interrupted_partial_requests_loop_back.remote(instance_ids)
                psrl_logger.info(f"All interrupted requests on the synchronized instances {instance_ids} have been looped back")
            await self.agent_loop_workers[0].resume_routing.remote()
            psrl_logger.info(f"Resumed routing after synchronization")
            
    async def check_no_activate_tasks(self, instance_id: int) -> bool:
        """
        Check whether the instance has no active tasks.
        """
        futures = []
        if self.rank_0_is_model_owner:
            futures.append(self.rollout_wg_list[instance_id].execute_rank_zero_async("get_active_task_num"))
        else:
            warnings.warn(f"Check no active tasks on instance {instance_id} in SPMD-style may cause undefined behavior, need to check the behavior")
            futures.extend(self.rollout_wg_list[instance_id].execute_all_async("get_active_task_num"))
        active_task_nums = await asyncio.gather(*futures)
        return all(active_task_num == 0 for active_task_num in active_task_nums)

    async def check_should_sync(self, instance_id: int) -> bool:
        """
        Check whether to synchronize with PS for the instance.
        """
        assert self.config.psrl.sync_and_mig_strategy.method == "status_based", "Partial rollout is only supported for status-based sync strategy"
        assert self.config.psrl.status_collection.enable, "Partial rollout is only supported when status collection is enabled"
        assert self.config.psrl.partial_rollout.enable, "Partial rollout is not enabled"
        # TODO(lhy): refactor the router to be a global router
        # psrl_logger.info(f"Checking whether to synchronize with PS for instance {instance_id}, ps model version: {self.ps_model_version}")
        return await self.agent_loop_workers[0].check_should_sync.remote(instance_id, self.ps_model_version)
    
    async def check_and_migrate(self, wait_interrupted_partial_requests_loop_back: bool = True):
        """
        Check if any instance is starving and do migration if necessary.
        """
        assert self.config.psrl.sync_and_mig_strategy.mig.enable, "Rollout migration is not enabled"
        assert self.config.psrl.status_collection.enable, "Rollout migration is only supported when status collection is enabled"
        # psrl_logger.info("Checking if any instance is starving and doing migration if necessary")
        migrate_instance_ids = await self.agent_loop_workers[0].check_should_migrate.remote()
        if migrate_instance_ids:
            psrl_logger.info(f"Migrating instances {migrate_instance_ids} to new PS model version {self.ps_model_version}")
            await self.agent_loop_workers[0].interrupt_routing.remote()
            psrl_logger.info(f"Interrupted routing for migration")
            await self.exec_command(Command(
                type=CommandType.ABORT,
                instance_ids=migrate_instance_ids,
            ), blocking=True)
            if wait_interrupted_partial_requests_loop_back:
                await self.agent_loop_workers[0].wait_interrupted_partial_requests_loop_back.remote(migrate_instance_ids)
                psrl_logger.info(f"All interrupted requests on the migrated instances {migrate_instance_ids} have been looped back")
            await self.agent_loop_workers[0].resume_routing.remote()
            psrl_logger.info(f"Resumed routing after migration")