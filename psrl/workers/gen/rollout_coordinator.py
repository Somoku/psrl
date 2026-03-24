import asyncio
import logging
import os
import warnings

import numpy as np
import ray

from psrl.utils.logger import (
    DualOutputHandler,
    EventType,
    log_dual_events,
)
from psrl.utils.server.command import Command, CommandExtension, CommandType
from psrl.workers.gen.stats_collector import EngineStats

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@ray.remote
class RolloutCoordinator(CommandExtension):
    def __init__(
        self,
        config,
        rollout_router,
        rollout_wg_list,
        validate_wg_list,
        agent_loop_workers,
        status_queues,
    ):
        """
        Initialize the RolloutCoordinator.
        Coordinates and manages rollout instances for PSRL.

        This class handles:
        - Registering and tracking rollout instances
        - Managing model version synchronization across instances
        - Handling command execution (abort, sync)
        - Collecting and distributing engine status information
        - Coordinating interruption and resumption of generation tasks

        Args:
            config: Configuration object containing PSRL settings
            rollout_router: Handle to the rollout router actor
            rollout_wg_list: List of rollout worker groups
            validate_wg_list: List of validation worker groups
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

        self.rollout_wg_list = rollout_wg_list
        self.rollout_wg_size = len(rollout_wg_list)
        self.validate_wg_list = validate_wg_list
        self.validate_wg_size = len(validate_wg_list)

        # All rollout and validate worker groups
        self.gen_wg_list = self.rollout_wg_list + self.validate_wg_list
        self.gen_wg_size = self.rollout_wg_size + self.validate_wg_size

        self.n_rollout_instances = self.config.psrl.deployment.n_rollout_instances
        self.n_validate_instances = (
            self.config.psrl.deployment.n_validate_instances if self.config.psrl.colocate_validate_and_train else 0
        )

        assert self.rollout_wg_size == self.n_rollout_instances, (
            "The number of rollout worker groups must be the same as the number of rollout instances, "
            f"but got {self.rollout_wg_size} and {self.n_rollout_instances}."
        )
        assert self.validate_wg_size == self.n_validate_instances, (
            "The number of validate worker groups must be the same as the number of validate instances, "
            f"but got {self.validate_wg_size} and {self.n_validate_instances}."
        )
        self.agent_loop_workers = agent_loop_workers
        self.rollout_router = rollout_router

        # Stats collection
        self.status_queues = status_queues
        assert len(self.status_queues) == self.gen_wg_size, (
            "The number of status queues must be the same as the number of rollout instances, "
            f"but got {len(self.status_queues)} and {self.gen_wg_size}."
        )

        # Background event handler
        self.running_loop = None
        self.command_handler_task = None
        self.sync_task = None
        self.process_status_queue_tasks = []
        self.broadcast_status_to_router_task = None
        self.stop_command_handler = False
        self.stop_sync_and_migrate = False
        self.stop_process_status_queue = [False] * self.gen_wg_size
        self.stop_broadcast_status_to_router = False

        # Asyncio event loop order control
        self._is_init_model = asyncio.Event()
        self._is_init_nixl_client = asyncio.Event()

        # Version tracking
        self.instance_to_latest_stale_model_version: dict[
            int, int
        ] = {}  # The latest stale model version of each instance
        self.instance_to_model_version: dict[int, int] = {}  # Track the model version of each instance
        self.ps_model_version = 0  # Current model version in the parameter server
        self.ready_buffers = set() # The set of ready buffers

        # Engine status tracking
        self.instance_to_engine_status: dict[int, EngineStats] = {}  # Track the latest engine stats of each instance

        # Build logger
        self.log_prefix = "RolloutCoordinator"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))

    def world_size(self):
        """Get the total world size (number of rollout and validate instances)."""
        return sum([rollout_wg.world_size for rollout_wg in self.gen_wg_list])

    def resume_instances(self, instance_ids: list[int]):
        """Notify that the given instances have resumed processing.

        Args:
            instance_ids (list[int]): List of instance IDs that have resumed processing.
        """
        for instance_id in instance_ids:
            self.stop_process_status_queue[instance_id] = False

    def pause_instances(self, instance_ids: list[int]):
        """Notify that the given instances have paused processing.

        Args:
            instance_ids (list[int]): List of instance IDs that have paused processing.
        """
        for instance_id in instance_ids:
            self.stop_process_status_queue[instance_id] = True

    def _get_wg_list_and_size(self, tag: str):
        """Get the worker group list and size based on the given tag.

        Args:
            tag (str): Tag to specify which instances to get ('rollout', 'validate', 'all')
        Returns:
            tuple: (worker group list, worker group size)
        """
        if tag == "rollout":
            return self.rollout_wg_list, self.rollout_wg_size
        elif tag == "validate":
            return self.validate_wg_list, self.validate_wg_size
        elif tag == "all":
            return self.gen_wg_list, self.gen_wg_size
        else:
            raise ValueError(f"Unknown tag {tag} for getting worker group list and size")

    async def init_model(self, tag: str = "rollout", init_mode: str = "full"):
        """Init the model on rollout instances and register to ps manager.

        Args:
            tag (str): Tag to specify which instances to initialize ('rollout', 'validate', 'all')
            init_mode (str): Initialization mode ('full', 'empty', etc.)
                'full' mode will load the full model weights,
                'empty' mode will load dummy model weights.
        """
        wg_list, wg_size = self._get_wg_list_and_size(tag)
        futures = []
        for i in range(wg_size):
            futures.append(wg_list[i].execute_rank_zero_async("init_and_register_model", init_mode))
        await asyncio.gather(*futures)
        self._is_init_model.set()

    async def init_route_strategy(self, tag: str = "rollout"):
        """Init the route strategy on rollout instances.

        This method estimates the maximum model length on each instance
        and uses it to budget the kv cache size for each instance.

        Args:
            tag (str): Tag to specify which instances to initialize ('rollout', 'validate', 'all')
        """
        assert self.rollout_router is not None, "Rollout router is not set in RolloutCoordinator"
        await self._is_init_model.wait()

        wg_list, wg_size = self._get_wg_list_and_size(tag)
        futures = []
        for i in range(wg_size):
            futures.append(wg_list[i].execute_rank_zero_async("estimate_max_model_len"))
        max_model_lens = await asyncio.gather(*futures)
        psrl_logger.info(f"Max model lens on {tag} instances: {max_model_lens}")
        wg_idx_range = range(self.rollout_wg_size, self.gen_wg_size) if tag == "validate" else range(wg_size)
        instance_to_max_model_len = {i: max(max_model_lens[j]) for i, j in zip(wg_idx_range, range(wg_size))}
        # Use the max model len to budget the kv cache size for each instance
        await self.rollout_router.init_route_strategy.remote(instance_to_max_model_len=instance_to_max_model_len)

    async def init_nixl_client(self):
        """Init the NIXL client on rollout and validate instances."""
        await self._is_init_model.wait()
        futures = []
        for i in range(self.gen_wg_size):
            futures.append(self.gen_wg_list[i].execute_rank_zero_async("init_nixl_client"))
        await asyncio.gather(*futures)
        psrl_logger.info(f"Initialized NIXL client on all {self.gen_wg_size} instances.")
        self._is_init_nixl_client.set()

    async def nixl_protocol(self, full_tag: str = "all"):
        """Run the NIXL server protocol on rollout and validate instances.

        Args:
            full_tag (str): Tag to specify which instances to run the protocol
                            in 'full' mode ('rollout', 'validate', 'all')
        """
        await self._is_init_nixl_client.wait()

        if full_tag == "all":
            full_tag_list = ["full"] * self.gen_wg_size
        elif full_tag == "rollout":
            full_tag_list = ["full"] * self.rollout_wg_size + ["meta"] * self.validate_wg_size
        elif full_tag == "validate":
            full_tag_list = ["meta"] * self.rollout_wg_size + ["full"] * self.validate_wg_size
        else:
            raise ValueError(f"Unknown full_tag {full_tag} for nixl_convert_params")

        futures = []
        for i in range(self.gen_wg_size):
            futures.append(self.gen_wg_list[i].execute_rank_zero_async("nixl_protocol", full_tag_list[i]))
        await asyncio.gather(*futures)

    async def nixl_convert_params(self):
        """Convert the model parameters to unified format on rollout and validate instances."""
        await self._is_init_nixl_client.wait()
        futures = []
        for i in range(self.gen_wg_size):
            futures.append(self.gen_wg_list[i].execute_rank_zero_async("nixl_convert_params"))
        await asyncio.gather(*futures)

    async def sleep(self, tag: str = "all"):
        """Make rollout instances sleep and release GPU memory.

        Args:
            tag (str): Tag to specify which instances to sleep ('rollout', 'validate', 'all')
        """
        await self._is_init_model.wait()

        wg_list, wg_size = self._get_wg_list_and_size(tag)
        futures = []
        for i in range(wg_size):
            futures.append(wg_list[i].execute_rank_zero_async("sleep"))
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

        # Start the status collection tasks
        if self.config.psrl.status_collection.enable:
            for instance_id in range(self.gen_wg_size):
                self.process_status_queue_tasks.append(
                    self.running_loop.create_task(self._process_status_queue(instance_id))
                )
                self.process_status_queue_tasks[instance_id].add_done_callback(
                    lambda f: f.result()
                )  # To avoid silent error in async tasks
        # Start the task to broadcast the engine status to the router
        self.broadcast_status_to_router_task = self.running_loop.create_task(self._broadcast_status_to_router())
        self.broadcast_status_to_router_task.add_done_callback(
            lambda f: f.result()
        )  # To avoid silent error in async tasks
        # Start the model synchronization and rollout migration loop
        if self.config.psrl.sync_and_mig_strategy.method == "greedy":
            self.sync_task = self.running_loop.create_task(self._greedy_sync_and_migrate_loop())
            self.sync_task.add_done_callback(lambda f: f.result())  # To avoid silent error in async tasks
        elif self.config.psrl.sync_and_mig_strategy.method == "status_based":
            assert self.config.psrl.status_collection.enable, (
                "Status-based sync strategy is only supported when status collection is enabled"
            )
            self.sync_task = self.running_loop.create_task(self._status_based_sync_and_migrate_loop())
            self.sync_task.add_done_callback(lambda f: f.result())  # To avoid silent error in async tasks
        else:
            raise NotImplementedError(
                f"Sync strategy {self.config.psrl.sync_and_mig_strategy.method} is not supported"
            )
        # Check if rollout migration is enabled
        if self.config.psrl.sync_and_mig_strategy.mig.enable:
            assert self.config.psrl.status_collection.enable, (
                "Rollout migration is only supported when status collection is enabled"
            )
            assert self.config.psrl.partial_rollout.enable, (
                "Rollout migration is only supported when partial rollout is enabled"
            )

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
        self.stop_process_status_queue = [True] * self.gen_wg_size
        self.stop_broadcast_status_to_router = True

        tasks_to_wait = [self.command_handler_task]
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
                psrl_logger.info(
                    f"Receive command: type = {command_type}, kwargs = {command.get_kwargs()}, args = {command_args}"
                )

                result = None
                # Process the command based on its type
                if command_type == CommandType.ABORT:
                    instance_to_uids = command_args.get("instance_to_uids", None)
                    instance_ids = command_args.get("instance_ids", None)
                    if instance_to_uids is None and instance_ids is None:
                        raise ValueError("ABORT command must contain 'instance_to_uids' or 'instance_ids' in args.")

                    psrl_logger.info(
                        f"Received ABORT command with instance_to_uids: "
                        f"{instance_to_uids} and instance_ids: {instance_ids}"
                    )
                    futures = []

                    if instance_to_uids is not None:
                        for instance_id, uids in instance_to_uids.items():
                            if not uids:
                                continue
                            if not isinstance(uids, (list, set)):
                                uids = [uids]
                            abort_requests = set(uids)  # Ensure uniqueness
                            assert instance_id < len(self.rollout_wg_list), (
                                f"Validate instance should not be interrupted, but got instance_id {instance_id} "
                                f"which is out of rollout instance range [0, {len(self.rollout_wg_list)})."
                            )
                            futures.append(
                                self.gen_wg_list[instance_id].execute_rank_zero_async(
                                    "interrupt_requests", abort_requests
                                )
                            )
                    if instance_ids is not None:
                        for instance_id in instance_ids:
                            futures.append(
                                self.gen_wg_list[instance_id].execute_rank_zero_async(
                                    "interrupt_requests", None
                                )
                            )

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
                        raise ValueError(
                            "SYNC command must contain 'instance_ids' and 'curr_ps_model_version' in args."
                        )
                    psrl_logger.info(
                        f"Received SYNC command for instances {instance_ids} "
                        f"with PS model version {curr_ps_model_version}"
                    )

                    # Sync with PS (interrupt, pull model, and resume generation)
                    interrupt_futures = []
                    sync_futures = []

                    for instance_id in instance_ids:
                        interrupt_future = self.gen_wg_list[instance_id].execute_rank_zero_async(
                            "interrupt_generation"
                        )
                        interrupt_futures.append(interrupt_future)
                    interrupted_request_nums = await asyncio.gather(*interrupt_futures)
                    for i, instance_id in enumerate(instance_ids):
                        psrl_logger.info(
                            f"Syncing with PS on instance {instance_id}, "
                            f"interrupted {interrupted_request_nums[i]} requests"
                        )

                    for instance_id in instance_ids:
                        sync_future = self.gen_wg_list[instance_id].execute_rank_zero_async(
                            "sync_with_ps", curr_ps_model_version
                        )
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
            psrl_logger.debug(
                f"Updated engine status for instance "
                f"{recv_stats.instance_id}: {self.instance_to_engine_status[instance_id]}"
            )

    async def _broadcast_status_to_router(self):
        """
        Broadcast the engine status to the router.
        """
        assert self.rollout_router is not None, "Rollout router is not set in RolloutCoordinator"

        while not self.stop_broadcast_status_to_router:
            # Broadcast the engine status to the router every coordinator sync interval
            await asyncio.sleep(self.config.psrl.status_collection.coordinator_sync_interval_in_ms / 1000)
            await self.rollout_router.update_instance_status.remote(self.instance_to_engine_status)

    async def _is_routing(self) -> bool:
        """Check if any agent loop worker is currently routing requests
        (i.e., the router is currently routing requests).

        Returns:
            bool: True if any agent loop worker is currently routing requests,
                False otherwise.
        """
        is_routing = await self.rollout_router.is_routing.remote()
        return is_routing

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
            for instance_id in range(self.rollout_wg_size):
                # Check whether engine status is stale (the instance is currently being synchronized with PS)
                if self.instance_to_model_version.get(
                    instance_id, 0
                ) <= self.instance_to_latest_stale_model_version.get(instance_id, -1):
                    have_syncing_instance = True
                    continue
                # Check whether instance version lags behind PS version
                if self.instance_to_model_version.get(instance_id, 0) == self.ps_model_version:
                    continue
                # Check whether current instance workload is empty if forbid partial rollout
                if not self.config.psrl.partial_rollout.enable:
                    if not await self.check_no_activate_tasks(instance_id):
                        continue
                # Check whether the training side can seamlessly continue to train after the synchronization
                if self.config.psrl.sync_and_mig_strategy.sync.seamless_train_version >= self.ps_model_version:
                    if self.ps_model_version not in self.ready_buffers:
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
                # Ignore validate instances for weight synchronization
                if instance_id >= self.rollout_wg_size:
                    continue
                # Check whether engine status is stale (the instance is currently being synchronized with PS)
                if engine_stats.model_version <= self.instance_to_latest_stale_model_version.get(instance_id, -1):
                    have_syncing_instance = True
                    continue
                # We do not synchronize with PS if the router is currently routing requests
                if await self._is_routing():
                    # psrl_logger.info(f"Skipping synchronization with PS for instance
                    # {instance_id} because the router is currently routing requests")
                    continue
                # Check whether instance version lags behind PS version
                if self.instance_to_model_version.get(instance_id, 0) == self.ps_model_version:
                    continue
                # Check whether current instance workload is empty (forbid partial rollout)
                # or satisfies the partial rollout policy
                if self.config.psrl.partial_rollout.enable:
                    if not (await self.check_should_sync(instance_id)):
                        continue
                else:
                    if engine_stats.get_waiting_and_running_queue_size() > 0:
                        continue
                # Check whether the training side can seamlessly continue to train after the synchronization
                if self.config.psrl.sync_and_mig_strategy.sync.seamless_train_version >= self.ps_model_version:
                    if self.ps_model_version not in self.ready_buffers:
                        continue
                # Add the instance to the sync list
                sync_instance_ids.append(instance_id)
                '''
                # NOTE(lhy): currently, we only synchronize with PS for one instance at a time
                # But the model pulling time can be overlapped
                break
                '''

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
        assert self.ps_model_version > 0, "PS model version must be greater than 0"
        assert (self.ps_model_version - 1) in self.ready_buffers, "PS model version must be greater than the ready buffers"
        self.ready_buffers.remove(self.ps_model_version - 1)
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
        psrl_logger.info(
            f"Updated rollout instance {rollout_instance_id} model version: {old_version} -> {version_tag}"
        )
        
    def update_ready_buffer(self, ready_buffer: int):
        """
        Update the ready buffer.
        """
        self.ready_buffers.add(ready_buffer)
        psrl_logger.info(
            f"Updated ready buffers to: {self.ready_buffers}"
        )

    async def sync_with_ps(
        self,
        instance_ids: list[int],
        wait_model_sync: bool = False,
        wait_interrupted_partial_requests_loop_back: bool = True,
    ):
        """
        Synchronize with PS for the given instance IDs.
        """
        # Add batching SYNC command to the command queue to interrupt the instance
        # This will stop the instance, pull the model weights from PS, and resume generation.
        # But this will not block the current loop.
        # NOTE(lhy): we don't need to update the instance version here because the version
        # is updated in the `sync_with_ps` method of the GenWorker
        # when calling `pull_model` or `pull_model_async` from the GenWorker, the ps manager
        # will update the instance version.
        # However, we need to update the latest stale model version here to avoid stale stats
        # being handled after the synchronization.
        with log_dual_events(
            f"Synchronize rollout instances {instance_ids} with PS "
            f"(model pull is {'non-blocking' if not wait_model_sync else 'blocking'} "
            f"for the coordinator)",
            psrl_logger,
            level=logging.INFO,
            event_type=EventType.OTHER,
        ):
            for instance_id in instance_ids:
                self.instance_to_latest_stale_model_version[instance_id] = self.instance_to_model_version.get(
                    instance_id, 0
                )
            await self.rollout_router.pause_routing.remote()
            psrl_logger.info("Paused routing for synchronization")
            await self.rollout_router.update_currently_syncing_instances.remote(instance_ids, self.ps_model_version)
            psrl_logger.info("Updated currently syncing instances")
            await self.exec_command(
                Command(
                    type=CommandType.SYNC,
                    instance_ids=instance_ids,
                    curr_ps_model_version=self.ps_model_version,
                    wait_model_sync=wait_model_sync,
                ),
                blocking=True,
            )
            psrl_logger.info("Executed SYNC command")
            if wait_interrupted_partial_requests_loop_back and self.config.psrl.partial_rollout.enable:
                psrl_logger.info("Waiting for interrupted partial requests loop back")
                await self.rollout_router.wait_interrupted_partial_requests_loop_back.remote(instance_ids)
                psrl_logger.info(
                    f"All interrupted requests on the synchronized instances {instance_ids} have been looped back"
                )
            await self.rollout_router.resume_routing.remote()
            psrl_logger.info("Resumed routing after synchronization")

    async def check_no_activate_tasks(self, instance_id: int) -> bool:
        """
        Check whether the instance has no active tasks.
        """
        active_task_num = await self.gen_wg_list[instance_id].execute_rank_zero_async("get_active_task_num")
        return active_task_num == 0

    async def check_should_sync(self, instance_id: int) -> bool:
        """
        Check whether to synchronize with PS for the instance.
        """
        assert self.config.psrl.sync_and_mig_strategy.method == "status_based", (
            "Partial rollout is only supported for status-based sync strategy"
        )
        assert self.config.psrl.status_collection.enable, (
            "Partial rollout is only supported when status collection is enabled"
        )
        assert self.config.psrl.partial_rollout.enable, "Partial rollout is not enabled"
        # TODO(lhy): refactor the router to be a global router
        # psrl_logger.info(
        #     f"Checking whether to synchronize with PS for instance {instance_id}, "
        #     f"ps model version: {self.ps_model_version}"
        # )
        return await self.rollout_router.check_should_sync.remote(instance_id, self.ps_model_version)

    async def check_and_migrate(self, wait_interrupted_partial_requests_loop_back: bool = True):
        """
        Check if any instance is starving and do migration if necessary.
        """
        assert self.config.psrl.sync_and_mig_strategy.mig.enable, "Rollout migration is not enabled"
        assert self.config.psrl.status_collection.enable, (
            "Rollout migration is only supported when status collection is enabled"
        )
        # psrl_logger.info("Checking if any instance is starving and doing migration if necessary")
        migrate_instance_ids = await self.rollout_router.check_should_migrate.remote()
        if migrate_instance_ids:
            with log_dual_events(
                f"Migrating instances {migrate_instance_ids}",
                psrl_logger,
                level=logging.INFO,
                event_type=EventType.OTHER,
            ):
                await self.rollout_router.pause_routing.remote()
                psrl_logger.info("Interrupted routing for migration")
                await self.exec_command(
                    Command(
                        type=CommandType.ABORT,
                        instance_ids=migrate_instance_ids,
                    ),
                    blocking=True,
                )
                if wait_interrupted_partial_requests_loop_back:
                    await self.rollout_router.wait_interrupted_partial_requests_loop_back.remote(
                        migrate_instance_ids
                    )
                    psrl_logger.info(
                        f"All interrupted requests on the migrated instances {migrate_instance_ids} have been looped back"
                    )
                await self.rollout_router.resume_routing.remote()
                psrl_logger.info("Resumed routing after migration")
