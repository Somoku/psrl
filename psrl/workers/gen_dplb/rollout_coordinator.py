import asyncio
import json
import logging
import os
import time

import aiohttp
import numpy as np
import ray

from psrl.utils.common.http_utils import find_available_port
from psrl.utils.elastic_rm.diagnostics import log_elastic_rm_backlog_diag
from psrl.utils.logger import (
    DualOutputHandler,
    EventType,
    log_dual_events,
)
from psrl.utils.server.command import Command, CommandExtension, CommandType
from psrl.workers.gen_dplb.stats_collector import EngineStats
from psrl.workers.gen_dplb.utils import DEFAULT_MAX_CONNECTIONS, DEFAULT_TIMEOUT, RolloutInstanceId
from psrl.workers.gen_dplb.zmq_queue import ZMQPullQueue

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


STATUS_QUEUE_POLL_TIMEOUT_SECS = 0.2


class RolloutCoordinator(CommandExtension):
    def __init__(
        self,
        config,
        ps_manager: ray.actor.ActorHandle,
        rollout_router: ray.actor.ActorHandle | str,
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
        """
        super().__init__()

        self.config = config
        self.staleness = self.config.psrl.staleness
        self.ps_manager = ps_manager

        # Rollout replica tracking
        self.rollout_replicas = {}
        self.server_handles = {}
        self.replica_ids = set()
        self.instance_ids = set()
        # For convenience, maintain separate lists for rollout and validate instances
        self.tag_to_replica_ids = {"rollout": set(), "validate": set()}

        self.rollout_router = rollout_router
        self.use_rust_gateway = isinstance(rollout_router, str)
        self.gateway_client: aiohttp.ClientSession | None = None
        self.gateway_addr: str | None = None
        self.gateway_base_url: str | None = None
        if self.use_rust_gateway:
            self.gateway_addr = rollout_router
            self.gateway_base_url = self.gateway_addr
            connector = aiohttp.TCPConnector(
                limit=DEFAULT_MAX_CONNECTIONS,
                limit_per_host=DEFAULT_MAX_CONNECTIONS,
                ttl_dns_cache=300,
                enable_cleanup_closed=True,
            )
            timeout = aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)
            self.gateway_client = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
            )
        else:
            self.gateway_addr = None
            self.gateway_base_url = None

        # Stats collection
        status_host = ray.util.get_node_ip_address().strip("[]")
        status_port = find_available_port(base_port=28000)
        self.status_sink_endpoint = f"tcp://{status_host}:{status_port}"
        self.status_queue = ZMQPullQueue(endpoint=self.status_sink_endpoint)
        self.replica_idx_to_replica_id: dict[int, str] = {}

        # Background event handler
        self.running_loop = None
        self.command_handler_task = None
        self.sync_task = None
        self.process_status_queue_task = None
        self.sync_status_to_router_task = None
        self.stats_recorder_task = None
        self.stop_command_handler = False
        self.stop_sync_and_migrate = False
        self.stop_process_status_queue = False
        self.stop_sync_status_to_router = False
        self.stop_stats_recorder = False

        # Asyncio event loop order control
        self._is_init_nixl_client = asyncio.Event()

        # Version tracking
        # The latest stale model version of each instance
        self.instance_to_latest_stale_model_version: dict[RolloutInstanceId, int] = {}
        # Track the model version of each instance
        self.instance_to_model_version: dict[RolloutInstanceId, int] = {}
        # Tracks model version each instance will have after its next sync.
        self.instance_to_version_after_sync: dict[RolloutInstanceId, int] = {}

        self.ps_model_version = 0  # Current model version in the parameter server
        self.ready_buffers = set()  # The set of ready buffers

        # Engine status tracking
        # Track the latest engine stats of each instance
        self.instance_to_engine_status: dict[RolloutInstanceId, EngineStats] = {}

        # Build logger
        self.log_prefix = "RolloutCoordinator"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))
        psrl_logger.info("Initialized RolloutCoordinator")

        # Stats recorder (opt-in)
        self._stats_recorder = None
        if self.config.psrl.status_collection.stats_recorder.enable:
            from psrl.workers.gen_dplb.stats_recorder import StatsRecorder

            self._stats_recorder = StatsRecorder(
                self.config.psrl,
                os.path.expanduser(self.config.psrl.logging_path),
            )
            self._stats_recorder.write_config(
                routing_strategy=self.config.psrl.routing_strategy.method,
                partial_rollout=self.config.psrl.partial_rollout.enable,
            )

    def add_worker(
        self,
        rollout_replica,
        server_handle,
        replica_id: str,
        dp_size: int,
        is_validate: bool = False,
        model_version: int = 0,
    ):
        """Add a rollout replica to the coordinator.

        Args:
            rollout_replica: Rollout replica object
            server_handle: Handle to the rollout replica actor
            replica_id (str): ID of the rollout replica
            instance_num (int): Number of instances in the rollout replica
        """
        self.rollout_replicas[replica_id] = rollout_replica
        self.server_handles[replica_id] = server_handle
        self.replica_ids.add(replica_id)
        self.instance_ids.update([(replica_id, i) for i in range(dp_size)])
        self.replica_idx_to_replica_id[rollout_replica.replica_rank] = replica_id

        tag = "validate" if is_validate else "rollout"
        self.tag_to_replica_ids[tag].add(replica_id)

        # Initialize version_after_sync for newly registered instances
        for i in range(dp_size):
            instance_id = (replica_id, i)
            self.instance_to_model_version[instance_id] = model_version
            self.instance_to_version_after_sync[instance_id] = model_version
            self.instance_to_engine_status[instance_id] = EngineStats(
                replica_idx=rollout_replica.replica_rank,
                data_parallel_rank=i,
                model_version=model_version,
                snapshot=EngineStats.get_default_snapshot(),
            )

    def get_status_sink_endpoint(self) -> str:
        return self.status_sink_endpoint

    def get_all_instance_ids(self) -> list:
        """Return all registered (replica_id, dp_rank) instance IDs, sorted for determinism."""
        return sorted(self.instance_ids)

    def _get_sleep_level(self) -> int:
        """Sleep level for server.sleep(). Rollout uses level=2 (full GPU release). Subclasses may override."""
        return 2

    async def _do_sleep_instance(self, replica_id: str, data_parallel_rank: int) -> None:
        """Execute sleep on one (replica_id, dp_rank). Rollout path uses nixl_sleep. Subclasses may override."""
        await self.server_handles[replica_id].nixl_sleep.remote(
            level=self._get_sleep_level(),
            data_parallel_rank=data_parallel_rank,
        )

    async def _do_wake_up_instance(self, replica_id: str, data_parallel_rank: int) -> None:
        """Execute wake_up on one (replica_id, dp_rank). Rollout path uses nixl_wake_up. Subclasses may override."""
        await self.server_handles[replica_id].nixl_wake_up.remote(
            data_parallel_rank=data_parallel_rank,
        )

    async def _gateway_post_json(self, path: str, payload, params: dict | None = None):
        if self.gateway_base_url is None:
            raise RuntimeError("Rust gateway base url is not initialized")
        url = f"{self.gateway_base_url}{path}"
        async with self.gateway_client.post(url, json=payload, params=params) as resp:
            resp.raise_for_status()
            text = await resp.text()
            if not text.strip():
                return {}
            return json.loads(text)

    async def _gateway_get_json(self, path: str, params: dict | None = None):
        if self.gateway_base_url is None:
            raise RuntimeError("Rust gateway base url is not initialized")
        url = f"{self.gateway_base_url}{path}"
        psrl_logger.debug(f"Making GET request to {url} with params {params}")
        async with self.gateway_client.get(url, params=params) as resp:
            resp.raise_for_status()
            text = await resp.text()
            if not text.strip():
                return {}
            return json.loads(text)

    async def _set_routing_loop_running(self, running: bool):
        path = "/routing_loop/resume" if running else "/routing_loop/pause"
        # When pausing, pass wait=true so the call only returns after the routing loop
        # finishes its current dispatch round (is_routing becomes false). This prevents
        # a race where update_currently_syncing_instances and the SYNC command are issued
        # while the loop is still assigning requests using the pre-sync version.
        params = {"wait": "true"} if not running else {}
        psrl_logger.info(f"Setting routing loop running state to {running} via {path}")
        data = await self._gateway_post_json(path, payload={}, params=params)
        expected = running
        actual = data.get("running")
        if actual is not None and bool(actual) != expected:
            raise RuntimeError(f"Unexpected routing loop state after {path}: expected={expected}, got={actual}")

    async def _fetch_filtered_request_meta(self, version_tag: int) -> list[tuple[int, bool]]:
        data = await self._gateway_get_json("/routing_loop/filter", params={"version_tag": version_tag})
        requests = data.get("requests", [])
        filtered_request_meta: list[tuple[int, bool]] = []
        for req in requests:
            request_id = int(req.get("request_id"))
            if request_id is None:
                continue
            is_validate = bool(req.get("is_validate", False))
            filtered_request_meta.append((request_id, is_validate))
        return filtered_request_meta

    def world_size(self):
        """Get the total world size (number of rollout and validate instances)."""
        return sum([rollout_replica.world_size for rollout_replica in self.rollout_replicas.values()])

    def _tag_to_server(self, tag: str):
        """Get the rollout server handles and number for a given tag.

        Args:
            tag (str): Tag to specify which instances to get ('rollout', 'validate', 'all')
        Returns:
            list: List of worker group handles
        """
        if tag in ["rollout", "validate"]:
            replica_ids = self.tag_to_replica_ids[tag]
        elif tag == "all":
            replica_ids = self.replica_ids
        else:
            raise ValueError(f"Unknown tag {tag} for getting server handles and number")
        server_handles = [self.server_handles[replica_id] for replica_id in replica_ids]
        return server_handles

    async def init_nixl_client(self):
        """Init the NIXL client on rollout and validate instances."""
        futures = []
        for server_handle in self.server_handles.values():
            futures.append(server_handle.init_nixl_client.remote())
        await asyncio.gather(*futures)
        psrl_logger.info(f"Initialized NIXL client on all {len(self.server_handles)} replicas.")
        self._is_init_nixl_client.set()

    async def nixl_protocol(self, full_tag: str = "all"):
        """Run the NIXL server protocol on rollout and validate instances.

        Args:
            full_tag (str): Tag to specify which instances to run the protocol
                            in 'full' mode ('rollout', 'validate', 'all')
        """
        await self._is_init_nixl_client.wait()

        if full_tag == "all":
            rollout_tag = "full"
            validate_tag = "full"
        elif full_tag == "rollout":
            rollout_tag = "full"
            validate_tag = "meta"
        elif full_tag == "validate":
            rollout_tag = "meta"
            validate_tag = "full"
        else:
            raise ValueError(f"Unknown full_tag {full_tag} for nixl_protocol")

        futures = []
        for replica_id in self.tag_to_replica_ids["rollout"]:
            futures.append(self.server_handles[replica_id].nixl_protocol.remote(rollout_tag))
        for replica_id in self.tag_to_replica_ids["validate"]:
            futures.append(self.server_handles[replica_id].nixl_protocol.remote(validate_tag))
        await asyncio.gather(*futures)

    async def nixl_convert_params(self):
        """Convert the model parameters to unified format on rollout and validate instances."""
        await self._is_init_nixl_client.wait()
        futures = []
        for server_handle in self.server_handles.values():
            futures.append(server_handle.nixl_convert_params.remote())
        await asyncio.gather(*futures)

    async def initial_pull_from_ps(self, tag: str = "rollout") -> None:
        futures = [server_handle.pull_model.remote() for server_handle in self._tag_to_server(tag)]
        await asyncio.gather(*futures)
        psrl_logger.info(f"Initial PS pull complete for {len(futures)} replicas with tag {tag}.")

    async def sleep(self, tag: str = "all"):
        """Make rollout instances sleep and release GPU memory.

        Args:
            tag (str): Tag to specify which instances to sleep ('rollout', 'validate', 'all')
        """
        server_handles = self._tag_to_server(tag)
        futures = []
        for server_handle in server_handles:
            futures.append(server_handle.sleep.remote(level=2))
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
        if self.command_handler_task is not None and not self.command_handler_task.done():
            return

        # Start the background tasks
        self.running_loop = asyncio.get_running_loop()
        self.command_handler_task = self.running_loop.create_task(self._command_handler_loop())
        self.command_handler_task.add_done_callback(lambda f: f.result())  # To avoid silent error in async tasks

        # Start the status collection tasks
        if self.config.psrl.status_collection.enable:
            self.process_status_queue_task = self.running_loop.create_task(self._process_status_queue())
            self.process_status_queue_task.add_done_callback(lambda f: f.result())
        # Start the task to broadcast the engine status to the router
        self.sync_status_to_router_task = self.running_loop.create_task(self._sync_status_to_router())
        self.sync_status_to_router_task.add_done_callback(lambda f: f.result())  # To avoid silent error in async tasks
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

        # Start the stats recorder loop (opt-in)
        if self.config.psrl.status_collection.stats_recorder.enable:
            self.stats_recorder_task = self.running_loop.create_task(self._stats_recorder_loop())
            self.stats_recorder_task.add_done_callback(lambda f: f.result())

    async def stop_busy_loop(self):
        """
        Stop all background tasks and clean up resources.

        This method gracefully shuts down:
        - Command handler task
        - Engine status sync task
        - Stats recorder task (if enabled)
        """
        if self.command_handler_task is None or self.command_handler_task.done():
            return

        # Stop the background tasks
        self.stop_command_handler = True
        self.stop_sync_and_migrate = True
        self.stop_sync_status_to_router = True
        self.stop_process_status_queue = True
        self.stop_stats_recorder = True

        psrl_logger.info("Before waiting for all background tasks")
        await self.command_handler_task
        if self.process_status_queue_task is not None:
            await self.process_status_queue_task
            psrl_logger.info("Finished process status queue task.")
        if self.sync_status_to_router_task is not None:
            await self.sync_status_to_router_task
            psrl_logger.info("Finished syncing status to router.")
        if self.sync_task is not None:
            await self.sync_task
            psrl_logger.info("Finished sync task.")
        if self.stats_recorder_task is not None:
            await self.stats_recorder_task
            psrl_logger.info("Finished stats recorder task.")
        if self._stats_recorder is not None:
            self._stats_recorder.close()
        psrl_logger.info("All background tasks have been stopped.")
        self.status_queue.close()
        if self.gateway_client is not None and not self.gateway_client.closed:
            await self.gateway_client.close()
        psrl_logger.info("Cleaned up resources in RolloutCoordinator.")

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
                psrl_logger.debug(
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
                        replica_to_uids = {}
                        for instance_id, uids in instance_to_uids.items():
                            if not uids:
                                continue
                            replica_id = instance_id[0]
                            if not isinstance(uids, (list, set)):
                                uids = [uids]
                            abort_requests = set(uids)  # Ensure uniqueness
                            replica_to_uids.setdefault(replica_id, set()).update(abort_requests)
                        for replica_id, abort_requests in replica_to_uids.items():
                            abort_requests = [str(req_id) for req_id in abort_requests]
                            futures.append(self.server_handles[replica_id].abort_requests.remote(abort_requests))
                    if instance_ids is not None:
                        for instance_id in instance_ids:
                            # TODO(linsh): merge dp ranks of the same replica to reduce RPC calls
                            replica_id, data_parallel_rank = instance_id
                            futures.append(
                                self.server_handles[replica_id].abort_all_requests.remote(
                                    data_parallel_rank=data_parallel_rank
                                )
                            )

                    if not futures:
                        interrupted_request_num = 0
                    else:
                        interrupted_request_status = await asyncio.gather(*futures)
                        interrupted_request_nums = [
                            status if isinstance(status, int) else status.get("aborted_count", 0)
                            for status in interrupted_request_status
                        ]
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

                    # Skip instances that are sleeping
                    is_sleeping = await asyncio.gather(
                        *[
                            self.server_handles[instance_id[0]].is_sleeping.remote(data_parallel_rank=instance_id[1])
                            for instance_id in instance_ids
                        ]
                    )
                    sync_instance_ids = [
                        instance_id for instance_id, sleeping in zip(instance_ids, is_sleeping) if not sleeping
                    ]
                    sleeping_instance_ids = [
                        instance_id for instance_id, sleeping in zip(instance_ids, is_sleeping) if sleeping
                    ]
                    if sleeping_instance_ids:
                        psrl_logger.info(f"Skipping SYNC for sleeping instances: {sleeping_instance_ids}")

                    if not sync_instance_ids:
                        self._complete_command(command_id, None)
                    else:
                        # Sync with PS (interrupt, pull model, and resume generation)
                        sync_futures = []

                        for instance_id in instance_ids:
                            replica_id, data_parallel_rank = instance_id
                            sync_future = self.server_handles[replica_id].sync_with_ps.remote(
                                curr_ps_model_version,
                                pause_generation=True,
                                data_parallel_rank=data_parallel_rank,
                            )
                            sync_futures.append(sync_future)

                        # Post process the command result
                        if wait_model_sync:
                            await asyncio.gather(*sync_futures)
                            self._complete_command(command_id, None)
                        else:
                            # NOTE(linsh): sometimes it's not necessary for the caller to wait for pulling from PS
                            self._complete_command(command_id, None)
                            await asyncio.gather(*sync_futures)  # Wait for the sync to complete
                elif command_type == CommandType.SLEEP:
                    instance_ids = command_args.get("instance_ids", None)
                    if not isinstance(instance_ids, list):
                        instance_ids = [instance_ids]
                    if instance_ids is None:
                        raise ValueError("SLEEP command must contain 'instance_ids' in args.")

                    # Pause routing to prevent new requests from being dispatched
                    # to the instances that are going to sleep
                    if self.use_rust_gateway:
                        await self._set_routing_loop_running(False)
                    else:
                        await self.rollout_router.pause_routing.remote()

                    sleep_futures = []
                    for instance_id in instance_ids:
                        replica_id, data_parallel_rank = instance_id
                        sleep_futures.append(self._do_sleep_instance(replica_id, data_parallel_rank))
                    await asyncio.gather(*sleep_futures)
                    self._complete_command(command_id, None)
                elif command_type == CommandType.WAKE_UP:
                    instance_ids = command_args.get("instance_ids", None)
                    if not isinstance(instance_ids, list):
                        instance_ids = [instance_ids]
                    if instance_ids is None:
                        raise ValueError("WAKE_UP command must contain 'instance_ids' in args.")
                    psrl_logger.info(f"Received WAKE_UP command for instances {instance_ids}")

                    wake_up_futures = []
                    for instance_id in instance_ids:
                        replica_id, data_parallel_rank = instance_id
                        wake_up_futures.append(self._do_wake_up_instance(replica_id, data_parallel_rank))
                    await asyncio.gather(*wake_up_futures)

                    # Resume routing after the instances have woken up
                    if self.use_rust_gateway:
                        await self._set_routing_loop_running(True)
                    else:
                        await self.rollout_router.resume_routing.remote()

                    self._complete_command(command_id, None)
                else:
                    raise ValueError(f"Unknown command type: {command_type}")

            await asyncio.sleep(0)

        psrl_logger.info("Background command handler of rollout coordinator has finished.")

    async def _process_status_queue(self):
        psrl_logger.info("Starting to process ZMQ status stream")
        while not self.stop_process_status_queue:
            # Naive implementation version
            """
            recv_stats = await self.status_queue.get_async(block=True, timeout=None)
            replica_id = self.replica_idx_to_replica_id.get(recv_stats.replica_idx)
            if replica_id is None:
                raise RuntimeError(f"Received engine stats with unknown replica_idx {recv_stats.replica_idx}")
            instance_id = (replica_id, recv_stats.data_parallel_rank)
            self.instance_to_engine_status[instance_id] = recv_stats
            psrl_logger.debug(
                f"Updated engine status for instance "
                f"{instance_id}: {self.instance_to_engine_status[instance_id]} "
            )
            """

            # Coalesce implementation version
            try:
                recv_stats = await self.status_queue.get_async(
                    block=True,
                    timeout=STATUS_QUEUE_POLL_TIMEOUT_SECS,
                )
            except TimeoutError:
                # Periodically wake up so shutdown flag can be observed even when no messages arrive.
                continue
            latest_by_instance: dict[RolloutInstanceId, EngineStats] = {}

            # Put first blocking message
            replica_id = self.replica_idx_to_replica_id.get(recv_stats.replica_idx)
            if replica_id is None:
                raise RuntimeError(f"Received engine stats with unknown replica_idx {recv_stats.replica_idx}")
            instance_id = (replica_id, recv_stats.data_parallel_rank)
            latest_by_instance[instance_id] = recv_stats

            # Drain all currently queued messages and coalesce by instance
            while True:
                next_stats = await self.status_queue.get_async(block=False)
                if next_stats is None:
                    break
                replica_id = self.replica_idx_to_replica_id.get(next_stats.replica_idx)
                if replica_id is None:
                    raise RuntimeError(f"Received engine stats with unknown replica_idx {next_stats.replica_idx}")
                instance_id = (replica_id, next_stats.data_parallel_rank)
                latest_by_instance[instance_id] = next_stats

            # Apply only the newest snapshot per instance
            # psrl_logger.info(f"Received {len(latest_by_instance)} engine stats updates, updating instance_to_engine_status with {latest_by_instance}...")  # noqa: E501
            self.instance_to_engine_status.update(latest_by_instance)
        psrl_logger.info("Stopped processing ZMQ status stream.")

    def get_instance_engine_status_snapshot(self) -> dict[RolloutInstanceId, dict]:
        """
        Return a lightweight snapshot map for elastic scaling decisions.
        """
        snapshot = {}
        for instance_id, engine_status in self.instance_to_engine_status.items():
            snapshot[instance_id] = engine_status.snapshot
        return snapshot

    async def get_router_backlog_size(self) -> int:
        """Return pending request count in rollout router queue."""
        t_enter = time.monotonic()
        model_tag = str(self.config.gen_actor_rollout_ref.model.path).rstrip("/").split("/")[-1]
        log_elastic_rm_backlog_diag(
            psrl_logger,
            "stage=RolloutCoordinator_enter model=%s elapsed_since_entry_s=0.000",
            model_tag,
        )
        if self.rollout_router is None:
            return 0
        log_elastic_rm_backlog_diag(
            psrl_logger,
            "stage=RolloutCoordinator_before_router_rpc model=%s since_enter_s=%.3f",
            model_tag,
            time.monotonic() - t_enter,
        )
        t_rpc = time.monotonic()
        if self.use_rust_gateway:
            # TODO: add `pending_request_num` to the `/routing_loop/status` endpoint in Rust gateway
            pending = int(self._gateway_get_json("/routing_loop/status").get("pending_request_num", 0))
            pass
        else:
            pending = int(await self.rollout_router.get_pending_request_count.remote())
        log_elastic_rm_backlog_diag(
            psrl_logger,
            "stage=RolloutCoordinator_after_router_rpc model=%s pending=%d router_rpc_s=%.3f since_enter_s=%.3f",
            model_tag,
            pending,
            time.monotonic() - t_rpc,
            time.monotonic() - t_enter,
        )
        return pending

    async def _sync_status_to_router(self):
        """
        Broadcast the engine status to the router.
        """
        assert self.rollout_router is not None, "Rollout router is not set in RolloutCoordinator"

        while not self.stop_sync_status_to_router:
            # Broadcast the engine status to the router every coordinator sync interval
            await asyncio.sleep(self.config.psrl.status_collection.coordinator_sync_interval_in_ms / 1000)
            if self.use_rust_gateway:
                if not self.instance_to_engine_status:
                    continue
                updates = []
                for instance_id, engine_status in self.instance_to_engine_status.items():
                    replica_id, dp_rank = instance_id
                    updates.append(
                        {
                            "base_worker_id": replica_id,
                            "dp_rank": dp_rank,
                            "snapshot": {
                                "timestamp": engine_status.snapshot.get("timestamp"),
                                "scheduler_stats": engine_status.snapshot.get("scheduler_stats", {}),
                            },
                        }
                    )
                await self._gateway_post_json("/workers/stats", payload={"updates": updates})
            else:
                await self.rollout_router.update_instance_status.remote(self.instance_to_engine_status)
        psrl_logger.info("Stopped syncing engine status to router.")

    async def _stats_recorder_loop(self):
        """Periodically snapshot per-replica stats to JSONL files."""
        psrl_logger.info("Starting stats recorder loop")
        interval = self.config.psrl.status_collection.stats_recorder.interval_in_s
        while not self.stop_stats_recorder:
            await asyncio.sleep(interval)
            if not self.stop_stats_recorder:
                self._stats_recorder.record(self.instance_to_engine_status)
        psrl_logger.info("Stopped stats recorder loop.")

    async def _is_routing(self) -> bool:
        """Check if any agent loop worker is currently routing requests
        (i.e., the router is currently routing requests).

        Returns:
            bool: True if any agent loop worker is currently routing requests,
                False otherwise.
        """
        if self.use_rust_gateway:
            data = await self._gateway_get_json("/routing_loop/status")
            is_routing = bool(data.get("is_routing", False))
        else:
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
            for instance_id in self.instance_ids:
                # Ignore validate instances for weight synchronization
                replica_id, _ = instance_id
                if replica_id not in self.tag_to_replica_ids["rollout"]:
                    continue
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
                psrl_logger.info(f"Sync with ps: {sync_instance_ids}")
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
                replica_id, _ = instance_id
                if replica_id not in self.tag_to_replica_ids["rollout"]:
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
                """
                # NOTE(lhy): currently, we only synchronize with PS for one instance at a time
                # But the model pulling time can be overlapped
                break
                """

            if sync_instance_ids:
                psrl_logger.info(f"Sync with ps {sync_instance_ids}")
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
        assert (self.ps_model_version - 1) in self.ready_buffers, (
            "PS model version must be greater than the ready buffers"
        )
        self.ready_buffers.remove(self.ps_model_version - 1)
        psrl_logger.info(f"Updated PS model version to {version}")

    # This is called by the PS manager to update the rollout instance model version after pulling
    def set_rollout_instance_model_version(self, rollout_instance_id: RolloutInstanceId, version_tag: int):
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
        psrl_logger.info(f"Updated ready buffers to: {self.ready_buffers}")

    async def sync_with_ps(
        self,
        instance_ids: list[RolloutInstanceId],
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

            if self.use_rust_gateway:
                await self._set_routing_loop_running(False)
            else:
                await self.rollout_router.pause_routing.remote()
            psrl_logger.info("Paused routing for synchronization")

            if self.use_rust_gateway:
                for instance_id in instance_ids:
                    self.instance_to_version_after_sync[instance_id] = self.ps_model_version
                updates = []
                for instance_id in instance_ids:
                    replica_id, dp_rank = instance_id
                    updates.append(
                        {
                            "base_worker_id": replica_id,
                            "dp_rank": dp_rank,
                            "version_tag": self.ps_model_version,
                        }
                    )
                await self._gateway_post_json("/workers/version_tag", payload={"updates": updates})
                psrl_logger.info(
                    f"Pushed version_after_sync to Rust gateway for "
                    f"{len(instance_ids)} instances to {self.ps_model_version}"
                )
            else:
                # Original path: update via Router Ray RPC
                await self.rollout_router.update_currently_syncing_instances.remote(
                    instance_ids, self.ps_model_version
                )

            await self.exec_command(
                Command(
                    type=CommandType.SYNC,
                    instance_ids=instance_ids,
                    curr_ps_model_version=self.ps_model_version,
                    wait_model_sync=wait_model_sync,
                ),
                blocking=True,
            )
            if wait_interrupted_partial_requests_loop_back and self.config.psrl.partial_rollout.enable:
                await self._wait_interrupted_partial_requests_loop_back(instance_ids)
                psrl_logger.info(
                    f"All interrupted requests on the synchronized instances {instance_ids} have been looped back"
                )

            if self.use_rust_gateway:
                await self._set_routing_loop_running(True)
            else:
                await self.rollout_router.resume_routing.remote()
            psrl_logger.info("Resumed routing after synchronization")

    async def check_no_activate_tasks(self, instance_id: RolloutInstanceId) -> bool:
        """
        Check whether the instance has no active tasks.
        """
        replica_id, data_parallel_rank = instance_id
        active_task_num = await self.server_handles[replica_id].get_active_task_num.remote(data_parallel_rank)
        return active_task_num == 0

    async def check_should_sync(self, instance_id: RolloutInstanceId) -> bool:
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

        if self.use_rust_gateway:
            return await self._check_should_sync(instance_id)
        else:
            # Fallback: delegate to Router via Ray RPC (original path)
            return await self.rollout_router.check_should_sync.remote(instance_id)

    async def _check_should_sync(self, instance_id: RolloutInstanceId) -> bool:
        instance_status = self.instance_to_engine_status[instance_id]
        if instance_status.get_waiting_queue_size() > 0:
            return False

        # 1. Check if there are any requests version satisfies the condition before synchronization
        current_instance_version = self.instance_to_version_after_sync[instance_id]
        if await self.ps_manager.check_aborted_model_versions.remote(current_instance_version):
            filtered_request_ids = []
        else:
            filtered_request_meta = await self._fetch_filtered_request_meta(current_instance_version)
            filtered_request_ids = [request_meta[0] for request_meta in filtered_request_meta]

        # 2. Check if there are any requests
        # that can be RESERVED for the instance but no need to reserve new entry
        # before synchronization
        if len(filtered_request_ids) > 0:
            is_aborted = await self.ps_manager.check_aborted_requests.remote(filtered_request_ids, remove=False)
            filtered_request_ids = [
                request_id for i, request_id in enumerate(filtered_request_ids) if not is_aborted[i]
            ]
            can_reserve_without_new_reserve_entry = await self.ps_manager.can_reserve_request.remote(
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
            throughput = self.instance_to_engine_status[instance_id].get_generation_throughput()
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
        else:
            raise ValueError(f"Unknown sync indicator: {self.config.psrl.sync_and_mig_strategy.sync.indicator}")
        return True

    async def check_and_migrate(self, wait_interrupted_partial_requests_loop_back: bool = True):
        """
        Check if any instance is starving and do migration if necessary.
        """
        assert self.config.psrl.sync_and_mig_strategy.mig.enable, "Rollout migration is not enabled"
        assert self.config.psrl.status_collection.enable, (
            "Rollout migration is only supported when status collection is enabled"
        )

        if self.use_rust_gateway:
            migrate_instance_ids = await self._check_should_migrate()
        else:
            # Fallback: delegate to Router via Ray RPC (original path)
            migrate_instance_ids = await self.rollout_router.check_should_migrate.remote()

        if migrate_instance_ids:
            with log_dual_events(
                f"Migrating instances {migrate_instance_ids}",
                psrl_logger,
                level=logging.INFO,
                event_type=EventType.OTHER,
            ):
                if self.use_rust_gateway:
                    await self._set_routing_loop_running(False)
                else:
                    await self.rollout_router.pause_routing.remote()
                psrl_logger.info("Paused routing for migration")
                await self.exec_command(
                    Command(
                        type=CommandType.ABORT,
                        instance_ids=migrate_instance_ids,
                    ),
                    blocking=True,
                )
                if wait_interrupted_partial_requests_loop_back:
                    await self._wait_interrupted_partial_requests_loop_back(migrate_instance_ids)
                    psrl_logger.info(
                        f"All interrupted requests on the migrated instances "
                        f"{migrate_instance_ids} have been looped back"
                    )
                if self.use_rust_gateway:
                    await self._set_routing_loop_running(True)
                else:
                    await self.rollout_router.resume_routing.remote()
                psrl_logger.info("Resumed routing after migration")

    async def _check_should_migrate(self) -> list[RolloutInstanceId]:
        filtered_instance_ids = []
        instance_to_status = self.instance_to_engine_status
        for instance_id in self.instance_ids:
            if instance_to_status[instance_id].get_waiting_queue_size() != 0:
                continue
            instance_version = self.instance_to_version_after_sync[instance_id]
            if await self.ps_manager.check_aborted_model_versions.remote(instance_version):
                continue
            filtered_request_meta = await self._fetch_filtered_request_meta(instance_version)
            filtered_request_ids = [meta[0] for meta in filtered_request_meta]
            if len(filtered_request_ids) > 0:
                is_aborted = await self.ps_manager.check_aborted_requests.remote(filtered_request_ids, remove=False)
                filtered_request_ids = [
                    request_id for i, request_id in enumerate(filtered_request_ids) if not is_aborted[i]
                ]
                is_validate_list = [request_meta[1] for request_meta in filtered_request_meta]
                can_reserve = await self.ps_manager.can_reserve_request.remote(
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

    async def _wait_interrupted_partial_requests_loop_back(self, instance_ids: list[RolloutInstanceId]):
        futures = []
        for instance_id in instance_ids:
            replica_id, dp_rank = instance_id
            futures.append(self.server_handles[replica_id].wait_for_requests_to_drain.remote())
        await asyncio.gather(*futures)
