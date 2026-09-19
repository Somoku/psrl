"""RolloutCoordinator composes loop mixins into the public coordinator API."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time

import aiohttp
import ray
from ray.actor import ActorHandle

from psrl.utils.common.http_utils import find_available_port
from psrl.utils.elastic_rm.diagnostics import log_elastic_rm_backlog_diag
from psrl.utils.logger import DualOutputHandler
from psrl.utils.server.command import CommandExtension
from psrl.workers.gen.smg_adapter import (
    ROUTING_LOOP_STATUS_PATH,
    WORKERS_STATS_PATH,
    build_weight_version_updates,
)
from psrl.workers.gen.stats_collector import EngineStats
from psrl.workers.gen.utils import DEFAULT_MAX_CONNECTIONS, DEFAULT_TIMEOUT, RolloutInstanceId
from psrl.workers.gen.zmq_queue import ZMQPullQueue

from . import base as base_module
from . import command_loop as command_loop_module
from . import status_loop as status_loop_module
from .base import CoordinatorBase
from .command_loop import CommandHandlerMixin
from .session import thunder_agent as thunder_agent_module
from .session.thunder_agent import ThunderAgentSessionMixin
from .status_loop import StatusMixin
from .sync_and_migrate import GreedySyncMixin, StatusBasedSyncMixin, SyncAndMigrateMixin
from .sync_and_migrate import greedy as greedy_module
from .sync_and_migrate import status_based as status_based_module
from .sync_and_migrate import sync_and_migrate_mixin as sync_and_migrate_module

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

# Loop modules use distinct loggers. Modules targeting one file must share a
# handler because concurrent write-mode handlers would truncate each other.
_LOG_FILE_TO_MODULE_LOGGERS = {
    "RolloutCoordinator": [command_loop_module.psrl_logger, base_module.psrl_logger],
    "SyncAndMigStrategy": [
        sync_and_migrate_module.psrl_logger,
        greedy_module.psrl_logger,
        status_based_module.psrl_logger,
    ],
    "Status": [status_loop_module.psrl_logger],
    "SessionStrategy": [thunder_agent_module.psrl_logger],
}


class RolloutCoordinator(
    CommandHandlerMixin,
    StatusMixin,
    SyncAndMigrateMixin,
    GreedySyncMixin,
    StatusBasedSyncMixin,
    ThunderAgentSessionMixin,
    CoordinatorBase,
    CommandExtension,
):
    def __init__(
        self,
        config,
        ps_manager: ray.actor.ActorHandle,
        rollout_gateway_url: str,
        session_router_url: str | None = None,
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
        - Session hang/continue scheduling (ThunderAgent port)

        Args:
            config: Configuration object containing PSRL settings
            rollout_gateway_url: HTTP base URL of the SMG rollout gateway.
            session_router_url: HTTP base URL of the SessionRouter (for session
                hang/continue control). None disables the thunder_agent loop.
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

        if not rollout_gateway_url:
            raise ValueError("Rollout gateway URL must not be empty.")
        self.rollout_gateway_url = rollout_gateway_url.rstrip("/")
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
        self.model_sync_tasks: set[asyncio.Task] = set()
        self.replica_sync_tasks: dict[str, asyncio.Task] = {}
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
        # Absolute KV-cache token capacity per instance (max_concurrency * max_model_len,
        # obtained via get_total_kv_cache_tokens). Used by the hang/continue scheduler.
        self.instance_to_total_kv_tokens: dict[RolloutInstanceId, int] = {}

        # --- Session hang/continue scheduling (ThunderAgent port) ---
        self.session_router_url = session_router_url.rstrip("/") if session_router_url else None
        self._thunder_agent_cfg = self.config.psrl.rollout_coordination.session_strategy.thunder_agent
        self._thunder_agent_enabled = bool(self._thunder_agent_cfg.enable and self.session_router_url is not None)
        self.thunder_agent_task = None
        self.stop_thunder_agent = False
        self._thunder_scheduler = None
        self._session_client: aiohttp.ClientSession | None = None

        # LMCache MP coordinator subprocess handle (started before init_model, if enabled).
        self._lmcache_coordinator_proc: subprocess.Popen | None = None
        self._lmcache_coordinator_url: str | None = None

        # Route each loop logger to its configured file.
        self.log_prefix = "RolloutCoordinator"
        self._attach_loop_log_handlers()
        psrl_logger.info("Initialized RolloutCoordinator")

        # Stats recorder (opt-in)
        self._stats_recorder = None
        if self.config.psrl.status_collection.stats_recorder.enable:
            from psrl.workers.gen.stats_recorder import StatsRecorder

            self._stats_recorder = StatsRecorder(
                self.config.psrl,
                os.path.expanduser(self.config.psrl.logging_path),
            )
            self._stats_recorder.write_config(
                routing_strategy=self.config.psrl.rollout_coordination.routing_strategy.method,
                partial_rollout=self.config.psrl.rollout_coordination.partial_rollout.enable,
            )

    def _attach_loop_log_handlers(self) -> None:
        """
        Route each loop module logger to its configured file.

        Loggers targeting one file share a `DualOutputHandler` to prevent
        write-mode handles from truncating each other.
        """
        logging_path = self.config.psrl.logging_path

        # coordinator.py's own logger shares RolloutCoordinator.log with the
        # command/base loggers, so create that file's handler first and reuse it.
        coordinator_handler = self._ensure_dual_handler(psrl_logger, logging_path, "RolloutCoordinator")
        for log_prefix, module_loggers in _LOG_FILE_TO_MODULE_LOGGERS.items():
            shared_handler = coordinator_handler if log_prefix == "RolloutCoordinator" else None
            for module_logger in module_loggers:
                shared_handler = self._ensure_dual_handler(
                    module_logger, logging_path, log_prefix, handler=shared_handler
                )

    @staticmethod
    def _ensure_dual_handler(target_logger, logging_path, log_prefix, handler=None):
        """Attach ``handler`` (or a fresh DualOutputHandler for ``log_prefix``) to
        ``target_logger`` unless a DualOutputHandler is already present, and return
        the handler so callers can reuse it across loggers sharing one file."""
        existing = next((h for h in target_logger.handlers if isinstance(h, DualOutputHandler)), None)
        if existing is not None:
            return existing
        if handler is None:
            handler = DualOutputHandler(logging_path, log_prefix)
        target_logger.addHandler(handler)
        return handler

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
            dp_size (int): Number of instances in the rollout replica
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

    async def _do_sleep_instance(self, replica_id: str) -> None:
        """Execute sleep on one replica. Rollout path uses nixl_sleep. Subclasses may override."""
        await self.server_handles[replica_id].nixl_sleep.remote(level=self._get_sleep_level())

    async def _do_wake_up_instance(self, replica_id: str) -> None:
        """Execute wake_up on one replica. Rollout path uses nixl_wake_up. Subclasses may override."""
        await self.server_handles[replica_id].nixl_wake_up.remote()

    def world_size(self):
        """Get the total world size (number of rollout and validate instances)."""
        return sum([rollout_replica.world_size for rollout_replica in self.rollout_replicas.values()])

    def _tag_to_server(self, tag: str):
        """Get the rollout server handles for a given tag.

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

    def _get_ordered_server_items(self, tag: str = "all") -> list[tuple[int, str, ActorHandle]]:
        """Return `(replica_idx, replica_id, server_handle)` sorted by replica index."""
        if tag in ["rollout", "validate"]:
            replica_ids = self.tag_to_replica_ids[tag]
        elif tag == "all":
            replica_ids = self.replica_ids
        else:
            raise ValueError(f"Unknown tag {tag} for getting server handles")
        return [
            (self.rollout_replicas[replica_id].replica_rank, replica_id, self.server_handles[replica_id])
            for replica_id in sorted(replica_ids, key=lambda rid: self.rollout_replicas[rid].replica_rank)
        ]

    async def init_nixl_client(self):
        """Init the NIXL client on rollout and validate instances."""
        futures = []
        for server_handle in self.server_handles.values():
            futures.append(server_handle.init_nixl_client.remote())
        await asyncio.gather(*futures)
        psrl_logger.info(f"Initialized NIXL clients: replica_count={len(self.server_handles)}.")
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
        psrl_logger.info(f"Initial parameter server pull complete: replicas={len(futures)}, tag={tag!r}.")
        # Pulled replicas must advertise the parameter server version to the gateway.
        pulled_replica_ids = self.replica_ids if tag == "all" else self.tag_to_replica_ids[tag]
        pulled_instance_ids = [inst_id for inst_id in self.instance_ids if inst_id[0] in pulled_replica_ids]
        for instance_id in pulled_instance_ids:
            self.instance_to_model_version[instance_id] = self.ps_model_version
            self.instance_to_version_after_sync[instance_id] = self.ps_model_version
        if pulled_instance_ids:
            updates = build_weight_version_updates(pulled_instance_ids, self.ps_model_version)
            await self._publish_weight_version_updates(updates)
            psrl_logger.info(
                f"Published initial weight version: version={self.ps_model_version}, "
                f"instances={len(pulled_instance_ids)}, tag={tag!r}."
            )

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
        if self.config.psrl.rollout_coordination.sync_and_mig_strategy.method == "greedy":
            self.sync_task = self.running_loop.create_task(self._greedy_sync_and_migrate_loop())
            self.sync_task.add_done_callback(lambda f: f.result())  # To avoid silent error in async tasks
        elif self.config.psrl.rollout_coordination.sync_and_mig_strategy.method == "status_based":
            assert self.config.psrl.status_collection.enable, (
                "Status-based sync strategy is only supported when status collection is enabled"
            )
            self.sync_task = self.running_loop.create_task(self._status_based_sync_and_migrate_loop())
            self.sync_task.add_done_callback(lambda f: f.result())  # To avoid silent error in async tasks
        else:
            raise NotImplementedError(
                f"Sync strategy {self.config.psrl.rollout_coordination.sync_and_mig_strategy.method} is not supported"
            )
        # Check if rollout migration is enabled
        if self.config.psrl.rollout_coordination.sync_and_mig_strategy.mig.enable:
            assert self.config.psrl.status_collection.enable, (
                "Rollout migration is only supported when status collection is enabled"
            )
            assert self.config.psrl.rollout_coordination.partial_rollout.enable, (
                "Rollout migration is only supported when partial rollout is enabled"
            )

        # Start the stats recorder loop (opt-in)
        if self.config.psrl.status_collection.stats_recorder.enable:
            self.stats_recorder_task = self.running_loop.create_task(self._stats_recorder_loop())
            self.stats_recorder_task.add_done_callback(lambda f: f.result())

        # Start the session hang/continue scheduling loop (opt-in, ThunderAgent port)
        if self._thunder_agent_enabled:
            self.thunder_agent_task = self.running_loop.create_task(self._thunder_agent_loop())
            self.thunder_agent_task.add_done_callback(lambda f: f.result())

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
        self.stop_thunder_agent = True

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
        if self.thunder_agent_task is not None:
            await self.thunder_agent_task
            psrl_logger.info("Finished thunder agent task.")
        if self.model_sync_tasks:
            await asyncio.gather(*self.model_sync_tasks, return_exceptions=True)
        if self._stats_recorder is not None:
            self._stats_recorder.close()
        psrl_logger.info("All background tasks have been stopped.")
        self.status_queue.close()
        if not self.gateway_client.closed:
            await self.gateway_client.close()
        if self._session_client is not None and not self._session_client.closed:
            await self._session_client.close()
        psrl_logger.info("Cleaned up resources in RolloutCoordinator.")

    async def get_router_backlog_size(self) -> int:
        """Return pending request count in rollout router queue."""
        t_enter = time.monotonic()
        model_tag = str(self.config.gen_actor_rollout_ref.model.path).rstrip("/").split("/")[-1]
        log_elastic_rm_backlog_diag(
            psrl_logger,
            "stage=RolloutCoordinator_enter model=%s elapsed_since_entry_s=0.000",
            model_tag,
        )
        log_elastic_rm_backlog_diag(
            psrl_logger,
            "stage=RolloutCoordinator_before_router_rpc model=%s since_enter_s=%.3f",
            model_tag,
            time.monotonic() - t_enter,
        )
        t_rpc = time.monotonic()
        status = await self._gateway_get_json(ROUTING_LOOP_STATUS_PATH)
        pending_value = status.get("pending_request_num", status.get("queue_len"))
        if pending_value is None:
            worker_stats = await self._gateway_get_json(WORKERS_STATS_PATH)
            stats = worker_stats if isinstance(worker_stats, list) else worker_stats.get("workers", [])
            pending = sum(int(item.get("running_requests", 0)) for item in stats if isinstance(item, dict))
        else:
            pending = int(pending_value)
        log_elastic_rm_backlog_diag(
            psrl_logger,
            "stage=RolloutCoordinator_after_router_rpc model=%s pending=%d router_rpc_s=%.3f since_enter_s=%.3f",
            model_tag,
            pending,
            time.monotonic() - t_rpc,
            time.monotonic() - t_enter,
        )
        return pending

    # This is called by the PS manager to update the PS model version after pushing
    def set_ps_model_version(self, version: int):
        """
        Set the current PS model version.

        This method updates the internal PS model version.

        Args:
            version (int): The new PS model version to set.
        """
        self.ps_model_version = version
        assert self.ps_model_version > 0, "PS model version must be greater than 0."
        # NOTE(claude): Skip the ready-buffer check until resume consumes its first buffer.
        if self.ready_buffers:
            assert (self.ps_model_version - 1) in self.ready_buffers, (
                "PS model version must be greater than the ready buffers."
            )
            self.ready_buffers.remove(self.ps_model_version - 1)
        psrl_logger.info(f"Updated PS model version to {version}")

    def init_ps_model_version(self, version: int):
        """
        Initialize the PS model version during resume without the ready_buffers assertion.

        Unlike set_ps_model_version (which expects the previous buffer to be consumed),
        this method is used during checkpoint resume where no buffers have been consumed yet.

        Args:
            version (int): The PS model version to initialize to.
        """
        self.ps_model_version = version
        psrl_logger.info(f"Initialized resume PS model version: version={version}.")

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
        if rollout_instance_id in self.instance_to_engine_status:
            self.instance_to_engine_status[rollout_instance_id].model_version = version_tag
        psrl_logger.info(
            f"Updated rollout model version: instance_id={rollout_instance_id!r}, "
            f"old_version={old_version!r}, new_version={version_tag}."
        )

    def update_ready_buffer(self, ready_buffer: int):
        """
        Update the ready buffer.
        """
        self.ready_buffers.add(ready_buffer)
        psrl_logger.info(f"Updated ready buffers to: {self.ready_buffers}")

    # ------- FUNCTIONS FOR LMCACHE MP -------

    async def init_lmcache_p2p(self) -> None:
        """
        Verify that node-local MP servers have registered with the coordinator.

        MP servers self-register at startup and discover peers through the
        coordinator, so this only confirms discovery is live. Must be called
        after init_model() has completed on all instances.
        """
        from psrl.utils.common.http_utils import get, init_http_client

        assert self._lmcache_coordinator_url is not None, (
            "start_lmcache_coordinator() must be called before init_lmcache_p2p()"
        )

        # Ensure the HTTP client is initialized in the Coordinator process.
        init_http_client(server_concurrency=4, rollout_engine_num=1)

        instances = await get(f"{self._lmcache_coordinator_url}/instances", max_retries=3)
        if isinstance(instances, dict):
            instances = instances.get("instances", [])
        registered = {inst.get("instance_id") for inst in instances or []}
        assert registered, (
            "[LMCache] Coordinator reports no registered MP instances. MP servers may not have started correctly."
        )
        psrl_logger.info(f"[LMCache] Coordinator registered instances: {sorted(i for i in registered if i)}.")

    async def start_lmcache_coordinator(self) -> str:
        """
        Start the shared LMCache MP coordinator subprocess.

        Must be called BEFORE init_model() so the coordinator is listening when
        each node's MP server registers at startup.

        Returns:
            str: Base URL of the coordinator, e.g. `"http://10.0.0.1:9300"`.
        """
        from psrl.utils.kv_cache.runtime import start_coordinator

        lmcache_cfg = self._build_lmcache_coordinator_config()
        log_dir = os.path.expanduser(self.config.psrl.logging_path)
        log_path = os.path.join(log_dir, "lmcache_coordinator.log")
        self._lmcache_coordinator_proc = await start_coordinator(lmcache_cfg, log_path)
        self._lmcache_coordinator_url = f"http://{lmcache_cfg.coordinator_host}:{lmcache_cfg.coordinator_port}"
        psrl_logger.info(f"[LMCache] Coordinator started at {self._lmcache_coordinator_url!r}.")
        return self._lmcache_coordinator_url

    def _build_lmcache_coordinator_config(self):
        """
        Build the LMCache config used to launch the cluster-wide coordinator.

        Returns:
            LMCacheConfig: Configuration with the coordinator host resolved from
                the PS manager IP and the port from the LMCache config.
        """
        from omegaconf import OmegaConf

        from psrl.utils.kv_cache.config import LMCacheConfig

        raw = OmegaConf.to_container(self.config.psrl.get("lmcache", OmegaConf.create()), resolve=True) or {}
        raw["coordinator_host"] = str(self.config.psrl.get("ps_manager_ip", "127.0.0.1"))
        return LMCacheConfig(**raw)
