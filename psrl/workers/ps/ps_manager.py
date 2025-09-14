import ray
import os
import logging
import asyncio
from collections.abc import Mapping
from typing import Optional, List, Dict, Union, Tuple, Set

from torch import Tensor
from torch.distributed.tensor import DTensor
from omegaconf import DictConfig
from dataclasses import dataclass
from verl import DataProto

from psrl.utils.ray import add_lock
from psrl.utils.logger import get_ps_logger, setup_ps_logger, get_worker_info, log_single_event, EventType, deprecated
from psrl.utils.server.command import CommandType, Command
from psrl.utils.nixl import NIXLMetaServer
from psrl.workers.ps.staleness_controller import BufferStatus, StalenessInventory, EntryInfo
from psrl.workers.ps.ps_worker_group import PSWorkerGroup
from psrl.workers.ps.request_status_tracker import RequestStatusTracker

# Use the unified PS logger
psrl_logger = get_ps_logger()


# TODO(lhy): may support other tag format
def tag_to_int(version_tag: Union[str, int]) -> int:
    """Convert a version tag to an integer."""
    if isinstance(version_tag, str):
        return int(version_tag)
    elif isinstance(version_tag, int):
        return version_tag
    else:
        raise ValueError(f"Invalid version tag type: {type(version_tag)}. Expected str or int.")

@dataclass
class RolloutInstanceStatus:
    version_tag: Union[str, int]
    
@dataclass
class ModelStore:
    """Model Storage in the Parameter Server.
    
    This class holds the model state dictionary (ref) and its version tag.
    """
    version_tag: Union[str, int]
    # 'cpu' mode will store the actual model weights in `model_state_dict`
    model_state_dict: Optional[Mapping[str, Union[Tensor, DTensor]]] = None
    # 'cpu_ref' mode will store the Ray object reference in `model_state_dict_ref`
    model_state_dict_ref: Optional[ray.ObjectRef] = None  # ray object_ref


# TODO(lhy): Ensure PSManager is a singleton
@add_lock
class PSManager(RequestStatusTracker):
    def __init__(
        self,
        psrl_config: DictConfig,
        group_post_process_fn: Optional[callable] = None,
        buffer_post_process_fn: Optional[callable] = None,
    ) -> None:
        """Initialize the Parameter Server (PS) Manager.
        
        The PS Manager is responsible for managing model versions, staleness buffers, 
        and rollout requests. It coordinates between rollout workers and training workers
        through a staleness-controlled buffer system.
        
        Args:
            psrl_config (DictConfig): Configuration object containing parameters such as 
                rollout_n, staleness buffer entries, staleness limit, etc.
            group_post_process_fn (Optional[callable]): Optional function to post-process 
                grouped entry data before occupying the buffer
            buffer_post_process_fn (Optional[callable]): Optional function to post-process 
                ready buffer data
        """
        RequestStatusTracker.__init__(self)

        self.psrl_config = psrl_config
        if self.psrl_config.redundant_rollout.enable:
            self.rollout_n = self.psrl_config.redundant_rollout.redundant_rollout_n
            self.alg_rollout_n = self.psrl_config.redundant_rollout.alg_rollout_n
        else:
            self.rollout_n = self.psrl_config.rollout_n
            self.alg_rollout_n = self.rollout_n
        
        self.group_post_process_fn = group_post_process_fn
        self.buffer_post_process_fn = buffer_post_process_fn
        
        # PS worker specific attributes
        self.rollout_instance_tracker: Dict[Union[str, int], RolloutInstanceStatus] = {}  # Maps rollout instance IDs to their corresponding info
        self.model_store: Optional[ModelStore] = None  # The current model store, which contains the model state dict and version tag
        
        # Staleness buffer management
        self.staleness_inventory: Optional[StalenessInventory] = None  # The staleness inventory for managing stale entries
        
        # Waiting lists for training batches
        self._buffer_waiters: Dict[int, List[asyncio.Future]] = {}  # Maps buffer IDs to a set of futures waiting for that buffer
        # Waiting lists for model versions
        self._version_waiters: Dict[int, List[asyncio.Future]] = {}  # Maps version tags to a set of futures waiting for that version
        
        # Set of buffer ids that have been logged as ready, to avoid duplicate logging
        self.logged_ready_buffer_ids: Set[int] = set()
        
        self.max_ready_buffer_id = -1

        # Initialize the staleness inventory
        entries_per_buffer = self.psrl_config.staleness_buffer_entries * self.alg_rollout_n
        self.staleness_inventory = StalenessInventory(
            num_entries=entries_per_buffer,
            staleness=self.psrl_config.staleness,
            buffer_post_process_fn=self.buffer_post_process_fn,
        )
        
        # Track finished child requests for Group Sampling
        self.rollout_request_tracker: Dict[Union[str, int], List[EntryInfo]] = {} # Maps parent request ids to "occupied" child entries

        # NIXL related attributes
        self.expected_agents = 0
        self.nixl_meta_server: Optional[NIXLMetaServer] = None
        self.ps_worker_group: Optional[PSWorkerGroup] = None
        self.ps_nixl_agent_names: Optional[List[str]] = None
        self.ps_nixl_train_storage_client_names: Optional[List[str]] = None
        self.ps_nixl_gen_storage_client_names: Optional[List[str]] = None
            
        # Build logger
        self.log_prefix = f"PSManager"
        setup_ps_logger(self.psrl_config.logging_path, self.log_prefix)
        psrl_logger.info(f"Initialized on {get_worker_info()}.")

    @deprecated("It is too slow to get the PS handle by `ray.get_runtime_context()`")
    def get_ps_manager_handle(self):
        """Get the PS handle."""
        return ray.get_runtime_context().current_actor

    def set_agent_loop_manager(self, agent_loop_manager: ray.actor.ActorHandle):
        """Set the reference to the agent loop manager."""
        self.staleness_inventory.set_agent_loop_manager(agent_loop_manager)

    def register_rollout_instance(self, rollout_instance_id: Union[str, int]):
        """Register a new rollout instance with the PS Manager.
        
        Args:
            rollout_instance_id (Union[str, int]): Unique identifier for the rollout instance
        """
        self.rollout_instance_tracker[rollout_instance_id] = RolloutInstanceStatus(
            version_tag=0
        )
        
    # ------- STALENESS INVENTORY MANAGEMENT -------

    def _group_post_process(self, parent_id: int, entry_infos: List[EntryInfo]) -> bool:
        """Apply post-processing function to a group of entry infos.
        
        This method retrieves data from the data pool for each entry, applies
        the group post-processing function, and stores the processed data back.
        
        Args:
            parent_id (int): The parent request id for the group
            entry_infos (List[EntryInfo]): List of entry info objects to process
        
        Returns:
            bool: whether the group data is reserved
        """
        assert self.group_post_process_fn is not None, "Group post-processing function is not set."

        data_list = [self.staleness_inventory.pop_from_data_pool(entry_info) for entry_info in entry_infos]
        group_data = DataProto.concat(data_list)
        processed_group_data = self.group_post_process_fn(group_data)
        
        if not processed_group_data:
            psrl_logger.info(f"Post-processing function returned empty data for parent {parent_id}. Retrying later.")
            pending_buffers = self.staleness_inventory._buffer_ids_by_status[BufferStatus.PENDING]
            candidate_ids = list(pending_buffers)
            if not candidate_ids:
                waiting_buffer_id = self.staleness_inventory.buffer_id
            else:
                waiting_buffer_id = min(candidate_ids)

            # Notify agent loop manager to retry new requests
            self.staleness_inventory.notify_request_retry(waiting_buffer_id)
            return False
        else:
            processed_group_data_list = processed_group_data.chunk(len(processed_group_data))
            for entry_info, processed_group_data in zip(entry_infos, processed_group_data_list):
                self.staleness_inventory.update_to_data_pool(entry_info, processed_group_data)
            return True

    def get_max_reserve_num(self, model_version) -> int:
        """Get the maximum number of entries that can be reserved for a specific model version.
        
        Args:
            model_version (int): The model version to reserve entries for
            
        Returns:
            int: The maximum number of entries that can be reserved for the given model version
        """
        max_staleness_buffer_id = model_version + self.psrl_config.staleness
        return self.staleness_inventory.get_empty_entries_total_num(max_staleness_buffer_id)
    
    def filter_reserve_parent_ids(self, parent_ids: List[Union[str, int]]) -> List[Union[str, int]]:
        """Filter out parent ids that have already been tracked, to avoid duplicate reservation.
        
        Args:
            parent_ids (List[Union[str, int]]): The parent request ids to filter
            
        Returns:
            List[Union[str, int]]: The filtered parent request ids
        """
        filter_parent_ids = []
        for parent_id in parent_ids:
            if parent_id not in self.rollout_request_tracker.keys():
                filter_parent_ids.append(parent_id)
        return filter_parent_ids
        
    def reserve_rollout_instance_request(
        self,
        rollout_instance_id: Union[str, int],
        request_id: Union[str, int],
        model_version: int,
        reserve_num: int = 1,
        by_parent: bool = False,
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Reserve a request for a specific rollout instance.
        
        This method will reserve buffer entries for requests in the specified rollout instance,
        without storing actual rollout data in the staleness inventory.
        For Group Sampling, it supports reserving multiple buffer entries if `by_parent` is True.
        Note that the request will be ignored if the group entries have been reserved.
        
        Args:
            rollout_instance_id (Union[str, int]): The rollout instance id
            request_id (Union[str, int]): The request id
            model_version (int): The model version
            reserve_num (int): The number of entries to reserve (for group sampling)
            by_parent (bool): Whether to reserve entries by parent request
            
        Returns:
            Tuple[Optional[int], Optional[int]]: The buffer id and entry id
        """
        assert rollout_instance_id in self.rollout_instance_tracker, f"Rollout instance {rollout_instance_id} is not registered."
        if not by_parent:
            assert reserve_num == 1, "Non-group sampling should reserve one entry per request."
        else:
            assert reserve_num == self.rollout_n, "Group sampling should reserve `rollout_n` entries per parent request."
        
        # Initialize the reserved entry and buffer ids
        entry_ids = []
        buffer_ids = []
        max_staleness_buffer_id = model_version + self.psrl_config.staleness
        if by_parent:
            # Group Sampling
            parent_id = request_id
            reserve_ids = [parent_id * self.rollout_n + i for i in range(self.rollout_n)]
            self.rollout_request_tracker.setdefault(parent_id, [])
        else:
            reserve_ids = [request_id]
        for reserve_id in reserve_ids:
            # Create an entry in the staleness inventory
            # note that model_version may be a future version of the current rollout instance
            entry_info = EntryInfo(
                rollout_instance_id=rollout_instance_id,
                request_id=reserve_id,
                model_version=model_version
            )
        
            buffer_id, entry_id = self.staleness_inventory.reserve_data(
                entry_info=entry_info,
                max_staleness_buffer_id=max_staleness_buffer_id
            )
            entry_ids.append(entry_id)
            buffer_ids.append(buffer_id)
        
        # TODO(lhy): better handle the case where the staleness inventory is full
        if buffer_ids is None or entry_ids is None:
            pass
        
        return buffer_ids, entry_ids

    def try_awake_waiters(self):
        """Check for ready buffers and wake up waiters.
        
        This method checks if there are any new ready buffers for training and
        processes them by waking up waiters and handling staleness control.
        It also sends interruption commands for partial rollout if enabled.
        """
        # Check whether there exists ready buffer for training
        max_ready_buffer_id = self.staleness_inventory.max_ready_buffer_id()
        if (
            max_ready_buffer_id is not None and
            max_ready_buffer_id > self.max_ready_buffer_id
        ):
            self.max_ready_buffer_id = max_ready_buffer_id
            curr_ps_model_version = self.get_ps_model_version()
        
            # Notify the request status manager to abort requests
            # whose version_tag is equal to `min_ready_buffer_id - S`
            if max_ready_buffer_id >= self.psrl_config.staleness:
                # Abort requests with version_tag equal to `min_ready_buffer_id - S`
                version_to_abort = max_ready_buffer_id - self.psrl_config.staleness
                psrl_logger.debug(f"Aborting requests with version tag {version_to_abort} due to ready buffer {max_ready_buffer_id}.")
                self.abort_requests_of_version(version_to_abort)
                psrl_logger.debug(f"Abort requests of version {version_to_abort} done")
            
            # Notify the rollout server to check interruption
            if self.psrl_config.partial_rollout.enable:
                psrl_logger.debug(f"Start to check and sync for ready buffer {max_ready_buffer_id} with current PS model version {curr_ps_model_version}.")
                # Create async task to avoid blocking
                command = Command(
                    type=CommandType.CHECK_AND_SYNC,
                    buffer_id=max_ready_buffer_id,
                    curr_ps_model_version=curr_ps_model_version,
                )
                asyncio.create_task(self.execute_command(self.rollout_coordinator, command, blocking=False))
        
        min_ready_buffer_id = self.staleness_inventory.min_ready_buffer_id()
        if min_ready_buffer_id is not None:
            self.process_ready_buffer(min_ready_buffer_id)

    async def execute_command(self, server, command: Command, blocking: bool = False):
        """Execute a command on a server asynchronously.
        
        Args:
            server: The server actor to execute the command on
            command (Command): The command to execute
            blocking (bool): Whether to wait for command completion
            
        Raises:
            ValueError: If command execution fails
        """
        try:
            await server.exec_command.remote(command, blocking=blocking)
        except Exception as e:
            raise ValueError(f"Failed to execute command {command}: {e}")

    def log_ready_buffer(self, buffer_id: int):
        """Log the ready buffer."""
        if buffer_id not in self.logged_ready_buffer_ids:
            log_single_event(f"Buffer {buffer_id} is ready", psrl_logger, event_type=EventType.BUFFER_READY)
            self.logged_ready_buffer_ids.add(buffer_id)

    def process_ready_buffer(self, min_ready_buffer_id):
        """
        Notify the rollout server to check abortion and interruption when there is a ready buffer.
        
        This method is called when a buffer is ready to be processed.
        - Abortion: when a buffer is full, all requests with version_tag equal to `buffer_id - S` should be aborted.
        - Interruption: check the workload of each rollout instance and whether the abortion led by interruption will
        influence the training process, to determine whether to interrupt the rollout instance.
        
        Args:
            min_ready_buffer_id (int): The minimum ready buffer ID to process
        """
        self.log_ready_buffer(min_ready_buffer_id)
        # If there are ready buffers, wake up the waiters for the minimum ready buffer
        self._awake_training_batch_waiters(min_ready_buffer_id)

    def occupy_rollout_instance_request(
        self,
        rollout_instance_id: Union[str, int],
        request_id: Union[str, int],
        data: DataProto
    ):
        """Occupy a request for a specific rollout instance.
        
        Args:
            rollout_instance_id (Union[str, int]): The rollout instance id
            request_id (Union[str, int]): The request id
            data (DataProto): The data to occupy
        """
        assert rollout_instance_id in self.rollout_instance_tracker, f"Rollout instance {rollout_instance_id} is not registered."
        
        curr_rollout_instance_model_version = self.get_rollout_instance_model_version(rollout_instance_id)
        entry_info = EntryInfo(
            rollout_instance_id=rollout_instance_id,
            request_id=request_id,
            model_version=curr_rollout_instance_model_version
        )
        # Remove the request from the training ready requests in the request status manager
        self.remove_train_ready_request(request_id)
        
        self.staleness_inventory.occupy_data(
            entry_info=entry_info,
            data=data
        )
        self.try_awake_waiters()

    def store_and_maybe_occupy_rollout_instance_request(
        self,
        rollout_instance_id: Union[str, int],
        request_id: int,
        version_tag: int,
        data: DataProto,
        parent_id: Optional[Union[str, int]]=None,
    ):
        """
        Store a finished request in the staleness inventory, maybe occupy the buffer
        if one of the following requirements is met:
        (1). No parent requests (`parent_id` is None). Note that in this case the data will
        directly occupy the buffer, bypassing storing in the data pool.
        (2). Collect required `rollout_n` requests for group sampling.
        
        Args:
            rollout_instance_id (Union[str, int]): The rollout instance id
            request_id (int): The request id
            version_tag (int): The model version tag for the request
            data (DataProto): The data to store
            parent_id (Optional[Union[str, int]]): The parent request id (for group sampling)
        """
        assert rollout_instance_id in self.rollout_instance_tracker, f"Rollout instance {rollout_instance_id} is not registered."
        
        entry_info = EntryInfo(
            rollout_instance_id=rollout_instance_id,
            request_id=request_id,
            model_version=version_tag
        )
        
        # Remove the request from the training ready requests in the request status manager
        self.remove_train_ready_request(request_id)
        
        if parent_id is not None:
            # Group Sampling
            assert self.rollout_n > 1, "rollout_n must be greater than 1 to use parent_id."
            self.staleness_inventory.add_to_data_pool(
                entry_info=entry_info,
                data=data,
            )
            
            if parent_id not in self.rollout_request_tracker:
                self.rollout_request_tracker[parent_id] = []
                
            self.rollout_request_tracker[parent_id].append(entry_info)
            psrl_logger.debug(f"Store data for parent {parent_id} with info {entry_info}, "
                              f"request num: {len(self.rollout_request_tracker[parent_id])}")
            
            if len(self.rollout_request_tracker[parent_id]) == self.alg_rollout_n:
                psrl_logger.debug(f"Reached required {self.alg_rollout_n} samples for parent {parent_id}")
                entry_infos = self.rollout_request_tracker.pop(parent_id)
                psrl_logger.debug(f"Popped entry_infos from rollout_request_tracker for parent_id {parent_id}, entry count: {len(entry_infos)}")
                
                all_child_ids = set(range(self.rollout_n))
                stored_child_ids = set([int(entry_info.request_id) % self.rollout_n for entry_info in entry_infos])
                abort_child_ids = all_child_ids - stored_child_ids
                psrl_logger.debug(f"Stored child IDs: {stored_child_ids}, Abort child IDs: {abort_child_ids}")
                
                # Remove the sample data from the buffer in the request status manager
                psrl_logger.debug(f"Removing request data of {parent_id} from request status manager buffer")
                self.remove_request_data_from_buffer(parent_id)
                psrl_logger.debug(f"Successfully removed request data of {parent_id} from buffer")
                
                # Notify the request status manager to abort the child requests
                if abort_child_ids:
                    psrl_logger.debug(f"Aborting child requests {abort_child_ids} for parent request {parent_id}.")
                    self.abort_requests(list(abort_child_ids))
                    psrl_logger.debug(f"Successfully aborted {len(abort_child_ids)} child requests")

                reserve_data = True
                if self.group_post_process_fn:
                    reserve_data = self._group_post_process(parent_id, entry_infos)

                if reserve_data:
                    psrl_logger.debug(f"Occupying data for {len(entry_infos)} entry_infos")
                    for i, entry_info in enumerate(entry_infos):
                        psrl_logger.debug(f"Occupying data for entry_info {i+1}/{len(entry_infos)}: {entry_info}")
                        self.staleness_inventory.occupy_data(entry_info=entry_info)
                
                    self.try_awake_waiters()
        else:
            # Remove the sample data from the buffer in the request status manager
            self.remove_request_data_from_buffer(request_id)
            # Directly occupy data, bypassing storing process
            self.staleness_inventory.occupy_data(
                entry_info=entry_info,
                data=data,
            )
            self.try_awake_waiters()

    async def wait_for_training_batch(
        self,
        buffer_id: int
    ) -> DataProto:
        """Await a training batch for a specific buffer ID."""
        self.staleness_inventory.ensure_buffer_exists(buffer_id)
        
        if self.staleness_inventory.get_buffer_status(buffer_id) == BufferStatus.READY:
            # If the buffer is ready, return immediately
            return self.staleness_inventory.consume_buffer(buffer_id)
        
        # TODO(lhy): support more consumption strategies, now only support waiting for the buffer to be ready
        # 1. Partial rollout if buffer status is STUCK
        # 2. Truncate if buffer status is STUCK
        # 3. Drop the RESERVED entry if buffer status is STUCK and move some OCCUPIED entries from other buffers 
        
        psrl_logger.info(f"Buffer {buffer_id} is not ready, waiting for it to be ready.")
        fut = asyncio.get_event_loop().create_future()
        if buffer_id not in self._buffer_waiters:
            self._buffer_waiters[buffer_id] = []
        self._buffer_waiters[buffer_id].append(fut)
        result = await fut
        # Once resumed, return
        return result
        
    def _awake_training_batch_waiters(self, buffer_id: int):
        """Wake up all training waiters for a specific buffer.
        
        When a buffer becomes ready, this method consumes the buffer data
        and sets the result for all futures waiting for this buffer.
        
        Args:
            buffer_id (int): The buffer ID to wake waiters for
        """
        # Wake all Futures waiting for this buffer
        if buffer_id in self._buffer_waiters:
            buffer_data = self.staleness_inventory.consume_buffer(buffer_id)
            assert len(self._buffer_waiters[buffer_id]) == 1, \
                f"Expected only one waiter for buffer {buffer_id}, but found {len(self._buffer_waiters[buffer_id])}."
            # Set the result for all futures
            for fut in self._buffer_waiters[buffer_id]:
                if not fut.done():
                    fut.set_result(buffer_data)
            # Remove the key after waking all waiters
            del self._buffer_waiters[buffer_id]

    # ------- MODEL VERSION MANAGEMENT -------

    def get_all_rollout_instance_model_versions(self) -> Dict[Union[str, int], int]:
        """Get all rollout instance model versions.
        
        Returns:
            Dict[Union[str, int], int]: A dictionary mapping rollout instance IDs to their model versions
        """
        return {instance_id: instance_status.version_tag for instance_id, instance_status in self.rollout_instance_tracker.items()}

    def get_rollout_instance_model_version(self, rollout_instance_id: Union[str, int]) -> int:
        """Get the model version for a specific rollout instance.
        
        Args:
            rollout_instance_id (Union[str, int]): The rollout instance id
            
        Returns:
            int: The model version for the specified rollout instance
        """
        assert rollout_instance_id in self.rollout_instance_tracker, f"Rollout instance {rollout_instance_id} is not registered."
        
        return tag_to_int(self.rollout_instance_tracker[rollout_instance_id].version_tag)
        
    def get_ps_model_version(self) -> int:
        """Get the current model version."""
        if self.model_store is None:
            return 0  # If no model is stored, return version 0
        return tag_to_int(self.model_store.version_tag)
    
    def _update_rollout_instance_model_version_tag_to_latest(self, rollout_instance_id: Union[str, int]):
        """Update the rollout instance model version to the latest model version."""
        assert rollout_instance_id in self.rollout_instance_tracker, f"Rollout instance {rollout_instance_id} is not registered."
        assert self.rollout_coordinator is not None, "Rollout coordinator is not set. Please set it before updating rollout instance model version."

        if self.rollout_instance_tracker[rollout_instance_id].version_tag != self.model_store.version_tag:
            self.rollout_instance_tracker[rollout_instance_id].version_tag = self.model_store.version_tag
            # Sync the rollout instance model version in the rollout server
            ray.get(self.rollout_coordinator.set_rollout_instance_model_version.remote(
                rollout_instance_id=rollout_instance_id,
                version_tag=self.model_store.version_tag,
            ))
            psrl_logger.debug(f"Updated rollout instance {rollout_instance_id} model version to {self.model_store.version_tag}.")
     
    async def wait_for_ps_model_version(self, target_version: int):
        """
        Waiting for model weights of the target version in the PS worker.
        If model current PS model version >= target_version, return immediately.
        """
        ps_model_version = self.get_ps_model_version()
        if ps_model_version >= target_version:
            if ps_model_version > target_version:
                psrl_logger.warning(f"PS model version {ps_model_version} is greater than target version {target_version},"
                                    " which should not happen.")
            return
        
        # Otherwise, create a Future and store it in _version_waiters[target_version],
        # and await it until set_version wakes it up.
        fut = asyncio.get_event_loop().create_future()
        if target_version not in self._version_waiters:
            self._version_waiters[target_version] = []
        self._version_waiters[target_version].append(fut)
        await fut
        
    def _awake_ps_model_version_waiters(self, version: int):
        """Wake up all rollout waiters for this version if any."""
        # Wake all Futures waiting for this version
        if version in self._version_waiters:
            for fut in self._version_waiters[version]:
                if not fut.done():
                    fut.set_result(None)
            # Remove the key after waking all waiters
            del self._version_waiters[version]
     
    # ------- PS NIXL CONTROL PLANE -------
    
    def init_nixl_server(self, expected_agents: int):
        """Initialize the NIXL server for distributed communication.
        
        Args:
            expected_agents (int): Number of expected NIXL clients to connect
            
        Raises:
            ValueError: If server_mode is invalid or deprecated
        """
        self.expected_agents = expected_agents
        if self.psrl_config.nixl.server_mode == "storage_server":
            raise ValueError("Storage server mode is deprecated.")
        elif self.psrl_config.nixl.server_mode == "meta_server":
            self.nixl_meta_server = NIXLMetaServer(
                "NIXLMetaServer", 
                self.psrl_config.nixl
            )
        else:
            raise ValueError(f"Invalid NIXL server mode: {self.psrl_config.nixl.server_mode}")
        
    def nixl_protocol(self):
        """Execute the NIXL protocol for distributed communication setup.
        
        Connect to the nixl clients and sync the client shardings/infos/comm_plan/temp_mappings to all clients.
        This method orchestrates the complete NIXL protocol workflow:
        1. Wait for client shardings and create unified sharding
        2. Wait for client infos and create communication plan  
        3. Wait for client temp mappings and notify all clients
        
        The protocol ensures all NIXL clients are properly coordinated.
        """
        psrl_logger.info(f"nixl server protocol step 1: waiting for {self.expected_agents} clients to connect and send sharding")
        self.nixl_meta_server.wait_for_client_shardings(self.expected_agents)
        psrl_logger.info(f"nixl server protocol step 2: make unified sharding")
        self.nixl_meta_server.make_unified_sharding()
        psrl_logger.info(f"nixl server protocol step 3: notify all client shardings")
        self.nixl_meta_server.notify_all_client_shardings()
        psrl_logger.info(f"nixl server protocol step 4: waiting for {self.expected_agents} agents to send infos")
        self.nixl_meta_server.wait_for_client_infos(self.expected_agents)
        psrl_logger.info(f"nixl server protocol step 5: make comm plan")
        self.nixl_meta_server.make_comm_plan()
        psrl_logger.info(f"nixl server protocol step 6: notify all client infos and the global comm plan")
        self.nixl_meta_server.notify_all_client_infos_and_comm_plan()
        psrl_logger.info(f"nixl server protocol step 7: waiting for {self.expected_agents} agents to send temp mappings")
        self.nixl_meta_server.wait_for_client_temp_mappings(self.expected_agents)
        psrl_logger.info(f"nixl server protocol step 8: notify all client temp mappings")
        self.nixl_meta_server.notify_all_client_temp_mappings()
        psrl_logger.info(f"nixl server protocol done.")
    
    def bind_ps_worker_group(self, ps_worker_group: PSWorkerGroup):
        """Bind the PS worker group to the PSManager.
        
        This method establishes the connection between the PSManager and the
        PS worker group, enabling distributed model storage and retrieval.
        
        Args:
            ps_worker_group (PSWorkerGroup): The PS worker group to bind
        """
        self.ps_worker_group = ps_worker_group
        ps_nixl_agent_name_futures = self.ps_worker_group.execute_all_async("get_nixl_agent_name")
        ps_nixl_train_storage_client_name_futures = self.ps_worker_group.execute_all_async("get_nixl_train_storage_client_name")
        ps_nixl_gen_storage_client_name_futures = self.ps_worker_group.execute_all_async("get_nixl_gen_storage_client_name")
        self.ps_nixl_agent_names = ray.get(ps_nixl_agent_name_futures)
        self.ps_nixl_train_storage_client_names = ray.get(ps_nixl_train_storage_client_name_futures)
        self.ps_nixl_gen_storage_client_names = ray.get(ps_nixl_gen_storage_client_name_futures)

    def get_ps_worker_handle(self, client_name: str) -> ray.actor.ActorHandle:
        """Get the PS worker handle by the client name."""
        assert self.ps_worker_group is not None, "The PS worker group must be initialized before calling get_ps_worker_handle."
        worker = self.ps_worker_group.distinguish_worker_by_method(
            lambda worker: client_name == ray.get(worker.get_nixl_train_storage_client_name.remote()) or client_name == ray.get(worker.get_nixl_gen_storage_client_name.remote())
        )
        return worker
    
    def get_ps_nixl_agent_names(self) -> List[str]:
        """Get the NIXL agent name of the PS worker group."""
        assert self.ps_nixl_agent_names is not None, "The PS worker group must be initialized before calling get_ps_nixl_agent_names."
        return self.ps_nixl_agent_names

    def get_ps_nixl_train_storage_client_names(self) -> List[str]:
        """Get the NIXL train storage client name of the PS worker group."""
        assert self.ps_nixl_train_storage_client_names is not None, "The PS worker group must be initialized before calling get_ps_nixl_train_storage_client_name."
        return self.ps_nixl_train_storage_client_names
    
    def get_ps_nixl_gen_storage_client_names(self) -> List[str]:
        """Get the NIXL gen storage client name of the PS worker group."""
        assert self.ps_nixl_gen_storage_client_names is not None, "The PS worker group must be initialized before calling get_ps_nixl_gen_storage_client_name."
        return self.ps_nixl_gen_storage_client_names
            
    # ------- MODEL PUSH/PULL -------
    # Now we separate the control plane and data plane (ps_model = "nixl_cpu" or "nixl_gpu"), all the dataflow is handled by PSWorkerGroup.
    # And PSManager is only responsible for the control plane (i.e., PUSH/PULL methods only need to update the version tag, the actual model state dict is stored in the PS worker group).
        
    def push_model_state_dict_cpu(
        self,
        version_tag: Union[str, int],
        model_state_dict: Optional[Mapping[str, Union[Tensor, DTensor]]]
    ):
        """
        Push a model to the PS. In 'cpu' mode, store the real state dict. In 'cpu_ref' mode, this should not be called.
        This method will block until the state dict is received by the PS worker (potential bottleneck for large models).
        
        Args:
            version_tag (Union[str, int]): The version tag of the model
            model_state_dict (Optional[Mapping[str, Union[Tensor, DTensor]]]): The model state dict to push
        """
        assert self.psrl_config.ps_mode == "cpu", "push_model_state_dict_cpu should only be used in 'cpu' mode."

        self.model_store = ModelStore(
            version_tag=version_tag,
            model_state_dict=model_state_dict
        )
        # Awake all rollout waiters for this version
        self._awake_ps_model_version_waiters(version_tag)
        log_single_event(f"Model with version tag {version_tag} pushed successfully", psrl_logger, event_type=EventType.PUSH)

    # NOTE: If you manually wrap ObjectRef in a container (like list/tuple),ray will not recursively dereference all refs inside the container
    # Only the top-level task/actor arguments are expanded to real values, and ray will not traverse all nested structures to find ObjectRefs. 
    def push_model_state_dict_cpu_ref_list(
        self,
        version_tag: Union[str, int],
        model_state_dict_ref_list: List[ray.ObjectRef]
    ):
        """
        Push a model to the PS by storing a ray object_ref. Only used in 'cpu_ref' mode.
        This method is non-blocking for the PS worker and only updates metadata (no large data transfer here).
        
        Args:
            version_tag (Union[str, int]): The version tag of the model
            model_state_dict_ref_list (List[ray.ObjectRef]): The list of ray object_refs to push
        """
        assert self.psrl_config.ps_mode == "cpu_ref", "push_model_state_dict_ref should only be used in 'cpu_ref' mode."
        assert len(model_state_dict_ref_list) == 1, "Only one model state dict ref is supported in 'cpu_ref' mode."

        self.model_store = ModelStore(
            version_tag=version_tag,
            model_state_dict_ref=model_state_dict_ref_list[0]
        )
        # Awake all rollout waiters for this version
        self._awake_ps_model_version_waiters(version_tag)
        log_single_event(f"Model with version tag {version_tag} (ref) pushed successfully", psrl_logger, event_type=EventType.PUSH)
        
    def push_model_state_dict_nixl(self, version_tag: Union[str, int]):
        """
        Record the version tag of the model state dict pushed to the PS via NIXL.
        The actual model state dict is stored in the PS worker group.
        """
        assert self.psrl_config.ps_mode == "nixl_cpu" or self.psrl_config.ps_mode == "nixl_cpu_ref", "push_model_state_dict_nixl should only be used in 'nixl_cpu' or 'nixl_gpu' mode."
        self.model_store = ModelStore(
            version_tag=version_tag,
        )
        self._awake_ps_model_version_waiters(tag_to_int(version_tag))
        log_single_event(f"Model with version tag {version_tag} (nixl) pushed successfully", psrl_logger, event_type=EventType.PUSH)

    def pull_model_state_dict_cpu(
        self,
        rollout_instance_id: Union[str, int]
    ) -> Optional[Mapping[str, Union[Tensor, DTensor]]]:
        """
        Pull the latest model state dict from PS via CPU. Only used in 'cpu' mode.
        This will block until the state dict is transferred (potential bottleneck for large models).
        
        Args:
            rollout_instance_id (Union[str, int]): The rollout instance id
            
        Returns:
            Optional[Mapping[str, Union[Tensor, DTensor]]]: The model state dict
        """
        assert self.psrl_config.ps_mode == "cpu", "pull_model_state_dict_cpu should only be used in 'cpu' mode."
        assert self.model_store is not None, "Model instance is not initialized."

        log_single_event(f"Rollout instance {rollout_instance_id} pulling latest model with version tag {self.model_store.version_tag}", psrl_logger, event_type=EventType.PULL)
        self._update_rollout_instance_model_version_tag_to_latest(rollout_instance_id)
        return self.model_store.model_state_dict

    def pull_model_state_dict_cpu_ref(
        self,
        rollout_instance_id: Union[str, int]
    ) -> ray.ObjectRef:
        """
        Return the ray object_ref for the latest model state dict. Only used in 'cpu_ref' mode.
        This is a fast operation (no large data transfer here).
        
        Args:
            rollout_instance_id (Union[str, int]): The rollout instance id
            
        Returns:
            ray.ObjectRef: The ray object_ref for the latest model state dict
        """
        assert self.psrl_config.ps_mode == "cpu_ref", "get_model_state_dict_ref should only be used in 'cpu_ref' mode."
        assert self.model_store is not None, "Model instance is not initialized."

        log_single_event(f"Rollout instance {rollout_instance_id} pulling latest model with version tag {self.model_store.version_tag} (ref)", psrl_logger, event_type=EventType.PULL)
        self._update_rollout_instance_model_version_tag_to_latest(rollout_instance_id)
        return self.model_store.model_state_dict_ref
    
    def pull_model_state_dict_nixl(self, rollout_instance_id: Union[str, int]):
        """
        Pull the latest model state dict from PS via NIXL. Only used in 'nixl_cpu' or 'nixl_gpu' mode.
        This only updates the version tag of the model state dict pulled from the PS.
        The actual model state dict is stored in the PS worker group.
        """
        assert self.psrl_config.ps_mode == "nixl_cpu" or self.psrl_config.ps_mode == "nixl_gpu", "pull_model_state_dict_nixl should only be used in 'nixl_cpu' or 'nixl_gpu' mode."
        assert self.model_store is not None, "Model instance is not initialized."
        log_single_event(f"Rollout instance {rollout_instance_id} pulling latest model with version tag {self.model_store.version_tag} (nixl)", psrl_logger, event_type=EventType.PULL)
        self._update_rollout_instance_model_version_tag_to_latest(rollout_instance_id)
