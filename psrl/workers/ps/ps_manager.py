import ray
import logging
from collections.abc import Mapping
from typing import Optional, List, Dict, Union, Tuple, Set

from torch import Tensor
from torch.distributed.tensor import DTensor
from omegaconf import DictConfig
from dataclasses import dataclass
from verl import DataProto

from psrl.utils.ray import add_lock, add_busy_polling_lock
from psrl.utils.logger import get_ps_logger, setup_ps_logger, get_worker_info, log_single_event, EventType, deprecated, log_dual_events
from psrl.utils.server.command import CommandType, Command
from psrl.utils.nixl import NIXLMetaServer
from psrl.workers.ps.staleness_controller import BufferStatus, StalenessInventory, EntryInfo
from psrl.workers.ps.ps_worker_group import PSWorkerGroup
from psrl.workers.ps.request_status_tracker import RequestStatusTracker

# Use the unified PS logger
psrl_logger = get_ps_logger()

@dataclass
class RolloutInstanceStatus:
    version_tag: int
    
@dataclass
class ModelStore:
    """Model Storage in the Parameter Server.
    
    This class holds the model state dictionary (ref) and its version tag.
    """
    version_tag: int
    # 'cpu' mode will store the actual model weights in `model_state_dict`
    model_state_dict: Optional[Mapping[str, Union[Tensor, DTensor]]] = None
    # 'cpu_ref' mode will store the Ray object reference in `model_state_dict_ref`
    model_state_dict_ref: Optional[ray.ObjectRef] = None  # ray object_ref


# TODO(lhy): Ensure PSManager is a singleton
@add_busy_polling_lock
class PSManager(RequestStatusTracker):
    def __init__(
        self,
        psrl_config: DictConfig,
    ) -> None:
        """Initialize the Parameter Server (PS) Manager.
        
        The PS Manager is responsible for managing model versions, staleness buffers, 
        and rollout requests. It coordinates between rollout workers and training workers
        through a staleness-controlled buffer system.
        
        Args:
            psrl_config (DictConfig): Configuration object containing parameters such as 
                rollout_n, staleness buffer entries, staleness limit, etc.
        """
        RequestStatusTracker.__init__(self, psrl_config)

        if self.psrl_config.redundant_rollout.enable:
            self.rollout_n = self.psrl_config.redundant_rollout.redundant_rollout_n
            self.alg_rollout_n = self.psrl_config.redundant_rollout.alg_rollout_n
        else:
            self.rollout_n = self.psrl_config.rollout_n
            self.alg_rollout_n = self.rollout_n

        # PS worker specific attributes
        self.rollout_instance_tracker: Dict[int, RolloutInstanceStatus] = {}  # Maps rollout instance IDs to their corresponding info
        self.model_store: Optional[ModelStore] = None  # The current model store, which contains the model state dict and version tag
        
        # Staleness buffer management
        self.staleness_inventory: Optional[StalenessInventory] = None  # The staleness inventory for managing stale entries
        
        # Set to track versions to be aborted
        self.abort_versions = set()
        
        # Set for buffer ids ready for deletion
        self.ready_for_delete_buffer_ids = set()

        # Initialize the staleness inventory
        if self.psrl_config.redundant_rollout.enable:
            entries_per_buffer = self.psrl_config.redundant_rollout.redundant_global_batch_size
            ready_entries_per_buffer = self.psrl_config.redundant_rollout.alg_global_batch_size
        else:
            entries_per_buffer = self.psrl_config.staleness_buffer_entries
            ready_entries_per_buffer = entries_per_buffer

        self.staleness_inventory = StalenessInventory(
            num_entries=entries_per_buffer,
            ready_num_entries=ready_entries_per_buffer,
            staleness=self.psrl_config.staleness,
            rollout_n=self.rollout_n,
        )

        # NIXL related attributes
        self.expected_agents = 0
        self.nixl_meta_server: Optional[NIXLMetaServer] = None
        self.ps_worker_group: Optional[PSWorkerGroup] = None
        self.ps_nixl_agent_names: Optional[List[str]] = None
        self.ps_nixl_train_storage_client_names: Optional[List[str]] = None
        self.ps_nixl_gen_storage_client_names: Optional[List[str]] = None
        
        # The log is now merged with the request status tracker
        '''    
        # Build logger
        self.log_prefix = f"PSManager"
        setup_ps_logger(self.psrl_config.logging_path, self.log_prefix)
        '''
        psrl_logger.info(f"PSManager initialized on {get_worker_info()}.")

    @deprecated("It is too slow to get the PS handle by `ray.get_runtime_context()`")
    def get_ps_manager_handle(self):
        """Get the PS handle."""
        return ray.get_runtime_context().current_actor

    def register_rollout_instance(self, rollout_instance_id: int):
        """Register a new rollout instance with the PS Manager.
        
        Args:
            rollout_instance_id (int): Unique identifier for the rollout instance
        """
        self.rollout_instance_tracker[rollout_instance_id] = RolloutInstanceStatus(
            version_tag=0
        )
        
    # ------- STALENESS INVENTORY MANAGEMENT -------

    def get_max_reserve_num(self, model_version) -> int:
        """Get the maximum number of entries that can be reserved for a specific model version.
        
        Args:
            model_version (int): The model version to reserve entries for
        Returns:
            int: The maximum number of entries that can be reserved for the given model version
        """
        max_staleness_buffer_id = model_version + self.psrl_config.staleness
        return self.staleness_inventory.get_empty_entries_total_num(max_staleness_buffer_id)
    
    # Used when the model version on the rollout instance is ahead of the request version tag (we allow a old version request to be routed to a new version instance)
    def update_request_version_tag(
        self,
        request_id: int,
        new_version_tag: int,
    ):
        """Update the version tag of a specific request in the staleness inventory."""
        self.staleness_inventory.update_request_version_tag(
            request_id=request_id,
            new_version_tag=new_version_tag,
        )
     
    # Used when the request is routed to a new rollout instance (partial rollout)
    def update_request_instance_id(
        self,
        request_id: int,
        new_instance_id: int,
    ):
        """Update the instance id of a specific request in the staleness inventory."""
        self.staleness_inventory.update_request_instance_id(
            request_id=request_id,
            new_instance_id=new_instance_id,
        )
    
    def clear_occupied_entries(
        self,
        prompt_ids: Union[int, List[int]],
    ):
        """Clear occupied entries in the staleness inventory."""
        self.staleness_inventory.clear_occupied_entries(prompt_ids)
    
    def clear_reserved_entries(
        self,
        prompt_ids: Union[int, List[int]],
        move_across_buffer: bool = False,
    ):
        """Clear reserved entries in the staleness inventory."""
        self.staleness_inventory.clear_reserved_entries(prompt_ids, move_across_buffer)
    
    def get_min_pending_buffer(self) -> int:
        """Get the minimum pending buffer id in the staleness inventory."""
        pending_buffers = self.staleness_inventory.get_buffers_with_capacity()
        if not pending_buffers:
            return self.staleness_inventory.buffer_id
        else:
            return min(pending_buffers)

    def maybe_delete_buffer(self, buffer_id: int):
        """Maybe delete a buffer from the staleness inventory.
        
        When RESERVE entries are cleared from a buffer, we can not delete it immediately
        because we rely on the READY buffer status to awake training workers.
        
        This method checks if the buffer can be deleted based on the current PS model version
        because the PS model version indicates which buffers have been consumed by training workers.
        """
        if buffer_id in self.ready_for_delete_buffer_ids:
            for bid in sorted(list(self.ready_for_delete_buffer_ids)):
                if bid <= buffer_id:
                    psrl_logger.debug(f"Clearing buffer {bid} after model version {buffer_id} is pushed.")
                    self.staleness_inventory.delete_buffer(bid)
                    self.ready_for_delete_buffer_ids.discard(bid)
                else:
                    break

    def delete_buffer(self, buffer_id: int):
        """Delete a buffer from the staleness inventory."""
        self.staleness_inventory.delete_buffer(buffer_id)

    def ensure_buffer_exists(self, buffer_id: int):
        """Ensure a buffer exists in the staleness inventory."""
        self.staleness_inventory.ensure_buffer_exists(buffer_id)
        
    def can_reserve_request(
        self,
        request_idx: Union[int, List[int]],
        model_versions: List[int],
        without_new_reserve_entry: bool = False
    ) -> List[bool]:
        """
        Check whether a request can be reserved for a given group of model versions.
        
        Args:
            request_idx (int): The request index
            model_versions (List[int]): The model versions that need to be checked
            without_new_reserve_entry (bool): Whether to check if the request can be reserved without a new reserve entry
        Returns:
            List[bool]: Whether the request can be reserved for each model version
        """
        # psrl_logger.info(f"Checking if request {request_idx} can be reserved for model versions: {model_versions}")
        if not isinstance(request_idx, list):
            results = []
            for model_version in model_versions:
                entry_info = EntryInfo(
                    rollout_instance_id=-1, # Not important for this check
                    prompt_id=request_idx // self.rollout_n,
                    request_idx=request_idx % self.rollout_n,
                    model_version=model_version
                )
                if without_new_reserve_entry:
                    results.append(self.staleness_inventory.can_reserve_data_without_new_reserve_entry(entry_info, model_version))
                else:
                    results.append(self.staleness_inventory.can_reserve_data(entry_info, model_version))
            return results
        else:
            multi_results = []
            for request_id in request_idx:
                results = []
                for model_version in model_versions:
                    entry_info = EntryInfo(
                        rollout_instance_id=-1, # Not important for this check
                        prompt_id=request_id // self.rollout_n,
                        request_idx=request_id % self.rollout_n,
                        model_version=model_version
                    )
                    if without_new_reserve_entry:
                        results.append(self.staleness_inventory.can_reserve_data_without_new_reserve_entry(entry_info, model_version))
                    else:
                        results.append(self.staleness_inventory.can_reserve_data(entry_info, model_version))
                multi_results.append(results)
            return multi_results
        
    def get_reserve_indicator(
        self,
        request_id: int,
        model_versions: List[int],
    ) -> List[float]:
        """
        Get the indicator of reserving a request for a given model version.
        indicator = inf: cannot reserve
        indicator = -inf: can reserve without new reserve entry
        indicator = -x: can reserve with new reserve entry in x-th pending buffer id
        
        Args:
            request_id (int): The request id
            model_versions (List[int]): The model versions that need to be checked
        Returns:
            List[int]: The indicator of reserving a request for each model version
        """
        indicators = []
        for model_version in model_versions:
            entry_info = EntryInfo(
                rollout_instance_id=-1, # Not important for this check
                prompt_id=request_id // self.rollout_n,
                request_idx=request_id % self.rollout_n,
                model_version=model_version
            )
            if self.staleness_inventory.can_reserve_data_without_new_reserve_entry(entry_info, model_version):
                indicators.append(float('-inf'))
            elif self.staleness_inventory.can_reserve_data(entry_info, model_version):
                max_pending_buffer_id = self.staleness_inventory.get_max_pending_buffer_id(model_version + self.psrl_config.staleness)
                indicators.append(-max_pending_buffer_id)
            else:
                indicators.append(float('inf'))
        return indicators

    def reserve_rollout_instance_requests(
        self,
        rollout_instance_ids: Union[int, List[int]],
        request_ids: Union[int, List[int]],
        model_versions: Union[int, List[int]],
    ) -> Tuple[Optional[List[int]], Optional[List[int]]]:
        """
        Reserve requests for specific rollout instances and model versions.
        
        This method will reserve buffer entries for requests in the specified rollout instance,
        without storing actual rollout data in the staleness inventory.
        For Group Sampling, it supports reserving multiple buffer entries if `by_parent` is True.
        Note that the request will be ignored if the group entries have been reserved.
        
        Args:
            rollout_instance_ids (Union[int, List[int]]): The rollout instance ids
            request_ids (Union[int, List[int]]): The request ids
            model_versions (Union[int, List[int]]): The model versions
        Returns:
            Tuple[Optional[List[int]], Optional[List[int]]]: A tuple containing two lists:
                - List of reserved buffer ids
                - List of reserved entry ids
            If reservation fails, returns (None, None).
        """
        if not isinstance(rollout_instance_ids, list):
            rollout_instance_ids = [rollout_instance_ids]
        if not isinstance(request_ids, list):
            request_ids = [request_ids]
        if not isinstance(model_versions, list):
            model_versions = [model_versions]
        
        for rollout_instance_id in rollout_instance_ids:
            assert rollout_instance_id == -1 or rollout_instance_id in self.rollout_instance_tracker, f"Rollout instance {rollout_instance_id} is not registered."
        
        # Initialize the reserved entry and buffer ids
        entry_ids = []
        buffer_ids = []
        for rollout_instance_id, request_id, model_version in zip(rollout_instance_ids, request_ids, model_versions):
            max_staleness_buffer_id = model_version + self.psrl_config.staleness
            # Create an entry in the staleness inventory
            # note that model_version may be a future version of the current rollout instance
            entry_info = EntryInfo(
                rollout_instance_id=rollout_instance_id,
                prompt_id=request_id // self.rollout_n,
                request_idx=request_id % self.rollout_n,
                model_version=model_version
            )
        
            buffer_id, entry_id = self.staleness_inventory.reserve_data(
                entry_info=entry_info,
                max_staleness_buffer_id=max_staleness_buffer_id
            )
            # TODO(lhy): better handle the case where the staleness inventory is full
            if buffer_id is None or entry_id is None:
                raise RuntimeError(f"Failed to reserve entry for request {request_id} in rollout instance {rollout_instance_id} "
                                   f"with model version {model_version}. "
                                   f"Please check if the staleness inventory is full or the model version is too old.")
            
            entry_ids.append(entry_id)
            buffer_ids.append(buffer_id)
        
        return buffer_ids, entry_ids

    def abort_requests(self, request_ids: Union[List[int], int], abort_group: bool = True, blocking: bool = False):
        """Abort specific rollout requests.
        This method will abort the specified requests and update the staleness inventory accordingly.
        
        Args:
            request_ids (Union[List[int], int]): The unique identifiers of the requests to abort.
            abort_group (bool, optional): Whether to abort the entire group of requests
                if the group will not satisfy sampling requirements after abortion. Defaults to True.
            blocking (bool, optional): Whether to block until the abortion is complete. Defaults to False.
        """
        if not isinstance(request_ids, list):
            request_ids = [request_ids]
        
        request_ids = set(request_ids) # Ensure uniqueness
        abort_request_ids = request_ids
        prompt_id_to_abort_request_idxs = {}
        for request_id in request_ids:
            prompt_id = request_id // self.rollout_n
            if prompt_id not in prompt_id_to_abort_request_idxs:
                prompt_id_to_abort_request_idxs[prompt_id] = []
            prompt_id_to_abort_request_idxs[prompt_id].append(request_id % self.rollout_n)
        # Update the corresponding entries, and clear entries if necessary (handled at the end)
        clear_entries = []
        for prompt_id, abort_request_idxs in prompt_id_to_abort_request_idxs.items():
            if prompt_id not in self.staleness_inventory.data_tracker:
                continue
            buffer_id, entry_id = self.staleness_inventory.data_tracker[prompt_id]
            entry_info = self.staleness_inventory.buffers[buffer_id].entries[entry_id].entry_info
            all_requests_of_entry = entry_info.get_all_requests(self.rollout_n)
            rest_requests_of_entry = set(all_requests_of_entry) - set(abort_request_idxs)
            if abort_group and len(rest_requests_of_entry) < self.alg_rollout_n:
                abort_request_ids = abort_request_ids.union(set(rest_requests_of_entry))
                clear_entries.append(entry_info.prompt_id)
            else:
                # Update the entry_info to remove aborted request idxs
                origin_entry_version = min(entry_info.model_version) if isinstance(entry_info.model_version, list) else entry_info.model_version
                update_idxs = []
                assert isinstance(entry_info.request_idx, list), "entry_info.request_idx should be a list."
                for i, request_idx in enumerate(entry_info.request_idx):
                    if request_idx not in abort_request_idxs:
                        update_idxs.append(i)
                if isinstance(entry_info.rollout_instance_id, list):
                    entry_info.rollout_instance_id = [entry_info.rollout_instance_id[i] for i in update_idxs]
                if isinstance(entry_info.model_version, list):
                    entry_info.model_version = [entry_info.model_version[i] for i in update_idxs]
                entry_info.request_idx = [entry_info.request_idx[i] for i in update_idxs]
                self.staleness_inventory.buffers[buffer_id].entries[entry_id].entry_info = entry_info

        # Clear the entries
        self.staleness_inventory.clear_reserved_entries(clear_entries)
        # Abort the requests
        self._abort_requests(list(abort_request_ids), blocking)
        psrl_logger.debug(f"Abort requests done: {abort_request_ids=}, {clear_entries=}")

    def check_staleness_abort(self, buffer_id: int):
        """ Check and interrupt rollout instances if necessary based on ready buffers."""
        curr_ps_model_version = self.get_ps_model_version(debug_info="ps_manager")
        ready_buffer_ids = self.staleness_inventory.ready_buffer_ids().copy()
        abort_request_ids = set()
        curr_abort_versions = set()
        
        if buffer_id >= self.psrl_config.staleness:
            version_to_abort = buffer_id - self.psrl_config.staleness
            # NOTE(linsh): The READY order of buffers can not be guaranteed
            # so we need more strict checks to avoid aborting requests too early.
            # `curr_ps_model_version - 1` is READY and consumed by training workers
            # so we need to check from `curr_ps_model_version` to `version_to_abort + staleness`
            # to ensure all buffers in `[version_to_abort, version_to_abort + staleness]` are READY
            buffer_range = set(range(max(version_to_abort, curr_ps_model_version), version_to_abort + self.psrl_config.staleness + 1))
            psrl_logger.debug(f"Checking abort for version {version_to_abort}, buffer range {buffer_range} should be ready in {ready_buffer_ids}.")
            # When `buffer_id` buffer is READY, we need to check related versions [version_to_abort, version_to_abort + staleness]
            # that may need to be aborted due to the READY status of `buffer_id`.
            # Because `version_to_abort` can be aborted, for the following continuous buffers,
            # we only need to check whether `curr_buffer_id + staleness` is READY to decide whether to abort `curr_buffer_id`.
            if buffer_range.issubset(ready_buffer_ids):
                curr_abort_versions.add(version_to_abort)
                # Further check the continuous buffers to see if they can also be aborted
                # If the next buffer is also READY, then we can abort the current buffer as well in one go
                for curr_buffer_id in range(version_to_abort + 1, version_to_abort + self.psrl_config.staleness + 1):
                    if curr_buffer_id + self.psrl_config.staleness in ready_buffer_ids:
                        curr_abort_versions.add(curr_buffer_id)
                        self.abort_versions.discard(curr_buffer_id)
                    else:
                        break
                psrl_logger.info(f"Aborting requests with version tag in {curr_abort_versions} due to ready buffer {buffer_id}.")
                curr_abort_versions = sorted(list(curr_abort_versions))
                # Collect requests to abort
                for abort_version in curr_abort_versions:
                    requests_of_abort_version = self.get_requests_of_abort_version(abort_version)
                    psrl_logger.debug(f"Requests of version {abort_version} to abort: {requests_of_abort_version}")
                    abort_request_ids = abort_request_ids.union(requests_of_abort_version)
            else:
                self.abort_versions.add(version_to_abort)
        
        if abort_request_ids:
            with log_dual_events(f"Abort {len(abort_request_ids)} requests in staleness check", psrl_logger, level=logging.INFO, event_type=EventType.OTHER):
                self.abort_requests(list(abort_request_ids), blocking=True)

        # If the buffer has no RESERVE entries after clearing entries, delete it or mark for deletion
        for buffer_id in ready_buffer_ids:
            if self.staleness_inventory.buffers[buffer_id].get_reserve_entry_num() == 0:
                if curr_ps_model_version > buffer_id:
                    psrl_logger.debug(f"Deleting buffer {buffer_id} immediately after aborting requests due to larger ps version {curr_ps_model_version}.")
                    self.staleness_inventory.delete_buffer(buffer_id)
                else:
                    psrl_logger.debug(f"Marking buffer {buffer_id} ready for deletion after aborting requests.")
                    self.ready_for_delete_buffer_ids.add(buffer_id)

        psrl_logger.debug(f"Check staleness abort done for buffer {buffer_id}. Current PS model version {curr_ps_model_version}, "
                          f"original ready buffers {ready_buffer_ids}, abort versions {curr_abort_versions}, "
                          f"abort {len(abort_request_ids)} requests. "
                          f"After abortion, current ready buffers {self.staleness_inventory.ready_buffer_ids()}.")

    def occupy_rollout_instance_request(
        self,
        prompt_id: int,
        request_ids: Optional[Union[int, List[int]]] = None,
        accumulate_sample: Optional[bool] = True,
    ) -> Tuple[Optional[int], Optional[BufferStatus], EntryInfo]:
        """
        Store a finished request in the staleness inventory, maybe occupy the buffer
        if one of the following requirements is met:
        (1). No parent requests (`parent_id` is None). Note that in this case the data will
        directly occupy the buffer, bypassing storing in the data pool.
        (2). Collect required `rollout_n` requests for group sampling.
        
        Args:
            prompt_id (int): The prompt id of the request
            request_ids (Optional[Union[int, List[int]]]): The request ids to occupy
            accumulate_sample (Optional[bool]): Whether to accumulate samples for group sampling
        Returns:
            Tuple[Optional[int], Optional[BufferStatus], EntryInfo]: A tuple containing:
                - buffer_id (Optional[int]): The buffer id where the data is stored, or None if not occupied
                - occupy_num (Optional[BufferStatus]): The buffer status after occupation, or None if not occupied
                - entry_info (EntryInfo): The entry information of the occupied data
        """
        if request_ids is None:
            request_ids = [prompt_id]
        
        # Remove the request from the training ready requests in the request status manager
        self.remove_train_ready_request(request_ids)
        
        buffer_id, entry_id, occupy_num = self.staleness_inventory.occupy_data_with_reserve(prompt_id)
        if buffer_id is None:
            # Occupy failed due to staleness limit, return the old entry info for abortion
            old_buffer_id, old_entry_id = self.staleness_inventory.data_tracker[prompt_id]
            entry_info = self.staleness_inventory.buffers[old_buffer_id].entries[old_entry_id].entry_info
            self.staleness_inventory.clear_reserved_entries(prompt_id, move_across_buffer=False)
            return None, None, entry_info

        # Update ready num entries if not accumulate_sample, requiring more data to occupy
        if not accumulate_sample:
            self.staleness_inventory.buffers[buffer_id].ready_num_entries += 1
        entry_info = self.staleness_inventory.buffers[buffer_id].entries[entry_id].entry_info
        return buffer_id, occupy_num, entry_info

    # ------- MODEL VERSION MANAGEMENT -------

    def get_all_rollout_instance_model_versions(self) -> Dict[int, int]:
        """Get all rollout instance model versions.
        
        Returns:
            Dict[int, int]: A dictionary mapping rollout instance IDs to their model versions
        """
        return {instance_id: instance_status.version_tag for instance_id, instance_status in self.rollout_instance_tracker.items()}

    def get_rollout_instance_model_version(self, rollout_instance_id: int) -> int:
        """Get the model version for a specific rollout instance.
        
        Args:
            rollout_instance_id (int): The rollout instance id
            
        Returns:
            int: The model version for the specified rollout instance
        """
        assert rollout_instance_id in self.rollout_instance_tracker, f"Rollout instance {rollout_instance_id} is not registered."
        
        return self.rollout_instance_tracker[rollout_instance_id].version_tag
        
    def get_ps_model_version(self, debug_info: str = None) -> int:
        """Get the current model version."""
        psrl_logger.debug(f"Getting PS model version from model store: {self.model_store}, debug info: {debug_info}.")
        if self.model_store is None:
            return 0  # If no model is stored, return version 0
        return self.model_store.version_tag
    
    def _update_rollout_instance_model_version_tag_to_latest(self, rollout_instance_id: int):
        """Update the rollout instance model version to the latest model version."""
        assert rollout_instance_id in self.rollout_instance_tracker, f"Rollout instance {rollout_instance_id} is not registered."
        assert self.rollout_coordinator is not None, "Rollout coordinator is not set. Please set it before updating rollout instance model version."

        if self.rollout_instance_tracker[rollout_instance_id].version_tag != self.model_store.version_tag:
            self.rollout_instance_tracker[rollout_instance_id].version_tag = self.model_store.version_tag
            # Sync the rollout instance model version in the rollout coordinator
            self.rollout_coordinator.set_rollout_instance_model_version.remote(
                rollout_instance_id=rollout_instance_id,
                version_tag=self.model_store.version_tag,
            )
            psrl_logger.info(f"Updated rollout instance {rollout_instance_id} model version to {self.model_store.version_tag}.")

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
        version_tag: int,
        model_state_dict: Optional[Mapping[str, Union[Tensor, DTensor]]]
    ):
        """
        Push a model to the PS. In 'cpu' mode, store the real state dict. In 'cpu_ref' mode, this should not be called.
        This method will block until the state dict is received by the PS worker (potential bottleneck for large models).
        
        Args:
            version_tag (int): The version tag of the model
            model_state_dict (Optional[Mapping[str, Union[Tensor, DTensor]]]): The model state dict to push
        """
        assert self.psrl_config.ps_mode == "cpu", "push_model_state_dict_cpu should only be used in 'cpu' mode."

        self.model_store = ModelStore(
            version_tag=version_tag,
            model_state_dict=model_state_dict
        )
        if version_tag > 0:
            self.maybe_delete_buffer(version_tag - 1)
        self.rollout_coordinator.set_ps_model_version.remote(version_tag)
        log_single_event(f"Model with version tag {version_tag} pushed successfully", psrl_logger, event_type=EventType.PUSH)

    # NOTE: If you manually wrap ObjectRef in a container (like list/tuple),ray will not recursively dereference all refs inside the container
    # Only the top-level task/actor arguments are expanded to real values, and ray will not traverse all nested structures to find ObjectRefs. 
    def push_model_state_dict_cpu_ref_list(
        self,
        version_tag: int,
        model_state_dict_ref_list: List[ray.ObjectRef]
    ):
        """
        Push a model to the PS by storing a ray object_ref. Only used in 'cpu_ref' mode.
        This method is non-blocking for the PS worker and only updates metadata (no large data transfer here).
        
        Args:
            version_tag (int): The version tag of the model
            model_state_dict_ref_list (List[ray.ObjectRef]): The list of ray object_refs to push
        """
        assert self.psrl_config.ps_mode == "cpu_ref", "push_model_state_dict_ref should only be used in 'cpu_ref' mode."
        assert len(model_state_dict_ref_list) == 1, "Only one model state dict ref is supported in 'cpu_ref' mode."

        self.model_store = ModelStore(
            version_tag=version_tag,
            model_state_dict_ref=model_state_dict_ref_list[0]
        )
        if version_tag > 0:
            self.maybe_delete_buffer(version_tag - 1)
        self.rollout_coordinator.set_ps_model_version.remote(version_tag)
        log_single_event(f"Model with version tag {version_tag} (ref) pushed successfully", psrl_logger, event_type=EventType.PUSH)
        
    def push_model_state_dict_nixl(self, version_tag: int):
        """
        Record the version tag of the model state dict pushed to the PS via NIXL.
        The actual model state dict is stored in the PS worker group.
        """
        self.model_store = ModelStore(
            version_tag=version_tag,
        )
        if version_tag > 0:
            self.maybe_delete_buffer(version_tag - 1)
        self.rollout_coordinator.set_ps_model_version.remote(version_tag)
        log_single_event(f"Model with version tag {version_tag} (nixl) pushed successfully", psrl_logger, event_type=EventType.PUSH)

    def pull_model_state_dict_cpu(
        self,
        rollout_instance_id: int
    ) -> Optional[Mapping[str, Union[Tensor, DTensor]]]:
        """
        Pull the latest model state dict from PS via CPU. Only used in 'cpu' mode.
        This will block until the state dict is transferred (potential bottleneck for large models).
        
        Args:
            rollout_instance_id (int): The rollout instance id
            
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
        rollout_instance_id: int
    ) -> ray.ObjectRef:
        """
        Return the ray object_ref for the latest model state dict. Only used in 'cpu_ref' mode.
        This is a fast operation (no large data transfer here).
        
        Args:
            rollout_instance_id (int): The rollout instance id
            
        Returns:
            ray.ObjectRef: The ray object_ref for the latest model state dict
        """
        assert self.psrl_config.ps_mode == "cpu_ref", "get_model_state_dict_ref should only be used in 'cpu_ref' mode."
        assert self.model_store is not None, "Model instance is not initialized."

        log_single_event(f"Rollout instance {rollout_instance_id} pulling latest model with version tag {self.model_store.version_tag} (ref)", psrl_logger, event_type=EventType.PULL)
        self._update_rollout_instance_model_version_tag_to_latest(rollout_instance_id)
        return self.model_store.model_state_dict_ref
    
    def pull_model_state_dict_nixl(self, rollout_instance_id: int):
        """
        Pull the latest model state dict from PS via NIXL. Only used in 'nixl_cpu' or 'nixl_gpu' mode.
        This only updates the version tag of the model state dict pulled from the PS.
        The actual model state dict is stored in the PS worker group.
        """
        assert self.psrl_config.ps_mode == "nixl_cpu" or self.psrl_config.ps_mode == "nixl_gpu", "pull_model_state_dict_nixl should only be used in 'nixl_cpu' or 'nixl_gpu' mode."
        assert self.model_store is not None, "Model instance is not initialized."
        log_single_event(f"Rollout instance {rollout_instance_id} pulling latest model with version tag {self.model_store.version_tag} (nixl)", psrl_logger, event_type=EventType.PULL)
        self._update_rollout_instance_model_version_tag_to_latest(rollout_instance_id)
