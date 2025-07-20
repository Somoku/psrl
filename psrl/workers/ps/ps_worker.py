import ray
import os
import logging
import asyncio
import torch.distributed as dist
from collections.abc import Mapping
from typing import Optional, List, Dict, Union, Tuple, Set

from torch import Tensor
from torch.distributed.tensor import DTensor
from omegaconf import DictConfig
from dataclasses import dataclass
from verl import DataProto
from verl.single_controller.base.worker import Worker, DistGlobalInfo, DistRankInfo

from psrl.utils.ray import add_lock
from psrl.utils.logger import DualOutputHandler, get_worker_info, log_single_event, EventType
from psrl.workers.ps.staleness_controller import BufferStatus, StalenessInventory, EntryInfo
from psrl.utils.server.command import CommandType, Command

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))

# TODO: may support other tag format
# Question(linsh): why not just use int?
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

@add_lock
class PSRL_PSWorker(Worker):
    def __init__(self, psrl_config: DictConfig, request_status_manager) -> None:
        """
        Initialize the Parameter Server (PS) Worker, responsible for management of model versions,
        staleness buffers, and rollout requests.
        NOTE: PS workers are initialized on a node, but only the representative rank (rank 0) will perform
        the actual PS operations such as model storage and staleness buffer management.
        
        Args:
            psrl_config (DictConfig): Configuration object containing parameters such as rollout_n, staleness buffer entries, etc.
        """
        super().__init__()
        self.psrl_config = psrl_config
        if self.psrl_config.rollout_test.redundant_rollout.enable:
            self.rollout_n = self.psrl_config.rollout_test.redundant_rollout.redundant_rollout_n
            self.alg_rollout_n = self.psrl_config.rollout_test.redundant_rollout.alg_rollout_n
        else:
            self.rollout_n = self.psrl_config.rollout_n
            self.alg_rollout_n = self.rollout_n
        
        # Request status manager for tracking request statuses
        self.request_status_manager = request_status_manager
        
        # Rollout server reference
        self.rollout_server: Optional[ray.actor.ActorHandle] = None
        
        # PS worker specific attributes
        self.rollout_instance_tracker: Dict[Union[str, int], RolloutInstanceStatus] = {}  # Maps rollout instance IDs to their corresponding info
        self.model_store: Optional[ModelStore] = None  # The current model store, which contains the model state dict and version tag
        
        # Staleness buffer management
        self.staleness_inventory: Optional[StalenessInventory] = None  # The staleness inventory for managing stale entries
        # Track finished child requests for Group Sampling
        # TODO: move to request status manager? (leave for future merge with PS manager)
        self.rollout_request_tracker: Dict[Union[str, int], List[EntryInfo]] = {} # Maps parent request ids to "occupied" child entries
        
        # Waiting lists for training batches
        self._buffer_waiters: Dict[int, List[asyncio.Future]] = {}  # Maps buffer IDs to a set of futures waiting for that buffer
        # Waiting lists for model versions
        self._version_waiters: Dict[int, List[asyncio.Future]] = {}  # Maps version tags to a set of futures waiting for that version
        
        # Set of buffer ids that have been logged as ready, to avoid duplicate logging
        self.logged_ready_buffer_ids: Set[int] = set()
        
        if not dist.is_initialized():
            dist.init_process_group()
        assert self.world_size == dist.get_world_size(), "The world size of PSRL_PSWorker must match the torch distributed world size."
        
        if self.is_ps_representative_rank:
            # Initialize the staleness inventory
            entries_per_buffer = self.psrl_config.staleness_buffer_entries * self.alg_rollout_n
            self.staleness_inventory = StalenessInventory(
                num_entries=entries_per_buffer,
            )
            psrl_logger.debug(f"Staleness inventory initialized with {entries_per_buffer} entries per buffer")

        # Build logger
        self.log_prefix = f"PSWorker_R{self.rank}"
        psrl_logger.addHandler(DualOutputHandler(self.log_prefix))
        psrl_logger.info(f"Initialized on {get_worker_info()}.")

    def get_megatron_global_info(self):
        """Megatron distributed global info (for compatibility)"""
        tp_size = 1
        dp_size = 1
        pp_size = 1
        cp_size = 1
        info = DistGlobalInfo(tp_size=tp_size, dp_size=dp_size, pp_size=pp_size, cp_size=cp_size)
        return info

    def get_megatron_rank_info(self):
        """Megatron distributed rank info (for compatibility)"""
        tp_rank = 0
        dp_rank = 0
        pp_rank = 0
        cp_rank = 0
        info = DistRankInfo(tp_rank=tp_rank, dp_rank=dp_rank, pp_rank=pp_rank, cp_rank=cp_rank)
        return info

    @property   
    def is_ps_representative_rank(self) -> bool:
        """
        Check if the current rank is the representative rank.
        The representative rank is the rank 0 of the PS.
        """
        return self.rank == 0
        
    def get_ps_handle(self):
        """Get the PS handle."""
        assert self.is_ps_representative_rank, "Only the representative PS worker can get the PS handle."
        
        return ray.get_runtime_context().current_actor
    
    def set_rollout_server(self, rollout_server: ray.actor.ActorHandle):
        """Set the reference to the rollout server."""
        self.rollout_server = rollout_server

    def register_rollout_instance(self, rollout_instance_id: Union[str, int]):
        """Register a new rollout instance."""
        assert self.is_ps_representative_rank, "Only the representative PS worker can register a rollout instance."
        
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
        assert self.is_ps_representative_rank, "Only the representative PS worker can get the max reserve num."
    
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
        assert self.is_ps_representative_rank, "Only the representative PS worker can reserve a rollout instance request."
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
        
        # TODO: better handle the case where the staleness inventory is full
        if buffer_ids is None or entry_ids is None:
            pass
        
        return buffer_ids, entry_ids

    def try_awake_waiters(self):
        # Check whether there exists ready buffer for training
        min_ready_buffer_id = self.staleness_inventory.min_ready_buffer_id()
        if min_ready_buffer_id is not None:
            psrl_logger.debug(f"Found min ready buffer is {min_ready_buffer_id}")
            self.process_ready_buffer(min_ready_buffer_id)

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
        curr_ps_model_version = self.get_ps_model_version()
        
        # Notify the request status manager to abort requests
        # whose version_tag is equal to `min_ready_buffer_id - S`
        if min_ready_buffer_id >= self.psrl_config.staleness:
            # Abort requests with version_tag equal to `min_ready_buffer_id - S`
            version_to_abort = min_ready_buffer_id - self.psrl_config.staleness
            psrl_logger.debug(f"Aborting requests with version tag {version_to_abort} due to ready buffer {min_ready_buffer_id}.")
            ray.get(self.request_status_manager.abort_requests_of_version.remote(version_to_abort))
        
        # Notify the rollout server to check interruption
        ray.get(self.rollout_server.exec_command.remote(Command(
            type=CommandType.CHECK_AND_SYNC,
            buffer_id=min_ready_buffer_id,
            curr_ps_model_version=curr_ps_model_version,
        ), blocking=False))
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
        assert self.is_ps_representative_rank, "Only the representative PS worker can occupy a rollout instance request."
        assert rollout_instance_id in self.rollout_instance_tracker, f"Rollout instance {rollout_instance_id} is not registered."
        
        curr_rollout_instance_model_version = self.get_rollout_instance_model_version(rollout_instance_id)
        entry_info = EntryInfo(
            rollout_instance_id=rollout_instance_id,
            request_id=request_id,
            model_version=curr_rollout_instance_model_version
        )
        # Remove the request from the training ready requests in the request status manager
        ray.get(self.request_status_manager.remove_train_ready_request.remote(request_id))
        
        self.staleness_inventory.occupy_data(
            entry_info=entry_info,
            data=data
        )
        self.try_awake_waiters()

    def store_and_maybe_occupy_rollout_instance_request(
        self,
        rollout_instance_id: Union[str, int],
        request_id: Union[str, int],
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
            request_id (Union[str, int]): The request id
            data (DataProto): The data to store
            parent_id (Optional[Union[str, int]]): The parent request id (for group sampling)
        """
        psrl_logger.debug(f"store_and_maybe_occupy_rollout_instance_request called with rollout_instance_id={rollout_instance_id}, request_id={request_id}, parent_id={parent_id}")
        assert self.is_ps_representative_rank, "Only the representative PS worker can occupy a rollout instance request."
        assert rollout_instance_id in self.rollout_instance_tracker, f"Rollout instance {rollout_instance_id} is not registered."
        
        curr_rollout_instance_model_version = self.get_rollout_instance_model_version(rollout_instance_id)
        psrl_logger.debug(f"Current model version for rollout_instance_id {rollout_instance_id}: {curr_rollout_instance_model_version}")
        entry_info = EntryInfo(
            rollout_instance_id=rollout_instance_id,
            request_id=request_id,
            model_version=curr_rollout_instance_model_version
        )
        
        # Remove the request from the training ready requests in the request status manager
        psrl_logger.debug(f"Removing request_id {request_id} from train_ready_requests")
        ray.get(self.request_status_manager.remove_train_ready_request.remote(request_id))
        psrl_logger.debug(f"Successfully removed request_id {request_id} from train_ready_requests")
        
        if parent_id is not None:
            # Group Sampling
            assert self.rollout_n > 1, "rollout_n must be greater than 1 to use parent_id."
            psrl_logger.debug(f"Adding data for entry_info to staleness inventory")
            self.staleness_inventory.add_data(
                entry_info=entry_info,
                data=data,
            )
            
            if parent_id not in self.rollout_request_tracker:
                self.rollout_request_tracker[parent_id] = []
                
            self.rollout_request_tracker[parent_id].append(entry_info)
            psrl_logger.debug(f"Store data for parent {parent_id} with info {entry_info}, total requests: {len(self.rollout_request_tracker[parent_id])}")
            
            if len(self.rollout_request_tracker[parent_id]) == self.alg_rollout_n:
                psrl_logger.debug(f"Reached required alg_rollout_n={self.alg_rollout_n} for parent_id {parent_id}")
                entry_infos = self.rollout_request_tracker.pop(parent_id)
                psrl_logger.debug(f"Popped entry_infos from rollout_request_tracker for parent_id {parent_id}, entry count: {len(entry_infos)}")
                
                all_child_ids = set(range(self.rollout_n))
                stored_child_ids = set([int(entry_info.request_id) % self.rollout_n for entry_info in entry_infos])
                abort_child_ids = all_child_ids - stored_child_ids
                psrl_logger.debug(f"All child IDs: {all_child_ids}, Stored child IDs: {stored_child_ids}, Abort child IDs: {abort_child_ids}")
                
                # Remove the parent request (sample) data from the buffer in the request status manager
                psrl_logger.debug(f"Removing request data of {parent_id} from request status manager buffer")
                ray.get(self.request_status_manager.remove_request_data_from_buffer.remote(parent_id))
                psrl_logger.debug(f"Successfully removed request data of {parent_id} from buffer")
                
                # Notify the request status manager to abort the child requests
                if abort_child_ids:
                    psrl_logger.debug(f"Aborting child requests {abort_child_ids} for parent request {parent_id}.")
                    ray.get(self.request_status_manager.abort_requests.remote(abort_child_ids))
                    psrl_logger.debug(f"Successfully aborted {len(abort_child_ids)} child requests")

                psrl_logger.debug(f"Occupying data for {len(entry_infos)} entry_infos")
                for i, entry_info in enumerate(entry_infos):
                    psrl_logger.debug(f"Occupying data for entry_info {i+1}/{len(entry_infos)}: {entry_info}")
                    self.staleness_inventory.occupy_data(entry_info=entry_info)
                
                psrl_logger.debug(f"Trying to awake waiters after occupying data for parent_id {parent_id}")
                self.try_awake_waiters()
                psrl_logger.debug(f"Finished group sampling request processing for parent_id {parent_id}")
        else:
            # Directly occupy data, bypassing storing process
            psrl_logger.debug(f"No parent_id provided, directly occupying data for request_id {request_id}")
            self.staleness_inventory.occupy_data(
                entry_info=entry_info,
                data=data,
            )
            psrl_logger.debug(f"Successfully occupied data for entry_info: {entry_info}")

            psrl_logger.debug(f"Trying to awake waiters after direct data occupation")
            self.try_awake_waiters()
            psrl_logger.debug(f"Finished direct data occupation for request_id {request_id}")

    async def wait_for_training_batch(
        self,
        buffer_id: int
    ) -> DataProto:
        """Waiting for the training batch in the specified buffer."""
        assert self.is_ps_representative_rank, "Only the representative PS worker can await a training batch."
        self.staleness_inventory.ensure_buffer_exists(buffer_id)
        
        if self.staleness_inventory.get_buffer_status(buffer_id) == BufferStatus.READY:
            # If the buffer is ready, return immediately
            return self.staleness_inventory.consume_buffer(buffer_id)
        
        # TODO: support more consumption strategies, now only support waiting for the buffer to be ready
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
        """Awake all training waiters for the buffer if any."""
        # Wake all Futures waiting for this buffer
        if buffer_id in self._buffer_waiters:
            buffer_data = self.staleness_inventory.consume_buffer(buffer_id)
            psrl_logger.debug(f"Consume buffer {buffer_id}: {buffer_data}")
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
        assert self.is_ps_representative_rank, "Only the representative PS worker can get all rollout instance model versions."
        
        return {instance_id: tag_to_int(instance_status.version_tag) for instance_id, instance_status in self.rollout_instance_tracker.items()}

    def get_rollout_instance_model_version(self, rollout_instance_id: Union[str, int]) -> int:
        """Get the model version for a specific rollout instance.
        
        Args:
            rollout_instance_id (Union[str, int]): The rollout instance id
            
        Returns:
            int: The model version for the specified rollout instance
        """
        assert self.is_ps_representative_rank, "Only the representative PS worker can get the model version for a rollout instance."
        assert rollout_instance_id in self.rollout_instance_tracker, f"Rollout instance {rollout_instance_id} is not registered."
        
        return tag_to_int(self.rollout_instance_tracker[rollout_instance_id].version_tag)
        
    def get_ps_model_version(self) -> int:
        """Get current model version in the model store."""
        assert self.is_ps_representative_rank, "Only the representative PS worker can get the model version."
        if self.model_store is None:
            return 0  # If no model is stored, return version 0
        
        return tag_to_int(self.model_store.version_tag)
    
    def _update_rollout_instance_model_version_tag_to_latest(self, rollout_instance_id: Union[str, int]):
        """Update the rollout instance model version to the latest model version in `model_store`."""
        assert self.is_ps_representative_rank, "Only the representative PS worker can update the rollout instance model version to the latest."
        assert rollout_instance_id in self.rollout_instance_tracker, f"Rollout instance {rollout_instance_id} is not registered."

        self.rollout_instance_tracker[rollout_instance_id].version_tag = self.model_store.version_tag
        # Sync the rollout instance model version in the rollout server
        ray.get(self.rollout_server.set_rollout_instance_model_version.remote(
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
        assert self.is_ps_representative_rank, "Only the representative PS worker can push a model on CPU."
        assert self.psrl_config.ps_mode == "cpu", "push_model_state_dict_cpu should only be used in 'cpu' mode."

        self.model_store = ModelStore(
            version_tag=version_tag,
            model_state_dict=model_state_dict
        )
        # Awake all rollout waiters for this version
        self._awake_ps_model_version_waiters(tag_to_int(version_tag))
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
        assert self.is_ps_representative_rank, "Only the representative PS worker can push a model on CPU."
        assert self.psrl_config.ps_mode == "cpu_ref", "push_model_state_dict_ref should only be used in 'cpu_ref' mode."
        assert len(model_state_dict_ref_list) == 1, "Only one model state dict ref is supported in 'cpu_ref' mode."

        self.model_store = ModelStore(
            version_tag=version_tag,
            model_state_dict_ref=model_state_dict_ref_list[0]
        )
        # Awake all rollout waiters for this version
        self._awake_ps_model_version_waiters(tag_to_int(version_tag))
        log_single_event(f"Model with version tag {version_tag} (ref) pushed successfully", psrl_logger, event_type=EventType.PUSH)

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
        assert self.is_ps_representative_rank, "Only the representative PS worker can pull a model on CPU."
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
        assert self.is_ps_representative_rank, "Only the representative PS worker can provide model ref."
        assert self.psrl_config.ps_mode == "cpu_ref", "get_model_state_dict_ref should only be used in 'cpu_ref' mode."
        assert self.model_store is not None, "Model instance is not initialized."

        log_single_event(f"Rollout instance {rollout_instance_id} pulling latest model with version tag {self.model_store.version_tag} (ref)", psrl_logger, event_type=EventType.PULL)
        self._update_rollout_instance_model_version_tag_to_latest(rollout_instance_id)
        return self.model_store.model_state_dict_ref