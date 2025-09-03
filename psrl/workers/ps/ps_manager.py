import ray
import os
import sys
import logging
import asyncio
import torch
import torch.distributed as dist
from collections.abc import Mapping
from typing import Optional, Any, List, Dict, Union, Tuple, Set

from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.distributed.device_mesh import init_device_mesh
from time import sleep
from omegaconf import DictConfig, open_dict
from dataclasses import dataclass
from verl import DataProto
from verl.single_controller.base.worker import Worker, DistGlobalInfo, DistRankInfo
from verl.utils.fs import copy_to_local
from verl.utils.debug import log_gpu_memory_usage
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.workers.rollout.vllm_rollout import vllm_mode

from psrl.utils.ray import add_lock
from psrl.utils.logger import DualOutputHandler, get_worker_info, log_dual_events, log_single_event, EventType, deprecated
from psrl.utils.nixl import NIXLMetaServer
from psrl.workers.ps.staleness_controller import BufferStatus, StalenessInventory, StalenessBuffer, EntryCategory, EntryInfo, Entry
from psrl.workers.ps.ps_worker_group import PSWorkerGroup

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))


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
    version_tag: Union[str, int]
    # In 'cpu' mode, model_state_dict is the real state dict; in 'cpu_ref' mode, model_state_dict_ref is a ray object_ref
    model_state_dict: Optional[Mapping[str, Union[Tensor, DTensor]]] = None
    model_state_dict_ref: Optional[ray.ObjectRef] = None  # ray object_ref


# TODO(lhy): Ensure PSManager is a singleton
@add_lock
class PSManager:
    def __init__(self, psrl_config: DictConfig) -> None:
        self.psrl_config = psrl_config
        self.rollout_n = psrl_config.rollout_n
        
        # PS worker specific attributes
        self.rollout_instance_tracker: Dict[Union[str, int], RolloutInstanceStatus] = {}  # Maps rollout instance IDs to their corresponding info
        self.model_store: Optional[ModelStore] = None  # The current model store, which contains the model state dict and version tag
        self.staleness_inventory: Optional[StalenessInventory] = None  # The staleness inventory for managing stale entries
        
        self.rollout_request_tracker: Dict[Union[str, int], List[EntryInfo]] = {} # Maps parent request ids to "occupied" child entries
        
        # Waiting lists for training batches
        self._buffer_waiters: Dict[int, List[asyncio.Future]] = {}  # Maps buffer IDs to a set of futures waiting for that buffer
        # Waiting lists for model versions
        self._version_waiters: Dict[int, List[asyncio.Future]] = {}  # Maps version tags to a set of futures waiting for that version
        
        # Initialize the staleness inventory
        self.staleness_inventory = StalenessInventory(
            num_entries=self.psrl_config.staleness_buffer_entries * self.psrl_config.rollout_n,
        )
            
        # NIXL related attributes
        self.expected_agents = 0
        self.nixl_meta_server: Optional[NIXLMetaServer] = None
        self.ps_worker_group: Optional[PSWorkerGroup] = None
        self.ps_nixl_train_storage_client_names: Optional[List[str]] = None
        self.ps_nixl_gen_storage_client_names: Optional[List[str]] = None
            
        # Build logger
        self.log_prefix = f"PSManager"
        psrl_logger.addHandler(DualOutputHandler(self.psrl_config.logging_path, self.log_prefix))
        psrl_logger.info(f"Initialized on {get_worker_info()}.")
        self.logged_ready_buffer_ids: Set[int] = set()

    def get_megatron_global_info(self):
        # NOTE(ls): for compatibility with megatron worker
        tp_size = 1
        dp_size = 1
        pp_size = 1
        cp_size = 1
        info = DistGlobalInfo(tp_size=tp_size, dp_size=dp_size, pp_size=pp_size, cp_size=cp_size)
        return info

    def get_megatron_rank_info(self):
        # NOTE(ls): for compatibility with megatron worker
        tp_rank = 0
        dp_rank = 0
        pp_rank = 0
        cp_rank = 0
        info = DistRankInfo(tp_rank=tp_rank, dp_rank=dp_rank, pp_rank=pp_rank, cp_rank=cp_rank)
        return info
     
    @deprecated("It is too slow to get the PS handle by `ray.get_runtime_context()`")
    def get_ps_manager_handle(self):
        """Get the PS handle."""
        return ray.get_runtime_context().current_actor
    
    def register_rollout_instance(self, rollout_instance_id: Union[str, int]):
        """Register a new rollout instance."""
        self.rollout_instance_tracker[rollout_instance_id] = RolloutInstanceStatus(
            version_tag=0
        )
        
    # ------- STALENESS INVENTORY MANAGEMENT -------
        
    def get_max_reserve_num(self, model_version) -> int:
        """Get the maximum number of entries that can be reserved for a specific model version."""
        max_staleness_buffer_id = model_version + self.psrl_config.staleness
        return self.staleness_inventory.get_empty_entries_total_num(max_staleness_buffer_id)
    
    def filter_reserve_parent_ids(self, parent_ids: List[Union[str, int]]) -> List[Union[str, int]]:
        filter_parent_ids = []
        for parent_id in parent_ids:
            if parent_id not in self.rollout_request_tracker.keys():
                filter_parent_ids.append(parent_id)
        return filter_parent_ids
    
    def log_ready_buffer(self, buffer_id: int):
        """Log the ready buffer."""
        if buffer_id not in self.logged_ready_buffer_ids:
            log_single_event(f"Buffer {buffer_id} is ready", psrl_logger, event_type=EventType.BUFFER_READY)
            self.logged_ready_buffer_ids.add(buffer_id)
        
    def reserve_rollout_instance_request(
        self,
        rollout_instance_id: Union[str, int],
        request_id: Union[str, int],
        model_version: int,
        reserve_num: int = 1,
        by_parent: bool = False,
    ) -> Tuple[Optional[int], Optional[int]]:
        """Reserve a request for a specific rollout instance."""
        assert rollout_instance_id in self.rollout_instance_tracker, f"Rollout instance {rollout_instance_id} is not registered."
        
        entry_ids = []
        buffer_ids = []
        max_staleness_buffer_id = model_version + self.psrl_config.staleness
        if by_parent:
            parent_id = request_id
            self.rollout_request_tracker.setdefault(parent_id, [])
            for i in range(reserve_num):
                request_id = f"{parent_id}_r{i}"
                # Create an entry in the staleness inventory
                # note that model_version may be a future version of the current rollout instance
                entry_info = EntryInfo(
                    rollout_instance_id=rollout_instance_id,
                    request_id=request_id,
                    model_version=model_version
                )
            
                buffer_id, entry_id = self.staleness_inventory.reserve_data(
                    entry_info=entry_info,
                    max_staleness_buffer_id=max_staleness_buffer_id
                )
                entry_ids.append(entry_id)
                buffer_ids.append(buffer_id)
        else:
            # Create an entry in the staleness inventory
            # note that model_version may be a future version of the current rollout instance
            entry_info = EntryInfo(
                rollout_instance_id=rollout_instance_id,
                request_id=request_id,
                model_version=model_version
            )
            
            buffer_id, entry_id = self.staleness_inventory.reserve_data(
                entry_info=entry_info,
                max_staleness_buffer_id=max_staleness_buffer_id
            )
            entry_ids = [entry_id]
            buffer_ids = [buffer_id]
        
        # TODO(lhy): better handle the case where the staleness inventory is full
        if buffer_ids is None or entry_ids is None:
            pass
        
        return buffer_ids, entry_ids

    def occupy_rollout_instance_request(
        self,
        rollout_instance_id: Union[str, int],
        request_id: Union[str, int],
        data: DataProto
    ):
        """Occupy a request for a specific rollout instance."""
        assert rollout_instance_id in self.rollout_instance_tracker, f"Rollout instance {rollout_instance_id} is not registered."
        
        curr_rollout_instance_model_version = self.get_rollout_instance_model_version(rollout_instance_id)
        entry_info = EntryInfo(
            rollout_instance_id=rollout_instance_id,
            request_id=request_id,
            model_version=curr_rollout_instance_model_version
        )
        self.staleness_inventory.occupy_data(
            entry_info=entry_info,
            data=data
        )
        
        min_ready_buffer_id = self.staleness_inventory.min_ready_buffer_id()
        psrl_logger.debug(f"Occupy data with info {entry_info}, min ready buffer is {min_ready_buffer_id}")
        if min_ready_buffer_id is not None:
            self.log_ready_buffer(min_ready_buffer_id)
            # If there are ready buffers, wake up the waiters for the minimum ready buffer
            self._awake_training_batch_waiters(min_ready_buffer_id)

    def store_and_maybe_occupy_rollout_instance_request(
        self,
        rollout_instance_id: Union[str, int],
        request_id: Union[str, int],
        data: DataProto,
        parent_id: Optional[Union[str, int]]=None,
    ):
        """Notify the PS worker about a new finished request and 
            store the data in the staleness inventory data buffer.
            This is used when the request is a child request.
            If all child requests are finished, the PS worker will occupy the data.
        """
        assert rollout_instance_id in self.rollout_instance_tracker, f"Rollout instance {rollout_instance_id} is not registered."
        
        curr_rollout_instance_model_version = self.get_rollout_instance_model_version(rollout_instance_id)
        entry_info = EntryInfo(
            rollout_instance_id=rollout_instance_id,
            request_id=request_id,
            model_version=curr_rollout_instance_model_version
        )
        
        if parent_id is not None:
            assert self.rollout_n > 1, "rollout_n must be greater than 1 to use parent_id."
            self.staleness_inventory.add_data(
                entry_info=entry_info,
                data=data,
            )
            self.rollout_request_tracker[parent_id].append(entry_info)
            psrl_logger.info(f"[TRACE] Store data for parent {parent_id} with info {entry_info}, total requests: {len(self.rollout_request_tracker[parent_id])}")
            if len(self.rollout_request_tracker[parent_id]) == self.rollout_n:
                entry_infos = self.rollout_request_tracker.pop(parent_id)
                for entry_info in entry_infos:
                    self.staleness_inventory.occupy_data(entry_info=entry_info)
            
                min_ready_buffer_id = self.staleness_inventory.min_ready_buffer_id()
                psrl_logger.debug(f"Occupy data with info {entry_info}, min ready buffer is {min_ready_buffer_id}")
                if min_ready_buffer_id is not None:
                    self.log_ready_buffer(min_ready_buffer_id)
                    # If there are ready buffers, wake up the waiters for the minimum ready buffer
                    self._awake_training_batch_waiters(min_ready_buffer_id)
        else:
            self.staleness_inventory.occupy_data(
                entry_info=entry_info,
                data=data,
            )
            
            min_ready_buffer_id = self.staleness_inventory.min_ready_buffer_id()
            psrl_logger.debug(f"Occupy data with info {entry_info}, min ready buffer is {min_ready_buffer_id}")
            if min_ready_buffer_id is not None:
                self.log_ready_buffer(min_ready_buffer_id)
                # If there are ready buffers, wake up the waiters for the minimum ready buffer
                self._awake_training_batch_waiters(min_ready_buffer_id)

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
        """
        Check whether there are waiters
        for this buffer. If yes, wake them up.
        """
        # Wake all Futures waiting for this buffer
        if buffer_id in self._buffer_waiters:
            buffer_data = self.staleness_inventory.consume_buffer(buffer_id)
            psrl_logger.debug(f"Consume buffer {buffer_id}: {buffer_data}")
            assert len(self._buffer_waiters[buffer_id]) == 1, \
                f"Expected only one waiter for buffer {buffer_id}, but found {len(self._buffer_waiters[buffer_id])}."
            for fut in self._buffer_waiters[buffer_id]:
                if not fut.done():
                    fut.set_result(buffer_data)
            # Remove the key after waking all waiters
            del self._buffer_waiters[buffer_id]

    # ------- MODEL VERSION MANAGEMENT -------
        
    def get_rollout_instance_model_version(self, rollout_instance_id: Union[str, int]) -> int:
        """Get the model version for a specific rollout instance."""
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
        self.rollout_instance_tracker[rollout_instance_id].version_tag = self.model_store.version_tag
     
    async def wait_for_ps_model_version(self, target_version: int):
        """
        If model current PS model version >= target_version, return immediately.
        Otherwise, create a Future and store it in _version_waiters[target_version],
        and await it until set_version wakes it up.
        """
        if self.get_ps_model_version() >= target_version:
            return  
        
        # Otherwise, create a Future and wait for it
        fut = asyncio.get_event_loop().create_future()
        if target_version not in self._version_waiters:
            self._version_waiters[target_version] = []
        self._version_waiters[target_version].append(fut)
        await fut
        # Once resumed, return
        
    def _awake_ps_model_version_waiters(self, version: int):
        """
        Check whether there are waiters
        for this version. If yes, wake them up.
        """
        # Wake all Futures waiting for this version
        if version in self._version_waiters:
            for fut in self._version_waiters[version]:
                if not fut.done():
                    fut.set_result(None)
            # Remove the key after waking all waiters
            del self._version_waiters[version]
     
    # ------- PS NIXL CONTROL PLANE -------
    
    def init_nixl_server(self, expected_agents: int):
        """Initialize the nixl server."""
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
        """Connect to the nixl clients and sync the client shardings/infos/comm_plan/temp_mappings to all clients."""
        psrl_logger.info(f"nixl server protocol step 1: waiting for {self.expected_agents} agents to connect and send sharding")
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
        """Bind the PS worker group to the PSManager."""
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
        
    def push_model_state_dict_cpu(self, version_tag: Union[str, int], model_state_dict: Optional[Mapping[str, Union[Tensor, DTensor]]]):
        """
        Push a model to the PS. In 'cpu' mode, store the real state dict. In 'cpu_ref' mode, this should not be called.
        This method will block until the state dict is received by the PS worker (potential bottleneck for large models).
        """
        assert self.psrl_config.ps_mode == "cpu", "push_model_state_dict_cpu should only be used in 'cpu' mode."
        self.model_store = ModelStore(
            version_tag=version_tag,
            model_state_dict=model_state_dict
        )
        self._awake_ps_model_version_waiters(tag_to_int(version_tag))
        log_single_event(f"Model with version tag {version_tag} pushed successfully", psrl_logger, event_type=EventType.PUSH)

    # Tricky part: If you manually wrap ObjectRef in a container (like list/tuple), ray will not recursively dereference all refs inside the container
    # Only the top-level task/actor arguments are expanded to real values, and ray will not traverse all nested structures to find ObjectRefs. 
    def push_model_state_dict_cpu_ref_list(self, version_tag: Union[str, int], model_state_dict_ref_list: List[ray.ObjectRef]):
        """
        Push a model to the PS by storing a ray object_ref. Only used in 'cpu_ref' mode.
        This method is non-blocking for the PS worker and only updates metadata (no large data transfer here).
        """
        assert self.psrl_config.ps_mode == "cpu_ref", "push_model_state_dict_ref should only be used in 'cpu_ref' mode."
        self.model_store = ModelStore(
            version_tag=version_tag,
            model_state_dict_ref=model_state_dict_ref_list[0]
        )
        self._awake_ps_model_version_waiters(tag_to_int(version_tag))
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

    def pull_model_state_dict_cpu(self, rollout_instance_id: Union[str, int]) -> Optional[Mapping[str, Union[Tensor, DTensor]]]:
        """
        Pull the latest model state dict from PS via CPU. Only used in 'cpu' mode.
        This will block until the state dict is transferred (potential bottleneck for large models).
        """
        assert self.psrl_config.ps_mode == "cpu", "pull_model_state_dict_cpu should only be used in 'cpu' mode."
        assert self.model_store is not None, "Model instance is not initialized."
        log_single_event(f"Rollout instance {rollout_instance_id} pulling latest model with version tag {self.model_store.version_tag}", psrl_logger, event_type=EventType.PULL)
        self._update_rollout_instance_model_version_tag_to_latest(rollout_instance_id)
        return self.model_store.model_state_dict

    def pull_model_state_dict_cpu_ref(self, rollout_instance_id: Union[str, int]) -> ray.ObjectRef:
        """
        Return the ray object_ref for the latest model state dict. Only used in 'cpu_ref' mode.
        This is a fast operation (no large data transfer here).
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
    