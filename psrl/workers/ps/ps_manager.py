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
from verl.single_controller.base import Worker
from verl.utils.fs import copy_to_local
from verl.utils.debug import log_gpu_memory_usage
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.workers.rollout.vllm_rollout import vllm_mode
from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager

from psrl.utils.ray import add_lock
from psrl.utils.logger import DualOutputHandler, get_worker_info, log_dual_events, log_single_event, EventType, deprecated
from psrl.utils.nixl import NIXLMetaServer
from psrl.workers.ps.staleness_controller import BufferStatus, StalenessInventory, StalenessBuffer, EntryCategory, EntryInfo, Entry
from psrl.workers.ps.ps_worker_group import PSWorkerGroup

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))


# TODO: may support other tag format
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


# TODO: Ensure PSManager is a singleton
@ray.remote
@add_lock
class PSManager:
    def __init__(self, psrl_config: DictConfig) -> None:
        self.psrl_config = psrl_config
        
        # PS worker specific attributes
        self.rollout_instance_tracker: Dict[Union[str, int], RolloutInstanceStatus] = {}  # Maps rollout instance IDs to their corresponding info
        self.model_store: Optional[ModelStore] = None  # The current model store, which contains the model state dict and version tag
        self.staleness_inventory: Optional[StalenessInventory] = None  # The staleness inventory for managing stale entries
        
        # Waiting lists for training batches
        self._buffer_waiters: Dict[int, List[asyncio.Future]] = {}  # Maps buffer IDs to a set of futures waiting for that buffer
        # Waiting lists for model versions
        self._version_waiters: Dict[int, List[asyncio.Future]] = {}  # Maps version tags to a set of futures waiting for that version
        
        # Initialize the staleness inventory
        self.staleness_inventory = StalenessInventory(
            num_entries=self.psrl_config.staleness_buffer_entries,
        )
        
        # NIXL related attributes
        self.expected_clients = 0
        self.nixl_meta_server: Optional[NIXLMetaServer] = None
            
        # Build logger
        self.log_prefix = f"PSManager"
        psrl_logger.addHandler(DualOutputHandler(self.log_prefix))
        psrl_logger.info(f"Initialized on {get_worker_info()}.")
        self.logged_ready_buffer_ids: Set[int] = set()
     
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
    
    def log_ready_buffer(self, buffer_id: int):
        """Log the ready buffer."""
        if buffer_id not in self.logged_ready_buffer_ids:
            log_single_event(f"Buffer {buffer_id} is ready", psrl_logger, event_type=EventType.BUFFER_READY)
            self.logged_ready_buffer_ids.add(buffer_id)
        
    def reserve_rollout_instance_request(
        self,
        rollout_instance_id: Union[str, int],
        local_request_id: Union[str, int],
        model_version: int
    ) -> Tuple[Optional[int], Optional[int]]:
        """Reserve a request for a specific rollout instance."""
        assert rollout_instance_id in self.rollout_instance_tracker, f"Rollout instance {rollout_instance_id} is not registered."
        
        # Create an entry in the staleness inventory
        # note that model_version may be a future version of the current rollout instance
        entry_info = EntryInfo(
            rollout_instance_id=rollout_instance_id,
            local_request_id=local_request_id,
            model_version=model_version
        )
        
        max_staleness_buffer_id = model_version + self.psrl_config.staleness
        buffer_id, entry_id = self.staleness_inventory.reserve_data(
            entry_info=entry_info,
            max_staleness_buffer_id=max_staleness_buffer_id
        )
        
        # TODO: better handle the case where the staleness inventory is full
        if buffer_id is None or entry_id is None:
            pass
        
        return buffer_id, entry_id
        
    def occupy_rollout_instance_request(
        self,
        rollout_instance_id: Union[str, int],
        local_request_id: Union[str, int],
        data: DataProto
    ):
        """Occupy a request for a specific rollout instance."""
        assert rollout_instance_id in self.rollout_instance_tracker, f"Rollout instance {rollout_instance_id} is not registered."
        
        curr_rollout_instance_model_version = self.get_rollout_instance_model_version(rollout_instance_id)
        entry_info = EntryInfo(
            rollout_instance_id=rollout_instance_id,
            local_request_id=local_request_id,
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
        
    async def wait_for_training_batch(
        self,
        buffer_id: int
    ) -> DataProto:
        """Await a training batch for a specific buffer ID."""
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
            
    # ------- MODEL PUSH/PULL -------
    # Now we separate the control plane and data plane (ps_model = "nixl"), all the dataflow is handled by PSWorkerGroup.
    # And PSManager is only responsible for the control plane.
        
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
    
    # ------- PS NIXL DATAFLOW CONTROL -------
    
    def init_nixl_server(self, expected_clients: int):
        """Initialize the nixl server."""
        self.expected_clients = expected_clients
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
        psrl_logger.info(f"nixl server protocol step 1: waiting for {self.expected_clients} clients to connect and send sharding")
        self.nixl_meta_server.wait_for_client_shardings(self.expected_clients)
        psrl_logger.info(f"nixl server protocol step 2: notify all client shardings")
        self.nixl_meta_server.notify_all_client_shardings()
        psrl_logger.info(f"nixl server protocol step 3: waiting for {self.expected_clients} clients to send infos")
        self.nixl_meta_server.wait_for_client_infos(self.expected_clients)
        psrl_logger.info(f"nixl server protocol step 4: notify all client infos and the global comm plan")
        self.nixl_meta_server.notify_all_client_infos_and_comm_plan()
        psrl_logger.info(f"nixl server protocol step 5: waiting for {self.expected_clients} clients to send temp mappings")
        self.nixl_meta_server.wait_for_client_temp_mappings(self.expected_clients)
        psrl_logger.info(f"nixl server protocol step 6: notify all client temp mappings")
        self.nixl_meta_server.notify_all_client_temp_mappings()
    
    def bind_ps_worker_group(self, ps_worker_group: PSWorkerGroup):
        """Bind the PS worker group to the PSManager."""
        self.ps_worker_group = ps_worker_group