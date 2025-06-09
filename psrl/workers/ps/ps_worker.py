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

from psrl.utils.atomic import add_lock
from psrl.workers.ps.staleness_controller import BufferStatus, StalenessInventory, StalenessBuffer, EntryCategory, EntryInfo, Entry

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


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
    model_state_dict: Optional[Mapping[str, Union[Tensor, DTensor]]] = None


@add_lock
class PSRL_PSWorker(Worker):
    def __init__(self, psrl_config: DictConfig) -> None:
        super().__init__()
        self.psrl_config = psrl_config
        
        # PS worker specific attributes
        self.rollout_instance_tracker: Dict[Union[str, int], RolloutInstanceStatus] = {}  # Maps rollout instance IDs to their corresponding info
        self.model_store: Optional[ModelStore] = None  # The current model store, which contains the model state dict and version tag
        self.staleness_inventory: Optional[StalenessInventory] = None  # The staleness inventory for managing stale entries
        
        # Waiting lists for training batches
        self._buffer_waiters: Dict[int, List[asyncio.Future]] = {}  # Maps buffer IDs to a set of futures waiting for that buffer
        # Waiting lists for model versions
        self._version_waiters: Dict[int, List[asyncio.Future]] = {}  # Maps version tags to a set of futures waiting for that version
        
        if not dist.is_initialized():
            dist.init_process_group()
        assert self.world_size == dist.get_world_size(), "The world size of PSRL_PSWorker must match the torch distributed world size."
        
        if self.rank == 0:
            # Initialize the staleness inventory
            self.staleness_inventory = StalenessInventory(
                num_entries=self.psrl_config.staleness_buffer_entries,
            )
        
    def get_ps_handle(self):
        """Get the PS handle."""
        assert self.rank == 0, "Only the rank 0 PS worker can get the PS handle."
        return ray.get_current_actor()
    
    def register_rollout_instance(self, rollout_instance_id: Union[str, int]):
        """Register a new rollout instance."""
        assert self.rank == 0, "Only the rank 0 PS worker can register a rollout instance."
        self.rollout_instance_tracker[rollout_instance_id] = RolloutInstanceStatus(
            version_tag=0
        )
        
    # ------- STALENESS INVENTORY MANAGEMENT -------
        
    def get_max_reserve_num(self, model_version) -> int:
        """Get the maximum number of entries that can be reserved for a specific model version."""
        assert self.rank == 0, "Only the rank 0 PS worker can get the max reserve num."
    
        max_staleness_buffer_id = model_version + self.psrl_config.staleness
        return self.staleness_inventory.get_empty_entries_total_num(max_staleness_buffer_id)
        
    def reserve_rollout_instance_request(
        self,
        rollout_instance_id: Union[str, int],
        local_request_id: Union[str, int],
        model_version: int
    ) -> Tuple[Optional[int], Optional[int]]:
        """Reserve a request for a specific rollout instance."""
        assert self.rank == 0, "Only the rank 0 PS worker can reserve a rollout instance request."
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
        assert self.rank == 0, "Only the rank 0 PS worker can occupy a rollout instance request."
        assert rollout_instance_id in self.rollout_instance_tracker, f"Rollout instance {rollout_instance_id} is not registered."
        
        curr_rollout_instance_model_version = self.get_rollout_instance_model_version(self, rollout_instance_id)
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
        if min_ready_buffer_id is not None:
            # If there are ready buffers, wake up the waiters for the minimum ready buffer
            self._awake_training_batch_waiters(min_ready_buffer_id)
        
    async def wait_for_training_batch(
        self,
        buffer_id: int
    ) -> Optional[DataProto]:
        """Await a training batch for a specific buffer ID."""
        assert self.rank == 0, "Only the rank 0 PS worker can await a training batch."
        if self.staleness_inventory.get_buffer_status(buffer_id) == BufferStatus.READY:
            # If the buffer is ready, return immediately
            return self.staleness_inventory.consume_buffer(buffer_id)
        
        # TODO: support more consumption strategies, now only support waiting for the buffer to be ready
        # 1. Partial rollout if buffer status is STUCK
        # 2. Truncate if buffer status is STUCK
        # 3. Drop the RESERVED entry if buffer status is STUCK and move some OCCUPIED entries from other buffers 
        
        logger.info(f"<PS>: buffer {buffer_id} is not ready, waiting for it to be ready.")
        fut = asyncio.get_event_loop().create_future()
        if buffer_id not in self._buffer_waiters:
            self._buffer_waiters[buffer_id] = []
        self._buffer_waiters[buffer_id].append(fut)
        await fut
        # Once resumed, return
        
    def _awake_training_batch_waiters(self, buffer_id: int):
        """
        Check whether there are waiters
        for this buffer. If yes, wake them up.
        """
        # Wake all Futures waiting for this buffer
        if buffer_id in self._buffer_waiters:
            buffer_data = self.staleness_inventory.consume_buffer(buffer_id)
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
        assert self.rank == 0, "Only the rank 0 PS worker can get the model version for a rollout instance."
        assert rollout_instance_id in self.rollout_instance_tracker, f"Rollout instance {rollout_instance_id} is not registered."
        
        return tag_to_int(self.rollout_instance_tracker[rollout_instance_id].version_tag)
        
    def get_ps_model_version(self) -> int:
        """Get the current model version."""
        assert self.rank == 0, "Only the rank 0 PS worker can get the model version."
        if self.model_store is None:
            return 0  # If no model is stored, return version 0
        
        return tag_to_int(self.model_store.version_tag)
     
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
        
    def push_model_state_dict_cpu(
        self,
        version_tag: Union[str, int],
        model_state_dict: Optional[Mapping[str, Union[Tensor, DTensor]]],
    ):
        """Push a model to the PS."""
        assert self.rank == 0, "Only the rank 0 PS worker can push a model on CPU."
        
        self.model_store = ModelStore(
            version_tag=version_tag,
            model_state_dict=model_state_dict
        )
        self._awake_ps_model_version_waiters(tag_to_int(version_tag))
        
        logger.info(f"Model instance with version tag {version_tag} pushed successfully.")
        
    def get_model_state_dict_cpu(
        self
    ) -> Optional[Mapping[str, Union[Tensor, DTensor]]]:
        """Get the latest model state dict from PS."""
        assert self.rank == 0, "Only the rank 0 PS worker can get a model on CPU."
        assert self.model_store is not None, "Model instance is not initialized."
        
        return self.model_store.model_state_dict