import threading
import time
import torch
import logging
import os
import pickle
import ray
from copy import deepcopy
from typing import Dict, Any, Set, Optional, List, Tuple
from dataclasses import dataclass
from nixl._api import nixl_agent, nixl_agent_config
from omegaconf import DictConfig

from psrl.utils.logger import deprecated, get_worker_info
from psrl.utils.nixl.network_topology import get_local_ip, get_local_gpu_id
from psrl.utils.nixl.nixl_spec import NIXLTensorDescInfo, NIXLClientType, NIXLClientInfo, NIXLSharding, NIXLShardMetaInfo, NIXLInterface
from psrl.utils.nixl.comm_plan import NIXLCommPlan, CommunicationPlanner


psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))


# Utility function to combine tag and shard_idx
def make_shard_tag(tag: bytes, shard_idx) -> bytes:
    """
    Combine tag and shard_idx (int or tuple of int) into a single bytes object as message identifier.
    """
    if not isinstance(tag, bytes):
        raise TypeError(f"tag must be bytes, got {type(tag)}")
    if isinstance(shard_idx, int):
        idx_bytes = bytes([shard_idx])
    elif isinstance(shard_idx, tuple):
        if not all(isinstance(i, int) and 0 <= i < 256 for i in shard_idx):
            raise ValueError("All elements in shard_idx tuple must be int in [0, 255]")
        idx_bytes = bytes(shard_idx)
    else:
        raise TypeError("shard_idx must be int or tuple of int")
    return tag + idx_bytes


@deprecated("Use NIXLMetaServer instead")
class NIXLStorageServer:
    """
    NIXL initiator (server): holds the state dict, registers tensors, and notifies all clients with its descs.
    """
    def __init__(self, server_name: str, server_ip: str, server_port: int = 23456, cuda: int = -1):
        self.server_name = server_name
        self.server_ip = server_ip
        self.server_port = server_port
        self.cuda = cuda
        self.state_dict: Dict[str, torch.Tensor] = {}
        self.tensor_descs: Dict[str, NIXLTensorDescInfo] = {}
        self.agent = nixl_agent(
            self.server_name,
            nixl_agent_config(True, True, self.server_port)
        )
        self.client_infos: Set[str] = set()
        self._init_device()

    def _init_device(self):
        if self.cuda >= 0:
            torch.set_default_device(f"cuda:{self.cuda}")
        else:
            torch.set_default_device("cpu")

    def register_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """
        Register each tensor in the state_dict with NIXL. Build key->desc mapping.
        """
        self.state_dict = state_dict
        for key, tensor in state_dict.items():
            desc = self.agent.register_memory([tensor])
            if not desc:
                raise RuntimeError(f"Memory registration failed for key {key}.")
            desc_bytes = self.agent.get_serialized_descs(desc)
            self.tensor_descs[key] = NIXLTensorDescInfo(desc_bytes_list=[desc_bytes], shard_dim=-1, shard_mesh=1, shard_indices=[0])

    def wait_for_client_infos(self, expected_clients: int = 1, timeout: float = 60.0):
        """
        Wait for all clients to connect and synchronize metadata.
        """
        start = time.time()
        while len(self.client_infos) < expected_clients:
            notifs = self.agent.get_new_notifs()
            for client_name in notifs:
                self.client_infos.add(client_name)
            if time.time() - start > timeout:
                raise TimeoutError("Timeout waiting for clients.")
            time.sleep(0.1)

    def get_serialized_descs(self) -> Dict[str, bytes]:
        """
        Return a dict mapping key to serialized desc for all tensors.
        """
        return {k: info.desc for k, info in self.tensor_descs.items()}

    def notify_all_client_infos(self):
        """
        Notify all connected clients with the server's descs (key->serialized_desc dict as bytes).
        """
        # Use ClientInfo to serialize
        for client in self.client_infos:
            info = NIXLClientInfo(
                name=self.server_name,
                type=NIXLClientType.PS,
                descs=self.tensor_descs,
                meta=self.agent.get_agent_metadata()
            )
            self.agent.send_notif(client, info.serialize())

    def shutdown(self):
        [self.agent.remove_remote_agent(client) for client in self.client_infos]
        self.agent.invalidate_local_metadata(self.server_ip, self.server_port)
        for info in self.tensor_descs.values():
            self.agent.deregister_memory(info.get_desc(self.agent, 0))


class NIXLMetaServer:
    """
    NIXL meta server: only stores client meta and desc info, not state dict.
    """
    def __init__(self, server_name: str, nixl_config: DictConfig):
        self.server_name = server_name
        self.server_ip = nixl_config.server_ip
        self.server_port = nixl_config.server_port
        self.agent = nixl_agent(
            self.server_name,
            nixl_agent_config(True, True, self.server_port)
        )
        self.client_sharding_dicts: Dict[str, Dict[str, NIXLSharding]] = {}
        self.client_infos: Dict[str, NIXLClientInfo] = {}
        
        self.client_unified_sharding_dicts: Dict[str, Dict[str, NIXLSharding]] = {}
        self.comm_plan: Optional[NIXLCommPlan] = None
        self._client_temp_mappings: Dict[str, Dict] = {}
        
        self._is_all_client_shardings_recved = False
        self._is_all_client_infos_recved = False
        self._is_all_temp_mappings_recved = False
        
    def wait_for_client_shardings(self, expected_clients: int = 1, timeout: float = 60.0):
        """
        Wait for all clients to connect and send sharding.
        """
        psrl_logger.info(f"Waiting for {expected_clients} clients to connect and send sharding...")
        if self._is_all_client_shardings_recved:
            # TODO: support elastic adding new clients after all clients are connected
            assert len(self.client_sharding_dicts) == expected_clients, f"Expected {expected_clients} clients, but got {len(self.client_sharding_dicts)}"
            return True
        start = time.time()
        while len(self.client_sharding_dicts) < expected_clients:
            notifs = self.agent.get_new_notifs()
            for client_name, notif_list in notifs.items():
                for notif in notif_list:
                    try:
                        self.client_sharding_dicts[client_name] = pickle.loads(notif)
                    except Exception as e:
                        continue
            if time.time() - start > timeout:
                raise TimeoutError("Timeout waiting for clients.")
            time.sleep(0.1)
        self._is_all_client_shardings_recved = True
        psrl_logger.info(f"All {len(self.client_sharding_dicts)} clients sent sharding after {time.time() - start} seconds.")

    def wait_for_client_infos(self, expected_clients: int = 1, timeout: float = 60.0):
        """
        Wait for all clients to connect and send client infos.
        """
        psrl_logger.info(f"Waiting for {expected_clients} clients to send client infos...")
        if self._is_all_client_infos_recved:
            # TODO: support elastic adding new clients after all clients are connected
            assert len(self.client_infos) == expected_clients, f"Expected {expected_clients} clients, but got {len(self.client_infos)}"
            return True
        start = time.time()
        while len(self.client_infos) < expected_clients:
            notifs = self.agent.get_new_notifs()
            for client_name, notif_list in notifs.items():
                for notif in notif_list:
                    try:
                        self.client_infos[client_name] = NIXLClientInfo.deserialize(notif)
                    except Exception as e:
                        continue
            if time.time() - start > timeout:
                raise TimeoutError("Timeout waiting for clients.")
            time.sleep(0.1)
        self._is_all_client_infos_recved = True
        psrl_logger.info(f"All {len(self.client_infos)} clients sent client infos after {time.time() - start} seconds.")
    
    def wait_for_client_temp_mappings(self, expected_clients: int = 1, timeout: float = 60.0):
        """
        Wait for all clients to send temporary mappings.
        """
        psrl_logger.info(f"Waiting for {expected_clients} clients to send temp mappings...")
        if self._is_all_temp_mappings_recved:
            assert len(self._client_temp_mappings) == expected_clients, f"Expected {expected_clients} clients, but got {len(self._client_temp_mappings)}"
            return True
        start = time.time()
        while len(self._client_temp_mappings) < expected_clients:
            notifs = self.agent.get_new_notifs()
            for client_name, notif_list in notifs.items():
                for notif in notif_list:
                    try:
                        self._client_temp_mappings[client_name] = pickle.loads(notif)
                    except Exception as e:
                        continue
            if time.time() - start > timeout:
                raise TimeoutError("Timeout waiting for clients temp mappings.")
            time.sleep(0.1)
        self._is_all_temp_mappings_recved = True
        psrl_logger.info(f"All {len(self._client_temp_mappings)} clients sent temp mappings after {time.time() - start} seconds.")
    
    def make_unified_sharding(self):
        """
        Make unified sharding for all clients.
        """
        assert self._is_all_client_shardings_recved, "Not all clients sent sharding yet."
        assert not self.client_unified_sharding_dicts, "Unified sharding already made."
        # We first need to guarantee that all client shardings have the same keys
        all_keys = set()
        for client_name, sharding_dict in self.client_sharding_dicts.items():
            all_keys.update(sharding_dict.keys())
        # Then we can make the unified sharding for each client
        # That is, for each key, we need to find the new representation of (shard_dim, shard_mesh, shard_indices) for the mutual slice of all clients
        for key in all_keys:
            shard_mesh_list = []
            for client_name, sharding_dict in self.client_sharding_dicts.items():
                if key not in sharding_dict:
                    raise RuntimeError(f"Key {key} not found in sharding of client {client_name}.")
                shard_mesh_list.append(sharding_dict[key].shard_mesh)
            finest_shard_mesh = NIXLSharding.find_finest_shard_mesh(shard_mesh_list)
            for client_name, sharding_dict in self.client_sharding_dicts.items():
                if client_name not in self.client_unified_sharding_dicts:
                    self.client_unified_sharding_dicts[client_name] = {}
                self.client_unified_sharding_dicts[client_name][key] = deepcopy(sharding_dict[key])
                self.client_unified_sharding_dicts[client_name][key].refactor_based_on_finer_shard_mesh(finest_shard_mesh)
    
    def make_comm_plan(self):
        """
        Make communication plan for all clients.
        """
        assert self._is_all_client_infos_recved, "Not all clients sent client infos yet."
        assert not self.comm_plan, "Communication plan already made."
        
        psrl_logger.info("Making communication plan...")
        start = time.time()
        self.comm_plan = CommunicationPlanner().make_comm_plan(self.client_infos)
        psrl_logger.info(f"Communication plan made after {time.time() - start} seconds.")
        
    def notify_all_client_shardings(self):
        """
        Notify all connected clients with their sharding.
        """
        assert self._is_all_client_shardings_recved, "Not all clients sent sharding yet."
        assert self.client_unified_sharding_dicts, "Unified sharding not made yet." 
        for client_name, sharding_dict in self.client_unified_sharding_dicts.items():
            payload = pickle.dumps(sharding_dict)
            self.agent.send_notif(client_name, payload)
        
    def notify_all_client_infos_and_comm_plan(self):
        """
        Notify all connected clients with all client infos and optional comm plan.
        """
        assert self._is_all_client_infos_recved, "Not all clients sent client infos yet."
        assert self.comm_plan, "Communication plan not made yet."
        # Prepare notification data
        notification_data = {
            'client_infos': {client_name: self.client_infos[client_name].serialize() for client_name in self.client_infos},
            'comm_plan': self.comm_plan.serialize() if self.comm_plan else None
        }
        payload = pickle.dumps(notification_data)
        
        for client in self.client_infos:
            # Send notification with client infos and optional comm plan
            self.agent.send_notif(client, payload)
            
    def notify_all_client_temp_mappings(self):
        """
        Notify all connected clients with all temp mappings.
        """
        assert self._is_all_temp_mappings_recved, "Not all clients sent temp mappings yet."
        # Prepare notification data with all clients' temp mappings
        payload = pickle.dumps(self._client_temp_mappings)
        for client in self.client_infos:
            # Send notification with all temp mappings
            self.agent.send_notif(client, payload)

    def shutdown(self):
        """
        Shutdown the meta server.
        """
        for client in self.client_infos:
            self.agent.remove_remote_agent(client)
        self.agent.invalidate_local_metadata(self.server_ip, self.server_port)


class NIXLStorageClient:
    """
    NIXL target (client): supports both storage_server and meta_server mode.
    In meta_server mode, can connect to other clients for direct read/write.
    """
    def __init__(
        self, 
        client_name: str, 
        server_name: str, 
        use_gpu: bool, 
        client_type: NIXLClientType,
        nixl_config: DictConfig,
        nixl_interface: NIXLInterface = NIXLInterface()
    ):
        self.client_name = client_name
        self.server_name = server_name
        if use_gpu:
            assert torch.cuda.is_available(), "CUDA is not available."
        self.device = torch.device("cuda" if use_gpu else "cpu")
        self.client_type = client_type
        self.mode = nixl_config.server_mode  # "storage_server" or "meta_server"
        self.server_ip = nixl_config.server_ip
        self.server_port = nixl_config.server_port
        self.max_pinned_temp_memory_slots = nixl_config.max_pinned_temp_memory_slots # None means no pinned temp memory
        self.nixl_interface = nixl_interface
        
        self.client_port = 0 if self.nixl_interface.port_scanner is None else \
            ray.get(self.nixl_interface.port_scanner.find_free_port.remote(host=get_worker_info()[0]))
        self.agent = nixl_agent(
            self.client_name,
            nixl_agent_config(True, True, self.client_port)
        )
        self.local_client_info: Optional[NIXLClientInfo] = None
        self.xfer_handles: Dict[tuple, Any] = {}  # (key, tag, op_type) or (key, tag, op_type, target_client) -> handle
        self._is_connected = False
        
        # Original tensor mapping for contiguous tensors
        self._original_tensor_mapping: Dict[Tuple[str, Tuple[int, ...]], torch.Tensor] = {}  # (key, shard_idx) -> original_tensor
        # Temporary memory management for non-contiguous tensors
        self._temp_pinned_idx_mapping: Dict[Tuple[str, Tuple[int, ...]], int] = {}  # (key, shard_idx) -> pinned_idx
        self._temp_tensor_mapping: Dict[Tuple[str, Tuple[int, ...]], torch.Tensor] = {}  # (key, shard_idx) -> contiguous_tensor
        self._temp_desc_mapping: Dict[Tuple[str, Tuple[int, ...]], bytes] = {}  # (key, shard_idx) -> desc_bytes
        self._temp_meta_mapping: Dict[Tuple[str, Tuple[int, ...]], NIXLShardMetaInfo] = {}  # (key, shard_idx) -> meta_info
        self._pinned_slot_running_xfer: Dict[int, tuple] = {} # pinned_idx -> (key, tag, op_type, target_client)
        
        # Deprecated: storage_server mode
        self.server_client_info: Optional[NIXLClientInfo] = None
        self._storage_server_infos_fetched = False
        # meta_server mode
        self._unified_sharding_dict: Optional[Dict[str, NIXLSharding]] = None  # key -> sharding
        self._unified_sharding_dict_fetched = False
        self._all_client_infos: Dict[str, NIXLClientInfo] = {}  # name -> ClientInfo
        self._all_client_infos_fetched = False
        self._comm_plan: Optional[NIXLCommPlan] = None  # Communication plan
        self._all_temp_mappings: Dict[str, Dict[Tuple[str, int], bytes]] = {}  # client_name -> temp_desc_mapping
        self._all_temp_mappings_fetched = False

    def release_temp_memory(self):
        """Release all temporary memory and deregister descriptors"""
        for (key, shard_idx), desc_bytes in self._temp_desc_mapping.items():
            # Deregister the descriptor
            desc = self.agent.deserialize_descs(desc_bytes)
            self.agent.deregister_memory(desc)
        
        # Clear all temporary mappings
        self._temp_tensor_mapping.clear()
        self._temp_desc_mapping.clear()
        self._temp_meta_mapping.clear()

    def reallocate_temp_memory(self):
        """Reallocate temporary memory for non-contiguous shards"""
        assert self.local_client_info is not None, "Local client info not registered."
        assert self.max_pinned_temp_memory_slots is None, "temporary memory reallocation is forbidden if pinned temp memory is enabled"
        
        for key, tensor_info in self.local_client_info.descs.items():
            for idx, meta_info in enumerate(tensor_info.shard_meta_infos):
                if not meta_info.is_contiguous:
                    # Recreate temporary contiguous tensor
                    contiguous_tensor = torch.empty(
                        meta_info.shape, 
                        dtype=meta_info.dtype,
                        device=meta_info.device,
                        memory_format=torch.contiguous_format
                    )
                    
                    # Register the new contiguous tensor
                    desc = self.agent.register_memory([contiguous_tensor])
                    if not desc:
                        raise RuntimeError(f"Memory registration failed for key {key} shard {idx} (realloc).")
                    desc_bytes = self.agent.get_serialized_descs(desc)
                    
                    # Store temporary mappings
                    shard_idx = tensor_info.sharding.shard_indices[idx]
                    self._temp_tensor_mapping[(key, shard_idx)] = contiguous_tensor
                    self._temp_desc_mapping[(key, shard_idx)] = desc_bytes
                    self._temp_meta_mapping[(key, shard_idx)] = meta_info
        
    def _get_local_original_tensor(self, key: str, shard_idx: Tuple[int, ...]) -> Optional[torch.Tensor]:
        """Get original tensor mapping for non-contiguous shard"""
        assert self.mode == "meta_server", "get_local_original_tensor only valid in meta_server mode"
        return self._original_tensor_mapping.get((key, shard_idx))
        
    def _get_local_temp_tensor(self, key: str, shard_idx: Tuple[int, ...]) -> Optional[torch.Tensor]:
        """Get temporary tensor mapping for non-contiguous shard"""
        assert self.mode == "meta_server", "get_local_temp_tensor only valid in meta_server mode"
        return self._temp_tensor_mapping.get((key, shard_idx))
    
    def _get_temp_desc(self, client_name: str, key: str, shard_idx: Tuple[int, ...]) -> Optional[bytes]:
        """Get temporary descriptor for non-contiguous shard"""
        assert self.mode == "meta_server", "get_temp_desc only valid in meta_server mode"
        assert self._all_temp_mappings_fetched, "All temp mappings not fetched yet."
        if client_name in self._all_temp_mappings:
            return self._all_temp_mappings[client_name].get((key, shard_idx), None)
        return None

    def register_local_tensors(self, state_dict: Dict[str, torch.Tensor], sharding_dict: Dict[str, NIXLSharding] = {}):
        """
        Register local tensors with NIXL. Build key->desc mapping.
        Args:
            state_dict: {key: torch.Tensor}
            sharding_dict: {key: NIXLSharding}
        """
        # If pinned temp memory is enabled, we need to first scan the state_dict and find all the tensors that are not contiguous
        # Then we need to find the largest uncontiguous tensor and allocate max_pinned_temp_memory_slots times of its size as pinned memory (a tensor like this: [max_pinned_temp_memory_slots, *])
        # Then we enumerate the uncontiguous tensors again and map them with the pinned memory in a round-robin manner (the first uncontiguous tensor map to [0, *], the second to [1, *], the (max_pinned_temp_memory_slots+1)-th to [0, *] again, etc.)
        # We should record a mapping from the uncontiguous tensor to the index of the pinned memory
        _uncontiguous_tensor_list = []
        if self.max_pinned_temp_memory_slots is not None:
            # Scan the state_dict and find all the tensors that are not contiguous
            for key, tensor in state_dict.items():
                assert key in sharding_dict, f"Key {key} not found in sharding_dict."
                if tensor.device == torch.device("meta"):
                    continue
                sharding = sharding_dict[key]
                local_sharded_tensors = sharding.get_local_sharded_tensors(tensor)
                for local_sharded_tensor in local_sharded_tensors:
                    if not local_sharded_tensor.is_contiguous():
                        _uncontiguous_tensor_list.append(local_sharded_tensor)
            # Find the largest uncontiguous tensor and allocate pinned memory for it
            largest_uncontiguous_tensor = max(_uncontiguous_tensor_list, key=lambda x: x.numel() * x.element_size())
            self._pinned_memory = torch.empty(self.max_pinned_temp_memory_slots, largest_uncontiguous_tensor.numel(), dtype=largest_uncontiguous_tensor.dtype, device=largest_uncontiguous_tensor.device)
        
        descs = {}
        uncontiguous_tensor_pinned_idx = 0
        for key, tensor in state_dict.items():
            assert key in sharding_dict, f"Key {key} not found in sharding_dict."
            sharding = sharding_dict[key]
            shard_indices = sharding.shard_indices
            # assert sharding.is_contiguous_sharding(), "Only contiguous sharding is supported for now."
            # Split registration
            desc_bytes_list = []
            shard_meta_info_list = []
            local_sharded_tensors = sharding.get_local_sharded_tensors(tensor)
            
            for local_pos, local_sharded_tensor in enumerate(local_sharded_tensors):
                # Store the original tensor mapping
                # If the tensor is on meta device, allocate on-the-fly
                if local_sharded_tensor.device == torch.device("meta"):
                    local_sharded_tensor = torch.empty_like(local_sharded_tensor, device=self.device)
                self._original_tensor_mapping[(key, shard_indices[local_pos])] = local_sharded_tensor
                
                # Create meta info for non-contiguous shard
                is_contiguous = local_sharded_tensor.is_contiguous()
                meta_info = NIXLShardMetaInfo(
                    dtype=local_sharded_tensor.dtype,
                    device=local_sharded_tensor.device,
                    shape=local_sharded_tensor.shape,
                    stride=local_sharded_tensor.stride(),
                    is_contiguous=is_contiguous
                )
                shard_meta_info_list.append(meta_info)
                    
                # Check if the shard is contiguous
                if local_sharded_tensor.is_contiguous():
                    # Contiguous shard: register directly
                    desc = self.agent.register_memory([local_sharded_tensor])
                    if not desc:
                        raise RuntimeError(f"Memory registration failed for key {key} shard {shard_indices[local_pos]}.")
                    desc_bytes = self.agent.get_serialized_descs(desc)
                    desc_bytes_list.append(desc_bytes)
                else:
                    if self.max_pinned_temp_memory_slots is None:
                        # Non-contiguous shard: create temporary contiguous memory
                        # Create a new contiguous tensor with the same shape and dtype
                        contiguous_tensor = torch.empty_like(local_sharded_tensor)
                    else:
                        # Non-contiguous shard: map to pinned memory
                        pinned_slot = self._pinned_memory[uncontiguous_tensor_pinned_idx]
                        required_bytes = local_sharded_tensor.numel() * local_sharded_tensor.element_size()
                        available_bytes = pinned_slot.numel() * pinned_slot.element_size()
                        assert available_bytes >= required_bytes, f"Pinned slot {uncontiguous_tensor_pinned_idx} has {available_bytes} bytes, but the {key} shard {shard_indices[local_pos]} requires {required_bytes} bytes."
                        # Make a view of the pinned slot
                        # We use the base_addr of pinned_slot as the base_addr of the contiguous tensor and the shape/dtype of the contiguous tensor is the same as the local_sharded_tensor
                        contiguous_tensor = torch.frombuffer(
                            pinned_slot, 
                            dtype=local_sharded_tensor.dtype,
                            count=local_sharded_tensor.numel()
                        ).reshape(local_sharded_tensor.shape)
                        self._temp_pinned_idx_mapping[(key, shard_indices[local_pos])] = uncontiguous_tensor_pinned_idx
                        uncontiguous_tensor_pinned_idx = (uncontiguous_tensor_pinned_idx + 1) % self.max_pinned_temp_memory_slots
                     
                    # Register the contiguous tensor   
                    desc = self.agent.register_memory([contiguous_tensor])
                    if not desc:
                        raise RuntimeError(f"Memory registration failed for key {key} shard {shard_indices[local_pos]} (contiguous temp).")
                    desc_bytes = self.agent.get_serialized_descs(desc)
                    # Store None in desc_bytes_list to indicate this shard uses temp memory
                    desc_bytes_list.append(desc_bytes)
                    # Store temporary mappings
                    self._temp_tensor_mapping[(key, shard_indices[local_pos])] = contiguous_tensor
                    self._temp_desc_mapping[(key, shard_indices[local_pos])] = desc_bytes
                    self._temp_meta_mapping[(key, shard_indices[local_pos])] = meta_info
            
            # Create the tensor descriptor info
            descs[key] = NIXLTensorDescInfo(
                desc_bytes_list=desc_bytes_list,
                sharding=sharding,
                shard_meta_infos=shard_meta_info_list
            )
            
        # Create the client info
        self.local_client_info = NIXLClientInfo(
            name=self.client_name,
            ip=get_local_ip(),
            gpu_id=get_local_gpu_id(),
            type=self.client_type,
            descs=descs,
            meta=self.agent.get_agent_metadata()
        )
        
    def connect_to_server(self, timeout: float = 60.0):
        """
        Connect to the storage/meta server.
        """
        assert not self._is_connected, "Already connected to server"
        self.agent.fetch_remote_metadata(self.server_name, self.server_ip, self.server_port)
        self.agent.send_local_metadata(self.server_ip, self.server_port)
        start = time.time()
        ready = False
        while not ready:
            ready = self.agent.check_remote_metadata(self.server_name)
            if time.time() - start > timeout:
                raise TimeoutError("Timeout waiting for server metadata to be fetched and connected.")
            time.sleep(0.1)
        self._is_connected = True
        
    def send_local_sharding(self, sharding_dict: Dict[str, NIXLSharding]):
        """
        Send local sharding to the server.
        """
        assert self.mode == "meta_server", "send_local_sharding only valid in meta_server mode"
        assert self._is_connected, "Not connected to server"
        self.agent.send_notif(self.server_name, pickle.dumps(sharding_dict))

    def send_local_info(self):
        """
        Send local client info to the server.
        For storage_server mode, notify the server that the client is ready.
        For meta_server mode, send the local client info to the server.
        """
        assert self._is_connected, "Not connected to server"
        if self.mode == "storage_server":
            self.agent.send_notif(self.server_name, b"client_ready")
        elif self.mode == "meta_server":
            # Send ClientInfo to meta server
            if self.local_client_info is None:
                raise RuntimeError("Local client info not registered.")
            self.agent.send_notif(self.server_name, self.local_client_info.serialize())
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
    def send_local_temp_mapping(self):
        """Send local temporary mappings to the server"""
        assert self.mode == "meta_server", "send_local_temp_mapping only valid in meta_server mode"
        assert self._is_connected, "Not connected to server"
        self.agent.send_notif(self.server_name, pickle.dumps(self._temp_desc_mapping))
        
    def wait_for_server_sharding(self, timeout: float = 60.0):
        """
        Wait for the server sharding to be fetched.
        """
        assert self.mode == "meta_server", "wait_for_server_sharding only valid in meta_server mode"
        assert self._is_connected, "Not connected to server"
        if self._unified_sharding_dict_fetched:
            return
        start = time.time()
        while True:
            notifs = self.agent.get_new_notifs()
            if self.server_name in notifs and notifs[self.server_name]:
                self._unified_sharding_dict = pickle.loads(notifs[self.server_name][0])
                break
            if time.time() - start > timeout:
                raise TimeoutError("Timeout waiting for server sharding notification.")
            time.sleep(0.1)
        self._unified_sharding_dict_fetched = True
        return self._unified_sharding_dict
        
    def wait_for_server_info(self, timeout: float = 60.0):
        """
        Wait for the server info to be fetched.
        For storage_server mode, wait for the storage server info to be fetched.
        For meta_server mode, wait for all client infos (stored in the server) to be fetched.
        """
        assert self._is_connected, "Not connected to server"
        if self.mode == "storage_server":
            if self._storage_server_infos_fetched:
                return
            start = time.time()
            while True:
                notifs = self.agent.get_new_notifs()
                if self.server_name in notifs and notifs[self.server_name]:
                    info_bytes = notifs[self.server_name][0]
                    # Deserialize the storage server info
                    self.server_client_info = NIXLClientInfo.deserialize(info_bytes)
                    self._storage_server_infos_fetched = True
                    break
                if time.time() - start > timeout:
                    raise TimeoutError("Timeout waiting for server descs notification.")
                time.sleep(0.1)
        elif self.mode == "meta_server":
            # Wait for all client infos (stored in the server) to be fetched
            if self._all_client_infos_fetched:
                return
            start = time.time()
            while True:
                notifs = self.agent.get_new_notifs()
                if self.server_name in notifs and notifs[self.server_name]:
                    notification_bytes = notifs[self.server_name][0]
                    notification_data = pickle.loads(notification_bytes)
                    # Process client infos
                    if isinstance(notification_data, dict) and 'client_infos' in notification_data:
                        # New format: includes communication plan
                        all_client_infos = notification_data['client_infos']
                        for client_name, info_bytes in all_client_infos.items():
                            info = NIXLClientInfo.deserialize(info_bytes)
                            self._all_client_infos[client_name] = info
                        # Process communication plan
                        if notification_data.get('comm_plan'):
                            self._comm_plan = NIXLCommPlan.deserialize(notification_data['comm_plan'])
                        else:
                            self._comm_plan = None
                    else:
                        # Old format: only client infos
                        all_client_infos = notification_data
                        for client_name, info_bytes in all_client_infos.items():
                            info = NIXLClientInfo.deserialize(info_bytes)
                            self._all_client_infos[client_name] = info
                            self._comm_plan = None
                    break
                if time.time() - start > timeout:
                    raise TimeoutError("Timeout waiting for meta server client infos.")
                time.sleep(0.1)
            self._all_client_infos_fetched = True
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def wait_for_server_temp_mappings(self, timeout: float = 60.0):
        """Wait for the server temporary mappings to be fetched."""
        assert self.mode == "meta_server", "wait_for_server_temp_mappings only valid in meta_server mode"
        assert self._is_connected, "Not connected to server"
        if self._all_temp_mappings_fetched:
            return
        start = time.time()
        while True:
            notifs = self.agent.get_new_notifs()
            if self.server_name in notifs and notifs[self.server_name]:
                self._all_temp_mappings = pickle.loads(notifs[self.server_name][0])
                break
            if time.time() - start > timeout:
                raise TimeoutError("Timeout waiting for temp mappings.")
            time.sleep(0.1)
        self._all_temp_mappings_fetched = True

    # --- storage_server mode read/write  ---
    @deprecated("Use client_read instead")
    def read(self, key: str, tag: bytes):
        """
        Read from the storage server.
        """
        if self.mode != "storage_server":
            raise RuntimeError("read(key, tag) only valid in storage_server mode")
        self.wait_for_server_info()
        local_desc = self.local_client_info.get_tensor_desc(self.agent, 0).trim()
        server_desc = self.server_client_info.get_tensor_desc(self.agent, 0).trim()
        handle = self.agent.initialize_xfer(
            "READ", local_desc, server_desc, self.server_name, tag
        )
        if not handle:
            raise RuntimeError(f"Creating READ transfer failed for key {key}.")
        state = self.agent.transfer(handle)
        if state == "ERR":
            raise RuntimeError(f"Posting READ transfer failed for key {key}.")
        self.xfer_handles[(key, tag, "READ")] = handle

    @deprecated("Use client_write instead")
    def write(self, key: str, tag: bytes):
        """
        Write to the storage server.
        """
        if self.mode != "storage_server":
            raise RuntimeError("write(key, tag) only valid in storage_server mode")
        self.wait_for_server_info()
        local_desc = self.local_client_info.get_tensor_desc(self.agent, 0).trim()
        server_desc = self.server_client_info.get_tensor_desc(self.agent, 0).trim()
        handle = self.agent.initialize_xfer(
            "WRITE", local_desc, server_desc, self.server_name, tag
        )
        if not handle:
            raise RuntimeError(f"Creating WRITE transfer failed for key {key}.")
        state = self.agent.transfer(handle)
        if state == "ERR":
            raise RuntimeError(f"Posting WRITE transfer failed for key {key}.")
        self.xfer_handles[(key, tag, "WRITE")] = handle

    # --- meta_server mode: client-to-client read/write ---
    def _ensure_client_info_fetched(self, target_client: str):
        """Ensure connection to target client is established."""
        assert target_client in self._all_client_infos, f"Target client {target_client} not found in client infos."
        meta = self._all_client_infos[target_client].meta
        nixl_agent_name_bytes = self.agent.add_remote_agent(meta)
        assert nixl_agent_name_bytes.decode() == target_client, f"NIXL agent name {nixl_agent_name_bytes.decode()} does not match target client: {target_client}"

    def client_read(self, target_client: str, key: str, tag: bytes, comm_plan: Optional[NIXLCommPlan] = None):
        """Read from another client (meta_server mode), supports shard alignment and communication plan."""
        if self.mode != "meta_server":
            raise RuntimeError("client_read only valid in meta_server mode")
        plan = comm_plan or self._comm_plan
        self._ensure_client_info_fetched(target_client)
        remote_info = self._all_client_infos[target_client].get_tensor_desc_info(key)
        local_info = self.local_client_info.get_tensor_desc_info(key)
        shards_to_transfer = []
        if plan and self.client_type == NIXLClientType.PULL_SIDE:
            pull_plan = plan.get_pull_plan(self.client_name, key)
            if target_client in pull_plan:
                shards_to_transfer = pull_plan[target_client]
        else:
            # Default behavior: align shards
            for shard_idx in local_info.sharding.shard_indices:
                if shard_idx in remote_info.sharding.shard_indices:
                    shards_to_transfer.append(shard_idx)
        for shard_idx in shards_to_transfer:
            assert shard_idx in local_info.sharding.shard_indices and shard_idx in remote_info.sharding.shard_indices, \
                f"Shard {shard_idx} not found in local or remote shards for key {key}"
            local_pos = local_info.sharding.shard_indices.index(shard_idx)
            remote_pos = remote_info.sharding.shard_indices.index(shard_idx)
            
            # Get local descriptor (check if it's a temporary one)
            local_desc_bytes = local_info.desc_bytes_list[local_pos]
            if local_desc_bytes is None:
                # Use temporary descriptor for non-contiguous shard
                local_desc_bytes = self._get_temp_desc(self.client_name, key, shard_idx)
                if local_desc_bytes is None:
                    raise RuntimeError(f"No temporary descriptor found for key {key} shard {shard_idx}")
                # Wait for the pinned slot to be available
                if self.max_pinned_temp_memory_slots is not None:
                    pinned_idx = self._temp_pinned_idx_mapping[(key, shard_idx)]
                    if pinned_idx in self._pinned_slot_running_xfer:
                        key, tag, op_type, target_client = self._pinned_slot_running_xfer[pinned_idx]
                        start_time = time.time()
                        self.wait(key, tag, op_type, target_client=target_client)
                        end_time = time.time()
                        print(f"{self.client_name} read uncontiguous {(key, shard_idx)}, pinned slot {pinned_idx} is available after {end_time - start_time} seconds")
                    self._pinned_slot_running_xfer[pinned_idx] = (key, tag, "READ", target_client)  
            
            # Get remote descriptor (check if it's a temporary one)
            remote_desc_bytes = remote_info.desc_bytes_list[remote_pos]
            if remote_desc_bytes is None:
                raise NotImplementedError("Not implemented for non-contiguous shards on the remote side.")
                # Use temporary descriptor for non-contiguous shard
                remote_desc_bytes = self._get_temp_desc(target_client, key, remote_pos)
                if remote_desc_bytes is None:
                    raise RuntimeError(f"No remote temporary descriptor found for key {key} shard {shard_idx}")
            
            local_desc = self.agent.deserialize_descs(local_desc_bytes).trim()
            remote_desc = self.agent.deserialize_descs(remote_desc_bytes).trim()
            
            handle = self.agent.initialize_xfer(
                "READ", local_desc, remote_desc, target_client, make_shard_tag(tag, shard_idx)
            )
            if not handle:
                raise RuntimeError(f"Creating client READ transfer failed for key {key} shard {shard_idx}.")
            state = self.agent.transfer(handle)
            if state == "ERR":
                raise RuntimeError(f"Posting client READ transfer failed for key {key} shard {shard_idx}.")
            self.xfer_handles[(key, make_shard_tag(tag, shard_idx), "READ", target_client)] = handle

    def client_write(self, target_client: str, key: str, tag: bytes, comm_plan: Optional[NIXLCommPlan] = None):
        """Write to another client (meta_server mode), supports shard alignment and communication plan."""
        if self.mode != "meta_server":
            raise RuntimeError("client_write only valid in meta_server mode")
        plan = comm_plan or self._comm_plan
        self._ensure_client_info_fetched(target_client)
        remote_info = self._all_client_infos[target_client].get_tensor_desc_info(key)
        local_info = self.local_client_info.get_tensor_desc_info(key)
        shards_to_transfer = []
        if plan and self.client_type == NIXLClientType.PUSH_SIDE:
            push_plan = plan.get_push_plan(self.client_name, key)
            if target_client in push_plan:
                shards_to_transfer = push_plan[target_client]
        else:
            # Default behavior: align shards
            for shard_idx in local_info.sharding.shard_indices:
                if shard_idx in remote_info.sharding.shard_indices:
                    shards_to_transfer.append(shard_idx)
        for shard_idx in shards_to_transfer:
            assert shard_idx in local_info.sharding.shard_indices and shard_idx in remote_info.sharding.shard_indices, \
                f"Shard {shard_idx} not found in local or remote shards for key {key}"
            local_pos = local_info.sharding.shard_indices.index(shard_idx)
            remote_pos = remote_info.sharding.shard_indices.index(shard_idx)
            
            # Check if local shard is non-contiguous and needs data copying
            local_desc_bytes = local_info.desc_bytes_list[local_pos]
            if local_desc_bytes is None:
                # Non-contiguous shard: copy data to temporary contiguous memory
                original_tensor = self._get_local_original_tensor(key, shard_idx)
                if original_tensor is None:
                    raise RuntimeError(f"No original tensor mapping found for key {key} shard {shard_idx}")
                contiguous_tensor = self._get_local_temp_tensor(key, shard_idx)
                if contiguous_tensor is None:
                    raise RuntimeError(f"No temporary tensor mapping found for key {key} shard {shard_idx}")
                # Wait for the pinned slot to be available
                if self.max_pinned_temp_memory_slots is not None:
                    pinned_idx = self._temp_pinned_idx_mapping[(key, shard_idx)]
                    if pinned_idx in self._pinned_slot_running_xfer:
                        key, tag, op_type, target_client = self._pinned_slot_running_xfer[pinned_idx]
                        start_time = time.time()
                        self.wait(key, tag, op_type, target_client=target_client)
                        end_time = time.time()
                        print(f"{self.client_name} write uncontiguous {(key, shard_idx)}, pinned slot {pinned_idx} is available after {end_time - start_time} seconds")
                    self._pinned_slot_running_xfer[pinned_idx] = (key, tag, "WRITE", target_client)  
                # Copy data from original non-contiguous tensor to temporary contiguous tensor
                contiguous_tensor.copy_(original_tensor)
                
                # Use temporary descriptor
                local_desc_bytes = self._get_temp_desc(self.client_name, key, shard_idx)
                if local_desc_bytes is None:
                    raise RuntimeError(f"No temporary descriptor found for key {key} shard {shard_idx} in {self.client_name}'s temp descs")
            
            # Get remote descriptor (check if it's a temporary one)
            remote_desc_bytes = remote_info.desc_bytes_list[remote_pos]
            if remote_desc_bytes is None:
                raise NotImplementedError("Not implemented for non-contiguous shards on the remote side.")
                # Use temporary descriptor for non-contiguous shard
                remote_desc_bytes = self._get_temp_desc(target_client, key, shard_idx)
                if remote_desc_bytes is None:
                    raise RuntimeError(f"No remote temporary descriptor found for key {key} shard {shard_idx}")
            
            assert local_info.get_shard_size_bytes(local_pos) == remote_info.get_shard_size_bytes(remote_pos), \
                f"Shard size mismatch for key {key} shard {shard_idx}: {local_info.get_shard_size_bytes(local_pos)} != {remote_info.get_shard_size_bytes(remote_pos)}"
            
            local_desc = self.agent.deserialize_descs(local_desc_bytes).trim()
            remote_desc = self.agent.deserialize_descs(remote_desc_bytes).trim()
            
            handle = self.agent.initialize_xfer(
                "WRITE", local_desc, remote_desc, target_client, make_shard_tag(tag, shard_idx)
            )
            if not handle:
                raise RuntimeError(f"Creating client WRITE transfer failed for key {key} shard {shard_idx}.")
            state = self.agent.transfer(handle)
            if state == "ERR":
                raise RuntimeError(f"Posting client WRITE transfer failed for key {key} shard {shard_idx}.")
            self.xfer_handles[(key, make_shard_tag(tag, shard_idx), "WRITE", target_client)] = handle

    def wait(self, key: str, tag: bytes, op_type: str, target_client: Optional[str] = None, timeout: float = 60.0):
        """
        Wait for a transfer to be completed.
        """
        if self.mode == "storage_server":
            handle = self.xfer_handles.get((key, tag, op_type))
            if handle is None:
                raise RuntimeError(f"No handle for ({key}, {tag}, {op_type})")
            start = time.time()
            while True:
                state = self.agent.check_xfer_state(handle)
                if state == "ERR":
                    raise RuntimeError(f"Transfer error for ({key}, {tag}, {op_type})")
                elif state == "DONE":
                    break
                if time.time() - start > timeout:
                    raise TimeoutError(f"Timeout waiting for transfer ({key}, {tag}, {op_type})")
                time.sleep(0.05)
        elif self.mode == "meta_server":
            # Shard tag, wait for all shards
            info = self.local_client_info.get_tensor_desc_info(key)
            for shard_idx in info.sharding.shard_indices:
                handle = self.xfer_handles.get((key, make_shard_tag(tag, shard_idx), op_type, target_client))
                if handle is None:
                    continue  # This shard did not do transfer
                start = time.time()
                while True:
                    state = self.agent.check_xfer_state(handle)
                    if state == "ERR":
                        raise RuntimeError(f"Transfer error for ({key}, {tag}, {op_type}, shard {shard_idx})")
                    elif state == "DONE":
                        # For non-contiguous shards, sync data back to original tensor after READ
                        if op_type == "READ":
                            local_pos = info.sharding.shard_indices.index(shard_idx)
                            if info.desc_bytes_list[local_pos] is None:
                                # Non-contiguous shard: copy data from temporary to original
                                original_tensor = self._get_local_original_tensor(key, shard_idx)
                                if original_tensor is None:
                                    raise RuntimeError(f"No original tensor mapping found for key {key} shard {shard_idx}")
                                contiguous_tensor = self._get_local_temp_tensor(key, shard_idx)
                                if contiguous_tensor is None:
                                    raise RuntimeError(f"No temporary tensor mapping found for key {key} shard {shard_idx}")
                                # Copy data from temporary contiguous tensor back to original non-contiguous tensor
                                original_tensor.copy_(contiguous_tensor)
                        
                        self.xfer_handles.pop((key, make_shard_tag(tag, shard_idx), op_type, target_client))
                        break
                    if time.time() - start > timeout:
                        raise TimeoutError(f"Timeout waiting for transfer ({key}, {tag}, {op_type}, shard {shard_idx})")
                    time.sleep(0.05)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def shutdown(self):
        # Release temporary memory first
        self.release_temp_memory()
        
        if self.local_client_info:
            for info in self.local_client_info.descs.values():
                for local_pos in range(info.num_local_shards):
                    # Only deregister if the descriptor is not None (not a temporary one)
                    if info.desc_bytes_list[local_pos] is not None:
                        self.agent.deregister_memory(info.get_desc(self.agent, local_pos))
