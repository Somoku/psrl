import logging
import os
import pickle
import time
from typing import Any

import nixl._bindings as nixlBind
import ray
import torch
from nixl._api import nixl_agent, nixl_agent_config
from omegaconf import DictConfig

from psrl.utils.logger import deprecated, get_worker_info
from psrl.utils.nixl.comm_plan import NIXLCommPlan
from psrl.utils.nixl.network_topology import get_local_gpu_id, get_local_ip
from psrl.utils.nixl.nixl_spec import (
    NIXLClientInfo,
    NIXLClientType,
    NIXLInterface,
    NIXLSharding,
    NIXLShardMetaInfo,
    NIXLTensorInfo,
)

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))


# Utility function to combine tag, shard_idx, src_client, target_client, key
def make_xfer_tag(
    tag: str,
    src_client: str,
    target_client: str,
    key: str,
    shard_idx: tuple[int, ...] | None = None,
) -> bytes:
    """
    Combine tag, shard_idx, src_client, target_client, and key into a unique bytes object as message identifier.
    Uses pickle to ensure uniqueness and handle various data types safely.
    """
    # Create a tuple containing all components to ensure uniqueness
    if not shard_idx:
        components = (tag, src_client, target_client, key)
    else:
        components = (tag, src_client, target_client, key, shard_idx)

    # Use pickle to serialize the tuple, ensuring unique representation
    return pickle.dumps(components)


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
        nixl_interface: NIXLInterface | None = None,
        binded_agent: nixl_agent | None = None,
        client_group_id: int = -1,  # -1 is the default client group
        logging_path: str | None = None,
    ):
        self.client_name = client_name
        self.server_name = server_name
        if use_gpu:
            assert torch.cuda.is_available(), "CUDA is not available."
        # self.device = torch.device("cuda:0" if use_gpu else "cpu")
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}" if use_gpu else "cpu")
        self.client_type = client_type
        self.mode = nixl_config.server_mode  # "storage_server" or "meta_server"
        self.server_ip = nixl_config.server_ip
        self.server_port = nixl_config.server_port
        self.max_pinned_temp_memory_slots = (
            nixl_config.max_pinned_temp_memory_slots
        )  # None means no pinned temp memory
        self.nixl_interface = nixl_interface if nixl_interface is not None else NIXLInterface()
        self.client_group_id = client_group_id

        # Initialize NIXL agent
        if binded_agent is None:
            self.client_port = (
                0
                if self.nixl_interface.port_scanner is None
                else ray.get(self.nixl_interface.port_scanner.find_free_port.remote(host=get_worker_info()[0]))
            )
            self.agent = nixl_agent(self.client_name, nixl_agent_config(True, True, self.client_port))
        else:
            self.agent = binded_agent

        self.local_client_info: NIXLClientInfo | None = None
        self.xfer_handles: dict[bytes, Any] = {}  # xfer_tag -> handle
        self._is_connected = False

        # Original tensor mapping for contiguous tensors
        self._original_tensor_mapping: dict[
            tuple[str, tuple[int, ...]], torch.Tensor
        ] = {}  # (key, shard_idx) -> original_tensor
        # Temporary memory management for non-contiguous tensors
        self._temp_tensor_mapping: dict[
            tuple[str, tuple[int, ...]], torch.Tensor
        ] = {}  # (key, shard_idx) -> contiguous_tensor
        self._temp_desc_bytes_mapping: dict[tuple[str, tuple[int, ...]], bytes] = {}  # (key, shard_idx) -> desc_bytes
        self._temp_meta_mapping: dict[
            tuple[str, tuple[int, ...]], NIXLShardMetaInfo
        ] = {}  # (key, shard_idx) -> meta_info
        # If we use pinned memory, we need to record the mapping
        # from the uncontiguous tensor to the index of the pinned memory
        self._temp_pinned_idx_mapping: dict[tuple[str, tuple[int, ...]], int] = {}  # (key, shard_idx) -> pinned_idx
        self._pinned_slot_running_xfer: dict[
            tuple[torch.Size, torch.dtype, int], tuple
        ] = {}  # (shape, dtype, pinned_idx) -> (key, tag, op_type, target_client)
        self._pinned_memory: dict[tuple[torch.Size, torch.dtype], list[tuple[torch.Tensor, bytes]]] | None = None
        self._contiguous_event_cache: dict[
            tuple[str, tuple[int, ...]], torch.cuda.streams.Event
        ] = {}  # (key, shard_idx) -> cudaEvent

        # Optimization: merge multiple contiguous transfers into one
        self._cached_xfer_descs = []  # [("READ", local_desc, remote_desc, target_agent, tag, target_client)]

        # Deprecated: storage_server mode
        self.server_client_info: NIXLClientInfo | None = None
        self._storage_server_infos_fetched = False
        # meta_server mode
        self._target_client_connected: dict[str, bool] = {}  # target_client -> connected
        self._unified_sharding_dict: dict[str, NIXLSharding] | None = None  # key -> sharding
        self._unified_sharding_dict_fetched = False
        self._all_client_infos: dict[str, NIXLClientInfo] = {}  # name -> ClientInfo
        self._all_client_infos_fetched = False
        self._comm_plan: NIXLCommPlan | None = None  # Communication plan
        self._all_temp_mappings: dict[str, dict[tuple[str, int], bytes]] = {}  # client_name -> temp_desc_mapping
        self._all_temp_mappings_fetched = False

        # logging
        """
        if logging_path is not None:
            self.log_prefix = "NIXLStorageClient_" + self.client_name
            psrl_logger.addHandler(DualOutputHandler(logging_path, self.log_prefix))
            psrl_logger.info(f"NIXLStorageClient {self.client_name} initialized.")
        """

    def release_temp_memory(self):
        """Release all temporary memory and deregister descriptors"""
        _cached_deregister_descs = []
        for _, desc_bytes in self._temp_desc_bytes_mapping.items():
            # Deregister the descriptor
            if desc_bytes in _cached_deregister_descs:
                continue
            _cached_deregister_descs.append(desc_bytes)
            try:
                desc = self.agent.deserialize_descs(desc_bytes)
                self.agent.deregister_memory(desc)
            except Exception as e:
                psrl_logger.warning(f"Failed to deregister descriptor {desc_bytes}: {e}")
                raise e

        # Clear all temporary mappings
        self._temp_tensor_mapping = {}
        self._temp_desc_bytes_mapping = {}
        self._temp_meta_mapping = {}

    def reallocate_temp_memory(self):
        """Reallocate temporary memory for non-contiguous shards"""
        assert self.local_client_info is not None, "Local client info not registered."
        assert self.max_pinned_temp_memory_slots is None, (
            "temporary memory reallocation is forbidden if pinned temp memory is enabled"
        )

        for key, tensor_info in self.local_client_info.tensor_infos.items():
            for idx, meta_info in enumerate(tensor_info.shard_meta_infos):
                if not meta_info.is_contiguous:
                    # Recreate temporary contiguous tensor
                    contiguous_tensor = torch.empty(
                        meta_info.shape,
                        dtype=meta_info.dtype,
                        device=self.device,
                        requires_grad=False,
                    )
                    contiguous_meta_info = NIXLShardMetaInfo(
                        dtype=contiguous_tensor.dtype,
                        device=contiguous_tensor.device,
                        shape=contiguous_tensor.shape,
                        stride=contiguous_tensor.stride,
                        is_contiguous=True,
                    )

                    # Register the new contiguous tensor
                    desc = self.agent.register_memory([contiguous_tensor])
                    if not desc:
                        raise RuntimeError(f"Memory registration failed for key {key} shard {idx} (realloc).")
                    desc_bytes = self.agent.get_serialized_descs(desc)

                    # Store temporary mappings
                    shard_idx = tensor_info.sharding.shard_indices[idx]
                    self._temp_tensor_mapping[(key, shard_idx)] = contiguous_tensor
                    self._temp_desc_bytes_mapping[(key, shard_idx)] = desc_bytes
                    self._temp_meta_mapping[(key, shard_idx)] = contiguous_meta_info

    def _get_local_original_tensor(self, key: str, shard_idx: tuple[int, ...]) -> torch.Tensor | None:
        """Get original tensor mapping for non-contiguous shard"""
        assert self.mode == "meta_server", "get_local_original_tensor only valid in meta_server mode"
        return self._original_tensor_mapping.get((key, shard_idx))

    def _get_local_temp_tensor(self, key: str, shard_idx: tuple[int, ...]) -> torch.Tensor | None:
        """Get temporary tensor mapping for non-contiguous shard"""
        assert self.mode == "meta_server", "get_local_temp_tensor only valid in meta_server mode"
        return self._temp_tensor_mapping.get((key, shard_idx))

    def _get_temp_desc_bytes(self, client_name: str, key: str, shard_idx: tuple[int, ...]) -> bytes | None:
        """Get temporary descriptor for non-contiguous shard"""
        assert self.mode == "meta_server", "get_temp_desc only valid in meta_server mode"
        assert self._all_temp_mappings_fetched, "All temp mappings not fetched yet."
        assert client_name in self._all_temp_mappings, f"Client {client_name} not found in temp mappings."
        # Currently, temp mappings are only used locally
        assert client_name == self.client_name, f"Client {client_name} is not the current client."
        return self._all_temp_mappings[client_name].get((key, shard_idx), None)

    def get_original_tensor_mapping(
        self,
    ) -> dict[tuple[str, tuple[int, ...]], torch.Tensor]:
        """Get original tensor mapping"""
        assert self.mode == "meta_server", "get_original_tensor_mapping only valid in meta_server mode"
        return self._original_tensor_mapping

    def get_temp_tensor_mapping(
        self,
    ) -> dict[tuple[str, tuple[int, ...]], torch.Tensor]:
        """Get temp tensor mapping"""
        assert self.mode == "meta_server", "get_temp_tensor_mapping only valid in meta_server mode"
        return self._temp_tensor_mapping

    def register_local_tensors(
        self,
        state_dict: dict[str, torch.Tensor],
        sharding_dict: dict[str, NIXLSharding] | None = None,
        binded_meta_tensor_mapping: (dict[tuple[str, tuple[int, ...]], torch.Tensor] | None) = None,
    ):
        """
        Register local tensors with NIXL. Build key->desc mapping.
        Args:
            state_dict: {key: torch.Tensor}
            sharding_dict: {key: NIXLSharding}
            binded_meta_tensor_mapping: {(key, shard_idx): torch.Tensor}
        """
        # If pinned temp memory is enabled, we need to first scan the state_dict
        # and find all the tensors that are not contiguous
        # Then we need to find all types (shape and dtype) of uncontiguous tensor
        # and allocate max_pinned_temp_memory_slots times of their size as pinned memory
        # (each pinned memory tensor is like this: [max_pinned_temp_memory_slots, *])
        # Then we enumerate the uncontiguous tensors again and map them with the pinned memory in a round-robin manner
        # (the first uncontiguous tensor map to [0, *], the second to [1, *],
        # the (max_pinned_temp_memory_slots+1)-th to [0, *] again, etc.)
        # We should record a mapping from the uncontiguous tensor to the index of the pinned memory
        # log_env_info(psrl_logger)
        if sharding_dict is None:
            sharding_dict = {}
        _uncontiguous_tensor_mapping: dict[
            tuple[Any, Any], list[tuple[str, tuple[int, ...], torch.Tensor]]
        ] = {}  # (shape, dtype) -> [key, shard_idx, uncontiguous_tensor]
        if self.max_pinned_temp_memory_slots is not None:
            # Scan the state_dict and find all the tensors that are not contiguous
            for key, tensor in state_dict.items():
                assert key in sharding_dict, f"Key {key} not found in sharding_dict."
                if tensor.device == torch.device("meta"):
                    continue
                sharding = sharding_dict[key]
                shard_indices = sharding.shard_indices
                local_sharded_tensors = sharding.get_local_sharded_tensors(tensor)
                for local_pos, local_sharded_tensor in enumerate(local_sharded_tensors):
                    if not local_sharded_tensor.is_contiguous():
                        if (
                            local_sharded_tensor.shape,
                            local_sharded_tensor.dtype,
                        ) not in _uncontiguous_tensor_mapping:
                            _uncontiguous_tensor_mapping[(local_sharded_tensor.shape, local_sharded_tensor.dtype)] = []
                        _uncontiguous_tensor_mapping[(local_sharded_tensor.shape, local_sharded_tensor.dtype)].append(
                            (key, shard_indices[local_pos], local_sharded_tensor)
                        )
            # Find all types of uncontiguous tensor and allocate pinned memory for them
            if _uncontiguous_tensor_mapping:
                self._pinned_memory = {}
                for (
                    shape,
                    dtype,
                ), uncontiguous_tensor_list in _uncontiguous_tensor_mapping.items():
                    self._pinned_memory[(shape, dtype)] = []
                    for slot_idx in range(self.max_pinned_temp_memory_slots):
                        memory_slot = torch.empty(*shape, dtype=dtype, device=self.device, requires_grad=False)
                        # psrl_logger.info(
                        #     f"{self.client_name} registering memory for the "
                        #     f"{slot_idx}-th slot of (shape: {shape}, dtype: {dtype})"
                        # )
                        try:
                            temp_desc = self.agent.register_memory([memory_slot])
                        except Exception as e:
                            raise RuntimeError(
                                f"{self.client_name} memory registration failed for "
                                f"the {slot_idx}-th slot of (shape: {shape}, dtype: {dtype}) : {e}"
                            ) from e
                        if not temp_desc:
                            raise RuntimeError(
                                f"{self.client_name} memory registration failed for "
                                f"the {slot_idx}-th slot of (shape: {shape}, dtype: {dtype})"
                            )
                        # psrl_logger.info(
                        #     f"{self.client_name} memory registration succeeded for "
                        #     f"the {slot_idx}-th slot of (shape: {shape}, dtype: {dtype})"
                        # )
                        temp_desc_bytes = self.agent.get_serialized_descs(temp_desc)
                        self._pinned_memory[(shape, dtype)].append((memory_slot, temp_desc_bytes))
                    for i, (key, shard_idx, uncontiguous_tensor) in enumerate(uncontiguous_tensor_list):
                        self._temp_pinned_idx_mapping[(key, shard_idx)] = i % self.max_pinned_temp_memory_slots

        # psrl_logger.info(f"{self.client_name}: temp pinned memory mapping is: {self._temp_pinned_idx_mapping}")
        tensor_infos = {}
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
                # If the tensor is on meta device, allocate on-the-fly or binded from the external tensor
                if local_sharded_tensor.device == torch.device("meta"):
                    if (
                        binded_meta_tensor_mapping is not None
                        and (key, shard_indices[local_pos]) in binded_meta_tensor_mapping
                    ):
                        local_sharded_tensor = binded_meta_tensor_mapping[(key, shard_indices[local_pos])]
                    else:
                        local_sharded_tensor = torch.empty_like(local_sharded_tensor, device=self.device)
                assert local_sharded_tensor.device == self.device, (
                    f"Local sharded tensor {key} shard {shard_indices[local_pos]} is not "
                    f"on device {self.device}, but on {local_sharded_tensor.device}, "
                    f"torch current device is {torch.cuda.current_device()}, "
                    f"CUDA_VISIBLE_DEVICES is {os.environ.get('CUDA_VISIBLE_DEVICES', 'None')}"
                )
                self._original_tensor_mapping[(key, shard_indices[local_pos])] = local_sharded_tensor

                # Create meta info for non-contiguous shard
                is_contiguous = local_sharded_tensor.is_contiguous()
                meta_info = NIXLShardMetaInfo(
                    dtype=local_sharded_tensor.dtype,
                    device=local_sharded_tensor.device,
                    shape=local_sharded_tensor.shape,
                    stride=local_sharded_tensor.stride(),
                    is_contiguous=is_contiguous,
                )
                shard_meta_info_list.append(meta_info)

                # Check if the shard is contiguous
                psrl_logger.debug(
                    f"{self.client_name} key {key} shard {shard_indices[local_pos]} "
                    f"register local tensor with shape {local_sharded_tensor.shape} and "
                    f"dtype {local_sharded_tensor.dtype}"
                )
                if local_sharded_tensor.is_contiguous():
                    # Contiguous shard: register directly
                    try:
                        desc = self.agent.register_memory([local_sharded_tensor])
                    except Exception as e:
                        raise RuntimeError(
                            f"{self.client_name} memory registration failed for "
                            f"key {key} shard {shard_indices[local_pos]}: {e}"
                        ) from e
                    if not desc:
                        raise RuntimeError(
                            f"{self.client_name} memory registration failed for "
                            f"key {key} shard {shard_indices[local_pos]}."
                        )
                    desc_bytes = self.agent.get_serialized_descs(desc)
                    desc_bytes_list.append(desc_bytes)
                else:
                    if self.max_pinned_temp_memory_slots is None:
                        # Non-contiguous shard: create temporary contiguous memory
                        # Create a new contiguous tensor with the same shape and dtype
                        # raise RuntimeError(
                        #     "Non-contiguous shard requires pinned memory, "
                        #     "but pinned memory is not enabled."
                        # )
                        contiguous_tensor = torch.empty_like(
                            local_sharded_tensor,
                            device=self.device,
                            requires_grad=False,
                        )
                        try:
                            temp_desc = self.agent.register_memory([contiguous_tensor])
                        except Exception as e:
                            raise RuntimeError(
                                f"{self.client_name} memory registration failed for "
                                f"key {key} shard {shard_indices[local_pos]} : {e}"
                            ) from e
                        if not temp_desc:
                            raise RuntimeError(
                                f"{self.client_name} memory registration failed for "
                                f"key {key} shard {shard_indices[local_pos]}."
                            )
                        temp_desc_bytes = self.agent.get_serialized_descs(temp_desc)
                    else:
                        assert self._pinned_memory is not None, "Pinned memory is not initialized."
                        assert (
                            local_sharded_tensor.shape,
                            local_sharded_tensor.dtype,
                        ) in self._pinned_memory, (
                            f"Pinned memory does not have slot for {key} shard {shard_indices[local_pos]}."
                        )
                        assert (
                            key,
                            shard_indices[local_pos],
                        ) in self._temp_pinned_idx_mapping, (
                            f"Pinned memory does not have slot for {key} shard {shard_indices[local_pos]}."
                        )
                        # Non-contiguous shard: map to pinned memory
                        pinned_slot, temp_desc_bytes = self._pinned_memory[
                            (local_sharded_tensor.shape, local_sharded_tensor.dtype)
                        ][self._temp_pinned_idx_mapping[(key, shard_indices[local_pos])]]
                        assert pinned_slot.dtype == local_sharded_tensor.dtype, (
                            f"Pinned slot {self._temp_pinned_idx_mapping[(key, shard_indices[local_pos])]} "
                            f"has {pinned_slot.dtype} dtype, but the {key} shard {shard_indices[local_pos]} "
                            f"requires {local_sharded_tensor.dtype} dtype."
                        )
                        assert pinned_slot.shape == local_sharded_tensor.shape, (
                            f"Pinned slot {self._temp_pinned_idx_mapping[(key, shard_indices[local_pos])]} "
                            f"has {pinned_slot.shape} shape, but the {key} shard {shard_indices[local_pos]} "
                            f"requires {local_sharded_tensor.shape} shape."
                        )
                        contiguous_tensor = pinned_slot

                    # Store None in desc_bytes_list to indicate this shard uses temp memory
                    desc_bytes_list.append(None)
                    # Build the contiguous meta info
                    contiguous_meta_info = NIXLShardMetaInfo(
                        dtype=contiguous_tensor.dtype,
                        device=contiguous_tensor.device,
                        shape=contiguous_tensor.shape,
                        stride=contiguous_tensor.stride(),
                        is_contiguous=True,
                    )
                    # Store temporary mappings
                    self._temp_tensor_mapping[(key, shard_indices[local_pos])] = contiguous_tensor
                    self._temp_desc_bytes_mapping[(key, shard_indices[local_pos])] = temp_desc_bytes
                    self._temp_meta_mapping[(key, shard_indices[local_pos])] = contiguous_meta_info

            # Create the tensor descriptor info
            tensor_infos[key] = NIXLTensorInfo(
                desc_bytes_list=desc_bytes_list,
                sharding=sharding,
                shard_meta_infos=shard_meta_info_list,
            )

        # Create the client info
        self.local_client_info = NIXLClientInfo(
            name=self.client_name,
            node_ip=get_local_ip(),
            node_gpu_id=get_local_gpu_id(),
            type=self.client_type,
            tensor_infos=tensor_infos,
            meta=self.agent.get_agent_metadata(),
            client_group_id=self.client_group_id,
        )
        psrl_logger.debug(
            f"Local client info is built, "
            f"temp pinned idx mapping is: {self._temp_pinned_idx_mapping}, "
            f"temp meta mapping is: {self._temp_meta_mapping}"
        )

    def connect_to_server(self, timeout: float = 600.0):
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

    def send_local_sharding(self, sharding_dict: dict[str, NIXLSharding]):
        """
        Send local sharding to the server.
        """
        assert self.mode == "meta_server", "send_local_sharding only valid in meta_server mode"
        assert self._is_connected, "Not connected to server"
        self.agent.send_notif(self.server_name, pickle.dumps({self.client_name: sharding_dict}))

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
            self.agent.send_notif(
                self.server_name,
                pickle.dumps({self.client_name: self.local_client_info.serialize()}),
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def send_local_temp_mapping(self):
        """Send local temporary mappings to the server"""
        assert self.mode == "meta_server", "send_local_temp_mapping only valid in meta_server mode"
        assert self._is_connected, "Not connected to server"
        self.agent.send_notif(
            self.server_name,
            pickle.dumps({self.client_name: self._temp_desc_bytes_mapping}),
        )

    def wait_for_server_sharding(self, timeout: float = 600.0):
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
                client_sharding_dicts = pickle.loads(notifs[self.server_name][0])
                assert isinstance(client_sharding_dicts, dict) and len(client_sharding_dicts) == 1, (
                    f"Expected a dict with one client sharding dict, but got {client_sharding_dicts}"
                )
                self._unified_sharding_dict = next(iter(client_sharding_dicts.values()))
                break
            if time.time() - start > timeout:
                raise TimeoutError("Timeout waiting for server sharding notification.")
            time.sleep(0.1)
        self._unified_sharding_dict_fetched = True
        return self._unified_sharding_dict

    def wait_for_server_info(self, timeout: float = 600.0):
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
                    if isinstance(notification_data, dict) and "client_infos" in notification_data:
                        # New format: includes communication plan
                        all_client_infos = notification_data["client_infos"]
                        for client_name, info_bytes in all_client_infos.items():
                            info = NIXLClientInfo.deserialize(info_bytes)
                            self._all_client_infos[client_name] = info
                        # Process communication plan
                        if notification_data.get("comm_plan"):
                            self._comm_plan = NIXLCommPlan.deserialize(notification_data["comm_plan"])
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

    def wait_for_server_temp_mappings(self, timeout: float = 600.0):
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
    def read(self, key: str, tag: str):
        """
        Read from the storage server.
        """
        if self.mode != "storage_server":
            raise RuntimeError("read(key, tag) only valid in storage_server mode")
        self.wait_for_server_info()
        local_desc = self.local_client_info.get_tensor_desc(self.agent, 0).trim()
        server_desc = self.server_client_info.get_tensor_desc(self.agent, 0).trim()
        handle = self.agent.initialize_xfer("READ", local_desc, server_desc, self.server_name, tag)
        if not handle:
            raise RuntimeError(f"Creating READ transfer failed for key {key}.")
        state = self.agent.transfer(handle)
        if state == "ERR":
            raise RuntimeError(f"Posting READ transfer failed for key {key}.")
        self.xfer_handles[(key, tag, "READ")] = handle

    @deprecated("Use client_write instead")
    def write(self, key: str, tag: str):
        """
        Write to the storage server.
        """
        if self.mode != "storage_server":
            raise RuntimeError("write(key, tag) only valid in storage_server mode")
        self.wait_for_server_info()
        local_desc = self.local_client_info.get_tensor_desc(self.agent, 0).trim()
        server_desc = self.server_client_info.get_tensor_desc(self.agent, 0).trim()
        handle = self.agent.initialize_xfer("WRITE", local_desc, server_desc, self.server_name, tag)
        if not handle:
            raise RuntimeError(f"Creating WRITE transfer failed for key {key}.")
        state = self.agent.transfer(handle)
        if state == "ERR":
            raise RuntimeError(f"Posting WRITE transfer failed for key {key}.")
        self.xfer_handles[(key, tag, "WRITE")] = handle

    # --- meta_server mode: client-to-client read/write ---
    def _ensure_client_info_fetched(self, target_client: str):
        """Ensure connection to target client is established."""
        if target_client in self._target_client_connected:
            return
        assert target_client in self._all_client_infos, (
            f"Target client {target_client} not found in client infos: {self._all_client_infos.keys()}"
        )
        meta = self._all_client_infos[target_client].meta
        try:
            self.agent.add_remote_agent(meta)
            # nixl_agent_name_bytes = self.agent.add_remote_agent(meta)
            # assert nixl_agent_name_bytes.decode() == target_client, (
            #     f"NIXL agent name {nixl_agent_name_bytes.decode()} "
            #     f"does not match target client: {target_client}"
            # )
        except Exception as e:
            psrl_logger.error(f"Error adding remote agent {target_client}: {e}")
            raise e
        self._target_client_connected[target_client] = True

    def client_read(
        self,
        target_agent: str,
        target_client: str,
        key: str,
        tag: str,
        comm_plan: NIXLCommPlan | None = None,
        merge_and_cache_xfer: bool | None = False,
    ) -> list[tuple[int, ...]]:
        """Read from another client (meta_server mode), supports shard alignment and communication plan."""
        if self.mode != "meta_server":
            raise RuntimeError("client_read only valid in meta_server mode")
        plan = comm_plan or self._comm_plan
        self._ensure_client_info_fetched(target_client)
        remote_info = self._all_client_infos[target_client].get_tensor_info(key)
        local_info = self.local_client_info.get_tensor_info(key)
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
            assert (
                shard_idx in local_info.sharding.shard_indices and shard_idx in remote_info.sharding.shard_indices
            ), f"Shard {shard_idx} not found in local or remote shards for key {key}"
            local_pos = local_info.sharding.shard_indices.index(shard_idx)
            remote_pos = remote_info.sharding.shard_indices.index(shard_idx)

            # For non-contiguous shards, record the running key and shard idx
            is_contiguous, running_key, running_shard_idx = True, None, None
            # Get local descriptor (check if it's a temporary one)
            local_desc_bytes = local_info.desc_bytes_list[local_pos]
            if local_desc_bytes is not None:
                assert local_info.shard_meta_infos[local_pos].can_xfer_to(remote_info.shard_meta_infos[remote_pos]), (
                    f"Shard meta info mismatch for key {key} shard {shard_idx}: "
                    f"{local_info.shard_meta_infos[local_pos]} != {remote_info.shard_meta_infos[remote_pos]}"
                )
            else:
                is_contiguous = False
                meta_info = self._temp_meta_mapping[(key, shard_idx)]
                assert meta_info.can_xfer_to(remote_info.shard_meta_infos[remote_pos]), (
                    f"Temporary shard meta info mismatch for key {key} shard {shard_idx}: "
                    f"{meta_info} != {remote_info.shard_meta_infos[remote_pos]}"
                )
                # Use temporary descriptor for non-contiguous shard
                local_desc_bytes = self._get_temp_desc_bytes(self.client_name, key, shard_idx)
                if local_desc_bytes is None:
                    raise RuntimeError(f"No temporary descriptor found for key {key} shard {shard_idx}")
                # Wait for the pinned slot to be available
                if self.max_pinned_temp_memory_slots is not None:
                    pinned_idx = self._temp_pinned_idx_mapping[(key, shard_idx)]
                    slot_key = (meta_info.shape, meta_info.dtype, pinned_idx)
                    if slot_key in self._pinned_slot_running_xfer:
                        (
                            running_key,
                            running_tag,
                            running_op_type,
                            running_target_client,
                            running_shard_idx,
                        ) = self._pinned_slot_running_xfer[slot_key]
                        # start_time = time.time()
                        self.wait(
                            running_key,
                            running_tag,
                            running_op_type,
                            target_client=running_target_client,
                            shard_idx=running_shard_idx,
                        )
                        # end_time = time.time()
                        # psrl_logger.info(
                        #     f"{self.client_name} read uncontiguous {(key, shard_idx)}, "
                        #     f"pinned slot {pinned_idx} is available, time: {end_time - start_time}s"
                        # )
                    self._pinned_slot_running_xfer[slot_key] = (
                        key,
                        tag,
                        "READ",
                        target_client,
                        shard_idx,
                    )

            # Get remote descriptor (check if it's a temporary one)
            remote_desc_bytes = remote_info.desc_bytes_list[remote_pos]
            if remote_desc_bytes is None:
                raise NotImplementedError("Not implemented for non-contiguous shards on the remote side.")
                # Use temporary descriptor for non-contiguous shard
                remote_desc_bytes = self._get_temp_desc_bytes(target_client, key, remote_pos)
                if remote_desc_bytes is None:
                    raise RuntimeError(f"No remote temporary descriptor found for key {key} shard {shard_idx}")

            # Double check the shard size
            assert local_info.get_shard_size_bytes(local_pos) == remote_info.get_shard_size_bytes(remote_pos), (
                f"Shard size mismatch for key {key} shard {shard_idx}: "
                f"{local_info.get_shard_size_bytes(local_pos)} != {remote_info.get_shard_size_bytes(remote_pos)}"
            )
            local_desc = self.agent.deserialize_descs(local_desc_bytes).trim()
            remote_desc = self.agent.deserialize_descs(remote_desc_bytes).trim()
            # Contiguous xfer can be merged and executed together later
            if merge_and_cache_xfer and is_contiguous:
                self._cached_xfer_descs.append(("READ", local_desc, remote_desc, target_agent, tag, target_client))
                return []
            # Real xfer
            try:
                if running_key is not None and running_shard_idx is not None:
                    assert (
                        running_key,
                        running_shard_idx,
                    ) in self._contiguous_event_cache, (
                        f"Running key {running_key} shard {running_shard_idx} not found in contiguous event cache"
                    )
                    self._contiguous_event_cache[(running_key, running_shard_idx)].synchronize()
                    self._contiguous_event_cache.pop((running_key, running_shard_idx))
                tag = make_xfer_tag(tag, self.client_name, target_client, key, shard_idx)
                if tag not in self.xfer_handles:
                    self.xfer_handles[tag] = self.agent.initialize_xfer(
                        "READ",
                        local_desc,
                        remote_desc,
                        target_agent,
                        make_xfer_tag(tag, self.client_name, target_client, key, shard_idx),
                    )
                handle = self.xfer_handles[tag]
            except Exception as e:
                raise RuntimeError(
                    f"{self.client_name} creating client READ transfer to {target_client} failed for "
                    f"key {key} shard {shard_idx}: {e}"
                ) from e
            if not handle:
                raise RuntimeError(
                    f"{self.client_name} creating client READ transfer to {target_client} failed for "
                    f"key {key} shard {shard_idx}."
                )
            # start_time = time.time()
            state = self.agent.transfer(handle)
            # end_time = time.time()
            # psrl_logger.info(
            #     f"{self.client_name} posted client READ transfer to {target_client} for "
            #     f"key {key} shard {shard_idx}, time: {end_time - start_time}s"
            # )
            if state == "ERR":
                raise RuntimeError(
                    f"{self.client_name} posting client READ transfer to {target_client} failed for "
                    f"key {key} shard {shard_idx}."
                )
        return shards_to_transfer

    def client_write(
        self,
        target_agent: str,
        target_client: str,
        key: str,
        tag: str,
        comm_plan: NIXLCommPlan | None = None,
        merge_and_cache_xfer: bool | None = False,
    ) -> list[tuple[int, ...]]:
        """Write to another client (meta_server mode), supports shard alignment and communication plan."""
        if self.mode != "meta_server":
            raise RuntimeError("client_write only valid in meta_server mode")
        plan = comm_plan or self._comm_plan
        self._ensure_client_info_fetched(target_client)
        remote_info = self._all_client_infos[target_client].get_tensor_info(key)
        local_info = self.local_client_info.get_tensor_info(key)
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
            assert (
                shard_idx in local_info.sharding.shard_indices and shard_idx in remote_info.sharding.shard_indices
            ), f"Shard {shard_idx} not found in local or remote shards for key {key}"
            local_pos = local_info.sharding.shard_indices.index(shard_idx)
            remote_pos = remote_info.sharding.shard_indices.index(shard_idx)

            # Check if local shard is non-contiguous and needs data copying
            is_contiguous = True
            local_desc_bytes = local_info.desc_bytes_list[local_pos]
            if local_desc_bytes is not None:
                assert local_info.shard_meta_infos[local_pos].can_xfer_to(remote_info.shard_meta_infos[remote_pos]), (
                    f"Shard meta info mismatch for key {key} shard {shard_idx}: "
                    f"{local_info.shard_meta_infos[local_pos]} != {remote_info.shard_meta_infos[remote_pos]}"
                )
            else:
                is_contiguous = False
                meta_info = self._temp_meta_mapping[(key, shard_idx)]
                assert meta_info.can_xfer_to(remote_info.shard_meta_infos[remote_pos]), (
                    f"Temporary shard meta info mismatch for key {key} shard {shard_idx}: "
                    f"{meta_info} != {remote_info.shard_meta_infos[remote_pos]}"
                )
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
                    slot_key = (meta_info.shape, meta_info.dtype, pinned_idx)
                    if slot_key in self._pinned_slot_running_xfer:
                        (
                            running_key,
                            running_tag,
                            running_op_type,
                            running_target_client,
                            running_shard_idx,
                        ) = self._pinned_slot_running_xfer[slot_key]
                        start_time = time.time()
                        self.wait(
                            running_key,
                            running_tag,
                            running_op_type,
                            target_client=running_target_client,
                            shard_idx=running_shard_idx,
                        )
                        end_time = time.time()
                        psrl_logger.debug(
                            f"{self.client_name} write uncontiguous {(key, shard_idx)}, "
                            f"pinned slot {pinned_idx} is available, time: {end_time - start_time}s"
                        )
                    self._pinned_slot_running_xfer[slot_key] = (
                        key,
                        tag,
                        "WRITE",
                        target_client,
                        shard_idx,
                    )
                # Copy data from original non-contiguous tensor to temporary contiguous tensor
                self._contiguous_event_cache[(key, shard_idx)] = torch.cuda.Event()
                contiguous_tensor.copy_(original_tensor.detach())
                self._contiguous_event_cache[(key, shard_idx)].record()
                # Use temporary descriptor
                local_desc_bytes = self._get_temp_desc_bytes(self.client_name, key, shard_idx)
                if local_desc_bytes is None:
                    raise RuntimeError(
                        f"No temporary descriptor found for key {key} shard {shard_idx} "
                        f"in {self.client_name}'s temp descs"
                    )

            # Get remote descriptor (check if it's a temporary one)
            remote_desc_bytes = remote_info.desc_bytes_list[remote_pos]
            if remote_desc_bytes is None:
                raise NotImplementedError("Not implemented for non-contiguous shards on the remote side.")
                # Use temporary descriptor for non-contiguous shard
                remote_desc_bytes = self._get_temp_desc_bytes(target_client, key, shard_idx)
                if remote_desc_bytes is None:
                    raise RuntimeError(f"No remote temporary descriptor found for key {key} shard {shard_idx}")

            # Double check the shard size
            assert local_info.get_shard_size_bytes(local_pos) == remote_info.get_shard_size_bytes(remote_pos), (
                f"Shard size mismatch for key {key} shard {shard_idx}: "
                f"{local_info.get_shard_size_bytes(local_pos)} != {remote_info.get_shard_size_bytes(remote_pos)}"
            )
            local_desc = self.agent.deserialize_descs(local_desc_bytes).trim()
            remote_desc = self.agent.deserialize_descs(remote_desc_bytes).trim()
            if merge_and_cache_xfer and is_contiguous:
                self._cached_xfer_descs.append(("WRITE", local_desc, remote_desc, target_agent, tag, target_client))
                return []
            # Real xfer
            try:
                if (key, shard_idx) in self._contiguous_event_cache:
                    self._contiguous_event_cache[(key, shard_idx)].synchronize()
                    self._contiguous_event_cache.pop((key, shard_idx))
                tag = make_xfer_tag(tag, self.client_name, target_client, key, shard_idx)
                if tag not in self.xfer_handles:
                    self.xfer_handles[tag] = self.agent.initialize_xfer(
                        "WRITE",
                        local_desc,
                        remote_desc,
                        target_agent,
                        make_xfer_tag(tag, self.client_name, target_client, key, shard_idx),
                    )
                handle = self.xfer_handles[tag]
            except Exception as e:
                raise RuntimeError(
                    f"{self.client_name} creating client WRITE transfer to {target_client} failed for "
                    f"key {key} shard {shard_idx}: {e}"
                ) from e
            if not handle:
                raise RuntimeError(
                    f"{self.client_name} creating client WRITE transfer to {target_client} failed for "
                    f"key {key} shard {shard_idx}."
                )
            state = self.agent.transfer(handle)
            if state == "ERR":
                raise RuntimeError(
                    f"{self.client_name} posting client WRITE transfer to {target_client} failed for "
                    f"key {key} shard {shard_idx}."
                )
        return shards_to_transfer

    # NOTE(lhy): This use low-level NIXL API to merge fragmented transfers into a single transfer,
    # which is more efficient than finishing each transfer individually.
    def merge_and_finish_cached_xfer(self, timeout: float = 600.0):
        """Merge and finish cached transfers."""
        if hasattr(self, "_cached_xfer_descs"):
            _cached_xfer_descs_by_op_type = {}
            for (
                op_type,
                local_desc,
                remote_desc,
                target_agent,
                tag,
                target_client,
            ) in self._cached_xfer_descs:
                # Group by op_type, target agent and tag
                if op_type not in _cached_xfer_descs_by_op_type:
                    _cached_xfer_descs_by_op_type[op_type] = {}
                if target_client not in _cached_xfer_descs_by_op_type[op_type]:
                    _cached_xfer_descs_by_op_type[op_type][target_client] = {}
                assert local_desc.descCount() == 1 and remote_desc.descCount() == 1, (
                    f"Local and remote descriptor count should be 1, "
                    f"but found {local_desc.descCount()} and {remote_desc.descCount()} "
                    f"for op type {op_type} target client {target_client} tag {tag}"
                )
                if tag not in _cached_xfer_descs_by_op_type[op_type][target_client]:
                    _cached_xfer_descs_by_op_type[op_type][target_client][tag] = [
                        {"mem_type": local_desc.getType(), "descs": [local_desc[0]]},
                        {"mem_type": remote_desc.getType(), "descs": [remote_desc[0]]},
                        target_agent,
                    ]
                else:
                    assert (
                        _cached_xfer_descs_by_op_type[op_type][target_client][tag][0]["mem_type"]
                        == local_desc.getType()
                        and _cached_xfer_descs_by_op_type[op_type][target_client][tag][1]["mem_type"]
                        == remote_desc.getType()
                    ), (
                        f"Mem type mismatch for op type {op_type} target client {target_client} tag {tag}: "
                        f"{_cached_xfer_descs_by_op_type[op_type][target_client][tag][0]['mem_type']} != "
                        f"{local_desc.getType()} or "
                        f"{_cached_xfer_descs_by_op_type[op_type][target_client][tag][1]['mem_type']} != "
                        f"{remote_desc.getType()}"
                    )
                    _cached_xfer_descs_by_op_type[op_type][target_client][tag][0]["descs"].append(local_desc[0])
                    _cached_xfer_descs_by_op_type[op_type][target_client][tag][1]["descs"].append(remote_desc[0])
                    assert target_agent == _cached_xfer_descs_by_op_type[op_type][target_client][tag][2], (
                        f"Target agent mismatch for op type {op_type} target client {target_client} tag {tag}: "
                        f"{target_agent} != {_cached_xfer_descs_by_op_type[op_type][target_client][tag][2]}"
                    )
            for op_type, target_client_dict in _cached_xfer_descs_by_op_type.items():
                for target_client, tag_dict in target_client_dict.items():
                    for tag, xfer_desc_meta in tag_dict.items():
                        (
                            merged_local_desc_dict,
                            merged_remote_desc_dict,
                            target_agent,
                        ) = (
                            xfer_desc_meta[0],
                            xfer_desc_meta[1],
                            xfer_desc_meta[2],
                        )
                        try:
                            merged_local_desc = nixlBind.nixlXferDList(
                                merged_local_desc_dict["mem_type"],
                                merged_local_desc_dict["descs"],
                            )
                            merged_remote_desc = nixlBind.nixlXferDList(
                                merged_remote_desc_dict["mem_type"],
                                merged_remote_desc_dict["descs"],
                            )
                            start_time = time.time()
                            handle = self.agent.initialize_xfer(
                                op_type,
                                merged_local_desc,
                                merged_remote_desc,
                                target_agent,
                                make_xfer_tag(
                                    tag,
                                    self.client_name,
                                    target_client,
                                    f"merged_xfer_for_{tag}",
                                ),
                            )
                            end_time = time.time()
                            psrl_logger.debug(
                                f"{self.client_name} created client {op_type} transfer to {target_client} "
                                f"for tag {tag} with {merged_local_desc.descCount()} merged descriptors, "
                                f"time: {end_time - start_time}s"
                            )
                        except Exception as e:
                            raise RuntimeError(
                                f"{self.client_name} creating client {op_type} transfer to {target_client} failed "
                                f"for tag {tag} with {merged_local_desc.descCount()} merged descriptors: {e}, "
                                f"local desc with type {merged_local_desc.getType()}: "
                                f"{[merged_local_desc[i] for i in range(merged_local_desc.descCount())]}, "
                                f"remote desc with type {merged_remote_desc.getType()}: "
                                f"{[merged_remote_desc[i] for i in range(merged_remote_desc.descCount())]}, "
                                f"target agent: {target_agent}"
                            ) from e
                        if not handle:
                            raise RuntimeError(
                                f"{self.client_name} creating client {op_type} transfer to {target_client} failed "
                                f"for tag {tag} with {merged_local_desc.descCount()} merged descriptors."
                            )
                        start_time = time.time()
                        state = self.agent.transfer(handle)
                        end_time = time.time()
                        psrl_logger.info(
                            f"{self.client_name} posted client {op_type} transfer to {target_client} for tag {tag} "
                            f"with {merged_local_desc.descCount()} merged descriptors, time: {end_time - start_time}s"
                        )
                        if state == "ERR":
                            raise RuntimeError(
                                f"{self.client_name} posting client {op_type} transfer to {target_client} failed "
                                f"for tag {tag} with {merged_local_desc.descCount()} merged descriptors."
                            )
                        start = time.time()
                        while True:
                            try:
                                state = self.agent.check_xfer_state(handle)
                            except Exception as e:
                                raise RuntimeError(
                                    f"Checking merged transfer state for ({op_type}, {target_client}, {tag}) "
                                    f"from {self.client_name} failed: {e}"
                                ) from e
                            if state == "ERR":
                                raise RuntimeError(
                                    f"Merged transfer error for ({op_type}, {target_client}, {tag}) "
                                    f"from {self.client_name}"
                                )
                            elif state == "DONE":
                                break
                            if time.time() - start > timeout:
                                raise TimeoutError(
                                    f"Timeout waiting for merged transfer ({op_type}, {target_client}, {tag}) "
                                    f"from {self.client_name}"
                                )
                            time.sleep(0.0001)
                        end = time.time()
                        psrl_logger.debug(
                            f"{self.client_name} finished client {op_type} transfer to {target_client} for tag {tag} "
                            f"with {merged_local_desc.descCount()} merged descriptors, time: {end - start}s"
                        )
            self._cached_xfer_descs = []

    def wait(
        self,
        key: str,
        tag: str,
        op_type: str,
        target_client: str | None = None,
        shard_idx: int | None = None,
        timeout: float = 600.0,
    ):
        """
        Wait for a transfer to be completed.
        """
        if self.mode == "storage_server":
            handle = self.xfer_handles.get((key, tag, op_type))
            if handle is None:
                raise RuntimeError(f"No handle for ({key}, {tag}, {op_type}) from {self.client_name}")
            start = time.time()
            while True:
                try:
                    state = self.agent.check_xfer_state(handle)
                except Exception as e:
                    raise RuntimeError(
                        f"Checking transfer state for ({key}, {tag}, {op_type}) from {self.client_name} failed: {e}"
                    ) from e
                if state == "ERR":
                    raise RuntimeError(f"Transfer error for ({key}, {tag}, {op_type}) from {self.client_name}")
                elif state == "DONE":
                    break
                if time.time() - start > timeout:
                    raise TimeoutError(
                        f"Timeout waiting for transfer ({key}, {tag}, {op_type}) from {self.client_name}"
                    )
                time.sleep(0.0001)
        elif self.mode == "meta_server":
            # Shard tag, wait for all shards
            info = self.local_client_info.get_tensor_info(key)
            waiting_shard_indices = [shard_idx] if shard_idx is not None else info.sharding.shard_indices
            for shard_idx in waiting_shard_indices:
                handle = self.xfer_handles.get(make_xfer_tag(tag, self.client_name, target_client, key, shard_idx))
                if handle is None:
                    # psrl_logger.info(
                    #     f"Transfer ({key}, {tag}, {op_type}, shard {shard_idx}) "
                    #     f"from {self.client_name} to {target_client} not found, continue"
                    # )
                    continue  # This shard did not do transfer
                start = time.time()
                while True:
                    try:
                        state = self.agent.check_xfer_state(handle)
                    except Exception as e:
                        raise RuntimeError(
                            f"Checking transfer state for ({key}, {tag}, {op_type}, shard {shard_idx}) "
                            f"from {self.client_name} to {target_client} failed: {e}"
                        ) from e
                    if state == "ERR":
                        raise RuntimeError(
                            f"Transfer error for ({key}, {tag}, {op_type}, shard {shard_idx}) "
                            f"from {self.client_name} to {target_client}"
                        )
                    elif state == "DONE":
                        # psrl_logger.info(
                        #     f"Transfer ({key}, {tag}, {op_type}, shard {shard_idx}) "
                        #     f"from {self.client_name} to {target_client} done, "
                        #     f"time cost: {time.time() - start} seconds"
                        # )
                        # For non-contiguous shards, sync data back to original tensor after READ
                        if op_type == "READ":
                            local_pos = info.sharding.shard_indices.index(shard_idx)
                            if info.desc_bytes_list[local_pos] is None:
                                # Non-contiguous shard: copy data from temporary to original
                                original_tensor = self._get_local_original_tensor(key, shard_idx)
                                if original_tensor is None:
                                    raise RuntimeError(
                                        f"No original tensor mapping found for key {key} shard {shard_idx}"
                                    )
                                contiguous_tensor = self._get_local_temp_tensor(key, shard_idx)
                                if contiguous_tensor is None:
                                    raise RuntimeError(
                                        f"No temporary tensor mapping found for key {key} shard {shard_idx}"
                                    )
                                # Copy data from temporary contiguous tensor back to original non-contiguous tensor
                                self._contiguous_event_cache[(key, shard_idx)] = torch.cuda.Event()
                                original_tensor.copy_(contiguous_tensor)
                                self._contiguous_event_cache[(key, shard_idx)].record()
                        # NOTE(lhy): can keep the handle for future reuse
                        # but no obvious performance gain, so we just pop it here
                        self.xfer_handles.pop(make_xfer_tag(tag, self.client_name, target_client, key, shard_idx))
                        break
                    if time.time() - start > timeout:
                        raise TimeoutError(
                            f"Timeout waiting for transfer ({key}, {tag}, {op_type}, shard {shard_idx}) "
                            f"from {self.client_name} to {target_client}"
                        )
                    time.sleep(0.0001)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def shutdown(self):
        # Release temporary memory first
        self.release_temp_memory()

        if self.local_client_info:
            for info in self.local_client_info.tensor_infos.values():
                for local_pos in range(info.num_local_shards):
                    # Only deregister if the descriptor is not None (not a temporary one)
                    if info.desc_bytes_list[local_pos] is not None:
                        self.agent.deregister_memory(info.get_desc(self.agent, local_pos))


class NIXLMultiStorageClients:
    """
    Multiple NIXLStorageClient instances can be registered to the same NIXL agent.
    This is useful for multi-precision (e.g., train use fp32, gen use bf16).
    """

    def __init__(
        self,
        agent_name: str,
        multi_client_names: list[str],
        server_name: str,
        use_gpu: bool,
        multi_client_types: list[NIXLClientType],
        nixl_config: DictConfig,
        nixl_interface: NIXLInterface | None = None,
        client_group_id: int = -1,  # -1 is the default client group
    ):
        self.agent_name = agent_name
        self.multi_client_names = multi_client_names
        self.server_name = server_name
        if use_gpu:
            assert torch.cuda.is_available(), "CUDA is not available."
        self.device = torch.device("cuda:0" if use_gpu else "cpu")
        assert nixl_config.server_mode == "meta_server", "NIXLMultiStorageClient only supports meta_server mode"
        self.server_ip = nixl_config.server_ip
        self.server_port = nixl_config.server_port
        self.nixl_interface = nixl_interface if nixl_interface is not None else NIXLInterface()

        # Initialize NIXL agent
        self.client_port = (
            0
            if self.nixl_interface.port_scanner is None
            else ray.get(self.nixl_interface.port_scanner.find_free_port.remote(host=get_worker_info()[0]))
        )
        self.agent = nixl_agent(self.agent_name, nixl_agent_config(True, True, self.client_port))

        # Initialize multi clients
        self.multi_clients: list[NIXLStorageClient] = []
        for client_name, client_type in zip(multi_client_names, multi_client_types):
            self.multi_clients.append(
                NIXLStorageClient(
                    client_name,
                    server_name,
                    use_gpu,
                    client_type,
                    nixl_config,
                    nixl_interface,
                    binded_agent=self.agent,
                    client_group_id=client_group_id,
                )
            )

        self._is_connected = False
        self._multi_unified_sharding_dicts_fetched = False

    def get_client_by_name(self, client_name: str) -> NIXLStorageClient:
        for client in self.multi_clients:
            if client.client_name == client_name:
                return client
        raise ValueError(f"Client {client_name} not found")

    def release_temp_memory(self):
        for client in self.multi_clients:
            client.release_temp_memory()

    def reallocate_temp_memory(self):
        for client in self.multi_clients:
            client.reallocate_temp_memory()

    def connect_to_server(self, timeout: float = 600.0):
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
        for client in self.multi_clients:
            client._is_connected = True

    def send_local_sharding(self, multi_sharding_dicts: dict[str, dict[str, NIXLSharding]]):
        assert self._is_connected, "Not connected to server"
        # multi_sharding_dicts: {client_name: {key: NIXLSharding}}
        self.agent.send_notif(self.server_name, pickle.dumps(multi_sharding_dicts))

    def send_local_info(self):
        assert self._is_connected, "Not connected to server"
        for client in self.multi_clients:
            assert client.local_client_info is not None, "Local client info not registered"
        self.agent.send_notif(
            self.server_name,
            pickle.dumps({client.client_name: client.local_client_info.serialize() for client in self.multi_clients}),
        )

    def send_local_temp_mapping(self):
        assert self._is_connected, "Not connected to server"
        for client in self.multi_clients:
            assert client._temp_desc_bytes_mapping is not None, "Temp desc bytes mapping not registered"
        self.agent.send_notif(
            self.server_name,
            pickle.dumps({client.client_name: client._temp_desc_bytes_mapping for client in self.multi_clients}),
        )

    def wait_for_server_sharding(self, timeout: float = 600.0):
        assert self._is_connected, "Not connected to server"
        start = time.time()
        if self._multi_unified_sharding_dicts_fetched:
            return
        while True:
            notifs = self.agent.get_new_notifs()
            if self.server_name in notifs and notifs[self.server_name]:
                client_sharding_dicts = pickle.loads(notifs[self.server_name][0])
                assert isinstance(client_sharding_dicts, dict), (
                    f"Expected a dict of client sharding dicts, but got {client_sharding_dicts}"
                )
                for client_name, sharding_dict in client_sharding_dicts.items():
                    assert client_name in self.multi_client_names, (
                        f"Client {client_name} not found in {self.multi_client_names}"
                    )
                    self.multi_clients[
                        self.multi_client_names.index(client_name)
                    ]._unified_sharding_dict = sharding_dict
                break
            if time.time() - start > timeout:
                raise TimeoutError("Timeout waiting for server sharding notification.")
            time.sleep(0.1)
        self._multi_unified_sharding_dicts_fetched = True
        return {client.client_name: client._unified_sharding_dict for client in self.multi_clients}

    def wait_for_server_info(self, timeout: float = 600.0):
        assert self._is_connected, "Not connected to server"
        self.multi_clients[0].wait_for_server_info(timeout)
        if len(self.multi_clients) > 1:
            for client in self.multi_clients[1:]:
                client._all_client_infos = self.multi_clients[0]._all_client_infos
                client._comm_plan = self.multi_clients[0]._comm_plan
                client._all_client_infos_fetched = True

    def wait_for_server_temp_mappings(self, timeout: float = 600.0):
        assert self._is_connected, "Not connected to server"
        self.multi_clients[0].wait_for_server_temp_mappings(timeout)
        if len(self.multi_clients) > 1:
            for client in self.multi_clients[1:]:
                client._all_temp_mappings = self.multi_clients[0]._all_temp_mappings
                client._all_temp_mappings_fetched = True

    def client_read(
        self,
        cur_client: str,
        target_agent: str,
        target_client: str,
        key: str,
        tag: str,
        comm_plan: NIXLCommPlan | None = None,
    ):
        assert self._is_connected, "Not connected to server"
        client = self.get_client_by_name(cur_client)
        client.client_read(target_agent, target_client, key, tag, comm_plan)

    def client_write(
        self,
        cur_client: str,
        target_agent: str,
        target_client: str,
        key: str,
        tag: str,
        comm_plan: NIXLCommPlan | None = None,
    ):
        assert self._is_connected, "Not connected to server"
        client = self.get_client_by_name(cur_client)
        client.client_write(target_agent, target_client, key, tag, comm_plan)

    def wait(
        self,
        cur_client: str,
        key: str,
        tag: str,
        op_type: str,
        target_client: str | None = None,
        timeout: float = 600.0,
    ):
        assert self._is_connected, "Not connected to server"
        client = self.get_client_by_name(cur_client)
        client.wait(key, tag, op_type, target_client, timeout)

    def shutdown(self):
        # TODO(lhy): better shutdown logic
        # May release twice if multi clients have shared memory
        for client in self.multi_clients:
            client.shutdown()
