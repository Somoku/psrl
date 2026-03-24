import logging
import os
import pickle
import time
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import nixl._bindings as nixlBind
import torch
from nixl._api import nixl_agent, nixl_agent_config
from omegaconf import DictConfig

from psrl.utils.common.patch_utils import apply_tms_patch
from psrl.utils.common.utils import lazy_import_to_globals
from psrl.utils.logger import deprecated, get_worker_info
from psrl.utils.nixl.comm_plan import NIXLCommPlan
from psrl.utils.nixl.network_topology import get_local_gpu_id, get_local_ip
from psrl.utils.nixl.nixl_spec import (
    NIXLClientInfo,
    NIXLClientType,
    NIXLSharding,
    NIXLShardMetaInfo,
    NIXLTensorInfo,
    find_free_port_with_scope,
)

if TYPE_CHECKING:
    from torch_memory_saver import torch_memory_saver
else:
    torch_memory_saver = None  # type: ignore

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
        replica_idx: int = 0,
        worker_index: int = 0,
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
        self.client_group_id = client_group_id
        self.enable_nixl_for_temp_buffers = nixl_config.enable_tms_for_temp_buffers and use_gpu

        if self.enable_nixl_for_temp_buffers:
            lazy_import_to_globals("torch_memory_saver", "torch_memory_saver")
            apply_tms_patch()

            psrl_logger.info(f"NIXLStorageClient {self.client_name} enabled TMS for temporary buffers.")

        # Initialize NIXL agent
        if binded_agent is None:
            self.client_port = find_free_port_with_scope(
                replica_idx=replica_idx,
                worker_index=worker_index,
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
        self._pinned_slot_running_write_xfer: dict[
            tuple[torch.Size, torch.dtype, int], tuple
        ] = {}  # (shape, dtype, pinned_idx) -> (key, tag, op_type, target_client)
        self._pinned_slot_running_read_xfer: dict[
            tuple[torch.Size, torch.dtype, int], tuple
        ] = {}  # (shape, dtype, pinned_idx) -> (key, tag, op_type, target_client)
        self._pinned_memory: dict[tuple[torch.Size, torch.dtype], list[torch.Tensor]] | None = None
        self._read_contiguous_event_cache: dict[
            tuple[str, tuple[int, ...]], torch.cuda.streams.Event
        ] = {}  # (key, shard_idx) -> cudaEvent
        self._write_contiguous_event_cache: dict[
            tuple[str, tuple[int, ...]], torch.cuda.streams.Event
        ] = {}  # (key, shard_idx) -> cudaEvent

        # Registry for all local registrations (desc_bytes -> desc object)
        self._registered_descs: dict[bytes, Any] = {}

        # mem_type -> [(base_addr, nbytes, device_id, mem_type)] tuples for storage registration
        self._mtype_to_reg_region_lists: dict[str, list[tuple[int, int, int, str]]] = {}
        self._reg_regions: set[tuple[int, int, int, str]] = set()

        # Mapping from (key, shard_idx) to registered desc slice info
        self.contig_desc_slice_map: dict[tuple[str, tuple[int, ...]], tuple[int, int, int, str]] = {}
        self.temp_desc_slice_map: dict[tuple[str, tuple[int, ...]], tuple[int, int, int, str]] = {}

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
        """Release all temporary memory and deregister descriptors.

        NOTE(linsh): deregistration is done globally. Here we just clear the local mappings.
        """
        # Clear all temporary mappings
        self._temp_tensor_mapping = {}
        self._temp_desc_bytes_mapping = {}
        self._temp_meta_mapping = {}
        self._pinned_memory = None
        self._temp_pinned_idx_mapping = {}
        self.temp_desc_slice_map = {}

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

    def _track_registered_desc(self, desc) -> bytes:
        """Cache registered desc object and return serialized bytes."""
        desc_bytes = self.agent.get_serialized_descs(desc)
        self._registered_descs[desc_bytes] = desc
        return desc_bytes

    def _record_region_registration(self, tensor: torch.Tensor) -> tuple[int, int, int, str]:
        """Record the base storage registration entry for a tensor view.

        Returns a storage key tuple of (base_addr, nbytes, device_id, mem_type).
        """
        storage = tensor.untyped_storage()
        base_addr = int(storage.data_ptr())
        nbytes = int(storage.nbytes())
        device_id = int(tensor.get_device()) if tensor.is_cuda else 0
        mem_type = "cuda" if tensor.is_cuda else "cpu"
        storage_region = (base_addr, nbytes, device_id, mem_type)
        if storage_region not in self._reg_regions:
            self._reg_regions.add(storage_region)
            self._mtype_to_reg_region_lists.setdefault(mem_type, []).append((base_addr, nbytes, device_id, ""))
        return storage_region

    def _build_xfer_desc_bytes(self, addr: int, length: int, device_id: int, mem_type: str) -> bytes:
        """Build serialized xfer descriptors for a contiguous slice."""
        xfer_descs = self.agent.get_xfer_descs([(addr, length, device_id)], mem_type=mem_type)
        return self.agent.get_serialized_descs(xfer_descs)

    def _deserialize_to_xfer_descs(self, desc_bytes: bytes):
        """Deserialize desc bytes and ensure xfer descriptors are returned."""
        descs = self.agent.deserialize_descs(desc_bytes)
        if isinstance(descs, nixlBind.nixlRegDList):
            return descs.trim()
        return descs

    def _ensure_xfer_descs(self, descs):
        """Normalize reg/xfer descriptors to xfer list."""
        if isinstance(descs, nixlBind.nixlRegDList):
            return descs.trim()
        return descs

    def _deregister_all_descs(self):
        """Deregister all cached descriptors in one pass."""
        if not self._registered_descs:
            return
        descs = list(self._registered_descs.values())
        self._registered_descs = {}
        for desc in descs:
            self.agent.deregister_memory(desc)

    def _merge_contiguous_regions(
        self, region_list: list[tuple[int, int, int, str]]
    ) -> list[tuple[int, int, int, str]]:
        """Merge contiguous memory regions to reduce registration calls.

        Args:
            region_list: List of (base_addr, nbytes, device_id, mem_type) tuples.
        Returns:
            Merged list of (base_addr, nbytes, device_id, mem_type) tuples.
        """
        if not region_list:
            return []

        # Sort regions by base address
        sorted_regions = sorted(region_list, key=lambda x: x[0])
        merged_regions = []
        current_base, current_size, current_device_id, current_mem_type = sorted_regions[0]

        for base, size, device_id, mem_type in sorted_regions[1:]:
            if current_base + current_size == base and current_device_id == device_id and current_mem_type == mem_type:
                # Merge contiguous region
                current_size += size
            else:
                # Append the current region and start a new one
                merged_regions.append((current_base, current_size, current_device_id, current_mem_type))
                current_base, current_size, current_device_id, current_mem_type = base, size, device_id, mem_type

        # Append the last region
        merged_regions.append((current_base, current_size, current_device_id, current_mem_type))
        return merged_regions

    def register_local_tensors(
        self,
        state_dict: dict[str, torch.Tensor],
        sharding_dict: dict[str, NIXLSharding] | None = None,
        binded_meta_tensor_mapping: (dict[tuple[str, tuple[int, ...]], torch.Tensor] | None) = None,
        meta_only: bool = False,
    ):
        """
        Register local tensors with NIXL. Build key->desc mapping.
        Args:
            state_dict: {key: torch.Tensor}
            sharding_dict: {key: NIXLSharding}
            binded_meta_tensor_mapping: {(key, shard_idx): torch.Tensor}
            meta_only: whether to skip registering real tensors
        """

        if (
            self.enable_nixl_for_temp_buffers
            and self.local_client_info is not None
            and self.local_client_info.is_registered
        ):
            # Re-register local tensors
            assert self._mtype_to_reg_region_lists is not None, "No registered regions found."
            for mem_type, reg_list in self._mtype_to_reg_region_lists.items():
                if not reg_list:
                    continue
                reg_descs = self.agent.register_memory(reg_list, mem_type=mem_type)
                self._track_registered_desc(reg_descs)

            # Rebuild desc bytes for all shards
            for (key, shard_idx), slice_info in self.contig_desc_slice_map.items():
                slice_addr, slice_len, device_id, mem_type = slice_info
                desc_bytes = self._build_xfer_desc_bytes(slice_addr, slice_len, device_id, mem_type)
                self.local_client_info.tensor_infos[key].desc_bytes_list[
                    self.local_client_info.tensor_infos[key].sharding.shard_indices.index(shard_idx)
                ] = desc_bytes

            for (key, shard_idx), slice_info in self.temp_desc_slice_map.items():
                slice_addr, slice_len, device_id, mem_type = slice_info
                desc_bytes = self._build_xfer_desc_bytes(slice_addr, slice_len, device_id, mem_type)
                self._temp_desc_bytes_mapping[(key, shard_idx)] = desc_bytes
                self.local_client_info.tensor_infos[key].temp_desc_bytes_list[
                    self.local_client_info.tensor_infos[key].sharding.shard_indices.index(shard_idx)
                ] = desc_bytes
            return

        tms_ctx = torch_memory_saver.region(tag="nixl") if self.enable_nixl_for_temp_buffers else nullcontext()
        with tms_ctx:
            # If pinned temp memory is enabled, we need to first scan the state_dict
            # and find all the tensors that are not contiguous
            # Then we need to find all types (shape and dtype) of uncontiguous tensor
            # and allocate max_pinned_temp_memory_slots times of their size as pinned memory
            # (each pinned memory tensor is like this: [max_pinned_temp_memory_slots, *])
            # Then we enumerate the uncontiguous tensors again and
            # map them with the pinned memory in a round-robin manner
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
                                _uncontiguous_tensor_mapping[
                                    (local_sharded_tensor.shape, local_sharded_tensor.dtype)
                                ] = []
                            _uncontiguous_tensor_mapping[
                                (local_sharded_tensor.shape, local_sharded_tensor.dtype)
                            ].append((key, shard_indices[local_pos], local_sharded_tensor))
                # Find all types of uncontiguous tensor and allocate pinned memory for them
                if _uncontiguous_tensor_mapping:
                    self._pinned_memory = {}
                    for (
                        shape,
                        dtype,
                    ), uncontiguous_tensor_list in _uncontiguous_tensor_mapping.items():
                        self._pinned_memory[(shape, dtype)] = []
                        for i, (key, shard_idx, uncontiguous_tensor) in enumerate(uncontiguous_tensor_list):
                            self._temp_pinned_idx_mapping[(key, shard_idx)] = i % self.max_pinned_temp_memory_slots
                        if meta_only:
                            continue

                        # Optimization: allocate a big pinned memory tensor and chunk it
                        # to reduce the number of registration calls
                        pinned_memory = torch.empty(
                            (self.max_pinned_temp_memory_slots, *shape),
                            dtype=dtype,
                            device=self.device,
                            requires_grad=False,
                        )
                        self._record_region_registration(pinned_memory)
                        memory_slots = torch.chunk(pinned_memory, self.max_pinned_temp_memory_slots, dim=0)
                        assert len(memory_slots) == self.max_pinned_temp_memory_slots, (
                            f"Expected {self.max_pinned_temp_memory_slots} memory slots, but got {len(memory_slots)}."
                        )
                        for slot in memory_slots:
                            memory_slot = slot.squeeze(0)
                            self._pinned_memory[(shape, dtype)].append(memory_slot)

            tensor_infos = {}
            for key, tensor in state_dict.items():
                assert key in sharding_dict, f"Key {key} not found in sharding_dict."
                sharding = sharding_dict[key]
                shard_indices = sharding.shard_indices
                # assert sharding.is_contiguous_sharding(), "Only contiguous sharding is supported for now."
                # Split registration
                desc_bytes_list = []
                temp_desc_bytes_list = []
                shard_meta_info_list = []

                # If the tensor is on meta device, allocate on-the-fly or binded from the external tensor
                # Case 1: allocate on-the-fly
                if tensor.device == torch.device("meta") and binded_meta_tensor_mapping is None and not meta_only:
                    tensor = torch.empty_like(tensor, device=self.device)
                local_sharded_tensors = sharding.get_local_sharded_tensors(tensor)

                # NOTE(linsh): we assume all shards have the same contiguity property
                if (
                    local_sharded_tensors[0].is_contiguous()
                    and tensor.device != torch.device("meta")
                    and not meta_only
                ):
                    self._record_region_registration(tensor)

                for local_pos, local_sharded_tensor in enumerate(local_sharded_tensors):
                    # Store the original tensor mapping
                    # If the tensor is on meta device, allocate on-the-fly or binded from the external tensor
                    # Case 2: binded from the external tensor
                    if (
                        local_sharded_tensor.device == torch.device("meta")
                        and binded_meta_tensor_mapping is not None
                        and (key, shard_indices[local_pos]) in binded_meta_tensor_mapping
                    ):
                        local_sharded_tensor = binded_meta_tensor_mapping[(key, shard_indices[local_pos])]
                    if not meta_only:
                        assert local_sharded_tensor.device == self.device, (
                            f"Local sharded tensor {key} shard {shard_indices[local_pos]} is not "
                            f"on device {self.device}, but on {local_sharded_tensor.device}, "
                            f"torch current device is {torch.cuda.current_device()}, "
                            f"CUDA_VISIBLE_DEVICES is {os.environ.get('CUDA_VISIBLE_DEVICES', 'None')}"
                        )
                    self._original_tensor_mapping[(key, shard_indices[local_pos])] = local_sharded_tensor

                    # Create meta info for shards
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
                        f"dtype {local_sharded_tensor.dtype}, is_contiguous = {is_contiguous}"
                    )
                    if not meta_only:
                        if local_sharded_tensor.is_contiguous():
                            # Contiguous shard: batch register
                            if binded_meta_tensor_mapping is None:
                                storage_key = self._record_region_registration(local_sharded_tensor)
                            slice_addr = int(local_sharded_tensor.data_ptr())
                            slice_len = int(local_sharded_tensor.numel() * local_sharded_tensor.element_size())
                            device_id = int(local_sharded_tensor.get_device()) if local_sharded_tensor.is_cuda else 0
                            mem_type = "cuda" if local_sharded_tensor.is_cuda else "cpu"
                            self.contig_desc_slice_map[(key, shard_indices[local_pos])] = (
                                slice_addr,
                                slice_len,
                                device_id,
                                mem_type,
                            )
                        else:
                            if self.max_pinned_temp_memory_slots is None:
                                # Non-contiguous shard: create temporary contiguous memory
                                # Create a new contiguous tensor with the same shape and dtype
                                contiguous_tensor = torch.empty_like(
                                    local_sharded_tensor,
                                    device=self.device,
                                    requires_grad=False,
                                )
                                storage_key = self._record_region_registration(contiguous_tensor)
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
                                pinned_slot = self._pinned_memory[
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
                                storage_key = self._record_region_registration(contiguous_tensor)

                            # Build the contiguous meta info
                            contiguous_meta_info = NIXLShardMetaInfo(
                                dtype=contiguous_tensor.dtype,
                                device=contiguous_tensor.device,
                                shape=contiguous_tensor.shape,
                                stride=contiguous_tensor.stride(),
                                is_contiguous=True,
                            )
                            # Store temporary mappings
                            temp_slice_addr = int(contiguous_tensor.data_ptr())
                            temp_slice_len = int(contiguous_tensor.numel() * contiguous_tensor.element_size())
                            self.temp_desc_slice_map[(key, shard_indices[local_pos])] = (
                                temp_slice_addr,
                                temp_slice_len,
                                storage_key[2],
                                storage_key[3],
                            )
                            self._temp_tensor_mapping[(key, shard_indices[local_pos])] = contiguous_tensor
                            self._temp_meta_mapping[(key, shard_indices[local_pos])] = contiguous_meta_info

                    # Placeholder for desc bytes, will be filled after global registration
                    desc_bytes_list.append(None)
                    temp_desc_bytes_list.append(None)

                # Create the tensor descriptor info
                tensor_infos[key] = NIXLTensorInfo(
                    desc_bytes_list=desc_bytes_list,
                    temp_desc_bytes_list=temp_desc_bytes_list,
                    sharding=sharding,
                    shard_meta_infos=shard_meta_info_list,
                )

            # Batch register all tensors and cache the reg list once.
            if not meta_only and self._mtype_to_reg_region_lists:
                for mem_type, reg_list in self._mtype_to_reg_region_lists.items():
                    if not reg_list:
                        continue
                    reg_list = self._merge_contiguous_regions(reg_list)
                    self._mtype_to_reg_region_lists[mem_type] = reg_list
                    reg_descs = self.agent.register_memory(reg_list, mem_type=mem_type)
                    self._track_registered_desc(reg_descs)

                for (key, shard_idx), slice_info in self.contig_desc_slice_map.items():
                    slice_addr, slice_len, device_id, mem_type = slice_info
                    desc_bytes = self._build_xfer_desc_bytes(slice_addr, slice_len, device_id, mem_type)
                    tensor_infos[key].desc_bytes_list[tensor_infos[key].sharding.shard_indices.index(shard_idx)] = (
                        desc_bytes
                    )

                for (key, shard_idx), slice_info in self.temp_desc_slice_map.items():
                    slice_addr, slice_len, device_id, mem_type = slice_info
                    desc_bytes = self._build_xfer_desc_bytes(slice_addr, slice_len, device_id, mem_type)
                    self._temp_desc_bytes_mapping[(key, shard_idx)] = desc_bytes
                    tensor_infos[key].temp_desc_bytes_list[
                        tensor_infos[key].sharding.shard_indices.index(shard_idx)
                    ] = desc_bytes

            # Create the client info
            self.local_client_info = NIXLClientInfo(
                name=self.client_name,
                node_ip=get_local_ip(),
                node_gpu_id=get_local_gpu_id(),
                type=self.client_type,
                tensor_infos=tensor_infos,
                meta=self.agent.get_agent_metadata(),
                client_group_id=self.client_group_id,
                is_registered=not meta_only,
            )
            psrl_logger.debug(
                f"Local client info is built, "
                f"temp pinned idx mapping is: {self._temp_pinned_idx_mapping}, "
                f"temp meta mapping is: {self._temp_meta_mapping}"
            )

    def deregister_local_tensors(self):
        """Deregister all local tensors"""
        assert self.local_client_info is not None, "Local client info not registered."
        # Deregister all registered regions.
        self._deregister_all_descs()
        if not self.enable_nixl_for_temp_buffers:
            self.release_temp_memory()

            # Clear all mappings
            self.local_client_info = None
            self._original_tensor_mapping = {}
            self._pinned_slot_running_read_xfer = {}
            self._pinned_slot_running_write_xfer = {}
            self._read_contiguous_event_cache = {}
            self._write_contiguous_event_cache = {}
            self.xfer_handles = {}
            self._reg_regions = set()
            self._mtype_to_reg_region_lists = {}
            self.contig_desc_slice_map = {}

        psrl_logger.debug(f"{self.client_name} deregistered all local tensors.")

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

    def send_local_info_to(self, dst_agent_names: list[str]):
        """Send local client info to specified destination agents (meta_server mode).

        Args:
            dst_agent_names: List of destination agent names.
        """
        assert self._is_connected, "Not connected to server"
        payload_dict = {
            self.client_name: {
                "info": self.local_client_info.serialize(),
                "temp_mapping": self._temp_desc_bytes_mapping,
            }
        }
        payload = pickle.dumps(payload_dict)
        for dst_agent_name in dst_agent_names:
            self.agent.send_notif(dst_agent_name, payload)

    def wait_for_update_infos(self, expected_agents: int, timeout: float = 600.0):
        """Wait for updated client infos from other clients (meta_server mode).

        Args:
            expected_agents: Number of expected client infos to be updated.
            timeout: Timeout in seconds.
        """
        assert self._is_connected, "Not connected to server"
        psrl_logger.info(f"{self.client_name}: Waiting for {expected_agents} updated client infos...")
        start = time.time()
        already_recved_agents = set()
        while len(already_recved_agents) < expected_agents:
            notifs = self.agent.get_new_notifs()
            for agent_name, notif_list in notifs.items():
                for notif in notif_list:
                    try:
                        multi_infos = pickle.loads(notif)
                        assert isinstance(multi_infos, dict), f"Expected a dict of multi_infos, but got {multi_infos}"
                        for client_name, info_and_temp_mapping in multi_infos.items():
                            info = info_and_temp_mapping["info"]
                            client_temp_mapping = info_and_temp_mapping["temp_mapping"]
                            client_info = NIXLClientInfo.deserialize(info)
                            self._all_client_infos[client_name] = client_info
                            self._all_temp_mappings[client_name] = client_temp_mapping
                        already_recved_agents.add(agent_name)
                        psrl_logger.info(
                            f"Already received {len(already_recved_agents)} agents: {already_recved_agents}"
                        )
                    except Exception:
                        continue
            if time.time() - start > timeout:
                raise TimeoutError("Timeout waiting for agents.")
            time.sleep(0.1)

    def broadcast_update_client_infos(self, dst_agent_names: list[str], update_client_names: list[str]):
        """Broadcast updated client infos to specified destination agents (meta_server mode).

        Args:
            dst_agent_names: List of destination agent names.
            update_client_names: List of client names whose infos are to be broadcasted.
        """
        payload_dict = {}
        for client_name in update_client_names:
            client_info = self._all_client_infos[client_name]
            client_temp_mapping = self._all_temp_mappings[client_name]
            payload_dict[client_name] = {"info": client_info.serialize(), "temp_mapping": client_temp_mapping}
        payload = pickle.dumps(payload_dict)
        for dst_agent_name in dst_agent_names:
            self.agent.send_notif(dst_agent_name, payload)

    # --- storage_server mode read/write  ---
    @deprecated("Use client_read instead")
    def read(self, key: str, tag: str):
        """
        Read from the storage server.
        """
        if self.mode != "storage_server":
            raise RuntimeError("read(key, tag) only valid in storage_server mode")
        self.wait_for_server_info()
        local_desc = self._ensure_xfer_descs(self.local_client_info.get_tensor_desc(self.agent, 0))
        server_desc = self._ensure_xfer_descs(self.server_client_info.get_tensor_desc(self.agent, 0))
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
        local_desc = self._ensure_xfer_descs(self.local_client_info.get_tensor_desc(self.agent, 0))
        server_desc = self._ensure_xfer_descs(self.server_client_info.get_tensor_desc(self.agent, 0))
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
            pull_plan = plan.get_rollout_pull_plan(self.client_name, key)
            if target_client in pull_plan:
                shards_to_transfer = pull_plan[target_client]
        elif plan and self.client_type == NIXLClientType.PUSH_SIDE:
            push_plan = plan.get_train_pull_plan(self.client_name, key)
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
                    f"{meta_info} != {remote_info.shard_meta_infos[remote_pos]} "
                    f"during client_read from {self.client_name} to {target_client}"
                )
                # Use temporary descriptor for non-contiguous shard
                local_desc_bytes = self._get_temp_desc_bytes(self.client_name, key, shard_idx)
                if local_desc_bytes is None:
                    raise RuntimeError(f"No temporary descriptor found for key {key} shard {shard_idx}")
                # Wait for the pinned slot to be available
                if self.max_pinned_temp_memory_slots is not None:
                    pinned_idx = self._temp_pinned_idx_mapping[(key, shard_idx)]
                    slot_key = (meta_info.shape, meta_info.dtype, pinned_idx)
                    if slot_key in self._pinned_slot_running_read_xfer:
                        (
                            running_key,
                            running_tag,
                            running_op_type,
                            running_target_client,
                            running_shard_idx,
                        ) = self._pinned_slot_running_read_xfer[slot_key]
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
                    self._pinned_slot_running_read_xfer[slot_key] = (
                        key,
                        tag,
                        "READ",
                        target_client,
                        shard_idx,
                    )

            # Get remote descriptor (check if it's a temporary one)
            remote_desc_bytes = remote_info.desc_bytes_list[remote_pos]
            if remote_desc_bytes is None:
                # Use temporary descriptor for non-contiguous shard
                remote_desc_bytes = remote_info.temp_desc_bytes_list[remote_pos]
                if remote_desc_bytes is None:
                    raise RuntimeError(f"No remote temporary descriptor found for key {key} shard {shard_idx}")

            # Double check the shard size
            assert local_info.get_shard_size_bytes(local_pos) == remote_info.get_shard_size_bytes(remote_pos), (
                f"Shard size mismatch for key {key} shard {shard_idx}: "
                f"{local_info.get_shard_size_bytes(local_pos)} != {remote_info.get_shard_size_bytes(remote_pos)}"
            )
            local_desc = self._deserialize_to_xfer_descs(local_desc_bytes)
            remote_desc = self._deserialize_to_xfer_descs(remote_desc_bytes)
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
                    ) in self._read_contiguous_event_cache, (
                        f"Running key {running_key} shard {running_shard_idx} not found in contiguous event cache"
                    )
                    self._read_contiguous_event_cache[(running_key, running_shard_idx)].synchronize()
                    self._read_contiguous_event_cache.pop((running_key, running_shard_idx))
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
                    if slot_key in self._pinned_slot_running_write_xfer:
                        (
                            running_key,
                            running_tag,
                            running_op_type,
                            running_target_client,
                            running_shard_idx,
                        ) = self._pinned_slot_running_write_xfer[slot_key]
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
                    self._pinned_slot_running_write_xfer[slot_key] = (
                        key,
                        tag,
                        "WRITE",
                        target_client,
                        shard_idx,
                    )
                # Copy data from original non-contiguous tensor to temporary contiguous tensor
                self._write_contiguous_event_cache[(key, shard_idx)] = torch.cuda.Event()
                contiguous_tensor.copy_(original_tensor.detach())
                self._write_contiguous_event_cache[(key, shard_idx)].record()
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
                # Use temporary descriptor for non-contiguous shard
                remote_desc_bytes = remote_info.temp_desc_bytes_list[remote_pos]
                if remote_desc_bytes is None:
                    raise RuntimeError(f"No remote temporary descriptor found for key {key} shard {shard_idx}")

            # Double check the shard size
            assert local_info.get_shard_size_bytes(local_pos) == remote_info.get_shard_size_bytes(remote_pos), (
                f"Shard size mismatch for key {key} shard {shard_idx}: "
                f"{local_info.get_shard_size_bytes(local_pos)} != {remote_info.get_shard_size_bytes(remote_pos)}"
            )
            local_desc = self._deserialize_to_xfer_descs(local_desc_bytes)
            remote_desc = self._deserialize_to_xfer_descs(remote_desc_bytes)
            if merge_and_cache_xfer and is_contiguous:
                self._cached_xfer_descs.append(("WRITE", local_desc, remote_desc, target_agent, tag, target_client))
                return []
            # Real xfer
            try:
                if (key, shard_idx) in self._write_contiguous_event_cache:
                    self._write_contiguous_event_cache[(key, shard_idx)].synchronize()
                    self._write_contiguous_event_cache.pop((key, shard_idx))
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
                                self._read_contiguous_event_cache[(key, shard_idx)] = torch.cuda.Event()
                                original_tensor.copy_(contiguous_tensor)
                                self._read_contiguous_event_cache[(key, shard_idx)].record()
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
        self._deregister_all_descs()


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
        replica_idx: int = 0,
        worker_index: int = 0,
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
        # Initialize NIXL agent
        self.client_port = (
            find_free_port_with_scope(
                replica_idx=replica_idx,
                worker_index=worker_index,
            )
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
                    replica_idx=replica_idx,
                    worker_index=worker_index,
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

    def wait_for_update_infos(self, expected_agents: int, timeout: float = 600.0):
        assert self._is_connected, "Not connected to server"
        self.multi_clients[0].wait_for_update_infos(expected_agents, timeout)
        if len(self.multi_clients) > 1:
            for client in self.multi_clients[1:]:
                client._all_client_infos = self.multi_clients[0]._all_client_infos
                client._all_temp_mappings = self.multi_clients[0]._all_temp_mappings

    def broadcast_update_client_infos(self, dst_agent_names: list[str], update_client_names: list[str]):
        assert self._is_connected, "Not connected to server"
        self.multi_clients[0].broadcast_update_client_infos(dst_agent_names, update_client_names)

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
