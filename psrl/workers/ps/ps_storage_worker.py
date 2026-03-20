import gc
import json
import os
from dataclasses import dataclass

import torch
from accelerate import init_empty_weights
from omegaconf import DictConfig
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq
from verl.utils.fs import copy_to_local

from psrl.utils.converter.hf_converter import convert_hf_inplace
from psrl.utils.logger import get_ps_logger, get_worker_info, setup_ps_logger
from psrl.utils.nixl import (
    GLOBAL_META_SERVER_NAME,
    GLOBAL_PS_CLIENT_NAME,
    NIXLClientType,
    NIXLInterface,
    NIXLMultiStorageClients,
)

# Use the unified PS logger
psrl_logger = get_ps_logger()


# TODO(lhy): Implement the PSStoragePlan
# support zero/half/full redundancy for PSStorageWorker
@dataclass
class PSStoragePlan:
    train_model_dtype: torch.dtype
    gen_model_dtype: torch.dtype
    storage_style: str = "hf"  # ["hf", "hsdp"]
    hsdp_pattern: dict[str, int] | None = None  # e.g. {"replicate": 1, "fully_shard": 2}

    def __post_init__(self):
        if self.storage_style == "hsdp":
            assert self.hsdp_pattern is not None, "hsdp_pattern is required."
            assert all(isinstance(value, int) for value in self.hsdp_pattern.values()), (
                "hsdp_pattern values must be integers."
            )
            assert all(value >= 0 for value in self.hsdp_pattern.values()), "hsdp_pattern values must be non-negative."

    def train_gen_model_share(self) -> bool:
        return self.train_model_dtype == self.gen_model_dtype


class PSStorageWorker:
    """A worker that only stores the data and uses NIXL to communicate."""

    def __init__(
        self,
        storage_plan: PSStoragePlan,
        model_config: DictConfig,
        psrl_config: DictConfig,
        nixl_interface: NIXLInterface,
    ) -> None:
        self.storage_plan = storage_plan
        self.model_config = model_config
        self.psrl_config = psrl_config
        self.nixl_interface = nixl_interface
        self.train_meta_hf_model: torch.nn.Module | None = None
        self.gen_meta_hf_model: torch.nn.Module | None = None

        # NIXL
        self.nixl_multi_storage_clients = None

        # Build logger
        self.rank = int(os.environ.get("RANK"))
        self.log_prefix = f"PSStorageWorker_R{self.rank}"
        setup_ps_logger(self.psrl_config.logging_path, self.log_prefix)
        psrl_logger.info(f"Initialized on {get_worker_info()}.")

        # NOTE(lhy): currently hard code the net device to bond1
        # os.environ["UCX_NET_DEVICES"] = "bond1"

    def get_replica_id(self) -> int:
        """
        Get the replica id (dp id) of the storage worker.
        """
        if self.storage_plan.storage_style == "hf":
            return self.rank
        elif self.storage_plan.storage_style == "hsdp":
            return self.rank // self.storage_plan.hsdp_pattern["fully_shard"]
        else:
            raise ValueError(f"Invalid storage style: {self.storage_plan.storage_style}")

    def init_nixl_client(self):
        """Initialize the NIXL client."""
        # NOTE(lhy): the init_nixl_client is called before the initialization of the actor module now
        # Because in UCX 1.18.0, this may enhance the communication performance
        # assert self.train_meta_hf_model and self.gen_meta_hf_model, \
        #     "The HuggingFace models must be initialized before calling init_nixl_client."
        if self.psrl_config.nixl.server_mode == "storage_server":
            raise ValueError("Storage server mode is deprecated.")
        elif self.psrl_config.nixl.server_mode == "meta_server":
            self.use_gpu = self.psrl_config.ps_mode == "nixl_gpu"
            # TODO(lhy): maybe support train and gen use different ps mode
            self.agent_name = f"{GLOBAL_PS_CLIENT_NAME}_{self.rank}"
            self.client_for_push_name = f"{self.agent_name}_for_push"
            self.client_for_pull_name = f"{self.agent_name}_for_pull"
            self.nixl_multi_storage_clients = NIXLMultiStorageClients(
                agent_name=self.agent_name,
                multi_client_names=[
                    self.client_for_push_name,
                    self.client_for_pull_name,
                ],
                server_name=GLOBAL_META_SERVER_NAME,
                use_gpu=self.use_gpu,
                multi_client_types=[
                    NIXLClientType.PS_FOR_PUSH,
                    NIXLClientType.PS_FOR_PULL,
                ],
                nixl_config=self.psrl_config.nixl,
                nixl_interface=self.nixl_interface,
                # client_group_id=self.get_replica_id(),
                logging_path=self.psrl_config.logging_path,
            )
        else:
            raise ValueError(f"Invalid NIXL server mode: {self.psrl_config.nixl.server_mode}")
        psrl_logger.info(
            f"NIXL multi storage clients initialized on port {self.nixl_multi_storage_clients.client_port}."
        )

    def _nixl_protocol_phase1(self):
        """Execute protocol phase 1: from step 0 to step 3 (before register_local_tensors)."""
        psrl_logger.info("nixl client protocol step 0: convert_hf_inplace")
        unified_train_meta_state_dict, local_train_sharding_dict = convert_hf_inplace(self.train_meta_hf_model)
        unified_gen_meta_state_dict, local_gen_sharding_dict = convert_hf_inplace(self.gen_meta_hf_model)
        unified_multi_meta_state_dicts = {
            self.client_for_push_name: unified_train_meta_state_dict,
            self.client_for_pull_name: unified_gen_meta_state_dict,
        }
        psrl_logger.info("nixl client protocol step 1: connect_to_server")
        self.nixl_multi_storage_clients.connect_to_server()
        psrl_logger.info("nixl client protocol step 2: send_local_sharding")
        multi_local_sharding_dicts = {
            self.client_for_push_name: local_train_sharding_dict,
            self.client_for_pull_name: local_gen_sharding_dict,
        }
        self.nixl_multi_storage_clients.send_local_sharding(multi_local_sharding_dicts)
        psrl_logger.info("nixl client protocol step 3: wait_for_server_sharding")
        unified_multi_sharding_dicts = self.nixl_multi_storage_clients.wait_for_server_sharding()
        for client_name, sharding_dict in unified_multi_sharding_dicts.items():
            assert sharding_dict is not None, f"Sharding dict for client {client_name} is None"
        return unified_multi_meta_state_dicts, unified_multi_sharding_dicts

    def _nixl_protocol_phase2(self):
        """Execute protocol phase 2: from step 5 to step 8 (remaining steps)."""
        psrl_logger.info("nixl client protocol step 5: send_local_info")
        self.nixl_multi_storage_clients.send_local_info()
        psrl_logger.info("nixl client protocol step 6: wait_for_server_info")
        self.nixl_multi_storage_clients.wait_for_server_info()
        psrl_logger.info("nixl client protocol step 7: send_local_temp_mapping")
        self.nixl_multi_storage_clients.send_local_temp_mapping()
        psrl_logger.info("nixl client protocol step 8: wait_for_server_temp_mappings")
        self.nixl_multi_storage_clients.wait_for_server_temp_mappings()
        psrl_logger.info("nixl client protocol done.")

    def nixl_protocol(self):
        psrl_logger.info("nixl protocol start with two phases.")
        unified_multi_meta_state_dicts, unified_multi_sharding_dicts = self._nixl_protocol_phase1()
        # Sequentially register in the main thread (thread-safe torch allocation)
        psrl_logger.info("nixl client protocol step 4: register_local_tensors")
        client_for_push = self.nixl_multi_storage_clients.get_client_by_name(self.client_for_push_name)
        client_for_pull = self.nixl_multi_storage_clients.get_client_by_name(self.client_for_pull_name)
        client_for_push.register_local_tensors(
            unified_multi_meta_state_dicts[self.client_for_push_name],
            unified_multi_sharding_dicts[self.client_for_push_name],
        )
        if self.storage_plan.train_gen_model_share():
            original_tensor_mapping = client_for_push.get_original_tensor_mapping()
            client_for_pull.register_local_tensors(
                unified_multi_meta_state_dicts[self.client_for_pull_name],
                unified_multi_sharding_dicts[self.client_for_pull_name],
                binded_meta_tensor_mapping=original_tensor_mapping,
            )
        else:
            # raise NotImplementedError("Gen model not share with train model is not implemented yet.")
            client_for_pull.register_local_tensors(
                unified_multi_meta_state_dicts[self.client_for_pull_name],
                unified_multi_sharding_dicts[self.client_for_pull_name],
            )
        self._nixl_protocol_phase2()

    def nixl_wait_for_update_infos(self, info_num: int):
        """Wait for update infos from the storage client.

        Args:
            info_num (int): Number of infos to wait for
        """
        self.nixl_multi_storage_clients.wait_for_update_infos(info_num)

    def nixl_broadcast_update_client_infos(self, dst_agent_names: list[str], update_client_names: list[str]):
        """Broadcast updated client infos to destination agents.

        Args:
            dst_agent_names (list[str]): List of destination agent names
            update_client_names (list[str]): List of updated client names
        """
        self.nixl_multi_storage_clients.broadcast_update_client_infos(dst_agent_names, update_client_names)

    def get_nixl_agent_name(self) -> str:
        """Get the name of the NIXL agent."""
        return self.agent_name

    def get_nixl_train_storage_client_name(self) -> str:
        """Get the name of the NIXL train storage client."""
        return self.client_for_push_name

    def get_nixl_gen_storage_client_name(self) -> str:
        """Get the name of the NIXL gen storage client."""
        return self.client_for_pull_name

    def init_model(self):
        """
        Initialize the model skeleton on the meta device.

        Only the parameter shapes / dtypes are materialised here; no actual
        weight data is loaded.  Call ``load_weights_to_registered_tensors()``
        *after* the full NIXL protocol (``nixl_protocol()``) has completed so
        that all meta-device tensors have been replaced by real allocated
        buffers, and only then copy the checkpoint weights into them.
        """
        local_path = copy_to_local(self.model_config.path, use_shm=self.model_config.get("use_shm", False))
        model_config = AutoConfig.from_pretrained(
            local_path,
            trust_remote_code=self.model_config.get("trust_remote_code", False),
        )
        if type(model_config) in AutoModelForVision2Seq._model_mapping.keys():
            model_class = AutoModelForVision2Seq
        else:
            model_class = AutoModelForCausalLM

        if self.psrl_config.ps_mode in ("nixl_cpu", "nixl_gpu"):
            with init_empty_weights():
                self.train_meta_hf_model = model_class.from_config(
                    model_config,
                    torch_dtype=self.storage_plan.train_model_dtype,
                    trust_remote_code=self.model_config.get("trust_remote_code", False),
                )
                if self.storage_plan.train_gen_model_share():
                    self.gen_meta_hf_model = self.train_meta_hf_model
                else:
                    self.gen_meta_hf_model = model_class.from_config(
                        model_config,
                        torch_dtype=self.storage_plan.gen_model_dtype,
                        trust_remote_code=self.model_config.get("trust_remote_code", False),
                    )
        else:
            raise ValueError(f"Invalid PS mode: {self.psrl_config.ps_mode}")
        psrl_logger.info(f"init_model (meta-only) done on {get_worker_info()}.")

    # ------------------------------------------------------------------
    # Post-protocol weight loading
    # ------------------------------------------------------------------

    def load_weights_to_registered_tensors(self):
        """
        Stream HuggingFace checkpoint weights into the already-registered NIXL
        buffers (``_original_tensor_mapping`` entries inside each sub-client).

        Must be called **after** ``nixl_protocol()`` has completed so that all
        meta-device tensors have been replaced by real allocated slices.

        Memory strategy
        ---------------
        * Checkpoint shards are opened lazily with ``safetensors.safe_open``
          (one file at a time, one tensor at a time).
        * Each source tensor is immediately deleted after being copied into the
          registered destination(s), so peak extra RAM equals roughly the size
          of a single parameter tensor.
        * When train and gen clients share the same underlying buffers
          (``train_gen_model_share() == True``) the copy is done only once.
        * The HF meta-model references are released at the end to free any
          remaining structure overhead.
        """
        assert self.nixl_multi_storage_clients is not None, (
            "NIXL clients must be initialized (call init_nixl_client()) before loading weights."
        )

        local_path = copy_to_local(self.model_config.path, use_shm=self.model_config.get("use_shm", False))
        shard_files = self._discover_safetensors_shards(local_path)
        psrl_logger.info(f"load_weights_to_registered_tensors: {len(shard_files)} shard file(s) under {local_path}")

        push_client = self.nixl_multi_storage_clients.get_client_by_name(self.client_for_push_name)
        pull_client = self.nixl_multi_storage_clients.get_client_by_name(self.client_for_pull_name)
        shared = self.storage_plan.train_gen_model_share()

        # Collect all keys expected by every sub-client so we can warn about missing ones.
        expected_keys: set[str] = set(push_client.local_client_info.tensor_infos.keys())
        if not shared:
            expected_keys |= set(pull_client.local_client_info.tensor_infos.keys())

        loaded_keys: set[str] = set()

        for shard_file in shard_files:
            psrl_logger.info(f"load_weights_to_registered_tensors: opening {shard_file}")
            with safe_open(shard_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key not in expected_keys:
                        continue  # this PS worker doesn't hold this key
                    src_tensor = f.get_tensor(key)  # loaded to CPU
                    # Delegate slicing + copy to each client
                    push_client.load_state_dict_into_registered_tensors({key: src_tensor})
                    if not shared:
                        pull_client.load_state_dict_into_registered_tensors({key: src_tensor})
                    del src_tensor
                    loaded_keys.add(key)

        missing = expected_keys - loaded_keys
        if missing:
            raise RuntimeError(
                f"load_weights_to_registered_tensors: {len(missing)} key(s) not found in checkpoint: "
                f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}"
            )
        psrl_logger.info(f"load_weights_to_registered_tensors: loaded {len(loaded_keys)}/{len(expected_keys)} key(s).")

        gc.collect()

    @staticmethod
    def _discover_safetensors_shards(local_path: str) -> list[str]:
        """
        Return an ordered list of absolute safetensors shard file paths.

        Looks for (in priority order):
        1. ``model.safetensors.index.json``  — multi-shard checkpoint
        2. ``model.safetensors``             — single-file checkpoint

        Raises a helpful ``FileNotFoundError`` / ``RuntimeError`` for unknown
        or unsupported (pytorch_model.bin) formats.
        """
        index_json = os.path.join(local_path, "model.safetensors.index.json")
        single_sf = os.path.join(local_path, "model.safetensors")

        if os.path.isfile(index_json):
            with open(index_json) as fh:
                index = json.load(fh)
            # weight_map: param_name -> relative shard filename
            weight_map: dict[str, str] = index.get("weight_map", index)
            # Deduplicate while preserving encounter order
            seen: set[str] = set()
            ordered: list[str] = []
            for rel_path in weight_map.values():
                if rel_path not in seen:
                    seen.add(rel_path)
                    ordered.append(os.path.join(local_path, rel_path))
            return ordered

        if os.path.isfile(single_sf):
            return [single_sf]

        # Legacy pytorch_model.bin — not supported
        pt_bin = os.path.join(local_path, "pytorch_model.bin")
        pt_idx = os.path.join(local_path, "pytorch_model.bin.index.json")
        if os.path.isfile(pt_bin) or os.path.isfile(pt_idx):
            raise RuntimeError(
                f"Checkpoint at '{local_path}' uses pytorch_model.bin format, which is not supported. "
                "Convert it to safetensors first:\n"
                "  python -c \"from transformers import AutoModel; m = AutoModel.from_pretrained('<path>'); "
                "m.save_pretrained('<path>', safe_serialization=True)\""
            )

        raise FileNotFoundError(
            f"No safetensors checkpoint found under '{local_path}'. "
            "Expected 'model.safetensors' or 'model.safetensors.index.json'."
        )

    def _build_transfer_key_cache(self, src_original_state_dict):
        self._transfer_key_cache = {"src_dict_id": id(src_original_state_dict)}
        for key_tuple in src_original_state_dict.keys():
            k, shard_idx = key_tuple
            if k not in self._transfer_key_cache:
                self._transfer_key_cache[k] = []
            self._transfer_key_cache[k].append((k, shard_idx))

    def transfer_train_to_gen(self, key: str, shards_to_transfer: list[tuple[int, ...]] | None = None):
        if self.storage_plan.train_gen_model_share():
            return
        src_client = self.nixl_multi_storage_clients.get_client_by_name(self.client_for_push_name)
        target_client = self.nixl_multi_storage_clients.get_client_by_name(self.client_for_pull_name)
        src_original_state_dict = src_client.get_original_tensor_mapping()
        target_original_state_dict = target_client.get_original_tensor_mapping()
        # src_temp_state_dict = src_client.get_temp_tensor_mapping()
        # target_temp_state_dict = target_client.get_temp_tensor_mapping()
        # assert len(src_temp_state_dict) == 0 and len(target_temp_state_dict) == 0, "Temp state dict should be empty"
        if not hasattr(self, "_transfer_key_cache") or self._transfer_key_cache.get("src_dict_id") != id(
            src_original_state_dict
        ):
            self._build_transfer_key_cache(src_original_state_dict)
        matching_keys = self._transfer_key_cache.get(key, [])
        for key_shard_idx_tuple in matching_keys:
            if shards_to_transfer is not None and key_shard_idx_tuple[1] not in shards_to_transfer:
                continue
            target_original_state_dict[key_shard_idx_tuple].copy_(src_original_state_dict[key_shard_idx_tuple])
        if self.use_gpu:
            torch.cuda.synchronize()

    def shutdown(self):
        self.nixl_multi_storage_clients.shutdown()

    def debug_log_info(self, label: str = ""):
        """
        Log info for both push (train) and pull (gen) clients on this PS worker.
        Called via Ray RPC from the train worker for precision debugging.
        """
        push_client = self.nixl_multi_storage_clients.get_client_by_name(self.client_for_push_name)
        pull_client = self.nixl_multi_storage_clients.get_client_by_name(self.client_for_pull_name)
        push_client.log_shard_info(label=label)
        pull_client.log_shard_info(label=label)
