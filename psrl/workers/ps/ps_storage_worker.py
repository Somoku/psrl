import ray
import torch
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq
from omegaconf import DictConfig
from accelerate import init_empty_weights

from verl.utils.fs import copy_to_local

from psrl.utils.nixl import NIXLClientType, NIXLInterface, NIXLMultiStorageClients, GLOBAL_META_SERVER_NAME, GLOBAL_PS_CLIENT_NAME
from psrl.utils.converter.hf_converter import convert_hf_inplace
from psrl.utils.logger import get_ps_logger, setup_ps_logger, get_worker_info, log_dual_events, log_single_event, EventType

# Use the unified PS logger
psrl_logger = get_ps_logger()


# TODO(lhy): Implement the PSStoragePlan
# support zero/half/full redundancy for PSStorageWorker
@dataclass
class PSStoragePlan:
    train_model_dtype: torch.dtype
    gen_model_dtype: torch.dtype
    
    def train_gen_model_share(self) -> bool:
        return self.train_model_dtype == self.gen_model_dtype


class PSStorageWorker:
    """A worker that only stores the data and uses NIXL to communicate."""
    
    def __init__(self, storage_plan: PSStoragePlan, model_config: DictConfig, psrl_config: DictConfig, nixl_interface: NIXLInterface) -> None:
        self.storage_plan = storage_plan
        self.model_config = model_config
        self.psrl_config = psrl_config
        self.nixl_interface = nixl_interface
        self.train_meta_hf_model: Optional[torch.nn.Module] = None
        self.gen_meta_hf_model: Optional[torch.nn.Module] = None
        
        # NIXL
        self.nixl_multi_storage_clients = None
        
        # Build logger
        self.rank = int(os.environ.get("RANK"))
        self.log_prefix = f"PSStorageWorker_R{self.rank}"
        setup_ps_logger(self.psrl_config.logging_path, self.log_prefix)
        psrl_logger.info(f"Initialized on {get_worker_info()}.")
        
    def init_nixl_client(self):
        """Initialize the NIXL client."""
        assert self.train_meta_hf_model and self.gen_meta_hf_model, "The HuggingFace models must be initialized before calling init_nixl_client."
        if self.psrl_config.nixl.server_mode == "storage_server":
            raise ValueError("Storage server mode is deprecated.")
        elif self.psrl_config.nixl.server_mode == "meta_server":
            use_gpu = self.psrl_config.ps_mode == "nixl_gpu"
            # TODO(lhy): maybe support train and gen use different ps mode
            self.agent_name = f"{GLOBAL_PS_CLIENT_NAME}_{self.rank}"
            self.client_for_push_name = f"{self.agent_name}_for_push"
            self.client_for_pull_name = f"{self.agent_name}_for_pull"
            self.nixl_multi_storage_clients = NIXLMultiStorageClients(
                agent_name=self.agent_name,
                multi_client_names=[self.client_for_push_name, self.client_for_pull_name],
                server_name=GLOBAL_META_SERVER_NAME,
                use_gpu=use_gpu,
                multi_client_types=[NIXLClientType.PS_FOR_PUSH, NIXLClientType.PS_FOR_PULL],
                nixl_config=self.psrl_config.nixl,
                nixl_interface=self.nixl_interface
            )
        else:
            raise ValueError(f"Invalid NIXL server mode: {self.psrl_config.nixl.server_mode}")
        psrl_logger.info(f"NIXL multi storage clients initialized on port {self.nixl_multi_storage_clients.client_port}.")
        
    def _nixl_protocol_phase1(self):
        """Execute protocol phase 1: from step 0 to step 3 (before register_local_tensors)."""
        psrl_logger.info(f"nixl client protocol step 0: convert_hf_inplace")
        unified_train_meta_state_dict, local_train_sharding_dict = convert_hf_inplace(self.train_meta_hf_model)
        unified_gen_meta_state_dict, local_gen_sharding_dict = convert_hf_inplace(self.gen_meta_hf_model)
        unified_multi_meta_state_dicts = {
            self.client_for_push_name: unified_train_meta_state_dict,
            self.client_for_pull_name: unified_gen_meta_state_dict
        }
        psrl_logger.info(f"nixl client protocol step 1: connect_to_server")
        self.nixl_multi_storage_clients.connect_to_server()
        psrl_logger.info(f"nixl client protocol step 2: send_local_sharding")
        multi_local_sharding_dicts = {
            self.client_for_push_name: local_train_sharding_dict,
            self.client_for_pull_name: local_gen_sharding_dict
        }
        self.nixl_multi_storage_clients.send_local_sharding(multi_local_sharding_dicts)
        psrl_logger.info(f"nixl client protocol step 3: wait_for_server_sharding")
        unified_multi_sharding_dicts = self.nixl_multi_storage_clients.wait_for_server_sharding()
        for client_name, sharding_dict in unified_multi_sharding_dicts.items():
            assert sharding_dict is not None, f"Sharding dict for client {client_name} is None"
        return unified_multi_meta_state_dicts, unified_multi_sharding_dicts
            
    def _nixl_protocol_phase2(self):
        """Execute protocol phase 2: from step 5 to step 8 (remaining steps)."""
        psrl_logger.info(f"nixl client protocol step 5: send_local_info")
        self.nixl_multi_storage_clients.send_local_info()
        psrl_logger.info(f"nixl client protocol step 6: wait_for_server_info")
        self.nixl_multi_storage_clients.wait_for_server_info()
        psrl_logger.info(f"nixl client protocol step 7: send_local_temp_mapping")
        self.nixl_multi_storage_clients.send_local_temp_mapping()
        psrl_logger.info(f"nixl client protocol step 8: wait_for_server_temp_mappings")
        self.nixl_multi_storage_clients.wait_for_server_temp_mappings()
        psrl_logger.info(f"nixl client protocol done.")
           
    def nixl_protocol(self):
        psrl_logger.info("nixl protocol start with two phases.")
        unified_multi_meta_state_dicts, unified_multi_sharding_dicts = self._nixl_protocol_phase1()
        # Sequentially register in the main thread (thread-safe torch allocation)
        psrl_logger.info("nixl client protocol step 4: register_local_tensors")
        client_for_push = self.nixl_multi_storage_clients.get_client_by_name(self.client_for_push_name)
        client_for_pull = self.nixl_multi_storage_clients.get_client_by_name(self.client_for_pull_name)
        client_for_push.register_local_tensors(unified_multi_meta_state_dicts[self.client_for_push_name], unified_multi_sharding_dicts[self.client_for_push_name])
        if self.storage_plan.train_gen_model_share():
            original_tensor_mapping = client_for_push.get_original_tensor_mapping()
            client_for_pull.register_local_tensors(
                unified_multi_meta_state_dicts[self.client_for_pull_name], 
                unified_multi_sharding_dicts[self.client_for_pull_name], 
                binded_meta_tensor_mapping=original_tensor_mapping
            )
        else:
            # raise NotImplementedError("Gen model not share with train model is not implemented yet.")
            client_for_pull.register_local_tensors(
                unified_multi_meta_state_dicts[self.client_for_pull_name], 
                unified_multi_sharding_dicts[self.client_for_pull_name]
            )
        self._nixl_protocol_phase2()
        
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
        """Initialize the model."""
        local_path = copy_to_local(self.model_config.path, use_shm=self.model_config.get("use_shm", False))
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=self.model_config.get("trust_remote_code", False))
        if type(model_config) in AutoModelForVision2Seq._model_mapping.keys():
            model_class = AutoModelForVision2Seq
        else:
            model_class = AutoModelForCausalLM

        if self.psrl_config.ps_mode == "nixl_cpu":
            # Initialize the model on meta device
            with init_empty_weights():
                self.train_meta_hf_model = model_class.from_config(
                    model_config, 
                    torch_dtype=self.storage_plan.train_model_dtype,
                    trust_remote_code=self.model_config.get("trust_remote_code", False)
                )
                if self.storage_plan.train_gen_model_share():
                    self.gen_meta_hf_model = self.train_meta_hf_model
                else:
                    self.gen_meta_hf_model = model_class.from_config(
                        model_config, 
                        torch_dtype=self.storage_plan.gen_model_dtype,
                        trust_remote_code=self.model_config.get("trust_remote_code", False)
                    )
        elif self.psrl_config.ps_mode == "nixl_gpu":
            raise NotImplementedError("NIXL GPU mode is not implemented yet.")
        else:
            raise ValueError(f"Invalid PS mode: {self.psrl_config.ps_mode}")
    
    def _build_transfer_key_cache(self, src_original_state_dict):
        self._transfer_key_cache = {
            'src_dict_id': id(src_original_state_dict) 
        }
        for key_tuple in src_original_state_dict.keys():
            k, shard_idx = key_tuple
            if k not in self._transfer_key_cache:
                self._transfer_key_cache[k] = []
            self._transfer_key_cache[k].append((k, shard_idx))
    
    def transfer_train_to_gen(self, key: str):
        if self.storage_plan.train_gen_model_share():
            return
        src_client = self.nixl_multi_storage_clients.get_client_by_name(self.client_for_push_name)
        target_client = self.nixl_multi_storage_clients.get_client_by_name(self.client_for_pull_name)
        src_original_state_dict = src_client.get_original_tensor_mapping()
        target_original_state_dict = target_client.get_original_tensor_mapping()
        # src_temp_state_dict = src_client.get_temp_tensor_mapping()
        # target_temp_state_dict = target_client.get_temp_tensor_mapping()
        # assert len(src_temp_state_dict) == 0 and len(target_temp_state_dict) == 0, "Temp state dict should be empty"
        if not hasattr(self, '_transfer_key_cache') or self._transfer_key_cache.get('src_dict_id') != id(src_original_state_dict):
            self._build_transfer_key_cache(src_original_state_dict)
        matching_keys = self._transfer_key_cache.get(key, [])
        for key_shard_idx_tuple in matching_keys:
            target_original_state_dict[key_shard_idx_tuple].copy_(src_original_state_dict[key_shard_idx_tuple])

    def shutdown(self):
        self.nixl_multi_storage_clients.shutdown()
