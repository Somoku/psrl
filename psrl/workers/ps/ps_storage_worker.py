import ray
import torch
import os
import logging
from typing import Optional
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq
from omegaconf import DictConfig
from accelerate import init_empty_weights

from verl.utils.fs import copy_to_local

from psrl.utils.nixl import NIXLClientType, NIXLInterface, NIXLStorageClient, global_meta_server_name
from psrl.utils.state_dict.hf_converter import convert_hf_inplace
from psrl.utils.logger import DualOutputHandler, get_worker_info, log_dual_events, log_single_event, EventType


psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))


# TODO: Implement the PSStoragePlan
class PSStoragePlan:
    pass


class PSStorageWorker:
    """A worker that only stores the data and uses NIXL to communicate."""
    
    def __init__(self, config: DictConfig, psrl_config: DictConfig, nixl_interface: NIXLInterface) -> None:
        self.config = config
        self.psrl_config = psrl_config
        self.nixl_interface = nixl_interface
        self.meta_hf_model: Optional[torch.nn.Module] = None
        
        # Build logger
        self.rank = int(os.environ.get("RANK"))
        self.log_prefix = f"PSStorageWorker_R{self.rank}"
        psrl_logger.addHandler(DualOutputHandler(self.log_prefix))
        psrl_logger.info(f"Initialized on {get_worker_info()}.")
        
    def init_nixl_client(self):
        """Initialize the NIXL client."""
        assert self.meta_hf_model, "The HuggingFace model must be initialized before calling init_nixl_client."
        if self.psrl_config.nixl_server_mode == "storage_server":
            raise ValueError("Storage server mode is deprecated.")
        elif self.psrl_config.nixl_server_mode == "meta_server":
            use_gpu = self.psrl_config.ps_mode == "gpu"
            self.nixl_storage_client = NIXLStorageClient(
                client_name=f"NIXLPSClient_{self.rank}",
                server_name=global_meta_server_name,
                server_ip=self.psrl_config.nixl_server_ip,
                server_port=self.psrl_config.nixl_server_port,
                use_gpu=use_gpu,
                mode=self.psrl_config.nixl_server_mode,
                client_type=NIXLClientType.PS,
                nixl_interface=self.nixl_interface
            )
        else:
            raise ValueError(f"Invalid NIXL server mode: {self.psrl_config.nixl_server_mode}")
        psrl_logger.info(f"NIXL client initialized on port {self.nixl_storage_client.client_port}.")
        
    def nixl_protocol(self):
        # Register the state dict and sharding dict to the NIXL client
        psrl_logger.info(f"nixl client protocol step 0: convert_hf_inplace")
        unified_meta_state_dict, local_sharding_dict = convert_hf_inplace(self.meta_hf_model)
        psrl_logger.info(f"nixl client protocol step 1: connect_to_server")
        self.nixl_storage_client.connect_to_server()
        psrl_logger.info(f"nixl client protocol step 2: send_local_sharding")
        self.nixl_storage_client.send_local_sharding(local_sharding_dict)
        psrl_logger.info(f"nixl client protocol step 3: wait_for_server_sharding")
        unified_sharding_dict = self.nixl_storage_client.wait_for_server_sharding()
        psrl_logger.info(f"nixl client protocol step 4: register_local_tensors")
        self.nixl_storage_client.register_local_tensors(unified_meta_state_dict, unified_sharding_dict)
        psrl_logger.info(f"nixl client protocol step 5: send_local_info")
        self.nixl_storage_client.send_local_info()
        psrl_logger.info(f"nixl client protocol step 6: wait_for_server_info")
        self.nixl_storage_client.wait_for_server_info()
        psrl_logger.info(f"nixl client protocol step 7: send_local_temp_mapping")
        self.nixl_storage_client.send_local_temp_mapping()
        psrl_logger.info(f"nixl client protocol step 8: wait_for_server_temp_mappings")
        self.nixl_storage_client.wait_for_server_temp_mappings()
        psrl_logger.info(f"nixl client protocol done.")

    def init_model(self):
        """Initialize the model."""
        local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.get("use_shm", False))
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=self.config.model.get("trust_remote_code", False))
        if type(model_config) in AutoModelForVision2Seq._model_mapping.keys():
            model_class = AutoModelForVision2Seq
        else:
            model_class = AutoModelForCausalLM

        if self.psrl_config.ps_mode == "nixl_cpu":
            # Initialize the model on meta device
            with init_empty_weights():
                self.meta_hf_model = model_class.from_config(
                    model_config, 
                    torch_dtype=torch.bfloat16, # TODO: read from config
                    trust_remote_code=self.config.model.get("trust_remote_code", False)
                )
        elif self.psrl_config.ps_mode == "nixl_gpu":
            raise NotImplementedError("NIXL GPU mode is not implemented yet.")
        else:
            raise ValueError(f"Invalid PS mode: {self.psrl_config.ps_mode}")
    
    def shutdown(self):
        self.nixl_storage_client.shutdown()
