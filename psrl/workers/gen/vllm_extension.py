import os
import logging
from typing import List, Union
from omegaconf import DictConfig

import vllm
import torch
try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from verl.utils.fs import copy_to_local
from verl.utils.device import get_device_id
from verl.utils.vllm_utils import patch_vllm_moe_model_weight_loader

from psrl.utils.nixl import NIXLInterface, NIXLStorageClient, GLOBAL_META_SERVER_NAME, GLOBAL_GEN_CLIENT_NAME, NIXLClientType
from psrl.utils.converter import create_parameter_mapping
from psrl.utils.converter.vllm_converter import convert_vllm_inplace

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

class vLLMWorkerExtension:
    def load_weights(self, weights, blocking: bool = True):
        """
        Load weights into the vLLM model runner.
        
        This method rebuilds the weights using the provided function and arguments stemming from `reduce_tensor` calls,
        transfers them to the current CUDA device, and loads them into the vLLM model runner.
        If the weight is a DTensor, it converts it to a full tensor before loading.
        If `blocking` is True, it ensures that all operations are completed before returning.
        If an error occurs during the process, it logs the error and returns None.

        Args:
            weights (List[tuple]): A list of tuples where each tuple contains:
                - name (str): The name of the weight.
                - handle (tuple): A tuple containing the function and its arguments to rebuild the weight.
            blocking (bool): If True, will block until all operations are completed.

        Returns:
            loaded_params: The loaded parameters from the model runner.

        Raises:
            Exception: If there is an error during the loading process.
        """
        try:
            def rebuild_weights_generator():
                current_device = torch.cuda.current_device()
                for name, handle in weights:
                    func, args = handle
                    list_args = list(args)
                    # CPU bundle: (type(tensor), storage, metadata)
                    if len(list_args) == 3:
                        tensor = func(*list_args)
                        tensor = tensor.to(current_device, non_blocking=True)
                        if isinstance(tensor, DTensor):
                            tensor = tensor.full_tensor()
                    else:
                        list_args[6] = get_device_id()
                        tensor = func(*list_args)
                        if isinstance(tensor, DTensor):
                            tensor = tensor.full_tensor()
                    yield (name, tensor)
            
            rebuild_weights = rebuild_weights_generator()
            torch.cuda.synchronize()
            loaded_params = self.model_runner.model.load_weights(weights=rebuild_weights)
            if blocking:
                # Ensure all operations are completed before returning
                torch.cuda.synchronize()
        except Exception as e:
            raise ValueError(f"Error in vLLMWorkerExtension.load_weights: {e}")
        return loaded_params

    def cuda_synchronize(self):
        """Synchronize the CUDA device."""
        try:
            torch.cuda.synchronize()
        except Exception as e:
            raise ValueError(f"Error in vLLMWorkerExtension.cuda_synchronize: {e}")
        return None

    def patch_vllm_moe_model_weight_loader(self) -> None:
        """Patch the vLLM model weight loader for MoE models."""
        try:
            patch_vllm_moe_model_weight_loader(self.model_runner.model)
        except Exception as e:
            raise ValueError(f"Error in vLLMWorkerExtension.patch_vllm_moe_model_weight_loader: {e}")
        return None
    
    # ----------------------------- NIXL Related -----------------------------
    # Because the model is on another process since vllm V1, we must call the nixl methods via rpc
    def get_instance_local_rank(self):
        from vllm.distributed.parallel_state import get_world_group
        return get_world_group().rank
        
    def get_instance_local_tp_rank(self):
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
        return get_tensor_model_parallel_rank()
    
    def init_nixl_client(self, nixl_config: DictConfig, nixl_interface_after_rpc: Union[dict, NIXLInterface], instance_id: int):
        # Reconstruct the nixl_interface (the RPC call serializes the nixl_interface to a dict)
        if isinstance(nixl_interface_after_rpc, dict):
            nixl_interface = NIXLInterface(
                port_scanner=nixl_interface_after_rpc['port_scanner']
            )
        else:
            nixl_interface = nixl_interface_after_rpc
        # NIXL attributes
        self.unified_state_dict = None
        self.unified_sharding_dict = None
        # Initialize the NIXL client
        self.nixl_storage_client = NIXLStorageClient(
            client_name=f"{GLOBAL_GEN_CLIENT_NAME}_I{instance_id}_R{self.get_instance_local_rank()}",
            server_name=GLOBAL_META_SERVER_NAME,
            use_gpu=True,
            client_type=NIXLClientType.PULL_SIDE,
            nixl_config=nixl_config,  
            nixl_interface=nixl_interface
        )
        psrl_logger.info(f"NIXL client initialized on port {self.nixl_storage_client.client_port}.")
    
    def nixl_protocol(self, config: DictConfig):
        # Register the state dict and sharding dict to the NIXL client
        psrl_logger.info(f"nixl client protocol step 0: convert_vllm_inplace")
        vllm_model = self.model_runner.model
        param_mapping = create_parameter_mapping(type(vllm_model), copy_to_local(config.model.path))
        unified_state_dict, local_sharding_dict = convert_vllm_inplace(param_mapping, vllm_model, tp_rank=self.get_instance_local_tp_rank())
        psrl_logger.info(f"nixl client protocol step 1: connect_to_server")
        self.nixl_storage_client.connect_to_server()
        psrl_logger.info(f"nixl client protocol step 2: send_local_sharding")
        self.nixl_storage_client.send_local_sharding(local_sharding_dict)
        psrl_logger.info(f"nixl client protocol step 3: wait_for_server_sharding")
        unified_sharding_dict = self.nixl_storage_client.wait_for_server_sharding()
        # psrl_logger.info(f"unified_sharding_dict: {unified_sharding_dict}")
        psrl_logger.info(f"nixl client protocol step 4: register_local_tensors")
        self.nixl_storage_client.register_local_tensors(unified_state_dict, unified_sharding_dict)
        psrl_logger.info(f"nixl client protocol step 5: send_local_info")
        self.nixl_storage_client.send_local_info()
        psrl_logger.info(f"nixl client protocol step 6: wait_for_server_info")
        self.nixl_storage_client.wait_for_server_info()
        psrl_logger.info(f"nixl client protocol step 7: send_local_temp_mapping")
        self.nixl_storage_client.send_local_temp_mapping()
        psrl_logger.info(f"nixl client protocol step 8: wait_for_server_temp_mappings")
        self.nixl_storage_client.wait_for_server_temp_mappings()
        psrl_logger.info(f"nixl client protocol done.")
        self.unified_state_dict = unified_state_dict
        self.unified_sharding_dict = unified_sharding_dict

    def nixl_pull_model_core(self, ps_nixl_agent_names, ps_nixl_gen_storage_client_names):
        if not hasattr(self, "pull_times"):
            self.pull_times = 0
        self.pull_times += 1
        wait_operations = []
        for target_agent_name, target_client_name in zip(ps_nixl_agent_names, ps_nixl_gen_storage_client_names): 
            for key in self.unified_state_dict:
                self.nixl_storage_client.client_read(target_agent_name, target_client_name, key, f"gen_pull_{self.pull_times}")
                wait_operations.append((key, target_client_name))
        # Generation cannot be overlapped with the NIXL pull, so we need to wait for all operations to complete
        for key, target_client_name in wait_operations:
            self.nixl_storage_client.wait(key, f"gen_pull_{self.pull_times}", "READ", target_client=target_client_name)