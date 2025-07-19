import os
import logging
from typing import List

import vllm
import torch
try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from verl.utils.device import get_device_id
from verl.utils.vllm_utils import patch_vllm_moe_model_weight_loader

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))

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
            loaded_params = self.model_runner.model.load_weights(weights=rebuild_weights)
            if blocking:
                # Ensure all operations are completed before returning
                torch.cuda.synchronize()
        except Exception as e:
            psrl_logger.error(f"Error in vLLMWorkerExtension.load_weights: {e}")
            return None
        return loaded_params

    def cuda_synchronize(self):
        """Synchronize the CUDA device."""
        try:
            torch.cuda.synchronize()
        except Exception as e:
            psrl_logger.error(f"Error in vLLMWorkerExtension.cuda_synchronize: {e}")
            return None
        return None

    def patch_vllm_moe_model_weight_loader(self) -> None:
        """Patch the vLLM model weight loader for MoE models."""
        try:
            patch_vllm_moe_model_weight_loader(self.model_runner.model)
        except Exception as e:
            psrl_logger.error(f"Error in vLLMWorkerExtension.patch_vllm_moe_model_weight_loader: {e}")
            return None
        return None
