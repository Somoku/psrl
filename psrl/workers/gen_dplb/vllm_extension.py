import logging
import os
import time

import torch
from omegaconf import DictConfig
from torch.distributed.tensor import DTensor
from verl.utils.device import get_device_id
from verl.utils.fs import copy_to_local
from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.v1.core.kv_cache_utils import estimate_max_model_len

from psrl.utils.converter import create_parameter_mapping
from psrl.utils.converter.vllm_converter import convert_vllm_inplace
from vllm.model_executor.models.interfaces import SupportsWeightLayoutSpec
from psrl.utils.nixl import (
    GLOBAL_GEN_CLIENT_NAME,
    GLOBAL_META_SERVER_NAME,
    NIXLClientType,
    NIXLStorageClient,
)

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class vLLMWorkerExtension(vLLMColocateWorkerExtension):
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
        return loaded_params

    # ----------------------------- NIXL Related -----------------------------
    # Because the model is on another process since vllm V1, we must call the nixl methods via rpc
    def get_instance_local_rank(self):
        from vllm.distributed.parallel_state import get_world_group

        return get_world_group().rank

    def get_instance_local_tp_rank(self):
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        return get_tensor_model_parallel_rank()

    def init_nixl_client(
        self,
        nixl_config: DictConfig,
        replica_idx: int,
        logging_path: str | None = None,
    ):
        # NIXL attributes
        self.unified_state_dict = None
        self.unified_sharding_dict = None
        # Initialize the NIXL client
        self.nixl_storage_client = NIXLStorageClient(
            client_name=f"{GLOBAL_GEN_CLIENT_NAME}_I{replica_idx}_R{self.get_instance_local_rank()}",
            server_name=GLOBAL_META_SERVER_NAME,
            use_gpu=True,
            client_type=NIXLClientType.PULL_SIDE,
            nixl_config=nixl_config,
            replica_idx=replica_idx,
            worker_index=self.get_instance_local_rank(),
            # client_group_id=instance_id,
            logging_path=logging_path,
        )
        psrl_logger.info(f"NIXL client initialized on port {self.nixl_storage_client.client_port}.")

    def nixl_convert_params(self, model_config: DictConfig):
        """Convert the model parameters to unified format.

        Args:
            config (DictConfig): Configuration object containing training settings.
        """
        vllm_model = self.model_runner.model
        if isinstance(vllm_model, CUDAGraphWrapper):
            vllm_model = vllm_model.unwrap()
        param_mapping = (
            None
            if isinstance(vllm_model, SupportsWeightLayoutSpec)
            else create_parameter_mapping(type(vllm_model), copy_to_local(model_config["path"]))
        )
        self.unified_state_dict, self.local_sharding_dict = convert_vllm_inplace(
            vllm_model,
            tp_rank=self.get_instance_local_tp_rank(),
            parameter_mapping=param_mapping,
        )

    def nixl_protocol(self, model_config: DictConfig, mode: str = "full"):
        """Run the NIXL server protocol.

        Args:
            model_config (DictConfig): Configuration object containing training settings.
            mode (str): Mode of registration, either 'meta' or 'full'.
                'meta' mode converts to meta tensors and skip registering their memory.
                'full' mode converts to full tensors.

            NOTE: ps storage may init with meta tensors, the register step would be different.
        """
        # Register the state dict and sharding dict to the NIXL client
        meta_only = mode == "meta"
        if self.unified_state_dict is None or self.local_sharding_dict is None:
            self.nixl_convert_params(model_config)
        psrl_logger.info("nixl client protocol step 1: connect_to_server")
        self.nixl_storage_client.connect_to_server()
        psrl_logger.info("nixl client protocol step 2: send_local_sharding")
        self.nixl_storage_client.send_local_sharding(self.local_sharding_dict)
        psrl_logger.info("nixl client protocol step 3: wait_for_server_sharding")
        unified_sharding_dict = self.nixl_storage_client.wait_for_server_sharding()
        # psrl_logger.info(f"unified_sharding_dict: {unified_sharding_dict}")
        psrl_logger.info("nixl client protocol step 4: register_local_tensors")
        self.nixl_storage_client.register_local_tensors(
            self.unified_state_dict, unified_sharding_dict, meta_only=meta_only
        )
        psrl_logger.info("nixl client protocol step 5: send_local_info")
        self.nixl_storage_client.send_local_info()
        psrl_logger.info("nixl client protocol step 6: wait_for_server_info")
        self.nixl_storage_client.wait_for_server_info()
        psrl_logger.info("nixl client protocol step 7: send_local_temp_mapping")
        self.nixl_storage_client.send_local_temp_mapping()
        psrl_logger.info("nixl client protocol step 8: wait_for_server_temp_mappings")
        self.nixl_storage_client.wait_for_server_temp_mappings()
        psrl_logger.info("nixl client protocol done.")
        self.unified_sharding_dict = unified_sharding_dict

    def nixl_register_after_wake_up(self):
        """Register the model parameters to NIXL after wake up from sleep.

        After sleep/wake_up, the physical memory backing the model weights
        has changed while virtual addresses remain the same. This method performs
        local re-registration:
        1. Reset nixl agent (clears UCX rcache)
        2. Re-registers memory with new physical pages (generates new rkeys)
        """
        torch.cuda.synchronize()
        # Reset nixl agent and reregister to handle physical memory changes
        self.nixl_storage_client.register_local_tensors(self.unified_state_dict, self.unified_sharding_dict)

    def nixl_deregister(self):
        """Deregister the model parameters from NIXL."""
        self.nixl_storage_client.deregister_local_tensors()

    def nixl_send_local_info_to(self, dst_agent_names: str | list[str]):
        """
        Send local NIXL info to the specified destination agents.
        """
        if isinstance(dst_agent_names, str):
            dst_agent_names = [dst_agent_names]
        self.nixl_storage_client.send_local_info_to(dst_agent_names)

    def nixl_wait_for_update_infos(self, info_num: int):
        """Wait for infos of updated clients for global synchronization.

        Args:
            info_num (int): Number of infos to wait for.
        """
        self.nixl_storage_client.wait_for_update_infos(info_num)

    def nixl_pull_model_core(self, ps_nixl_agent_names, ps_nixl_gen_storage_client_names):
        """Pull the model parameters from PS workers via NIXL.

        Args:
            ps_nixl_agent_names (list[str]): List of PS NIXL agent names
            ps_nixl_train_storage_client_names (list[str]): List of PS NIXL train storage client names
        """
        if not hasattr(self, "pull_times"):
            self.pull_times = 0
        self.pull_times += 1
        wait_operations = []
        time_start = time.time()
        for key in self.unified_state_dict:
            for target_agent_name, target_client_name in zip(ps_nixl_agent_names, ps_nixl_gen_storage_client_names):
                shards_to_transfer = self.nixl_storage_client.client_read(
                    target_agent_name,
                    target_client_name,
                    key,
                    f"gen_pull_{self.pull_times}",
                )
                # shards_to_transfer = self.nixl_storage_client.client_read(
                #     target_agent_name, target_client_name, key, "gen_pull", merge_and_cache_xfer=False
                # )
                if len(shards_to_transfer) > 0:
                    wait_operations.append((key, target_client_name, shards_to_transfer))
        # Generation cannot be overlapped with the NIXL pull, so we need to wait for all operations to complete
        for key, target_client_name, shards_to_transfer in wait_operations:
            self.nixl_storage_client.wait(
                key,
                f"gen_pull_{self.pull_times}",
                "READ",
                target_client=target_client_name,
            )
            # self.nixl_storage_client.wait(key, "gen_pull", "READ", target_client=target_client_name)
        self.nixl_storage_client.merge_and_finish_cached_xfer()
        self.cuda_synchronize()
        self.nixl_storage_client.clear_intermediate_cached_data()
        time_end = time.time()
        psrl_logger.info(
            f"{self.nixl_storage_client}: NIXL pull model core done ({self.pull_times} times). "
            f"time: {time_end - time_start}s"
        )

    def estimate_max_model_len(self):
        """Estimate the maximum model length that can fit in the available KV cache memory."""
        assert hasattr(self, "available_kv_cache_memory_bytes"), "available_kv_cache_memory_bytes must be set"
        assert hasattr(self, "vllm_config"), "vllm_config must be set"
        kv_cache_spec = self.get_kv_cache_spec()
        assert kv_cache_spec is not None, "kv_cache_spec must not be None"
        # It use the binary search to estimate the max model length
        actual_max_model_len = self.vllm_config.model_config.max_model_len
        # Set the max model length to the upper limit of the estimation
        self.vllm_config.model_config.max_model_len = self.vllm_config.additional_config.get(
            "max_model_len_used_in_estimation",
            self.vllm_config.model_config.max_model_len * 8192,
        )
        estimated_max_model_len = estimate_max_model_len(
            self.vllm_config, kv_cache_spec, self.available_kv_cache_memory_bytes
        )
        # Restore the actual max model length
        self.vllm_config.model_config.max_model_len = actual_max_model_len
        return estimated_max_model_len
