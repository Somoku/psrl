import ray
import os
import logging
import threading
from omegaconf import DictConfig

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import StateDictType, FullStateDictConfig

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import get_device_id
from verl.utils.debug import log_gpu_memory_usage, GPUMemoryLogger
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.fsdp_utils import (
    fsdp_version,
    load_fsdp_model_to_gpu,
    offload_fsdp_model_to_cpu,
)

from psrl.workers.train import TrainInterface
from psrl.utils.logger import DualOutputHandler, get_worker_info, log_dual_events, log_single_event, EventType
from psrl.utils.nixl import NIXLClientType, NIXLInterface, NIXLStorageClient, GLOBAL_META_SERVER_NAME, GLOBAL_TRAIN_CLIENT_NAME
from psrl.utils.state_dict import convert_fsdp_inplace


psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

def get_fsdp_full_state_dict(model: torch.nn.Module, offload_to_cpu: bool = True, rank0_only: bool = True):
    """
    Get the full state dict from an FSDP model.

    Args:
        model (torch.nn.Module): The FSDP model to get state dict from
        offload_to_cpu (bool, optional): Whether to offload the state dict to CPU. Defaults to True.
        rank0_only (bool, optional): Whether to only get state dict on rank 0. Defaults to True.

    Returns:
        dict: The full state dict of the model

    Raises:
        NotImplementedError: If the FSDP version is unknown
    """
    if fsdp_version(model) == 1:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.api import StateDictType, FullStateDictConfig
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=offload_to_cpu, rank0_only=rank0_only)):
            state_dict = model.state_dict()
        return state_dict
    elif fsdp_version(model) == 2:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
        state_dict_config = StateDictOptions(
            full_state_dict=True, 
            cpu_offload=offload_to_cpu, 
            broadcast_from_rank0=not rank0_only
        )
        state_dict = get_model_state_dict(model, options=state_dict_config)
        return state_dict
    else:
        raise NotImplementedError(f"Unknown FSDP version {fsdp_version}")

class PSRL_FSDPTrainWorker(ActorRolloutRefWorker):
    def __init__(
        self,
        config: DictConfig,
        role: str,
        psrl_config: DictConfig,
        train_interface: TrainInterface,
        nixl_interface: NIXLInterface,
        **kwargs
    ) -> None:
        super().__init__(config, role, **kwargs)
        self.psrl_config = psrl_config
        self.train_interface = train_interface
        self.nixl_interface = nixl_interface
        
        # NIXL
        self.nixl_storage_client = None
        self.unified_state_dict = None
        self.unified_sharding_dict = None
        # NIXL wait threads
        self.nixl_wait_thread = None  # Single thread for all wait operations
        self.nixl_wait_thread_lock = threading.Lock()
        self.nixl_wait_completed = threading.Event()
        
        # Build logger
        self.log_prefix = f"TrainWorker_R{self.rank}"
        psrl_logger.addHandler(DualOutputHandler(self.psrl_config.logging_path, self.log_prefix))
        psrl_logger.info(f"Initialized on {get_worker_info()}.")
     
    def get_node_id(self) -> str:
        """
        Get the node id of the train worker.
        """
        return ray.get_runtime_context().get_node_id()
        
    @property   
    def is_train_representative_rank(self) -> bool:
        """
        Check if the current rank is the representative rank.
        The representative rank is the rank 0 of the PS.
        """
        return self.rank == 0
    
    def init_nixl_client(self):
        """Initialize the NIXL client."""
        assert self.actor_module_fsdp, "The actor module must be initialized before calling init_nixl_client."
        if self.psrl_config.nixl.server_mode == "storage_server":
            raise ValueError("Storage server mode is deprecated.")
        elif self.psrl_config.nixl.server_mode == "meta_server":
            self.nixl_storage_client = NIXLStorageClient(
                client_name=f"{GLOBAL_TRAIN_CLIENT_NAME}_{self.rank}",
                server_name=GLOBAL_META_SERVER_NAME,
                use_gpu=True,
                client_type=NIXLClientType.PUSH_SIDE,
                nixl_config=self.psrl_config.nixl,
                nixl_interface=self.nixl_interface
            )
        else:
            raise ValueError(f"Invalid NIXL server mode: {self.psrl_config.nixl.server_mode}")
        psrl_logger.info(f"NIXL client initialized on port {self.nixl_storage_client.client_port}.")
       
    def nixl_protocol(self):
        # Register the state dict and sharding dict to the NIXL client
        psrl_logger.info(f"nixl client protocol step 0: convert_fsdp_inplace")
        unified_state_dict, local_sharding_dict = convert_fsdp_inplace(self.config.actor.strategy, self.actor_module_fsdp)
        psrl_logger.info(f"nixl client protocol step 1: connect_to_server")
        self.nixl_storage_client.connect_to_server()
        psrl_logger.info(f"nixl client protocol step 2: send_local_sharding")
        self.nixl_storage_client.send_local_sharding(local_sharding_dict)
        psrl_logger.info(f"nixl client protocol step 3: wait_for_server_sharding")
        unified_sharding_dict = self.nixl_storage_client.wait_for_server_sharding()
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
        
    def ray_push_model(self) -> None:
        """
        Push the model weights to the PS via ray. In 'cpu' mode, push the full state dict. In 'cpu_ref' mode, push a ray object_ref.
        In 'cpu' mode, the PS worker will block on large model transfer (potential bottleneck).
        In 'cpu_ref' mode, only the train worker blocks on ray.put, PS worker is non-blocking.
        """
        ps_manager_handle = self.train_interface.ps_manager_handle
        curr_ps_model_version = ray.get(ps_manager_handle.get_ps_model_version.remote())
        next_ps_model_version = curr_ps_model_version + 1
        # Gather the model state dict on rank 0
        # assert fsdp_version(self.actor_module_fsdp) == 1, "FSDP version 2 is not supported yet."
        psrl_logger.info(f"Gathering the full state dict on the CPU of the representive rank.")
        full_state_dict = get_fsdp_full_state_dict(self.actor_module_fsdp, offload_to_cpu=True, rank0_only=True)
        if self.is_train_representative_rank:
            assert len(full_state_dict) > 0, "The model state dict shouldn't be empty on the representive worker."
            psrl_logger.info(f"Push the model via CPU on the representive rank (async).")
            if self.psrl_config.ps_mode == "cpu":
                # In 'cpu' mode, push the full state dict (PS worker will block on transfer)
                # But the training side does not need to wait for the push to complete, as it can be overlapped with the next-iteration training
                ps_manager_handle.push_model_state_dict_cpu.remote(next_ps_model_version, full_state_dict)
            elif self.psrl_config.ps_mode == "cpu_ref":
                # In 'cpu_ref' mode, push a ray object_ref (PS worker is non-blocking)
                # But the training side needs to wait for the push to complete, as `ray.put` is blocking
                object_ref = ray.put(full_state_dict)  # This blocks until the state dict is in the object store
                ps_manager_handle.push_model_state_dict_cpu_ref_list.remote(next_ps_model_version, [object_ref]) # Tricky part: manually wrap the object_ref in a list to avoid ray dereferencing the full state dict
            else:
                raise NotImplementedError(f"PSRL TrainWorker does not support PS mode '{self.psrl_config.ps_mode}' yet.")
        else:
            assert len(full_state_dict) == 0, "The model state dict should be empty on non-representative workers."
    
    def nixl_push_model(self) -> None:
        """
        Push the model weights to the PS via NIXL.
        
        Usage example:
            # Start the push operation (this will start a background wait thread)
            worker.nixl_push_model()
            
            # Do other work while push is happening in background...
            
            # Wait for all push operations to complete
            success = worker.wait_for_nixl_push_completion(timeout=60.0)
            if success:
                print("All NIXL push operations completed successfully")
            else:
                print("Some NIXL push operations timed out")
                
            # Or check thread status
            status = worker.get_nixl_wait_thread_status()
            print(f"Thread alive: {status.get('alive', False)}")
        """
        assert self.psrl_config.ps_mode == "nixl_cpu" or self.psrl_config.ps_mode == "nixl_gpu", "push_model_state_dict_nixl should only be used in 'nixl_cpu' or 'nixl_gpu' mode."
        ps_manager_handle = self.train_interface.ps_manager_handle
        curr_ps_model_version = ray.get(ps_manager_handle.get_ps_model_version.remote())
        next_ps_model_version = curr_ps_model_version + 1
        ps_nixl_storage_client_names = ray.get(ps_manager_handle.get_ps_nixl_storage_client_names.remote())
        psrl_logger.info(f"Pushing the model to the PS via NIXL on {len(ps_nixl_storage_client_names)} clients.")
        
        # Clear previous wait thread
        with self.nixl_wait_thread_lock:
            if self.nixl_wait_thread is not None and self.nixl_wait_thread.is_alive():
                raise RuntimeError("Previous NIXL wait thread is still running, you should wait for it to complete before calling nixl_push_model again.")
            self.nixl_wait_thread = None
            self.nixl_wait_completed.clear()
        
        # Collect all operations to wait for
        wait_operations = []
        for target_client_name in ps_nixl_storage_client_names: 
            for key in self.unified_state_dict:
                self.nixl_storage_client.client_write(target_client_name, key, b"train_push")
                wait_operations.append((key, target_client_name))
        
        # Start a single background thread to wait for all operations
        def wait_all_operations():
            try:
                psrl_logger.debug(f"Starting to wait for {len(wait_operations)} NIXL operations...")
                for key, target_client_name in wait_operations:
                    self.nixl_storage_client.wait(key, b"train_push", "WRITE", target_client=target_client_name)
                    psrl_logger.debug(f"Wait completed for key {key} to target {target_client_name}")
                psrl_logger.debug("All NIXL wait operations completed successfully.")
                ray.get(ps_manager_handle.push_model_state_dict_nixl.remote(next_ps_model_version))
                self.nixl_wait_completed.set()
            except Exception as e:
                psrl_logger.error(f"Error in NIXL wait thread: {e}")
                # Don't set the event on error, so wait_for_nixl_push_completion can detect failure
        
        wait_thread = threading.Thread(target=wait_all_operations, daemon=True)
        wait_thread.start()
        # Store the thread reference
        with self.nixl_wait_thread_lock:
            self.nixl_wait_thread = wait_thread
    
    def wait_for_nixl_push_completion(self, timeout: float = None) -> bool:
        """
        Wait for the NIXL push wait thread to complete.
        
        Args:
            timeout (float, optional): Maximum time to wait in seconds. If None, wait indefinitely.
            
        Returns:
            bool: True if the thread completed successfully, False if timeout occurred or thread failed.
        """
        with self.nixl_wait_thread_lock:
            if self.nixl_wait_thread is None:
                psrl_logger.debug("No NIXL wait thread to wait for.")
                return True
            
            psrl_logger.info("Waiting for NIXL wait thread to complete...")
            if timeout is not None:
                # Use the event to wait with timeout
                if self.nixl_wait_completed.wait(timeout=timeout):
                    # Event was set, check if thread actually completed successfully
                    self.nixl_wait_thread.join(timeout=1.0)  # Brief join to catch any exceptions
                    if self.nixl_wait_thread.is_alive():
                        psrl_logger.warning("NIXL wait thread is still alive after event was set.")
                        return False
                    psrl_logger.info("NIXL wait thread completed successfully.")
                    return True
                else:
                    psrl_logger.warning("Timeout waiting for NIXL wait thread to complete.")
                    return False
            else:
                # Wait indefinitely
                self.nixl_wait_thread.join()
                if self.nixl_wait_thread.is_alive():
                    psrl_logger.warning("NIXL wait thread is still alive after join.")
                    return False
                psrl_logger.info("NIXL wait thread completed successfully.")
                return True
    
    def get_nixl_wait_thread_status(self) -> dict:
        """
        Get the status of the NIXL wait thread.
        
        Returns:
            dict: Dictionary containing thread status information.
        """
        with self.nixl_wait_thread_lock:
            if self.nixl_wait_thread is None:
                return {
                    'has_thread': False,
                    'alive': False,
                    'completed': True
                }
            return {
                'has_thread': True,
                'alive': self.nixl_wait_thread.is_alive(),
                'completed': self.nixl_wait_completed.is_set()
            }
            
    def push_model(self):
        if self.psrl_config.ps_mode == "cpu" or self.psrl_config.ps_mode == "cpu_ref":
            self.ray_push_model()
        elif self.psrl_config.ps_mode == "nixl_cpu" or self.psrl_config.ps_mode == "nixl_gpu":
            self.nixl_push_model()
            # TODO(lhy): wait for the push to complete before the next iteration optimizer update
            # This will enable the NIXL push to be overlapped with the next iteration training
            self.wait_for_nixl_push_completion()
        else:
            raise NotImplementedError(f"PSRL TrainWorker does not support PS mode '{self.psrl_config.ps_mode}' yet.")
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def init_model(self):
        with log_dual_events("Initialize model", psrl_logger, event_type=EventType.INIT):
            super().init_model()
    
    # The log_prob in training side is only used when there is a proxy policy    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @GPUMemoryLogger(role="compute_log_prob", logger=psrl_logger)
    def compute_log_prob(self, data: DataProto):
        # NOTE: compared with verl, we replace `old_log_probs` with `proxy_log_probs` in the output.
        # when is_lora is True, we use the actor without lora applied to calculate the log_prob
        # which is mostly used for ref log_prob calculation
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        # Support all hardwares
        from contextlib import nullcontext

        is_lora = data.meta_info.pop("is_lora", False)
        adapter_ctx = self.actor.actor_module.disable_adapter() if is_lora else nullcontext()
        data = data.to(get_device_id())
        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            with adapter_ctx:
                output, entropys = self.actor.compute_log_prob(data=data, calculate_entropy=True)
            output = DataProto.from_dict(
                tensors={"proxy_log_probs": output, "entropys": entropys},
                meta_info={"temperature": self.config.rollout.temperature},
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during compute_log_prob", logger=psrl_logger)

        return output
                
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        # The model weights are pushed to the PS via CPU
        with log_dual_events("Train actor", psrl_logger, event_type=EventType.TRAIN):
            output = super().update_actor(data)
        with log_dual_events("Push model", psrl_logger, event_type=EventType.PUSH):
            self.push_model()
        return output
            