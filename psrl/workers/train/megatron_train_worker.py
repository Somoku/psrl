import ray
import os
import logging

from omegaconf import DictConfig

from verl import DataProto
from verl.models.mcore import get_mcore_weight_converter
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.debug import log_gpu_memory_usage, GPUMemoryLogger
from verl.utils.device import get_device_id, get_torch_device
from verl.utils.megatron_utils import (
    load_megatron_model_to_gpu,
    offload_megatron_model_to_cpu,
    per_tensor_generator,
)
from verl.workers.megatron_workers import ActorRolloutRefWorker
from verl.utils.fs import copy_to_local

from psrl.workers.train import TrainInterface, PSRL_BaseTrainWorker
from psrl.utils.logger import DualOutputHandler, get_worker_info, log_dual_events, EventType
from psrl.utils.nixl import NIXLClientType, NIXLInterface, NIXLStorageClient, GLOBAL_META_SERVER_NAME, GLOBAL_TRAIN_CLIENT_NAME
from psrl.utils.converter import create_parameter_mapping
from psrl.utils.converter.megatron_converter import convert_megatron_inplace


psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class PSRL_MegatronTrainWorker(ActorRolloutRefWorker, PSRL_BaseTrainWorker):
    def __init__(
        self, 
        config: DictConfig, 
        role: str, 
        psrl_config: DictConfig, 
        train_interface: TrainInterface,
        nixl_interface: NIXLInterface
    ) -> None:
        ActorRolloutRefWorker.__init__(self, config, role)
        PSRL_BaseTrainWorker.__init__(self, psrl_config, train_interface, nixl_interface)
        
        self.layer_name_mapping = {
            "qkv_layer_name": "self_attention.linear_qkv.",
            "gate_proj_layer_name": "linear_fc1.",
        }
        self.weight_converter = None
        
        # Build logger
        self.log_prefix = f"TrainWorker_R{self.rank}"
        psrl_logger.addHandler(DualOutputHandler(self.psrl_config.logging_path, self.log_prefix))
        psrl_logger.info(f"Initialized on {get_worker_info()}.")
        
    @property   
    def is_train_representative_rank(self) -> bool:
        """
        Check if the current rank is the representative rank.
        The representative rank is the rank 0 of the PS.
        """
        return self.rank == 0
       
    def init_nixl_client(self):
        """Initialize the NIXL client."""
        assert self.actor_module, "The actor module must be initialized before calling init_nixl_client."
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
        psrl_logger.info(f"nixl client protocol step 0: convert_megatron_inplace")
        parameter_mapping = create_parameter_mapping("Megatron", copy_to_local(self.config.model.path))
        unified_state_dict, local_sharding_dict = convert_megatron_inplace(parameter_mapping, self.actor_module)
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
        Push the model weights to the PS. In 'cpu' mode, push the full state dict. In 'cpu_ref' mode, push a ray object_ref.
        In 'cpu' mode, the PS worker will block on large model transfer (potential bottleneck).
        In 'cpu_ref' mode, only the train worker blocks on ray.put, PS worker is non-blocking.
        """
        ps_manager_handle = self.train_interface.ps_manager_handle
        curr_ps_model_version = ray.get(ps_manager_handle.get_ps_model_version.remote())
        next_ps_model_version = curr_ps_model_version + 1
        # Gather the model state dict on rank 0
        psrl_logger.info(f"Gathering the full state dict on the CPU of the representative rank.")
        if self.weight_converter is None:
            self.weight_converter = get_mcore_weight_converter(self.actor_model_config, self.dtype)
        per_tensor_param = per_tensor_generator(
            self.actor_module,
            self.actor_model_config,
            self.weight_converter,
            self.tf_config,
            self.layer_name_mapping,
        )
        full_state_dict = {}
        for name, param in per_tensor_param:
            if self.is_train_representative_rank:
                full_state_dict[name] = param.to("cpu", non_blocking=True)
        if self.is_train_representative_rank:
            assert len(full_state_dict) > 0, "The model state dict shouldn't be empty on the representative worker."
            psrl_logger.info(f"Push the model via CPU on the representative rank (async).")
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
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        with log_dual_events("Initialize model", psrl_logger, event_type=EventType.INIT):
            ActorRolloutRefWorker.init_model(self)
    
    # The log_prob in training side is only used when there is a proxy policy    
    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    @GPUMemoryLogger(role="compute_log_prob", logger=psrl_logger)
    def compute_log_prob(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module, load_grad=False)
            log_gpu_memory_usage("After load actor params and grad during compute_log_prob", logger=psrl_logger)
        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        data = data.to(get_device_id())
        output, entropys = self.actor.compute_log_prob(data=data, calculate_entropy=True)
        output = DataProto.from_dict(
            tensors={"proxy_log_probs": output, "entropys": entropys},
            meta_info={"temperature": self.config.rollout.temperature},
        )
        output = output.to("cpu")
        # clear kv cache
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
            log_gpu_memory_usage("After offload actor params and grad during compute_log_prob", logger=psrl_logger)
        get_torch_device().empty_cache()
        return output
                
    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        with log_dual_events("Train actor", psrl_logger, event_type=EventType.TRAIN):
            output = ActorRolloutRefWorker.update_actor(self, data)
        with log_dual_events("Push model", psrl_logger, event_type=EventType.PUSH):
            PSRL_BaseTrainWorker.push_model(self)
        return output
        