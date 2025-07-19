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

from psrl.workers.train import TrainInterface
from psrl.utils.logger import DualOutputHandler, get_worker_info, log_dual_events, EventType


psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))

class PSRL_MegatronTrainWorker(ActorRolloutRefWorker):
    def __init__(self, config: DictConfig, role: str, psrl_config: DictConfig, train_interface: TrainInterface) -> None:
        super().__init__(config, role)
        self.psrl_config = psrl_config
        self.train_interface = train_interface
        self.layer_name_mapping = {
            "qkv_layer_name": "self_attention.linear_qkv.",
            "gate_proj_layer_name": "linear_fc1.",
        }
        self.weight_converter = None
        
        # Build logger
        self.log_prefix = f"TrainWorker_R{self.rank}"
        psrl_logger.addHandler(DualOutputHandler(self.log_prefix))
        psrl_logger.info(f"Initialized on {get_worker_info()}.")
        
    @property   
    def is_train_representative_rank(self) -> bool:
        """
        Check if the current rank is the representative rank.
        The representative rank is the rank 0 of the PS.
        """
        return self.rank == 0
        
    def push_model_cpu(self) -> None:
        """
        Push the model weights to the PS. In 'cpu' mode, push the full state dict. In 'cpu_ref' mode, push a ray object_ref.
        In 'cpu' mode, the PS worker will block on large model transfer (potential bottleneck).
        In 'cpu_ref' mode, only the train worker blocks on ray.put, PS worker is non-blocking.
        """
        ps_handle = self.train_interface.ps_handle
        curr_ps_model_version = ray.get(ps_handle.get_ps_model_version.remote())
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
                self.train_interface.ps_handle.push_model_state_dict_cpu.remote(next_ps_model_version, full_state_dict)
            elif self.psrl_config.ps_mode == "cpu_ref":
                # In 'cpu_ref' mode, push a ray object_ref (PS worker is non-blocking)
                # But the training side needs to wait for the push to complete, as `ray.put` is blocking
                object_ref = ray.put(full_state_dict)  # This blocks until the state dict is in the object store
                self.train_interface.ps_handle.push_model_state_dict_cpu_ref_list.remote(next_ps_model_version, [object_ref]) # Tricky part: manually wrap the object_ref in a list to avoid ray dereferencing the full state dict
            else:
                raise NotImplementedError(f"PSRL TrainWorker does not support PS mode '{self.psrl_config.ps_mode}' yet.")
        else:
            assert len(full_state_dict) == 0, "The model state dict should be empty on non-representative workers."
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        with log_dual_events("Initialize model", psrl_logger, event_type=EventType.INIT):
            super().init_model()
    
    # The log_prob in training side is only used when there is a proxy policy    
    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    @GPUMemoryLogger(role="compute_log_prob", logger=psrl_logger)
    def compute_log_prob(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module, load_grad=False)
            log_gpu_memory_usage("After load actor params and grad during compute_log_prob", logger=psrl_logger)

        # Support all hardwares
        data = data.to(get_device_id())
        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        # perform recompute log_prob
        output, entropys = self.actor.compute_log_prob(data=data, calculate_entropy=True)
        output = DataProto.from_dict(
            tensors={"old_log_probs": output, "entropys": entropys},
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
        # The model weights are pushed to the PS via CPU
        if self.psrl_config.ps_mode == "cpu" or self.psrl_config.ps_mode == "cpu_ref":
            with log_dual_events("Train actor", psrl_logger, event_type=EventType.TRAIN):
                output = super().update_actor(data)
            with log_dual_events("Push model", psrl_logger, event_type=EventType.PUSH):
                self.push_model_cpu()
            return output
        else:
            raise NotImplementedError(f"PSRL TrainWorker does not support PS mode '{self.psrl_config.ps_mode}' yet.")
            