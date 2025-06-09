import ray
import os
import sys
import logging
import torch
import torch.distributed as dist

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy, StateDictType, FullStateDictConfig
from typing import Any, Callable, ClassVar, Optional, Union, List
from time import sleep
from omegaconf import DictConfig, open_dict
from dataclasses import dataclass
from verl import DataProto
from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import get_torch_device
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.workers.rollout.vllm_rollout import vllm_mode
from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class TrainInterface:
    """Info for the PSRL TrainWorker."""
    ps_handle: ray.actor.ActorHandle


class PSRL_TrainWorker(ActorRolloutRefWorker):
    def __init__(self, config: DictConfig, role: str, psrl_config: DictConfig, train_interface: TrainInterface) -> None:
        super().__init__(config, role)
        self.psrl_config = psrl_config
        self.train_interface = train_interface
        
    def push_model_cpu(self) -> None:
        """Push the model weights to the PS via CPU."""
        ps_handle = self.train_interface.ps_handle
        curr_ps_model_version = ray.get(ps_handle.get_ps_model_version.remote())
        next_ps_model_version = curr_ps_model_version + 1
        # Gather the model state dict on rank 0
        # TODO: support FSDP2
        assert fsdp_version(self.actor_module_fsdp) == 1, "FSDP version 2 is not supported yet."
        with FSDP.state_dict_type(self.actor_module_fsdp, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            full_state_dict = self.actor_module_fsdp.state_dict()
        if self.rank == 0:
            assert len(full_state_dict) > 0, "The model state dict shouldn't be empty on rank 0 worker."
            logger.info(f"<TrainWorker_{self.rank}>: push the model via CPU (async).")
            # Do not need to wait for the push to complete, as it can be overlapped with the next-iteration training
            self.train_interface.ps_handle.push_model_state_dict_cpu.remote(next_ps_model_version, full_state_dict)
        else:
            assert len(full_state_dict) == 0, "The model state dict should be empty on non-rank-0 workers."
    
    # The log_prob in training side is only used when there is a proxy policy    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        # Support all hardwares
        data = data.to(torch.cuda.current_device())
        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
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
            log_gpu_memory_usage("After offload actor model during compute_log_prob", logger=logger)

        return output
                
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        # The model weights are pushed to the PS via CPU
        if self.psrl_config.ps_mode == "cpu":
            logger.info(f"<TrainWorker_{self.rank}>: train actor.")
            super().update_actor(data)
            self.push_model_cpu()
        else:
            raise NotImplementedError(f"PSRL TrainWorker does not support PS mode '{self.psrl_config.ps_mode}' yet.")
            