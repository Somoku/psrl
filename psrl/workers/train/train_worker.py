import ray
import os
import sys
import logging
import torch
import torch.distributed as dist

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy, StateDictType, FullStateDictConfig
from typing import Any, Callable, ClassVar, Optional, Union, List, Dict
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

from psrl.utils.logger import DualOutputHandler, get_worker_info, log_dual_events, log_single_event, EventType
from psrl.utils.nixl import NIXLClientType, NIXLInterface, NIXLStorageClient, global_meta_server_name
from psrl.utils.state_dict import convert_fsdp_inplace

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))


@dataclass
class TrainInterface:
    """Info for the PSRL TrainWorker."""
    ps_manager_handle: ray.actor.ActorHandle


class PSRL_TrainWorker(ActorRolloutRefWorker):
    def __init__(self, config: DictConfig, role: str, psrl_config: DictConfig, train_interface: TrainInterface, nixl_interface: NIXLInterface) -> None:
        super().__init__(config, role)
        self.psrl_config = psrl_config
        self.train_interface = train_interface
        self.nixl_interface = nixl_interface
        
        # Build logger
        self.log_prefix = f"TrainWorker_R{self.rank}"
        psrl_logger.addHandler(DualOutputHandler(self.log_prefix))
        psrl_logger.info(f"Initialized on {get_worker_info()}.")
     
    def get_node_id(self) -> str:
        """
        Get the node id of the train worker.
        """
        return ray.get_runtime_context().get_node_id()
        
    @property   
    def is_train_representive_rank(self) -> bool:
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
                client_name=f"NIXLTrainClient_{self.rank}",
                server_name=global_meta_server_name,
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
        
    def push_model_cpu(self) -> None:
        """
        Push the model weights to the PS. In 'cpu' mode, push the full state dict. In 'cpu_ref' mode, push a ray object_ref.
        In 'cpu' mode, the PS worker will block on large model transfer (potential bottleneck).
        In 'cpu_ref' mode, only the train worker blocks on ray.put, PS worker is non-blocking.
        """
        ps_manager_handle = self.train_interface.ps_manager_handle
        curr_ps_model_version = ray.get(ps_manager_handle.get_ps_model_version.remote())
        next_ps_model_version = curr_ps_model_version + 1
        # Gather the model state dict on rank 0
        # TODO: support FSDP2
        assert fsdp_version(self.actor_module_fsdp) == 1, "FSDP version 2 is not supported yet."
        psrl_logger.info(f"Gathering the full state dict on the CPU of the representive rank.")
        with FSDP.state_dict_type(self.actor_module_fsdp, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            full_state_dict = self.actor_module_fsdp.state_dict()
        if self.is_train_representive_rank:
            assert len(full_state_dict) > 0, "The model state dict shouldn't be empty on the representive worker."
            psrl_logger.info(f"Push the model via CPU on the representive rank (async).")
            if self.psrl_config.ps_mode == "cpu":
                # In 'cpu' mode, push the full state dict (PS worker will block on transfer)
                # But the training side does not need to wait for the push to complete, as it can be overlapped with the next-iteration training
                self.train_interface.ps_manager_handle.push_model_state_dict_cpu.remote(next_ps_model_version, full_state_dict)
            elif self.psrl_config.ps_mode == "cpu_ref":
                # In 'cpu_ref' mode, push a ray object_ref (PS worker is non-blocking)
                # But the training side needs to wait for the push to complete, as `ray.put` is blocking
                object_ref = ray.put(full_state_dict)  # This blocks until the state dict is in the object store
                self.train_interface.ps_manager_handle.push_model_state_dict_cpu_ref_list.remote(next_ps_model_version, [object_ref]) # Tricky part: manually wrap the object_ref in a list to avoid ray dereferencing the full state dict
            else:
                raise NotImplementedError(f"PSRL TrainWorker does not support PS mode '{self.psrl_config.ps_mode}' yet.")
        else:
            assert len(full_state_dict) == 0, "The model state dict should be empty on non-representive workers."
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        with log_dual_events("Initialize model", psrl_logger, event_type=EventType.INIT):
            super().init_model()
    
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
            log_gpu_memory_usage("After offload actor model during compute_log_prob", logger=psrl_logger)

        return output
                
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
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
            