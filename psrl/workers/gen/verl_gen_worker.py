import os
import queue
import asyncio
import threading
import logging
import numpy as np
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Optional, Union, List
from collections import deque, defaultdict
from transformers import AutoConfig

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.distributed.device_mesh import init_device_mesh
from torch.multiprocessing.reductions import reduce_tensor

import ray
from ray.util.queue import Queue as RayQueue

from verl import DataProto
# from verl.single_controller.base import Worker
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_tokenizer
from verl.utils.device import get_torch_device, get_device_name, get_device_id
from verl.utils.fs import copy_to_local
from verl.utils.model import get_generation_config, update_model_config
from verl.utils.debug import log_gpu_memory_usage

from psrl.utils.ray import RayLock, AsyncRayLock
from psrl.utils.server.command import CommandType, Command
from psrl.utils.logger import DualOutputHandler, get_worker_info, log_dual_events, log_single_event, deprecated, EventType
from psrl.utils.converter import create_parameter_mapping
from psrl.utils.converter.vllm_converter import convert_vllm_inplace
from psrl.utils.nixl import NIXLInterface
from psrl.workers.gen import PSRL_vLLMRollout, GenInterface
from psrl.workers.ps.request_status_tracker import PSRL_RequestStatus

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

class PSRL_VerlGenWorker(ActorRolloutRefWorker):

    def __init__(
        self,
        config: DictConfig,
        role: str,
        psrl_config: DictConfig,
        gen_interface: GenInterface,
        nixl_interface: NIXLInterface,
        status_queue: RayQueue, 
        **kwargs,
    ) -> None:
        """
        Initialize the PSRL FSDP GenWorker.
        
        Args:
            config (DictConfig): The configuration for the worker.
            role (str): The role of the worker (e.g., "gen").
            psrl_config (DictConfig): The PSRL configuration.
            gen_interface (GenInterface): The interface for generation.
            nixl_interface (NIXLInterface): The interface for NIXL storage.
            **kwargs: Additional keyword arguments, including 'seed'.
        """
        super().__init__(config, role, **kwargs)
        self.psrl_config = psrl_config
        self.gen_interface = gen_interface
    
    def set_rollout_coordinator(self, rollout_coordinator):
        """Set the rollout coordinator for this GenWorker."""
        self.coordinator_handle = rollout_coordinator
    
    def get_instance_representative_rank(self) -> int:
        """
        The representative rank is the rank 0 of the rollout instance in current implementation (i.e., DP=1).
        """
        return 0
    
    def get_instance_id(self) -> int:
        """
        Get the ID of the rollout instance.
        It is given by the gen_interface and is an unique ID for the dist group in current implementation (i.e., DP=1).
        """
        return self.gen_interface.rollout_instance_id
    
    @property   
    def is_instance_representative_rank(self) -> bool:
        """
        Check if the current rank is the representative rank.
        The representative rank is the rank 0 of the rollout instance in current implementation (i.e., DP=1).
        """
        return self.rank == self.get_instance_representative_rank()
        
    def register_rollout_instance(self):
        """Register the rollout instance in the PS worker."""
        if hasattr(self, "_is_rollout_instance_registered"):
            return
        if self.is_instance_representative_rank:
            # Only the representative rank needs to register the rollout instance
            ray.get(self.gen_interface.ps_manager_handle.register_rollout_instance.remote(self.get_instance_id()))
        self._is_rollout_instance_registered = True
    
    def ray_pull_model(self) -> None:
        """
        Pull the model state dict from PS via CPU and update the rollout model weights.
        In 'cpu' mode, pull the full state dict (potential bottleneck for large models).
        In 'cpu_ref' mode, get the ray object_ref and ray.get it (parallel, non-blocking for PS worker).
        """
        ps_manager_handle = self.gen_interface.ps_manager_handle
        model = self.rollout.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
        device = get_torch_device().current_device()
        
        if self.psrl_config.ps_mode == "cpu" or self.psrl_config.ps_mode == "cpu_ref":
            if self.psrl_config.ps_mode == "cpu":
                # In 'cpu' mode, pull the full state dict (PS worker will block on transfer)
                model_state_dict_cpu = ray.get(ps_manager_handle.pull_model_state_dict_cpu.remote(self.get_instance_id()))
            elif self.psrl_config.ps_mode == "cpu_ref":
                # In 'cpu_ref' mode, get the object_ref and ray.get it (PS worker is non-blocking)
                object_ref = ray.get(ps_manager_handle.pull_model_state_dict_cpu_ref.remote(self.get_instance_id()))
                model_state_dict_cpu = ray.get(object_ref)  # This blocks until the state dict is available in the object store
            # Load the model state dict to the vllm model
            # sharding will be handled automatically inside vllm
            self.rollout_sharding_manager.update_params(model_state_dict_cpu)
            # model.load_weights(((name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param.to(device, non_blocking=True)) for name, param in model_state_dict_cpu.items()))
            # NOTE(lhy): Do we need to clear the cache after loading the model?
            get_torch_device().empty_cache()
            torch.cuda.synchronize()
            torch.distributed.barrier()
        else:
            raise NotImplementedError(f"PSRL VerlGenWorker does not support PS mode '{self.psrl_config.ps_mode}' yet.")
            
    def pull_model(self) -> None:
        assert self.config.rollout.mode == "sync", "Only support `sync` mode."
        if self.psrl_config.ps_mode == "cpu" or self.psrl_config.ps_mode == "cpu_ref":
            self.ray_pull_model()
        elif self.psrl_config.ps_mode == "nixl_cpu" or self.psrl_config.ps_mode == "nixl_gpu":
            raise NotImplementedError(f"PSRL VerlGenWorker does not support PS mode '{self.psrl_config.ps_mode}' yet.")
        else:
            raise NotImplementedError(f"PSRL VerlGenWorker does not support PS mode '{self.psrl_config.ps_mode}' yet.")
 
    # ------------- DEBUG METHODS -------------
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def verl_generate_sequences_all(self, prompts: DataProto):
        assert self.psrl_config.gen_mode == "batch", "verl_generate_sequences is only supported in batch mode"
        assert self.config.rollout.get("pipeline_model_parallel_size", 1) == 1, "verl_generate_sequences is only supported in pure tensor parallel mode"
        self.register_rollout_instance()
        with self.rollout_sharding_manager:
            if prompts.meta_info.get("need_pull_model", True):
                self.pull_model()
            prompts = prompts.to(get_device_id())
            meta_info = {
                "eos_token_id": self.generation_config.eos_token_id
                if self.generation_config is not None
                else self.tokenizer.eos_token_id,
                "pad_token_id": self.generation_config.pad_token_id
                if self.generation_config is not None
                else self.tokenizer.pad_token_id,
            }
            prompts.meta_info.update(meta_info)
            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            outputs = self.rollout.generate_sequences(prompts)
            outputs = self.rollout_sharding_manager.postprocess_data(outputs)
            outputs = outputs.to("cpu")
            get_torch_device().empty_cache()
            return outputs

    def verl_generate_sequences(self, prompts: DataProto):
        assert self.psrl_config.gen_mode == "batch", "verl_generate_sequences is only supported in batch mode"
        assert self.config.rollout.get("pipeline_model_parallel_size", 1) == 1, "verl_generate_sequences is only supported in pure tensor parallel mode"
        self.register_rollout_instance()
        if not hasattr(self, "_is_entered_sharding_manager"):
            self.rollout_sharding_manager.__enter__()
            self._is_entered_sharding_manager = True
        if prompts.meta_info.get("need_pull_model", True):
            self.pull_model()
        prompts = prompts.to(get_device_id())
        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        outputs = self.rollout.generate_sequences(prompts)
        return outputs.chunk(chunks=self.world_size)[self.rank].to("cpu")