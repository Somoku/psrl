import ray
import os
import uuid
import sys
import logging
import torch
import torch.distributed as dist
import numpy as np

from torch.distributed.tensor import DTensor
from torch.distributed.device_mesh import init_device_mesh
from typing import Any, Callable, ClassVar, Optional, Union, List
from time import sleep
from omegaconf import DictConfig, open_dict
from dataclasses import dataclass
from verl import DataProto
from verl.utils.device import get_torch_device
from verl.utils.fs import copy_to_local
from verl.utils.debug import log_gpu_memory_usage
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.workers.rollout.vllm_rollout import vllm_mode
from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager

from psrl.utils.atomic import RayLock
from psrl.utils.dataset import DatasetType, DatasetHandle
from psrl.utils.logger import DualLogger
from psrl.workers.gen import PSRL_vLLMRollout
from psrl.workers.ps import PSRL_PSWorker


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class GenInterface:
    """Info for the PSRL GenWorker."""
    rollout_instance_id: int
    dataset_handle: ray.actor.ActorHandle
    ps_handle: ray.actor.ActorHandle


class PSRL_GenWorker(ActorRolloutRefWorker):
    def __init__(self, config: DictConfig, role: str, psrl_config: DictConfig, gen_interface: GenInterface) -> None:
        super().__init__(config, role)
        self.psrl_config = psrl_config
        self.gen_interface = gen_interface
        
    def _build_rollout(self, trust_remote_code=False):
        tp = self.config.rollout.tensor_model_parallel_size
        assert self.world_size == tp, "Only support dp=1 for now"
        self.rollout_device_mesh = init_device_mesh("cuda", mesh_shape=(1, tp), mesh_dim_names=["dp", "infer_tp"])
        rollout_name = self.config.rollout.name
        assert rollout_name == "vllm" and vllm_mode == "spmd" and self.config.rollout.mode == "sync", "Only support vllm spmd sync rollout for now"
        
        log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
        local_path = copy_to_local(self.config.model.path)
        rollout = PSRL_vLLMRollout(
            model_path=local_path,
            config=self.config.rollout,
            tokenizer=self.tokenizer,
            model_hf_config=self.actor_model_config,
            device_mesh=self.rollout_device_mesh,
            trust_remote_code=trust_remote_code,
        )
        log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)
        
        rollout_sharding_manager = FSDPVLLMShardingManager(
            module=self.actor_module_fsdp,
            inference_engine=rollout.inference_engine,
            model_config=self.actor_model_config,
            full_params="hf" in self.config.rollout.load_format,
            device_mesh=self.rollout_device_mesh,
            offload_param=self._is_offload_param,
        )
        log_gpu_memory_usage("After building sharding manager", logger=logger)
        
        return rollout, rollout_sharding_manager
    
    def pull_model_cpu(self) -> None:
        """
        Pull the model state dict from PS on CPU and update the rollout model weights.
        """
        ps_handle = self.gen_interface.ps_handle
        model_state_dict_cpu = ray.get(ps_handle.get_model_state_dict_cpu().remote())
        model = self.rollout.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner
        device = get_torch_device().current_device()
        
        # Load the model state dict to the vllm model
        # sharding will be handled automatically inside vllm
        model.load_weights(((name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param) for name, param in model_state_dict_cpu.items()))
        
        # Question: Do we need to clear the cache after loading the model?
        get_torch_device().empty_cache()

    
    def get_prompts_on_device(self, batch: DataProto) -> DataProto:
        """
        Convert a batch dictionary to DataProto.
        """    
        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_inputs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.extend(["multi_modal_data", "multi_modal_inputs"])
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        prompts = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        
        # The batch_size of prompts is already the number of sequences to generate per instance
        prompts = prompts.to(get_torch_device().current_device())
        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id if self.generation_config is not None else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id if self.generation_config is not None else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
    
    def batch_gen(self, batch_dict: dict) -> None:
        """
        Generate sequences in batch mode.
        """
        # Get the PS worker handle
        ps_handle = self.gen_interface.ps_handle
        # Get the current rollout instance id
        rollout_instance_id = self.gen_interface.rollout_instance_id
        # Get the model versions
        curr_rollout_instance_model_version = ray.get(ps_handle.get_rollout_instance_model_version.remote(rollout_instance_id))
        curr_ps_model_version = ray.get(ps_handle.get_ps_model_version.remote())
        needed_model_version = curr_rollout_instance_model_version # By default, we will use the current rollout instance model version
        
        logger.info(f"<GenWorker_{rollout_instance_id}:{self.rank}>: begin batch generation with model version {curr_rollout_instance_model_version} and PS model version {curr_ps_model_version}.")
        
        # Preprocess the data
        # Convert the batch dictionary to DataProto
        batch: DataProto = DataProto.from_single_dict(batch_dict)
        # Convert the batch to prompts on device (i.e, cuda)
        prompts = self.get_prompts_on_device(batch)
        
        # Reserve data for the following requests
        batch_size = prompts.batch["input_ids"].size(0)
        assert batch_size == len(batch), f"Batch size {batch_size} does not match the length of the batch {len(batch)}."
        curr_request_id = self.rollout.get_curr_request_id()
        
        # Step 1: Determine the model version to use and reserve requests in the PS worker
        # TODO (Done already): Implement a wrapper for ps_handle to gurantee the `reserve_num` and `reserve_rollout_instance_request` are merged and called atomically
        # Add asyncio method to the ps_handle ray actor to ensure that there won't be race condition when multiple GenWorkers are trying to reserve requests
        with RayLock(ps_handle):
            # Check if we can reserve the requests
            # If not, we will wait until the requests can be reserved (the waiting will take place later)
            max_reserve_num = ray.get(ps_handle.get_max_reserve_num.remote(curr_rollout_instance_model_version))
            if max_reserve_num < batch_size:
                # Need to pull new model version
                needed_model_version = curr_ps_model_version
                # If the current PS model version is still not enough, we will wait for the training side to update the model version
                while ray.get(ps_handle.get_max_reserve_num.remote(needed_model_version)) < batch_size:
                    needed_model_version += 1
            # TODO: Maybe we can support partial reservation, currently we fix the batch size outside
            # assert max_reserve_num >= batch_size, f"Cannot reserve {batch_size} requests, only {max_reserve_num} requests can be reserved."
            request_ids = list(range(curr_request_id + 1, curr_request_id + batch_size + 1))
            futures = []
            for request_id in request_ids:
                futures.append(ps_handle.reserve_rollout_instance_request.remote(
                    rollout_instance_id=rollout_instance_id,
                    local_request_id=request_id,
                    model_version=needed_model_version
                ))
            results = ray.get(futures)
            for request_id, (buffer_id, entry_id) in zip(request_ids, results):
                assert buffer_id is not None and entry_id is not None, f"Failed to reserve rollout instance {rollout_instance_id} request {request_id}."
        
        # Step 2: Pull the model version if needed (may need waiting)
        if needed_model_version != curr_rollout_instance_model_version:
            logger.info(f"<GenWorker_{rollout_instance_id}:{self.rank}>: begin waiting for model version {needed_model_version}.")
            ray.get(ps_handle.wait_for_ps_model_version.remote(needed_model_version)) # This will block until the PS worker has the needed model version 
            logger.info(f"<GenWorker_{rollout_instance_id}:{self.rank}>: end waiting for model version {needed_model_version}.")
            # The PS model version may be higher than the needed model version
            # if a pushing happens between step 1 and step 2
            # but that is ok since a higher model version will not break the staleness
            logger.info(f"<GenWorker_{rollout_instance_id}:{self.rank}>: begin pulling the model.")
            self.pull_model_cpu()
            logger.info(f"<GenWorker_{rollout_instance_id}:{self.rank}>: end pulling the model.")
            
        # Step 3: Generate sequences
        outputs : DataProto = self.rollout.generate(prompts)
        
        # Step 4: Union the generated sequences with the input batch
        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
        # repeat to align with repeated responses in rollout
        batch = batch.repeat(repeat_times=self.config.rollout.n, interleave=True)
        batch = batch.union(outputs)
        sequences : List[DataProto] = batch.chunk(len(batch))
        
        # Step 4: Occupy requests in the PS worker
        futures = []
        for request_id, sequence in zip(request_ids, sequences):
            # Occupy the request in the PS worker
            futures.append(ps_handle.occupy_rollout_instance_request.remote(
                rollout_instance_id=rollout_instance_id,
                local_request_id=request_id,
                data=sequence
            ))
        ray.get(futures)
      
    def stream_gen(self) -> None:
        """
        Generate sequences in stream mode.
        """
        raise NotImplementedError("Stream generation is not implemented yet.")
      
    def busy_loop_generate_sequences(self) -> None:
        """
        Busy loop to generate sequences.
        """
        # Get the current rollout instance id
        rollout_instance_id = self.gen_interface.rollout_instance_id
        # Get the dataset handle
        dataset_handle = self.gen_interface.dataset_handle
        
        # Build logger
        self.log_filename = f"GenWorker_{self.gen_interface.rollout_instance_id}:{self.rank}.log"
        self.original_stdout = sys.stdout
        sys.stdout = DualLogger(self.original_stdout, self.log_filename)
        logger.info(f"<GenWorker_{self.gen_interface.rollout_instance_id}:{self.rank}>: logger initialized.")
        
        # Register the rollout instance in the PS worker
        if self.rank == 0:
            # Only the rank 0 worker needs to register the rollout instance
            ray.get(self.gen_interface.ps_handle.register_rollout_instance.remote(self.gen_interface.rollout_instance_id))
        
        ending = False
        
        # Currently only need enter once for the rollout sharding manager
        # because we use the old_log_prob directly from the vllm rollout
        # otherwise, we need to enter the rollout sharding manager for each batch
        self.rollout_sharding_manager.__enter__()
        
        if self.psrl_config.gen_mode == "batch":
            while True:
                batch_dict = None
                # TODO: better implementation, currently need busy polling to get the current batch
                while True:
                    # Get the current batch from the dataset handle
                    try:
                        batch_dict = ray.get(dataset_handle.get_rollout_instance_batch_nowait.remote(DatasetType.train, rollout_instance_id))
                    except ray.RayTaskError as e:
                        if isinstance(e.cause, StopIteration):
                            logger.info("All data is generated.")
                        else:
                            logger.info(f"Unknown exception happened during obtaining training data: {type(e.cause)}")
                            raise
                        ending = True
                        break   
                    if batch_dict is not None:
                        break 
                    sleep(0.1)   
                if ending:
                    break
                # Core generation
                self.batch_gen(batch_dict=batch_dict)
                
        elif self.psrl_config.gen_mode == "stream":
            # TODO: Implement stream generation
            # should use the dataset_handle rather than the batched prompts inside the stream_gen method
            # the dataset_handle should support obtaining data in a streaming manner
            self.stream_gen()
            
        else:
            raise ValueError(f"Unsupported generation mode: {self.config.rollout.gen_mode}")
            
        self.rollout_sharding_manager.__exit__()
        
        # Shutdown the logger
        sys.stdout.close()  
        sys.stdout = self.original_stdout