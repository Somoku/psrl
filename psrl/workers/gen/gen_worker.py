import ray
import os
import uuid
import time
import logging
import torch
import torch.distributed as dist
import numpy as np

from ray.exceptions import RayTaskError
from torch.distributed.tensor import DTensor
from torch.distributed.device_mesh import init_device_mesh
from typing import Any, Callable, ClassVar, Optional, Union, List
from time import sleep
from omegaconf import DictConfig, open_dict
from dataclasses import dataclass
from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import get_torch_device
from verl.utils.fs import copy_to_local
from verl.utils.debug import log_gpu_memory_usage
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.workers.rollout.vllm_rollout import vllm_mode
from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager

from psrl.utils.ray import RayLock
from psrl.utils.dataset import DatasetType, DatasetHandle
from psrl.utils.logger import DualOutputHandler, get_worker_info, log_dual_events, log_single_event, EventType
from psrl.utils.state_dict import create_parameter_mapping, convert_vllm_inplace
from psrl.utils.nixl import NIXLClientType, NIXLInterface, NIXLStorageClient, global_meta_server_name
from psrl.workers.gen import PSRL_vLLMRollout


psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))


@dataclass
class GenInterface:
    """Info for the PSRL GenWorker."""
    rollout_instance_id: int
    dataset_handle: ray.actor.ActorHandle
    ps_manager_handle: ray.actor.ActorHandle


class PSRL_GenWorker(ActorRolloutRefWorker):
    def __init__(self, config: DictConfig, role: str, psrl_config: DictConfig, gen_interface: GenInterface, nixl_interface: NIXLInterface) -> None:
        super().__init__(config, role)
        self.psrl_config = psrl_config
        self.gen_interface = gen_interface
        self.nixl_interface = nixl_interface
        self.instance_dist_group = None

        # Build logger
        self.log_prefix = f"GenWorker_I{self.get_instance_id()}_R{self.get_instance_local_rank()}"
        psrl_logger.addHandler(DualOutputHandler(self.log_prefix))
        psrl_logger.info(f"Initialized on {get_worker_info()}.")
        
    def init_nixl_client(self):
        assert self.rollout, "Rollout must be initialized before calling init_nixl_client."
        """Initialize the NIXL client."""
        if self.psrl_config.nixl.server_mode == "storage_server":
            raise ValueError("Storage server mode is deprecated.")
        elif self.psrl_config.nixl.server_mode == "meta_server":
            self.nixl_storage_client = NIXLStorageClient(
                client_name=f"NIXLGenClient_I{self.get_instance_id()}_R{self.get_instance_local_rank()}",
                server_name=global_meta_server_name,
                use_gpu=True,
                client_type=NIXLClientType.PULL_SIDE,
                nixl_config=self.psrl_config.nixl,  
                nixl_interface=self.nixl_interface
            )
        else:
            raise ValueError(f"Invalid NIXL server mode: {self.psrl_config.nixl.server_mode}")
        psrl_logger.info(f"NIXL client initialized on port {self.nixl_storage_client.client_port}.")
        
    def nixl_protocol(self):
        # Register the state dict and sharding dict to the NIXL client
        psrl_logger.info(f"nixl client protocol step 0: convert_vllm_inplace")
        vllm_model = self.rollout.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model.state_dict()
        param_mapping = create_parameter_mapping(type(vllm_model), copy_to_local(self.config.model.path))
        unified_state_dict, local_sharding_dict = convert_vllm_inplace(param_mapping, vllm_model, tp_rank=self.rank)
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
        
    def get_node_id(self) -> str:
        """
        Get the node id of the rollout instance.
        """
        return ray.get_runtime_context().get_node_id()
    
    def get_instance_representive_rank(self) -> int:
        """
        The representative rank is the rank 0 of the rollout instance in current implementation (i.e., DP=1).
        """
        return 0
    
    def get_instance_ranks(self) -> List[int]:
        """
        Get the ranks of the rollout instance.
        The rollout instance is all the ranks of the dist group in current implementation (i.e., DP=1).
        """
        return list(range(self.world_size))
    
    def get_instance_local_rank(self) -> int:
        """
        Get the local rank of the rollout instance.
        It is just the global rank in the current implementation (i.e., DP=1).
        """
        return self.rank
    
    def get_instance_id(self) -> int:
        """
        Get the ID of the rollout instance.
        It is given by the gen_interface and is an unique ID for the dist group in current implementation (i.e., DP=1).
        """
        return self.gen_interface.rollout_instance_id
    
    @property   
    def is_instance_representive_rank(self) -> bool:
        """
        Check if the current rank is the representative rank.
        The representative rank is the rank 0 of the rollout instance in current implementation (i.e., DP=1).
        """
        return self.rank == self.get_instance_representive_rank()
    
    def _broadcast_int_val_from_representive_rank(self, val: Optional[int] = None) -> None:
        if self.instance_dist_group == None:
            # If the instance dist group is not set, we will create it
            self.instance_dist_group = dist.new_group(ranks=self.get_instance_ranks())
        if self.is_instance_representive_rank:
            # If the current rank is the representative rank, we will broadcast the value to all instance ranks
            dist.broadcast(torch.tensor([val], dtype=torch.int64).cuda(), src=self.get_instance_representive_rank(), group=self.instance_dist_group)
        else:
            # If the current rank is not the representative rank, we will receive the value from the representative rank
            val_tensor = torch.zeros(1, dtype=torch.int64).cuda()
            dist.broadcast(val_tensor, src=self.get_instance_representive_rank(), group=self.instance_dist_group)
            return val_tensor.item()
        
    def _build_rollout(self, trust_remote_code=False):
        tp = self.config.rollout.tensor_model_parallel_size
        assert self.world_size == tp, "Only support dp=1 for now"
        self.rollout_device_mesh = init_device_mesh("cuda", mesh_shape=(1, tp), mesh_dim_names=["dp", "infer_tp"])
        rollout_name = self.config.rollout.name
        assert rollout_name == "vllm" and vllm_mode == "spmd" and self.config.rollout.mode == "sync", "Only support vllm spmd sync rollout for now"
        
        log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=psrl_logger)
        local_path = copy_to_local(self.config.model.path)
        rollout = PSRL_vLLMRollout(
            model_path=local_path,
            config=self.config.rollout,
            tokenizer=self.tokenizer,
            model_hf_config=self.actor_model_config,
            device_mesh=self.rollout_device_mesh,
            trust_remote_code=trust_remote_code,
        )
        log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=psrl_logger)
        
        rollout_sharding_manager = FSDPVLLMShardingManager(
            module=self.actor_module_fsdp,
            inference_engine=rollout.inference_engine,
            model_config=self.actor_model_config,
            full_params="hf" in self.config.rollout.load_format,
            device_mesh=self.rollout_device_mesh,
            offload_param=self._is_offload_param,
        )
        log_gpu_memory_usage("After building sharding manager", logger=psrl_logger)
        
        return rollout, rollout_sharding_manager
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        with log_dual_events("Initialize model", psrl_logger, event_type=EventType.INIT):
            super().init_model()
    
    def pull_model(self) -> None:
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
            model.load_weights(((name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param.to(device, non_blocking=True)) for name, param in model_state_dict_cpu.items()))
            # Question: Do we need to clear the cache after loading the model?
            # get_torch_device().empty_cache()
            torch.cuda.synchronize()
        else:
            raise NotImplementedError(f"PSRL GenWorker does not support PS mode '{self.psrl_config.ps_mode}' yet.")
    
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
        
        return prompts
    
    def batch_gen(self, batch_dict: dict) -> None:
        """
        Generate sequences in batch mode.
        """
        # Preprocess the data
        # Convert the batch dictionary to DataProto
        batch: DataProto = DataProto.from_single_dict(batch_dict)
        # Convert the batch to prompts on device (i.e, cuda)
        prompts = self.get_prompts_on_device(batch)
        
        # Reserve data for the following requests
        batch_size = prompts.batch["input_ids"].size(0)
        assert batch_size == len(batch), f"Batch size {batch_size} does not match the length of the batch {len(batch)}."
        curr_request_id = self.rollout.get_curr_request_id()
        psrl_logger.info(f"curr_request_id is {curr_request_id}, begin to reserve the next {batch_size} requests.")
        
        # Step 1: Determine the model version to use and reserve requests in the PS worker
        # This is take place only on the representative rank of the rollout instance
        # Get the PS worker handle
        with log_dual_events("Reserve requests", psrl_logger, event_type=EventType.OTHER):
            ps_manager_handle = self.gen_interface.ps_manager_handle
            # Get the current rollout instance id
            rollout_instance_id = self.get_instance_id()
            # Get the model versions
            curr_rollout_instance_model_version = ray.get(ps_manager_handle.get_rollout_instance_model_version.remote(rollout_instance_id))
            if self.is_instance_representive_rank:
                curr_ps_model_version = ray.get(ps_manager_handle.get_ps_model_version.remote())
                needed_model_version = curr_rollout_instance_model_version # By default, we will use the current rollout instance model version
                # TODO (Done already): Implement a wrapper for ps_manager_handle to gurantee the `reserve_num` and `reserve_rollout_instance_request` are merged and called atomically
                # Add asyncio method to the ps_manager_handle ray actor to ensure that there won't be race condition when multiple GenWorkers are trying to reserve requests
                with RayLock(ps_manager_handle):
                    # Check if we can reserve the requests
                    # If not, we will wait until the requests can be reserved (the waiting will take place later)
                    max_reserve_num = ray.get(ps_manager_handle.get_max_reserve_num.remote(curr_rollout_instance_model_version))
                    if max_reserve_num < batch_size:
                        # Need to pull new model version
                        needed_model_version = curr_ps_model_version
                        # If the current PS model version is still not enough, we will wait for the training side to update the model version
                        while ray.get(ps_manager_handle.get_max_reserve_num.remote(needed_model_version)) < batch_size:
                            needed_model_version += 1
                    # TODO: Maybe we can support partial reservation, currently we fix the batch size outside
                    # assert max_reserve_num >= batch_size, f"Cannot reserve {batch_size} requests, only {max_reserve_num} requests can be reserved."
                    request_ids = list(range(curr_request_id + 1, curr_request_id + batch_size + 1))
                    futures = []
                    for request_id in request_ids:
                        futures.append(ps_manager_handle.reserve_rollout_instance_request.remote(
                            rollout_instance_id=rollout_instance_id,
                            local_request_id=request_id,
                            model_version=needed_model_version
                        ))
                    results = ray.get(futures)
                    for request_id, (buffer_id, entry_id) in zip(request_ids, results):
                        assert buffer_id is not None and entry_id is not None, f"Failed to reserve rollout instance {rollout_instance_id} request {request_id}."
                # Use the pytorch distributed communication here to broadcast the model version
                self._broadcast_int_val_from_representive_rank(needed_model_version)
            else:
                # If not the representative rank, we will get the needed_model_version from the representative rank
                needed_model_version = self._broadcast_int_val_from_representive_rank()
        
        # Step 2: Pull the model version if needed (may need waiting)
        # All the ranks should participate
        if needed_model_version != curr_rollout_instance_model_version:
            with log_dual_events(f"Wait for model version {needed_model_version}", psrl_logger, event_type=EventType.WAIT):
                ray.get(ps_manager_handle.wait_for_ps_model_version.remote(needed_model_version)) # This will block until the PS worker has the needed model version 
            # The PS model version may be higher than the needed model version
            # if a pushing happens between step 1 and step 2
            # but that is ok since a higher model version will not break the staleness
            with log_dual_events("Pull model", psrl_logger, event_type=EventType.PULL):
                self.pull_model()
            
        # Step 3: Generate sequences
        # All the ranks should participate
        # Note that the actual model version may be higher than the needed model version
        actual_model_version = ray.get(ps_manager_handle.get_rollout_instance_model_version.remote(rollout_instance_id))
        with log_dual_events(f"Core generation with model version {actual_model_version}", psrl_logger, event_type=EventType.GEN):
            outputs : DataProto = self.rollout.generate_sequences(prompts)
        
        # Step 4: Union the generated sequences with the input batch
        # This is take place only on the representative rank of the rollout instance
        if self.is_instance_representive_rank:
            batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
            # repeat to align with repeated responses in rollout
            batch = batch.repeat(repeat_times=self.config.rollout.n, interleave=True)
            batch = batch.union(outputs)
            sequences : List[DataProto] = batch.chunk(len(batch))
        
        # Step 5: Occupy requests in the PS worker
        # This is take place only on the representative rank of the rollout instance
        if self.is_instance_representive_rank:
            with log_dual_events("Occupy requests", psrl_logger, event_type=EventType.OTHER):
                futures = []
                for request_id, sequence in zip(request_ids, sequences):
                    # Occupy the request in the PS worker
                    futures.append(ps_manager_handle.occupy_rollout_instance_request.remote(
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
        rollout_instance_id = self.get_instance_id()
        # Get the dataset handle
        dataset_handle = self.gen_interface.dataset_handle
        
        # Register the rollout instance in the PS worker
        if self.is_instance_representive_rank:
            # Only the representive rank needs to register the rollout instance
            ray.get(self.gen_interface.ps_manager_handle.register_rollout_instance.remote(self.get_instance_id()))
        
        ending = False
        
        # Currently only need enter once for the rollout sharding manager
        # because we use the old_log_prob directly from the vllm rollout
        # otherwise, we need to enter the rollout sharding manager for each batch
        self.rollout_sharding_manager.__enter__()
        
        if self.psrl_config.gen_mode == "batch":
            round = 0
            while True:
                batch_dict = None
                # TODO: better implementation, currently need busy polling to get the current batch
                while True:
                    # Get the current batch from the dataset handle
                    try:
                        batch_dict = ray.get(dataset_handle.get_rollout_instance_batch_nowait.remote(
                            DatasetType.train, 
                            rollout_instance_id,
                            self.get_instance_local_rank()
                        ))
                    except RayTaskError as e:
                        if isinstance(e.cause, StopIteration):
                            psrl_logger.info("All data is generated.")
                        else:
                            psrl_logger.info(f"Unknown exception happened during obtaining training data: {type(e.cause)}")
                            raise
                        ending = True
                        break   
                    if batch_dict is not None:
                        break 
                    sleep(0.1)   
                if ending:
                    break
                # Core generation
                psrl_logger.info(f"Begin round {round} batch generation.")
                self.batch_gen(batch_dict=batch_dict)
                round += 1
                
        elif self.psrl_config.gen_mode == "stream":
            # TODO: Implement stream generation
            # should use the dataset_handle rather than the batched prompts inside the stream_gen method
            # the dataset_handle should support obtaining data in a streaming manner
            self.stream_gen()
            
        else:
            raise ValueError(f"Unsupported generation mode: {self.config.rollout.gen_mode}")
            
        self.rollout_sharding_manager.__exit__()