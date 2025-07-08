import ray
import os
import queue
import asyncio
import threading
import logging
import numpy as np

import torch
import torch.distributed as dist

from torch.distributed.tensor import DTensor
from torch.distributed.device_mesh import init_device_mesh
from torch.multiprocessing.reductions import reduce_tensor
from typing import Any, Optional, List
from omegaconf import DictConfig
from dataclasses import dataclass
from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import get_torch_device
from verl.utils.fs import copy_to_local
from verl.utils.debug import log_gpu_memory_usage
from verl.workers.megatron_workers import ActorRolloutRefWorker

from psrl.utils.ray import RayLock, AsyncLock
from psrl.utils.logger import DualOutputHandler, get_worker_info, log_dual_events, EventType
from psrl.workers.gen import PSRL_vLLMRollout, GenInterface
from psrl.workers.sharding_manager import PSRL_MegatronASyncvLLMShardingManager, PSRL_MegatronvLLMShardingManager


psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))

class PSRL_MegatronGenWorker(ActorRolloutRefWorker):

    @staticmethod
    def configure_worker(
        config,
        num_gpus: int | float,
        dp_idx: int,
        bundle_indices: list[int],
    ) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
        resources: dict[str, Any] = {}
        init_kwargs: dict[str, Any] = {}
        env_vars: dict[str, str] = {}

        if config is not None and hasattr(config, "rollout") and config.rollout.mode == "sync":
            return resources, env_vars, init_kwargs
        
        resources["num_gpus"] = num_gpus
        psrl_logger.info("Configuring PSRL GenWorker...")
        # Initialize configuration

        if bundle_indices is not None:

            bundle_id = bundle_indices[0] // len(bundle_indices)
            # NOTE: bundle_id is 0 if we prepare pg for each dp manually
            seed = dp_idx + 1000 + bundle_id

            init_kwargs["seed"] = seed
            env_vars["VLLM_CACHE_ROOT"] = os.path.expanduser(f"~/.cache/vllm/vllm_{seed}")

        is_part_of_parallel_workers = (
            bundle_indices is not None and len(bundle_indices) > 1
        ) or bundle_indices is None

        if is_part_of_parallel_workers:
            resources["num_gpus"] = 0
            resources["num_cpus"] = 0
            env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
        env_vars["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        env_vars["VLLM_SKIP_P2P_CHECK"] = "1"
        return resources, env_vars, init_kwargs

    def __init__(self, config: DictConfig, role: str, psrl_config: DictConfig, gen_interface: GenInterface, **kwargs) -> None:
        super().__init__(config, role)
        self.seed = kwargs.get("seed", 0)
        self.psrl_config = psrl_config
        self.gen_interface = gen_interface
        self.instance_dist_group = None
        self.dtype = self.config.rollout.dtype
        
        self._rollout_running = False
        self._generate_thread = None
        self.request_queue = queue.Queue()
        self.active_tasks = set()
        self.version_to_active_tasks: dict[int, set[asyncio.Task]] = {}

        self.version_task_lock = asyncio.Lock()
        self.version_to_task_num: dict[int, int] = {}
        self.require_version_update_event = asyncio.Event()
        self.wait_on_version_events: dict[int, Optional[asyncio.Event]] = {}
        self.wait_on_version_events[0] = None

        def patched_reduce(self):
            # We inherit from RuntimeError so the only thing in args is the message.
            args = self.args
            assert len(args) == 1
            msg = args[0]
            first_useful_frame = None  # self.first_useful_frame is not serializable...
            return (reconstruct_exception, (type(self), msg, self.inner_exception, None))

        def reconstruct_exception(cls, msg, inner_exception, first_useful_frame):
            try:
                instance = cls(msg)
                if hasattr(instance, 'inner_exception'):
                    instance.inner_exception = inner_exception
                return instance
            except Exception:
                return RuntimeError(f"{cls.__name__}: {msg}")

        torch._dynamo.exc.BackendCompilerFailed.__reduce__ = patched_reduce

        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        if self.psrl_config.gen_mode == "batch":
            self.request_num_queue = queue.Queue()
        else:
            assert self.config.rollout.mode == "psrl_async", \
                "Only support psrl_async mode for stream generation, please set rollout.mode to psrl_async in the config."

        self._request_queue_lock = threading.Lock()

        self._async_interrupt_event = asyncio.Event()
        self._async_resume_event = asyncio.Event()
        self._generate_loop = None
        self.gen_task = None
        
        # Build logger
        self.log_prefix = f"GenWorker_I{self.get_instance_id()}_R{self.get_instance_local_rank()}"
        psrl_logger.addHandler(DualOutputHandler(self.log_prefix))
        psrl_logger.info(f"Initialized on {get_worker_info()}.")
    
    def get_instance_representative_rank(self) -> int:
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
    def is_instance_representative_rank(self) -> bool:
        """
        Check if the current rank is the representative rank.
        The representative rank is the rank 0 of the rollout instance in current implementation (i.e., DP=1).
        """
        return self.rank == self.get_instance_representative_rank()
    
    def _broadcast_int_val_from_representative_rank(self, val: Optional[int] = None) -> None:
        if self.instance_dist_group == None:
            # If the instance dist group is not set, we will create it
            self.instance_dist_group = dist.new_group(ranks=self.get_instance_ranks())
        if self.is_instance_representative_rank:
            # If the current rank is the representative rank, we will broadcast the value to all instance ranks
            dist.broadcast(torch.tensor([val], dtype=torch.int64).cuda(), src=self.get_instance_representative_rank(), group=self.instance_dist_group)
        else:
            # If the current rank is not the representative rank, we will receive the value from the representative rank
            val_tensor = torch.zeros(1, dtype=torch.int64).cuda()
            dist.broadcast(val_tensor, src=self.get_instance_representative_rank(), group=self.instance_dist_group)
            return val_tensor.item()
        
    def _build_rollout(self, trust_remote_code=False):
        layer_name_mapping = {
            "qkv_layer_name": "self_attention.linear_qkv.",
            "gate_proj_layer_name": "linear_fc1.",
        }
        tp = self.config.rollout.tensor_model_parallel_size
        pp = self.config.rollout.pipeline_model_parallel_size
        assert self.world_size == tp * pp, "Only support dp=1 for now"
        self.rollout_device_mesh = init_device_mesh("cuda", mesh_shape=(1, pp, tp), mesh_dim_names=["dp", "pp", "infer_tp"])
        rollout_name = self.config.rollout.name
        # assert rollout_name == "vllm" and vllm_mode == "spmd" and self.config.rollout.mode == "sync", "Only support vllm spmd sync rollout for now"
        assert rollout_name == "vllm", "Only support vLLM rollout for now"
        
        log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=psrl_logger)
        local_path = copy_to_local(self.config.model.path)
        rollout = PSRL_vLLMRollout(
            model_path=local_path,
            config=self.config.rollout,
            tokenizer=self.tokenizer,
            model_hf_config=self.actor_model_config,
            device_mesh=self.rollout_device_mesh,
            trust_remote_code=trust_remote_code,
            seed=self.seed,
        )
        log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=psrl_logger)
        from verl.models.mcore import get_mcore_weight_converter
        weight_converter = get_mcore_weight_converter(self.actor_model_config, self.dtype)

        rollout_sharding_manager_cls = PSRL_MegatronvLLMShardingManager if self.config.rollout.mode == "sync" else PSRL_MegatronASyncvLLMShardingManager
        rollout_sharding_manager = rollout_sharding_manager_cls(
            actor_module=self.actor_module,
            inference_engine=rollout.inference_engine,
            model_config=self.actor_model_config,
            transformer_config=self.tf_config,
            rollout_config=self.config.rollout,
            layer_name_mapping=layer_name_mapping,
            weight_converter=weight_converter,
            device_mesh=self.rollout_device_mesh,
            seed=self.seed,
            offload_param=self._is_offload_param,
            bridge=None,
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
        ps_handle = self.gen_interface.ps_handle
        model = self.rollout.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
        device = get_torch_device().current_device()
        
        if self.psrl_config.ps_mode == "cpu" or self.psrl_config.ps_mode == "cpu_ref":
            if self.psrl_config.ps_mode == "cpu":
                # In 'cpu' mode, pull the full state dict (PS worker will block on transfer)
                model_state_dict_cpu = ray.get(ps_handle.pull_model_state_dict_cpu.remote(self.get_instance_id()))
            elif self.psrl_config.ps_mode == "cpu_ref":
                # In 'cpu_ref' mode, get the object_ref and ray.get it (PS worker is non-blocking)
                object_ref = ray.get(ps_handle.pull_model_state_dict_cpu_ref.remote(self.get_instance_id()))
                model_state_dict_cpu = ray.get(object_ref)  # This blocks until the state dict is available in the object store
            # Load the model state dict to the vllm model
            # sharding will be handled automatically inside vllm
            model.load_weights(((name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param.to(device, non_blocking=True)) for name, param in model_state_dict_cpu.items()))
            # Question: Do we need to clear the cache after loading the model?
            # get_torch_device().empty_cache()
            torch.cuda.synchronize()
        else:
            raise NotImplementedError(f"PSRL GenWorker does not support PS mode '{self.psrl_config.ps_mode}' yet.")
    
    async def pull_model_async(self) -> None:
        """
        Pull the model state dict from PS via CPU and update the rollout model weights.
        In 'cpu' mode, pull the full state dict (potential bottleneck for large models).
        In 'cpu_ref' mode, get the ray object_ref and ray.get it (parallel, non-blocking for PS worker).
        """
        assert self.config.rollout.mode == "psrl_async", "Only support psrl_async mode for async pull model."
        ps_handle = self.gen_interface.ps_handle
        device = get_torch_device().current_device()
        
        if self.psrl_config.ps_mode == "cpu" or self.psrl_config.ps_mode == "cpu_ref":
            if self.psrl_config.ps_mode == "cpu":
                # In 'cpu' mode, pull the full state dict (PS worker will block on transfer)
                model_state_dict_cpu = await ps_handle.pull_model_state_dict_cpu.remote(self.get_instance_id())
            elif self.psrl_config.ps_mode == "cpu_ref":
                # In 'cpu_ref' mode, get the object_ref and ray.get it (PS worker is non-blocking)
                object_ref = await ps_handle.pull_model_state_dict_cpu_ref.remote(self.get_instance_id())
                model_state_dict_cpu = await object_ref  # This blocks until the state dict is available in the object store
            # Load the model state dict to the vllm model
            # sharding will be handled automatically inside vllm
            # NOTE: transfer from CPU to GPU is handled inside vLLM extension function `load_weights`.
            params_to_load = [(name, reduce_tensor(param.full_tensor()) if isinstance(param, DTensor) else reduce_tensor(param)) for name, param in model_state_dict_cpu.items()]
            loaded_params = await self.rollout.inference_engine.collective_rpc(
                "load_weights",
                args=(params_to_load,),
            )

            if loaded_params is None:
                psrl_logger.error(f"Error: Worker failed to update weights. Result: {loaded_params}")
                raise
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
    
    def batch_gen(self, batch_data: DataProto, rollout_queue) -> None:
        """
        Generate sequences in batch mode.
        """
        # Convert the batch to prompts on device (i.e, cuda)
        batch_data = batch_data.to(get_torch_device().current_device())
        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id if self.generation_config is not None else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id if self.generation_config is not None else self.tokenizer.pad_token_id,
        }
        batch_data.meta_info.update(meta_info)
        prompts = batch_data
        
        # Reserve data for the following requests
        batch_size = prompts.batch["input_ids"].size(0)
        assert batch_size == len(batch_data), f"Batch size {batch_size} does not match the length of the batch {len(batch_data)}."

        if self.config.rollout.n > 1:
            parent_ids = prompts.non_tensor_batch["parent_id"]
            parent_ids = np.unique(parent_ids)
        else:
            reserve_size = batch_size
        
        # Step 1: Determine the model version to use and reserve requests in the PS worker
        # This is take place only on the representative rank of the rollout instance
        # Get the PS worker handle
        with log_dual_events("Reserve requests", psrl_logger, event_type=EventType.OTHER):
            ps_handle = self.gen_interface.ps_handle
            # Get the current rollout instance id
            rollout_instance_id = self.get_instance_id()
            # Get the model versions
            curr_rollout_instance_model_version = ray.get(ps_handle.get_rollout_instance_model_version.remote(rollout_instance_id))
            if self.is_instance_representative_rank:
                curr_ps_model_version = ray.get(ps_handle.get_ps_model_version.remote())
                needed_model_version = curr_rollout_instance_model_version # By default, we will use the current rollout instance model version
                # TODO (Done already): Implement a wrapper for ps_handle to gurantee the `reserve_num` and `reserve_rollout_instance_request` are merged and called atomically
                # Add asyncio method to the ps_handle ray actor to ensure that there won't be race condition when multiple GenWorkers are trying to reserve requests
                with RayLock(ps_handle):
                    # Filter out the parent_ids for new requests that need to be reserved
                    if self.config.rollout.n > 1:
                        parent_ids = ray.get(ps_handle.filter_reserve_parent_ids.remote(parent_ids))
                        reserve_num = len(parent_ids)
                        reserve_size = reserve_num * self.config.rollout.n
                    # Check if we can reserve the requests
                    # If not, we will wait until the requests can be reserved (the waiting will take place later)
                    max_reserve_num = ray.get(ps_handle.get_max_reserve_num.remote(curr_rollout_instance_model_version))
                    if max_reserve_num < reserve_size:
                        # Need to pull new model version
                        needed_model_version = curr_ps_model_version
                        # If the current PS model version is still not enough, we will wait for the training side to update the model version
                        while ray.get(ps_handle.get_max_reserve_num.remote(needed_model_version)) < batch_size:
                            needed_model_version += 1
                    # TODO: Maybe we can support partial reservation, currently we fix the batch size outside
                    # assert max_reserve_num >= batch_size, f"Cannot reserve {batch_size} requests, only {max_reserve_num} requests can be reserved."
                    futures = []
                    reserve_ids = parent_ids if self.config.rollout.n > 1 else prompts.non_tensor_batch["uid"]
                    if self.config.rollout.n > 1:
                        for parent_id in reserve_ids:
                            futures.append(ps_handle.reserve_rollout_instance_request.remote(
                                rollout_instance_id=int(rollout_instance_id),
                                request_id=str(parent_id),
                                model_version=needed_model_version,
                                reserve_num=self.config.rollout.n,
                                by_parent=True,
                            ))
                    else:
                        for request_id in reserve_ids:
                            futures.append(ps_handle.reserve_rollout_instance_request.remote(
                                rollout_instance_id=int(rollout_instance_id),
                                request_id=str(request_id),
                                model_version=needed_model_version,
                            ))
                    results = ray.get(futures)
                    for request_id, (buffer_ids, entry_ids) in zip(reserve_ids, results):
                        assert buffer_ids is not None and entry_ids is not None, f"Failed to reserve rollout instance {rollout_instance_id} request {request_id}."
                # Use the pytorch distributed communication here to broadcast the model version
                self._broadcast_int_val_from_representative_rank(needed_model_version)
            else:
                # If not the representative rank, we will get the needed_model_version from the representative rank
                needed_model_version = self._broadcast_int_val_from_representative_rank()
        
        # Step 2: Pull the model version if needed (may need waiting)
        # All the ranks should participate
        if needed_model_version != curr_rollout_instance_model_version:
            with log_dual_events(f"Wait for model version {needed_model_version}", psrl_logger, event_type=EventType.WAIT):
                ray.get(ps_handle.wait_for_ps_model_version.remote(needed_model_version)) # This will block until the PS worker has the needed model version 
            # The PS model version may be higher than the needed model version
            # if a pushing happens between step 1 and step 2
            # but that is ok since a higher model version will not break the staleness
            with log_dual_events("Pull model", psrl_logger, event_type=EventType.PULL):
                self.pull_model()
            
        # Step 3: Generate sequences
        # All the ranks should participate
        # Note that the actual model version may be higher than the needed model version
        actual_model_version = ray.get(ps_handle.get_rollout_instance_model_version.remote(rollout_instance_id))
        prompts.non_tensor_batch["rollout_instance_id"] = np.array([rollout_instance_id] * len(prompts.batch))
        with log_dual_events(f"Core generation with model version {actual_model_version}", psrl_logger, event_type=EventType.GEN):
            outputs : DataProto = self.rollout.generate_sequences(prompts)
        
        # Step 4: Put the outputs to the rollout queue
        # This is take place only on the representative rank of the rollout instance
        if self.is_instance_representative_rank:
            rollout_queue.put(outputs)

        # NOTE: Occupy operation is moved to data processor

    async def stream_gen(self, rollout_queue):
        max_inflight_requests = self.config.rollout.max_inflight_requests
        
        stop_add_request = False

        def create_task_done_callback(require_version: int):
            def task_done_callback(task):
                self.version_to_active_tasks[require_version].discard(task)
            return task_done_callback
        
        async def process_request(request, needed_model_version):
            curr_rollout_instance_model_version = await ps_handle.get_rollout_instance_model_version.remote(rollout_instance_id)

            if needed_model_version != curr_rollout_instance_model_version:
                psrl_logger.info(f"Waiting for model version update, need {needed_model_version}, current {curr_rollout_instance_model_version}")
                assert needed_model_version > 0, "wait on model version 0 is not allowed, please check the model version logic"
                await self.wait_on_version_events[needed_model_version].wait()
                psrl_logger.info(f"Model version {needed_model_version} update done, proceeding with generation")

            actual_model_version = await ps_handle.get_rollout_instance_model_version.remote(rollout_instance_id)
            if actual_model_version != needed_model_version:
                psrl_logger.warning(f"Actual model version for generation is {actual_model_version}, needed model version is {needed_model_version}")
            
            request = request.to(get_torch_device().current_device())
            meta_info = {
                "eos_token_id": self.generation_config.eos_token_id if self.generation_config is not None else self.tokenizer.eos_token_id,
                "pad_token_id": self.generation_config.pad_token_id if self.generation_config is not None else self.tokenizer.pad_token_id,
            }
            request.meta_info.update(meta_info)
            request.non_tensor_batch["rollout_instance_id"] = np.array([rollout_instance_id] * len(request.batch))
            
            with log_dual_events(f"Core generation with model version {actual_model_version}", psrl_logger, event_type=EventType.GEN):
                result = await self.rollout.generate_sequences_async(request)

            rollout_queue.put(result)
            async with self.version_task_lock:
                self.version_to_task_num[needed_model_version] -= 1
                if self.version_to_task_num[needed_model_version] == 0:
                    self.wait_on_version_events.pop(needed_model_version, None)
                    self.version_to_task_num.pop(needed_model_version)
                    if (
                        self.version_to_task_num and
                        min(self.version_to_task_num.keys()) > needed_model_version and
                        not self.require_version_update_event.is_set()
                    ):
                        psrl_logger.info(f"All tasks for model version {needed_model_version} done, "
                                            f"waiting for new model version >= {min(self.version_to_task_num.keys())}")
                        self.require_version_update_event.set()
        
        async with self.sharding_manager:
            while self._rollout_running:
                if self._async_interrupt_event and self._async_interrupt_event.is_set():
                    psrl_logger.info(f"Generation interrupted, waiting for resume...")
                    await self._async_resume_event.wait()
                    psrl_logger.info(f"Generation resumed")
                
                while not stop_add_request:
                    if len(self.active_tasks) < max_inflight_requests:
                        request_data = None
                        try:
                            with self._request_queue_lock:
                                request_data = self.request_queue.get(block=False)
                            if request_data is None:
                                psrl_logger.info(f"[TRACE] Rank {self.rank}: Received end signal")
                                stop_add_request = True
                                await asyncio.sleep(0)
                                break
                            else:
                                assert len(request_data) == 1, \
                                    f"Expected batch_data length to be 1, got {len(request_data)}"
                        except queue.Empty:
                            break

                        with log_dual_events("Reserve requests", psrl_logger, event_type=EventType.OTHER):
                            ps_handle = self.gen_interface.ps_handle
                            # Get the current rollout instance id
                            rollout_instance_id = self.get_instance_id()
                            # Get the model versions
                            curr_rollout_instance_model_version = await ps_handle.get_rollout_instance_model_version.remote(rollout_instance_id)
                            # curr_ps_model_version = await ps_handle.get_ps_model_version.remote()
                            needed_model_version = curr_rollout_instance_model_version # By default, we will use the current rollout instance model version
                            # TODO (Done already): Implement a wrapper for ps_handle to gurantee the `reserve_num` and `reserve_rollout_instance_request` are merged and called atomically
                            # Add asyncio method to the ps_handle ray actor to ensure that there won't be race condition when multiple GenWorkers are trying to reserve requests
                            async with AsyncLock(ps_handle):
                                if self.config.rollout.n > 1:
                                    parent_ids = request_data.non_tensor_batch["parent_id"]
                                    parent_ids = np.unique(parent_ids)
                                    filtered_parent_ids = await ps_handle.filter_reserve_parent_ids.remote(parent_ids)
                                    reserve_num = len(filtered_parent_ids)
                                    reserve_size = reserve_num * self.config.rollout.n
                                else:
                                    reserve_size = len(request_data)
                                # Check if we can reserve the requests
                                # If not, we will wait until the requests can be reserved (the waiting will take place later)
                                max_reserve_num = await ps_handle.get_max_reserve_num.remote(curr_rollout_instance_model_version)
                                if max_reserve_num < reserve_size:
                                    curr_ps_model_version = await ps_handle.get_ps_model_version.remote()
                                    # Need to pull new model version
                                    needed_model_version = curr_ps_model_version
                                    # If the current PS model version is still not enough, we will wait for the training side to update the model version
                                    while (await ps_handle.get_max_reserve_num.remote(needed_model_version)) < reserve_size:
                                        needed_model_version += 1
                                # TODO: Maybe we can support partial reservation, currently we fix the batch size outside
                                # assert max_reserve_num >= batch_size, f"Cannot reserve {batch_size} requests, only {max_reserve_num} requests can be reserved."
                                reserve_ids = filtered_parent_ids if self.config.rollout.n > 1 else request_data.non_tensor_batch["uid"]
                                if len(reserve_ids) > 0:
                                    futures = []
                                    for request_id in reserve_ids:
                                        futures.append(ps_handle.reserve_rollout_instance_request.remote(
                                            rollout_instance_id=int(rollout_instance_id),
                                            request_id=str(request_id),
                                            model_version=needed_model_version,
                                            reserve_num=self.config.rollout.n,
                                            by_parent=self.config.rollout.n > 1,
                                        ))
                                    results = await asyncio.gather(*futures)
                                    psrl_logger.debug(f"Reserved requests for rollout instance {rollout_instance_id} with {results=}")
                                    for request_id, (buffer_ids, entry_ids) in zip(reserve_ids, results):
                                        assert buffer_ids is not None and entry_ids is not None, f"Failed to reserve rollout instance {rollout_instance_id} request {request_id}."
                        
                        # Check model version and add waiting tasks which are woken up by event
                        async with self.version_task_lock:
                            self.version_to_task_num[needed_model_version] = self.version_to_task_num.get(needed_model_version, 0) + 1
                            if needed_model_version != curr_rollout_instance_model_version:
                                if needed_model_version not in self.wait_on_version_events:
                                    psrl_logger.info(f"Creating wait_on_version_event for model version {needed_model_version}")
                                    self.wait_on_version_events[needed_model_version] = asyncio.Event()
                                    if min(self.version_to_task_num.keys()) == needed_model_version and not self.require_version_update_event.is_set():
                                        psrl_logger.info(f"Setting require_version_update_event for model version {needed_model_version}")
                                        self.require_version_update_event.set()

                        task = self._generate_loop.create_task(process_request(request_data, needed_model_version))
                        task.add_done_callback(create_task_done_callback(needed_model_version))
                        if needed_model_version not in self.version_to_active_tasks:
                            self.version_to_active_tasks[needed_model_version] = set()
                        self.version_to_active_tasks[needed_model_version].add(task)
                        await asyncio.sleep(0)
                    else:
                        break

                # Pull model and wake up waiting tasks
                if self.require_version_update_event.is_set():
                    psrl_logger.info(f"Require_version_update_event is set, checking for model version update")
                    curr_rollout_instance_model_version = await ps_handle.get_rollout_instance_model_version.remote(rollout_instance_id)
                    psrl_logger.info(f"Current rollout instance model version is {curr_rollout_instance_model_version}, waiting for update")
                    # wait_tasks = self.version_to_active_tasks[curr_rollout_instance_model_version]
                    # await asyncio.gather(*wait_tasks, return_exceptions=False)
                    # self.version_to_active_tasks.pop(curr_rollout_instance_model_version, None)
                    async with self.version_task_lock:
                        needed_model_version = min(self.version_to_task_num.keys())
                    with log_dual_events(f"Wait for model version {needed_model_version}", psrl_logger, event_type=EventType.WAIT):
                        await self.gen_interface.ps_handle.wait_for_ps_model_version.remote(needed_model_version) # This will block until the PS worker has the needed model version 
                    
                    # The PS model version may be higher than the needed model version
                    # if a pushing happens between step 1 and step 2
                    # but that is ok since a higher model version will not break the staleness
                    with log_dual_events("Pull model", psrl_logger, event_type=EventType.PULL):
                        await self.pull_model_async()
                    self.require_version_update_event.clear()
                    self.wait_on_version_events[needed_model_version].set()
                if stop_add_request:
                    curr_rollout_instance_model_version = await ps_handle.get_rollout_instance_model_version.remote(rollout_instance_id)
                    async with self.version_task_lock:
                        max_active_task_version = max(self.version_to_active_tasks.keys(), default=-1)
                    if max_active_task_version > curr_rollout_instance_model_version:
                        psrl_logger.info(f"Waiting for all tasks with model version {max_active_task_version} to finish, current model version is {curr_rollout_instance_model_version}")
                        await self.require_version_update_event.wait()
                    elif max_active_task_version == -1:
                        psrl_logger.info(f"All tasks done, stopping generation")
                        break
                await asyncio.sleep(0)

    def busy_loop_generate_sequences(self, rollout_queue) -> None:
        """
        Busy loop to generate sequences.
        """
        # Register the rollout instance in the PS worker
        if self.is_instance_representative_rank:
            # Only the representative rank needs to register the rollout instance
            ray.get(self.gen_interface.ps_handle.register_rollout_instance.remote(self.get_instance_id()))
        
        # Currently only need enter once for the rollout sharding manager
        # because we use the old_log_prob directly from the vllm rollout
        # otherwise, we need to enter the rollout sharding manager for each batch
        
        if self.psrl_config.gen_mode == "batch":
            def batch_gen_loop():
                with self.sharding_manager:
                    round = 0
                    while True:
                        request_num = self.request_num_queue.get()
                        if request_num is None:
                            psrl_logger.info("Received end signal, all data is generated. Stopping generation.")
                            break
                        if request_num == 0:
                            psrl_logger.info("Received request_num 0, skipping generation.")
                            continue

                        assert request_num > 0, f"Received invalid request_num: {request_num}, should be greater than 0."
                        
                        requests = []
                        with self._request_queue_lock:
                            for _ in range(request_num):
                                # Get the next request from the request queue
                                try:
                                    request = self.request_queue.get_nowait()
                                except queue.Empty:
                                    raise ValueError("Request queue is empty, cannot get next request.")
                                requests.append(request)
                        batch_data = DataProto.concat(requests)
                        psrl_logger.info(f"Begin round {round} batch with {len(batch_data)} requests for batch generation.")
                        self.batch_gen(batch_data, rollout_queue)
                        round += 1

            if self._generate_loop is None:
                self._generate_loop = asyncio.new_event_loop()
            
            def run_generate(loop):
                asyncio.set_event_loop(loop)
                self._async_resume_event.set()
                loop.run_until_complete(asyncio.to_thread(batch_gen_loop))
            
            self._rollout_running = True
            self._generate_thread = threading.Thread(
                target=run_generate,
                args=(self._generate_loop,),
                daemon=True
            )
            self._generate_thread.start()
                
        elif self.psrl_config.gen_mode == "stream":
            try:
                self._generate_loop = asyncio.get_running_loop()
                self.gen_task = self._generate_loop.create_task(self.stream_gen(rollout_queue))
                self._rollout_running = True
                psrl_logger.info(f"Stream generation task created")
            except RuntimeError:
                psrl_logger.error(f"No event loop running, cannot start stream generation")
                raise RuntimeError("Stream generation requires an active event loop in the main thread")
            
        else:
            raise ValueError(f"Unsupported generation mode: {self.config.rollout.gen_mode}")
        
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def shutdown_generate(self):
        if self._generate_thread and self._generate_thread.is_alive():
            self._rollout_running = False
            asyncio.gather(self.gen_task)
            if self._generate_loop and self._generate_loop.is_running():
                self._generate_loop.call_soon_threadsafe(self._generate_loop.stop)
            self._generate_thread.join()
            self._generate_thread = None
        
        self._generate_loop = None
        self._async_interrupt_event = None
        self._async_resume_event = None

        while not self.request_queue.empty():
            self.request_queue.get_nowait()
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def interrupt_all_requests(self, rollout_queue):
        print(f"[TRACE] Rank {self.rank}: Start to interrupt all requests")

        if self._generate_thread and self._generate_thread.is_alive():
            self._generate_loop.call_soon_threadsafe(self._async_interrupt_event.set)
            self._generate_loop.call_soon_threadsafe(self._async_resume_event.clear)
        else:
            self._async_interrupt_event.set()
            self._async_resume_event.clear()
        
        with self._request_queue_lock:
            psrl_logger.debug(f"Interrupting all requests, current queue size: {self.request_queue.qsize()}")
            interrupted_requests = []
            while not self.request_queue.empty():
                try:
                    request = self.request_queue.get_nowait()
                    if request is None:
                        break
                    request.meta_info["interrupted"] = True
                    interrupted_requests.append(request)
                except queue.Empty:
                    break
            
            # TODO: implement replay buffer and refactor this part
            for request in interrupted_requests:
                rollout_queue.put(request)
            
            psrl_logger.debug(f"Interrupted {len(interrupted_requests)} requests, current queue size: {self.request_queue.qsize()}")

        if self._generate_thread and self._generate_thread.is_alive() and self._generate_loop:
            future = asyncio.run_coroutine_threadsafe(
                self.rollout.interrupt_all_requests_async(), 
                self._generate_loop
            )
            interrupted_request_num = future.result()
            # clean all tasks in self.active_tasks
            for task in self.active_tasks:
                if not task.done():
                    task.cancel()
            self.active_tasks.clear()
            return interrupted_request_num
        else:
            return asyncio.run(self.rollout.interrupt_all_requests_async())

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def resume_generate(self):
        if self._generate_thread and self._generate_thread.is_alive():
            self._generate_loop.call_soon_threadsafe(self._async_interrupt_event.clear)
            self._generate_loop.call_soon_threadsafe(self._async_resume_event.set)
        else:
            self._async_interrupt_event.clear()
            self._async_resume_event.set()

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def add_request(self, requests):
        if isinstance(requests, DataProto):
            if len(requests) > 0:
                requests = [requests[i:i+1] for i in range(len(requests))]
                with self._request_queue_lock:
                    if self.psrl_config.gen_mode == "batch":
                        self.request_num_queue.put(len(requests))
                    for request in requests:
                        self.request_queue.put(request)
        elif requests is None:
            with self._request_queue_lock:
                if self.psrl_config.gen_mode == "batch":
                    self.request_num_queue.put(None)
                self.request_queue.put(requests)
        else:
            raise ValueError(f"Unsupported request type: {type(requests)}. Expected DataProto or None.")
