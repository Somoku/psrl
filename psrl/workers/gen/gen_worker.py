import os
import queue
import asyncio
import threading
import logging
import numpy as np
from omegaconf import DictConfig
from typing import Any, Optional, List
from collections import deque, defaultdict
from transformers import AutoConfig

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.multiprocessing.reductions import reduce_tensor

import ray
from ray.util.queue import Queue as RayQueue

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_tokenizer
from verl.utils.device import get_torch_device, get_device_name
from verl.utils.fs import copy_to_local
from verl.utils.model import get_generation_config
from verl.utils.debug import log_gpu_memory_usage
from verl.workers.fsdp_workers import ActorRolloutRefWorker

from psrl.utils.server.command import CommandType, Command
from psrl.utils.logger import DualOutputHandler, get_worker_info, log_dual_events, EventType, deprecated
from psrl.utils.state_dict import create_parameter_mapping, convert_vllm_inplace
from psrl.utils.nixl import NIXLClientType, NIXLInterface, NIXLStorageClient, GLOBAL_META_SERVER_NAME, GLOBAL_GEN_CLIENT_NAME
from psrl.workers.gen import PSRL_vLLMRollout, GenInterface
from psrl.workers.ps.request_status_tracker import RequestStatus

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

class PSRL_GenWorker(Worker):
    @staticmethod
    def configure_worker(
        config,
        num_gpus: int | float,
        dp_idx: int,
        bundle_indices: list[int],
    ) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
        """
        Provides complete worker configuration (resource assignment, init args and environment variables)
        for vLLM tensor and pipeline parallelism.
       
        Args:
            config (DictConfig): The configuration for the worker.
            num_gpus (int | float): The number of GPUs available for the worker.
            dp_idx (int): The data parallel index of the worker.
            bundle_indices (list[int]): Indices of the bundles to which this worker belongs.

        Returns:
            tuple: A tuple containing:
                - resources (dict[str, Any]): Resources assigned to the worker.
                - env_vars (dict[str, str]): Environment variables for the worker.
                - init_kwargs (dict[str, Any]): Initialization arguments for the worker.
        """
        resources: dict[str, Any] = {}
        init_kwargs: dict[str, Any] = {}
        env_vars: dict[str, str] = {}

        # Nothing to do for SPMD-style synchronous rollout engines
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
            # Need to give each DP group its own vllm cache to address:
            # https://github.com/vllm-project/vllm/issues/18851
            env_vars["VLLM_CACHE_ROOT"] = os.path.expanduser(f"~/.cache/vllm/vllm_{seed}")

        # Check if this worker is part of a parallel group (TP or TP+PP).
        is_part_of_parallel_workers = (
            bundle_indices is not None and len(bundle_indices) > 1
        ) or bundle_indices is None

        # Leave the GPU assignment management of inner parallel workers to vLLM + Ray
        if is_part_of_parallel_workers:
            resources["num_gpus"] = 0
            resources["num_cpus"] = 0
            env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"

        env_vars["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        env_vars["VLLM_SKIP_P2P_CHECK"] = "1"
        return resources, env_vars, init_kwargs

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
        super().__init__()
        self.config = config
        self.dtype = self.config.rollout.dtype
        self.psrl_config = psrl_config
        self.gen_interface = gen_interface
        self.nixl_interface = nixl_interface
        self.status_queue = status_queue
        self.instance_dist_group = None

        self._lora_rank = self.config.model.get("lora_rank", 0)
        self._is_lora = self._lora_rank > 0

        self.seed = kwargs.get("seed", 0)
        
        self.curr_rollout_instance_model_version = 0 # Current model version for the rollout instance
        
        # Rollout loop management
        self._generate_loop = asyncio.get_running_loop() # Background async loop for generation
        self.gen_task = None # Generation task in stream mode
        self._generate_thread = None # Generation thread in batch mode
        self._rollout_running = False
        self.active_tasks = set() # Active tasks for the current generation loop
        
        # Async event management
        self._async_interrupt_event = asyncio.Event()
        self._async_resume_event = asyncio.Event()
        
        # Task for version update
        self.version_update_task = None
        
        # Task for engine status collection
        self.status_collection_task = None
        
        # Rollout request management
        self.request_queue = deque()
        if self.psrl_config.gen_mode == "batch":
            self.request_num_queue = queue.Queue()
        else:
            assert self.config.rollout.mode == "psrl_async", \
                "Only support psrl_async mode for stream generation, please set rollout.mode to psrl_async in the config."

        # Request management
        self.version_task_lock = asyncio.Lock()
        self.version_to_task_num: dict[int, int] = {}
        self.version_to_active_tasks: dict[int, set[asyncio.Task]] = defaultdict(lambda: set())
        self.request_id_to_active_tasks: dict[int, set[asyncio.Task]] = defaultdict(lambda: set())
        self.pending_version_requests: dict[int, List[DataProto]] = defaultdict(lambda: [])
        self.require_version_update_event = asyncio.Event()
        self.wait_on_version_events: dict[int, Optional[asyncio.Event]] = {}
        self.wait_on_version_events[0] = None

        # NIXL
        self.nixl_storage_client = None
        self.unified_state_dict = None
        self.unified_sharding_dict = None

        # For async model pulling
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        
        # Build logger
        self.log_prefix = f"GenWorker_I{self.get_instance_id()}_R{self.get_instance_local_rank()}"
        psrl_logger.addHandler(DualOutputHandler(self.psrl_config.logging_path, self.log_prefix))
        psrl_logger.info(f"Initialized on {get_worker_info()}.")

    def set_rollout_coordinator(self, rollout_coordinator):
        """Set the rollout coordinator for this GenWorker."""
        self.coordinator_handle = rollout_coordinator

    def _build_distributed(self):
        """Build the distributed process group for the rollout instance."""
        # Initialize the distributed process group
        if not dist.is_initialized():
            is_cuda_available = torch.cuda.is_available()
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            dist.init_process_group(backend="cpu:gloo,cuda:nccl" if is_cuda_available else "cpu:gloo,npu:hccl", rank=rank, world_size=world_size)

    def init_nixl_client(self):
        assert self.rollout, "Rollout must be initialized before calling init_nixl_client."
        """Initialize the NIXL client."""
        if self.psrl_config.nixl.server_mode == "storage_server":
            raise ValueError("Storage server mode is deprecated.")
        elif self.psrl_config.nixl.server_mode == "meta_server":
            self.nixl_storage_client = NIXLStorageClient(
                client_name=f"{GLOBAL_GEN_CLIENT_NAME}_I{self.get_instance_id()}_R{self.get_instance_local_rank()}",
                server_name=GLOBAL_META_SERVER_NAME,
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
        vllm_model = self.rollout.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
        param_mapping = create_parameter_mapping(type(vllm_model), copy_to_local(self.config.model.path))
        unified_state_dict, local_sharding_dict = convert_vllm_inplace(param_mapping, vllm_model, tp_rank=self.rank)
        psrl_logger.info(f"nixl client protocol step 1: connect_to_server")
        self.nixl_storage_client.connect_to_server()
        psrl_logger.info(f"nixl client protocol step 2: send_local_sharding")
        self.nixl_storage_client.send_local_sharding(local_sharding_dict)
        psrl_logger.info(f"nixl client protocol step 3: wait_for_server_sharding")
        unified_sharding_dict = self.nixl_storage_client.wait_for_server_sharding()
        # psrl_logger.info(f"unified_sharding_dict: {unified_sharding_dict}")
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

    def get_node_id(self) -> str:
        """
        Get the node id of the rollout instance.
        """
        return ray.get_runtime_context().get_node_id()

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
        # TODO(lhy): This is a temporary solution to broadcast the int value from the representive rank to all instance ranks
        # A more general solution is to use the `torch.distributed.broadcast_object_list` (https://pytorch.org/docs/stable/distributed.html#torch.distributed.broadcast_object_list)
        """Broadcast an integer value from the representative rank to all instance ranks."""
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

    def register_rollout_instance(self):
        """Register the rollout instance in the PS worker."""
        if self.is_instance_representative_rank:
            # Only the representative rank needs to register the rollout instance
            ray.get(self.gen_interface.ps_manager_handle.register_rollout_instance.remote(self.get_instance_id()))

    def _build_rollout(self, trust_remote_code=False):
        """
        Build the rollout engine and sharding manager for the PSRL GenWorker.
        NOTE: This method only supports building for one rollout instance at a time.
        """
        from torch.distributed.device_mesh import init_device_mesh

        self._build_distributed()

        tp = self.config.rollout.tensor_model_parallel_size
        pp = self.config.rollout.pipeline_model_parallel_size
        assert self.world_size == tp * pp, "Only support dp=1 for now"
        
        self.rollout_device_mesh = init_device_mesh(
            get_device_name(), mesh_shape=(1, pp, tp), mesh_dim_names=["dp", "pp", "infer_tp"]
        )
        rollout_name = self.config.rollout.name
        assert rollout_name == "vllm", "Only support vLLM rollout for now"
        
        # Build the rollout engine
        log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=psrl_logger)
        local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.get("use_shm", False))
        lora_kwargs = (
            {"lora_kwargs": {"enable_lora": True, "max_loras": 1, "max_lora_rank": self._lora_rank}}
            if self._is_lora
            else {}
        )
        self.actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code, attn_implementation="flash_attention_2")
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)
        rollout = PSRL_vLLMRollout(
            model_path=local_path,
            config=self.config.rollout,
            tokenizer=self.tokenizer,
            device_mesh=self.rollout_device_mesh,
            trust_remote_code=trust_remote_code,
            seed=self.seed,
            status_queue=self.status_queue,
            instance_id=self.get_instance_id(),
            **lora_kwargs,
        )
        log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=psrl_logger)
        
        return rollout
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def init_model(self):
        with log_dual_events("Initialize model", psrl_logger, event_type=EventType.INIT):
            self.rollout = self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))
    
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
            model.load_weights(((name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param.to(device, non_blocking=True)) for name, param in model_state_dict_cpu.items()))
            # Question: Do we need to clear the cache after loading the model?
            # get_torch_device().empty_cache()
            torch.cuda.synchronize()
        else:
            raise NotImplementedError(f"PSRL GenWorker does not support PS mode '{self.psrl_config.ps_mode}' yet.")
    
    async def ray_pull_model_async(self) -> None:
        """
        Pull the model state dict from PS via CPU and update the rollout model weights.
        In 'cpu' mode, pull the full state dict (potential bottleneck for large models).
        In 'cpu_ref' mode, get the ray object_ref and ray.get it (parallel, non-blocking for PS worker).
        """
        assert self.config.rollout.mode == "psrl_async", "Only support psrl_async mode for async pull model."
        ps_manager_handle = self.gen_interface.ps_manager_handle
        
        if self.psrl_config.ps_mode == "cpu" or self.psrl_config.ps_mode == "cpu_ref":
            if self.psrl_config.ps_mode == "cpu":
                # In 'cpu' mode, pull the full state dict (PS worker will block on transfer)
                model_state_dict_cpu = await ps_manager_handle.pull_model_state_dict_cpu.remote(self.get_instance_id())
            elif self.psrl_config.ps_mode == "cpu_ref":
                # In 'cpu_ref' mode, get the object_ref and ray.get it (PS worker is non-blocking)
                object_ref = await ps_manager_handle.pull_model_state_dict_cpu_ref.remote(self.get_instance_id())
                model_state_dict_cpu = await object_ref  # This blocks until the state dict is available in the object store
            # Load the model state dict to the vllm model
            # sharding will be handled automatically inside vllm
            # NOTE: transfer from CPU to GPU is handled inside vLLM extension function `load_weights`.
            params_to_load = [(name, reduce_tensor(param.full_tensor()) if isinstance(param, DTensor) else reduce_tensor(param)) for name, param in model_state_dict_cpu.items()]
            try:
                loaded_params = await self.rollout.inference_engine.collective_rpc(
                    "load_weights",
                    args=(params_to_load,),
                )
            except Exception as e:
                psrl_logger.error(f"Failed to load model parameters: {e}")
                raise e

            if loaded_params is None:
                psrl_logger.error(f"Worker failed to update weights. Result: {loaded_params}")
                raise
        else:
            raise NotImplementedError(f"PSRL GenWorker does not support PS mode '{self.psrl_config.ps_mode}' yet.")
    
    def nixl_pull_model(self) -> None:
        """
        Pull the model state dict from PS via NIXL and update the rollout model weights.
        """
        assert self.psrl_config.ps_mode == "nixl_cpu" or self.psrl_config.ps_mode == "nixl_gpu", "pull_model_state_dict_nixl should only be used in 'nixl_cpu' or 'nixl_gpu' mode."

        ps_manager_handle = self.gen_interface.ps_manager_handle
        ps_nixl_storage_client_names = ray.get(ps_manager_handle.get_ps_nixl_storage_client_names.remote())
        wait_operations = []
        for target_client_name in ps_nixl_storage_client_names: 
            for key in self.unified_state_dict:
                self.nixl_storage_client.client_read(target_client_name, key, b"gen_pull")
                wait_operations.append((key, target_client_name))
        # Generation cannot be overlapped with the NIXL pull, so we need to wait for all operations to complete
        for key, target_client_name in wait_operations:
            self.nixl_storage_client.wait(key, b"gen_pull", "READ", target_client=target_client_name)
        ps_manager_handle.pull_model_state_dict_nixl.remote(self.get_instance_id()) # This only updates the model version
        psrl_logger.info(f"NIXL pull model done.")

    async def nixl_pull_model_async(self) -> None:
        """
        Pull the model state dict from PS via NIXL and update the rollout model weights.
        """
        assert self.psrl_config.ps_mode == "nixl_cpu" or self.psrl_config.ps_mode == "nixl_gpu", "pull_model_state_dict_nixl should only be used in 'nixl_cpu' or 'nixl_gpu' mode."

        ps_manager_handle = self.gen_interface.ps_manager_handle
        ps_nixl_storage_client_names = await ps_manager_handle.get_ps_nixl_storage_client_names.remote()
        wait_operations = []
        for target_client_name in ps_nixl_storage_client_names: 
            for key in self.unified_state_dict:
                self.nixl_storage_client.client_read(target_client_name, key, b"gen_pull")
                wait_operations.append((key, target_client_name))
        # Generation cannot be overlapped with the NIXL pull, so we need to wait for all operations to complete
        for key, target_client_name in wait_operations:
            self.nixl_storage_client.wait(key, b"gen_pull", "READ", target_client=target_client_name)
        ps_manager_handle.pull_model_state_dict_nixl.remote(self.get_instance_id()) # This only updates the model version
        psrl_logger.info(f"NIXL pull model done.")

    def pull_model(self):
        """
        Pull the model state dict from PS and update the rollout model weights.
        """
        with log_dual_events("Pull model", psrl_logger, event_type=EventType.PULL):
            if self.psrl_config.ps_mode == "cpu" or self.psrl_config.ps_mode == "cpu_ref":
                self.ray_pull_model()
            elif self.psrl_config.ps_mode == "nixl_cpu" or self.psrl_config.ps_mode == "nixl_gpu":
                self.nixl_pull_model()

    async def pull_model_async(self):
        """
        Pull the model state dict from PS and update the rollout model weights in async mode.
        """
        with log_dual_events("Pull model", psrl_logger, event_type=EventType.PULL):
            if self.psrl_config.ps_mode == "cpu" or self.psrl_config.ps_mode == "cpu_ref":
                await self.ray_pull_model_async()
            elif self.psrl_config.ps_mode == "nixl_cpu" or self.psrl_config.ps_mode == "nixl_gpu":
                await self.nixl_pull_model_async()
    
    def get_prompts_on_device(self, batch: DataProto) -> DataProto:
        """Get generation prompts from the batch and move them to the current device."""    
        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_inputs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.extend(["multi_modal_inputs"])
        if "multi_modal_data" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        if "interaction_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("interaction_kwargs")
        if "index" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("index")
        if "agent_name" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("agent_name")
        prompts = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        
        # The batch_size of prompts is already the number of sequences to generate per instance
        prompts = prompts.to(get_torch_device().current_device())
        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        
        return prompts
    
    @deprecated("It is deprecated after we suppport agent loop. "
                "Refer to `generate` to generate sequences in batch.")
    def batch_gen(self, batch_data: DataProto, rollout_queue) -> None:
        """
        Generate sequences in batch mode.
        This method handles the generation of sequences in batch mode, where all requests are processed at once.
        It reserves requests in the PS worker, pulls the model if needed, and generates sequences using the rollout engine.

        Args:
            batch_data (DataProto): The batch data containing prompts and metadata.
            rollout_queue (queue.Queue): The queue to put the generated sequences.
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
        
        # NOTE: We no longer need to reserve requests, as the request already carries the version_tag
        needed_model_version = list(batch_data.non_tensor_batch["version_tag"])
        # Ensure all requests have the same model version
        for version in needed_model_version:
            if version != needed_model_version[0]:
                raise ValueError(f"All requests should have the same model version, but got {needed_model_version}.")
        needed_model_version = needed_model_version[0]  # Use the first one as the needed model version
        """
        # Step 1: Determine the model version to use and reserve requests in the PS worker
        # This is take place only on the representative rank of the rollout instance
        # Get the PS worker handle
        with log_dual_events("Reserve requests", psrl_logger, event_type=EventType.OTHER):
            ps_manager_handle = self.gen_interface.ps_manager_handle
            # Get the current rollout instance id
            rollout_instance_id = self.get_instance_id()
            # Get the model versions
            curr_rollout_instance_model_version = ray.get(ps_manager_handle.get_rollout_instance_model_version.remote(rollout_instance_id))
            if self.is_instance_representative_rank:
                curr_ps_model_version = ray.get(ps_manager_handle.get_ps_model_version.remote())
                needed_model_version = curr_rollout_instance_model_version # By default, we will use the current rollout instance model version
                # RayLock ensures that there won't be race condition when multiple GenWorkers are trying to reserve requests
                with RayLock(ps_manager_handle):
                    # Filter out the parent_ids for new requests that need to be reserved
                    if self.config.rollout.n > 1:
                        parent_ids = ray.get(ps_manager_handle.filter_reserve_parent_ids.remote(parent_ids))
                        reserve_num = len(parent_ids)
                        reserve_size = reserve_num * self.config.rollout.n
                    # Check if we can reserve the requests
                    # If not, we will wait until the requests can be reserved (the waiting will take place later)
                    max_reserve_num = ray.get(ps_manager_handle.get_max_reserve_num.remote(curr_rollout_instance_model_version))
                    if max_reserve_num < reserve_size:
                        # Need to pull new model version
                        needed_model_version = curr_ps_model_version
                        # If the current PS model version is still not enough, we will wait for the training side to update the model version
                        while ray.get(ps_manager_handle.get_max_reserve_num.remote(needed_model_version)) < batch_size:
                            needed_model_version += 1
                    # TODO(lhy): Maybe we can support partial reservation, currently we fix the batch size outside
                    # assert max_reserve_num >= batch_size, f"Cannot reserve {batch_size} requests, only {max_reserve_num} requests can be reserved."
                    futures = []
                    reserve_ids = parent_ids if self.config.rollout.n > 1 else prompts.non_tensor_batch["uid"]
                    if self.config.rollout.n > 1:
                        for parent_id in reserve_ids:
                            futures.append(ps_manager_handle.reserve_rollout_instance_request.remote(
                                rollout_instance_id=int(rollout_instance_id),
                                request_id=str(parent_id),
                                model_version=needed_model_version,
                                reserve_num=self.config.rollout.n,
                                by_parent=True,
                            ))
                    else:
                        for request_id in reserve_ids:
                            futures.append(ps_manager_handle.reserve_rollout_instance_request.remote(
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
        """
        
        rollout_instance_id = self.get_instance_id()
        ps_manager_handle = self.gen_interface.ps_manager_handle
        curr_rollout_instance_model_version = ray.get(ps_manager_handle.get_rollout_instance_model_version.remote(rollout_instance_id))
        
        # Pull the model version if needed (may need waiting)
        # All the ranks should participate
        if needed_model_version != curr_rollout_instance_model_version:
            with log_dual_events(f"Wait for model version {needed_model_version}", psrl_logger, event_type=EventType.WAIT):
                ray.get(ps_manager_handle.wait_for_ps_model_version.remote(needed_model_version)) # This will block until the PS worker has the needed model version 
            # The PS model version may be higher than the needed model version
            # if a pushing happens between step 1 and step 2
            # but that is ok since a higher model version will not break the staleness
            with log_dual_events("Pull model", psrl_logger, event_type=EventType.PULL):
                self.pull_model()
            
        # Generate sequences
        # All the ranks should participate
        # Note that the actual model version may be higher than the needed model version
        actual_model_version = ray.get(ps_manager_handle.get_rollout_instance_model_version.remote(rollout_instance_id))
        prompts.non_tensor_batch["rollout_instance_id"] = np.array([rollout_instance_id] * len(prompts.batch))
        with log_dual_events(f"Core generation with model version {actual_model_version}", psrl_logger, event_type=EventType.GEN):
            outputs : DataProto = self.rollout.generate_sequences(prompts)
        
        # Put the outputs to the rollout queue
        # This is take place only on the representative rank of the rollout instance
        if self.is_instance_representative_rank:
            rollout_queue.put(outputs)

        # NOTE: Occupy operation is moved to data processor

    @deprecated("It is deprecated after we support agent loop. "
                "Refer to `generate_async` to add a new async generation request.")
    async def stream_gen(self, rollout_queue, replay_buffer):
        """
        Generate sequences in stream mode.
        This method handles the async generation loop, processing requests from the request queue.
        It manages the model versioning and ensures that requests are processed in a timely manner.
        
        Args:
            rollout_queue (queue.Queue): The queue to put the generated sequences.
            replay_buffer (queue.Queue): The buffer to store the generated sequences for replay.
        """
        stop_add_request = False
        max_inflight_requests = self.config.rollout.max_inflight_requests
        rollout_instance_id = self.get_instance_id()
        ps_manager_handle = self.gen_interface.ps_manager_handle
        
        # Task processing function
        async def process_request(request, needed_model_version):
            result, request_status = await self._generate_async_task(request, needed_model_version)
            if request_status == RequestStatus.RUNNING:
                rollout_queue.put(result)
            elif request_status == RequestStatus.ROLLOUT_INTERRUPTED:
                replay_buffer.put(result)
            else:
                psrl_logger.error(f"Unknown request status {request_status}, cannot put result to the queue.")
            await asyncio.sleep(0)  # Yield control to allow other tasks to run

            # Check if we need to update the model version
            min_version = -1
            async with self.version_task_lock:
                self.version_to_task_num[needed_model_version] -= 1
                if self.version_to_task_num[needed_model_version] == 0:
                    self.version_to_task_num.pop(needed_model_version)
                    min_version = min(self.version_to_task_num.keys(), default=-1)
            if (
                min_version > needed_model_version and
                not self.require_version_update_event.is_set()
            ):
                psrl_logger.debug(f"All tasks for model version {needed_model_version} done, "
                                    f"waiting for new model version = {min(self.version_to_task_num.keys())}")
                self.require_version_update_event.set()
        
        # Start the main loop for async generation
        while self._rollout_running:
            # Wait for resuming if the generation is interrupted
            if self._async_interrupt_event and self._async_interrupt_event.is_set():
                psrl_logger.debug(f"Generation interrupted, waiting for resume...")
                await self._async_resume_event.wait()
                psrl_logger.debug(f"Generation resumed")
            
            curr_rollout_instance_model_version = self.curr_rollout_instance_model_version
            if not stop_add_request and len(self.request_queue) > 0:
                if len(self.active_tasks) < max_inflight_requests:
                    request_data = None
                    try:
                        request_data = self.request_queue.popleft()
                        if request_data is None:
                            stop_add_request = True
                            continue
                        else:
                            assert len(request_data) == 1, \
                                f"Expected request_data length to be 1, got {len(request_data)}"

                        needed_model_version = int(request_data.non_tensor_batch["version_tag"])
                        request_id = int(request_data.non_tensor_batch["uid"])

                        '''
                        # NOTE(linsh): We no longer need to reserve requests, as the request already carries the version_tag
                        with log_dual_events("Reserve requests", psrl_logger, event_type=EventType.OTHER):
                            ps_manager_handle = self.gen_interface.ps_manager_handle
                            # Get the current rollout instance id
                            rollout_instance_id = self.get_instance_id()
                            # Get the model versions
                            curr_rollout_instance_model_version = await ps_manager_handle.get_rollout_instance_model_version.remote(rollout_instance_id)
                            # curr_ps_model_version = await ps_manager_handle.get_ps_model_version.remote()
                            needed_model_version = curr_rollout_instance_model_version # By default, we will use the current rollout instance model version
                            # Add asyncio method to the ps_manager_handle ray actor to ensure that there won't be race condition when multiple GenWorkers are trying to reserve requests
                            async with AsyncLock(ps_manager_handle):
                                if self.config.rollout.n > 1:
                                    parent_ids = request_data.non_tensor_batch["parent_id"]
                                    parent_ids = np.unique(parent_ids)
                                    filtered_parent_ids = await ps_manager_handle.filter_reserve_parent_ids.remote(parent_ids)
                                    reserve_num = len(filtered_parent_ids)
                                    reserve_size = reserve_num * self.config.rollout.n
                                else:
                                    reserve_size = len(request_data)
                                # Check if we can reserve the requests
                                # If not, we will wait until the requests can be reserved (the waiting will take place later)
                                max_reserve_num = await ps_manager_handle.get_max_reserve_num.remote(curr_rollout_instance_model_version)
                                if max_reserve_num < reserve_size:
                                    curr_ps_model_version = await ps_manager_handle.get_ps_model_version.remote()
                                    # Need to pull new model version
                                    needed_model_version = curr_ps_model_version
                                    # If the current PS model version is still not enough, we will wait for the training side to update the model version
                                    while (await ps_manager_handle.get_max_reserve_num.remote(needed_model_version)) < reserve_size:
                                        needed_model_version += 1
                                # TODO(lhy): Maybe we can support partial reservation, currently we fix the batch size outside
                                # assert max_reserve_num >= batch_size, f"Cannot reserve {batch_size} requests, only {max_reserve_num} requests can be reserved."
                                reserve_ids = filtered_parent_ids if self.config.rollout.n > 1 else request_data.non_tensor_batch["uid"]
                                if len(reserve_ids) > 0:
                                    futures = []
                                    for request_id in reserve_ids:
                                        futures.append(ps_manager_handle.reserve_rollout_instance_request.remote(
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
                        '''
                        
                        if needed_model_version > curr_rollout_instance_model_version:
                            # Add the request to the pending version requests
                            self.pending_version_requests[needed_model_version].append(request_data)
                            async with self.version_task_lock:
                                self.version_to_task_num[needed_model_version] = self.version_to_task_num.get(needed_model_version, 0) + 1
                                min_version = min(self.version_to_task_num.keys())
                            if min_version == needed_model_version and not self.require_version_update_event.is_set():
                                psrl_logger.debug(f"Setting version update event for model version {needed_model_version}")
                                self.require_version_update_event.set()
                        elif needed_model_version == curr_rollout_instance_model_version:
                            # Process the request immediately
                            async with self.version_task_lock:
                                self.version_to_task_num[needed_model_version] = self.version_to_task_num.get(needed_model_version, 0) + 1
                            task = self._generate_loop.create_task(process_request(request_data, needed_model_version))
                            task.add_done_callback(self._create_task_done_callback(request_id, needed_model_version))
                            self.version_to_active_tasks[needed_model_version].add(task)
                            self.request_id_to_active_tasks[request_id].add(task)
                            self.active_tasks.add(task)
                        else:
                            raise ValueError(f"Needed model version {needed_model_version} is less than current rollout instance model version {curr_rollout_instance_model_version}. This should not happen.")
                        await asyncio.sleep(0) # Yield control to the event loop
                    except IndexError:
                        await asyncio.sleep(0)

            # Pull model and wake up waiting tasks
            if self.require_version_update_event.is_set():
                psrl_logger.debug(f"Require_version_update_event is set, checking for model version update")
                
                async with self.version_task_lock:
                    needed_model_version = min(self.version_to_task_num.keys())

                psrl_logger.debug(f"Current rollout instance model version is {self.curr_rollout_instance_model_version}, waiting for update to version {needed_model_version}")
                with log_dual_events(f"Wait for model version {needed_model_version}", psrl_logger, event_type=EventType.WAIT):
                    # This will block until the PS worker has the needed model version
                    await self.gen_interface.ps_manager_handle.wait_for_ps_model_version.remote(needed_model_version)
                
                # NOTE: The PS model version may be higher than the needed model version
                # if a pushing happens between waiting and pulling
                # but that is ok since a higher model version will not break the staleness
                with log_dual_events("Pull model", psrl_logger, event_type=EventType.PULL):
                    await self.pull_model_async()

                self.curr_rollout_instance_model_version = await ps_manager_handle.get_rollout_instance_model_version.remote(rollout_instance_id)
                if self.curr_rollout_instance_model_version > needed_model_version:
                    psrl_logger.warning(f"Actual model version for generation is {self.curr_rollout_instance_model_version}, needed model version is {needed_model_version}")

                self.require_version_update_event.clear()
                
                # Add pending requests of the updated model version to the request queue
                requests_of_needed_model_version = self.pending_version_requests.pop(needed_model_version, [])
                if requests_of_needed_model_version:
                    for request in reversed(requests_of_needed_model_version):
                        request_id = int(request.non_tensor_batch["uid"])
                        task = self._generate_loop.create_task(process_request(request, needed_model_version))
                        task.add_done_callback(self._create_task_done_callback(request_id, needed_model_version))
                        self.version_to_active_tasks[needed_model_version].add(task)
                        self.request_id_to_active_tasks[request_id].add(task)
                        self.active_tasks.add(task)
            
            if stop_add_request:
                async with self.version_task_lock:
                    max_active_task_version = max(self.version_to_task_num.keys(), default=-1)
                
                if max_active_task_version > curr_rollout_instance_model_version:
                    # Wait until all tasks with the current model version are done
                    psrl_logger.debug(f"Waiting for all tasks with model version {max_active_task_version} to finish, current model version is {curr_rollout_instance_model_version}")
                    await self.require_version_update_event.wait()
                elif max_active_task_version == -1:
                    # No more tasks to process
                    psrl_logger.info(f"All tasks done, stopping generation")
                    break
            await asyncio.sleep(0)

    @deprecated("It is deprecated after we support agent loop. "
                "Refer to `generate_async` to add a new async generation request.")
    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def busy_loop_generate_sequences(self, rollout_queue, replay_buffer) -> None:
        """
        Start the generation process in batch or stream mode.

        This method will register the rollout instance in the PS worker and start the generation loop.
        It will handle both batch and stream generation modes based on the configuration.
        
        Args:
            rollout_queue (Queue): The queue to put the generated sequences.
            replay_buffer (ReplayBuffer): The replay buffer to store the generated sequences.
        """
        # Register the rollout instance in the PS worker
        if self.is_instance_representative_rank:
            # Only the representative rank needs to register the rollout instance
            ray.get(self.gen_interface.ps_manager_handle.register_rollout_instance.remote(self.get_instance_id()))

        if self.psrl_config.gen_mode == "batch":
            # Batch generation loop
            def batch_gen_loop():
                # Currently only need enter once for the rollout sharding manager
                # because we use the old_log_prob directly from the vllm rollout
                # otherwise, we need to enter the rollout sharding manager for each batch
                batch_num = 0
                while True:
                    # Get request num for the next batch
                    request_num = self.request_num_queue.get()
                    if request_num is None:
                        psrl_logger.info("Received end signal, all data is generated. Stopping generation.")
                        break
                    if request_num == 0:
                        continue

                    assert request_num > 0, f"Received invalid request_num: {request_num}, should be greater than 0."
                    
                    requests = []
                    for _ in range(request_num):
                        # Get next batch from the request queue
                        try:
                            request = self.request_queue.popleft()
                        except IndexError:
                            raise ValueError("Request queue is empty, cannot get next request.")
                        requests.append(request)
                    batch_data = DataProto.concat(requests)
                    psrl_logger.debug(f"Begin {batch_num}-batch with {len(batch_data)} requests for batch generation.")
                    self.batch_gen(batch_data, rollout_queue)
                    batch_num += 1

            if self._generate_loop is None:
                self._generate_loop = asyncio.new_event_loop()
            
            # Run the batch generation loop in a separate thread
            def run_generate(loop):
                asyncio.set_event_loop(loop)
                self._async_resume_event.set()
                loop.run_until_complete(asyncio.to_thread(batch_gen_loop))
            
            self._generate_thread = threading.Thread(
                target=run_generate,
                args=(self._generate_loop,),
                daemon=True,
            )
            self._generate_thread.start()
            self._rollout_running = True
        elif self.psrl_config.gen_mode == "stream":
            try:
                self._generate_loop = asyncio.get_running_loop()
                # Create the background task for streaming generation
                self.gen_task = self._generate_loop.create_task(self.stream_gen(rollout_queue, replay_buffer))
                self._rollout_running = True
            except RuntimeError:
                raise RuntimeError("Stream generation requires an active event loop in the main thread")
            
        else:
            raise ValueError(f"Unsupported generation mode: {self.config.rollout.gen_mode}")

    @deprecated("It is deprecated after we support agent loop.")
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def shutdown_generate(self):
        """Shutdown the generation process and clean up all tasks and requests."""
        self._rollout_running = False
        # Shutdown the generation loop
        if self._generate_thread and self._generate_thread.is_alive():
            if self._generate_loop and self._generate_loop.is_running():
                self._generate_loop.call_soon_threadsafe(self._generate_loop.stop)
            self._generate_thread.join()
            self._generate_thread = None
        else:
            asyncio.gather(self.gen_task)
        
        self._generate_loop = None
        self._async_interrupt_event.clear()
        self._async_resume_event.clear()

        self.request_queue.clear()
    
    async def _async_interrupt_requests(self, request_ids=None):
        """Interrupt requests in the engine queue (waiting and running).
        
        If `request_ids` is None, it will interrupt all requests.
        If `request_ids` is provided, it will only interrupt the specified requests.
        
        Returns:
            int: The number of requests interrupted.
        """
        if not request_ids:
            # Interrupt all requests
            interrupt_request_num = await self.rollout.interrupt_all_requests_async()
            # Wait for all tasks to finish
            if self.active_tasks:
                await asyncio.gather(*self.active_tasks, return_exceptions=True)
            psrl_logger.debug(f"Interrupted all {interrupt_request_num} requests")
            return interrupt_request_num
        
        request_tasks = set()
        for request_id in request_ids:
            if request_id in self.request_id_to_active_tasks:
                request_tasks.update(self.request_id_to_active_tasks[request_id])
            else:
                psrl_logger.warning(f"Request ID {request_id} not found in active tasks.")
        psrl_logger.debug(f"Found {len(request_tasks)} active tasks for request IDs: {request_ids}")
        if request_tasks:
            await self.rollout.interrupt_requests_async(request_ids)
            # Wait for all tasks to finish
            await asyncio.gather(*request_tasks, return_exceptions=True)
            psrl_logger.debug(f"Interrupt all tasks done, clearing active tasks for request IDs: {request_ids}")
        interrupt_request_num = len(request_tasks)
        return interrupt_request_num
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    async def interrupt_requests(self, request_ids):
        """Interrupt specific requests in the engine queue (waiting and running)."""
        return await self._async_interrupt_requests(request_ids)
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    async def interrupt_all_requests(self):
        """Interrupt all requests in the engine queue (waiting and running)."""
        return await self._async_interrupt_requests()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    async def interrupt_generation(self):
        """Interrupt the generation process."""
        # Interrupt creating generation tasks
        if self._generate_thread and self._generate_thread.is_alive():
            self._generate_loop.call_soon_threadsafe(self._async_interrupt_event.set)
            self._generate_loop.call_soon_threadsafe(self._async_resume_event.clear)
        else:
            self._async_interrupt_event.set()
            self._async_resume_event.clear()
        
        # Interrupt all requests in the engine queue (waiting and running)
        interrupted_running_request_num = await self._async_interrupt_requests()
        
        # Wait and clean all tasks in self.active_tasks
        await asyncio.gather(*self.active_tasks, return_exceptions=True)
        self.active_tasks.clear()

        return {
            "interrupted_running_request_num": interrupted_running_request_num,
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def resume_generation(self):
        """Resume the generation process."""
        if self._generate_thread and self._generate_thread.is_alive():
            self._generate_loop.call_soon_threadsafe(self._async_resume_event.set)
            self._generate_loop.call_soon_threadsafe(self._async_interrupt_event.clear)
        else:
            self._async_resume_event.set()
            self._async_interrupt_event.clear()

    @deprecated("It is deprecated after we support agent loop.")
    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def add_request(self, requests):
        """Add requests to the request queue."""
        if isinstance(requests, DataProto):
            if len(requests) > 0:
                requests = [requests[i:i+1] for i in range(len(requests))]
                if self.psrl_config.gen_mode == "batch":
                    self.request_num_queue.put(len(requests))
                for request in requests:
                    self.request_queue.append(request)
            else:
                raise ValueError("Cannot add an empty DataProto to the request queue.")
        elif requests is None:
            if self.psrl_config.gen_mode == "batch":
                self.request_num_queue.put(None)
            self.request_queue.append(requests)
        else:
            raise ValueError(f"Unsupported request type: {type(requests)}. Expected DataProto or None.")

    def _create_task_done_callback(self, request_id: int, require_version: int):
        # Remove from the active tasks tracker when the task is done
        def task_done_callback(task):
            self.request_id_to_active_tasks[request_id].discard(task)
            self.version_to_active_tasks[require_version].discard(task)
            self.active_tasks.discard(task)
        return task_done_callback

    async def _generate_async_task(self, request: DataProto, needed_model_version: int):
        """
        An async task to generate sequences for a single request.
        This method handles the generation for a single request, managing model versioning
        and ensuring that the request is processed correctly.
        
        Args:
            request (DataProto): The generation request.
            needed_model_version (int): The model version required for this request.
        
        Returns:
            tuple: A tuple containing the generated sequences and the update status.
        """
        assert len(request) == 1, f"Expected request length to be 1, got {len(request)}"
        
        actual_model_version = await self.gen_interface.ps_manager_handle.get_rollout_instance_model_version.remote(self.get_instance_id())
        if actual_model_version != needed_model_version:
            psrl_logger.warning(f"Actual model version for generation is {actual_model_version}, needed model version is {needed_model_version}")
        
        # Update the request status to ROLLOUT_RUNNING
        request_ids = request.non_tensor_batch.get("uid", None)
        rollout_instance_id = self.get_instance_id()
        update_status_success = await self.gen_interface.ps_manager_handle.update_request_status.remote(
            request_ids.tolist(),
            RequestStatus.ROLLOUT_RUNNING,
            rollout_instance_id=rollout_instance_id,
        )
        if update_status_success[0]:
            # Prepare the request for generation
            request = request.to(get_torch_device().current_device())
            meta_info = {
                "eos_token_id": self.generation_config.eos_token_id if self.generation_config is not None else self.tokenizer.eos_token_id,
                "pad_token_id": self.generation_config.pad_token_id if self.generation_config is not None else self.tokenizer.pad_token_id,
            }
            request.meta_info.update(meta_info)
            request.non_tensor_batch["rollout_instance_id"] = np.array([rollout_instance_id] * len(request.batch))
        
            # Start the generation
            with log_dual_events(f"Core generation with model version {actual_model_version}", psrl_logger, event_type=EventType.GEN):
                vllm_outputs = await self.rollout.raw_generate_sequences_async(request)

            vllm_output = vllm_outputs[0][1] if isinstance(vllm_outputs, list) else vllm_outputs[1]
            assert len(vllm_output.outputs) == 1, \
                f"Expected no repeat in generation, got {len(vllm_output.outputs)} outputs."

            interrupted = (vllm_output.outputs[0].finish_reason == "abort")

            # Update the request status to ROLLOUT_COMPLETED or ROLLOUT_INTERRUPTED,
            # depending on `interrupted` field in the result
            update_status = RequestStatus.ROLLOUT_INTERRUPTED if interrupted else RequestStatus.RUNNING
            update_status_success = await self.gen_interface.ps_manager_handle.update_request_status.remote(
                request_ids.tolist(),
                update_status,
            )
            if update_status_success[0]:
                filtered_result = vllm_output
                return filtered_result, update_status
        return None, None

    def generate(self, requests: DataProto):
        """
        Generate sequences in batch mode.
        This method handles a batch of generation requests, managing model versioning
        and ensuring that the requests are processed in a timely manner.
        
        Args:
            requests (DataProto): The batch of generation requests.
        
        Returns:
            tuple: A tuple containing the generated sequences, the indices of the requests
                   that were successfully processed, and their corresponding update statuses.
        """
        rollout_instance_id = self.get_instance_id()

        curr_rollout_instance_model_version = self.curr_rollout_instance_model_version
        request_ids = requests.non_tensor_batch["uid"]
        needed_model_versions = requests.non_tensor_batch["version_tag"]
        # assert all version tag in needed_model_versions are equal
        if not all(version == needed_model_versions[0] for version in needed_model_versions):
            raise ValueError(f"All version tags must be the same, got {needed_model_versions}")
        needed_model_version = needed_model_versions[0]

        if needed_model_version >= curr_rollout_instance_model_version:
            if needed_model_version > curr_rollout_instance_model_version:
                # self.wait_on_version_events[needed_model_versions[0]] = asyncio.Event()
                # self.require_version_update_event.set()

                # await self.require_version_update_event.wait()
                with log_dual_events(f"Wait for model version {needed_model_version}", psrl_logger, event_type=EventType.WAIT):
                    # This will block until the PS worker has the needed model version
                    ray.get(self.gen_interface.ps_manager_handle.wait_for_ps_model_version.remote(needed_model_version))

                # NOTE: The PS model version may be higher than the needed model version
                # if a pushing happens between waiting and pulling
                # but that is ok since a higher model version will not break the staleness
                with log_dual_events("Pull model", psrl_logger, event_type=EventType.PULL):
                    self.pull_model()

                self.curr_rollout_instance_model_version = ray.get(self.gen_interface.ps_manager_handle.get_rollout_instance_model_version.remote(rollout_instance_id))
                psrl_logger.debug(f"Updated current rollout instance model version to {self.curr_rollout_instance_model_version}")
                if self.curr_rollout_instance_model_version > needed_model_version:
                    psrl_logger.warning(f"Actual model version for generation is {self.curr_rollout_instance_model_version}, needed model version is {needed_model_version}")

            update_status_success = ray.get(self.gen_interface.ps_manager_handle.update_request_status.remote(
                request_ids.tolist(),
                RequestStatus.ROLLOUT_RUNNING,
                rollout_instance_id=rollout_instance_id,
            ))
            filtered_request_idxs = [i for i, success in enumerate(update_status_success) if success]
            if filtered_request_idxs:
                filtered_requests = requests[filtered_request_idxs]
                # Prepare the request for generation
                filtered_requests = filtered_requests.to(get_torch_device().current_device())
                meta_info = {
                    "eos_token_id": self.generation_config.eos_token_id if self.generation_config is not None else self.tokenizer.eos_token_id,
                    "pad_token_id": self.generation_config.pad_token_id if self.generation_config is not None else self.tokenizer.pad_token_id,
                }
                filtered_requests.meta_info.update(meta_info)
                filtered_requests.non_tensor_batch["rollout_instance_id"] = np.array([rollout_instance_id] * len(filtered_requests.batch))

                # Start the generation
                with log_dual_events(f"Core generation with model version {self.curr_rollout_instance_model_version}", psrl_logger, event_type=EventType.GEN):
                    vllm_outputs = self.rollout.raw_generate_sequences(filtered_requests)

                vllm_outputs = [vllm_outputs[i] for i in range(len(vllm_outputs))]

                # Update the request status to ROLLOUT_COMPLETED or ROLLOUT_INTERRUPTED,
                # depending on `interrupted` field in the result
                update_statuses = [RequestStatus.ROLLOUT_INTERRUPTED if vllm_outputs[i].outputs[0].finish_reason == "abort" else RequestStatus.RUNNING for i in range(len(vllm_outputs))]
                update_status_success = ray.get(self.gen_interface.ps_manager_handle.update_request_status.remote(
                    request_ids.tolist(),
                    update_statuses,
                ))
                filtered_request_idxs = [i for i, success in enumerate(update_status_success) if success]
                if filtered_request_idxs:
                    filtered_result = [vllm_outputs[i] for i in filtered_request_idxs]
                    return filtered_result, filtered_request_idxs, update_statuses
                return None, None, None
        else:
            raise ValueError(f"Needed model version {needed_model_version} is less than current rollout instance model version {curr_rollout_instance_model_version}. This should not happen.")

    async def generate_async(self, request: DataProto):
        """
        Generate sequences asynchronously.
        This method handles a single async generation request, managing model versioning
        and ensuring that the request is processed in a timely manner.
        
        Args:
            request (DataProto): The async generation request.
        """
        assert len(request) == 1, f"Expected request length to be 1, got {len(request)}"
        rollout_instance_id = self.get_instance_id()

        # Wait for resuming if the generation is interrupted
        if self._async_interrupt_event and self._async_interrupt_event.is_set():
            psrl_logger.debug(f"Generation interrupted, waiting for resume...")
            await self._async_resume_event.wait()
            psrl_logger.debug(f"Generation resumed")

        curr_rollout_instance_model_version = self.curr_rollout_instance_model_version
        request_id = int(request.non_tensor_batch["uid"][0])
        needed_model_version = int(request.non_tensor_batch["version_tag"][0])
        if needed_model_version >= curr_rollout_instance_model_version:
            if needed_model_version > curr_rollout_instance_model_version:
                if needed_model_version not in self.wait_on_version_events:
                    self.wait_on_version_events[needed_model_version] = asyncio.Event()
                    # self.require_version_update_event.set()

                    await self.require_version_update_event.wait()
                    with log_dual_events(f"Wait for model version {needed_model_version}", psrl_logger, event_type=EventType.WAIT):
                        # This will block until the PS worker has the needed model version
                        await self.gen_interface.ps_manager_handle.wait_for_ps_model_version.remote(needed_model_version)
                    
                    # NOTE: The PS model version may be higher than the needed model version
                    # if a pushing happens between waiting and pulling
                    # but that is ok since a higher model version will not break the staleness
                    with log_dual_events("Pull model", psrl_logger, event_type=EventType.PULL):
                        await self.pull_model_async()

                    self.curr_rollout_instance_model_version = await self.gen_interface.ps_manager_handle.get_rollout_instance_model_version.remote(rollout_instance_id)
                    psrl_logger.debug(f"Updated current rollout instance model version to {self.curr_rollout_instance_model_version}")
                    if self.curr_rollout_instance_model_version > needed_model_version:
                        psrl_logger.warning(f"Actual model version for generation is {self.curr_rollout_instance_model_version}, needed model version is {needed_model_version}")

                    awake_version_events = [event for version, event in self.wait_on_version_events.items() if version == needed_model_version]
                    if awake_version_events:
                        psrl_logger.debug(f"Awake version events for model versions: {needed_model_version}")
                        for event in awake_version_events:
                            event.set()

                    self.require_version_update_event.clear()
                    
                await self.wait_on_version_events[needed_model_version].wait()
            task = self._generate_loop.create_task(
                self._generate_async_task(request, needed_model_version)
            )
            self.version_to_active_tasks[needed_model_version].add(task)
            self.request_id_to_active_tasks[request_id].add(task)
            self.active_tasks.add(task)
            task.add_done_callback(self._create_task_done_callback(
                int(request.non_tensor_batch["uid"][0]),
                needed_model_version,
            ))
            # Wait for the task to finish
            result = await task
            return result
        else:
            raise ValueError(f"Needed model version {needed_model_version} is less than current rollout instance model version {curr_rollout_instance_model_version}. This should not happen.")

    async def push_task(self, request_id: int, needed_model_version: int):
        """
        Push a new task for the given model version.
        
        Args:
            request_id (int): The ID of the request.
            needed_model_version (int): The model version needed for the request.
        """
        psrl_logger.debug(f"Pushing task for request_id {request_id} with needed_model_version {needed_model_version}")
        self.version_to_task_num[needed_model_version] = self.version_to_task_num.get(needed_model_version, 0) + 1
    
    async def pop_task(self, request_id: int, needed_model_version: int):
        """
        Pop a finished task for the given model version.
        If there are no more tasks for this model version, it will set the
        require_version_update_event to wake up the version update event handler.
        
        Args:
            request_id (int): The ID of the request.
            needed_model_version (int): The model version needed for the request.
        """
        psrl_logger.debug(f"Popping task for request_id {request_id} with needed_model_version {needed_model_version}")
        if needed_model_version in self.version_to_task_num:
            self.version_to_task_num[needed_model_version] -= 1
            if self.version_to_task_num[needed_model_version] == 0:
                self.version_to_task_num.pop(needed_model_version)
                self.wait_on_version_events.pop(needed_model_version, None)
                psrl_logger.debug(f"All tasks for model version {needed_model_version} done")
                if not self.require_version_update_event.is_set():
                    psrl_logger.debug(f"Setting require_version_update_event for model version {needed_model_version}")
                    # If there are no more tasks for this model version, we can set the event
                    # to wake up the version update event handler
                    self.require_version_update_event.set()
        else:
            psrl_logger.warning(f"Model version {needed_model_version} not found in version_to_task_num.")
