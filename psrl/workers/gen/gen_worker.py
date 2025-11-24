import os
import queue
import asyncio
import warnings
import logging
import torch
import ray
import time
import numpy as np
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
from typing import Any, Optional, List
from collections import deque, defaultdict
from transformers import AutoConfig
from ray.util.queue import Queue as RayQueue
from torch.distributed.tensor import DTensor
from torch.distributed.device_mesh import init_device_mesh
from torch.multiprocessing.reductions import reduce_tensor

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_processor, hf_tokenizer, omega_conf_to_dataclass
from verl.utils.device import get_torch_device, get_device_name
from verl.utils.fs import copy_to_local
from verl.utils.model import get_generation_config, update_model_config

from psrl.utils.logger import DualOutputHandler, get_worker_info, log_dual_events, log_single_event, log_begin_event, log_end_event, deprecated, EventType
from psrl.utils.nixl import NIXLInterface
from psrl.workers.gen import PSRL_vLLMRollout
from psrl.workers.config import HFModelConfig, RolloutConfig
from psrl.workers.ps.request_status_tracker import PSRL_RequestStatus


psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@dataclass
class GenInterface:
    """Info for the PSRL GenWorker."""
    rollout_instance_id: int
    ps_manager_handle: ray.actor.ActorHandle
    status_queue: RayQueue


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
        if config.rollout.disable_attn:
            warnings.warn("CAUTION: you are disabling the attention, this should only be used for analysis purposes, not for training!")
            env_vars["VLLM_DISABLE_ATTN"] = "1"
        return resources, env_vars, init_kwargs

    def __init__(
        self,
        config: DictConfig,
        role: str,
        psrl_config: DictConfig,
        gen_interface: GenInterface,
        nixl_interface: NIXLInterface,
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
        self.instance_dist_group = None
        
        if self.psrl_config.redundant_rollout.enable:
            self.avg_max_active_tasks_len = self.psrl_config.redundant_rollout.redundant_global_batch_size * self.psrl_config.redundant_rollout.redundant_rollout_n // self.psrl_config.deployment.n_rollout_instances
        else:
            self.avg_max_active_tasks_len = self.psrl_config.staleness_buffer_entries * self.psrl_config.rollout_n // self.psrl_config.deployment.n_rollout_instances
        self.log_active_tasks_interval = self.avg_max_active_tasks_len // 8

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
        self._is_init_model = asyncio.Event()
        self._is_init_nixl_client = asyncio.Event()
        self._async_interrupt_event = asyncio.Event()
        self._async_resume_event = asyncio.Event()
        
        # Task for version update
        self.version_update_task = None
        
        # Rollout request management
        self.request_queue = deque()
        if self.psrl_config.gen_mode == "batch":
            self.request_num_queue = queue.Queue()
        else:
            assert self.config.rollout.mode == "psrl_async", \
                "Only support psrl_async mode for stream generation, please set rollout.mode to psrl_async in the config."

        # Request management
        self.version_to_task_num: dict[int, int] = {}
        self.request_id_to_active_tasks: dict[int, set[asyncio.Task]] = defaultdict(lambda: set())
        self.pending_version_requests: dict[int, List[DataProto]] = defaultdict(lambda: [])
        
        # Version ordering management
        self.version_ready_events: dict[int, asyncio.Event] = {}  # Events for when a version is ready to execute
        
        # NIXL
        self.nixl_storage_client = None
        self.unified_state_dict = None
        self.unified_sharding_dict = None
        
        # NIXL cache
        self._cached_ps_nixl_agent_names = None
        self._cached_ps_nixl_gen_storage_client_names = None

        # For async model pulling
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        
        # Build logger
        # Only the representative rank will build the logger
        if self.is_instance_representative_rank:
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

    async def estimate_max_model_len(self):
        """
        Estimate the max model length for the rollout instance.
        """
        await self._is_init_model.wait()
        assert self.rollout, "Rollout must be initialized before calling estimate_max_model_len."
        if self.config.rollout.mode == "sync":
            max_model_len = self.rollout.inference_engine.collective_rpc(
                "estimate_max_model_len", 
                args=(),
            )
        elif self.config.rollout.mode == "psrl_async":
            max_model_len = await self.rollout.inference_engine.collective_rpc(
                "estimate_max_model_len", 
                args=(),
            )
        else:
            raise ValueError(f"Invalid rollout mode: {self.config.rollout.mode}")
        return max_model_len

    async def init_nixl_client(self):
        """
        Initialize the NIXL client.
        This is implemented via rpc call in the vLLM extension.
        """
        await self._is_init_model.wait()
        assert self.rollout, "Rollout must be initialized before calling init_nixl_client."
        psrl_logger.info(f"NIXL client initialization begin via rpc call.")
        if self.psrl_config.nixl.server_mode == "storage_server":
            raise ValueError("Storage server mode is deprecated.")
        elif self.psrl_config.nixl.server_mode == "meta_server":
            if self.config.rollout.mode == "sync":
                self.rollout.inference_engine.collective_rpc(
                    "init_nixl_client", 
                    args=(self.psrl_config.nixl, self.nixl_interface, self.get_instance_id(), self.psrl_config.logging_path),
                )
            elif self.config.rollout.mode == "psrl_async":
                await self.rollout.inference_engine.collective_rpc(
                    "init_nixl_client", 
                    args=(self.psrl_config.nixl, self.nixl_interface, self.get_instance_id(), self.psrl_config.logging_path),
                )
            else:
                raise ValueError(f"Invalid rollout mode: {self.config.rollout.mode}")
        else:
            raise ValueError(f"Invalid NIXL server mode: {self.psrl_config.nixl.server_mode}")
        self._is_init_nixl_client.set()
        psrl_logger.info(f"NIXL client initialized via rpc call.")
        
    async def nixl_protocol(self):
        """
        Register the state dict and sharding dict to the NIXL client.
        This is implemented via rpc call in the vLLM extension.
        """
        await self._is_init_model.wait()
        await self._is_init_nixl_client.wait()
        assert self.rollout, "Rollout must be initialized before calling nixl_protocol."
        psrl_logger.info(f"NIXL protocol begin via rpc call.")
        if self.config.rollout.mode == "sync":
            self.rollout.inference_engine.collective_rpc(
                "nixl_protocol", 
                args=(self.config,),
            )
        elif self.config.rollout.mode == "psrl_async":
            await self.rollout.inference_engine.collective_rpc(
                "nixl_protocol", 
                args=(self.config,),
            )
        else:
            raise ValueError(f"Invalid rollout mode: {self.config.rollout.mode}")
        psrl_logger.info(f"NIXL protocol done via rpc call.")

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
    
    def get_instance_local_tp_rank(self) -> int:
        """
        Get the local tp rank of the rollout instance.
        """
        tp_rank = self.rank % self.config.rollout.get("tensor_model_parallel_size", 1)
        if self.config.rollout.mode == "sync" and self.rollout:
            from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
            vllm_tp_rank = get_tensor_model_parallel_rank()
            assert tp_rank == vllm_tp_rank, "The tp rank of the rollout instance is not consistent with the vllm tp rank."
        return tp_rank
    
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
    
    @deprecated("Use _broadcast_val_from_representative_rank instead")
    def _broadcast_int_val_from_representative_rank(self, val: Optional[int] = None) -> None:
        # NOTE(lhy): This is a temporary solution to broadcast the int value from the representative rank to all instance ranks
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

    async def register_rollout_instance(self):
        """Register the rollout instance in the PS worker."""
        assert self.rollout, "Rollout must be initialized before calling register_rollout_instance."
        if hasattr(self, "_is_rollout_instance_registered"):
            return
        if self.is_instance_representative_rank:
            # Only the representative rank needs to register the rollout instance
            await self.gen_interface.ps_manager_handle.register_rollout_instance.remote(self.get_instance_id())
        self._is_rollout_instance_registered = True
        
    def _broadcast_val_from_representative_rank(self, val: Optional[Any] = None) -> Any:
        # Use torch.distributed.broadcast_object_list for generic object broadcasting
        if self.instance_dist_group is None:
            # Create the instance distribution group if it doesn't exist
            self.instance_dist_group = dist.new_group(ranks=self.get_instance_ranks())
        # Create object list for broadcasting
        obj_list = [val]
        if self.is_instance_representative_rank:
            # Current rank is the representative rank, broadcast object to all instance ranks
            dist.broadcast_object_list(
                obj_list,
                src=self.get_instance_representative_rank(),
                group=self.instance_dist_group
            )
            return val
        else:
            # Current rank is not the representative rank, receive object from representative rank
            dist.broadcast_object_list(
                obj_list,
                src=self.get_instance_representative_rank(),
                group=self.instance_dist_group
            )
            return obj_list[0]
        
    def _build_rollout(self, trust_remote_code=False):
        """
        Build the rollout engine and sharding manager for the PSRL GenWorker.
        NOTE: This method only supports building for one rollout instance at a time.
        """
        rollout_name = self.config.rollout.name
        assert rollout_name == "vllm", "Only support vLLM rollout for now"
        try:
            rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)
            model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model, dataclass_type=HFModelConfig)
        except Exception as e:
            psrl_logger.error(f"Failed to parse rollout config or model config: {e}")
            raise
        self.model_config = model_config
        tp = self.config.rollout.get("tensor_model_parallel_size", 1)
        pp = self.config.rollout.get("pipeline_model_parallel_size", 1)
        assert self.world_size == tp * pp, "Only support dp=1 for now"

        if self.config.rollout.mode == "sync":
            self._build_distributed()
        elif self.config.rollout.mode == "psrl_async":
            # NOTE(lhy): No need to build distributed for psrl_async mode
            # vllm will handle the distributed communication internally
            pass
        else:
            raise ValueError(f"Invalid rollout mode: {self.config.rollout.mode}")
        
        # Build the rollout engine
        local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.get("use_shm", False))
        # Get the tokenizer
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)
        
        # Get the model config
        self.model_hf_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code, attn_implementation="flash_attention_2")
        # patch for kimi-vl
        if getattr(self.model_hf_config, "model_type", None) == "kimi_vl":
            self.model_hf_config.text_config.topk_method = "greedy"
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))
        if isinstance(override_model_config, dict) and "model_config" in override_model_config:
            override_config_kwargs.update(override_model_config["model_config"]) # Megatron model style
        else:
            override_config_kwargs.update(override_model_config) # FSDP model style
        update_model_config(self.model_hf_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            psrl_logger.info(f"Model config after override: {self.model_hf_config}")

        get_torch_device().manual_seed(self.seed)

        psrl_logger.info(f"Building {rollout_name} rollout with seed {self.seed}.")
        rollout = PSRL_vLLMRollout(
            psrl_config=self.psrl_config,
            config=rollout_config, 
            model_config=model_config,
            seed=self.seed, 
            status_queue=self.gen_interface.status_queue, 
            instance_id=self.get_instance_id(),
            nixl_interface=self.nixl_interface,
        )  
        
        return rollout
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def init_model(self):
        with log_dual_events("Initialize model", psrl_logger, event_type=EventType.INIT):
            self.rollout = self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))
        self._is_init_model.set()
        
    def get_active_task_num(self) -> int:
        """
        Get the number of active tasks.
        """
        return len(self.active_tasks)
        
    def log_active_tasks(self, task_added: bool = False, task_done: bool = False):
        """
        Log the active tasks.
        """
        assert task_added ^ task_done, "Exactly one of task_added or task_done must be True"
        active_task_num = self.get_active_task_num()
        psrl_logger.debug(f"Active tasks: {active_task_num}")
        if task_added and active_task_num == 1:
            self.active_tasks_start_time = time.time()
            log_begin_event(f"Streaming generate with model version {self.curr_rollout_instance_model_version}", psrl_logger, event_type=EventType.GEN)
        if task_done and active_task_num == 0:
            duration = time.time() - self.active_tasks_start_time
            log_end_event(f"Streaming generate with model version {self.curr_rollout_instance_model_version}", psrl_logger, event_type=EventType.GEN, duration=duration)
        if active_task_num % self.log_active_tasks_interval == 0:
            log_single_event(f"Active tasks: {active_task_num} ({active_task_num / self.avg_max_active_tasks_len * 100:.2f}%)", psrl_logger, event_type=EventType.OTHER)
    
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
            if not self.psrl_config.profile.fix_weight:
                model.load_weights(((name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param.to(device, non_blocking=True)) for name, param in model_state_dict_cpu.items()))
            # NOTE(lhy): Do we need to clear the cache after loading the model?
            # get_torch_device().empty_cache()
            # torch.cuda.synchronize()
        else:
            raise NotImplementedError(f"PSRL GenWorker does not support PS mode '{self.psrl_config.ps_mode}' yet.")
    
    async def ray_pull_model_async(self) -> None:
        """
        Pull the model state dict from PS via CPU and update the rollout model weights.
        In 'cpu' mode, pull the full state dict (potential bottleneck for large models).
        In 'cpu_ref' mode, get the ray object_ref and await it (parallel, non-blocking for PS worker).
        """
        assert self.config.rollout.mode == "psrl_async", "Only support psrl_async mode for async pull model."
        ps_manager_handle = self.gen_interface.ps_manager_handle
        
        if self.psrl_config.ps_mode == "cpu" or self.psrl_config.ps_mode == "cpu_ref":
            if self.psrl_config.ps_mode == "cpu":
                # In 'cpu' mode, pull the full state dict (PS worker will block on transfer)
                model_state_dict_cpu = await ps_manager_handle.pull_model_state_dict_cpu.remote(self.get_instance_id())
            elif self.psrl_config.ps_mode == "cpu_ref":
                # In 'cpu_ref' mode, get the object_ref and await it (PS worker is non-blocking)
                object_ref = await ps_manager_handle.pull_model_state_dict_cpu_ref.remote(self.get_instance_id())
                model_state_dict_cpu = await object_ref  # This blocks until the state dict is available in the object store
            # Load the model state dict to the vllm model
            # sharding will be handled automatically inside vllm
            # NOTE(linsh): transfer from CPU to GPU is handled inside vLLM extension function `load_weights`.
            params_to_load = [(name, reduce_tensor(param.full_tensor()) if isinstance(param, DTensor) else reduce_tensor(param)) for name, param in model_state_dict_cpu.items()]
            if not self.psrl_config.profile.fix_weight:
                loaded_params = await self.rollout.inference_engine.collective_rpc(
                    "load_weights",
                    args=(params_to_load,),
                )
                if loaded_params is None:
                    psrl_logger.error(f"Worker failed to update weights. Result: {loaded_params}")
                    raise
        else:
            raise NotImplementedError(f"PSRL GenWorker does not support PS mode '{self.psrl_config.ps_mode}' yet.")
    
    def nixl_pull_model(self) -> None:
        """
        Pull the model state dict from PS via NIXL and update the rollout model weights.
        This is implemented via rpc call in the vLLM extension.
        """
        assert self.psrl_config.ps_mode == "nixl_cpu" or self.psrl_config.ps_mode == "nixl_gpu", "pull_model_state_dict_nixl should only be used in 'nixl_cpu' or 'nixl_gpu' mode."
        ps_manager_handle = self.gen_interface.ps_manager_handle
        if self._cached_ps_nixl_agent_names is None:
            self._cached_ps_nixl_agent_names = ray.get(ps_manager_handle.get_ps_nixl_agent_names.remote())
        if self._cached_ps_nixl_gen_storage_client_names is None:
            self._cached_ps_nixl_gen_storage_client_names = ray.get(ps_manager_handle.get_ps_nixl_gen_storage_client_names.remote())
        if not self.psrl_config.profile.fix_weight:
            self.rollout.inference_engine.collective_rpc(
                "nixl_pull_model_core", 
                args=(self._cached_ps_nixl_agent_names, self._cached_ps_nixl_gen_storage_client_names)
            )
        ray.get(ps_manager_handle.pull_model_state_dict_nixl.remote(self.get_instance_id())) # This only updates the model version
        psrl_logger.info(f"NIXL pull model done.")
       
    async def nixl_pull_model_async(self) -> None:
        """
        Pull the model state dict from PS via NIXL and update the rollout model weights.
        This is implemented via rpc call in the vLLM extension.
        """
        assert self.psrl_config.ps_mode == "nixl_cpu" or self.psrl_config.ps_mode == "nixl_gpu", "pull_model_state_dict_nixl should only be used in 'nixl_cpu' or 'nixl_gpu' mode."
        ps_manager_handle = self.gen_interface.ps_manager_handle
        if self._cached_ps_nixl_agent_names is None:
            self._cached_ps_nixl_agent_names = await ps_manager_handle.get_ps_nixl_agent_names.remote()
        if self._cached_ps_nixl_gen_storage_client_names is None:
            self._cached_ps_nixl_gen_storage_client_names = await ps_manager_handle.get_ps_nixl_gen_storage_client_names.remote()
        if not self.psrl_config.profile.fix_weight:
            await self.rollout.inference_engine.collective_rpc(
                "nixl_pull_model_core", 
                args=(self._cached_ps_nixl_agent_names, self._cached_ps_nixl_gen_storage_client_names)
            )
        await ps_manager_handle.pull_model_state_dict_nixl.remote(self.get_instance_id()) # This only updates the model version
        psrl_logger.info(f"NIXL pull model done.")
        
    def pull_model(self) -> None:
        assert self.config.rollout.mode == "sync", "Only support `sync` mode."
        assert len(self.active_tasks) == 0, f"Cannot pull model while there are {len(self.active_tasks)} active tasks."

        if self.psrl_config.ps_mode == "cpu" or self.psrl_config.ps_mode == "cpu_ref":
            self.ray_pull_model()
        elif self.psrl_config.ps_mode == "nixl_cpu" or self.psrl_config.ps_mode == "nixl_gpu":
            self.nixl_pull_model()
        else:
            raise NotImplementedError(f"PSRL GenWorker does not support PS mode '{self.psrl_config.ps_mode}' yet.")
        # Important: the prefix cache needs to be cleared after pulling the model
        self.rollout.inference_engine.reset_prefix_cache()
        
    async def pull_model_async(self) -> None:
        assert self.config.rollout.mode == "psrl_async", "Only support `psrl_async` mode."
        assert len(self.active_tasks) == 0, f"Cannot pull model while there are {len(self.active_tasks)} active tasks"

        if self.psrl_config.ps_mode == "cpu" or self.psrl_config.ps_mode == "cpu_ref":
            await self.ray_pull_model_async()
        elif self.psrl_config.ps_mode == "nixl_cpu" or self.psrl_config.ps_mode == "nixl_gpu":
            await self.nixl_pull_model_async()
        else:
            raise NotImplementedError(f"PSRL GenWorker does not support PS mode '{self.psrl_config.ps_mode}' yet.")
        # Important: the prefix cache needs to be cleared after pulling the model
        await self.rollout.inference_engine.reset_prefix_cache()
   
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
            psrl_logger.debug(f"Interrupted requests with IDs: {request_ids}")
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
        interrupted_request_num = await self.interrupt_all_requests()
        
        # Wait and clean all tasks in self.active_tasks
        await asyncio.gather(*self.active_tasks, return_exceptions=True)
        self.active_tasks.clear()

        return interrupted_request_num

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def resume_generation(self):
        """Resume the generation process."""
        if self._generate_thread and self._generate_thread.is_alive():
            self._generate_loop.call_soon_threadsafe(self._async_resume_event.set)
            self._generate_loop.call_soon_threadsafe(self._async_interrupt_event.clear)
        else:
            self._async_resume_event.set()
            self._async_interrupt_event.clear()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    async def sync_with_ps(self, ps_version: int, interrupt_generation: bool = False) -> int:
        """
        Synchronize the rollout instance with the parameter server.
        
        This method combines three operations into one:
        1. (Optional) Interrupt the current generation process
        2. Pull the latest model weights from the parameter server
        3. Resume the generation process
        
        Returns:
            int: The number of requests that were interrupted during the sync process.
        """
        if self.curr_rollout_instance_model_version >= ps_version:
            psrl_logger.warning(f"No need to sync with PS for instance {self.get_instance_id()}, current model version {self.curr_rollout_instance_model_version} is greater than or equal to the required PS version {ps_version}")
            return 0

        # Step 1: Interrupt generation
        if interrupt_generation:
            psrl_logger.info(f"Interrupting generation on instance {self.get_instance_id()}")
            interrupted_request_num = await self.interrupt_generation()
            psrl_logger.info(f"Interrupted {interrupted_request_num} requests on instance {self.get_instance_id()}")
        else:
            assert len(self.active_tasks) == 0, "Should not have any active tasks when syncing with PS, please call `self.interrupt_generation()` in advance or set `interrupt_generation` to False"
        
        # Step 2: Pull model
        with log_dual_events("Pull model (partial rollout)", psrl_logger, event_type=EventType.PULL):
            if self.config.rollout.mode == "psrl_async":
                await self.pull_model_async()
            else:
                self.pull_model()
                
        # NOTE(lhy): The version obtained from the PS manager is the actual model version after the pull
        # It may be higher than the required version due to the pushing happens between waiting and pulling
        self.curr_rollout_instance_model_version = await self.gen_interface.ps_manager_handle.get_rollout_instance_model_version.remote(self.get_instance_id())
        assert self.curr_rollout_instance_model_version >= ps_version, \
            f"Current rollout instance model version should not be less than the required PS version, but got {self.curr_rollout_instance_model_version} vs. {ps_version}"
        if self.curr_rollout_instance_model_version > ps_version:
            psrl_logger.warning(f"Actual model version after pull (partial rollout) is {self.curr_rollout_instance_model_version}, which is higher than the required PS version {ps_version}")
        if self.rollout.stat_collector is not None:
            self.rollout.stat_collector.record_model_version_update(self.curr_rollout_instance_model_version)
        
        # Step 3: Resume generation
        psrl_logger.info(f"Resuming generation on instance {self.get_instance_id()}")
        self.resume_generation()
        psrl_logger.info(f"Generation resumed on instance {self.get_instance_id()}")

    def _create_task_done_callback(self, request_id: int, require_version: int):
        # Remove from the active tasks tracker when the task is done
        def task_done_callback(task):
            self.request_id_to_active_tasks[request_id].discard(task)
            self.active_tasks.discard(task)
            self.log_active_tasks(task_done=True)
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
        assert self.curr_rollout_instance_model_version >= needed_model_version, \
            f"Rollout model version should not be less than needed version, but got {self.curr_rollout_instance_model_version} for needed {needed_model_version}"
        
        # Update the request status to ROLLOUT_RUNNING
        request_ids = request.non_tensor_batch.get("uid", None)
        rollout_instance_id = self.get_instance_id()
        
        # Only update the model version if the request is prompt-only
        if "raw_response_ids" in request.non_tensor_batch:
            # Indicate it is a partial rollout request, use the original version tag in the request
            model_version = request.non_tensor_batch.get("version_tag", None)[0]
        else:
            model_version = self.curr_rollout_instance_model_version
            if needed_model_version != model_version:
                psrl_logger.warning(f"Update version_tag of request {request.non_tensor_batch['uid'][0]} "
                                    f"from {needed_model_version} to {model_version} due to inconsistent model pull")
                # Update version tag in staleness inventory
                await self.gen_interface.ps_manager_handle.update_request_version_tag.remote(request_ids[0], model_version)
            request.non_tensor_batch["version_tag"] = np.array([model_version], dtype=int)

        update_status_success = await self.gen_interface.ps_manager_handle.update_request_status.remote(
            request_ids.tolist(),
            PSRL_RequestStatus.ROLLOUT_RUNNING,
            rollout_instance_id=rollout_instance_id,
            model_version=model_version,
        )
        if update_status_success[0]:
            # Prepare the request for generation
            meta_info = {
                "eos_token_id": self.generation_config.eos_token_id if self.generation_config is not None else self.tokenizer.eos_token_id,
                "pad_token_id": self.generation_config.pad_token_id if self.generation_config is not None else self.tokenizer.pad_token_id,
            }
            request.meta_info.update(meta_info)
            request.non_tensor_batch["rollout_instance_id"] = np.array([rollout_instance_id] * len(request.batch))
        
            # Start the generation
            with log_dual_events(f"Core generation with model version {self.curr_rollout_instance_model_version}", psrl_logger, level=logging.DEBUG, event_type=EventType.GEN):
                vllm_outputs = await self.rollout.raw_generate_sequences_async(request)

            vllm_output = vllm_outputs[0][1] if isinstance(vllm_outputs, list) else vllm_outputs[1]
            assert len(vllm_output.outputs) == 1, \
                f"Expected no repeat in generation, got {len(vllm_output.outputs)} outputs."

            result = self.rollout.post_process_outputs(request, vllm_output)

            interrupted = result.non_tensor_batch["interrupted"][0]
            interrupted_by_scheduler = result.non_tensor_batch["interrupted_by_scheduler"][0]
            
            # Update the request status to ROLLOUT_INTERRUPTED_BY_SCHEDULER or ROLLOUT_INTERRUPTED or RUNNING,
            if interrupted_by_scheduler:
                update_status = PSRL_RequestStatus.ROLLOUT_INTERRUPTED_BY_SCHEDULER
            elif interrupted:
                update_status = PSRL_RequestStatus.ROLLOUT_INTERRUPTED
            else:
                update_status = PSRL_RequestStatus.RUNNING
            update_status_success = await self.gen_interface.ps_manager_handle.update_request_status.remote(
                request_ids.tolist(),
                update_status,
            )
            if update_status_success[0]:
                return result, update_status
        return None, None

    def generate(self, requests: DataProto, consolidate: bool = True, return_only_on_representative_rank: bool = True):
        """
        Generate sequences in batch mode.
        This method handles a batch of generation requests, managing model versioning
        and ensuring that the requests are processed in a timely manner.
        
        Args:
            requests (DataProto): The batch of generation requests.
        
        Returns:
            tuple: A tuple containing the generated sequences and their corresponding update statuses.
        """
        rollout_instance_id = self.get_instance_id()

        curr_rollout_instance_model_version = self.curr_rollout_instance_model_version
        request_ids = requests.non_tensor_batch["uid"]
        psrl_logger.debug(f"Rollout instance {rollout_instance_id} is generating requests with request ids: {request_ids}")
        needed_model_versions = requests.non_tensor_batch["version_tag"]
        # assert all version tag in needed_model_versions are equal
        if not all(version == needed_model_versions[0] for version in needed_model_versions):
            raise ValueError(f"All version tags must be the same, got {needed_model_versions}")
        needed_model_version = needed_model_versions[0]

        if needed_model_version >= curr_rollout_instance_model_version:
            if needed_model_version > curr_rollout_instance_model_version:

                with log_dual_events(f"Wait for model version {needed_model_version}", psrl_logger, event_type=EventType.WAIT):
                    # Busy polling until the PS worker has the needed model version
                    while ray.get(self.gen_interface.ps_manager_handle.get_ps_model_version.remote(debug_info="gen_worker")) < needed_model_version:
                        time.sleep(1)

                with log_dual_events("Pull model", psrl_logger, event_type=EventType.PULL):
                    self.pull_model()

                # NOTE(lhy): The version obtained from the PS manager is the actual model version after the pull
                # It may be higher than the required version due to the pushing happens between waiting and pulling
                self.curr_rollout_instance_model_version = ray.get(self.gen_interface.ps_manager_handle.get_rollout_instance_model_version.remote(rollout_instance_id))
                psrl_logger.debug(f"Updated current rollout instance model version to {self.curr_rollout_instance_model_version}")
                assert self.curr_rollout_instance_model_version >= needed_model_version, \
                    f"Current rollout instance model version should not be less than needed version, but got {self.curr_rollout_instance_model_version} vs. {needed_model_version}"
                if self.curr_rollout_instance_model_version > needed_model_version:
                    psrl_logger.warning(f"Actual model version for generation is {self.curr_rollout_instance_model_version}, needed model version is {needed_model_version}")
                if self.rollout.stat_collector is not None:
                    self.rollout.stat_collector.record_model_version_update(self.curr_rollout_instance_model_version)

            update_status_success = ray.get(self.gen_interface.ps_manager_handle.update_request_status.remote(
                request_ids.tolist(),
                PSRL_RequestStatus.ROLLOUT_RUNNING,
                rollout_instance_id=rollout_instance_id,
            ))
            filtered_request_idxs = [i for i, success in enumerate(update_status_success) if success]
            if filtered_request_idxs:
                filtered_requests = requests[filtered_request_idxs]
                # Prepare the request for generation
                meta_info = {
                    "eos_token_id": self.generation_config.eos_token_id if self.generation_config is not None else self.tokenizer.eos_token_id,
                    "pad_token_id": self.generation_config.pad_token_id if self.generation_config is not None else self.tokenizer.pad_token_id,
                }
                filtered_requests.meta_info.update(meta_info)
                filtered_requests.non_tensor_batch["rollout_instance_id"] = np.array([rollout_instance_id] * len(filtered_requests.batch))

                # Start the generation
                with log_dual_events(f"Core generation with model version {self.curr_rollout_instance_model_version}", psrl_logger, event_type=EventType.GEN):
                    vllm_outputs = self.rollout.raw_generate_sequences(filtered_requests)

                # vllm_outputs = [vllm_outputs[i] for i in range(len(vllm_outputs))]

                if return_only_on_representative_rank and not self.is_instance_representative_rank:
                    return None, None
                
                # Update the request status to ROLLOUT_COMPLETED or ROLLOUT_INTERRUPTED,
                # depending on `interrupted` field in the result
                with log_dual_events(f"Update request status", psrl_logger, event_type=EventType.OTHER):
                    update_statuses = [PSRL_RequestStatus.ROLLOUT_INTERRUPTED if vllm_outputs[i].outputs[0].finish_reason == "abort" else PSRL_RequestStatus.RUNNING for i in range(len(vllm_outputs))]
                    update_status_success = ray.get(self.gen_interface.ps_manager_handle.update_request_status.remote(
                        request_ids.tolist(),
                        update_statuses,
                    ))
                    filtered_request_idxs = [i for i, success in enumerate(update_status_success) if success]
                    if filtered_request_idxs:
                        filtered_requests = filtered_requests[filtered_request_idxs]
                        filtered_result = [vllm_outputs[i] for i in filtered_request_idxs]
                        filtered_update_statuses = [update_statuses[i] for i in filtered_request_idxs]
                        # NOTE(lhy): The DataProto will be huge and slow to transfer using ray, so we postprocess the data inside the vllm rollout
                        if consolidate:
                            filtered_result = self.rollout.post_process_outputs(filtered_requests, filtered_result)
                        return filtered_result, filtered_update_statuses
                    return None, None
        else:
            raise ValueError(f"Needed model version {needed_model_version} is less than current rollout instance model version {curr_rollout_instance_model_version}. This should not happen.")

    async def generate_async(self, request: DataProto, consolidate: bool = True):
        """
        Generate sequences asynchronously.
        This method handles a single async generation request, managing model versioning
        and ensuring that the request is processed in a timely manner.
        
        Args:
            request (DataProto): The async generation request.
        """
        assert consolidate, "Consolidate must be True for async generation for now. Because the postprocess is need to be done inside the vllm rollout to mark the requests that are interrupted by the scheduler."
        assert len(request) == 1, f"Expected request length to be 1, got {len(request)}"

        # Wait for resuming if the generation is interrupted
        if self._async_interrupt_event and self._async_interrupt_event.is_set():
            psrl_logger.debug(f"Generation interrupted, waiting for resume...")
            await self._async_resume_event.wait()
            psrl_logger.debug(f"Generation resumed")

        request_id = int(request.non_tensor_batch["uid"][0])
        needed_model_version = int(request.non_tensor_batch["version_tag"][0])
        
        # The router should guarantee the request is assigned to a rollout instance that can directly generate with the needed model version.
        assert needed_model_version <= self.curr_rollout_instance_model_version, \
            f"Needed model version {needed_model_version} should not be greater than current rollout instance model version {self.curr_rollout_instance_model_version}."
        
        # All the partial rollout requests (with version tag less than the current rollout instance model version) should be updated to the current rollout instance model version
        if needed_model_version < self.curr_rollout_instance_model_version:
            psrl_logger.debug(f"Request {request_id} needed model version {needed_model_version} is less than "
                              f"current rollout instance model version {self.curr_rollout_instance_model_version}, "
                              f"we'll update needed model version to {self.curr_rollout_instance_model_version}.")
            needed_model_version = self.curr_rollout_instance_model_version

        task = self._generate_loop.create_task(
            self._generate_async_task(request, needed_model_version)
        )
        task.add_done_callback(self._create_task_done_callback(
            int(request.non_tensor_batch["uid"][0]),
            needed_model_version,
        ))
        self.request_id_to_active_tasks[request_id].add(task)
        self.active_tasks.add(task)
        self.log_active_tasks(task_added=True)
        # Wait for the task to finish
        result = await task
        return result