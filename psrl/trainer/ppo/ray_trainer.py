import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import ray
import requests
import torch
from omegaconf import OmegaConf, open_dict
from ray.exceptions import RayTaskError
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from tqdm import tqdm
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls_fused
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.utils import WorkerType
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.seqlen_balancing import (
    get_seqlen_balanced_partitions,
    log_seqlen_unbalance,
)
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.tracking import ValidationGenerationsLogger

from psrl.trainer.ppo.utils import (
    PSRL_compute_advantage,
    PSRL_Role,
    ResourcePoolManager,
    apply_kl_penalty,
    compute_response_mask,
    need_critic,
    need_reference_policy,
    need_reward_model,
)
from psrl.utils.dataset import DataProcessor, DatasetType
from psrl.utils.logger import (
    DualOutputHandler,
    EventType,
    log_data_protocol,
    log_dual_events,
)
from psrl.utils.nixl import (
    GLOBAL_GEN_CLIENT_NAME,
    GLOBAL_META_SERVER_NAME,
    GLOBAL_PS_CLIENT_NAME,
    GLOBAL_TRAIN_CLIENT_NAME,
)
from psrl.workers.agent_loop import PSRL_AgentLoopManager, PSRL_AgentLoopWorker
from psrl.workers.agent_loop.prometheus_utils import update_prometheus_config
from psrl.workers.agent_loop.router import RolloutRouter
from psrl.workers.gen_dplb.rollout_coordinator import RolloutCoordinator
from psrl.workers.gen_dplb.rollout_gateway import RolloutGateway
from psrl.workers.gen_dplb.vllm_async_server import GenInterface, PSRL_vLLMReplica
from psrl.workers.ps import (
    PSClassWithInitArgs,
    PSManager,
    PSResourcePool,
    PSResourceSpec,
    PSStoragePlan,
    PSStorageWorker,
    PSWorkerGroup,
)
from psrl.workers.reward.reward_manager import RewardManager
from psrl.workers.train import TrainInterface

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class PSRL_RayPPOTrainer:
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[PSRL_Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        collate_fn=None,
        group_post_process_fn=None,
        buffer_post_process_fn=None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process and is responsible for managing the training process.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[PSRL_Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resources.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data.
            reward_fn: Function to compute rewards for the training data.
            val_reward_fn: Function to compute rewards for the validation data.
            collate_fn: Optional function to collate data into batches.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.collate_fn = collate_fn
        self.group_post_process_fn = group_post_process_fn
        self.buffer_post_process_fn = buffer_post_process_fn

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls  # NOTE(lhy): ray_worker_group_cls is used only in train side
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.train_actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.train_actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.train_actor_rollout_ref.model.get("lora_adapter_path") is not None

        # CPU workers for Streaming Rollout
        self.data_processor = None
        self.agent_loop_manager = None
        self.rollout_coordinator = None
        self.reward_manager = None

        self.rollout_replicas = []
        self.tag_to_server_handles = {}
        self.tag_to_base_worker_ids = {}
        self.server_addresses = []

        # Rollout gateway handle
        self.rollout_gateway = None

        # HTTP session for rollout gateway control
        self._gateway_http_session = requests.Session()
        configured_n_rollout_instances = self.config.psrl.deployment.n_rollout_instances
        configured_n_validate_instances = (
            self.config.psrl.deployment.n_validate_instances if self.config.psrl.colocate_validate_and_train else 0
        )
        gateway_pool_size = max(configured_n_rollout_instances + configured_n_validate_instances, 32)
        gateway_adapter = requests.adapters.HTTPAdapter(
            pool_connections=gateway_pool_size,
            pool_maxsize=gateway_pool_size,
            max_retries=0,
        )
        self._gateway_http_session.mount("http://", gateway_adapter)
        self._gateway_http_session.mount("https://", gateway_adapter)
        self._gateway_http_timeout = None

        # Parameter server handle for other workers to access
        self.ps_manager_handle = None
        self.ps_manager_grpc_port = None

        # Async rollout mode for training worker
        self.async_rollout_mode = False

        # Indicate whether current mode is rollout mode in actor
        self.is_rollout_mode_in_actor = (
            self.config.psrl.colocate_validate_and_train and self.config.trainer.val_before_train
        )
        psrl_logger.info(
            f"Initializing PSRL_RayPPOTrainer with is_rollout_mode_in_actor: {self.is_rollout_mode_in_actor}"
        )

        # Mappings from worker to node id and ps index for NIXL
        self.worker_to_node_id = {}
        self.worker_to_ps_idx = {}

        self.n_rollout_instances = self.config.psrl.deployment.n_rollout_instances
        self.n_validate_instances = (
            self.config.psrl.deployment.n_validate_instances if self.config.psrl.colocate_validate_and_train else 0
        )

        if self.config.psrl.redundant_rollout.enable:
            self.max_concurrency = (
                self.config.psrl.redundant_rollout.redundant_rollout_n
                * self.config.psrl.redundant_rollout.redundant_global_batch_size
                * (self.config.psrl.staleness + 1)
            )
        else:
            self.max_concurrency = (
                self.rollout_n  # instance variable set earlier in __init__
                * self.config.psrl.staleness_buffer_entries
                * (self.config.psrl.staleness + 1)
            )

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.train_actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        # Build logger
        self.log_prefix = "MainRayTrainer"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))
        psrl_logger.info("Initialized major ray trainer (single controller).")

        self._initialize_queue_buffers()

        self._validate_config()

        self._init_ps_manager()

        # initialize data processor
        # NOTE(lhy): data processor must be initialized before initializing other workers
        # so that the total_training_steps can be obtained and the optimizer config
        # (related to weight decay, lr schedule, etc.) can be set
        # otherwise, it will cause error when running Megatron backend
        self._init_data_processor()

    def _initialize_queue_buffers(self):
        if self.config.psrl.redundant_rollout.enable:
            self.rollout_n = self.config.psrl.redundant_rollout.redundant_rollout_n
            self.alg_rollout_n = self.config.psrl.redundant_rollout.alg_rollout_n
        else:
            self.rollout_n = self.config.gen_actor_rollout_ref.rollout.n
            self.alg_rollout_n = self.rollout_n
        assert self.rollout_n >= self.alg_rollout_n, (
            f"Rollout n {self.rollout_n} must be greater than or equal to alg_rollout_n {self.alg_rollout_n}."
        )

        # Data queue is the communication handle between the data processor and the rollout server.
        # The size of the queue is determined by the batch size and the rollout n.
        self.data_queue_size = (
            self.config.data.get("gen_batch_size", self.config.data.train_batch_size) * self.rollout_n
        )

        psrl_logger.debug(
            "Initialized data_queue with sizes: %d.",
            self.data_queue_size,
        )

    def _validate_config(self):
        config = self.config
        # number of GPUs used in training
        train_n_gpus = config.psrl.deployment.train_ngpus_per_node * config.psrl.deployment.train_nnodes
        if config.train_actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.train_actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.train_actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            context_parallel_size = config.train_actor_rollout_ref.actor.megatron.context_parallel_size
            assert train_n_gpus % (model_parallel_size * context_parallel_size) == 0, (
                f"train_n_gpus ({train_n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times"
                f" context_parallel_size ({context_parallel_size})"
            )
            megatron_dp = train_n_gpus // (model_parallel_size * context_parallel_size)
            minimal_bsz = megatron_dp * config.train_actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = train_n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.train_actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            """Validate mutually exclusive micro batch size configuration options.

            Ensures that users don't set both deprecated micro_batch_size and
            the new micro_batch_size_per_gpu parameters simultaneously.

            Args:
                mbs: Deprecated micro batch size parameter value.
                mbs_per_gpu: New micro batch size per GPU parameter value.
                name (str): Configuration section name for error messages.

            Raises:
                ValueError: If both parameters are set or neither is set.
            """
            settings = {
                "train_actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "train_actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "train_actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                        f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        if not config.train_actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.train_actor_rollout_ref.actor.ppo_micro_batch_size,
                config.train_actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "train_actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.train_actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.train_actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "train_actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.train_actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.train_actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "train_actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(
                config.critic.ppo_micro_batch_size,
                config.critic.ppo_micro_batch_size_per_gpu,
                "critic",
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size,
                config.reward_model.micro_batch_size_per_gpu,
                "reward_model",
            )

        # Actor training
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.train_actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.train_actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.train_actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.train_actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert (
                    config.train_actor_rollout_ref.actor.ppo_mini_batch_size
                    % config.train_actor_rollout_ref.actor.ppo_micro_batch_size
                    == 0
                )
                assert config.train_actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= train_n_gpus

        assert config.train_actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.train_actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.train_actor_rollout_ref.actor.use_kl_loss:
            psrl_logger.info("NOTICE: You have both enabled in-reward kl and kl loss.")

        # Critic training
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= train_n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.train_actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"} and (
            config.train_actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1
            or config.train_actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1
        ):
            assert config.train_actor_rollout_ref.model.use_remove_padding, (
                "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
            )

        if self.use_critic and config.critic.strategy in {"fsdp", "fsdp2"}:
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, (
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."
                )

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )

        # Check eval config
        if config.train_actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.train_actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        # check multi_turn with tool config
        if config.train_actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.train_actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        # Check NIXL compatibility
        if self.config.psrl.ps_mode == "nixl_cpu" or self.config.psrl.ps_mode == "nixl_gpu":
            assert self.config.psrl.nixl.server_ip == self.config.psrl.ps_manager_ip, (
                "PSManager IP and NIXL server IP must be the same"
            )
            assert self.config.train_actor_rollout_ref.actor.strategy != "fsdp", (
                "FSDP1 is not supported for NIXL because it uses flat_param"
            )
            psrl_logger.info(
                f"NOTICE: NIXL is enabled. Actor strategy used is {self.config.train_actor_rollout_ref.actor.strategy}"
            )

        # Check validate mode
        if self.config.psrl.colocate_validate_and_train:
            assert self.config.psrl.tms.range == "train" or self.config.psrl.tms.range == "all", (
                "TMS range must be 'train' or 'all' when using colocate_validate_and_train"
            )
        else:
            assert self.config.psrl.fuse_rollout_with_validate, (
                "fuse_rollout_with_validate must be enabled when not colocate_validate_and_train"
            )

        # Check routing strategy
        if (
            self.config.psrl.routing_strategy.method == "request_num_balance"
            or self.config.psrl.routing_strategy.method == "throughput_balance"
        ):
            assert self.config.psrl.status_collection.enable, (
                "status collection must be enabled when using "
                "request num balance or throughput balance routing strategy"
            )

        # Check TMS configuration
        if self.config.psrl.tms.enable_cuda_graph:
            assert self.config.psrl.tms.range == "all", "TMS CUDA graph can only be enabled when TMS range is 'all'"
        if self.config.psrl.tms.range not in ["train", "all"]:
            assert (
                self.config.train_actor_rollout_ref.actor.strategy == "megatron"
                and self.config.train_actor_rollout_ref.actor.megatron.optimizer_offload
                or self.config.train_actor_rollout_ref.actor.strategy == "fsdp2"
                and self.config.train_actor_rollout_ref.actor.fsdp_config.optimizer_offload
            ), "Optimizer offload must be enabled when TMS is not enabled for training workers"

        psrl_logger.info("[validate_config] All configuration checks passed successfully!")

    def _init_ps_manager(self):
        """Initialize the PS manager for handling model version, requests condition and staleness."""
        # Set the validation rollout number in the config
        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "psrl"):
                    self.config.psrl.val_rollout_n = self.config.train_actor_rollout_ref.rollout.val_kwargs.n
        except Exception as e:
            psrl_logger.warning(f"Could not set val_rollout_n in config. Structure missing? Error: {e}")

        ip_to_node_id = {node["NodeManagerAddress"]: node["NodeID"] for node in ray.nodes()}
        assert self.config.psrl.ps_manager_ip in ip_to_node_id, (
            f"PSManager IP {self.config.psrl.ps_manager_ip} not found in ray nodes"
        )
        psrl_logger.info("Getting the handle of the PSManager")
        self.ps_manager_handle = (
            ray.remote(PSManager)
            .options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ip_to_node_id[self.config.psrl.ps_manager_ip], soft=False
                )
            )
            .remote(self.config.psrl)
        )
        self.ps_manager_grpc_port = ray.get(self.ps_manager_handle.start_grpc_server.remote())

    def stop_ps_manager(self):
        """Stop the PS manager."""
        if self.ps_manager_handle is not None:
            psrl_logger.info("Stopping PS manager...")
            ray.get(self.ps_manager_handle.shutdown_nixl_server.remote())
            psrl_logger.info("NIXL server stopped successfully.")
            ray.get(self.ps_manager_handle.stop_grpc_server.remote())
            self.ps_manager_handle = None
            psrl_logger.info("PS manager stopped successfully.")
        else:
            psrl_logger.warning("PS manager is not initialized, skipping stop operation.")

    def _init_data_processor(self):
        """Initialize the data processor for handling data preprocessing and batching."""
        if self.data_processor is not None:
            return

        # Initialize the data processor
        self.data_processor = DataProcessor.remote(
            self.config, self.tokenizer, self.processor, self.ps_manager_handle, collate_fn=self.collate_fn
        )

        # Get total training steps from the data processor where dataloaders are built
        self.total_training_steps = ray.get(self.data_processor.get_total_training_steps.remote())

        psrl_logger.info(f"Total training steps: {self.total_training_steps}")

        # Set the total training steps in the config
        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "train_actor_rollout_ref.actor.optim"):
                    self.config.train_actor_rollout_ref.actor.optim.total_training_steps = self.total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = self.total_training_steps
        except Exception as e:
            psrl_logger.warning(f"Could not set total_training_steps in config. Structure missing? Error: {e}")

    def start_data_processor(self):
        """Launch the data processor for processing data in the background."""
        assert self.data_processor is not None, "Data processor must be initialized before starting it."

        ray.get(self.data_processor.start_busy_loop.remote())

    def stop_data_processor(self):
        """Stop the data processor."""
        if self.data_processor is not None:
            psrl_logger.debug("Stopping data processor...")
            ray.get(self.data_processor.stop_busy_loop.remote())
            self.data_processor = None
            psrl_logger.debug("Data processor stopped successfully.")
        else:
            psrl_logger.warning("Data processor is not initialized, skipping stop operation.")

    def init_agent_loop_manager(self):
        if self.agent_loop_manager is not None:
            return

        # Initialize the agent loop manager
        self.agent_loop_manager = ray.remote(PSRL_AgentLoopManager).options(max_concurrency=self.max_concurrency).remote(
            self.config,
            self.data_queue_size,
            self.agent_loop_workers,
            self.ps_manager_handle,
            group_post_process_fn=self.group_post_process_fn,
            buffer_post_process_fn=self.buffer_post_process_fn,
        )

    def start_agent_loop_manager(self):
        """Start the agent loop manager to handle agent loops in the background."""
        assert self.agent_loop_manager is not None, "Agent loop manager must be initialized before starting it."

        ray.get(self.agent_loop_manager.start_busy_loop.remote())

    def stop_agent_loop_manager(self):
        """Stop the agent loop manager."""
        if self.agent_loop_manager is not None:
            psrl_logger.debug("Stopping agent loop manager...")
            ray.get(self.agent_loop_manager.stop_busy_loop.remote())
            self.agent_loop_manager = None
            psrl_logger.debug("Agent loop manager stopped successfully.")
        else:
            psrl_logger.warning("Agent loop manager is not initialized, skipping stop operation.")

    def init_rollout_router(self):
        if self.config.psrl.rollout_gateway.enable:
            self.rollout_router = RolloutGateway.remote(
                self.config,
                self.config.psrl.ps_manager_ip,
                self.ps_manager_grpc_port,
            )
            self.rollout_gateway_url = ray.get(self.rollout_router.launch_router.remote())
            psrl_logger.info(f"Rollout gateway launched at {self.rollout_gateway_url}")
        else:
            self.rollout_router = RolloutRouter.options(max_concurrency=self.max_concurrency).remote(
                self.config,
                self.ps_manager_handle,
                self.tokenizer,
            )
            self.rollout_gateway_url = None

    def init_rollout_coordinator(self):
        assert self.rollout_router is not None, (
            "Rollout router must be initialized before initializing rollout coordinator."
        )
        self.rollout_coordinator = RolloutCoordinator.remote(
            self.config,
            self.ps_manager_handle,
            self.rollout_gateway_url if self.config.psrl.rollout_gateway.enable else self.rollout_router,
        )

    def _post_gateway_worker_routing_control(self, action: str, instance_ids):
        assert action in {"pause", "resume"}, f"Unsupported action: {action}"
        assert self.rollout_gateway_url is not None, "rollout_gateway_url is not initialized"

        payload = []
        for instance_id in instance_ids:
            if isinstance(instance_id, tuple):
                base_worker_id, dp_rank = instance_id
                payload.append({"base_worker_id": base_worker_id, "dp_rank": dp_rank})
            else:
                base_worker_id = instance_id
                payload.append({"base_worker_id": base_worker_id})

        resp = self._gateway_http_session.post(
            f"{self.rollout_gateway_url.rstrip('/')}/workers/{action}",
            json=payload,
            timeout=self._gateway_http_timeout,
        )
        resp.raise_for_status()
        psrl_logger.info(
            f"After control action {action} over {instance_ids}, resp = {resp.json() if resp.content else {}}"
        )
        return resp.json() if resp.content else {}

    def start_rollout_coordinator(self):
        assert self.rollout_coordinator is not None, "Rollout coordinator must be initialized before starting it."

        ray.get(self.rollout_coordinator.start_busy_loop.remote())

    def stop_rollout_coordinator(self):
        """Stop the rollout coordinator."""
        if self.rollout_coordinator is not None:
            psrl_logger.debug("Stopping rollout coordinator...")
            ray.get(self.rollout_coordinator.stop_busy_loop.remote())
            self.rollout_coordinator = None
            psrl_logger.debug("Rollout coordinator stopped successfully.")
        else:
            psrl_logger.warning("Rollout coordinator is not initialized, skipping stop operation.")

    def init_reward_manager(self):
        """Initialize the reward manager for computing rewards during training."""
        assert self.data_processor is not None, (
            "Data processor must be initialized before starting reward computation."
        )
        assert self.rollout_coordinator is not None, (
            "Rollout server must be initialized before starting reward computation."
        )

        ip_to_node_id = {node["NodeManagerAddress"]: node["NodeID"] for node in ray.nodes()}
        self.reward_manager = (
            ray.remote(RewardManager)
            .options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ip_to_node_id[self.config.psrl.reward_service_ip],
                    soft=False,
                )
            )
            .remote(
                self.config,
                self.tokenizer,
                self.processor,
                self.ps_manager_handle,
            )
        )

    def start_reward_manager(self):
        """Start the reward manager to handle reward computation requests in the background."""
        assert self.reward_manager is not None, "Reward manager must be initialized before starting it."

        ray.get(self.reward_manager.start_busy_loop.remote())

    def stop_reward_manager(self):
        """Stop the reward manager."""
        if self.reward_manager is not None:
            psrl_logger.debug("Stopping reward manager...")
            ray.get(self.reward_manager.stop_busy_loop.remote())
            self.reward_manager = None
            psrl_logger.debug("Reward manager stopped successfully.")
        else:
            psrl_logger.warning("Reward manager is not initialized, skipping stop operation.")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        psrl_logger.info(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self,
        batch: DataProto,
        reward_extra_infos_dict: dict,
        timing_raw: dict,
        rollout_data_dir: str,
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        """Validate the model using the validation dataset.

        Note that we use the training side to do val for overlapping with generation.
        """
        with log_dual_events("Switch to rollout mode", psrl_logger, event_type=EventType.SWITCH):
            self.switch_to_rollout_mode()

        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_parent_ids = []

        test_batch_list = []
        batch_count = 0
        while True:
            try:
                test_data = ray.get(self.data_processor.get_single_controller_batch.remote(DatasetType.val))
                batch_count += 1
            except RayTaskError as e:
                if isinstance(e.cause, StopIteration):
                    psrl_logger.debug(
                        "Reached end of validation dataset after %d batches",
                        batch_count,
                    )
                    break
                else:
                    psrl_logger.error(f"Unknown exception happened during obtaining validation data: {type(e.cause)}")
                    raise
            test_batch = DataProto.from_single_dict(test_data)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            test_batch_list.append(test_batch)

        val_data_size = sum(len(batch.batch) for batch in test_batch_list)
        futures = []
        futures.append(self.ps_manager_handle.set_val_staleness_inventory_capacity.remote(val_data_size))
        futures.append(self.agent_loop_manager.set_val_buffer_size.remote(val_data_size))
        ray.get(futures)

        val_rollout_n = self.config.train_actor_rollout_ref.rollout.val_kwargs.n
        for test_batch in test_batch_list:
            batch_size = len(test_batch.batch)

            sample_ids = ray.get(self.data_processor.get_val_sample_ids.remote(batch_size))
            test_batch.non_tensor_batch["parent_id" if val_rollout_n > 1 else "uid"] = np.array(sample_ids)
            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=val_rollout_n, interleave=True)

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO(verl): Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_parent_ids.extend(
                test_batch.non_tensor_batch["parent_id" if val_rollout_n > 1 else "uid"]
            )

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "interaction_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("interaction_kwargs")
            if "agent_name" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("agent_name")
            non_tensor_batch_keys_to_pop.append("parent_id" if val_rollout_n > 1 else "uid")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            if val_rollout_n > 1:
                uid_list = []
                for i in range(batch_size):
                    for j in range(val_rollout_n):
                        child_id = sample_ids[i] * val_rollout_n + j
                        uid_list.append(child_id)
                test_gen_batch.non_tensor_batch["uid"] = np.array(uid_list)

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.train_actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            psrl_logger.debug(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            val_buffer_id = ray.get(self.agent_loop_manager.generate_validate_sequences.remote(test_gen_batch))
            with log_dual_events(f"Wait for validation batch {val_buffer_id}", psrl_logger, event_type=EventType.WAIT):
                test_output_gen_batch = ray.get(
                    self.agent_loop_manager.wait_for_validation_batch.remote(val_buffer_id)
                )

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(
                test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
            )

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        with log_dual_events("Switch to trainer mode", psrl_logger, event_type=EventType.SWITCH):
            self.switch_to_trainer_mode()

        return self._val_metrics_update(data_sources, sample_parent_ids, reward_extra_infos_dict, sample_turns)

    def _val_metrics_update(self, data_sources, sample_parent_ids, reward_extra_infos_dict, sample_turns):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_parent_ids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _run_all(self, tasks: list[asyncio.Task]):
        async def run_all():
            await asyncio.gather(*tasks)

        asyncio.run(run_all())

    def register_rollout_servers(self, rollout_replicas: list[PSRL_vLLMReplica]):
        futures = []
        # 1. register to rollout router
        for replica in rollout_replicas:
            if self.config.psrl.rollout_gateway.enable:
                futures.append(replica.servers[0].register_server_to_gateway.remote(self.rollout_gateway_url))
            else:
                futures.append(
                    self.rollout_router.add_worker.remote(
                        replica.servers[0],
                        str(replica.get_replica_id()),
                        replica.data_parallel_size,
                        replica.tensor_parallel_size,
                        replica.pipeline_parallel_size,
                        is_validate=False if replica.tag == "rollout" else True,
                    )
                )
        results = ray.get(futures)

        # 2. register to ps manager
        futures = []
        for replica in rollout_replicas:
            futures.append(replica.servers[0].register_rollout_instances_to_ps.remote())
        ray.get(futures)

        # 3. register to rollout coordinator
        futures = []
        for replica_id, replica in zip(results, rollout_replicas):
            futures.append(
                self.rollout_coordinator.add_worker.remote(
                    replica,
                    replica.servers[0],
                    replica_id,
                    replica.data_parallel_size,
                    is_validate=False if replica.tag == "rollout" else True,
                    model_version=0,
                )
            )
            if self.tag_to_base_worker_ids.get(replica.tag) is None:
                self.tag_to_base_worker_ids[replica.tag] = []
            self.tag_to_base_worker_ids[replica.tag].append(replica_id)
        ray.get(futures)

    def init_rollout_servers(
        self,
        worker_group_list: list[RayWorkerGroup],
        tag: str = "rollout",
    ):
        status_sink_endpoint = ray.get(self.rollout_coordinator.get_status_sink_endpoint.remote())
        if tag == "validate":
            rollout_world_size = (
                self.config.train_actor_rollout_ref.rollout.tensor_model_parallel_size
                * self.config.train_actor_rollout_ref.rollout.data_parallel_size
                * self.config.train_actor_rollout_ref.rollout.pipeline_model_parallel_size
            )
            rollout_config = self.config.train_actor_rollout_ref.rollout
        else:
            rollout_world_size = (
                self.config.gen_actor_rollout_ref.rollout.tensor_model_parallel_size
                * self.config.gen_actor_rollout_ref.rollout.data_parallel_size
                * self.config.gen_actor_rollout_ref.rollout.pipeline_model_parallel_size
            )
            rollout_config = self.config.gen_actor_rollout_ref.rollout
        model_config = self.config.train_actor_rollout_ref.model
        rollout_replicas = []
        init_tasks = []
        for worker_group in worker_group_list:
            world_size = worker_group.world_size
            num_replicas = world_size // rollout_world_size
            psrl_logger.info(f"[{tag}]: {world_size=}, {rollout_world_size=}, {num_replicas=}")
            curr_replica_num = len(self.rollout_replicas)
            new_rollout_replicas = []
            for replica_rank in range(num_replicas):
                gen_interface = GenInterface(
                    rollout_replica_idx=curr_replica_num + replica_rank,
                    ps_manager_handle=self.ps_manager_handle,
                    status_endpoint=status_sink_endpoint,
                )
                new_rollout_replicas.append(
                    PSRL_vLLMReplica(
                        replica_rank=curr_replica_num + replica_rank,
                        local_replica_rank=replica_rank,
                        psrl_config=self.config.psrl,
                        config=rollout_config,
                        model_config=model_config,
                        gen_interface=gen_interface,
                        gpus_per_node=(
                            self.config.psrl.deployment.rollout_ngpus_per_node_per_instance
                            if tag == "rollout"
                            else self.config.psrl.deployment.validate_ngpus_per_node_per_instance
                        ),
                        tag=tag,
                    )
                )
            rollout_replicas.extend(new_rollout_replicas)
            self.rollout_replicas.extend(new_rollout_replicas)
            init_tasks.extend([replica.init_model(worker_group) for replica in new_rollout_replicas])
        self._run_all(init_tasks)
        if tag not in self.tag_to_server_handles:
            self.tag_to_server_handles[tag] = []
        self.tag_to_server_handles[tag].extend([replica._server_handle for replica in rollout_replicas])
        psrl_logger.info(f"Current server num of {tag} is {len(self.tag_to_server_handles[tag])}")
        self.server_addresses.extend([replica._server_address for replica in rollout_replicas])

        # Update Prometheus configuration with server addresses
        if rollout_config.prometheus.enable:
            if rollout_config.disable_log_stats:
                raise ValueError("PROMETHEUS needs disable_log_stats==False, but it is currently True.")
            update_prometheus_config(rollout_config.prometheus, self.server_addresses, rollout_config.name)

        self.register_rollout_servers(rollout_replicas)

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)

        Note that we use multi-threading to speed up the initialization of worker groups.
        For rollout instances, we create multiple worker groups based on
        the number of instances specified in the configuration,
        instead of creating a unified worker group for all instances.
        """

        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # ---- Step 1: Create worker cls for each role ----
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(
                        self.config.global_profiler.global_tool_config.nsys,
                        "worker_nsight_options",
                    )
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(
                        self.config.global_profiler.global_tool_config.nsys,
                        "worker_nsight_options",
                    )
                )
        wg_kwargs["device_name"] = self.device_name

        # create rollout, actor and ps
        # PS need to be created before rollout and actor to pass the ps_manager_handle
        assert PSRL_Role.Rollout in self.role_worker_mapping and PSRL_Role.Actor in self.role_worker_mapping, (
            "Rollout and Actor must be in role_worker_mapping."
        )

        # map rollout resource pool to rollout cls
        for i in range(self.n_rollout_instances):
            rollout_config = self.config.gen_actor_rollout_ref.rollout
            rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[PSRL_Role.Rollout],
                config=rollout_config,
                model_config=self.config.train_actor_rollout_ref.model,
                device_mesh=None,
            )
            rollout_resource_pool = self.resource_pool_manager.get_resource_pool(PSRL_Role.Rollout, i)
            self.resource_pool_to_cls[rollout_resource_pool][f"rollout_{i}"] = rollout_cls

        # map validate resource pool to validate cls
        for i in range(self.n_validate_instances):
            rollout_config = self.config.train_actor_rollout_ref.rollout
            val_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[PSRL_Role.Validate],
                config=rollout_config,
                model_config=self.config.train_actor_rollout_ref.model,
                device_mesh=None,
            )
            val_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(PSRL_Role.Validate, i)
            self.resource_pool_to_cls[val_rollout_resource_pool][f"validate_{i}"] = val_rollout_cls

        # map training actor resource pool to actor cls
        train_interface = TrainInterface(ps_manager_handle=self.ps_manager_handle)
        actor_resource_pool = self.resource_pool_manager.get_resource_pool(PSRL_Role.Actor)
        actor_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[PSRL_Role.Actor],
            config=self.config.train_actor_rollout_ref,
            role="actor",
            psrl_config=self.config.psrl,
            train_interface=train_interface,
        )
        self.resource_pool_to_cls[actor_resource_pool]["actor"] = actor_cls

        # map critic resource pool to critic cls if critic is used
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(PSRL_Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[PSRL_Role.Critic],
                config=self.config.critic,
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # map reference policy resource pool to reference policy cls if reference policy is used
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(PSRL_Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[PSRL_Role.RefPolicy],
                config=self.config.train_actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # map reward model resource pool to reward model cls if used
        self.rm_wg = None
        if self.use_rm:
            resource_pool = self.resource_pool_manager.get_resource_pool(PSRL_Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                self.role_worker_mapping[PSRL_Role.RewardModel],
                config=self.config.reward_model,
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        if not self.use_critic and not self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(PSRL_Role.DummyPolicy)
            dummy_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[PSRL_Role.DummyPolicy],
                config=self.config.train_actor_rollout_ref,
                role="dummy",
            )
            self.resource_pool_to_cls[resource_pool]["dummy"] = dummy_policy_cls

        # ---- Step 2: Create worker groups for each role in parallel ----

        # initialize WorkerGroup
        psrl_logger.info("Initializing WorkerGroup for other roles")

        # NOTE(verl): if you want to use a different resource pool for each role,
        # which can support different parallel size,
        # you should not use `create_colocated_worker_cls_fused`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        def create_worker_group(resource_pool, class_dict, wg_kwargs=wg_kwargs):
            # if there is only one worker class in the resource pool, we can directly create a worker group
            # so that we can use 'execute_all_async' and other low-level APIs
            # NOTE(lhy): in newest verl, we can use `create_colocated_worker_cls_fused`
            # to create a fused worker group and low-level APIs can also be used
            if len(class_dict) == 1:
                role = next(iter(class_dict.keys()))
                ray_worker_group_cls = (
                    RayWorkerGroup if "rollout" in role or "validate" in role else self.ray_worker_group_cls
                )
                return {
                    role: ray_worker_group_cls(
                        resource_pool=resource_pool,
                        ray_cls_with_init=class_dict[role],
                        **wg_kwargs,
                    )
                }
            # colocate
            else:
                worker_dict_cls = create_colocated_worker_cls_fused(class_dict=class_dict)
                wg_dict = self.ray_worker_group_cls(
                    resource_pool=resource_pool,
                    ray_cls_with_init=worker_dict_cls,
                    **wg_kwargs,
                )
                return wg_dict.spawn(prefix_set=class_dict.keys())

        def _run_worker_group_tasks(tasks, label: str):
            """Create worker groups with a thread pool; safely handle empty task lists."""

            if not tasks:
                psrl_logger.info(f"No {label} worker group to create; skipping.")
                return

            # We create one thread per task; ThreadPoolExecutor requires max_workers > 0
            with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
                futures = {}
                for resource_pool, class_dict, task_wg_kwargs in tasks:
                    future = executor.submit(create_worker_group, resource_pool, class_dict, task_wg_kwargs)
                    futures[future] = (resource_pool, class_dict)
                for future in futures:
                    result = future.result()
                    all_wg.update(result)

        # multi-thread version
        train_tasks = []
        gen_tasks = []
        val_tasks = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            psrl_logger.info(f"Creating worker group for resource pool: {resource_pool}, classes: {class_dict}")
            if "ps" in class_dict:
                assert class_dict.keys() == {"ps"}, "PS resource pool should only have PS role."
                continue
            if any("rollout" in key for key in class_dict.keys()):
                assert len(class_dict) == 1, "Rollout resource pool should only have one worker class."
                gen_tasks.append((resource_pool, class_dict, wg_kwargs))
            elif any("validate" in key for key in class_dict.keys()):
                assert len(class_dict) == 1, "Validate resource pool should only have one worker class."
                val_tasks.append((resource_pool, class_dict, wg_kwargs))
            else:
                # NOTE(linsh): adapt wg_kwargs for fused train worker
                # if want to add specific env args.
                if self.config.psrl.tms.range in ["train", "all"] or self.config.psrl.tms.enable_nixl:
                    # add tms config to train workers
                    import torch_memory_saver

                    dynlib_path = os.path.join(
                        os.path.dirname(os.path.dirname(torch_memory_saver.__file__)),
                        "torch_memory_saver_hook_mode_preload.abi3.so",
                    )
                    assert os.path.exists(dynlib_path), f"LD_PRELOAD so file {dynlib_path} does not exist."

                    train_wg_kwargs = wg_kwargs.copy()
                    train_wg_kwargs["worker_env"] = {
                        "LD_PRELOAD": dynlib_path,
                        "TMS_INIT_ENABLE": "1",
                        "TMS_INIT_ENABLE_CPU_BACKUP": "0",
                        # NOTE(linsh): torch_memory_saver is not compatible with expandable segments
                        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
                        "PSRL_TMS_ENABLE": "1" if self.config.psrl.tms.range in ["train", "all"] else "",
                    }
                else:
                    train_wg_kwargs = wg_kwargs
                    # NOTE(lhy): Still cannot use expandable segments, will cause NIXL error
                    # train_wg_kwargs["worker_env"] = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
                train_tasks.append((resource_pool, class_dict, train_wg_kwargs))
        # We must execute train tasks first because rollout instances may occupy
        # the resources randomly and no structured resources are available for training
        _run_worker_group_tasks(train_tasks, label="train")
        _run_worker_group_tasks(gen_tasks, label="gen")
        _run_worker_group_tasks(val_tasks, label="validate")

        self.rollout_wg_list = [all_wg[f"rollout_{i}"] for i in range(self.n_rollout_instances)]
        self.validate_wg_list = [all_wg[f"validate_{i}"] for i in range(self.n_validate_instances)]

        # initialize rollout router for registration
        self.init_rollout_router()

        # create agent loop workers
        rollout_router = self.rollout_gateway_url if self.config.psrl.rollout_gateway.enable else self.rollout_router
        self.agent_loop_workers = []
        max_concurrency_per_worker = self.max_concurrency // self.config.gen_actor_rollout_ref.rollout.agent.num_workers
        for i in range(self.config.gen_actor_rollout_ref.rollout.agent.num_workers):
            self.agent_loop_workers.append(
                PSRL_AgentLoopWorker.options(
                    name=f"agent_loop_worker_{i}",
                    max_concurrency=max_concurrency_per_worker,
                ).remote(
                    self.config,
                    self.ps_manager_handle,
                    rollout_router,
                )
            )

        psrl_logger.info("Initializing models and NIXL clients")

        nixl_client_futures = []
        model_init_futures = []

        # ---- Step 3: Create and init PS WorkerGroup ----
        psrl_logger.info("Create PS WorkerGroup")
        train_model_dtype = (
            torch.bfloat16 if self.config.train_actor_rollout_ref.actor.strategy == "megatron" else torch.float32
        )
        storage_plan = PSStoragePlan(
            train_model_dtype=train_model_dtype,
            gen_model_dtype=PrecisionType.to_dtype(self.config.gen_actor_rollout_ref.rollout.dtype),
        )
        if self.config.psrl.ps_mode == "cpu" or self.config.psrl.ps_mode == "cpu_ref":
            # PSManager is used to store the model state dict
            # No need to create PS WorkerGroup
            pass
        elif self.config.psrl.ps_mode == "nixl_cpu" or self.config.psrl.ps_mode == "nixl_gpu":
            # PSManager is only used to build the nixl meta server
            # The PS WorkerGroup is used to store the model state dict
            # It is colocate with the rollout instances
            assert self.config.psrl.nixl.server_ip == self.config.psrl.ps_manager_ip, (
                "PSManager IP and NIXL server IP must be the same"
            )
            if self.config.psrl.ps_mode == "nixl_cpu":
                # ps is deployed on both generation and (maybe) actor nodes
                ps_node_ids = set()

                # Get all rollout instances' distinct node ids
                # TODO(linsh): refactor naming to `W{i}_I{j}_R{k}` to make it
                # more clear about (replica i, instance j, rank k)
                for i in range(self.n_rollout_instances):
                    rollout_instance_node_ids = all_wg[f"rollout_{i}"].execute_all_sync("get_node_id")
                    for node_id in rollout_instance_node_ids:
                        ps_node_ids.add(node_id)
                    self.worker_to_node_id.update(
                        {f"rollout_I{i}_R{idx}": node_id for idx, node_id in enumerate(rollout_instance_node_ids)}
                    )

                # Get all actor instances' distinct node ids
                actor_instance_node_ids = all_wg["actor"].execute_all_sync("get_node_id")
                for node_id in actor_instance_node_ids:
                    ps_node_ids.add(node_id)
                self.worker_to_node_id.update(
                    {f"actor_R{idx}": node_id for idx, node_id in enumerate(actor_instance_node_ids)}
                )

                # Get all validate instances' distinct node ids
                for i in range(self.n_validate_instances):
                    validate_instance_node_ids = all_wg[f"validate_{i}"].execute_all_sync("get_node_id")
                    for node_id in validate_instance_node_ids:
                        ps_node_ids.add(node_id)
                    self.worker_to_node_id.update(
                        {f"validate_I{i}_R{idx}": node_id for idx, node_id in enumerate(validate_instance_node_ids)}
                    )

                ps_spec_list = []
                ps_node_ids = list(ps_node_ids)
                # Map each worker to a PS index
                for i, node_id in enumerate(ps_node_ids):
                    self.worker_to_ps_idx.update(
                        {worker: i for worker, nid in self.worker_to_node_id.items() if nid == node_id}
                    )

                for node_id in ps_node_ids:
                    ps_spec_list.append(PSResourceSpec(node_id=node_id, attached_gpu_id=None))
                ps_resource_pool = PSResourcePool(ps_spec_list=ps_spec_list)
                psrl_logger.info(f"PS resource pool: {ps_resource_pool}")
                self.ps_wg = PSWorkerGroup(
                    resource_pool=ps_resource_pool,
                    ps_cls_with_init=PSClassWithInitArgs(
                        cls=ray.remote(PSStorageWorker),
                        storage_plan=storage_plan,
                        model_config=self.config.train_actor_rollout_ref.model,
                        psrl_config=self.config.psrl,
                    ),
                )
                if self.config.psrl.ps_mode == "nixl_cpu" or self.config.psrl.ps_mode == "nixl_gpu":
                    nixl_client_futures.extend(self.ps_wg.execute_all_async("init_nixl_client"))
                # Init model skeleton on meta device; weights are loaded after NIXL protocol completes.
                model_init_futures.extend(self.ps_wg.execute_all_async("init_model"))
                psrl_logger.info("PS model initialized successfully!")
            elif self.config.psrl.ps_mode == "nixl_gpu":
                raise NotImplementedError("PS mode 'nixl_gpu' is not implemented yet")
        else:
            raise ValueError(f"Invalid PS mode: {self.config.psrl.ps_mode}")

        # ---- Step 4: Initialize models in all worker groups ----

        psrl_logger.info("Initializing models in all rollout instances")
        # start rollout coordinator
        self.init_rollout_coordinator()

        # simutaneously init all rollout instances
        self.init_rollout_servers(self.rollout_wg_list, tag="rollout")

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        if not (self.use_critic or (self.use_reference_policy and not self.ref_in_actor)):
            # NOTE(linsh): when not using critic or reference policy,
            # if we directly call `init_model` of actor_wg, Ray will view the fused worker
            # as an async actor and run `run_async_func_or_coro_in_event_loop`, which will
            # make it invalid to call async function such as `trainer_mode` in `init_model`.
            # So here we create a dummy worker group and call its dummy method to avoid this issue.
            self.dummy_wg = all_wg["dummy"]
            self.dummy_wg.init_model()

        # ---- Step 5: Initialize NIXL clients, convert parameters format and execute NIXL protocol  ---

        if self.is_rollout_mode_in_actor:
            assert self.config.psrl.ps_mode == "nixl_cpu" or self.config.psrl.ps_mode == "nixl_gpu", (
                "Fused trainer and validator only support NIXL PS mode."
            )
            # init actor wg -> offload -> init validate wg
            psrl_logger.info("Initializing actor model")
            self.actor_wg = all_wg["actor"]
            nixl_client_futures.extend(self.actor_wg.execute_all_async("init_nixl_client"))
            ray.get(nixl_client_futures)
            psrl_logger.info("Initialized NIXL client in actor worker group")
            self.actor_wg.init_model("empty")
            ray.get(self.actor_wg.execute_all_async("nixl_convert_params"))
            ray.get(self.actor_wg.execute_all_async("nixl_sleep", "meta"))

            psrl_logger.info("Initializing validation model")
            self.init_rollout_servers(self.validate_wg_list, tag="validate")
            ray.get(model_init_futures)
            ray.get(self.rollout_coordinator.init_nixl_client.remote())
            ray.get(self.rollout_coordinator.nixl_convert_params.remote())
        else:
            # init validate wg -> offload -> init actor wg
            # ray.get(model_init_futures)
            psrl_logger.info("Initializing validation model")
            self.init_rollout_servers(self.validate_wg_list, tag="validate")
            ray.get(model_init_futures)
            ray.get(self.rollout_coordinator.init_nixl_client.remote())
            ray.get(self.rollout_coordinator.nixl_convert_params.remote())
            # Pause validate instances in the router before sleeping them, so that the router
            # does not route rollout requests to validate instances that are in sleep state.
            # (fuse_rollout_with_validate=True makes all instances available by default)
            init_paused_instance_ids = list(
                range(self.n_rollout_instances, self.n_rollout_instances + self.n_validate_instances)
            )
            ray.get(
                [
                    self.rollout_coordinator.sleep.remote("validate"),
                    self.rollout_router.pause_instances.remote(init_paused_instance_ids),
                ]
            )

            psrl_logger.info("Initializing actor model")
            self.actor_wg = all_wg["actor"]
            self.actor_wg.init_model("full")
            nixl_client_futures.extend(self.actor_wg.execute_all_async("init_nixl_client"))
            ray.get(nixl_client_futures)
            ray.get(self.actor_wg.execute_all_async("nixl_convert_params"))
        psrl_logger.info("All workers' models initialized successfully!")

        # initialize NIXL
        if self.config.psrl.ps_mode == "nixl_cpu" or self.config.psrl.ps_mode == "nixl_gpu":
            rollout_world_size = ray.get(self.rollout_coordinator.world_size.remote())
            psrl_logger.info(
                f"Initializing NIXL server with {self.ps_wg.world_size} PS workers, "
                f"{self.actor_wg.world_size} actor workers, {rollout_world_size} rollout workers"
            )
            expected_nixl_client_agents = self.ps_wg.world_size + self.actor_wg.world_size + rollout_world_size
            ray.get(self.ps_manager_handle.init_nixl_server.remote(expected_nixl_client_agents))
            actor_protocol_mode = "meta" if self.is_rollout_mode_in_actor else "full"
            rollout_full_tag = "all" if self.is_rollout_mode_in_actor else "rollout"

            with log_dual_events("Executing NIXL protocol", psrl_logger, event_type=EventType.INIT):
                futures = []
                futures.append(self.ps_manager_handle.nixl_protocol.remote())
                futures.extend(self.ps_wg.execute_all_async("nixl_protocol"))
                futures.extend(self.actor_wg.execute_all_async("nixl_protocol", actor_protocol_mode))
                futures.append(self.rollout_coordinator.nixl_protocol.remote(rollout_full_tag))
                ray.get(futures)
            
            # Now that all NIXL buffers are allocated (meta tensors replaced),
            # stream checkpoint weights into the PS registered tensors.
            # TODO(lhy): change all loading logic to this, not only val_before_train
            #            (disk -> PS CPUs -> train/gen GPUs)
            if self.is_rollout_mode_in_actor:
                with log_dual_events("Loading PS checkpoint weights", psrl_logger, event_type=EventType.INIT):
                    ray.get(self.ps_wg.execute_all_async("load_weights_to_registered_tensors"))

            psrl_logger.info("Binding PS worker group")
            self.ps_manager_handle.bind_ps_worker_group.remote(self.ps_wg)
            psrl_logger.info("PS worker group bound successfully!")

    def switch_to_rollout_mode(self):
        """Switch the PSRL colocate part to rollout mode for validation.

        This involves several steps to ensure that the system transitions smoothly
        from training to rollout mode, particularly when validation and training are
        colocated.
        1. Deregister actor clients from NIXL to free up resources.
        2. Wake up validation instances and allocate necessary resources.
        3. Sync with the PS manager to update client information.
        4. Broadcast updated client information to all relevant clients.
        5. Sync validation instances' model weights and status with the PS.
        6. Resume the generation process in the rollout coordinator.
        """
        if not self.config.psrl.colocate_validate_and_train or self.is_rollout_mode_in_actor:
            return

        _switch_start = time.time()
        psrl_logger.info("Switching to rollout mode...")

        _t = time.time()
        psrl_logger.info("Step 1 - Deregistering actor clients from NIXL...")
        # actor_wg nixl client deregister weight memory
        release_futures = self.actor_wg.execute_all_async("nixl_sleep", "full")
        ray.get(release_futures)
        psrl_logger.info(f"Step 1 done in {time.time() - _t:.2f}s.")

        _t = time.time()
        psrl_logger.info("Step 2 - Waking up validation instances...")
        # Allocate rollout space and register
        ray.get(
            [self.tag_to_server_handles["validate"][i].nixl_wake_up.remote() for i in range(self.n_validate_instances)]
        )
        psrl_logger.info(f"Step 2 done in {time.time() - _t:.2f}s.")

        _t = time.time()
        psrl_logger.info("Step 3 - Syncing with ps manager...")
        # sync with server
        updated_client_names = []  # to collect all updated client names for broadcasting
        futures = []
        for i in range(self.n_validate_instances):
            instance_id = self.n_rollout_instances + i
            worker_names = [name for name in self.worker_to_ps_idx if name.startswith(f"validate_I{i}_R")]
            tp_size = len(worker_names)
            for rank in range(len(worker_names)):
                updated_client_names.append(f"{GLOBAL_GEN_CLIENT_NAME}_I{instance_id}_R{rank}")
            futures.append(
                self.tag_to_server_handles["validate"][i].nixl_send_local_info_to.remote(GLOBAL_META_SERVER_NAME)
            )
        # wait for ps manager to collect all infos
        futures.append(self.ps_manager_handle.nixl_wait_for_update_infos.remote(self.n_validate_instances * tp_size))
        ray.get(futures)
        psrl_logger.info(f"Step 3 done in {time.time() - _t:.2f}s.")

        # broadcast to other clients
        _t = time.time()
        psrl_logger.info("Step 4 - PS manager broadcasting updated client infos...")
        self._broadcast_updated_client_infos_from_ps_manager(updated_client_names)
        psrl_logger.info(f"Step 4 done in {time.time() - _t:.2f}s.")

        _t = time.time()
        psrl_logger.info("Step 5 - Syncing validation instances' model weights & status with PS...")
        # sync validation instances with ps
        # the generation will be resumed in the rollout coordinator
        resumed_instance_ids = []
        validate_dp_size = self.config.train_actor_rollout_ref.rollout.data_parallel_size
        for base_worker_id in self.tag_to_base_worker_ids.get("validate", []):
            resumed_instance_ids.extend((base_worker_id, i) for i in range(validate_dp_size))

        ray.get(self.rollout_coordinator.sync_with_ps.remote(resumed_instance_ids))
        psrl_logger.info(f"Step 5 done in {time.time() - _t:.2f}s.")

        _t = time.time()
        psrl_logger.info("Step 6 - Resuming validation instances...")
        # resume validation instances in router and coordinator
        if self.config.psrl.rollout_gateway.enable:
            resumed_base_worker_ids = self.tag_to_base_worker_ids.get("validate", [])
            self._post_gateway_worker_routing_control("resume", resumed_base_worker_ids)
        else:
            ray.get(self.rollout_router.resume_instances.remote(resumed_instance_ids))
        psrl_logger.info(f"Step 6 done in {time.time() - _t:.2f}s.")

        self.is_rollout_mode_in_actor = True
        psrl_logger.info(f"Switched to rollout mode in {time.time() - _switch_start:.2f}s.")

    def switch_to_trainer_mode(self):
        """Switch the PSRL colocate part to trainer mode for training.

        This involves several steps to ensure that the system transitions smoothly
        from rollout to training mode, particularly when validation and training are
        colocated.
        1. Notify agent loop workers about paused validation instances.
        2. Interrupt the generation process in validation instances.
        3. Put validation instances to sleep and deregister from NIXL.
        4. Wake up the training actor and allocate necessary resources.
        5. Sync with the PS manager to update client information.
        6. Broadcast updated client information to all relevant clients.
        7. Pull the latest model weights from the PS to the actor.
        """
        # notify coordinator + interrupt + sleep + upload actor
        if not self.config.psrl.colocate_validate_and_train or not self.is_rollout_mode_in_actor:
            return

        _switch_start = time.time()
        psrl_logger.info("Switching to trainer mode...")

        _t = time.time()
        psrl_logger.info("Step 1 - Pausing validation instances...")
        # pause validation instances in router and coordinator
        paused_instance_ids = []
        validate_dp_size = self.config.train_actor_rollout_ref.rollout.data_parallel_size
        for base_worker_id in self.tag_to_base_worker_ids.get("validate", []):
            paused_instance_ids.extend((base_worker_id, i) for i in range(validate_dp_size))

        if self.config.psrl.rollout_gateway.enable:
            paused_base_worker_ids = self.tag_to_base_worker_ids.get("validate", [])
            self._post_gateway_worker_routing_control("pause", paused_base_worker_ids)
        else:
            ray.get(self.rollout_router.pause_instances.remote(paused_instance_ids))
        psrl_logger.info(f"Step 1 done in {time.time() - _t:.2f}s.")

        _t = time.time()
        psrl_logger.info("Step 2 - Interrupting generation of validation instances...")
        # interrupt generation and sleep
        ray.get(
            [
                self.tag_to_server_handles["validate"][i].pause_generation.remote(clear_cache=False)
                for i in range(self.n_validate_instances)
            ]
        )
        psrl_logger.info(f"Step 2 done in {time.time() - _t:.2f}s.")

        _t = time.time()
        psrl_logger.info("Step 3 - Putting validation instances to sleep...")
        # sleep validation instances and deregister from NIXL
        ray.get(
            [
                self.tag_to_server_handles["validate"][i].nixl_sleep.remote(level=2)
                for i in range(self.n_validate_instances)
            ]
        )
        psrl_logger.info(f"Step 3 done in {time.time() - _t:.2f}s.")

        _t = time.time()
        psrl_logger.info("Step 4 - Waking up training actor...")
        # Allocate trainer space and register
        ray.get(self.actor_wg.execute_all_async("nixl_wake_up"))
        psrl_logger.info(f"Step 4 done in {time.time() - _t:.2f}s.")

        _t = time.time()
        psrl_logger.info("Step 5 - Syncing with ps manager...")
        # sync with server
        update_client_names = []  # to collect all updated client names for broadcasting
        futures = []
        for i in range(self.actor_wg.world_size):
            update_client_names.append(f"{GLOBAL_TRAIN_CLIENT_NAME}_{i}")
        # sender side: actor workers
        futures.extend(self.actor_wg.execute_all_async("nixl_send_local_info_to", GLOBAL_META_SERVER_NAME))
        # receiver side: ps manager
        futures.append(self.ps_manager_handle.nixl_wait_for_update_infos.remote(self.actor_wg.world_size))
        ray.get(futures)
        psrl_logger.info(f"Step 5 done in {time.time() - _t:.2f}s.")

        _t = time.time()
        psrl_logger.info("Step 6 - PS manager broadcasting updated client infos...")
        self._broadcast_updated_client_infos_from_ps_manager(update_client_names)
        psrl_logger.info(f"Step 6 done in {time.time() - _t:.2f}s.")

        _t = time.time()
        psrl_logger.info("Step 7 - Pulling actor model from PS...")
        # pull actor model
        ray.get(self.actor_wg.execute_all_async("pull_model"))
        psrl_logger.info(f"Step 7 done in {time.time() - _t:.2f}s.")

        self.is_rollout_mode_in_actor = False
        psrl_logger.info(f"Switched to trainer mode in {time.time() - _switch_start:.2f}s.")

    def _make_broadcast_plan(self, src_agent_names, dst_agent_names) -> dict:
        """Create a broadcast plan mapping source agents to destination agents.

        Args:
            src_agent_names (list): List of source agent names.
            dst_agent_names (list): List of destination agent names.
        Returns:
            dict: A dictionary mapping each source agent to a list of destination agents.
        """
        # simple round-robin broadcast plan
        # NOTE(linsh): currently only PS manager broadcasting is implemented.
        # This method can be extended for more complex plans if needed.
        broadcast_plan = {src_agent: [] for src_agent in src_agent_names}
        for i, dst_agent in enumerate(dst_agent_names):
            src_agent = src_agent_names[i % len(src_agent_names)]
            broadcast_plan[src_agent].append(dst_agent)
        return broadcast_plan

    def _broadcast_updated_client_infos_from_ps_manager(self, updated_client_names: list):
        """Broadcast updated client infos from PS manager to all nixl clients.

        Args:
            updated_client_names (list): List of updated client names to broadcast.
        """
        src_agent_names = [GLOBAL_META_SERVER_NAME]
        dst_agent_names = []
        # 1. ps storage workers
        ps_worker_num = self.ps_wg.world_size
        for i in range(ps_worker_num):
            dst_agent_names.append(f"{GLOBAL_PS_CLIENT_NAME}_{i}")
        # 2. rollout workers
        for i in range(self.n_rollout_instances):
            worker_names = [name for name in self.worker_to_ps_idx if name.startswith(f"rollout_I{i}_R")]
            for rank in range(len(worker_names)):
                dst_agent_names.append(f"{GLOBAL_GEN_CLIENT_NAME}_I{i}_R{rank}")
        for i in range(self.n_validate_instances):
            worker_names = [name for name in self.worker_to_ps_idx if name.startswith(f"validate_I{i}_R")]
            instance_id = self.n_rollout_instances + i
            for rank in range(len(worker_names)):
                dst_agent_names.append(f"{GLOBAL_GEN_CLIENT_NAME}_I{instance_id}_R{rank}")
        # 3. actor workers
        actor_worker_num = self.actor_wg.world_size
        for i in range(actor_worker_num):
            dst_agent_names.append(f"{GLOBAL_TRAIN_CLIENT_NAME}_{i}")
        psrl_logger.debug(f"Destination agent names for broadcasting: {dst_agent_names}")

        broadcast_plan = self._make_broadcast_plan(src_agent_names, dst_agent_names)
        futures = []
        for src_agent, dst_agents in broadcast_plan.items():
            if src_agent == GLOBAL_META_SERVER_NAME:
                futures.append(
                    self.ps_manager_handle.nixl_broadcast_update_client_infos.remote(dst_agents, updated_client_names)
                )
            else:
                raise NotImplementedError(
                    "Only meta server broadcasting is implemented in _broadcast_updated_client_infos_from_ps_manager"
                )

        # recv broadcast results by all clients
        # 1. ps storage workers
        futures.extend(self.ps_wg.execute_all_async("nixl_wait_for_update_infos", 1))
        # 2. rollout workers
        futures.extend(
            [
                self.tag_to_server_handles["rollout"][i].nixl_wait_for_update_infos.remote(1)
                for i in range(self.n_rollout_instances)
            ]
        )
        futures.extend(
            [
                self.tag_to_server_handles["validate"][i].nixl_wait_for_update_infos.remote(1)
                for i in range(self.n_validate_instances)
            ]
        )
        # 3. actor workers
        futures.extend(self.actor_wg.execute_all_async("nixl_wait_for_update_infos", 1))
        ray.get(futures)

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        psrl_logger.info(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir,
                f"global_step_{self.global_steps}",
                "actor",
            )
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            psrl_logger.warning(
                "remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_wg.save_checkpoint(
            actor_local_path,
            actor_remote_path,
            self.global_steps,
            max_ckpt_to_keep=max_actor_ckpt_to_keep,
        )
        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir,
                    f"global_step_{self.global_steps}",
                    "critic",
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path,
                critic_remote_path,
                self.global_steps,
                max_ckpt_to_keep=max_critic_ckpt_to_keep,
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        ray.get(self.data_processor.save_train_dataloader.remote(dataloader_local_path))

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.train_actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.train_actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.train_actor_rollout_ref.actor.checkpoint
            and self.config.train_actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            psrl_logger.info("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                psrl_logger.info("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        psrl_logger.info(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        psrl_logger.info(f"Setting global step to {self.global_steps}")
        psrl_logger.info(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor (train only)
        self.actor_wg.load_checkpoint(
            actor_path,
            del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
        )

        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path,
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )

        # TODO(lhy): push the actor model state dict to the PS worker (though it is not necessary to do so)

        # load dataloader
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            ray.get(self.data_processor.load_train_dataloader.remote(dataloader_local_path))
        else:
            psrl_logger.info(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        dp_size = self._get_dp_size(self.actor_wg, "actor")
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=dp_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst,
            partitions=global_partition_lst,
            prefix=logging_prefix,
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        psrl_logger.info(
            f"Initialized tracking logger with project: {self.config.trainer.project_name}, "
            f"experiment: {self.config.trainer.experiment_name}"
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        if self.global_steps >= self.total_training_steps:
            psrl_logger.warning(
                f"Global steps {self.global_steps} >= total training steps {self.total_training_steps}, "
                "skipping training."
            )
            return

        self.init_agent_loop_manager()

        futures = []
        futures.append(self.data_processor.set_agent_loop_manager.remote(self.agent_loop_manager))
        futures.append(self.ps_manager_handle.set_rollout_coordinator.remote(self.rollout_coordinator))
        for agent_loop_worker in self.agent_loop_workers:
            futures.append(agent_loop_worker.set_agent_loop_manager.remote(self.agent_loop_manager))
        ray.get(futures)

        self.init_reward_manager()
        futures = []
        futures.append(self.data_processor.set_reward_manager.remote(self.reward_manager))
        futures.append(self.ps_manager_handle.set_reward_manager.remote(self.reward_manager))
        for agent_loop_worker in self.agent_loop_workers:
            futures.append(agent_loop_worker.set_reward_manager.remote(self.reward_manager))
        ray.get(futures)

        # Start data pipeline
        if not self.config.psrl.colocate:
            # Start rollout coordinator to handle rollouts and data generation
            psrl_logger.info("Starting rollout coordinator...")
            self.start_rollout_coordinator()
            psrl_logger.info("Rollout coordinator started successfully.")

            # Start agent loop manager to handle agent-environment interactions
            psrl_logger.info("Starting agent loop manager...")
            self.start_agent_loop_manager()
            psrl_logger.info("Agent loop manager started successfully.")

            # Start reward manager to handle reward computation requests
            psrl_logger.info("Starting reward manager...")
            self.start_reward_manager()
            psrl_logger.info("Reward manager started successfully.")

        psrl_logger.info("All data pipeline components started successfully.")

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            psrl_logger.info(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # Start data processor to handle data preprocessing and batching
        psrl_logger.info("Starting data processor...")
        self.start_data_processor()
        psrl_logger.info("Data processor started successfully.")

        # add tqdm
        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Training Progress",
        )

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        # busy loop for training
        while True:
            metrics = {}
            timing_raw = {}
            is_last_step = self.global_steps == self.total_training_steps

            with marked_timer("step", timing_raw):
                # Wait for the training batch to be ready
                with marked_timer("wait_for_gen", timing_raw, color="gray"):
                    if not self.config.psrl.colocate:
                        buffer_id = self.global_steps - 1
                        # will block until the training batch is ready
                        psrl_logger.debug("Waiting for training batch with buffer_id %d", buffer_id)
                        with log_dual_events(
                            f"Wait for training batch {buffer_id}",
                            psrl_logger,
                            event_type=EventType.WAIT,
                        ):
                            batch = ray.get(self.agent_loop_manager.wait_for_training_batch.remote(buffer_id))
                        # psrl_logger.info(
                        #     "Received training batch for step %d, batch size: %d",
                        #     self.global_steps,
                        #     len(batch) if batch is not None else 0,
                        # )
                        # psrl_logger.info(f"batch = {batch}")
                        with log_dual_events("Switch to trainer mode", psrl_logger, event_type=EventType.SWITCH):
                            self.switch_to_trainer_mode()
                    else:
                        from psrl.trainer.ppo.reward import compute_reward

                        batch = ray.get(self.agent_loop_manager.get_data.remote())
                        if batch is None:
                            psrl_logger.info(
                                "No more data from agent loop manager, ending training at step %d",
                                self.global_steps,
                            )
                            break
                        batch_keys_to_pop = [
                            "input_ids",
                            "attention_mask",
                            "position_ids",
                        ]
                        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
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
                        gen_batch = batch.pop(
                            batch_keys=batch_keys_to_pop,
                            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                        )
                        gen_batch.meta_info["need_pull_model"] = self.global_steps != 1
                        # Verl original colocate method
                        output_batch = self.actor_wg.generate_sequences(gen_batch)
                        batch = batch.union(output_batch)
                        reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                        batch.batch["reward"] = reward_tensor
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                
                # Log multi-turn or other metrics
                if "metrics" in batch.meta_info:
                    gen_metrics = batch.meta_info["metrics"]
                    metrics.update(reduce_metrics(gen_metrics))
                    batch.meta_info["metrics"].clear()

                if "response_mask" not in batch.batch.keys():
                    batch.batch["response_mask"] = compute_response_mask(batch)
                # Balance the number of valid tokens across DP ranks.
                # NOTE: This usually changes the order of data in the `batch`,
                # which won't affect the advantage calculation (since it's based on uid),
                # but might affect the loss calculation (due to the change of mini-batching).
                # Please take care when you implement group based adv computation such as GRPO and rloo
                # TODO(verl): Decouple the DP balancing and mini-batching.
                if self.config.trainer.balance_batch:
                    self._balance_batch(batch, metrics=metrics)

                # compute global_valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                batch.meta_info["temperature"] = self.config.gen_actor_rollout_ref.rollout.temperature

                # Operating Mode Selection:
                # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)

                if bypass_recomputing_logprobs:
                    from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                    apply_bypass_mode(
                        batch=batch,
                        rollout_corr_config=rollout_corr_config,
                        policy_loss_config=self.config.train_actor_rollout_ref.actor.policy_loss,
                    )

                    batch.batch.pop("rollout_log_probs")
                else:
                    # recompute log_probs in the training side
                    with marked_timer("recompute_log_prob", timing_raw, color="orange"):
                        with log_dual_events(
                            "Recompute log_prob on training side",
                            psrl_logger,
                            event_type=EventType.OTHER,
                        ):
                            recomputed_log_prob = self.actor_wg.compute_log_prob(batch)
                            entropys = recomputed_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.train_actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            metrics.update({"actor/entropy": entropy_agg.detach().item()})
                            recomputed_log_prob.batch.pop("entropys")
                            batch = batch.union(recomputed_log_prob)

                            if "rollout_log_probs" in batch.batch.keys():
                                rollout_old_log_probs = batch.batch["rollout_log_probs"]
                                recomputed_log_probs = batch.batch["recomputed_log_probs"]
                                attention_mask = batch.batch["attention_mask"]
                                responses = batch.batch["responses"]
                                response_length = responses.size(1)
                                response_mask = attention_mask[:, -response_length:]

                                rollout_probs = torch.exp(rollout_old_log_probs)
                                recomputed_probs = torch.exp(recomputed_log_probs)
                                probs_diff = torch.abs(rollout_probs - recomputed_probs)
                                probs_diff = torch.masked_select(probs_diff, response_mask.bool())
                                probs_diff_max = torch.max(probs_diff)
                                probs_diff_mean = torch.mean(probs_diff)
                                probs_diff_std = torch.std(probs_diff)
                                metrics.update(
                                    {
                                        "training/probs_diff_max": probs_diff_max.detach().item(),
                                        "training/probs_diff_mean": probs_diff_mean.detach().item(),
                                        "training/probs_diff_std": probs_diff_std.detach().item(),
                                    }
                                )
                    batch.batch["old_log_probs"] = batch.batch["recomputed_log_probs"]
                    batch.batch.pop("recomputed_log_probs")

                if self.use_reference_policy:
                    # compute reference log_prob
                    with marked_timer("ref", timing_raw, color="olive"):
                        with log_dual_events(
                            "Compute reference log_prob",
                            psrl_logger,
                            event_type=EventType.OTHER,
                        ):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                # compute values
                if self.use_critic:
                    with marked_timer("values", timing_raw, color="cyan"):
                        with log_dual_events(
                            "Compute critic values",
                            psrl_logger,
                            event_type=EventType.OTHER,
                        ):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                # compute reward model score
                if self.use_rm and "rm_scores" not in batch.batch.keys():
                    with marked_timer("reward", timing_raw, color="yellow"):
                        with log_dual_events(
                            "Compute reward model score",
                            psrl_logger,
                            event_type=EventType.OTHER,
                        ):
                            # compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)
                elif self.config.reward_model.launch_reward_fn_async:
                    # Overlap reward computation with log_prob computation in trainer
                    with marked_timer("async_reward_get", timing_raw, color="yellow"):
                        with log_dual_events(
                            "Wait for async reward model score",
                            psrl_logger,
                            event_type=EventType.OTHER,
                        ):
                            request_ids = batch.non_tensor_batch["uid"].tolist()
                            print(f"Waiting for reward of request_ids: {request_ids}")
                            assert self.reward_manager is not None, "Reward manager is not initialized"
                            request_id_to_reward = ray.get(
                                self.reward_manager.wait_for_reward_of_requests.remote(request_ids)
                            )

                        with log_dual_events(
                            "Post process async reward model score",
                            psrl_logger,
                            event_type=EventType.OTHER,
                        ):
                            scores = []
                            reward_extra_infos_dict_list = []
                            for request_id in request_ids:
                                reward_score = request_id_to_reward[request_id]["reward_score"]
                                extra_info = request_id_to_reward[request_id].get("reward_extra_info", {})
                                scores.append(reward_score)
                                reward_extra_infos_dict_list.append(extra_info)

                            prompt_length = batch.batch["prompts"].size(1)
                            response_length = batch.batch["attention_mask"][:, prompt_length:].sum(dim=1) - 1
                            rm_scores = torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32)
                            rm_scores[
                                torch.arange(batch.batch["response_mask"].size(0)),
                                response_length,
                            ] = torch.tensor(scores, dtype=torch.float32)
                            reward_tensor = rm_scores  # [bsz, response_length]

                            # add reward_extra_info to non_tensor_batch
                            reward_extra_infos_dict = defaultdict(list)
                            for reward_extra_infos in reward_extra_infos_dict_list:
                                for key, value in reward_extra_infos.items():
                                    if not isinstance(value, list):
                                        value = [value]
                                    reward_extra_infos_dict[key].extend(value)
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                else:
                    reward_tensor = batch.batch.pop("rm_scores", None)

                batch.batch["token_level_scores"] = reward_tensor

                with marked_timer("adv", timing_raw, color="brown"):
                    with log_dual_events("Compute advantage", psrl_logger, event_type=EventType.OTHER):
                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import (
                                compute_rollout_correction_and_add_to_batch,
                            )

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        log_data_protocol(
                            batch,
                            psrl_logger,
                            self.log_prefix + " before compute advantage",
                            level=logging.DEBUG,
                        )
                        batch = PSRL_compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.gen_actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                # update critic
                if self.use_critic:
                    with marked_timer("update_critic", timing_raw, color="pink"):
                        with log_dual_events("Update critic", psrl_logger, event_type=EventType.TRAIN):
                            critic_output = self.critic_wg.update_critic(batch)
                    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    metrics.update(critic_output_metrics)

                # implement critic warmup
                if self.config.trainer.critic_warmup <= self.global_steps:
                    # update actor
                    with marked_timer("update_actor", timing_raw, color="red"):
                        with log_dual_events("Update actor", psrl_logger, event_type=EventType.TRAIN):
                            batch.meta_info["multi_turn"] = self.config.gen_actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_wg.update_actor(batch)
                    # psrl_logger.info(
                    #     f"Update actor ppo_kl: {actor_output.meta_info['metrics']['actor/ppo_kl']}, "
                    #     f"len: {len(actor_output.meta_info['metrics']['actor/ppo_kl'])}"
                    # )
                    # psrl_logger.info(
                    #     f"Update actor pg_loss: {actor_output.meta_info['metrics']['actor/pg_loss']}, "
                    #     f"len: {len(actor_output.meta_info['metrics']['actor/pg_loss'])}"
                    # )
                    # psrl_logger.info(
                    #     f"Update actor grad_norm: {actor_output.meta_info['metrics']['actor/grad_norm']}, "
                    #     f"len: {len(actor_output.meta_info['metrics']['actor/grad_norm'])}"
                    # )
                    psrl_logger.info(f"{actor_output.meta_info['metrics']=}")
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                # Log rollout generations if enabled
                rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                if rollout_data_dir:
                    with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                        with log_dual_events(
                            "Dump rollout generations",
                            psrl_logger,
                            event_type=EventType.OTHER,
                        ):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            sample_gts = [
                                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                                for item in batch
                            ]
                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                gts=sample_gts,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        with log_dual_events("Validate", psrl_logger, event_type=EventType.VAL):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        with log_dual_events("Save checkpoint", psrl_logger, event_type=EventType.OTHER):
                            self._save_checkpoint()

            with marked_timer("stop_profile", timing_raw):
                next_step_profile = (
                    self.global_steps + 1 in self.config.global_profiler.steps
                    if self.config.global_profiler.steps is not None
                    else False
                )
                self._stop_profiling(
                    curr_step_profile and not next_step_profile
                    if self.config.global_profiler.profile_continuous_steps
                    else curr_step_profile
                )
                prev_step_profile = curr_step_profile
                curr_step_profile = next_step_profile

            steps_duration = timing_raw["step"]
            self.max_steps_duration = max(self.max_steps_duration, steps_duration)

            # training metrics
            metrics.update(
                {
                    "training/global_step": self.global_steps,
                }
            )
            # collect metrics
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            # TODO(verl): implement actual tflpo and theoretical tflpo
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

            # TODO(verl): make a canonical logger that supports various backend
            logger.log(data=metrics, step=self.global_steps)

            progress_bar.update(1)
            self.global_steps += 1

            if (
                hasattr(self.config.train_actor_rollout_ref.actor, "profiler")
                and self.config.train_actor_rollout_ref.actor.profiler.tool == "torch_memory"
            ):
                self.actor_wg.dump_memory_snapshot(
                    tag=f"post_update_step{self.global_steps}",
                    sub_dir=f"step{self.global_steps}",
                )

            if is_last_step:
                psrl_logger.info(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                break

        # Stop all components
        self.stop_data_processor()
        self.stop_agent_loop_manager()
        self.stop_rollout_coordinator()
        self.stop_reward_manager()
        self.stop_ps_manager()

        psrl_logger.info("Training completed successfully!")
