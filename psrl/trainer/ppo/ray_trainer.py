import asyncio
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
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from tqdm import tqdm
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import SubRayResourcePool, create_colocated_worker_cls_fused
from verl.trainer.distillation.losses import is_distillation_enabled
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, apply_kl_penalty, compute_response_mask
from verl.trainer.ppo.utils import WorkerType
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.seqlen_balancing import (
    calculate_workload,
    get_seqlen_balanced_partitions,
    log_seqlen_unbalance,
)
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import DistillationConfig, EngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

from psrl.trainer.ppo.utils import (
    PSRL_compute_advantage,
    PSRL_Role,
    ResourcePoolManager,
    extract_gen_rm_token_num,
    need_critic,
    need_reference_policy,
    need_teacher_policy,
    record_rollout_rm_metrics,
)
from psrl.utils.common.nixl_names import NIXL_META_SERVER_NAME
from psrl.utils.common.worker_naming import WorkerKey, ps_agent_name, train_client_name
from psrl.utils.dataset import DataProcessor, DatasetType
from psrl.utils.elastic_rm.cluster_topology import ClusterTopology
from psrl.utils.elastic_rm.elastic_executor import ElasticExecutor
from psrl.utils.logger import (
    DualOutputHandler,
    EventType,
    log_data_protocol,
    log_dual_events,
)
from psrl.utils.server.command import Command, CommandType
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
from psrl.workers.reward.reward_manager import RewardLoopManager
from psrl.workers.reward.reward_model import PSRL_RewardModelManager
from psrl.workers.reward.reward_model.gateway import RewardModelGateway
from psrl.workers.train import TrainInterface

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class PSRL_RayPPOTrainer(RayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[PSRL_Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
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

        # AGENT(VERL): PSRL use `config.train_actor_rollout_ref` instead of `config.actor_rollout_ref` in verl.

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        # AGENT(VERL): skip `hybrid_engine` in PSRL

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_teacher_policy = need_teacher_policy(self.config)

        # AGENT(VERL): skip `use_rm` in PSRL.

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

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.train_actor_rollout_ref.actor.get("use_prefix_grouper", False)

        # AGENT(VERL): skip legacy worker impl and dataloader in PSRL
        # PSRL's dataloader is moved to `data_processor`.

        # ---- PSRL specific initialization ----

        self.collate_fn = collate_fn
        self.group_post_process_fn = group_post_process_fn
        self.buffer_post_process_fn = buffer_post_process_fn

        # CPU workers for Streaming Rollout
        self.data_processor = None
        self.agent_loop_manager = None
        self.rollout_coordinator = None
        self.reward_manager = None

        self.reward_gateways: dict[str, ray.actor.ActorHandle] = {}
        self.reward_gateway_urls: dict[str, str] = {}
        self.reward_model_to_manager = {}

        # Elastic rm
        self.elastic_rm_mode = config.psrl.deployment.elastic_rm.enable
        self.elastic_executor = None

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

        # Mappings from WorkerKey to Ray node id and PS instance index for NIXL.
        self.worker_to_node_id: dict[WorkerKey, str] = {}
        self.worker_to_ps_idx: dict[WorkerKey, int] = {}

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

        self._initialize_queue_buffers()

        self._init_ps_manager()

        # initialize data processor
        # NOTE(lhy): data processor must be initialized before initializing other workers
        # so that the total_training_steps can be obtained and the optimizer config
        # (related to weight decay, lr schedule, etc.) can be set
        # otherwise, it will cause error when running Megatron backend
        self._init_data_processor()

        # Build logger
        self.log_prefix = "MainRayTrainer"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))
        psrl_logger.info("Initialized major ray trainer (single controller).")

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
        # AGENT(VERL): this logic is mapped to `_create_dataloader` in `verl/trainer/ppo/ray_trainer.py`
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
        self.agent_loop_manager = (
            ray.remote(PSRL_AgentLoopManager)
            .options(max_concurrency=self.max_concurrency)
            .remote(
                self.config,
                self.data_queue_size,
                self.agent_loop_workers,
                self.ps_manager_handle,
                group_post_process_fn=self.group_post_process_fn,
                buffer_post_process_fn=self.buffer_post_process_fn,
            )
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

    def init_reward_gateways(self):
        """Launch one smg gateway per named generative reward model."""
        for rm_cfg in self.config.reward.reward_models:
            if rm_cfg.reward_loop_type != "gen":
                continue
            reward_model_name = rm_cfg.get("reward_model_name", rm_cfg.model.path.split("/")[-1])
            gateway = RewardModelGateway.options(name=f"reward_gateway_{reward_model_name}").remote(
                self.config, reward_model_name
            )
            url = ray.get(gateway.launch_router.remote())
            self.reward_gateways[reward_model_name] = gateway
            self.reward_gateway_urls[reward_model_name] = url
            psrl_logger.info("Reward gateway for '%s' launched at %s", reward_model_name, url)

    def init_reward_model_servers(self, all_wg: dict):
        """
        Initialize reward model HTTP servers and register them to their smg gateways.

        Args:
            all_wg: dict mapping worker-group name → RayWorkerGroup.
        """
        for rm_cfg in self.config.reward.reward_models:
            if rm_cfg.reward_loop_type != "gen":
                continue
            reward_model_name = rm_cfg.get("reward_model_name", rm_cfg.model.path.split("/")[-1])
            reward_model_wg_list = [
                all_wg[f"reward_model_{reward_model_name}_{i}"] for i in range(rm_cfg.num_replicas)
            ]
            gateway_url = self.reward_gateway_urls[reward_model_name]
            self.reward_model_to_manager[reward_model_name] = PSRL_RewardModelManager(
                reward_model_name=reward_model_name,
                config=self.config,
                reward_model_config=rm_cfg,
                reward_model_wg_list=reward_model_wg_list,
                gateway_url=gateway_url,
            )
            psrl_logger.info("RewardModelManager for '%s' initialized.", reward_model_name)
        psrl_logger.info("reward_model_to_manager: %s", list(self.reward_model_to_manager))

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

    def init_elastic_rm_runtime(self):
        if not self.elastic_rm_mode:
            return
        if self.rollout_coordinator is None:
            raise RuntimeError("Rollout coordinator must be initialized before elastic_rm init.")

        psrl_logger.info("Initializing elastic_executor runtime (coordinator sleep/wake, ElasticExecutor).")
        rollout_model_name = self.config.gen_actor_rollout_ref.model.path.split("/")[-1]
        rollout_instance_num = len(self.rollout_wg_list)
        psrl_logger.info(
            "Elastic_RM: rollout model_name=%s, n_instances=%d",
            rollout_model_name,
            rollout_instance_num,
        )

        # Enable coordinator command handling before elastic sleep/wake orchestration.
        self.start_rollout_coordinator()
        psrl_logger.info("Rollout coordinator busy loop started for elastic_rm command handling.")

        # ── Build roles / coordinators and create ElasticExecutor early ──
        reward_coordinators: dict[str, ray.actor.ActorHandle] = {}
        for reward_model_name, manager in self.reward_model_to_manager.items():
            reward_coordinators[reward_model_name] = manager.reward_model_coordinator

        roles = [(PSRL_Role.Rollout, rollout_model_name)]
        roles.extend((PSRL_Role.RewardModel, rm_name) for rm_name in reward_coordinators)
        coordinators = {
            PSRL_Role.Rollout: {rollout_model_name: self.rollout_coordinator},
            PSRL_Role.RewardModel: reward_coordinators,
        }
        elastic_rm_cfg = OmegaConf.to_container(self.config.psrl.deployment.elastic_rm, resolve=True)
        assert isinstance(elastic_rm_cfg, dict), "elastic_rm config should be resolved as dict"

        self.elastic_executor = ElasticExecutor.remote(
            config=self.config,
            roles=roles,
            coordinators=coordinators,
            agent_loop_manager=self.agent_loop_manager,
            elastic_rm_config=elastic_rm_cfg,
        )
        psrl_logger.info(
            "Elastic_RM: ElasticExecutor created (roles=%s).",
            [(r.name, m) for r, m in roles],
        )

        # ── Sleep all rollout instances and register with executor ──
        rollout_all_ids: list = ray.get(self.rollout_coordinator.get_all_instance_ids.remote())
        if rollout_all_ids:
            psrl_logger.info(
                "Elastic_RM: putting all rollout instances to sleep (instance_ids=%s).",
                rollout_all_ids,
            )
            ray.get(
                self.rollout_coordinator.exec_command.remote(
                    Command(type=CommandType.SLEEP, instance_ids=rollout_all_ids),
                    blocking=True,
                )
            )
            psrl_logger.info("Elastic_RM: all rollout instances slept.")

        rollout_gpu_slots = [ClusterTopology.collect_gpu_slots_from_worker_group(wg) for wg in self.rollout_wg_list]
        ray.get(
            self.elastic_executor.register_role.remote(
                role_name=PSRL_Role.Rollout,
                model_name=rollout_model_name,
                instance_ids=rollout_all_ids,
                gpu_slots_per_instance=rollout_gpu_slots,
            )
        )
        psrl_logger.info("Elastic_RM: registered %d rollout instances.", len(rollout_all_ids))

        # ── Sleep all reward model instances and register with executor ──
        for reward_model_name, manager in self.reward_model_to_manager.items():
            rm_all_ids: list = ray.get(manager.reward_model_coordinator.get_all_instance_ids.remote())
            psrl_logger.info(
                "Elastic_RM: putting reward model replicas to sleep (name=%s, instance_ids=%s).",
                reward_model_name,
                rm_all_ids,
            )
            ray.get(
                manager.reward_model_coordinator.exec_command.remote(
                    Command(type=CommandType.SLEEP, instance_ids=rm_all_ids),
                    blocking=True,
                )
            )
            psrl_logger.info("Elastic_RM: reward model %s replicas slept.", reward_model_name)

            rm_gpu_slots = [
                ClusterTopology.collect_gpu_slots_from_worker_group(wg)
                for wg in self.reward_model_to_wg_list[reward_model_name]
            ]
            ray.get(
                self.elastic_executor.register_role.remote(
                    role_name=PSRL_Role.RewardModel,
                    model_name=reward_model_name,
                    instance_ids=rm_all_ids,
                    gpu_slots_per_instance=rm_gpu_slots,
                )
            )
            psrl_logger.info(
                "Elastic_RM: registered %d reward model instances (%s).", len(rm_all_ids), reward_model_name
            )

        # ── Select non-conflicting initial instances and wake them up ──
        min_awake_per_role = max(0, int(self.config.psrl.deployment.elastic_rm.min_awake_per_role))

        # Wake reward-model replicas first so each RM keeps at least one awake instance.
        for reward_model_name, manager in self.reward_model_to_manager.items():
            rm_all_ids = ray.get(manager.reward_model_coordinator.get_all_instance_ids.remote())
            rm_awake_num = max(min_awake_per_role, 0)
            rm_awake_ids = ray.get(
                self.elastic_executor.select_initial_awake_ids.remote(
                    role_name=PSRL_Role.RewardModel,
                    model_name=reward_model_name,
                    target_awake_num=rm_awake_num,
                    min_awake_num=min_awake_per_role if len(rm_all_ids) > 0 else 0,
                )
            )
            if rm_awake_ids:
                psrl_logger.info(
                    "Elastic_RM: waking up reward model replicas (name=%s, count=%d, ids=%s).",
                    reward_model_name,
                    len(rm_awake_ids),
                    rm_awake_ids,
                )
                ray.get(
                    manager.reward_model_coordinator.exec_command.remote(
                        Command(type=CommandType.WAKE_UP, instance_ids=rm_awake_ids),
                        blocking=True,
                    )
                )
                psrl_logger.info("Elastic_RM: reward model %s wake_up completed.", reward_model_name)

        rollout_awake_num = max(min_awake_per_role, rollout_instance_num)
        rollout_awake_ids = ray.get(
            self.elastic_executor.select_initial_awake_ids.remote(
                role_name=PSRL_Role.Rollout,
                model_name=rollout_model_name,
                target_awake_num=rollout_awake_num,
                min_awake_num=min_awake_per_role if rollout_instance_num > 0 else 0,
            )
        )
        if rollout_awake_ids:
            psrl_logger.info(
                "Elastic_RM: waking up rollout instances (count=%d, ids=%s).",
                len(rollout_awake_ids),
                rollout_awake_ids,
            )
            ray.get(
                self.rollout_coordinator.exec_command.remote(
                    Command(type=CommandType.WAKE_UP, instance_ids=rollout_awake_ids),
                    blocking=True,
                )
            )
            psrl_logger.info("Elastic_RM: rollout wake_up completed.")

        # ── Start monitor loop ──
        ray.get(self.elastic_executor.start_busy_loop.remote())
        psrl_logger.info("Elastic_RM: ElasticExecutor busy loop started; runtime ready.")

    def init_reward_manager(self):
        """Initialize a single reward manager for both training and validation reward computation."""
        ip_to_node_id = {node["NodeManagerAddress"]: node["NodeID"] for node in ray.nodes()}
        assert self.data_processor is not None, (
            "Data processor must be initialized before starting reward computation."
        )
        assert self.rollout_coordinator is not None, (
            "Rollout server must be initialized before starting reward computation."
        )

        self.reward_manager = (
            ray.remote(RewardLoopManager)
            .options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ip_to_node_id[self.config.psrl.reward_service_ip],
                    soft=False,
                )
            )
            .remote(
                config=self.config,
                tokenizer=self.tokenizer,
                processor=self.processor,
                ps_manager_handle=self.ps_manager_handle,
                reward_model_configs=self.config.reward.reward_models,
                reward_model_to_manager=self.reward_model_to_manager,
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

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            # AGENT(VERL): add PSRL specific `log_dual_events` here.
            with log_dual_events(
                "Dump rollout generations",
                psrl_logger,
                event_type=EventType.OTHER,
            ):
                inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                sample_gts = [
                    item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch
                ]

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

    def _validate(self, merged: bool = False):
        """Validate the model using the validation dataset.

        Note that we use the training side to do val for overlapping with generation.
        """
        # AGENT(VERL): PSRL add switch between train/rollout mode.
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
        # AGENT(VERL): PSRL use `parent_id` to group samples of the same prompt together,
        # while verl use `uid` instead.
        sample_parent_ids = []

        # AGENT(VERL): PSRL get validation data size from data processor and
        # set the capacity of staleness inventory and val buffer accordingly.
        val_data_size = ray.get(self.data_processor.get_val_data_size.remote())
        futures = []
        futures.append(self.ps_manager_handle.set_val_staleness_inventory_capacity.remote(val_data_size))
        futures.append(self.agent_loop_manager.set_val_buffer_size.remote(val_data_size))
        ray.get(futures)
        val_rollout_n = self.config.train_actor_rollout_ref.rollout.val_kwargs.n
        val_batch_num = ray.get(self.data_processor.get_val_batch_num.remote())
        assert self.reward_manager is not None, "Reward manager must be initialized before validation."

        for i in range(val_batch_num):
            test_data = ray.get(self.data_processor.get_single_controller_batch.remote(DatasetType.val))
            test_batch = DataProto.from_single_dict(test_data)
            batch_size = len(test_batch.batch)

            sample_ids = ray.get(self.data_processor.get_val_sample_ids.remote(batch_size))
            test_batch.non_tensor_batch["parent_id" if val_rollout_n > 1 else "uid"] = np.array(sample_ids)
            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=val_rollout_n, interleave=True)

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            # AGENT(VERL): PSRL add `uid` for each sample to distinguish different samples.
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

            # AGENT(VERL): PSRL use its own generate function for validation.
            val_buffer_id = ray.get(self.agent_loop_manager.generate_validate_sequences.remote(test_gen_batch))
            with log_dual_events(f"Wait for validation batch {val_buffer_id}", psrl_logger, event_type=EventType.WAIT):
                test_output_gen_batch = ray.get(
                    self.agent_loop_manager.wait_for_validation_batch.remote(val_buffer_id)
                )

            # TODO(linsh): refactor it to be verl-style.
            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            request_id_to_reward = ray.get(self.reward_manager.compute_score_for_validation.remote(test_batch))
            request_ids = test_batch.non_tensor_batch["uid"].tolist()
            scores = []
            reward_extra_infos_dict_list = []
            acc_list = []
            for request_id in request_ids:
                reward_score = request_id_to_reward[request_id]["reward_score"]
                # reward_extra_info is now {loop_key: per_loop_info_dict, ...}.
                # Merge all per-loop dicts into a single flat dict so downstream
                # code can access keys like "acc" regardless of which loop produced them.
                per_loop_infos = request_id_to_reward[request_id].get("reward_extra_info", {})
                extra_info = {}
                for per_loop_info in per_loop_infos.values():
                    extra_info.update(per_loop_info)
                acc = extra_info.get("acc", 0.0)
                scores.append(reward_score)
                reward_extra_infos_dict_list.append(extra_info)
                acc_list.append(acc)
            sample_scores.extend(scores)
            reward_extra_infos_dict["reward"].extend(scores)
            reward_extra_infos_dict["reward_extra_info"].extend(reward_extra_infos_dict_list)
            reward_extra_infos_dict["acc"].extend(acc_list)

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO(verl): Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_parent_ids.extend(test_batch.non_tensor_batch["parent_id" if val_rollout_n > 1 else "uid"])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * len(request_ids)))

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

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_parent_ids": sample_parent_ids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)

        with log_dual_events("Switch to trainer mode", psrl_logger, event_type=EventType.SWITCH):
            self.switch_to_trainer_mode()

        return self._val_metrics_update(data_sources, sample_parent_ids, reward_extra_infos_dict, sample_turns)

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
                    role=tag,
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

        # ---- Step 1: Create each role by mapping resource pool to worker class ----

        # create elastic reward model if needed
        elastic_shared_pool = None
        elastic_subpool_group_idx = 0
        if self.elastic_rm_mode:
            elastic_shared_pool = self.resource_pool_manager.get_resource_pool(PSRL_Role.Rollout)
            # Materialize placement groups once and reuse them across all elastic sub resource pools.
            elastic_shared_pool.get_placement_groups(strategy="STRICT_PACK", device_name=self.device_name)
            psrl_logger.info(
                "Elastic RM shared RayResourcePool: name_prefix=%s world_size=%d store=%s max_colocate_count=%s",
                elastic_shared_pool.name_prefix,
                elastic_shared_pool.world_size,
                list(elastic_shared_pool.store),
                elastic_shared_pool.max_colocate_count,
            )

        def _build_elastic_sub_resource_pool(subgroup_world_size: int, tag: str) -> SubRayResourcePool:
            nonlocal elastic_subpool_group_idx
            assert elastic_shared_pool is not None, "elastic_shared_pool must be initialized in elastic_rm_mode"
            if subgroup_world_size <= 0:
                raise ValueError(f"subgroup_world_size must be > 0, but got {subgroup_world_size} ({tag})")
            if subgroup_world_size > elastic_shared_pool.world_size:
                raise ValueError(
                    f"subgroup_world_size={subgroup_world_size} exceeds shared pool world_size="
                    f"{elastic_shared_pool.world_size} ({tag})"
                )

            # SubRayResourcePool requires a contiguous bundle range [start, start + subgroup_world_size).
            # We rotate start index to reduce collisions while allowing controlled oversubscription.
            max_start = elastic_shared_pool.world_size - subgroup_world_size
            if max_start == 0:
                start_bundle_index = 0
            else:
                start_bundle_index = (elastic_subpool_group_idx * subgroup_world_size) % (max_start + 1)
            elastic_subpool_group_idx += 1

            sub_rp = SubRayResourcePool(
                process_on_nodes=elastic_shared_pool.store,
                use_gpu=elastic_shared_pool.use_gpu,
                name_prefix=f"{elastic_shared_pool.name_prefix}_{tag}",
                max_colocate_count=elastic_shared_pool.max_colocate_count,
                detached=elastic_shared_pool.detached,
                accelerator_type=elastic_shared_pool.accelerator_type,
                resource_num_per_bundle=elastic_shared_pool.resource_num_per_bundle,
                placement_groups=elastic_shared_pool.pgs,
                start_bundle_index=start_bundle_index,
                subgroup_world_size=subgroup_world_size,
            )
            end_bundle = start_bundle_index + subgroup_world_size
            pg_ids = [getattr(pg, "id", None) for pg in (elastic_shared_pool.pgs or [])]
            psrl_logger.info(
                "Elastic SubRayResourcePool[%s]: type=%s name_prefix=%s subgroup_world_size=%d "
                "start_bundle_index=%d bundle_range=[%d, %d) shared_world_size=%d "
                "shared_store=%s pg_count=%s pg_ids=%s",
                tag,
                type(sub_rp).__name__,
                sub_rp.name_prefix,
                subgroup_world_size,
                start_bundle_index,
                start_bundle_index,
                end_bundle,
                elastic_shared_pool.world_size,
                list(elastic_shared_pool.store),
                len(elastic_shared_pool.pgs) if elastic_shared_pool.pgs is not None else 0,
                pg_ids,
            )
            return sub_rp

        # create rollout
        for i in range(self.n_rollout_instances):
            if self.elastic_rm_mode:
                rollout_world_size = (
                    self.config.gen_actor_rollout_ref.rollout.tensor_model_parallel_size
                    * self.config.gen_actor_rollout_ref.rollout.pipeline_model_parallel_size
                    * self.config.gen_actor_rollout_ref.rollout.get("data_parallel_size", 1)
                )
                rollout_resource_pool = _build_elastic_sub_resource_pool(
                    subgroup_world_size=rollout_world_size,
                    tag=f"rollout_{i}",
                )
            else:
                rollout_resource_pool = self.resource_pool_manager.get_resource_pool(PSRL_Role.Rollout, i)
            rollout_config = self.config.gen_actor_rollout_ref.rollout
            if self.config.psrl.deployment.heterogeneous_rollout.enable:
                rollout_config.rollout.tensor_model_parallel_size = (
                    self.config.psrl.deployment.heterogeneous_rollout.tensor_model_parallel_size_per_instance[i]
                )
                rollout_config.rollout.pipeline_model_parallel_size = (
                    self.config.psrl.deployment.heterogeneous_rollout.pipeline_model_parallel_size_per_instance[i]
                )

            rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[PSRL_Role.Rollout],
                config=rollout_config,
                model_config=self.config.train_actor_rollout_ref.model,
                device_mesh=None,
            )
            self.resource_pool_to_cls.setdefault(rollout_resource_pool, {})
            self.resource_pool_to_cls[rollout_resource_pool][f"rollout_{i}"] = rollout_cls

        # create validate
        for i in range(self.n_validate_instances):
            val_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(PSRL_Role.Validate, i)
            rollout_config = self.config.train_actor_rollout_ref.rollout
            val_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[PSRL_Role.Validate],
                config=rollout_config,
                model_config=self.config.train_actor_rollout_ref.model,
                device_mesh=None,
            )
            self.resource_pool_to_cls[val_rollout_resource_pool][f"validate_{i}"] = val_rollout_cls

        # create actor
        # AGENT(VERL): PSRL does not use `hybrid_engine`
        train_interface = TrainInterface(ps_manager_handle=self.ps_manager_handle)
        actor_resource_pool = self.resource_pool_manager.get_resource_pool(PSRL_Role.Actor)
        actor_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[PSRL_Role.Actor],
            config=self.config.train_actor_rollout_ref,
            role="actor",
            psrl_config=self.config.psrl,
            train_interface=train_interface,
            distillation_config=self.config.get("distillation", None),
        )
        self.resource_pool_to_cls[actor_resource_pool]["actor"] = actor_cls

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(PSRL_Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            # convert critic_cfg into TrainingWorkerConfig
            from verl.workers.engine_workers import TrainingWorkerConfig

            orig_critic_cfg = critic_cfg
            engine_config: EngineConfig = orig_critic_cfg.engine
            engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
            engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu

            critic_cfg = TrainingWorkerConfig(
                model_type="value_model",
                model_config=orig_critic_cfg.model,
                engine_config=engine_config,
                optimizer_config=orig_critic_cfg.optim,
                checkpoint_config=orig_critic_cfg.checkpoint,
            )

            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[PSRL_Role.Critic],
                config=critic_cfg,
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and PSRL_Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(PSRL_Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[PSRL_Role.RefPolicy],
                config=self.config.train_actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        if not self.use_critic and not self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(PSRL_Role.DummyPolicy)
            dummy_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[PSRL_Role.DummyPolicy],
                config=self.config.train_actor_rollout_ref,
                role="dummy",
            )
            self.resource_pool_to_cls[resource_pool]["dummy"] = dummy_policy_cls

        # Allocate resource pools for reward model replicas.
        for reward_model in self.config.reward.reward_models:
            if reward_model.reward_loop_type != "gen":
                continue
            reward_model_name = reward_model.get("reward_model_name", reward_model.model.path.split("/")[-1])
            reward_model_cfg = reward_model
            for i in range(reward_model.num_replicas):
                if self.elastic_rm_mode:
                    reward_model_world_size = (
                        reward_model_cfg.rollout.tensor_model_parallel_size
                        * reward_model_cfg.rollout.pipeline_model_parallel_size
                        * reward_model_cfg.rollout.get("data_parallel_size", 1)
                    )
                    reward_model_resource_pool = _build_elastic_sub_resource_pool(
                        subgroup_world_size=reward_model_world_size,
                        tag=f"reward_model_{reward_model_name}_{i}",
                    )
                else:
                    reward_model_resource_pool = self.resource_pool_manager.resource_pool_dict[
                        f"reward_pool_{reward_model_name}_{i}"
                    ]
                reward_cls = RayClassWithInitArgs(
                    cls=self.role_worker_mapping[PSRL_Role.RewardModel],
                    config=reward_model_cfg.rollout,
                    model_config=reward_model_cfg.model,
                    device_mesh=None,
                )
                self.resource_pool_to_cls[reward_model_resource_pool][f"reward_model_{reward_model_name}_{i}"] = (
                    reward_cls
                )

        # ---- Step 2: Create worker groups for each role in parallel ----
        psrl_logger.info("Initializing WorkerGroup for other roles")

        # initialize WorkerGroup
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
        reward_model_tasks = []
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
            elif any("reward_model" in key for key in class_dict.keys()):
                assert len(class_dict) == 1, "Reward model resource pool should only have one worker class."
                reward_model_tasks.append((resource_pool, class_dict, wg_kwargs))
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
        _run_worker_group_tasks(reward_model_tasks, label="reward_model")
        _run_worker_group_tasks(gen_tasks, label="gen")
        _run_worker_group_tasks(val_tasks, label="validate")

        # Initialize reward model gateways and servers
        self.init_reward_gateways()
        self.init_reward_model_servers(all_wg)

        self.rollout_wg_list = [all_wg[f"rollout_{i}"] for i in range(self.n_rollout_instances)]
        self.validate_wg_list = [all_wg[f"validate_{i}"] for i in range(self.n_validate_instances)]
        self.reward_model_to_wg_list = {}
        for reward_model in self.config.reward.reward_models:
            if reward_model.reward_loop_type != "gen":
                continue
            reward_model_name = reward_model.get("reward_model_name", reward_model.model.path.split("/")[-1])
            self.reward_model_to_wg_list[reward_model_name] = [
                all_wg[f"reward_model_{reward_model_name}_{i}"] for i in range(reward_model.num_replicas)
            ]

        # initialize rollout router for registration
        self.init_rollout_router()

        # create agent loop workers
        rollout_router = self.rollout_gateway_url if self.config.psrl.rollout_gateway.enable else self.rollout_router
        self.agent_loop_workers = []
        max_concurrency_per_worker = (
            self.max_concurrency // self.config.gen_actor_rollout_ref.rollout.agent.num_workers
        )
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
                        {
                            WorkerKey("rollout", i, idx): node_id
                            for idx, node_id in enumerate(rollout_instance_node_ids)
                        }
                    )

                # Get all actor instances' distinct node ids
                actor_instance_node_ids = all_wg["actor"].execute_all_sync("get_node_id")
                for node_id in actor_instance_node_ids:
                    ps_node_ids.add(node_id)
                self.worker_to_node_id.update(
                    {WorkerKey("actor", 0, idx): node_id for idx, node_id in enumerate(actor_instance_node_ids)}
                )

                # Get all validate instances' distinct node ids
                for i in range(self.n_validate_instances):
                    validate_instance_node_ids = all_wg[f"validate_{i}"].execute_all_sync("get_node_id")
                    for node_id in validate_instance_node_ids:
                        ps_node_ids.add(node_id)
                    self.worker_to_node_id.update(
                        {
                            WorkerKey("validate", i, idx): node_id
                            for idx, node_id in enumerate(validate_instance_node_ids)
                        }
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
                # NOTE(claude): dispatch preload immediately into each PS actor's serial queue; it will
                # start as soon as that actor's init_model completes, overlapping with gen/val/train
                # model initialization and NIXL protocol to hide disk I/O latency.
                preload_futures = self.ps_wg.execute_all_async("preload_checkpoint_to_cpu")
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
            self.critic_wg.reset()
            # assign critic loss
            from functools import partial

            from verl.workers.utils.losses import value_loss

            value_loss_ = partial(value_loss, config=orig_critic_cfg)
            self.critic_wg.set_loss_fn(value_loss_)

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if not (self.use_critic or (self.use_reference_policy and not self.ref_in_actor)):
            # NOTE(linsh): when not using critic or reference policy,
            # if we directly call `init_model` of actor_wg, Ray will view the fused worker
            # as an async actor and run `run_async_func_or_coro_in_event_loop`, which will
            # make it invalid to call async function such as `trainer_mode` in `init_model`.
            # So here we create a dummy worker group and call its dummy method to avoid this issue.
            self.dummy_wg = all_wg["dummy"]
            self.dummy_wg.init_model()

        # TODO(linsh): check OPD implementation
        # initialize teacher loop manager
        if self.use_teacher_policy:
            from verl.experimental.teacher_loop import TeacherModelManager

            teacher_resource_pool = self.resource_pool_manager.get_resource_pool(PSRL_Role.TeacherModel)
            self.teacher_model_manager = TeacherModelManager(
                config=self.config.distillation,
                resource_pool=teacher_resource_pool,
            )
            self.distillation_config: DistillationConfig = omega_conf_to_dataclass(self.config.distillation)
        else:
            self.teacher_model_manager = None
            self.distillation_config = None

        # AGENT(VERL): skip `async_rollout_manager`, `checkpoint_manager` in PSRL.

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
            self.actor_wg.init_model("empty")
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
            # write the preloaded checkpoint tensors into the PS registered buffers.
            with log_dual_events("Loading PS checkpoint weights", psrl_logger, event_type=EventType.INIT):
                # Ensure prefetch finished (likely already done; blocks only if preload
                # outlasted NIXL protocol, which would be unusual).
                ray.get(preload_futures)
                ray.get(self.ps_wg.execute_all_async("write_checkpoint_to_registered_tensors"))

            # Bind the PS worker group to PSManager before initial pull, so that PSManager's
            # ps_nixl_agent_names are populated and gen workers can call get_ps_nixl_agent_names.
            psrl_logger.info("Binding PS worker group")
            ray.get(self.ps_manager_handle.bind_ps_worker_group.remote(self.ps_wg))
            psrl_logger.info("PS worker group bound successfully!")

            # Pull checkpoint weights from PS into gen and actor workers via NIXL.
            # NOTE(lhy): gen/val/actor workers are now empty-initialized and must pull from PS on startup.
            # When is_rollout_mode_in_actor is True, the actor is sleeping (meta mode) at this point
            # and will pull from PS later in switch_to_trainer_mode, so skip here.
            initial_pull_tag = "all" if self.is_rollout_mode_in_actor else "rollout"
            with log_dual_events("Initial pull: PS → gen/actor workers", psrl_logger, event_type=EventType.INIT):
                initial_pull_futures = []
                initial_pull_futures.append(self.rollout_coordinator.initial_pull_from_ps.remote(initial_pull_tag))
                if not self.is_rollout_mode_in_actor:
                    initial_pull_futures.extend(self.actor_wg.execute_all_async("pull_model"))
                ray.get(initial_pull_futures)

        self.init_elastic_rm_runtime()

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
            tp_size = sum(1 for k in self.worker_to_ps_idx if k.role == "validate" and k.instance_id == i)
            for rank in range(tp_size):
                updated_client_names.append(
                    WorkerKey("validate", i, rank).to_nixl_client_name(self.n_rollout_instances)
                )
            futures.append(
                self.tag_to_server_handles["validate"][i].nixl_send_local_info_to.remote(NIXL_META_SERVER_NAME)
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
            update_client_names.append(train_client_name(i))
        # sender side: actor workers
        futures.extend(self.actor_wg.execute_all_async("nixl_send_local_info_to", NIXL_META_SERVER_NAME))
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
        src_agent_names = [NIXL_META_SERVER_NAME]
        dst_agent_names = []
        # 1. PS storage workers
        for i in range(self.ps_wg.world_size):
            dst_agent_names.append(ps_agent_name(i))
        # 2. rollout workers
        for i in range(self.n_rollout_instances):
            tp_size = sum(1 for k in self.worker_to_ps_idx if k.role == "rollout" and k.instance_id == i)
            for rank in range(tp_size):
                dst_agent_names.append(WorkerKey("rollout", i, rank).to_nixl_client_name())
        # 3. validate workers
        for i in range(self.n_validate_instances):
            tp_size = sum(1 for k in self.worker_to_ps_idx if k.role == "validate" and k.instance_id == i)
            for rank in range(tp_size):
                dst_agent_names.append(WorkerKey("validate", i, rank).to_nixl_client_name(self.n_rollout_instances))
        # 4. actor workers
        for i in range(self.actor_wg.world_size):
            dst_agent_names.append(train_client_name(i))
        psrl_logger.debug(f"Destination agent names for broadcasting: {dst_agent_names}")

        broadcast_plan = self._make_broadcast_plan(src_agent_names, dst_agent_names)
        futures = []
        for src_agent, dst_agents in broadcast_plan.items():
            if src_agent == NIXL_META_SERVER_NAME:
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

        # AGENT(VERL): PSRL use `actor_wg` while VERL use `actor_rollout_wg`.

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
        # AGENT(VERL): dataloader state is saved in `data_processor` in PSRL.
        ray.get(self.data_processor.save_train_dataloader.remote(dataloader_local_path))

        # AGENT(VERL): PSRL use `config.train_actor_rollout_ref` while VERL use `config.actor_rollout_ref.actor`.
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
        # AGENT(VERL): PSRL use `actor_wg` while VERL use `actor_rollout_wg`.
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
            # AGENT(VERL): dataloader state is saved in `data_processor` in PSRL.
            ray.get(self.data_processor.load_train_dataloader.remote(dataloader_local_path))
        else:
            psrl_logger.info(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        # AGENT(VERL): PSRL use `actor_wg` while VERL use `actor_rollout_wg`.
        if do_profile:
            self.actor_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        # AGENT(VERL): PSRL use `actor_wg` while VERL use `actor_rollout_wg`.
        if do_profile:
            self.actor_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_wg, "actor")

        # Use group-level balancing for PrefixGrouper to keep same-uid samples together
        # AGENT(VERL): PSRL use `parent_id` to group samples from the same episode,
        # while VERL use `uid` for the same purpose.
        if getattr(self, "use_prefix_grouper", False) and "parent_id" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["parent_id"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.train_actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        tu.assign_non_tensor(batch_td, compute_loss=False)
        output = self.critic_wg.infer_batch(batch_td)
        output = output.get()
        values = tu.get(output, "values")
        values = no_padding_2_padding(values, batch_td)
        values = tu.get_tensordict({"values": values.float()})
        values = DataProto.from_tensordict(values)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        # step 1: convert dataproto to tensordict.
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        metadata = {"calculate_entropy": False, "compute_loss": False}
        if self.ref_in_actor:
            metadata["no_lora_adapter"] = True
        tu.assign_non_tensor(batch_td, **metadata)
        if self.ref_in_actor:
            output = self.actor_wg.compute_log_prob(batch_td)
        else:
            output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
        # gather output
        log_probs = tu.get(output, "log_probs")
        # step 4. No padding to padding
        log_probs = no_padding_2_padding(log_probs, batch_td)
        # step 5: rebuild a tensordict and convert to dataproto
        ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
        ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
        # step 1: convert dataproto to tensordict.
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
        output = self.actor_wg.compute_log_prob(batch_td)
        # gather output
        entropy = tu.get(output, "entropy")
        log_probs = tu.get(output, "log_probs")
        routed_experts = tu.get(output, "routed_experts")

        old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
        # step 4. No padding to padding
        entropy = no_padding_2_padding(entropy, batch_td)
        log_probs = no_padding_2_padding(log_probs, batch_td)
        # step 5: rebuild a tensordict and convert to dataproto
        if routed_experts is not None:
            old_log_prob = tu.get_tensordict(
                {"old_log_probs": log_probs.float(), "entropys": entropy.float(), "routed_experts": routed_experts}
            )
        else:
            old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
        old_log_prob = DataProto.from_tensordict(old_log_prob)
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.gen_actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to no-padding
        batch_td = left_right_2_no_padding(batch_td)
        calculate_entropy = self.config.train_actor_rollout_ref.actor.entropy_coeff != 0.0
        distillation_use_topk = (
            self.distillation_config.distillation_loss.loss_settings.use_topk
            if is_distillation_enabled(self.config.get("distillation"))
            else False
        )
        ppo_mini_batch_size = self.config.train_actor_rollout_ref.actor.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.gen_actor_rollout_ref.rollout.n
        ppo_epochs = self.config.train_actor_rollout_ref.actor.ppo_epochs
        seed = self.config.train_actor_rollout_ref.actor.data_loader_seed
        shuffle = self.config.train_actor_rollout_ref.actor.shuffle
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=calculate_entropy,
            distillation_use_topk=distillation_use_topk,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
            compute_loss=True,
        )
        actor_output = self.actor_wg.update_actor(batch_td)
        actor_output = tu.get(actor_output, "metrics")
        actor_output = rename_dict(actor_output, "actor/")
        # modify key name
        actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
        actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})

        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to no-padding
        batch_td = left_right_2_no_padding(batch_td)
        ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.gen_actor_rollout_ref.rollout.n
        ppo_epochs = self.config.critic.ppo_epochs
        seed = self.config.critic.data_loader_seed
        shuffle = self.config.critic.shuffle
        tu.assign_non_tensor(
            batch_td,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
        )

        output = self.critic_wg.train_mini_batch(batch_td)
        output = output.get()
        output = tu.get(output, "metrics")
        output = rename_dict(output, "critic/")
        # modify key name
        output["perf/mfu/critic"] = output.pop("critic/mfu")
        critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        return critic_output

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
        # AGENT(VERL): ignore checkpoint manager in PSRL

        # AGENT(VERL): earlier check in PSRL
        if self.global_steps >= self.total_training_steps:
            psrl_logger.warning(
                f"Global steps {self.global_steps} >= total training steps {self.total_training_steps}, "
                "skipping training."
            )
            return

        # AGENT(VERL): PSRL specific initialization process
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

        # AGENT(VERL): continue verl's fit logics

        # perform validation before training
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            psrl_logger.info(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # AGENT(VERL): skip RolloutSkip part in PSRL

        # Start data processor to handle data preprocessing and batching
        # AGENT(VERL): PSRL specific data processor start
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
        # AGENT(VERL): PSRL use a busy loop for training,
        # while verl use epoch and iteration based loop.
        while True:
            if hasattr(self.train_actor_rollout_wg, "async_calls_finalize_fn_exec"):
                self.train_actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
            metrics = {}
            timing_raw = {}
            is_last_step = self.global_steps == self.total_training_steps

            with marked_timer("step", timing_raw):
                # Wait for the training batch to be ready
                # AGENT(VERL): wait for train batch first in PSRL's async RL.
                # verl will handle gen batch processing and generation here.
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
                        with log_dual_events("Switch to trainer mode", psrl_logger, event_type=EventType.SWITCH):
                            self.switch_to_trainer_mode()
                    else:
                        # NOTE(linsh): this code snippet is not actively maintained.
                        from verl.trainer.ppo.reward import compute_reward

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
                if self._should_compute_teacher_colocate(batch):
                    with marked_timer("teacher", timing_raw, color="cyan"):
                        batch_teacher = self._compute_teacher_colocate(batch)
                        batch = batch.union(batch_teacher)

                if "response_mask" not in batch.batch.keys():
                    batch.batch["response_mask"] = compute_response_mask(batch)
                # Balance the number of valid tokens across DP ranks.
                # NOTE: This usually changes the order of data in the `batch`,
                # which won't affect the advantage calculation (since it's based on uid),
                # but might affect the loss calculation (due to the change of mini-batching).
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
                if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                    from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                    apply_bypass_mode(
                        batch=batch,
                        rollout_corr_config=rollout_corr_config,
                        policy_loss_config=self.config.train_actor_rollout_ref.actor.policy_loss,
                    )
                else:
                    # recompute log_probs in the training side
                    with marked_timer("old_log_prob", timing_raw, color="orange"):
                        with log_dual_events(
                            "Recompute log_prob on training side",
                            psrl_logger,
                            event_type=EventType.OTHER,
                        ):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.train_actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                raise ValueError(
                                    "Detected conflicting router replay configuration: "
                                    "router_replay.mode='R2' and enable_rollout_routing_replay=True "
                                    "cannot be enabled simultaneously. "
                                    "The enable_rollout_routing_replay option is only used in R3 mode; "
                                    "it should not be set when using R2 mode."
                                )
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                if self.use_reference_policy:
                    # compute reference log_prob
                    with marked_timer("ref", timing_raw, color="olive"):
                        with log_dual_events(
                            "Compute reference log_prob",
                            psrl_logger,
                            event_type=EventType.OTHER,
                        ):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                # compute values
                if self.use_critic:
                    with marked_timer("values", timing_raw, color="cyan"):
                        with log_dual_events(
                            "Compute critic values",
                            psrl_logger,
                            event_type=EventType.OTHER,
                        ):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                # AGENT(VERL): PSRL specific reward computation logic.
                if self.config.reward.launch_reward_fn_async:
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
                            reward_metrics_dict_list = []
                            rm_generated_token_nums = []
                            for request_id in request_ids:
                                reward_score = request_id_to_reward[request_id]["reward_score"]
                                extra_info = request_id_to_reward[request_id].get("reward_extra_info", {})
                                reward_metrics = request_id_to_reward[request_id].get("reward_metrics", {})
                                scores.append(reward_score)
                                reward_extra_infos_dict_list.append(extra_info)
                                reward_metrics_dict_list.append(reward_metrics)
                                rm_generated_token_num = extract_gen_rm_token_num(extra_info)
                                rm_generated_token_nums.append(rm_generated_token_num)

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
                                    if not isinstance(value, list) and key in ("data_source", "original_reward_score"):
                                        value = [value]
                                    reward_extra_infos_dict[key].extend(value)
                                reward_extra_infos_dict["reward_extra_info"].append(reward_extra_infos)
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                            global_token_num = batch.meta_info.get("global_token_num")
                            if isinstance(global_token_num, list) and len(global_token_num) == len(
                                rm_generated_token_nums
                            ):
                                batch.meta_info["global_token_num"] = [
                                    int(token_num) + int(rm_token_num)
                                    for token_num, rm_token_num in zip(global_token_num, rm_generated_token_nums)
                                ]
                            else:
                                psrl_logger.warning(
                                    "Skip merging reward model token count to global_token_num due to shape mismatch: "
                                    "global_token_num=%s rm_generated_token_nums=%s",
                                    type(global_token_num),
                                    len(rm_generated_token_nums),
                                )
                else:
                    reward_tensor = batch.batch.pop("rm_scores", None)

                batch.batch["token_level_scores"] = reward_tensor

                metrics_logging_path = self.config.psrl.get("logging_path", None)
                metrics_output_path = (
                    os.path.join(metrics_logging_path, "rollout_rm_metrics.jsonl")
                    if metrics_logging_path is not None
                    else None
                )
                record_rollout_rm_metrics(batch, output_path=metrics_output_path)

                with marked_timer("adv", timing_raw, color="brown"):
                    with log_dual_events("Compute advantage", psrl_logger, event_type=EventType.OTHER):
                        # AGENT(VERL): reward combine is moved.

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

                        # AGENT(VERL): PSRL specific debug logging.
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
                            critic_output = self._update_critic(batch)
                    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    metrics.update(critic_output_metrics)

                # implement critic warmup
                if self.config.trainer.critic_warmup <= self.global_steps:
                    # update actor
                    with marked_timer("update_actor", timing_raw, color="red"):
                        with log_dual_events("Update actor", psrl_logger, event_type=EventType.TRAIN):
                            actor_output = self._update_actor(batch)

                    # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    # Check if the conditions for saving a checkpoint are met.
                    # The conditions include a mandatory condition (1) and
                    # one of the following optional conditions (2/3/4):
                    # 1. The save frequency is set to a positive value.
                    # 2. It's the last training step.
                    # 3. The current step number is a multiple of the save frequency.
                    # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            with log_dual_events("Save checkpoint", psrl_logger, event_type=EventType.OTHER):
                                self._save_checkpoint()

                    # AGENT(VERL): Skip checkpoint manager in PSRL.

                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                # Log rollout generations if enabled
                rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                if rollout_data_dir:
                    self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        with log_dual_events("Validate", psrl_logger, event_type=EventType.VAL):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                    metrics.update(val_metrics)

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
                    # AGENT(VERL): no epoch metrics in PSRL.
                }
            )
            # collect metrics
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            # GDPO per-component reward metrics
            gdpo_reward_keys = self.config.algorithm.get("gdpo_reward_keys", None)
            if gdpo_reward_keys and self.config.algorithm.adv_estimator in ("gdpo", AdvantageEstimator.GDPO):
                for key in gdpo_reward_keys:
                    if key in batch.non_tensor_batch:
                        vals = np.asarray(batch.non_tensor_batch[key], dtype=np.float32)
                        metrics[f"gdpo/{key}/mean"] = float(np.mean(vals))
                        metrics[f"gdpo/{key}/std"] = float(np.std(vals))
                        metrics[f"gdpo/{key}/max"] = float(np.max(vals))
                        metrics[f"gdpo/{key}/min"] = float(np.min(vals))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            # TODO(verl): implement actual tflpo and theoretical tflpo
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
            # compute variance proxy metrics
            gradient_norm = metrics.get("actor/grad_norm", None)
            metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))

            # AGENT(VERL): skip curriculum sampler processing here in PSRL.

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
                if hasattr(self.actor_wg, "async_calls_finalize_fn_exec"):
                    self.actor_wg.async_calls_finalize_fn_exec(blocking=True)
                psrl_logger.info(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                break

        # Stop all components
        self.stop_data_processor()
        self.stop_agent_loop_manager()
        self.stop_rollout_coordinator()
        self.stop_reward_manager()
        if self.elastic_executor is not None:
            ray.get(self.elastic_executor.stop_busy_loop.remote())
            self.elastic_executor = None
        self.stop_ps_manager()

        # AGENT(VERL): skip `on_batch_end` processing in PSRL.

        psrl_logger.info("Training completed successfully!")
