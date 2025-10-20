import os
import uuid
import torch
import logging
import asyncio
import numpy as np
from collections import defaultdict
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import ray
from ray.exceptions import RayTaskError
from ray.util.queue import Queue as RayQueue
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls_fused
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, apply_kl_penalty, compute_response_mask, RayPPOTrainer
from verl.trainer.ppo.utils import WorkerType
from verl.trainer.ppo.core_algos import agg_loss
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.metric import reduce_metrics
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.tracking import ValidationGenerationsLogger
from verl.utils.debug import marked_timer

from psrl.utils.dataset import DatasetType, DataProcessor
from psrl.utils.logger import DualOutputHandler, log_dual_events, log_single_event, EventType, log_data_protocol
from psrl.utils.nixl import NIXLInterface, GLOBAL_PORT_SCANNER
from psrl.workers.train import TrainInterface
from psrl.workers.gen import GenInterface, RolloutCoordinator
from psrl.workers.reward import RewardServer
from psrl.workers.ps import PSManager, PSWorkerGroup, PSResourceSpec, PSResourcePool, PSClassWithInitArgs, PSStoragePlan, PSStorageWorker
from psrl.workers.agent_loop import PSRL_AgentLoopManager, PSRL_AgentLoopWorker
from psrl.trainer.ppo.utils import PSRL_Role, PSRL_ResourcePoolManager, PSRL_compute_advantage, need_critic, need_reference_policy, need_reward_model

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

class PSRL_RayPPOTrainer(RayPPOTrainer):
    
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[PSRL_Role, WorkerType],
        resource_pool_manager: PSRL_ResourcePoolManager,
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
            resource_pool_manager (PSRL_ResourcePoolManager): Manager for Ray resources.
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
        self.ray_worker_group_cls = ray_worker_group_cls # NOTE(lhy): ray_worker_group_cls is used only in train side
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )
        
        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.train_actor_rollout_ref.model.get("lora_rank", 0) > 0
        
        # CPU workers for Streaming Rollout
        self.data_processor = None
        self.agent_loop_manager = None
        self.rollout_coordinator = None
        self.reward_server = None
        
        # Parameter server handle for other workers to access
        self.ps_manager_handle = None
        
        # Async rollout mode for training worker
        self.async_rollout_mode = False

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        # Build logger
        self.log_prefix = f"MainRayTrainer"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))
        psrl_logger.info(f"Initialized major ray trainer (single controller).")
        
        self._initialize_queue_buffers()

        self._validate_config()
        
        self._init_ps_manager()
        
        # initialize data processor
        # NOTE(lhy): data processor must be initialized before initializing other workers
        # so that the total_training_steps can be obtained and the optimizer config (related to weight decay, lr schedule, etc.) can be set
        # otherwise, it will cause error when running Megatron backend
        self._init_data_processor()

    def _initialize_queue_buffers(self):
        self.process_mode = self.config.psrl.gen_mode
        if self.config.psrl.redundant_rollout.enable:
            self.rollout_n = self.config.psrl.redundant_rollout.redundant_rollout_n
            self.alg_rollout_n = self.config.psrl.redundant_rollout.alg_rollout_n
        else:
            self.rollout_n = self.config.gen_actor_rollout_ref.rollout.n
            self.alg_rollout_n = self.rollout_n
        assert self.rollout_n >= self.alg_rollout_n, \
            f"Rollout n {self.rollout_n} must be greater than or equal to alg_rollout_n {self.alg_rollout_n}."
        
        # Data queue is the communication handle between the data processor and the rollout server.
        # It holds the data batches that are ready for processing.
        # The size of the queue is determined by the batch size and the process mode.
        # If process_mode is "stream", it will hold multiple requests for streaming processing.
        # If process_mode is "batch", it will hold a single batch for batch processing.      
        # Rollout queue is the communication handle between the rollout workers and the data processor (reward module).
        # It holds the rollout data that is ready for reward computation.
        # The size of the queue is the same as the whole batch size for streaming mode.
        # The size of the queue is the same as number of agent workers for batch mode.
        if self.process_mode == "stream":
            self.data_queue_size = self.config.data.get("gen_batch_size", self.config.data.train_batch_size) * self.rollout_n
            self.rollout_queue_size = self.data_queue_size
        else:
            self.data_queue_size = 1 
            self.rollout_queue_size = self.config.gen_actor_rollout_ref.rollout.get("agent", {}).get("num_workers", 1)

        # Status queue is used to store the status of the rollout workers.
        # The status is collected by the rollout coordinator and sent to the agent loop workers.
        self.status_queue = RayQueue()

        psrl_logger.debug("Initialized data_queue, rollout_queue, and status_queue with sizes: %d, %d, and unlimited respectively.",
                         self.data_queue_size, self.rollout_queue_size)

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
            assert train_n_gpus % (model_parallel_size * context_parallel_size) == 0, \
                f"train_n_gpus ({train_n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times" \
                f" context_parallel_size ({context_parallel_size})"
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
                config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic"
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
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
            assert config.train_actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.train_actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.train_actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )
        
        # Check NIXL compatibility
        if self.config.psrl.ps_mode == "nixl_cpu" or self.config.psrl.ps_mode == "nixl_gpu":
            assert self.config.psrl.nixl.server_ip == self.config.psrl.ps_manager_ip, "PSManager IP and NIXL server IP must be the same"
            assert self.config.train_actor_rollout_ref.actor.strategy != "fsdp", "FSDP1 is not supported for NIXL because it uses flat_param"
            psrl_logger.info(f"NOTICE: NIXL is enabled. Actor strategy used is {self.config.train_actor_rollout_ref.actor.strategy}")
            
        # Check log_prob mode
        if self.config.psrl.log_prob.mode == "rollout":
            assert self.config.psrl.log_prob.enable_rollout_engine_log_prob, "enable_rollout_engine_log_prob must be set when using rollout log_prob"
        elif self.config.psrl.log_prob.mode == "recompute":
            assert self.config.psrl.log_prob.enable_train_engine_recompute_log_prob, "enable_train_engine_recompute_log_prob must be set when using recompute log_prob"
        elif self.config.psrl.log_prob.mode == "tis":
            assert self.config.psrl.log_prob.enable_rollout_engine_log_prob and self.config.psrl.log_prob.enable_train_engine_recompute_log_prob, "enable_rollout_engine_log_prob and enable_train_engine_recompute_log_prob must be set when using TIS log_prob"
        else:
            raise ValueError(f"Invalid log_prob mode: {self.config.psrl.log_prob.mode}, must be one of ['rollout', 'recompute', 'tis']")

        # Check colocate mode
        if self.config.psrl.colocate:
            assert self.config.psrl.gen_mode == "batch", "gen_mode must be batch when using colocate mode"
            assert self.config.psrl.staleness == 0, "staleness must be 0 when using colocate mode"
            
        # Check rollout mode
        if self.config.psrl.gen_mode == "batch":
            assert self.config.gen_actor_rollout_ref.rollout.mode == "sync", "rollout mode must be sync when using batch mode"
        elif self.config.psrl.gen_mode == "stream":
            assert self.config.gen_actor_rollout_ref.rollout.mode == "psrl_async", "rollout mode must be async when using stream mode"
        else:
            raise ValueError(f"Invalid gen_mode: {self.config.psrl.gen_mode}, must be one of ['batch', 'stream']")

        psrl_logger.info("[validate_config] All configuration checks passed successfully!")
    
    def _init_ps_manager(self):
        """Initialize the PS manager for handling model version, requests condition and staleness."""
        ip_to_node_id = {node['NodeManagerAddress']: node['NodeID'] for node in ray.nodes()}
        assert self.config.psrl.ps_manager_ip in ip_to_node_id, f"PSManager IP {self.config.psrl.ps_manager_ip} not found in ray nodes"
        psrl_logger.info("Getting the handle of the PSManager")
        self.ps_manager_handle = ray.remote(PSManager).options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=ip_to_node_id[self.config.psrl.ps_manager_ip],
                soft=False
            )
        ).remote(self.config.psrl)
    
    def _init_data_processor(self):
        """Initialize the data processor for handling data preprocessing and batching."""
        if self.data_processor is not None:
            return
        
        # Initialize the data processor
        self.data_processor = DataProcessor.remote(
            self.config,
            self.tokenizer,
            self.processor,
            self.ps_manager_handle,
            collate_fn=self.collate_fn,
            process_mode=self.process_mode,
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
        self.agent_loop_manager = ray.remote(PSRL_AgentLoopManager).remote(
            self.config,
            self.data_queue_size,
            self.agent_loop_workers,
            self.ps_manager_handle,
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

    def init_rollout_coordinator(self):
        self.rollout_coordinator = RolloutCoordinator.remote(
            self.config,
            self.ps_manager_handle,
            self.rollout_wg_list,
            self.agent_loop_workers,
            self.status_queue,
        )

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

    def init_reward_server(self):
        """Initialize the reward server for computing rewards during training."""
        assert self.data_processor is not None, "Data processor must be initialized before starting reward computation."
        assert self.rollout_coordinator is not None, "Rollout server must be initialized before starting reward computation."
        
        self.reward_server = ray.remote(RewardServer).remote(
            self.config,
            self.tokenizer,
            self.processor,
            self.ps_manager_handle,
            self.agent_loop_manager,
            self.rollout_queue_size,
            reward_fn=self.reward_fn,
            use_rm=self.use_rm,
            group_post_process_fn=self.group_post_process_fn,
            buffer_post_process_fn=self.buffer_post_process_fn,
        )

    def start_reward_server(self):
        """Start the reward server to handle reward computation requests in the background."""
        assert self.reward_server is not None, "Reward server must be initialized before starting it."
        
        ray.get(self.reward_server.start_busy_loop.remote())

    def stop_reward_server(self):
        """Stop the reward server."""
        if self.reward_server is not None:
            psrl_logger.debug("Stopping reward server...")
            ray.get(self.reward_server.stop_busy_loop.remote())
            self.reward_server = None
            psrl_logger.debug("Reward server stopped successfully.")
        else:
            psrl_logger.warning("Reward server is not initialized, skipping stop operation.")

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

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        # TODO: check it
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self):
        """Validate the model using the validation dataset.
        
        Note that we use the training side to do val for overlapping with generation.
        """
        psrl_logger.debug("Starting validation process")
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_parent_ids = []

        batch_count = 0
        while True:
            try:
                test_data = ray.get(self.data_processor.get_single_controller_batch.remote(DatasetType.val))
                batch_count += 1
            except RayTaskError as e:
                if isinstance(e.cause, StopIteration):
                    psrl_logger.debug("Reached end of validation dataset after %d batches", batch_count)
                    break
                else:
                    psrl_logger.error(f"Unknown exception happened during obtaining validation data: {type(e.cause)}")
                    raise
            test_batch = DataProto.from_single_dict(test_data)

            if "parent_id" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["parent_id"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.train_actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO(verl): Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_parent_ids.extend(test_batch.non_tensor_batch["parent_id"])

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
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.train_actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            psrl_logger.debug(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_wg.world_size // self.config.train_actor_rollout_ref.rollout.tensor_model_parallel_size
                if not self.async_rollout_mode
                else self.config.train_actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            # switch to the inference engine and generate sequences
            # NOTE: `async_rollout_mode` regards to aysnc engine in verl, not the async rollout mode in PSRL as `psrl_async`.
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
            
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

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

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

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

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.
        
        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        
        Note that we use multi-threading to speed up the initialization of worker groups.
        For rollout instances, we create multiple worker groups based on the number of instances specified in the configuration,
        instead of creating a unified worker group for all instances.
        """
        
        self.resource_pool_manager.create_resource_pool()
        
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}
        
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        # create rollout, actor and ps
        # PS need to be created before rollout and actor to pass the ps_manager_handle
        assert (
            PSRL_Role.Rollout in self.role_worker_mapping and
            PSRL_Role.Actor in self.role_worker_mapping
        ), "Rollout and Actor must be in role_worker_mapping." 
        
        # create nixl interface
        nixl_interface = NIXLInterface(
            port_scanner=GLOBAL_PORT_SCANNER
        )
        
        # create rollout instances  
        for i in range(self.config.psrl.deployment.n_rollout_instances):
            gen_interface = GenInterface(
                rollout_instance_id=i,
                ps_manager_handle=self.ps_manager_handle,
                status_queue=self.status_queue,
            )
            rollout_config = self.config.gen_actor_rollout_ref
            if self.config.psrl.deployment.heterogeneous_rollout.enable:
                rollout_config.rollout.tensor_model_parallel_size = self.config.psrl.deployment.heterogeneous_rollout.tensor_model_parallel_size_per_instance[i]
                rollout_config.rollout.pipeline_model_parallel_size = self.config.psrl.deployment.heterogeneous_rollout.pipeline_model_parallel_size_per_instance[i]

            rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[PSRL_Role.Rollout],
                config=rollout_config,
                role='rollout',
                psrl_config=self.config.psrl,
                gen_interface=gen_interface,
                nixl_interface=nixl_interface,
            )
            rollout_resource_pool = self.resource_pool_manager.get_resource_pool(PSRL_Role.Rollout, i)
            self.resource_pool_to_cls[rollout_resource_pool][f"rollout_{i}"] = rollout_cls  
        
        # create actor (train only) 
        train_interface = TrainInterface(
            ps_manager_handle=self.ps_manager_handle
        )   
        actor_resource_pool = self.resource_pool_manager.get_resource_pool(PSRL_Role.Actor)
        actor_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[PSRL_Role.Actor],
            config=self.config.train_actor_rollout_ref,
            role='actor_rollout', # also need rollout for validation set
            psrl_config=self.config.psrl,
            train_interface=train_interface,
            nixl_interface=nixl_interface,
        )
        self.resource_pool_to_cls[actor_resource_pool]["actor"] = actor_cls

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(PSRL_Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[PSRL_Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(PSRL_Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[PSRL_Role.RefPolicy], config=self.config.train_actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        self.rm_wg = None
        if self.use_rm:
            resource_pool = self.resource_pool_manager.get_resource_pool(PSRL_Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[PSRL_Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        if not self.use_critic and not self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(PSRL_Role.DummyPolicy)
            dummy_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[PSRL_Role.DummyPolicy], config=self.config.train_actor_rollout_ref, role="dummy")
            self.resource_pool_to_cls[resource_pool]["dummy"] = dummy_policy_cls

        # initialize WorkerGroup
        psrl_logger.info("Initializing WorkerGroup for other roles")
        # NOTE(verl): if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls_fused`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        def create_worker_group(resource_pool, class_dict):
            # if there is only one worker class in the resource pool, we can directly create a worker group
            # so that we can use 'execute_all_async' and other low-level APIs
            # NOTE(lhy): in newest verl, we can use `create_colocated_worker_cls_fused` to create a fused worker group and low-level APIs can also be used
            if len(class_dict) == 1:
                role = next(iter(class_dict.keys()))
                if "rollout" in role:
                    return {role: RayWorkerGroup(
                        resource_pool=resource_pool,
                        ray_cls_with_init=class_dict[role],
                        **wg_kwargs
                    )}
                return {role: self.ray_worker_group_cls(
                    resource_pool=resource_pool,
                    ray_cls_with_init=class_dict[role],
                    **wg_kwargs
                )}
            # colocate
            else:
                worker_dict_cls = create_colocated_worker_cls_fused(class_dict=class_dict)
                wg_dict = self.ray_worker_group_cls(
                    resource_pool=resource_pool,
                    ray_cls_with_init=worker_dict_cls,
                    **wg_kwargs
                )
                return wg_dict.spawn(prefix_set=class_dict.keys())
        
        # coroutine version
        '''
        async def async_create_worker_groups():
            tasks = []
            for resource_pool, class_dict in self.resource_pool_to_cls.items():
                psrl_logger.info(f"Creating worker group for resource pool: {resource_pool}, classes: {class_dict}")
                if "ps" in class_dict:
                    assert class_dict.keys() == {"ps"}, "PS resource pool should only have PS role."
                    continue
                tasks.append((resource_pool, class_dict))
            
            async def create_single_worker_group(resource_pool, class_dict):
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = loop.run_in_executor(executor, create_worker_group, resource_pool, class_dict)
                    return await future
            
            # Concurrency control within max_concurrent tasks at a time
            # NOTE(lhy): currently set to 1 (the default sync version) to avoid the stuck issue when multiple bundles are trying to be placed at the same time (verl is using STRICT_PACK mode)
            # To reproduce the issue, you can set it to 16 and run `psrl/examples/precision_test/dapo/megatron_qwen_7b_aime.sh`
            # where GEN_NNODES=$(( ${NNODES} / 2 )) and TRAIN_NNODES=$(( ${NNODES} / 2 )) 
            max_concurrent = min(len(tasks), 1)
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def controlled_create(resource_pool, class_dict):
                async with semaphore:
                    return await create_single_worker_group(resource_pool, class_dict)
            
            coroutines = [controlled_create(rp, cd) for rp, cd in tasks]
            results = await asyncio.gather(*coroutines, return_exceptions=True)
            
            all_wg_async = {}
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    resource_pool, class_dict = tasks[i]
                    psrl_logger.error(f"Error creating worker group for {resource_pool}, class {class_dict}: {str(result)}")
                    raise result
                all_wg_async.update(result)
            
            return all_wg_async

        async_results = asyncio.run(async_create_worker_groups())
        all_wg.update(async_results)
        '''
        
        # multi-thread version 
        train_tasks = []
        gen_tasks = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            psrl_logger.info(f"Creating worker group for resource pool: {resource_pool}, classes: {class_dict}")
            if "ps" in class_dict:
                assert class_dict.keys() == {"ps"}, "PS resource pool should only have PS role."
                continue
            if any("rollout" in key for key in class_dict.keys()):
                assert len(class_dict) == 1, "Rollout resource pool should only have one worker class."
                gen_tasks.append((resource_pool, class_dict))
            else:
                train_tasks.append((resource_pool, class_dict))
        # We must execute train tasks first because rollout instances may occupy the resources randomly and no structured resources are available for training
        with ThreadPoolExecutor(max_workers=len(train_tasks)) as executor:  # max_workers is the number of threads to use
            futures = {}
            for resource_pool, class_dict in train_tasks:
                future = executor.submit(
                    create_worker_group,
                    resource_pool,
                    class_dict
                )
                futures[future] = (resource_pool, class_dict)
            for future in futures:
                result = future.result()
                all_wg.update(result)
        with ThreadPoolExecutor(max_workers=len(gen_tasks)) as executor:  # max_workers is the number of threads to use
            futures = {}
            for resource_pool, class_dict in gen_tasks:
                future = executor.submit(
                    create_worker_group,
                    resource_pool,
                    class_dict
                )
                futures[future] = (resource_pool, class_dict)
            for future in futures:
                result = future.result()
                all_wg.update(result)
        
        '''
        # sync version
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if "ps" in class_dict:
                assert class_dict.keys() == {"ps"}, "PS resource pool should only have one worker class."
                continue # PS is created first, so we skip it here
            all_wg.update(create_worker_group(resource_pool, class_dict))
        '''
        
        # create agent loop workers
        self.agent_loop_workers = []
        self.rollout_wg_list = [all_wg[f"rollout_{i}"] for i in range(self.config.psrl.deployment.n_rollout_instances)]
        for i in range(self.config.gen_actor_rollout_ref.rollout.agent.num_workers):
            self.agent_loop_workers.append(
                PSRL_AgentLoopWorker.options(
                    name=f"agent_loop_worker_{i}",
                ).remote(
                    self.config,
                    self.ps_manager_handle,
                    self.rollout_wg_list,
                )
            )
        
        # create PS WorkerGroup
        psrl_logger.info("Create PS WorkerGroup")
        train_model_dtype = torch.bfloat16 if self.config.train_actor_rollout_ref.actor.strategy == "megatron" else torch.float32
        storage_plan = PSStoragePlan(
            train_model_dtype=train_model_dtype,
            gen_model_dtype=self.config.gen_actor_rollout_ref.rollout.dtype
        )
        if self.config.psrl.ps_mode == "cpu" or self.config.psrl.ps_mode == "cpu_ref":
            # PSManager is used to store the model state dict 
            # No need to create PS WorkerGroup
            pass
        elif self.config.psrl.ps_mode == "nixl_cpu" or self.config.psrl.ps_mode == "nixl_gpu":
            # PSManager is only used to build the nixl meta server
            # The PS WorkerGroup is used to store the model state dict
            # It is colocate with the rollout instances
            assert self.config.psrl.nixl.server_ip == self.config.psrl.ps_manager_ip, "PSManager IP and NIXL server IP must be the same"
            if self.config.psrl.ps_mode == "nixl_cpu":
                # Get all rollout instances' distinct node ids
                ps_node_ids = set()
                for i in range(self.config.psrl.deployment.n_rollout_instances):
                    rollout_instance_node_ids = all_wg[f"rollout_{i}"].execute_all_sync("get_node_id")
                    for node_id in rollout_instance_node_ids:
                        ps_node_ids.add(node_id)
                ps_spec_list = []
                for node_id in ps_node_ids:
                    ps_spec_list.append(PSResourceSpec(
                        node_id=node_id,
                        attached_gpu_id=None
                    ))
                ps_resource_pool = PSResourcePool(ps_spec_list=ps_spec_list)
                psrl_logger.info(f"PS resource pool: {ps_resource_pool}")
                self.ps_wg = PSWorkerGroup(
                    resource_pool=ps_resource_pool,
                    ps_cls_with_init=PSClassWithInitArgs(
                        cls=ray.remote(PSStorageWorker),
                        storage_plan=storage_plan,
                        model_config=self.config.train_actor_rollout_ref.model,
                        psrl_config=self.config.psrl,
                        nixl_interface=nixl_interface
                    )
                )
                ray.get(self.ps_wg.execute_all_async("init_model"))
                psrl_logger.info("PS model initialized successfully!")
            elif self.config.psrl.ps_mode == "nixl_gpu":
                raise NotImplementedError("PS mode 'nixl_gpu' is not implemented yet")
        else:
            raise ValueError(f"Invalid PS mode: {self.config.psrl.ps_mode}")

        psrl_logger.info("Initializing models in all rollout instances")
        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        # simutaneously init all rollout instances
        rollout_futures = []
        for i in range(self.config.psrl.deployment.n_rollout_instances):
            rollout_futures.extend(self.rollout_wg_list[i].execute_all_async("init_model"))

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
        
        if not (
            self.use_critic or
            (self.use_reference_policy and not self.ref_in_actor)
        ):
            # NOTE(linsh): when not using critic or reference policy,
            # if we directly call `init_model` of actor_wg, Ray will view the fused worker
            # as an async actor and run `run_async_func_or_coro_in_event_loop`, which will
            # make it invalid to call async function such as `trainer_mode` in `init_model`.
            # So here we create a dummy worker group and call its dummy method to avoid this issue.
            self.dummy_wg = all_wg["dummy"]
            self.dummy_wg.init_model()

        # Concurrently initialize actor and rollout instances
        psrl_logger.info("Initializing actor model")
        self.actor_wg = all_wg["actor"]
        self.actor_wg.init_model()

        ray.get(rollout_futures)

        psrl_logger.info("All workers' models initialized successfully!")

        # start rollout coordinator
        self.init_rollout_coordinator()

        # initialize NIXL
        if self.config.psrl.ps_mode == "nixl_cpu" or self.config.psrl.ps_mode == "nixl_gpu":
            with log_dual_events(f"Initializing NIXL clients", psrl_logger, event_type=EventType.INIT):
                futures = []
                rollout_world_size = ray.get(self.rollout_coordinator.world_size.remote())
                psrl_logger.info(f"Initializing NIXL server with {self.ps_wg.world_size} PS workers, {self.actor_wg.world_size} actor workers, {rollout_world_size} rollout workers")
                expected_agents = self.ps_wg.world_size + \
                    self.actor_wg.world_size + \
                    rollout_world_size
                futures.append(self.ps_manager_handle.init_nixl_server.remote(expected_agents))
                futures.extend(self.ps_wg.execute_all_async("init_nixl_client"))
                futures.extend(self.actor_wg.execute_all_async("init_nixl_client"))
                futures.append(self.rollout_coordinator.init_nixl_client.remote())
                ray.get(futures)
            
            with log_dual_events(f"Executing NIXL protocol", psrl_logger, event_type=EventType.INIT):
                futures = []
                futures.append(self.ps_manager_handle.nixl_protocol.remote())
                futures.extend(self.ps_wg.execute_all_async("nixl_protocol"))
                futures.extend(self.actor_wg.execute_all_async("nixl_protocol"))
                futures.append(self.rollout_coordinator.nixl_protocol.remote())
                ray.get(futures)
            
            psrl_logger.info("Binding PS worker group")
            self.ps_manager_handle.bind_ps_worker_group.remote(self.ps_wg)
            psrl_logger.info("PS worker group bound successfully!")

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
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
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
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )
        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        ray.get(self.data_processor.save_train_dataloader.remote(dataloader_local_path))

        # latest checkpointed iteration tracker (for atomic usage)
        local_mkdir_safe(self.config.trainer.default_local_dir)
        psrl_logger.info(f"Saving latest checkpointed iteration to {self.config.trainer.default_local_dir}")
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
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
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
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
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
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
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
        psrl_logger.info("Initialized tracking logger with project: %s, experiment: %s", 
                         self.config.trainer.project_name, self.config.trainer.experiment_name)

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        
        if self.global_steps >= self.total_training_steps:
            psrl_logger.warning(
                f"Global steps {self.global_steps} >= total training steps {self.total_training_steps}, "
                "skipping training."
            )
            return

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            psrl_logger.info(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        self.init_agent_loop_manager()

        futures = []
        futures.append(self.data_processor.set_agent_loop_manager.remote(self.agent_loop_manager))
        futures.append(self.ps_manager_handle.set_rollout_coordinator.remote(self.rollout_coordinator))
        for i in range(self.config.psrl.deployment.n_rollout_instances):
            futures.extend(self.rollout_wg_list[i].execute_all_async("set_rollout_coordinator", self.rollout_coordinator))
        ray.get(futures)
        
        self.init_reward_server()
        futures = []
        futures.append(self.data_processor.set_reward_server.remote(self.reward_server))
        futures.append(self.ps_manager_handle.set_reward_server.remote(self.reward_server))
        for agent_loop_worker in self.agent_loop_workers:
            futures.append(agent_loop_worker.set_reward_server.remote(self.reward_server))
        ray.get(futures)

        # Start data pipeline
        # 1. Start data processor to handle data preprocessing and batching
        psrl_logger.info("Starting data processor...")
        self.start_data_processor()
        psrl_logger.info("Data processor started successfully.")
        
        if not self.config.psrl.colocate:
            # 2. Start rollout coordinator to handle rollouts and data generation
            psrl_logger.info("Starting rollout coordinator...")
            self.start_rollout_coordinator()
            psrl_logger.info("Rollout coordinator started successfully.")
            
            # 3. Start agent loop manager to handle agent-environment interactions
            psrl_logger.info("Starting agent loop manager...")
            self.start_agent_loop_manager()
            psrl_logger.info("Agent loop manager started successfully.")

            # 4. Start reward server to handle reward computation requests
            psrl_logger.info("Starting reward server...")
            self.start_reward_server()
            psrl_logger.info("Reward server started successfully.")
            
        psrl_logger.info("All data pipeline components started successfully.")

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

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
                        with log_dual_events(f"Wait for training batch {buffer_id}", psrl_logger, event_type=EventType.WAIT):
                            batch = ray.get(self.reward_server.wait_for_training_batch.remote(buffer_id)) 
                        psrl_logger.debug("Received training batch for step %d, batch size: %d", 
                                        self.global_steps, len(batch) if batch is not None else 0)
                    else:
                        from verl.trainer.ppo.reward import compute_reward
                        batch = ray.get(self.agent_loop_manager_handle.get_data.remote())
                        if batch is None:
                            psrl_logger.info("No more data from agent loop manager, ending training at step %d", self.global_steps)
                            break
                        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
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
                batch.meta_info['micro_batch_size'] = self.config.train_actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu
                batch.meta_info['max_token_len'] = self.config.train_actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu
                batch.meta_info['use_dynamic_bsz'] = self.config.train_actor_rollout_ref.rollout.log_prob_use_dynamic_bsz
                batch.meta_info['temperature'] = self.config.gen_actor_rollout_ref.rollout.temperature
                if self.config.psrl.log_prob.enable_rollout_engine_log_prob:
                    # batch.batch["rollout_log_probs"] can be used directly
                    pass 
                if self.config.psrl.log_prob.enable_train_engine_recompute_log_prob:
                    # recompute log_probs in the training side
                    with marked_timer("recompute_log_prob", timing_raw, color="orange"):
                        with log_dual_events("Recompute log_prob on training side", psrl_logger, event_type=EventType.OTHER):
                            recomputed_log_prob = self.actor_wg.compute_log_prob(batch)
                            entropys = recomputed_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.train_actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
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
                            
                # TODO(lhy): support TIS
                if self.config.psrl.log_prob.mode == "rollout":
                    batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]
                    batch.batch.pop("rollout_log_probs")
                elif self.config.psrl.log_prob.mode == "recompute":
                    batch.batch["old_log_probs"] = batch.batch["recomputed_log_probs"]
                    batch.batch.pop("recomputed_log_probs")
                elif self.config.psrl.log_prob.mode == "tis":
                    batch.batch["old_log_probs"] = batch.batch["recomputed_log_probs"]
                    batch.batch.pop("recomputed_log_probs")
                else:
                    raise ValueError(f"Invalid log_prob mode: {self.config.psrl.log_prob.mode}")

                if self.use_reference_policy:
                    # compute reference log_prob
                    with marked_timer("ref", timing_raw, color="olive"):
                        with log_dual_events("Compute reference log_prob", psrl_logger, event_type=EventType.OTHER):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                # compute values
                if self.use_critic:
                    with marked_timer("values", timing_raw, color="cyan"):
                        with log_dual_events("Compute critic values", psrl_logger, event_type=EventType.OTHER):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)
                
                # compute reward model score
                if self.use_rm and "rm_scores" not in batch.batch.keys():
                    with marked_timer("reward", timing_raw, color="yellow"):
                        with log_dual_events("Compute reward model score", psrl_logger, event_type=EventType.OTHER):
                            # compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)
                elif self.config.reward_model.launch_reward_fn_async:
                    # Overlap reward computation with log_prob computation in trainer
                    with marked_timer("async_reward_get", timing_raw, color="yellow"):
                        with log_dual_events("Get async reward model score", psrl_logger, event_type=EventType.OTHER):
                            future_rewards = batch.non_tensor_batch.pop("future_reward", None)
                            assert future_rewards is not None, "Reward tensor must be provided in async mode"
                            reward_tensor_list = []
                            reward_extra_infos_dict_list = []
                            for future_reward in future_rewards:
                                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                                reward_tensor_list.append(reward_tensor)
                                reward_extra_infos_dict_list.append(reward_extra_infos_dict)
                            reward_tensor = torch.cat(reward_tensor_list, dim=0)
                            reward_extra_infos_dict = defaultdict(list)
                            for reward_extra_infos in reward_extra_infos_dict_list:
                                for key, value in reward_extra_infos.items():
                                    reward_extra_infos_dict[key].extend(value)
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                else:
                    reward_tensor = batch.batch.pop("reward", None)
                
                batch.batch["token_level_scores"] = reward_tensor

                with marked_timer("adv", timing_raw, color="brown"):
                    with log_dual_events("Compute advantage", psrl_logger, event_type=EventType.OTHER):
                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor
                        
                        log_data_protocol(batch, psrl_logger, self.log_prefix + " before compute advantage", level=logging.DEBUG)
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
                    psrl_logger.info(f"Update actor ppo_kl: {actor_output.meta_info['metrics']['actor/ppo_kl']}, len: {len(actor_output.meta_info['metrics']['actor/ppo_kl'])}")
                    psrl_logger.info(f"Update actor pg_loss: {actor_output.meta_info['metrics']['actor/pg_loss']}, len: {len(actor_output.meta_info['metrics']['actor/pg_loss'])}")
                    psrl_logger.info(f"Update actor grad_norm: {actor_output.meta_info['metrics']['actor/grad_norm']}, len: {len(actor_output.meta_info['metrics']['actor/grad_norm'])}")
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                # Log rollout generations if enabled
                rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                if rollout_data_dir:
                    with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                        with log_dual_events("Dump rollout generations", psrl_logger, event_type=EventType.OTHER):
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
                    self.val_reward_fn is not None and
                    self.config.trainer.test_freq > 0 and
                    (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        with log_dual_events("Validate", psrl_logger, event_type=EventType.VAL):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or
                    self.global_steps % self.config.trainer.save_freq == 0
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
                    tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                )

            if is_last_step:
                psrl_logger.info(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                break

        # Stop all components
        psrl_logger.info("Stopping all data pipeline components...")
        self.stop_reward_server()
        self.stop_agent_loop_manager()
        self.stop_rollout_coordinator()
        self.stop_data_processor()

        psrl_logger.info("Training completed successfully!")