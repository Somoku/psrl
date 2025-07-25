import os
import uuid
import torch
import logging
import numpy as np
import ray
from ray.exceptions import RayTaskError
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from collections import defaultdict
from copy import deepcopy
from enum import Enum
from typing import Type
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.ray_trainer import WorkerType, AdvantageEstimator, ResourcePoolManager, apply_kl_penalty, compute_response_mask, compute_advantage, _timer, RayPPOTrainer
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.tracking import ValidationGenerationsLogger

from psrl.utils.nixl import NIXLInterface, global_port_scanner
from psrl.utils.dataset import DatasetType, DatasetHandle
from psrl.utils.logger import DualOutputHandler, log_dual_events, log_single_event, EventType
from psrl.workers.train import TrainInterface
from psrl.workers.gen import GenInterface
from psrl.workers.ps import PSManager, PSWorkerGroup, PSResourceSpec, PSResourcePool, PSClassWithInitArgs, PSWorker

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))


class PSRL_Role(Enum):
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6
    ParameterServer = 7


class PSRL_ResourcePoolManager(ResourcePoolManager):
    """
    Support multiple instances of the same role
    """
    mapping: dict[PSRL_Role, list[str]]
    
    def get_resource_pool(self, role: PSRL_Role, instance_id: int = 0) -> RayResourcePool:
        """Get the resource pool of the worker_cls for the given instance_id."""
        return self.resource_pool_dict[self.mapping[role][instance_id]]


class PSRL_RayPPOTrainer(RayPPOTrainer):
    
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[PSRL_Role, WorkerType],
        resource_pool_manager: PSRL_ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
    ):
        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = PSRL_Role.RefPolicy in role_worker_mapping
        self.use_rm = PSRL_Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()
        
        # Build logger
        self.log_prefix = f"Main_Ray_Trainer"
        psrl_logger.addHandler(DualOutputHandler(self.log_prefix))
        psrl_logger.info(f"Initialized major ray trainer (single controller).")

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            # AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]:
            self.use_critic = False
        else:
            # TODO: REMAX need to compute the advantage on input prompts and overlap this process with generation  
            if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                raise NotImplementedError("REMAX is not implemented yet, please use other advantage estimator")
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()
        
    def _validate_config(self):
        config = self.config
        # number of GPUs used in training
        train_n_gpus = config.psrl.deployment.train_ngpus_per_node * config.psrl.deployment.train_nnodes
        if config.train_actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = config.train_actor_rollout_ref.actor.megatron.tensor_model_parallel_size * config.train_actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            assert train_n_gpus % (model_parallel_size * config.train_actor_rollout_ref.actor.megatron.context_parallel_size) == 0, f"train_n_gpus ({train_n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times context_parallel_size ({config.train_actor_rollout_ref.actor.megatron.context_parallel_size})"
            megatron_dp = train_n_gpus // (model_parallel_size * config.train_actor_rollout_ref.actor.megatron.context_parallel_size)
            minimal_bsz = megatron_dp * config.train_actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = train_n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.train_actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size ({minimal_bsz})"

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
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
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

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
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.train_actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.train_actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.train_actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.train_actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.train_actor_rollout_ref.actor.ppo_mini_batch_size % config.train_actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.train_actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= train_n_gpus

        assert config.train_actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.train_actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.train_actor_rollout_ref.actor.use_kl_loss:
            psrl_logger.info("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= train_n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.train_actor_rollout_ref.actor.strategy == "fsdp" and (config.train_actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.train_actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.train_actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            psrl_logger.info("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.train_actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.train_actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.train_actor_rollout_ref.rollout.multi_turn.enable:
            assert config.train_actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        psrl_logger.info("[validate_config] All configuration checks passed successfully!")
    
    # all rollout instances share a same dataloader (a ray actor)
    def _create_dataloader(self):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: support different rollout instances use difference TP
        tp = self.config.psrl.deployment.rollout_ngpus_per_node_per_instance
        rollout_instances_tp = {i: tp for i in range(self.config.psrl.deployment.n_rollout_instances)}
        self.dataset_handle = DatasetHandle.remote(
            self.config,
            self.tokenizer,
            self.processor,
            rollout_instances_tp
        )
        self.dataset_handle.build_train_and_val_dataset.remote()
        self.dataset_handle.build_train_sampler.remote()
        self.dataset_handle.build_train_dataloader.remote()
        self.dataset_handle.build_val_dataloader.remote()

        total_training_steps = ray.get(self.dataset_handle.get_train_len.remote()) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        psrl_logger.info(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "train_actor_rollout_ref.actor.optim"):
                    self.config.train_actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            psrl_logger.info(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    # We use the training side to do val so that val and generation can overlap
    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        while True:
            try:
                test_data = ray.get(self.dataset_handle.get_single_controller_batch.remote(DatasetType.val))
            except RayTaskError as e:
                if isinstance(e.cause, StopIteration):
                    break
                else:
                    psrl_logger.info(f"Unknown exception happened during obtaining validation data: {type(e.cause)}")
                    raise
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.train_actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_inputs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.extend(["multi_modal_data", "multi_modal_inputs"])
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
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
            }
            psrl_logger.info(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_wg.world_size)
            # switch to the inference engine and generate sequences
            test_output_gen_batch_padded = self.actor_wg.generate_sequences(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            psrl_logger.info("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        # create rollout, actor and ps
        # PS need to be created before rollout and actor to pass the ps_manager_handle
        # assert PSRL_Role.Rollout in self.role_worker_mapping and PSRL_Role.Actor in self.role_worker_mapping and PSRL_Role.ParameterServer in self.role_worker_mapping, "Rollout, Actor and PS must be in role_worker_mapping." 
        assert PSRL_Role.Rollout in self.role_worker_mapping and PSRL_Role.Actor in self.role_worker_mapping, "Rollout, Actor must be in role_worker_mapping." 
         
        # create ps manager
        ip_to_node_id = {node['NodeManagerAddress']: node['NodeID'] for node in ray.nodes()}
        assert self.config.psrl.ps_manager_ip in ip_to_node_id, f"PSManager IP {self.config.psrl.ps_manager_ip} not found in ray nodes"
        psrl_logger.info("Getting the handle of the PSManager")
        self.ps_manager_handle = PSManager.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=ip_to_node_id[self.config.psrl.ps_manager_ip],
                soft=False
            )
        ).remote(self.config.psrl)
        
        nixl_interface = NIXLInterface(
            port_scanner=global_port_scanner
        )
         
        # create rollout instances  
        for i in range(self.config.psrl.deployment.n_rollout_instances):
            gen_interface = GenInterface(
                rollout_instance_id=i, 
                dataset_handle=self.dataset_handle,
                ps_manager_handle=self.ps_manager_handle
            )
            rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[PSRL_Role.Rollout],
                config=self.config.gen_actor_rollout,
                role='rollout',
                psrl_config=self.config.psrl,
                gen_interface=gen_interface,
                nixl_interface=nixl_interface
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
            nixl_interface=nixl_interface
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
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(PSRL_Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[PSRL_Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        psrl_logger.info("Initializing WorkerGroup for other roles")
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        def create_worker_group(resource_pool, class_dict):
            # if there is only one worker class in the resource pool, we can directly create a worker group
            # so that we can use 'execute_all_async' and other low-level APIs
            if len(class_dict) == 1:
                role = next(iter(class_dict.keys()))
                return {role: self.ray_worker_group_cls(
                    resource_pool=resource_pool,
                    ray_cls_with_init=class_dict[role],
                    **wg_kwargs
                )}
            # colocate
            else:
                worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
                wg_dict = self.ray_worker_group_cls(
                    resource_pool=resource_pool,
                    ray_cls_with_init=worker_dict_cls,
                    **wg_kwargs
                )
                return wg_dict.spawn(prefix_set=class_dict.keys())
        
        # multi-thread version 
        tasks = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if "ps" in class_dict:
                assert class_dict.keys() == {"ps"}, "PS resource pool should only have PS role."
                continue
            tasks.append((resource_pool, class_dict))
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:  # 最多同时处理所有任务
            futures = {}
            for resource_pool, class_dict in tasks:
                future = executor.submit(
                    create_worker_group,
                    resource_pool,
                    class_dict
                )
                futures[future] = (resource_pool, class_dict)
            for future in futures:
                try:
                    result = future.result()
                    all_wg.update(result)
                except Exception as e:
                    resource_pool, class_dict = futures[future]
                    psrl_logger.info(f"Error creating worker group for {resource_pool}, class {class_dict}: {str(e)}")
                    raise
        
        '''
        # sync version
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if "ps" in class_dict:
                assert class_dict.keys() == {"ps"}, "PS resource pool should only have one worker class."
                continue # PS is created first, so we skip it here
            all_wg.update(create_worker_group(resource_pool, class_dict))
        '''
        
        # create PS WorkerGroup
        psrl_logger.info("Create PS WorkerGroup")
        if self.config.psrl.ps_mode == "cpu" or self.config.psrl.ps_mode == "cpu_ref":
            # PSManager is used to store the model state dict 
            # No need to create PS WorkerGroup
            pass
        elif self.config.psrl.ps_mode == "nixl_cpu" or self.config.psrl.ps_mode == "nixl_gpu":
            # PSManager is only used to build the nixl meta server
            # The PS WorkerGroup is used to store the model state dict
            # It is colocate with the rollout instances
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
                        cls=ray.remote(PSWorker),
                        psrl_config=self.config.psrl,
                        nixl_interface=nixl_interface
                    )
                )
                self.ps_manager_handle.bind_ps_worker_group.remote(self.ps_wg)
            elif self.config.psrl.ps_mode == "nixl_gpu":
                raise NotImplementedError("PS mode 'nixl_gpu' is not implemented yet")
        else:
            raise ValueError(f"Invalid PS mode: {self.config.psrl.ps_mode}")

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        psrl_logger.info("Initializing actor model")
        self.actor_wg = all_wg["actor"]
        self.actor_wg.init_model()
        
        psrl_logger.info("Initializing models in all rollout instances")
        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        # simutaneously init all rollout instances
        self.rollout_wg_list = [all_wg[f"rollout_{i}"] for i in range(self.config.psrl.deployment.n_rollout_instances)]
        futures = []
        for i in range(self.config.psrl.deployment.n_rollout_instances):
            futures.extend(self.rollout_wg_list[i].execute_all_async("init_model"))
        ray.get(futures)
        
        psrl_logger.info("All workers' models initialized successfully!")
        
        psrl_logger.info("Initializing NIXL")
        futures = []
        expected_clients = self.ps_wg.world_size + \
            self.actor_wg.world_size + \
            sum([self.rollout_wg_list[i].world_size for i in range(self.config.psrl.deployment.n_rollout_instances)])
        futures.append(self.ps_manager_handle.init_nixl_server.remote(expected_clients))
        futures.extend(self.ps_wg.execute_all_async("init_nixl_client"))
        futures.extend(self.actor_wg.execute_all_async("init_nixl_client"))
        for i in range(self.config.psrl.deployment.n_rollout_instances):
            futures.extend(self.rollout_wg_list[i].execute_all_async("init_nixl_client"))
        ray.get(futures)
        psrl_logger.info("NIXL initialized successfully!")
        
        psrl_logger.info("Executing NIXL protocol")
        futures = []
        futures.append(self.ps_manager_handle.nixl_potocol.remote())
        futures.extend(self.ps_wg.execute_all_async("nixl_protocol"))
        futures.extend(self.actor_wg.execute_all_async("nixl_protocol"))
        for i in range(self.config.psrl.deployment.n_rollout_instances):
            futures.extend(self.rollout_wg_list[i].execute_all_async("nixl_protocol"))
        ray.get(futures)
        psrl_logger.info("NIXL protocol executed successfully!")

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        psrl_logger.info(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            psrl_logger.info("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        ray.get(self.dataset_handle.save_train_dataloader.remote(dataloader_local_path))

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
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
        self.actor_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load rollout instance
        for i in range(self.config.psrl.deployment.n_rollout_instances):
            self.rollout_wg_list[i].load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # TODO: push the actor model state dict to the PS worker (though it is not necessary to do so)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            ray.get(self.dataset_handle.load_train_dataloader.remote(dataloader_local_path))
        else:
            psrl_logger.info(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
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

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            psrl_logger.info(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        # rollout instances keep generating sequences in their busy loop
        for i in range(self.config.psrl.deployment.n_rollout_instances):
            self.rollout_wg_list[i].execute_all_async("busy_loop_generate_sequences")
        
        # busy loop for training
        while True:
            metrics = {}
            timing_raw = {}
            is_last_step = self.global_steps >= self.total_training_steps

            with _timer("step", timing_raw): 
                   
                with _timer("wait_for_gen", timing_raw):   
                    buffer_id = self.global_steps - 1
                    # will block until the training batch is ready
                    with log_dual_events(f"Wait for training batch {buffer_id}", psrl_logger, event_type=EventType.WAIT):
                        batch = ray.get(self.ps_manager_handle.wait_for_training_batch.remote(buffer_id)) 
                    psrl_logger.debug(f"Global step {self.global_steps} training batch: {batch}")
                    
                batch.batch["response_mask"] = compute_response_mask(batch)
                # balance the number of valid tokens on each dp rank.
                # Note that this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                if self.config.trainer.balance_batch:
                    self._balance_batch(batch, metrics=metrics)

                # compute global_valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                with _timer("reward", timing_raw):
                    with log_dual_events("Compute reward model score", psrl_logger, event_type=EventType.OTHER):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                if self.config.psrl.log_prob.enable_inference_engine_log_prob:
                    # log probs from vLLM could be buggy
                    batch.meta_info['micro_batch_size'] = self.config.gen_actor_rollout.rollout.log_prob_micro_batch_size_per_gpu
                    batch.meta_info['max_token_len'] = self.config.gen_actor_rollout.rollout.log_prob_max_token_len_per_gpu
                    batch.meta_info['use_dynamic_bsz'] = self.config.gen_actor_rollout.rollout.log_prob_use_dynamic_bsz
                    batch.meta_info['temperature'] = self.config.gen_actor_rollout.rollout.temperature
                    batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]
                else:
                    # TODO: support recompute old_log_probs in the generation side
                    raise NotImplementedError("Use training engine to compute log_prob is not supported in PSRL yet, please set enable_inference_engine_log_prob for now.")
                
                if self.config.psrl.log_prob.enable_proxy_log_prob:
                    # compute proxy log_prob
                    # AReal's algorithms require a proxy policy
                    with _timer("proxy_log_prob", timing_raw):
                        with log_dual_events("Compute proxy log_prob", psrl_logger, event_type=EventType.OTHER):
                            proxy_log_prob = self.actor_wg.compute_log_prob(batch)
                            batch = batch.union(proxy_log_prob)
                    # TODO: support AReal's revised PPO

                if self.use_reference_policy:
                    # compute reference log_prob
                    with _timer("ref", timing_raw):
                        with log_dual_events("Compute reference log_prob", psrl_logger, event_type=EventType.OTHER):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                # compute values
                if self.use_critic:
                    with _timer("values", timing_raw):
                        with log_dual_events("Compute critic values", psrl_logger, event_type=EventType.OTHER):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                with _timer("adv", timing_raw):
                    with log_dual_events("Compute reward", psrl_logger, event_type=EventType.OTHER):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        psrl_logger.info(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.gen_actor_rollout.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.gen_actor_rollout.rollout.multi_turn.enable,
                            config=self.config.algorithm
                        )

                # update critic
                if self.use_critic:
                    with _timer("update_critic", timing_raw):
                        with log_dual_events("Update critic", psrl_logger, event_type=EventType.TRAIN):
                            critic_output = self.critic_wg.update_critic(batch)
                    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    metrics.update(critic_output_metrics)

                # implement critic warmup
                if self.config.trainer.critic_warmup <= self.global_steps:
                    # update actor
                    with _timer("update_actor", timing_raw):
                        with log_dual_events("Update actor", psrl_logger, event_type=EventType.TRAIN):
                            batch.meta_info["multi_turn"] = self.config.gen_actor_rollout.rollout.multi_turn.enable
                            actor_output = self.actor_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                # Log rollout generations if enabled
                rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                if rollout_data_dir:
                    with _timer("dump_rollout_generations", timing_raw):
                        with log_dual_events("Dump rollout generations", psrl_logger, event_type=EventType.OTHER):
                            psrl_logger.info(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                # validate
                if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                    with _timer("testing", timing_raw):
                        with log_dual_events("Validate", psrl_logger, event_type=EventType.VAL):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                    with _timer("save_checkpoint", timing_raw):
                        with log_dual_events("Save checkpoint", psrl_logger, event_type=EventType.OTHER):
                            self._save_checkpoint()

            # training metrics
            metrics.update(
                {
                    "training/global_step": self.global_steps,
                }
            )
            # collect metrics
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            # TODO: implement actual tflpo and theoretical tflpo
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

            # TODO: make a canonical logger that supports various backend
            logger.log(data=metrics, step=self.global_steps)

            if is_last_step:
                psrl_logger.info(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                return

            progress_bar.update(1)
            self.global_steps += 1
