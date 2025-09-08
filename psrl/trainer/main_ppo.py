import os
import importlib.util
import random
import sys
import hydra
import ray
import logging
import torch
import numpy as np
from pprint import pprint
from omegaconf import OmegaConf

from verl.trainer.ppo.reward import load_reward_manager
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local


from psrl.trainer.ppo.ray_trainer import PSRL_ResourcePoolManager, PSRL_RayPPOTrainer, PSRL_Role

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true", 
                    "NCCL_DEBUG": "WARN", 
                    "VLLM_USE_V1": "1",
                    "VLLM_LOGGING_LEVEL": "WARN",
                    "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
                    "VLLM_DISABLE_COMPILE_CACHE": "1", # NOTE: workaround for vllm compile cache issue, see https://github.com/vllm-project/vllm/issues/18851
                    "PSRL_LOGGING_PATH": config.psrl.logging_path,
                }
            },
            num_cpus=config.ray_init.num_cpus,
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)

@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # print initial config
        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(config.train_actor_rollout_ref.model.path, use_shm=config.train_actor_rollout_ref.model.get("use_shm", False))

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        
        # Version validation for vllm.
        if config.gen_actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.gen_actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # define worker classes
        if config.train_actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"], "Critic strategy must be the same as actor strategy: 'fsdp' or 'fsdp2'."
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
            from psrl.workers.train.fsdp_train_worker import PSRL_FSDPTrainWorker as PSRL_TrainWorker
            from psrl.workers.gen.gen_worker import PSRL_GenWorker

            ray_worker_group_cls = RayWorkerGroup
        elif config.train_actor_rollout_ref.actor.strategy == "megatron":
            assert config.train_actor_rollout_ref.actor.strategy == config.critic.strategy
            # from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from psrl.workers.train.refactored_megatron_worker_group import RefactoredNVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
            from psrl.workers.train.megatron_train_worker import PSRL_MegatronTrainWorker as PSRL_TrainWorker
            from psrl.workers.gen.gen_worker import PSRL_GenWorker

            ray_worker_group_cls = RefactoredNVMegatronRayWorkerGroup
        else:
            raise NotImplementedError(f"Unsupported strategy: {config.train_actor_rollout_ref.actor.strategy}. "
                                        "Currently only 'fsdp', 'fsdp2', and 'megatron' are supported.")
        
        deployment_config = config.psrl.deployment
        rollout_pool_id_list = [f'rollout_pool_{i}' for i in range(deployment_config.n_rollout_instances)]
        train_pool_id = 'train_pool'
        # PS pool is not used, we now use self-defined PS worker group
        # ps_pool_id = 'ps_pool'
        
        # format: {pool_id: [ngpus_per_node] * nnodes}
        # nnodes will be the number of ray placement groups
        # and ngpus_per_node will be the number of ray bundles (currently all equals to {"CPU": self.max_colocate_count, "GPU": 1}) in each placement group
        # For now, PS nodes are colocated with rollout nodes (on CPUs)
        resource_pool_spec = {
            train_pool_id: [deployment_config.train_ngpus_per_node] * deployment_config.train_nnodes,
            # ps_pool_id: [deployment_config.ps_ngpus_per_node] * deployment_config.ps_nnodes,
        }
        if deployment_config.heterogeneous_rollout.enable:
            heterogeneous_deployment_config = deployment_config.heterogeneous_rollout
            assert len(heterogeneous_deployment_config.rollout_nnodes_per_instance) == heterogeneous_deployment_config.n_rollout_instances, \
                "The number of rollout nnodes per instance must match the number of rollout instances."
            assert len(heterogeneous_deployment_config.rollout_ngpus_per_node_per_instance) == heterogeneous_deployment_config.n_rollout_instances, \
                "The number of rollout ngpus per node per instance must match the number of rollout instances."
            assert len(heterogeneous_deployment_config.tensor_model_parallel_size_per_instance) == heterogeneous_deployment_config.n_rollout_instances, \
                "The number of tensor model parallel size per instance must match the number of rollout instances."
            assert len(heterogeneous_deployment_config.pipeline_model_parallel_size_per_instance) == heterogeneous_deployment_config.n_rollout_instances, \
                "The number of pipeline model parallel size per instance must match the number of rollout instances."

            for i in range(deployment_config.n_rollout_instances):
                rollout_pool_id = rollout_pool_id_list[i]
                resource_pool_spec[rollout_pool_id] = [heterogeneous_deployment_config.rollout_ngpus_per_node_per_instance[i]] * heterogeneous_deployment_config.rollout_nnodes_per_instance[i]
        else:
            for i in range(deployment_config.n_rollout_instances):
                rollout_pool_id = rollout_pool_id_list[i]
                resource_pool_spec[rollout_pool_id] = [deployment_config.rollout_ngpus_per_node_per_instance] * deployment_config.rollout_nnodes_per_instance
        role_worker_mapping = {
            PSRL_Role.Rollout: ray.remote(PSRL_GenWorker),
            PSRL_Role.Actor: ray.remote(PSRL_TrainWorker),
            PSRL_Role.Critic: ray.remote(CriticWorker),
            # PSRL_Role.ParameterServer: ray.remote(PSRL_PSWorker)
        }
        # multiple instances mapping
        mapping = {
            PSRL_Role.Rollout: rollout_pool_id_list,
            PSRL_Role.Actor: [train_pool_id],
            PSRL_Role.Critic: [train_pool_id],
            # PSRL_Role.ParameterServer: [ps_pool_id]
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[PSRL_Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[PSRL_Role.RewardModel] = [train_pool_id]

        # use reference model
        if config.algorithm.use_kl_in_reward or config.train_actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[PSRL_Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[PSRL_Role.RefPolicy] = [train_pool_id]

        reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {}))
        
        print(f"resource_pool_spec = {resource_pool_spec}, mapping = {mapping}")
        resource_pool_manager = PSRL_ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        
        from verl.utils.dataset.rl_dataset import collate_fn

        trainer = PSRL_RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            collate_fn=collate_fn,
            device_name=config.trainer.device,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    seed_everything(0)
    main()