import os
import importlib.util
import random
import sys
import hydra
import ray
import torch
import numpy as np
from pprint import pprint
from omegaconf import OmegaConf

from verl.single_controller.ray import RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker, RewardModelWorker
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local

from psrl.workers.rollout import PSRL_GenWorker
from psrl.workers.train import PSRL_TrainWorker
from psrl.workers.ps import PSRL_PSWorker
from psrl.trainer.ppo.ray_trainer import PSRL_ResourcePoolManager, PSRL_RayPPOTrainer, PSRL_Role


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def get_custom_reward_fn(config):
    """Load a custom reward function from a file."""
    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules["custom_module"] = module
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e

    function_name = reward_fn_config.get("name")
    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")
    raw_fn = getattr(module, function_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **reward_kwargs)

    return wrapped_fn


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # print initial config
        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

        # define worker classes
        assert config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"], "Currently only fsdp and fsdp2 are supported."
        assert config.critic.strategy in ["fsdp", "fsdp2"], "Currently only fsdp and fsdp2 are supported."
        ray_worker_group_cls = RayWorkerGroup
        
        deployment_config = config.psrl.deployment
        rollout_pool_id_list = [f'rollout_pool_{i}' for i in range(deployment_config.n_rollout_instances)]
        train_pool_id = 'train_pool'
        ps_pool_id = 'ps_pool'
        # format: {pool_id: [ngpus_per_node] * nnodes}
        # nnodes will be the number of ray placement groups
        # and ngpus_per_node will be the number of ray bundles (currently all equals to {"CPU": self.max_colocate_count, "GPU": 1}) in each placement group
        resource_pool_spec = {
            train_pool_id: [deployment_config.train_ngpus_per_node] * deployment_config.train_nnodes,
            ps_pool_id: [deployment_config.ps_ngpus_per_node] * deployment_config.ps_nnodes,
        }
        for i in range(deployment_config.n_rollout_instances):
            rollout_pool_id = rollout_pool_id_list[i]
            resource_pool_spec[rollout_pool_id] = [deployment_config.rollout_ngpus_per_node_per_instance] * deployment_config.rollout_nnodes_per_instance,
        role_worker_mapping = {
            PSRL_Role.Rollout: ray.remote(PSRL_GenWorker),
            PSRL_Role.Actor: ray.remote(PSRL_TrainWorker),
            PSRL_Role.Critic: ray.remote(CriticWorker),
            PSRL_Role.ParameterServer: ray.remote(PSRL_PSWorker)
        }
        # multiple instances mapping
        mapping = {
            PSRL_Role.Rollout: rollout_pool_id_list,
            PSRL_Role.Actor: [train_pool_id],
            PSRL_Role.Critic: [train_pool_id],
            PSRL_Role.ParameterServer: [ps_pool_id]
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            assert config.reward_model.strategy in ["fsdp", "fsdp2"], "Currently only fsdp and fsdp2 are supported."
            role_worker_mapping[PSRL_Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[PSRL_Role.RewardModel] = [train_pool_id]

        # use reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[PSRL_Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[PSRL_Role.RefPolicy] = [train_pool_id]

        reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1)
        resource_pool_manager = PSRL_ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        trainer = PSRL_RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    seed_everything(42)
    main()
