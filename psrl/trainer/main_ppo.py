import logging
import os
import random
import socket

# Must precede anything that reaches megatron.bridge. That chain pulls in modelopt ->
# tilelang, which loads its own `libcudart_stub.so` into the process; a later
# `flashinfer.comm` import then resolves cudaDeviceReset against the stub and dies with
# "undefined symbol: cudaDeviceReset". vLLM's FusedMoE -> all2all_utils -> nixl_ep path
# imports flashinfer, so the trainer fails at import time unless the real CUDA runtime is
# bound first. Order, not presence, is what matters -- the same fix the NIXL e2e harness
# already carries, applied here because this is the production entry point.
try:
    import flashinfer.comm  # noqa: F401
except Exception as _flashinfer_exc:  # pragma: no cover - optional on CPU-only runs
    # Logged, not swallowed. A bare `pass` here hides the one thing worth knowing: if
    # this import fails *now*, the next one -- made lazily by vLLM, deep inside
    # `verl.workers.rollout` -- fails identically but far from the cause, and reads as a
    # vLLM bug. The warning is what connects the two.
    import warnings as _warnings

    _warnings.warn(
        f"flashinfer.comm preload failed ({type(_flashinfer_exc).__name__}: "
        f"{_flashinfer_exc}); a later lazy import of it will fail the same way",
        stacklevel=2,
    )

import hydra
import numpy as np
import ray
import torch
import transfer_queue as tq
from omegaconf import OmegaConf
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.device import auto_set_device, is_cuda_available
from psrl.utils.logger import configure_verl_logging

from psrl.trainer.constants_ppo import get_ppo_ray_runtime_env
from psrl.trainer.ppo.utils import PSRL_Role
from psrl.utils.config import validate_config
from psrl.workers.config.reward_model import resolve_active_managers

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


def _configure_runtime_env_flags(config) -> None:
    """
    Propagate config-derived flags before Ray snapshots the driver environment.

    veRL's runtime environment forwards these variables at call time, so they must be set on
    `os.environ` before `ray.init`.

    Args:
        config: Hydra training configuration.
    """
    rollout_config = config.gen_actor_rollout_ref.rollout
    deterministic = rollout_config.full_determinism

    reward_config = OmegaConf.select(config, "reward", default=None)
    if reward_config is not None:
        deterministic = deterministic or any(
            bool(OmegaConf.select(manager, "rollout.full_determinism", default=False))
            for manager in resolve_active_managers(reward_config)
            if OmegaConf.select(manager, "reward_loop_type", default=None) == "gen"
        )

    if deterministic:
        os.environ["VERL_FULL_DETERMINISM"] = "1"
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
        os.environ["PYTHONHASHSEED"] = str(rollout_config.seed)

    trainer_logger = config.trainer.get("logger", [])
    logger_backends = [trainer_logger] if isinstance(trainer_logger, str) else trainer_logger or []
    if "rl_insight" in logger_backends:
        os.environ["VERL_RL_INSIGHT_ENABLE"] = "1"


def seed_everything(seed: int):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)

    config.transfer_queue.enable = True

    # validate config
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )

    # AGENT(VERL): Skip migrate legacy reward impl
    run_ppo(config, task_runner_class=TaskRunner)


@ray.remote(num_gpus=1, num_cpus=0)
class _GPUSlotReserver:
    """Lightweight Ray actor that holds a GPU slot without touching CUDA.

    Used to prevent Ray from scheduling job workers onto nodes that the job
    does not intend to use, which would otherwise cause multiple validate
    instances to land on the same GPU when the cluster has excess capacity.
    """

    def ping(self) -> str:
        return "ok"


def _reserve_excess_nodes(config) -> list:
    """Block GPU slots on nodes that are not needed by this job.

    Reads psrl.deployment.total_nnodes from config. If null/None, does
    nothing. Otherwise, compares against the live cluster node count and
    creates one _GPUSlotReserver actor per GPU on every excess node to
    prevent Ray from scheduling job workers there.

    Args:
        config: Hydra config with psrl.deployment fields.

    Returns:
        list: The list of _GPUSlotReserver actor handles (kept alive by caller).
    """
    total_nnodes = config.psrl.deployment.get("total_nnodes", None)
    if total_nnodes is None:
        return []
    total_nnodes = int(total_nnodes)

    alive_gpu_nodes = sorted(
        [n for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU", 0) > 0],
        key=lambda n: n["NodeID"],
    )
    cluster_nnodes = len(alive_gpu_nodes)

    if cluster_nnodes <= total_nnodes:
        psrl_logger.info(f"Cluster has {cluster_nnodes} GPU nodes, job needs {total_nnodes}. No reservation needed.")
        return []

    excess_nodes = alive_gpu_nodes[total_nnodes:]
    reservers = []
    for node in excess_nodes:
        node_id = node["NodeID"]
        n_gpus = int(node["Resources"]["GPU"])
        psrl_logger.info(
            f"Reserving {n_gpus} GPU slot(s) on excess node {node['NodeManagerAddress']} "
            f"(node_id={node_id}) to prevent stray worker placement."
        )
        for _ in range(n_gpus):
            actor = _GPUSlotReserver.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
            ).remote()
            reservers.append(actor)

    # Verify actors are ready before proceeding so placement is confirmed.
    ray.get([r.ping.remote() for r in reservers])
    psrl_logger.info(
        f"[PSRL] Reserved {len(reservers)} GPU slot(s) across "
        f"{len(excess_nodes)} excess node(s) "
        f"(cluster={cluster_nnodes}, job needs={total_nnodes})."
    )
    return reservers


# Define a function to run the PPO-like training process
def run_ppo(config, task_runner_class=None) -> None:
    """Initialize Ray cluster and run distributed PPO training process.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed PPO training including Ray initialization settings,
                model paths, and training hyperparameters.
        task_runner_class: For recipe to change TaskRunner.
    """
    _configure_runtime_env_flags(config)

    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env(config)
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        if config.transfer_queue.enable:
            # Add runtime environment variables for transfer queue
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        psrl_logger.info(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    # NOTE(claude): Keep the handle list alive for the entire job lifetime so Ray
    # does not garbage-collect the reservation actors before the job finishes.
    _slot_reservers = _reserve_excess_nodes(config)

    if task_runner_class is None:
        task_runner_class = ray.remote(num_cpus=1)(TaskRunner)  # please make sure main_task is not scheduled on head

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))
    ray.shutdown()

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


def colocate_bundle_fractions(psrl_config) -> tuple[float, float, float]:
    """Ray GPU-share per bundle for the (train, rollout, validate) worker groups.

    Fractions are scheduling shares, not memory splits: every worker still sees the whole
    card. ``(0.9, 1.0, 0.1)`` is validate-only colocation, where rollout has its own GPUs.
    With ``colocate_rollout_and_train`` all three roles share one node, and the values are
    chosen so each card hosts at most one worker of each role while they sum to no more
    than one GPU:

    * two train bundles do not fit (``2 * 0.6 > 1``),
    * a second rollout bundle does not fit beside a train one (``0.6 + 2 * 0.3 > 1``),
    * two validate bundles do not fit in the remainder (``0.6 + 0.3 + 2 * 0.09 > 1``).

    Raises:
        ValueError: the fractions exceed one GPU per card, or rollout colocation is
            requested without validate colocation (the two share the switch that frees
            the trainer before validation runs).
    """
    colocate_validate = bool(psrl_config.get("colocate_validate_and_train", False))
    if bool(psrl_config.get("colocate_rollout_and_train", False)):
        if not colocate_validate:
            raise ValueError("colocate_rollout_and_train requires colocate_validate_and_train")
        train = float(psrl_config.get("colocate_train_fraction", 0.6))
        rollout = float(psrl_config.get("colocate_rollout_fraction", 0.3))
        validate = float(psrl_config.get("colocate_validate_fraction", 0.09))
        total = train + rollout + validate
        if total > 1.0 + 1e-9:
            raise ValueError(
                "colocate bundle fractions must sum to at most 1 GPU per card, got "
                f"train={train} + rollout={rollout} + validate={validate} = {total}"
            )
        return train, rollout, validate
    train = 0.9 if colocate_validate else 1.0
    return train, 1.0, (1.0 - train if colocate_validate else 0.0)


@ray.remote(num_cpus=1)
class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.

    Attributes:
        role_worker_mapping: Dictionary mapping Role enums to Ray remote worker classes
        mapping: Dictionary mapping Role enums to resource pool IDs for GPU allocation
    """

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker (backend selected via config.actor.strategy)."""
        from psrl.workers.gen.vllm_rollout import PSRL_ServerAdapter
        from psrl.workers.train.engine_train_worker import PSRL_EngineTrainWorker as PSRL_TrainWorker

        self.role_worker_mapping[PSRL_Role.Actor] = ray.remote(PSRL_TrainWorker)
        self.role_worker_mapping[PSRL_Role.Rollout] = ray.remote(PSRL_ServerAdapter)
        if config.psrl.colocate_validate_and_train:
            self.role_worker_mapping[PSRL_Role.Validate] = ray.remote(PSRL_ServerAdapter)
        self.mapping[PSRL_Role.Actor] = ["train_pool"]

    def add_critic_worker(self, config):
        """
        Add a critic worker backed by veRL's unified training engine.
        """
        from psrl.workers.train.engine_train_worker import PSRL_CriticTrainWorker

        if need_critic(config):
            self.role_worker_mapping[PSRL_Role.Critic] = ray.remote(PSRL_CriticTrainWorker)
            self.mapping[PSRL_Role.Critic] = ["train_pool"]

    def add_ref_policy_worker(self, config):
        """Add a reference policy worker when KL loss or KL reward is enabled."""
        if not need_reference_policy(config):
            return

        from verl.workers.engine_workers import ActorRolloutRefWorker

        self.role_worker_mapping[PSRL_Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
        self.mapping[PSRL_Role.RefPolicy] = ["train_pool"]

    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager."""
        deployment_config = config.psrl.deployment
        # AGENT(VERL): use train_pool instead of global_pool
        train_pool_id = "train_pool"
        (
            train_bundle_resource_num,
            rollout_bundle_resource_num,
            validate_bundle_resource_num,
        ) = colocate_bundle_fractions(config.psrl)
        resource_pool_spec = {
            train_pool_id: [deployment_config.train_ngpus_per_node] * deployment_config.train_nnodes,
        }
        # Unequal bundle fractions prevent two colocated workers from claiming one GPU.
        resource_num_per_bundle = {
            train_pool_id: train_bundle_resource_num,
        }

        # Set the resource pool spec for each rollout instance.
        total_rollout_gpus = 0
        if deployment_config.elastic_rm.enable:
            rollout_pool_id_list = ["shared_rollout_pool"]
        else:
            rollout_pool_id_list = [f"rollout_pool_{i}" for i in range(deployment_config.n_rollout_instances)]

        # If heterogeneous rollout is enabled, we will use the heterogeneous rollout configuration.
        if deployment_config.heterogeneous_rollout.enable:
            heterogeneous_deployment_config = deployment_config.heterogeneous_rollout
            assert (
                len(heterogeneous_deployment_config.rollout_nnodes_per_instance)
                == heterogeneous_deployment_config.n_rollout_instances
            ), "The number of rollout nnodes per instance must match the number of rollout instances."
            assert (
                len(heterogeneous_deployment_config.rollout_ngpus_per_node_per_instance)
                == heterogeneous_deployment_config.n_rollout_instances
            ), "The number of rollout ngpus per node per instance must match the number of rollout instances."
            assert (
                len(heterogeneous_deployment_config.tensor_model_parallel_size_per_instance)
                == heterogeneous_deployment_config.n_rollout_instances
            ), "The number of tensor model parallel size per instance must match the number of rollout instances."
            assert (
                len(heterogeneous_deployment_config.pipeline_model_parallel_size_per_instance)
                == heterogeneous_deployment_config.n_rollout_instances
            ), "The number of pipeline model parallel size per instance must match the number of rollout instances."

            for i in range(deployment_config.n_rollout_instances):
                rollout_pool_id = rollout_pool_id_list[i]
                resource_pool_spec[rollout_pool_id] = [
                    heterogeneous_deployment_config.rollout_ngpus_per_node_per_instance[i]
                ] * heterogeneous_deployment_config.rollout_nnodes_per_instance[i]
        elif deployment_config.elastic_rm.enable:
            total_rollout_gpus = (
                deployment_config.elastic_rm.shared_ngpus_per_node * deployment_config.elastic_rm.shared_nnodes
            )
            deployment_config.n_rollout_instances = total_rollout_gpus // (
                config.gen_actor_rollout_ref.rollout.tensor_model_parallel_size
                * config.gen_actor_rollout_ref.rollout.pipeline_model_parallel_size
                * config.gen_actor_rollout_ref.rollout.data_parallel_size
            )
            psrl_logger.info(
                f"[Elastic RM] Maximum number of rollout instances = {deployment_config.n_rollout_instances}"
            )
            resource_pool_spec["shared_rollout_pool"] = [
                deployment_config.elastic_rm.shared_ngpus_per_node
            ] * deployment_config.elastic_rm.shared_nnodes
            rollout_pool_id_list = ["shared_rollout_pool"]
        else:
            for i in range(deployment_config.n_rollout_instances):
                rollout_pool_id = rollout_pool_id_list[i]
                resource_pool_spec[rollout_pool_id] = [
                    deployment_config.rollout_ngpus_per_node_per_instance
                ] * deployment_config.rollout_nnodes_per_instance

        for rollout_pool_id in rollout_pool_id_list:
            resource_num_per_bundle[rollout_pool_id] = rollout_bundle_resource_num

        # Set the resource pool spec for each validation instance.
        if config.psrl.colocate_validate_and_train:
            val_pool_id_list = [f"validate_pool_{i}" for i in range(deployment_config.n_validate_instances)]
            for i in range(deployment_config.n_validate_instances):
                validate_pool_id = val_pool_id_list[i]
                resource_pool_spec[validate_pool_id] = [
                    deployment_config.validate_ngpus_per_node_per_instance
                ] * deployment_config.validate_nnodes_per_instance
                resource_num_per_bundle[validate_pool_id] = validate_bundle_resource_num
            self.mapping[PSRL_Role.Validate] = val_pool_id_list

        self.mapping[PSRL_Role.Rollout] = rollout_pool_id_list

        # Reward model resource pool
        total_reward_pool_id_list = [] if not deployment_config.elastic_rm.enable else ["shared_rollout_pool"]
        for reward_model in resolve_active_managers(config.reward):
            if reward_model.reward_loop_type != "gen":
                continue
            if deployment_config.elastic_rm.enable:
                reward_model.num_replicas = total_rollout_gpus // (
                    reward_model.rollout.tensor_model_parallel_size
                    * reward_model.rollout.pipeline_model_parallel_size
                    * reward_model.rollout.data_parallel_size
                )
                psrl_logger.info(
                    f"[Elastic RM] Maximum number of reward model"
                    f"({reward_model.reward_model_name}) instances = {reward_model.num_replicas}"
                )
            elif reward_model.enable_resource_pool:
                if reward_model.n_gpus_per_node <= 0:
                    raise ValueError("reward_model.n_gpus_per_node must be greater than 0")
                if reward_model.nnodes <= 0:
                    raise ValueError("reward_model.nnodes must be greater than 0")

                reward_model_instances = reward_model.get("num_replicas", 1)
                reward_model_name = reward_model.get("reward_model_name", reward_model.model.path.split("/")[-1])
                reward_pool_id_list = [f"reward_pool_{reward_model_name}_{i}" for i in range(reward_model_instances)]
                for i in range(reward_model_instances):
                    resource_pool_spec[reward_pool_id_list[i]] = [
                        reward_model.rollout_ngpus_per_instance_per_node
                    ] * reward_model.rollout_nnodes_per_instance
                total_reward_pool_id_list.extend(reward_pool_id_list)
            else:
                raise ValueError("reward_model.enable_resource_pool must be True when elastic_rm.enable is False")

        self.mapping[PSRL_Role.RewardModel] = total_reward_pool_id_list

        from psrl.trainer.ppo.utils import ResourcePoolManager

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=self.mapping,
            resource_num_per_bundle=resource_num_per_bundle,
        )

        print(f"resource_pool_spec = {resource_pool_spec}, mapping = {self.mapping}")

        return resource_pool_manager

    def add_reward_model_worker(self, config):
        """Add reward model worker."""
        from psrl.workers.gen.vllm_rollout import PSRL_ServerAdapter

        self.role_worker_mapping[PSRL_Role.RewardModel] = ray.remote(PSRL_ServerAdapter)

    def add_dummy_worker(self, config):
        from psrl.trainer.ppo.utils import PSRL_DummyWorker

        self.role_worker_mapping[PSRL_Role.DummyPolicy] = ray.remote(PSRL_DummyWorker)
        self.mapping[PSRL_Role.DummyPolicy] = ["train_pool"]

    def run(self, config):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        configure_verl_logging()

        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        from concurrent.futures import ThreadPoolExecutor

        # Import `flashinfer.comm` here, before the workers are added, and not only at
        # module scope.
        #
        # The module-level preload is not sufficient in this actor, and the reason is
        # ordering rather than precedence: `add_*_worker` below imports `megatron.bridge`,
        # whose chain loads tilelang's `libcudart_stub.so` into the process; vLLM's later
        # lazy `flashinfer.comm` import then resolves `cudaDeviceReset` against that stub
        # and dies with "undefined symbol". Verified in an actor: importing
        # `megatron.bridge` first makes a following `flashinfer.comm` fail; importing
        # `flashinfer.comm` first makes it succeed.
        #
        # `LD_PRELOAD` of the real `libcudart.so.12` does *not* fix it either: the stub is
        # dlopen'd after exec, so preloading cannot keep it out. Measured -- with the real
        # runtime mapped (`/proc/self/maps` confirms) the failure still reproduces.
        #
        # Doing it here rather than relying on module import order is what makes it hold:
        # these `add_*_worker` calls run inside a ThreadPoolExecutor, so the imports that
        # pull in megatron happen on a worker thread, where module-level ordering in this
        # file says nothing about when they run.
        try:
            import flashinfer.comm  # noqa: F401
        except Exception as _flashinfer_exc:  # pragma: no cover - optional on CPU-only runs
            import warnings as _warnings

            _warnings.warn(
                f"flashinfer.comm preload failed ({type(_flashinfer_exc).__name__}: "
                f"{_flashinfer_exc}); a later lazy import of it will fail the same way",
                stacklevel=2,
            )

        # Overlap tq.init (network/storage I/O, ~38s) with add_workers (Python imports, ~79s)
        # since they have no shared dependencies. This reduces total from ~117s to ~79s.
        with ThreadPoolExecutor(max_workers=1) as executor:
            tq_future = executor.submit(tq.init, config.transfer_queue)
            self.add_actor_rollout_worker(config)
            self.add_critic_worker(config)
            self.add_ref_policy_worker(config)

            # AGENT(VERL): PSRL use reward model worker.
            self.add_reward_model_worker(config)

            # NOTE(linsh): Add a dummy worker so Ray does not classify fused workers as asynchronous.
            self.add_dummy_worker(config)

            tq_future.result()

        trainer = None
        try:
            resource_pool_manager = self.init_resource_pool_mgr(config)

            # NOTE(linsh): Lazily import `PSRL_RayPPOTrainer` here to avoid implicit `ray.init()`
            # during initialization of nixl modules.
            from psrl.trainer.ppo.ray_trainer import PSRL_RayPPOTrainer

            # Initialize the PPO trainer.
            trainer = PSRL_RayPPOTrainer(
                config=config,
                role_worker_mapping=self.role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
            )
            # Initialize the workers of the trainer.
            trainer.init_workers()
            # Start the training process.
            trainer.fit()
        finally:
            if trainer:
                trainer.replay_buffer.close()
            tq.close()


if __name__ == "__main__":
    seed_everything(0)
    main()
