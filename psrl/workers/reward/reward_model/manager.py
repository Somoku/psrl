import asyncio
import logging
import os

import ray
from omegaconf import DictConfig
from verl.single_controller.ray import RayWorkerGroup
from verl.utils import hf_tokenizer, omega_conf_to_dataclass
from verl.utils.fs import copy_to_local
from verl.workers.config import HFModelConfig

from psrl.workers.config import RolloutConfig
from psrl.workers.gen_dplb.vllm_async_server import GenInterface
from psrl.workers.reward.reward_model.coordinator import RewardModelCoordinator
from psrl.workers.reward.reward_model.replica import RewardModelReplica

psrl_logger = logging.getLogger(__name__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class RewardModelManager:
    """
    Manages reward model replicas for a single named reward model.

    Lifecycle:
    1. Creates a ``RewardModelCoordinator`` and retrieves its ZMQ status endpoint.
    2. For each replica worker-group, creates a ``RewardModelReplica``,
       calls ``init_model()`` to launch the vLLM HTTP server, then registers
       the server to the smg gateway and to the coordinator.
    3. Exposes ``get_gateway_url()`` for ``GenRewardManager`` to POST requests.
    """

    def __init__(
        self,
        reward_model_name: str,
        config: DictConfig,
        reward_model_config: DictConfig,
        reward_model_wg_list: list[RayWorkerGroup],
        gateway_url: str,
    ) -> None:
        self.reward_model_name = reward_model_name
        self.config = config
        self.reward_model_config = reward_model_config
        self.gateway_url = gateway_url

        # ── Build model config and tokenizer ────────────────────────────────
        model_cfg = reward_model_config.model
        local_path = copy_to_local(model_cfg.path, use_shm=model_cfg.get("use_shm", False))
        self.reward_model_tokenizer = hf_tokenizer(
            local_path,
            trust_remote_code=model_cfg.get("trust_remote_code", False),
        )
        self.hf_model_config = HFModelConfig(
            path=model_cfg.path,
            external_lib=model_cfg.get("external_lib"),
            trust_remote_code=model_cfg.get("trust_remote_code", False),
        )
        self.rollout_config: RolloutConfig = omega_conf_to_dataclass(reward_model_config.rollout)

        # ── Coordinator ──────────────────────────────────────────────────────
        self.reward_model_coordinator = ray.remote(RewardModelCoordinator).remote(
            config,
            reward_model_config,
            rollout_router=self.gateway_url,
        )

        # ── Replicas ─────────────────────────────────────────────────────────
        self.reward_model_wg_list = reward_model_wg_list
        self.replicas: list[RewardModelReplica] = []

        self._initialize_reward_replicas()
        self._register_reward_servers(self.replicas)

        ray.get(self.reward_model_coordinator.start_busy_loop.remote())
        ray.get(self.reward_model_coordinator.set_gateway_url.remote(gateway_url))

        psrl_logger.info(
            "RewardModelManager for '%s' initialized with %d replica(s).",
            reward_model_name,
            len(self.replicas),
        )

    def _initialize_reward_replicas(self):
        status_endpoint: str = ray.get(self.reward_model_coordinator.get_status_sink_endpoint.remote())

        init_tasks = []
        for i, wg in enumerate(self.reward_model_wg_list):
            gen_interface = GenInterface(
                role=f"reward_model_{self.reward_model_name}",
                rollout_replica_idx=i,
                status_endpoint=status_endpoint,
                ps_manager_handle=None,  # reward model: no PS sync
            )
            replica = RewardModelReplica(
                replica_rank=i,
                local_replica_rank=i,
                psrl_config=self.config.psrl,
                config=self.rollout_config,
                model_config=self.hf_model_config,
                gen_interface=gen_interface,
                reward_model_name=self.reward_model_name,
                gpus_per_node=self.reward_model_config.rollout_ngpus_per_instance_per_node,
            )
            init_tasks.append(replica.init_model(wg))
            self.replicas.append(replica)
        self._run_all(init_tasks)

    def _register_reward_servers(self, replicas: list[RewardModelReplica]):
        # Register to gateway
        reg_futures = [replica.servers[0].register_server_to_gateway.remote(self.gateway_url) for replica in replicas]
        worker_ids: list[str] = ray.get(reg_futures)

        # Register to coordinator
        coord_futures = [
            self.reward_model_coordinator.add_worker.remote(
                replica,
                replica.servers[0],
                worker_id,
                replica.data_parallel_size,
                is_validate=False,
                model_version=0,
            )
            for replica, worker_id in zip(replicas, worker_ids)
        ]
        ray.get(coord_futures)

    def get_gateway_url(self) -> str:
        """Return the smg gateway HTTP URL for this reward model."""
        return self.gateway_url

    def get_reward_model_tokenizer(self):
        """Return the reward model tokenizer (for prompt construction in GenRewardManager)."""
        return self.reward_model_tokenizer

    def _run_all(self, tasks: list[asyncio.Task]):
        async def run_all():
            await asyncio.gather(*tasks)

        asyncio.run(run_all())
