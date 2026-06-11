import logging
import multiprocessing
import os
import time

import ray
from omegaconf import DictConfig

from psrl.utils.common.http_utils import find_available_port
from psrl.utils.logger import DualOutputHandler
from psrl.workers.gen.rollout_gateway import _run_smg
from psrl.workers.gen.smg_adapter import build_reward_router_args

psrl_logger = logging.getLogger(__name__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@ray.remote
class RewardModelGateway:
    """
    Launches a dedicated smg router process for a single named reward model.

    Key differences from RolloutGateway:
    - routing policy: ``round_robin`` (no cost model, no staleness tracking)
    - no PS manager integration
    - ``enable_routing_loop=False``
    - port range starts at 8200 (avoids collision with rollout gateway at 8100)
    - one instance per reward model name
    """

    def __init__(self, config: DictConfig, model_name: str) -> None:
        self.config = config
        self.model_name = model_name
        self.smg_ip: str | None = None
        self.smg_port: int | None = None
        self.smg_url: str | None = None
        self.router_process: multiprocessing.Process | None = None

        self.log_prefix = f"RewardModelGateway-{model_name}"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))
        psrl_logger.info("Initialized RewardModelGateway for model_name=%s", model_name)

    def _init_router_args(self):
        """Build the CLI args namespace passed to smg RouterArgs."""
        return build_reward_router_args(
            self.config,
            self.smg_ip,
            self.smg_port,
            prometheus_port=find_available_port(base_port=4100),
        )

    def launch_router(self) -> str:
        """Start the smg subprocess and return the gateway HTTP URL."""
        if self.smg_url is not None:
            return self.smg_url

        self.smg_ip = ray.util.get_node_ip_address().strip("[]")
        self.smg_port = find_available_port(base_port=8300)  # 8100=rollout main, 8200=rollout session

        router_args = self._init_router_args()

        self.router_process = multiprocessing.Process(
            target=_run_smg,
            args=(router_args,),
            daemon=True,
        )
        self.router_process.start()
        time.sleep(3)
        assert self.router_process.is_alive(), (
            f"smg router for reward model '{self.model_name}' failed to start (port {self.smg_port})"
        )

        self.smg_url = f"http://{self.smg_ip}:{self.smg_port}"
        psrl_logger.info("RewardModelGateway for '%s' launched at %s", self.model_name, self.smg_url)
        return self.smg_url

    def shutdown_router(self) -> None:
        """Terminate the smg subprocess if running."""
        if self.router_process is not None and self.router_process.is_alive():
            self.router_process.terminate()
            self.router_process.join()
            psrl_logger.info("RewardModelGateway for '%s' shut down.", self.model_name)
