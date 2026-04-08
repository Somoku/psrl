import argparse
import logging
import multiprocessing
import os
import time

import ray
from omegaconf import DictConfig

from psrl.utils.common.http_utils import find_available_port
from psrl.utils.logger import DualOutputHandler
from psrl.workers.gen_dplb.rollout_gateway import _run_smg

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
        from smg.launch_router import RouterArgs

        cli_args = argparse.Namespace(
            # server
            host=self.smg_ip,
            port=self.smg_port,
            dp_aware=False,
            connection_mode="grpc",
            # pd disaggregation
            pd_disaggregation=False,
            prefill=None,
            decode=None,
            # routing — round-robin, no cost model
            policy="round_robin",
            prefill_policy=None,
            decode_policy=None,
            disable_retries=True,
            policy_balanced_concurrent_seqs_per_instance=1,
            policy_max_concurrent_seqs_per_instance=1024,
            policy_cost_model_path=None,
            policy_max_num_waiting_reqs_after_preemption=1000,
            policy_delta_throughput_threshold=0.5,
            policy_max_prompt_length=32768,
            policy_request_budget=1024,
            # no PSRL staleness routing
            # TODO(linsh): check if require routing loop
            enable_routing_loop=False,
            enable_multi_priority_queue=False,
            psrl_enable_group_sampling_on_multi_instances=False,
            psrl_check_interval_ms=10,
            psrl_ps_manager_ip=None,
            psrl_ps_manager_grpc_port=None,
            psrl_request_sort_indicator="short_length",
            psrl_candidate_sort_indicator="version",
            psrl_snapshot_staleness_threshold_in_ms=1000,
            psrl_max_num_waiting_reqs_after_preemption=1000,
            psrl_mig_enable=False,
            # TITO / service discovery
            enable_tito=False,
            tito_max_entries_per_session=-1,
            service_discovery=False,
            # observability
            prometheus_port=find_available_port(base_port=4100),
            request_timeout_secs=2**64 - 1,
            log_level="warn",
            log_dir=self.config.psrl.logging_path,
            api_key=None,
            disable_health_check=True,
        )

        router_args = RouterArgs.from_cli_args(cli_args, use_router_prefix=False)
        return router_args

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
