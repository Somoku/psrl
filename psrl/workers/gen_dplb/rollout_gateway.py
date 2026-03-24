import logging
import multiprocessing
import os
import time

import ray
from omegaconf import DictConfig

from psrl.utils.common.http_utils import find_available_port
from psrl.utils.logger import DualOutputHandler

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


def _run_smg(args):
    """Entry point for the smg router subprocess.

    This function is the target of ``multiprocessing.Process`` and runs
    ``launch_router()`` from the ``smg`` Python binding.  It must
    be a module-level function so that it can be pickled by multiprocessing.
    """
    try:
        from smg.launch_router import launch_router

        router = launch_router(args)
        if router is None:
            return 1
        return 0
    except Exception as e:
        psrl_logger.error(e)
        return 1


@ray.remote
class RolloutGateway:
    def __init__(
        self,
        config: DictConfig,
        ps_manager_grpc_ip,
        ps_manager_grpc_port,
    ):
        self.config = config

        # Initialize smg if enabled
        self.smg_ip = None
        self.smg_port = None
        self.smg_url = None
        self.smg_request_timeout_secs = None
        self.router_process = None

        self.ps_manager_grpc_ip = ps_manager_grpc_ip
        self.ps_manager_grpc_port = ps_manager_grpc_port

        # Build logger
        self.log_prefix = "RolloutGateway"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))
        psrl_logger.info("Initialized RolloutGateway")

    def _cfg_get(self, path: str, default=None):
        """Safely fetch nested OmegaConf fields via dot path."""
        node = self.config
        for part in path.split("."):
            if node is None or not hasattr(node, part):
                return default
            node = getattr(node, part)
        return node if node is not None else default

    def _estimate_balanced_concurrent_seqs_per_instance(self) -> int:
        """Mirror Python router-side balanced concurrency estimation logic."""
        rollout_n = int(self._cfg_get("psrl.rollout_n", 1))
        n_rollout_instances = int(self._cfg_get("psrl.deployment.n_rollout_instances", 1))
        n_rollout_instances = max(1, n_rollout_instances)

        if bool(self._cfg_get("psrl.redundant_rollout.enable", False)):
            redundant_global_batch_size = self._cfg_get("psrl.redundant_rollout.redundant_global_batch_size", None)
            if redundant_global_batch_size is not None:
                return max(
                    1,
                    int(redundant_global_batch_size) * rollout_n // n_rollout_instances,
                )

        staleness_buffer_entries = int(self._cfg_get("psrl.staleness_buffer_entries", 512))
        return max(1, staleness_buffer_entries * rollout_n // n_rollout_instances)

    def _init_router_args(self):
        import argparse

        from smg.launch_router import RouterArgs

        routing_method = str(self._cfg_get("psrl.routing_strategy.method", "request_num_balance"))
        enable_routing_loop = routing_method in {
            "request_num_balance",
            "throughput_optimal",
            "throughput_optimal_with_budget",
        }

        balanced_concurrent = self._estimate_balanced_concurrent_seqs_per_instance()
        max_prompt_length = int(
            self._cfg_get(
                "data.max_prompt_length",
                self._cfg_get("rollout.prompt_length", 8192),
            )
        )

        cli_args = argparse.Namespace(
            # server
            host=self.smg_ip,
            port=self.smg_port,
            dp_aware=True,
            connection_mode="grpc",
            # pd disaggregation
            pd_disaggregation=False,
            prefill=None,
            decode=None,
            # routing
            policy=routing_method,
            prefill_policy=None,
            decode_policy=None,
            disable_retries=True,
            policy_balanced_concurrent_seqs_per_instance=balanced_concurrent,
            policy_max_concurrent_seqs_per_instance=int(
                self._cfg_get("psrl.routing_strategy.max_concurrent_seqs_per_instance", 1024)
            ),
            policy_cost_model_path=self._cfg_get("psrl.routing_strategy.cost_model_path", None),
            policy_max_num_waiting_reqs_after_preemption=int(
                self._cfg_get("psrl.routing_strategy.max_num_waiting_reqs_after_preemption", 1000)
            ),
            policy_delta_throughput_threshold=float(
                self._cfg_get("psrl.routing_strategy.delta_throughput_threshold", 0.5)
            ),
            policy_max_prompt_length=max_prompt_length,
            policy_request_budget=int(self._cfg_get("psrl.routing_strategy.request_budget", 1024)),
            enable_routing_loop=enable_routing_loop,
            enable_multi_priority_queue=bool(
                self._cfg_get("psrl.routing_strategy.enable_multi_priority_queue", False)
            ),
            psrl_enable_group_sampling_on_multi_instances=bool(
                self._cfg_get("psrl.routing_strategy.enable_group_sampling_on_multi_instances", False)
            ),
            # psrl
            psrl_check_interval_ms=int(self._cfg_get("psrl.routing_strategy.check_interval_in_ms", 10)),
            psrl_ps_manager_ip=str(self.ps_manager_grpc_ip),
            psrl_ps_manager_grpc_port=int(self.ps_manager_grpc_port),
            psrl_request_sort_indicator=str(
                self._cfg_get("psrl.routing_strategy.request_sort_indicator", "short_length")
            ),
            psrl_candidate_sort_indicator=str(
                self._cfg_get("psrl.routing_strategy.candidate_sort_indicator", "version")
            ),
            psrl_snapshot_staleness_threshold_in_ms=int(
                self._cfg_get("psrl.routing_strategy.snapshot_staleness_threshold_in_ms", 1000)
            ),
            psrl_max_num_waiting_reqs_after_preemption=int(
                self._cfg_get("psrl.routing_strategy.max_num_waiting_reqs_after_preemption", 1000)
            ),
            psrl_mig_enable=bool(self._cfg_get("psrl.sync_and_mig_strategy.mig.enable", False)),
            # service discovery
            service_discovery=False,
            # observability / request
            prometheus_port=find_available_port(base_port=4000),
            request_timeout_secs=2**64 - 1,  # u64::MAX — effectively unlimited
            # log / auth / tls
            log_level="info",
            log_dir=self.config.psrl.logging_path,
            api_key=None,
            disable_health_check=True,
        )

        router_args = RouterArgs.from_cli_args(cli_args, use_router_prefix=False)
        return router_args

    def launch_router(self) -> str:
        if self.smg_url is not None:
            return

        # Get host from Ray actor runtime context
        self.smg_ip = ray.util.get_node_ip_address().strip("[]")

        # Find an available port automatically
        self.smg_port = find_available_port(base_port=8100)

        router_args = self._init_router_args()

        self.router_process = multiprocessing.Process(
            target=_run_smg,
            args=(router_args,),
        )
        self.router_process.daemon = True  # Set the process as a daemon
        self.router_process.start()
        # Wait 3 seconds
        time.sleep(3)
        assert self.router_process.is_alive()
        psrl_logger.info("Router launched at %s:%s", self.smg_ip, self.smg_port)
        self.smg_url = f"http://{self.smg_ip}:{self.smg_port}"
        return self.smg_url

    def shutdown_router(self):
        if self.smg_url is None:
            return

        self.router_process.terminate()
        self.router_process.join()
        psrl_logger.info("Router process terminated")
