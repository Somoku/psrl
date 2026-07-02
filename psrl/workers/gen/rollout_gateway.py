import logging
import multiprocessing
import os
import time

import ray
from omegaconf import DictConfig

from psrl.utils.common.http_utils import find_available_port
from psrl.utils.logger import DualOutputHandler
from psrl.workers.gen.smg_adapter import build_rollout_router_args

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


@ray.remote(num_cpus=0)
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
        self.session_router_process = None
        self.session_router_url = None

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

    def _estimate_http_client_concurrency(self) -> int:
        """Estimate the shared SMG HTTP budget for this rollout gateway."""
        server_max_concurrency = int(self._cfg_get("psrl.rollout_gateway.server_max_concurrency", 256))
        n_rollout_instances = int(self._cfg_get("psrl.deployment.n_rollout_instances", 1))
        colocate_validate = bool(self._cfg_get("psrl.colocate_validate_and_train", False))
        n_validate_instances = (
            int(self._cfg_get("psrl.deployment.n_validate_instances", 0)) if colocate_validate else 0
        )
        active_instances = max(1, n_rollout_instances + n_validate_instances)
        return server_max_concurrency * active_instances

    def _init_router_args(self):
        ps_manager_addr = f"{self.ps_manager_grpc_ip}:{int(self.ps_manager_grpc_port)}"
        return build_rollout_router_args(self.config, self.smg_ip, self.smg_port, ps_manager_addr)

    def launch_router(self) -> str:
        if self.smg_url is not None:
            return self.smg_url

        # Get host from Ray actor runtime context
        self.smg_ip = ray.util.get_node_ip_address().strip("[]")

        # Find an available port automatically
        self.smg_port = find_available_port(base_port=8100)

        router_args = self._init_router_args()

        # Set per-module Rust log filter for the SMG gateway subprocess.
        # EnvFilter::try_from_default_env() in SMG's init_logging reads RUST_LOG
        # before falling back to the configured log_level, so this takes precedence.
        rust_log_filter = str(self._cfg_get("psrl.rollout_gateway.rust_log_filter", ""))
        if rust_log_filter:
            os.environ["RUST_LOG"] = rust_log_filter

        self.router_process = multiprocessing.Process(
            target=_run_smg,
            args=(router_args,),
            daemon=True,
        )
        self.router_process.start()
        # Wait 3 seconds
        time.sleep(3)
        assert self.router_process.is_alive()
        psrl_logger.info("Router launched at %s:%s", self.smg_ip, self.smg_port)
        self.smg_url = f"http://{self.smg_ip}:{self.smg_port}"
        return self.smg_url

    def launch_session_router(self) -> str:
        """Launch the SessionRouter process alongside the SMG router.

        Returns:
            str: The session router URL.
        """
        if not self.smg_url:
            raise RuntimeError("SMG router must be launched before session router")

        from psrl.utils.common.http_utils import find_available_port

        session_port = find_available_port(base_port=8200)
        session_ip = self.smg_ip
        session_client_concurrency = self._estimate_http_client_concurrency()

        def _run_session_router(smg_url, host, port, client_concurrency, logging_path):
            import uvicorn

            from psrl.workers.gen.session_router import SessionRouter, psrl_logger as session_logger

            if logging_path:
                session_logger.addHandler(DualOutputHandler(logging_path, "SessionRouter"))

            router = SessionRouter(
                smg_url=smg_url,
                client_concurrency=client_concurrency,
            )
            uvicorn.run(router.app, host=host, port=port, log_level="warning")

        self.session_router_process = multiprocessing.Process(
            target=_run_session_router,
            args=(self.smg_url, session_ip, session_port, session_client_concurrency, self.config.psrl.logging_path),
        )
        self.session_router_process.daemon = True
        self.session_router_process.start()
        time.sleep(1)
        assert self.session_router_process.is_alive(), "Session router failed to start"
        self.session_router_url = f"http://{session_ip}:{session_port}"
        psrl_logger.info("Session router launched at %s", self.session_router_url)
        return self.session_router_url

    def shutdown_router(self):
        if self.session_router_process is not None:
            self.session_router_process.terminate()
            self.session_router_process.join()
            psrl_logger.info("Session router process terminated")
            self.session_router_process = None
            self.session_router_url = None

        if self.smg_url is None:
            return

        self.router_process.terminate()
        self.router_process.join()
        psrl_logger.info("Router process terminated")
