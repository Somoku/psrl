import argparse
import asyncio
import inspect
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pprint import pprint
from typing import Any

import aiohttp
import ray
from grpc_reflection.v1alpha import reflection
from omegaconf import OmegaConf
from ray.actor import ActorHandle
from smg_grpc_proto import vllm_engine_pb2, vllm_engine_pb2_grpc
from smg_grpc_servicer.vllm.preemption import PreemptionStatLogger
from smg_grpc_servicer.vllm.servicer import VllmEngineServicer
from torch.distributed.tensor import DTensor
from torch.multiprocessing.reductions import reduce_tensor
from verl.plugin.platform import get_platform
from verl.single_controller.ray import RayWorkerGroup
from verl.utils.device import get_resource_name
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.net_utils import is_valid_ipv6_address
from verl.utils.profiler import build_vllm_profiler_args
from verl.workers.config import HFModelConfig
from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.utils import get_vision_placeholder_token_ids
from verl.workers.rollout.vllm_rollout.utils import (
    build_cli_args_from_config,
    build_mtp_speculative_config,
    get_vllm_max_lora_rank,
)
from verl.workers.rollout.vllm_rollout.vllm_async_server import (
    vLLMHttpServer,
    vLLMReplica,
)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.parser.harmony_utils import get_encoding
from vllm.pooling_params import PoolingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.engine import PauseMode
from vllm.v1.engine.async_llm import AsyncLLM

import grpc
from psrl.sandbox.config import SandboxManagerConfig
from psrl.utils.kv_cache import KVCacheManager, LMCacheConfig
from psrl.utils.logger import (
    DualOutputHandler,
    EventType,
    get_worker_info,
    log_dual_events,
)
from psrl.utils.ray import shared_pull_model_context_async
from psrl.workers.config import RolloutConfig
from psrl.workers.gen.smg_adapter import build_worker_registration_payload, cfg_get, is_cache_aware_method
from psrl.workers.gen.stats_collector import DPLBStatCollector
from psrl.workers.gen.utils import DEFAULT_MAX_CONNECTIONS, DEFAULT_TIMEOUT
from psrl.workers.gen.zmq_queue import ZMQPushQueue

if os.getenv("VERL_USE_GPT_OSS", "0") == "1":
    get_encoding()

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@dataclass
class GenInterface:
    """Info for the PSRL GenWorker."""

    role: str
    rollout_replica_idx: int
    ps_manager_handle: ray.actor.ActorHandle | None = None  # None for reward model (no PS sync)
    status_endpoint: str | None = None  # None / "" → no ZMQ status reporting (reward model path)


class PSRL_vLLMHttpServer(vLLMHttpServer):
    def __init__(
        self,
        psrl_config,
        config: RolloutConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
        cuda_visible_devices: str,
        gen_interface: GenInterface,
    ):
        super().__init__(
            config,
            model_config,
            rollout_mode,
            workers,
            replica_rank,
            node_rank,
            gpus_per_node,
            nnodes,
            cuda_visible_devices,
        )

        # model weights will be loaded by pulling from ps
        self.config.load_format = "dummy"

        self.psrl_config = psrl_config
        self.gen_interface = gen_interface
        self.status_queue = None

        self.stat_collector = None
        self.curr_rollout_instance_model_version = None

        # Async event management
        self._is_init_model = asyncio.Event()
        self._is_init_nixl_client = asyncio.Event()

        # NIXL
        self.nixl_storage_client = None
        self.unified_state_dict = None
        self.unified_sharding_dict = None

        # NIXL cache
        self._cached_ps_nixl_agent_names = None
        self._cached_ps_nixl_gen_storage_client_names = None

        # Gateway HTTP client (connection pooled)
        self._gateway_client: aiohttp.ClientSession | None = None
        self._max_connections = DEFAULT_MAX_CONNECTIONS
        self._timeout = DEFAULT_TIMEOUT

        # For async model pulling
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        # Created before vLLM engine init so LMCache env vars and engine args
        # are visible to vLLM, then attached to the live engine in run_server().
        self.kv_cache_manager: KVCacheManager | None = None

        # NOTE(linsh): Detect pooling models before `generate` selects the inference path.
        self.is_pooling_model = config.get("runner", "generate") == "pooling"
        # Populated after engine startup. Generative models retain `None`.
        self.pooling_params: PoolingParams | None = None

    async def is_init_model(self):
        self._is_init_model.set()

    async def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ):
        await self.engine.collective_rpc(
            method=method,
            timeout=timeout,
            args=args,
            kwargs=kwargs,
        )

    def _get_worker_extension_cls(self) -> str:
        return "psrl.workers.gen.vllm_extension.vLLMWorkerExtension"

    def _build_kv_cache_manager(self) -> KVCacheManager:
        """Build and configure `KVCacheManager` before vLLM engine initialization."""
        lmcache_raw = (
            OmegaConf.to_container(
                self.psrl_config.get("lmcache", OmegaConf.create()),
                resolve=True,
            )
            or {}
        )

        lmcache_instance_id = f"psrl_instance_{self.get_replica_idx()}"
        if lmcache_raw.get("enable_p2p", False):
            lmcache_raw.setdefault(
                "controller_host",
                str(self.psrl_config.get("ps_manager_ip", "127.0.0.1")),
            )
            lmcache_raw["lmcache_instance_id"] = lmcache_instance_id

        # Off-GPU cache scoring requires LMCache events in vLLM's KV event stream.
        lmcache_raw["enable_kv_events"] = bool(
            lmcache_raw.get("enable", False)
            and is_cache_aware_method(self.psrl_config.rollout_coordination.routing_strategy.method)
            and float(
                cfg_get(
                    self.psrl_config,
                    "rollout_coordination.routing_strategy.cache_aware_policy.lmcache_overlap_weight",
                    0.0,
                )
            )
            > 0.0
        )

        lmcache_cfg = LMCacheConfig(**lmcache_raw)
        if lmcache_cfg.enable_p2p:
            self._configure_lmcache_p2p_runtime(lmcache_cfg)

        kv_cache_manager = KVCacheManager(lmcache_cfg)
        kv_cache_manager.set_instance_id(self.get_replica_idx())
        kv_cache_manager.apply_env_vars()
        return kv_cache_manager

    def _configure_lmcache_p2p_runtime(self, lmcache_cfg: LMCacheConfig) -> None:
        """Fill runtime-only LMCache P2P fields for this server actor."""
        from psrl.utils.common.http_utils import get_host_info
        from psrl.utils.nixl.port_scanner import get_port_scanner

        # A server actor owns all local vLLM workers for this replica node. In
        # DP mode, each local DP x TP worker may create an LMCache worker.
        lmcache_cfg.num_kv_workers = max(1, len(self.workers))
        _, worker_ip = get_host_info()
        lmcache_cfg.worker_host = worker_ip

        port_scanner = get_port_scanner(worker_ip)
        num_ports_needed = lmcache_cfg.num_kv_workers * 3
        ports = ray.get([port_scanner.find_free_port.remote() for _ in range(num_ports_needed)])
        n = lmcache_cfg.num_kv_workers
        lmcache_cfg.allocated_worker_ports = ports[0:n]
        lmcache_cfg.allocated_p2p_init_ports = ports[n : 2 * n]
        lmcache_cfg.allocated_p2p_lookup_ports = ports[2 * n : 3 * n]

    def _build_kv_events_args(self) -> dict:
        """
        Build vLLM `kv_events_config` for SMG event-driven routing.

        Returns an args fragment enabling the native ZMQ KV-event publisher when
        the rollout router uses the cache-aware strategy. Returns ``{}`` (no
        publisher) otherwise, so the feature is zero-cost when unused.

        Each DP rank uses an independent ZMQ port derived from `base_port`.
        """
        routing_method = self.psrl_config.rollout_coordination.routing_strategy.method
        if not is_cache_aware_method(routing_method):
            return {}

        # Allocate a free base port on this server's host (same PortScanner used
        # for LMCache worker ports) so the publisher endpoint never collides.
        from psrl.utils.common.http_utils import get_host_info
        from psrl.utils.nixl.port_scanner import get_port_scanner

        _, worker_ip = get_host_info()
        port_scanner = get_port_scanner(worker_ip)
        base_port = ray.get(port_scanner.find_free_port.remote())

        instance_id = f"psrl_instance_{self.get_replica_idx()}"
        kv_events_config = {
            "enable_kv_cache_events": True,
            "publisher": "zmq",
            "endpoint": f"tcp://*:{base_port}",
            "topic": f"kv@{instance_id}",
        }
        psrl_logger.info(
            f"[KVEvents] Publishing KV cache events on tcp://{worker_ip}:{base_port} "
            f"(topic=kv@{instance_id}, +dp_rank per rank)"
        )
        return {"kv_events_config": kv_events_config}

    async def launch_server(
        self, master_address: str | None = None, master_port: int | None = None, dp_rpc_port: int | None = None
    ):
        """Launch the vLLM gRPC server with PSRL-specific setup.

        AGENT(verl): This method is adapted from the original vLLMHttpServer.launch_server in verl.
        The main differences are:
        1. Additional setup for PSRL features (e.g., rollout scheduler, stat collector, scheduler abort processor).
        2. Use gRPC mode.
        """
        if self.node_rank != 0:
            assert master_address and master_port and dp_rpc_port, (
                "non-master node should provide master_address, master_port and dp_rpc_port"
            )
            self._master_address = master_address
            self._master_port = master_port
            self._dp_rpc_port = dp_rpc_port

        # 1. setup vllm serve cli args
        engine_kwargs = self.config.get("engine_kwargs", {}).get("vllm", {}) or {}
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if self.config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": self.config.get("limit_images")}

        self._preprocess_engine_kwargs(engine_kwargs)

        # Override default generation config from hugging face model config,
        # user can still override them by passing kwargs in each request.
        override_generation_config = self._get_override_generation_config()
        psrl_logger.info(f"override_generation_config: {override_generation_config}")

        psrl_logger.info(f"enable_sleep_mode: {self.config.enable_sleep_mode}")
        if not self.config.enable_sleep_mode:
            from verl.utils.device import set_expandable_segments

            set_expandable_segments(True)

        quantization, hf_overrides = self._apply_quantization()

        compilation_config = engine_kwargs.pop("compilation_config", None) or {}
        if isinstance(compilation_config, str):
            compilation_config = json.loads(compilation_config)
        compilation_config.setdefault("cudagraph_mode", "FULL_AND_PIECEWISE")

        # FULL cuda graph is not yet supported with DCP, downgrade to PIECEWISE
        dcp_size = engine_kwargs.get("decode_context_parallel_size", 1) or 1
        if dcp_size > 1 and compilation_config["cudagraph_mode"] == "FULL_AND_PIECEWISE":
            psrl_logger.warning(
                "FULL cuda graph is not supported with DCP (decode_context_parallel_size=%d), "
                "downgrading cudagraph_mode to PIECEWISE.",
                dcp_size,
            )
            compilation_config["cudagraph_mode"] = "PIECEWISE"
        if self.config.cudagraph_capture_sizes:
            compilation_config["cudagraph_capture_sizes"] = self.config.cudagraph_capture_sizes

        compilation_config = json.dumps(compilation_config)
        args = {
            "grpc": True,  # AGENT(VERL): use gRPC server in PSRL, different from verl
            "dtype": self.config.dtype,
            "load_format": self.config.load_format,
            "skip_tokenizer_init": False,
            "distributed_executor_backend": "mp",
            "worker_extension_cls": self._get_worker_extension_cls(),
            "trust_remote_code": self.model_config.trust_remote_code,
            "max_model_len": self.config.max_model_len,
            "max_num_seqs": self.config.max_num_seqs,
            "enable_chunked_prefill": self.config.enable_chunked_prefill,
            "max_num_batched_tokens": self.config.max_num_batched_tokens,
            "enable_prefix_caching": self.config.enable_prefix_caching,
            "enable_sleep_mode": self.config.enable_sleep_mode,
            "logprobs_mode": self.config.logprobs_mode,
            "enforce_eager": self.config.enforce_eager,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "disable_log_stats": self.config.disable_log_stats,
            "tensor_parallel_size": self.config.tensor_model_parallel_size,
            "seed": self.replica_rank + (self.config.get("seed") or 0),
            "override_generation_config": json.dumps(override_generation_config),
            "quantization": quantization,
            "hf_overrides": hf_overrides,
            "scheduling_policy": self.config.scheduling_policy,
            "compilation_config": compilation_config,
            # AGENT(VERL): thread runner/task through for pooling model support in PSRL
            "runner": self.config.get("runner", "generate"),
            **engine_kwargs,
        }

        self.kv_cache_manager = self._build_kv_cache_manager()
        args.update(self.kv_cache_manager.get_engine_kwargs())

        # SMG consumes the merged GPU and LMCache event stream through localhost.
        kv_events_args = self._build_kv_events_args()
        if kv_events_args:
            args.update(kv_events_args)

        # update profiler args
        profiler_args = build_vllm_profiler_args(
            self.profiler_controller.config, self.profiler_controller.tool_config, self.replica_rank
        )
        args.update(profiler_args)

        if self.config.prometheus.enable:
            if self.config.prometheus.served_model_name:
                # Extract model name from path if it's a full path
                served_model_name = self.config.prometheus.served_model_name
                if "/" in served_model_name:
                    # If it's a full path, extract the last part as model name
                    served_model_name = served_model_name.split("/")[-1]
                args["served_model_name"] = served_model_name

        # MTP is only available for LLM rollout models.
        if self.config.mtp is not None and self.config.mtp.enable and self.config.mtp.enable_rollout:
            args["speculative_config"] = build_mtp_speculative_config(
                self.config.mtp.method,
                self.config.mtp.num_speculative_tokens,
                args.get("speculative_config"),
            )

        # Always report data_parallel_size so SMG's DP discovery step can find it
        # in the gRPC server_info response (required for worker registration).
        args["data_parallel_size"] = self.config.data_parallel_size
        if self.config.data_parallel_size > 1:
            assert self.gpus_per_node % self.config.tensor_model_parallel_size == 0, (
                "gpus_per_node should be divisible by tensor_model_parallel_size"
            )
            data_parallel_size_local = self.gpus_per_node // self.config.tensor_model_parallel_size
            assert len(self.workers) == data_parallel_size_local * self.config.tensor_model_parallel_size, (
                f"num workers ({len(self.workers)}) should be equal to "
                f"dp_size_local ({data_parallel_size_local}) * tp_size ({self.config.tensor_model_parallel_size})"
            )
            dp_args = {
                "data_parallel_size_local": data_parallel_size_local,
                "data_parallel_start_rank": self.node_rank * data_parallel_size_local,
                "data_parallel_address": self._master_address,
                "data_parallel_rpc_port": self._dp_rpc_port,
            }
            args.update(dp_args)

        args.update({"enable_expert_parallel": self.config.expert_parallel_size > 1})

        # used for torch.distributed.init_process_group
        if self.nnodes > 1:
            args.update(
                {
                    "master_addr": self._master_address,
                    "master_port": self._master_port,
                    "node_rank": self.node_rank,
                    "nnodes": self.nnodes,
                    "data_parallel_address": self._master_address,
                    "data_parallel_rpc_port": self._dp_rpc_port,
                }
            )

        # update lora-related args
        lora_rank = self.model_config.lora.get("rank", 0)
        if lora_rank <= 0:
            lora_rank = (
                self.model_config.lora_rank
            )  # FIXME: fallback to lora_rank for now, we should unify lora settings.

        if self.model_config.lora.get("merge", False):
            lora_rank = 0

        if lora_rank > 0:
            lora_args = {
                "enable_lora": True,
                "max_loras": 1,
                "max_lora_rank": get_vllm_max_lora_rank(lora_rank),
            }
            if self.model_config.lora.get("fully_sharded_loras", False):
                lora_args["fully_sharded_loras"] = True
            args.update(lora_args)

        # Routing replay needs vLLM's fixed hybrid-attention routed-expert capture path.
        if self.config.enable_rollout_routing_replay:
            args.update({"enable_return_routed_experts": True})

        # AGENT(VERL): setup rollout scheduler for PSRL
        args["scheduler_cls"] = "psrl.workers.gen.rollout_scheduler.RolloutScheduler"
        args["additional_config"] = {
            "enable_weights_cpu_backup": self.config.enable_weights_cpu_backup,
        }

        server_args = ["serve", self.model_config.path] + build_cli_args_from_config(args)

        if self.replica_rank == 0:
            pprint(server_args)
            psrl_logger.info(f"{server_args=}")

        CMD_MODULES = self._get_cli_modules()
        parser = FlexibleArgumentParser(description=self._get_cli_description())
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        cmds = {}
        for cmd_module in CMD_MODULES:
            new_cmds = cmd_module.cmd_init()
            for cmd in new_cmds:
                cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
                cmds[cmd.name] = cmd
        server_args = parser.parse_args(args=server_args)
        server_args.model = server_args.model_tag
        if server_args.subparser in cmds:
            cmds[server_args.subparser].validate(server_args)

        # 3. launch server
        if self.node_rank == 0:
            await self.run_server(server_args)
        else:
            await self.run_headless(server_args)

        # AGENT(VERL): log server launch completion for PSRL
        if self.node_rank == 0:
            self.log_prefix = f"vLLMHTTPServer_Replica{self.get_replica_idx()}"
            psrl_logger.addHandler(DualOutputHandler(self.psrl_config.logging_path, self.log_prefix))
            psrl_logger.info(f"Initialized on {get_worker_info()}.")

    async def run_server(self, args: argparse.Namespace):
        engine_args = AsyncEngineArgs.from_cli_args(args)
        usage_context = UsageContext.OPENAI_API_SERVER
        vllm_config = engine_args.create_engine_config(usage_context=usage_context)
        vllm_config.parallel_config.data_parallel_master_port = self._dp_master_port
        # AGENT(VERL): wire preemption_notification_threshold into vLLM for the PSRL gateway loopback.
        vllm_config.scheduler_config.preemption_notification_threshold = (
            self.psrl_config.rollout_coordination.routing_strategy.max_num_waiting_reqs_after_preemption
        )
        # Wire prefill composition logging config into SchedulerConfig so RolloutScheduler
        # can read them directly without relying on environment variables.
        vllm_config.scheduler_config.psrl_prefill_composition_enable = (
            self.psrl_config.profile.prefill_composition.enable
        )
        vllm_config.scheduler_config.psrl_logging_path = str(self.psrl_config.logging_path)
        vllm_config.scheduler_config.psrl_replica_idx = self.get_replica_idx()

        fn_args = set(dict(inspect.signature(AsyncLLM.from_vllm_config).parameters).keys())
        kwargs = {}
        if "enable_log_requests" in fn_args:
            kwargs["enable_log_requests"] = engine_args.enable_log_requests
        if "disable_log_stats" in fn_args:
            kwargs["disable_log_stats"] = engine_args.disable_log_stats

        # AGENT(VERL): apply stat logger patch for PSRL
        # NOTE(linsh): enable custom stat collection for PSRL
        self.preemption_queue: asyncio.Queue = asyncio.Queue(maxsize=0)
        self.psrl_preemption_logger = PreemptionStatLogger(
            vllm_config,
            engine_index=0,
            preemption_queue=self.preemption_queue,
        )
        if not self.config.disable_log_stats and self.psrl_config.status_collection.enable:
            self.stat_collector = DPLBStatCollector(
                vllm_config,
                self.psrl_config,
                self.get_replica_idx(),
                self.gen_interface.role,
            )
            self.stat_collector.begin_record()
            _endpoint = self.gen_interface.status_endpoint or ""
            self.status_queue = ZMQPushQueue(_endpoint)
            self.stat_collector.init_output_queue(self.status_queue)
            for data_parallel_rank in range(self.config.data_parallel_size):
                self.stat_collector.record_model_version_update(0, data_parallel_rank)
            kwargs["stat_loggers"] = [self.stat_collector, self.psrl_preemption_logger]
        else:
            kwargs["stat_loggers"] = [self.psrl_preemption_logger]

        engine_client = AsyncLLM.from_vllm_config(vllm_config=vllm_config, usage_context=usage_context, **kwargs)

        # Don't keep the dummy data in memory
        await engine_client.reset_mm_cache()
        await engine_client.collective_rpc(
            method="monkey_patch_model",
            kwargs={
                "vocab_size": len(self.model_config.tokenizer),
                "banned_token_ids": get_vision_placeholder_token_ids(self.model_config.processor),
            },
        )

        if self.replica_rank == 0 and self.node_rank == 0:
            psrl_logger.info(f"Initializing a V1 LLM engine with config: {vllm_config}")

        # AGENT(VERL): use gRPC server instead of HTTP server in PSRL

        self.engine = engine_client
        assert self.kv_cache_manager is not None, "kv_cache_manager must be initialized before engine startup."
        self.kv_cache_manager.attach_engine(engine_client)
        psrl_logger.info("[KVCacheManager]: Engine attached after vLLM server initialization.")

        # self._server_port, self._server_task = await run_unvicorn(app, args, self._server_address)
        self._server_port = await self._start_grpc_server(engine_client)

        # NOTE(linsh): Initialize `PoolingParams` only for pooling models.
        if self.is_pooling_model:
            normalize = self.config.reward_kwargs.get("normalize", False)
            use_activation = self.config.reward_kwargs.get("use_activation", False)
            pooling_task = self.config.get("task", "classify")
            self.pooling_params = PoolingParams(
                normalize=normalize,
                use_activation=use_activation,
                task=pooling_task,
            )
            psrl_logger.info(
                "Initialized PoolingParams for pooling model: normalize=%s, use_activation=%s, task=%s",
                normalize,
                use_activation,
                pooling_task,
            )

    async def _start_grpc_server(self, engine_client: "AsyncLLM") -> int:
        """
        Start the engine gRPC server and return its bound port.

        The bound port is required for subsequent SMG registration.
        """
        start_time = time.time()
        kv_transfer_cfg = self.psrl_config.rollout_coordination.routing_strategy.get("kv_transfer", {})
        kv_transfer_enabled = bool(kv_transfer_cfg.get("enable", False))
        stats_log_interval_s = float(kv_transfer_cfg.get("stats_log_interval_s", 30)) if kv_transfer_enabled else 0.0
        servicer = VllmEngineServicer(
            engine_client,
            start_time,
            preemption_queue=self.preemption_queue,
            kv_cache_manager=self.kv_cache_manager,
            kv_transfer_stats_log_interval_s=stats_log_interval_s,
            enable_kv_event_replay=self.psrl_config.rollout_gateway.enable_kv_event_replay,
        )
        self.grpc_servicer = servicer

        server = grpc.aio.server(
            options=[
                # Model outputs require unlimited message sizes.
                ("grpc.max_send_message_length", -1),
                ("grpc.max_receive_message_length", -1),
                # Allow keepalive pings every 10 seconds without active calls.
                ("grpc.http2.min_recv_ping_interval_without_data_ms", 10000),
                ("grpc.keepalive_permit_without_calls", True),
                # Unlimited ping strikes prevent simultaneous channel setup from
                # terminating the shared HTTP/2 transport.
                ("grpc.http2.max_ping_strikes", 0),
            ],
        )
        vllm_engine_pb2_grpc.add_VllmEngineServicer_to_server(servicer, server)

        # Enable gRPC reflection so ``grpcurl`` and SMG's health-probe can discover services.
        service_names = (
            vllm_engine_pb2.DESCRIPTOR.services_by_name["VllmEngine"].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(service_names, server)

        # Port zero requests a free ephemeral port from the operating system.
        port = server.add_insecure_port(f"{self._server_address}:0")
        await server.start()

        # Keep a reference to the server object so it is not garbage-collected
        # while the process is alive.
        self._grpc_server = server

        psrl_logger.info(
            "gRPC server started on %s:%d (replica=%d, node_rank=%d)",
            self._server_address,
            port,
            self.get_replica_idx(),
            self.node_rank,
        )
        return port

    def _grpc_health_probe_target(self) -> str:
        host = self._server_address
        if host in ("0.0.0.0", "::", "[::]"):
            host = "127.0.0.1"
        return f"{host}:{self._server_port}"

    async def _probe_grpc_health_once(self, rpc_timeout_s: float = 5.0) -> bool:
        channel = grpc.aio.insecure_channel(
            self._grpc_health_probe_target(),
            options=[
                ("grpc.max_send_message_length", -1),
                ("grpc.max_receive_message_length", -1),
            ],
        )
        try:
            stub = vllm_engine_pb2_grpc.VllmEngineStub(channel)
            response = await asyncio.wait_for(
                stub.HealthCheck(vllm_engine_pb2.HealthCheckRequest()),
                timeout=rpc_timeout_s,
            )
            return bool(response.healthy)
        finally:
            await channel.close()

    async def _wait_for_grpc_servicer_ready(self) -> None:
        rollout_gateway_cfg = self.psrl_config.rollout_gateway
        timeout_s = float(getattr(rollout_gateway_cfg, "grpc_registration_health_timeout_s", 300))
        poll_interval_s = float(getattr(rollout_gateway_cfg, "grpc_registration_health_poll_interval_s", 1.0))
        rpc_timeout_s = float(getattr(rollout_gateway_cfg, "grpc_registration_health_rpc_timeout_s", 5.0))

        if self.grpc_servicer is None or self._server_port is None:
            raise RuntimeError("gRPC servicer is not initialized")

        deadline = time.monotonic() + timeout_s
        attempt = 0
        last_error = "unknown"
        while time.monotonic() < deadline:
            attempt += 1
            try:
                if await self._probe_grpc_health_once(rpc_timeout_s=rpc_timeout_s):
                    psrl_logger.info(
                        "gRPC servicer ready for gateway registration: replica=%s target=%s attempt=%d",
                        self.get_replica_idx(),
                        self._grpc_health_probe_target(),
                        attempt,
                    )
                    return
                last_error = "HealthCheck returned healthy=false"
            except Exception as exc:
                last_error = str(exc)

            psrl_logger.debug(
                "Waiting for gRPC servicer before gateway registration: replica=%s target=%s attempt=%d error=%s",
                self.get_replica_idx(),
                self._grpc_health_probe_target(),
                attempt,
                last_error,
            )
            await asyncio.sleep(poll_interval_s)

        raise TimeoutError(
            f"gRPC servicer not ready after {timeout_s:.0f}s "
            f"(replica={self.get_replica_idx()}, target={self._grpc_health_probe_target()}): {last_error}"
        )

    # AGENT(VERL): PSRL-specific async methods for server control and coordination.
    # We add `data_parallel_rank` parameters to these methods to support DP-aware control in PSRL.

    async def is_sleeping(self) -> bool:
        return await self.engine.is_sleeping()

    async def sleep(self, level: int):
        await self.engine.sleep(level)
        await self.engine.reset_encoder_cache()
        if self.psrl_config.tms.range in ["rollout", "all"]:
            # NOTE(linsh): TMS requires an explicit aggressive cache clear.
            aggressive_empty_cache(force_sync=True)

    async def wake_up(self):
        wake_up_tags = ["weights", "kv_cache"]
        if self.psrl_config.tms.enable_cuda_graph:
            wake_up_tags.append("graph")
        await self.engine.wake_up(tags=wake_up_tags)

    async def clear_kv_cache(self):
        await self.engine.reset_prefix_cache(reset_connector=True)
        await self.engine.reset_mm_cache()
        await self.engine.reset_encoder_cache()

    async def pause_generation(
        self,
        mode: PauseMode = "abort",
        wait_for_inflight_requests: bool = False,
        clear_cache: bool = True,
    ):
        await self.engine.pause_generation(
            mode=mode,
            wait_for_inflight_requests=wait_for_inflight_requests,
            clear_cache=clear_cache,
        )

    async def resume_generation(self):
        await self.engine.resume_generation()

    async def close_grpc_generate_admission(self):
        await self.grpc_servicer.close_generate_admission()

    async def open_grpc_generate_admission(self):
        await self.grpc_servicer.open_generate_admission()

    async def pause_for_sync(self):
        # Close the park gate before admission so new requests cannot reach a
        # paused engine.
        self.grpc_servicer.pause_generation_admission()
        await self.close_grpc_generate_admission()
        # KV cache clearing is deferred to pull_model() after weights are updated.
        await self.pause_generation(clear_cache=False)
        psrl_logger.info(f"Generation paused for parameter server sync: replica={self.get_replica_idx()}.")

    async def resume_after_sync(self):
        # Resume the engine before waking parked requests.
        await self.resume_generation()
        await self.open_grpc_generate_admission()
        self.grpc_servicer.resume_generation_admission()
        psrl_logger.info(f"Generation resumed on replica {self.get_replica_idx()}")

    async def fail_sync(self):
        await self.close_grpc_generate_admission()
        await self.pause_generation(clear_cache=False)
        # Failed parked requests must wake and reroute away from the quarantined engine.
        self.grpc_servicer.fail_generation_admission()

    async def wait_for_requests_to_drain(self):
        await self.engine.wait_for_requests_to_drain()

    async def abort_all_requests(
        self, reset_prefix_cache: bool = False, reject_request: bool = False
    ) -> dict[str, Any]:
        """
        Abort all ongoing requests asynchronously.

        This method is used to abort all requests, typically during shutdown or
        when a global interruption is needed.

        Args:
            reset_prefix_cache: Clear the KV cache after aborting.
            reject_request: Fail requests that arrive behind the parked gate instead of letting them
                wait for a resume. The SMG servicer owns admission, so this wakes the parked gate in
                its failing state and the gateway reroutes those requests. A later
                `resume_after_sync` clears the flag.

        Returns:
            The number of requests that were aborted.
        """
        # AGENT(VERL): the implementation is different from verl, skip when bump dependency.

        if reject_request:
            self.grpc_servicer.fail_generation_admission()

        request_states_snapshot = list(self.engine.output_processor.request_states.items())
        request_ids = [req_id for req_id, _ in request_states_snapshot]
        if not request_ids:
            return {"aborted_count": 0, "request_ids": []}

        await self.engine.abort(request_ids)

        # Try to reset prefix cache to ensure clean state
        if reset_prefix_cache:
            await self.clear_kv_cache()

        return {"aborted_count": len(request_ids), "request_ids": request_ids}

    async def abort_requests(self, request_ids: list[str]) -> int:
        """
        Abort specific requests by their IDs asynchronously.

        This method aborts only the specified requests, allowing selective
        interruption based on staleness or other criteria.

        Args:
            request_ids: List of request IDs to abort
        Returns:
            The number of requests that were aborted.
        """
        await self.engine.abort(request_ids)
        return len(request_ids)

    def get_replica_idx(self) -> int:
        return self.gen_interface.rollout_replica_idx

    def get_instance_num(self) -> int:
        return self.engine.engine_core.num_engines

    def get_active_task_num(self, data_parallel_rank: int) -> int:
        # The servicer owns in-flight accounting and tracks admissions across all DP engines.
        return self.grpc_servicer.active_generate_admissions

    async def register_rollout_instances_to_ps(self):
        if self.gen_interface.ps_manager_handle is None:
            return  # reward model: no PS instance registration
        if hasattr(self, "_is_rollout_instance_registered"):
            return
        rollout_instance_ids = [(self.base_worker_id, i) for i in range(self.get_instance_num())]
        await self.gen_interface.ps_manager_handle.register_rollout_instance.remote(rollout_instance_ids)
        self.curr_rollout_instance_model_version = [0] * self.get_instance_num()
        self._is_rollout_instance_registered = True

    async def register_server_to_gateway(self, gateway_url: str) -> str | None:
        if self.node_rank != 0:
            return None
        await self._wait_for_grpc_servicer_ready()
        total_kv_tokens = await self.get_total_kv_cache_tokens()
        # Register to rollout gateway
        gateway_url = gateway_url.rstrip("/")
        if self._gateway_client is None or self._gateway_client.closed:
            connector = aiohttp.TCPConnector(
                limit=self._max_connections,
                limit_per_host=self._max_connections,
                ttl_dns_cache=300,
                enable_cleanup_closed=True,
            )
            timeout = aiohttp.ClientTimeout(total=self._timeout)
            self._gateway_client = aiohttp.ClientSession(connector=connector, timeout=timeout)

        # SMG identifies transfer destinations by LMCache instance ID. Source
        # servicers resolve per-rank peer URLs from their registries.
        lmcache_instance_id = None
        if self.kv_cache_manager is not None and self.kv_cache_manager.config.enable_p2p:
            lmcache_instance_id = self.kv_cache_manager.config.lmcache_instance_id

        payload = build_worker_registration_payload(
            url=f"grpc://{self._server_address}:{self._server_port}",
            model_id=self.model_config.path,
            max_model_len=total_kv_tokens,
            dp_size=self.config.data_parallel_size,
            tp_size=self.config.tensor_model_parallel_size,
            pp_size=self.config.pipeline_model_parallel_size,
            lmcache_instance_id=lmcache_instance_id,
            worker_id=str(self.get_replica_idx()),
        )

        try:
            async with self._gateway_client.post(f"{gateway_url}/workers", json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)
            worker_id = data.get("worker_id")
            if not worker_id:
                raise ValueError(f"Missing worker_id in gateway response: {data}")

            self.base_worker_id = worker_id
            psrl_logger.info(
                "Registered rollout server to gateway: replica=%s, worker_id=%s, addr=%s",
                self.get_replica_idx(),
                worker_id,
                self._server_address,
            )
            return worker_id
        except Exception as e:
            psrl_logger.error(
                "Failed to register server to gateway at %s: %s",
                gateway_url,
                e,
            )
            raise

    async def generate(self, *args, **kwargs):
        """Reject veRL's direct generation path because SMG owns all inference through gRPC."""
        raise RuntimeError("Direct generation is disabled. Send inference requests through the SMG gRPC gateway.")

    ###### NIXL Integration ######

    async def init_nixl_client(self):
        await self._is_init_model.wait()
        await self.collective_rpc(
            method="init_nixl_client",
            args=(
                self.psrl_config.nixl,
                self.get_replica_idx(),
                self.psrl_config.logging_path,
            ),
        )

        self._is_init_nixl_client.set()

    async def nixl_convert_params(self):
        await self._is_init_model.wait()
        await self.collective_rpc(
            method="nixl_convert_params",
            args=(self.model_config,),
        )

    async def nixl_protocol(self, mode: str = "full"):
        await self._is_init_model.wait()
        await self._is_init_nixl_client.wait()
        await self.collective_rpc(
            method="nixl_protocol",
            args=(self.model_config, mode),
        )

    async def nixl_wake_up(self):
        await self._is_init_nixl_client.wait()
        await self.wake_up()
        await self.collective_rpc(method="nixl_register_after_wake_up")

    async def nixl_sleep(self, level: int):
        await self._is_init_nixl_client.wait()
        await self.sleep(level)
        await self.collective_rpc(method="nixl_deregister")

    async def nixl_send_local_info_to(self, dst_agent_names: str | list[str]):
        await self._is_init_nixl_client.wait()
        await self.collective_rpc(
            method="nixl_send_local_info_to",
            args=(dst_agent_names,),
        )

    async def nixl_wait_for_update_infos(self, info_num: int):
        await self._is_init_nixl_client.wait()
        await self.collective_rpc(
            method="nixl_wait_for_update_infos",
            args=(info_num,),
        )

    ###### KV Cache Management ######

    def _assert_kv_cache_manager(self) -> KVCacheManager:
        assert self.kv_cache_manager is not None, "KVCacheManager is not initialized. Call launch_server() first."
        assert self.kv_cache_manager.is_attached, (
            "KVCacheManager engine is not attached. Call launch_server() before KV cache operations."
        )
        return self.kv_cache_manager

    async def kv_pin(self, tokens: list[int], targets: list[str]) -> bool:
        """Pin the cached prefix blocks/chunks for `tokens`."""
        kv_cache_manager = self._assert_kv_cache_manager()
        assert tokens, "tokens must be a non-empty list."
        assert targets, "targets must be a non-empty list."
        assert all(target in ("gpu", "backend") for target in targets), (
            f"Invalid targets: {targets!r}. Must be a subset of ['gpu', 'backend']."
        )
        return await kv_cache_manager.pin(tokens, targets)

    async def kv_unpin(self, tokens: list[int], targets: list[str]) -> bool:
        """Unpin the cached prefix blocks/chunks for `tokens`."""
        kv_cache_manager = self._assert_kv_cache_manager()
        assert tokens, "tokens must be a non-empty list."
        assert targets, "targets must be a non-empty list."
        assert all(target in ("gpu", "backend") for target in targets), (
            f"Invalid targets: {targets!r}. Must be a subset of ['gpu', 'backend']."
        )
        return await kv_cache_manager.unpin(tokens, targets)

    def kv_set_peer_registry(
        self,
        registry: dict[str, list[str]],
        worker_zmq_urls: list[str] | None = None,
    ) -> None:
        """Set peer registry for direct LMCache transfer bypass.

        Args:
            registry: Maps lmcache_instance_id → rank-sorted list of peer_init_url.
            worker_zmq_urls: This replica's rank-sorted local LMCacheWorker ZMQ URLs.
        """
        assert self.kv_cache_manager is not None, "KVCacheManager is not initialized. Call launch_server() first."
        self.kv_cache_manager.set_peer_registry(registry, worker_zmq_urls)

    def set_lmcache_controller_url(self, controller_url: str) -> None:
        """Receive the shared LMCache Controller URL from `RolloutCoordinator`."""
        assert self.kv_cache_manager is not None, "KVCacheManager is not initialized. Call launch_server() first."
        from psrl.utils.common.http_utils import init_http_client

        init_http_client(server_concurrency=4, rollout_engine_num=1)
        self.kv_cache_manager.set_controller_url(controller_url)

    def kv_set_current_version(self, version: int) -> None:
        """
        Commit this replica's actual model version for KV cache tagging.

        The version is written locally only after this replica has completed its
        weight pull. This keeps LMCache tags aligned with the weights that produced
        the KV tensors and avoids advancing unrelated replicas.

        Args:
            version (int): The new model version number.
        """
        if self.kv_cache_manager is not None:
            self.kv_cache_manager.set_current_version(version)

    ###### Weights Update ######

    async def pull_model_for_sync(self, ps_version: int) -> int:
        data_parallel_ranks = range(self.get_instance_num())
        data_parallel_rank = 0  # take dp 0 as representative for the replica

        if self.curr_rollout_instance_model_version[data_parallel_rank] >= ps_version:
            curr_model_version = self.curr_rollout_instance_model_version[data_parallel_rank]
            self.kv_set_current_version(curr_model_version)
            return curr_model_version

        async with shared_pull_model_context_async(self.gen_interface.ps_manager_handle):
            with log_dual_events("Pull model (partial rollout)", psrl_logger, event_type=EventType.PULL):
                await self.pull_model()

        curr_model_version = await self.gen_interface.ps_manager_handle.get_rollout_instance_model_version.remote(
            (self.base_worker_id, data_parallel_rank)
        )
        assert curr_model_version >= ps_version, (
            f"Pulled model version is stale: current={curr_model_version}, required={ps_version}."
        )
        for dp_rank in data_parallel_ranks:
            self.curr_rollout_instance_model_version[dp_rank] = curr_model_version
        self.kv_set_current_version(curr_model_version)

        if curr_model_version > ps_version:
            psrl_logger.warning(
                f"Actual model version after pull (partial rollout) is "
                f"{curr_model_version}, "
                f"which is higher than the required PS version {ps_version}."
            )
        if self.stat_collector is not None:
            for dp_rank in data_parallel_ranks:
                self.stat_collector.record_model_version_update(
                    self.curr_rollout_instance_model_version[data_parallel_rank], dp_rank
                )
        return curr_model_version

    async def sync_with_ps(self, ps_version: int, pause_generation: bool = False):
        if pause_generation:
            await self.pause_for_sync()
        version = await self.pull_model_for_sync(ps_version)
        if pause_generation:
            await self.resume_after_sync()
        return version

    async def pull_model(self):
        if self.psrl_config.ps_mode == "cpu" or self.psrl_config.ps_mode == "cpu_ref":
            await self.ray_pull_model()
        elif self.psrl_config.ps_mode == "nixl_cpu" or self.psrl_config.ps_mode == "nixl_gpu":
            await self.nixl_pull_model()
        else:
            raise NotImplementedError(f"PSRL does not support PS mode '{self.psrl_config.ps_mode}' yet.")
        # Important: the prefix cache needs to be cleared after pulling the model
        await self.clear_kv_cache()
        if self.kv_cache_manager is not None and self.kv_cache_manager.should_clear_on_weight_update:
            await self.collective_rpc(method="lmcache_clear_all_from_backend")
            psrl_logger.debug("Cleared LMCache backend after model weight update.")

    async def nixl_pull_model(self) -> None:
        assert self.gen_interface.ps_manager_handle is not None, "nixl_pull_model requires a PS manager handle"
        assert self.psrl_config.ps_mode == "nixl_cpu" or self.psrl_config.ps_mode == "nixl_gpu", (
            "nixl_pull_model should only be used in 'nixl_cpu' or 'nixl_gpu' mode."
        )
        ps_manager_handle = self.gen_interface.ps_manager_handle
        if self._cached_ps_nixl_agent_names is None:
            self._cached_ps_nixl_agent_names = await ps_manager_handle.get_ps_nixl_agent_names.remote()
        if self._cached_ps_nixl_gen_storage_client_names is None:
            self._cached_ps_nixl_gen_storage_client_names = (
                await ps_manager_handle.get_ps_nixl_gen_storage_client_names.remote()
            )
        if not self.psrl_config.profile.fix_weight:
            await self.engine.collective_rpc(
                "nixl_pull_model_core",
                args=(
                    self._cached_ps_nixl_agent_names,
                    self._cached_ps_nixl_gen_storage_client_names,
                ),
            )
        data_parallel_ranks = range(self.get_instance_num())
        pulled_instance_ids = [(self.base_worker_id, data_parallel_rank) for data_parallel_rank in data_parallel_ranks]
        await ps_manager_handle.pull_model_state_dict_nixl.remote(
            pulled_instance_ids
        )  # This only updates the model version
        psrl_logger.info("NIXL pull model done.")

    async def ray_pull_model(self) -> None:
        assert self.gen_interface.ps_manager_handle is not None, "ray_pull_model requires a PS manager handle"
        ps_manager_handle = self.gen_interface.ps_manager_handle

        if self.psrl_config.ps_mode == "cpu" or self.psrl_config.ps_mode == "cpu_ref":
            data_parallel_ranks = range(self.get_instance_num())
            rollout_instance_ids = [(self.base_worker_id, dp_rank) for dp_rank in data_parallel_ranks]

            if self.psrl_config.ps_mode == "cpu":
                # In 'cpu' mode, pull the full state dict (PS worker will block on transfer)
                model_state_dict_cpu = await ps_manager_handle.pull_model_state_dict_cpu.remote(rollout_instance_ids)
            elif self.psrl_config.ps_mode == "cpu_ref":
                # In 'cpu_ref' mode, get the object_ref and await it (PS worker is non-blocking)
                object_ref = await ps_manager_handle.pull_model_state_dict_cpu_ref.remote(rollout_instance_ids)
                model_state_dict_cpu = (
                    await object_ref
                )  # This blocks until the state dict is available in the object store
            # `load_weights` handles sharding and CPU-to-GPU transfer.
            params_to_load = [
                (
                    name,
                    (reduce_tensor(param.full_tensor()) if isinstance(param, DTensor) else reduce_tensor(param)),
                )
                for name, param in model_state_dict_cpu.items()
            ]
            if not self.psrl_config.profile.fix_weight:
                loaded_params = await self.engine.collective_rpc(
                    "load_weights",
                    args=(params_to_load,),
                )
                if loaded_params is None:
                    raise RuntimeError(f"Worker failed to update weights. Result: {loaded_params}")
        else:
            raise NotImplementedError(f"PSRL GenWorker does not support PS mode '{self.psrl_config.ps_mode}' yet.")

    ###### Utility Methods ######

    async def get_total_kv_cache_tokens(self) -> int:
        await self._is_init_model.wait()
        # All ranks return the same value, so unwrap the first collective result.
        results = await self.engine.collective_rpc(method="get_total_kv_cache_tokens")
        if not results:
            raise RuntimeError("get_total_kv_cache_tokens collective_rpc returned no results")
        return int(results[0])


class PSRL_vLLMReplica(vLLMReplica):
    def __init__(
        self,
        replica_rank: int,
        local_replica_rank: int,
        psrl_config,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gen_interface: GenInterface,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
        tag: str = "rollout",
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)

        self.psrl_config = psrl_config
        self.gen_interface = gen_interface
        self.tag = tag

        self.local_replica_rank = local_replica_rank
        self.data_parallel_size = config.data_parallel_size
        self.tensor_parallel_size = config.tensor_model_parallel_size
        self.pipeline_parallel_size = config.pipeline_model_parallel_size

        self.servers: list[ActorHandle] = []
        self.server_class = ray.remote(PSRL_vLLMHttpServer)
        # The HTTP server only needs inference settings. Keep worker-local
        # sandbox configuration out of the Ray actor boundary.
        self.server_config = replace(
            self.config,
            agent=replace(
                self.config.agent,
                sandbox=SandboxManagerConfig(),
            ),
        )

    def _get_server_env_vars(self) -> dict[str, str]:
        """Return the platform rollout environment with PSRL overrides applied."""
        return {
            **{var: "1" for var in get_platform().ray_noset_envvars()},
            **get_platform().rollout_env_vars(),
            "NCCL_CUMEM_ENABLE": "0",
            "VLLM_DISABLE_ATTN": "1" if self.config.disable_attn else "0",
        }

    async def init_model(self, worker_group: RayWorkerGroup):
        """Init model by launching vLLM server in each node.

        Args:
            worker_group: RayWorkerGroup, fused workers where training engine(fsdp/megatron) have been initialized.
        """
        self.rollout_mode = RolloutMode.STANDALONE
        self.workers = worker_group.workers[
            self.world_size * self.local_replica_rank : self.world_size * (self.local_replica_rank + 1)
        ]
        await self.launch_servers()

    async def launch_servers(self):
        """Launch http server in each node."""
        # AGENT(VERL): sync with verl's update
        assert len(self.workers) == self.world_size, (
            f"Worker count mismatch: workers={len(self.workers)}, world_size={self.world_size}."
        )

        # get (node_id, CUDA_VISIBLE_DEVICES) of all workers
        worker_infos = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self: (
                        ray.get_runtime_context().get_node_id(),
                        ray.get_runtime_context().get_accelerator_ids()[get_resource_name()][0],
                    )
                )
                for worker in self.workers
            ]
        )
        worker_cuda_visible_devices = [worker_info[1] for worker_info in worker_infos]
        worker_node_ids = [worker_info[0] for worker_info in worker_infos]

        # create server actor in each node with node affinity and cuda visible devices
        nnodes, gpus_per_replica_node = self.nnodes, self.gpus_per_replica_node
        for node_rank in range(nnodes):
            workers = self.workers[node_rank * gpus_per_replica_node : (node_rank + 1) * gpus_per_replica_node]
            node_cuda_visible_devices = ",".join(
                worker_cuda_visible_devices[
                    node_rank * gpus_per_replica_node : (node_rank + 1) * gpus_per_replica_node
                ]
            )
            node_id = worker_node_ids[node_rank * gpus_per_replica_node]
            prefix = self._get_server_name_prefix()
            if self.is_reward_model:
                name = f"{prefix}server_reward_{self.replica_rank}_{node_rank}"
            elif self.is_teacher_model:
                name = f"{prefix}server_teacher_{self.replica_rank}_{node_rank}"
            else:
                name = f"{prefix}server_{self.tag}_{self.replica_rank}_{node_rank}"

            # Preserve platform-specific rollout environment while applying PSRL overrides.
            env_vars = self._get_server_env_vars()
            if self.psrl_config.tms.range == "all" or self.psrl_config.tms.enable_nixl:
                # add tms config to rollout workers
                import torch_memory_saver  # noqa: F401

                dynlib_path = os.path.join(
                    os.path.dirname(os.path.dirname(torch_memory_saver.__file__)),
                    "torch_memory_saver_hook_mode_preload.abi3.so",
                )
                assert os.path.exists(dynlib_path), f"Missing LD_PRELOAD shared object: path={dynlib_path!r}."

                vllm_patch_env = ""
                if self.psrl_config.tms.enable_cuda_graph:
                    vllm_patch_env = "TMS:GRAPH"
                elif self.psrl_config.tms.range == "all":
                    vllm_patch_env = "TMS"

                env_vars.update(
                    {
                        "LD_PRELOAD": dynlib_path,
                        "TMS_INIT_ENABLE": "0",
                        "TMS_INIT_ENABLE_CPU_BACKUP": "0",
                        "PSRL_VLLM_PATCHES": vllm_patch_env,
                    }
                )

            server = self.server_class.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                runtime_env={"env_vars": env_vars},
                name=name,
                max_concurrency=self.max_concurrency,
            ).remote(
                psrl_config=self.psrl_config,
                config=self.server_config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                workers=workers,
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                gpus_per_node=gpus_per_replica_node,
                nnodes=nnodes,
                cuda_visible_devices=node_cuda_visible_devices,
                gen_interface=self.gen_interface,
            )
            self.servers.append(server)

        # launch http server in each node
        master_address, master_port, dp_rpc_port = await self.servers[0].get_master_address.remote()
        await asyncio.gather(
            *[
                server.launch_server.remote(
                    master_address=master_address, master_port=master_port, dp_rpc_port=dp_rpc_port
                )
                for server in self.servers
            ]
        )

        # get http server address from first server
        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = (
            f"[{server_address}]:{server_port}"
            if is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )

        # AGENT(VERL): Only keep one server handle for PSRL
        server_handle = self.servers[0]
        self.servers = [server_handle]
        await self.servers[0].is_init_model.remote()
