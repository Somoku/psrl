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
from psrl.utils.kv_cache import KVCacheManager, LMCacheConfig, enable_for_pool
from psrl.utils.logger import (
    DualOutputHandler,
    EventType,
    get_worker_info,
    log_dual_events,
)
from psrl.utils.nixl.pull_engine import WeightRefreshFailed
from psrl.utils.ps_mode import has_ps_data_plane, uses_nixl
from psrl.utils.ray import shared_pull_model_context_async
from psrl.workers.config import RolloutConfig
from psrl.workers.gen.lmcache_kv_events import LMCacheKvEventBridge
from psrl.workers.gen.lmcache_servicer import PSRLVllmEngineServicer
from psrl.workers.gen.smg_adapter import (
    build_worker_registration_payload,
    is_cache_aware_method,
    resolve_lmcache_event_flags,
)
from psrl.workers.gen.stats_collector import DPLBStatCollector
from psrl.workers.gen.utils import DEFAULT_MAX_CONNECTIONS, DEFAULT_TIMEOUT
from psrl.workers.gen.zmq_queue import ZMQPushQueue
from psrl.workers.ps.control_plane_meter import GLOBAL_CONTROL_PLANE_METER

if os.getenv("VERL_USE_GPT_OSS", "0") == "1":
    get_encoding()

psrl_logger = logging.getLogger(__file__)


class _PullDeferred(RuntimeError):
    """A pull must be retried after the writer publishes: it was preempted, or the version
    closed before it could start.

    Raised *inside* the shared pull lock so the lock's exit releases the admission; waiting
    for the new version while holding it would deadlock against a publish that still takes
    the exclusive side (cast/PS paths), and would keep the old version's reader count alive
    for no reason.
    """
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
        tag: str = "rollout",
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
        #: Role of this server ("rollout", "validate", "reward_model_*",
        #: "teacher_*"); LMCache uses it to skip pools that cannot reuse KV.
        self.tag = tag
        self.status_queue = None

        self.stat_collector = None
        self.curr_rollout_instance_model_version = None

        # Async event management
        self._is_init_model = asyncio.Event()
        self._is_init_nixl_client = asyncio.Event()
        #: vLLM memory tags currently released, tracked here because vLLM only reports
        #: "sleeping" as a whole. See `weights_resident`.
        self._asleep_tags: set[str] = set()

        # NIXL
        self.nixl_storage_client = None
        self.unified_state_dict = None
        self.unified_sharding_dict = None

        # NIXL cache
        #: Version reported by the most recent pull, or None if that pull reported none
        #: (item C2). Cleared before every pull: a value left over from the previous
        #: sync would be recorded as this sync's version, which is worse than the round
        #: trip it saves because nothing downstream could detect it.
        self._last_pulled_model_version: int | None = None
        #: Set while a pull runs under the shared-pull lock, so the version commit can
        #: ride on the lock release instead of costing its own round trip (item C2).
        self._pull_commit_deferred: bool = False
        self._pull_commit_instance_ids = None

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

    async def _build_kv_cache_manager(self) -> KVCacheManager:
        """Build `KVCacheManager` and start this node's LMCache MP server."""
        lmcache_raw = (
            OmegaConf.to_container(
                self.psrl_config.get("lmcache", OmegaConf.create()),
                resolve=True,
            )
            or {}
        )

        # A validation run is single-turn, its prompts are unique within the
        # run, and a weight update fences every earlier entry by version. Its
        # stores can never be read back, so the copy is pure overhead. Reward
        # and teacher models produce no reusable generation KV either, and a
        # generate-only agent pool has no within-trajectory reuse.
        agent_loop = str(getattr(self.config.agent, "default_agent_loop", ""))
        if not enable_for_pool(
            lmcache_raw,
            self.tag,
            is_reward_model=getattr(self, "is_reward_model", False),
            is_teacher_model=getattr(self, "is_teacher_model", False),
            is_single_turn=agent_loop == "generate_only_agent",
        ):
            if lmcache_raw.get("enable", False):
                psrl_logger.info(
                    "[LMCache] Disabled for %s server %s: this pool cannot reuse KV.",
                    self.tag,
                    self.get_replica_idx(),
                )
            lmcache_raw["enable"] = False

        # One MP server per node owns all local KV ranks, so its identity is
        # node-scoped when a replica spans nodes and replica-scoped otherwise.
        replica_id = f"psrl_instance_{self.get_replica_idx()}"
        lmcache_instance_id = replica_id
        if self.nnodes > 1:
            lmcache_instance_id = f"{replica_id}_n{self.node_rank}"
        lmcache_raw["lmcache_instance_id"] = lmcache_instance_id
        # Registered with the coordinator so a transfer can reach every node of
        # this replica from either its replica id or a single node id.
        lmcache_raw["replica_id"] = replica_id

        # One shared resolver decides both flags, so the MP server, the
        # coordinator launcher and the router agree on when the off-GPU tier is
        # live.
        flags = resolve_lmcache_event_flags(self.psrl_config)
        lmcache_raw["enable_kv_events"] = flags.kv_events
        lmcache_raw["coordinator_event_reporting"] = flags.coordinator_reporting
        if lmcache_raw.get("enable_p2p", False) or flags.coordinator_reporting:
            # MP servers must be reachable in the coordinator registry, both for
            # P2P discovery and for a router to find the event stream. The YAML
            # default already points at the PS manager; this fills the host back
            # in when a config cleared it.
            lmcache_raw["coordinator_host"] = lmcache_raw.get("coordinator_host") or str(
                self.psrl_config.get("ps_manager_ip", "127.0.0.1")
            )

        lmcache_cfg = LMCacheConfig(**lmcache_raw)
        if lmcache_cfg.n_servers > 1:
            # The connector numbers ranks in node-contiguous blocks, one block
            # per configured server, and vLLM rejects that alongside DP.
            assert self.config.data_parallel_size == 1, (
                f"lmcache.mp_server_urls splits the replica across {lmcache_cfg.n_servers} servers, which "
                f"LMCacheMPConnector does not support with data_parallel_size="
                f"{self.config.data_parallel_size}. Leave it empty or disable data parallelism."
            )
            assert lmcache_cfg.n_servers == max(1, self.nnodes), (
                f"lmcache.mp_server_urls lists {lmcache_cfg.n_servers} servers but this replica spans "
                f"{self.nnodes} node(s). List every node's server, ordered by node rank."
            )
        kv_cache_manager = KVCacheManager(lmcache_cfg)
        if lmcache_cfg.enable:
            # Per-rank keys follow the connector's own geometry, which is
            # resolved from the engine config once the engine exists.
            await self._start_lmcache_mp_runtime(lmcache_cfg, kv_cache_manager)
        return kv_cache_manager

    def _set_lmcache_geometry(self, vllm_config) -> None:
        """
        Derive the KV rank fan-out from the engine config.

        The connector computes its fan-out from this same helper, so taking the
        value from the engine is what keeps backend pins and cross-instance
        transfer addressing the keys that were actually stored. Duplicating the
        formula instead silently mis-addresses multi-node replicas and MLA
        models, which replicate KV across TP ranks.

        Args:
            vllm_config: The vLLM config created for this server.
        """
        if self.kv_cache_manager is None or not self.kv_cache_manager.enabled:
            return
        from lmcache.integration.vllm.lmcache_mp_connector import (
            build_parallel_strategy_from_vllm_config,
        )

        n_servers = max(1, self.kv_cache_manager.config.n_servers)
        strategy = build_parallel_strategy_from_vllm_config(vllm_config, n_servers)
        self.kv_cache_manager.set_parallel_geometry(self.model_config.path, strategy.kv_world_size)

    async def _start_lmcache_mp_runtime(self, lmcache_cfg: LMCacheConfig, kv_cache_manager: KVCacheManager) -> None:
        """
        Allocate runtime ports, start the node-local MP server, and attach it.

        Args:
            lmcache_cfg (LMCacheConfig): Configuration whose runtime fields are filled here.
            kv_cache_manager (KVCacheManager): Manager that will own the runtime.
        """
        from psrl.utils.common.http_utils import get_host_info
        from psrl.utils.kv_cache.runtime import LMCacheMPRuntime
        from psrl.utils.nixl.port_scanner import get_port_scanner

        _, node_ip = get_host_info()
        port_scanner = get_port_scanner(node_ip)
        # Always a message-queue and HTTP port, plus one per enabled feature.
        wants_events = lmcache_cfg.coordinator_event_reporting
        # The event stream shares the management HTTP port, and consumers on
        # other nodes must be able to reach it like a transfer destination.
        routable_http = lmcache_cfg.enable_p2p or wants_events
        num_ports = 2 + (1 if lmcache_cfg.enable_p2p else 0)
        ports = ray.get([port_scanner.find_free_port.remote() for _ in range(num_ports)])

        next_port = 0
        lmcache_cfg.server_host = "127.0.0.1"
        lmcache_cfg.server_port = ports[next_port]
        next_port += 1
        lmcache_cfg.http_port = ports[next_port]
        next_port += 1
        if routable_http:
            lmcache_cfg.http_host = "0.0.0.0"
            lmcache_cfg.http_advertise_host = node_ip
        if lmcache_cfg.enable_p2p:
            lmcache_cfg.p2p_advertise_host = node_ip
            lmcache_cfg.p2p_transfer_port = ports[next_port]

        log_path = os.path.join(
            str(self.psrl_config.logging_path),
            f"lmcache_server_{self.get_replica_idx()}_n{self.node_rank}.log",
        )
        runtime = LMCacheMPRuntime(lmcache_cfg)
        await runtime.start(log_path)
        kv_cache_manager.attach_runtime(runtime)

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

        hf_overrides = self._apply_quantization()
        # PSRL's publisher casts the trainer's shards itself, so the engine must load the same
        # block layout. This is the transport's wire format, separate from veRL's trainer-side
        # QAT presets (which vLLM routes to compressed-tensors kernels).
        transport_quantization = (self.psrl_config.get("nixl") or {}).get("rollout_quantization", None)
        quantization_arg = None
        if transport_quantization is not None:
            if transport_quantization != "fp8":
                raise ValueError(
                    f"Unsupported psrl.nixl.rollout_quantization={transport_quantization!r}; expected null or fp8."
                )
            if hf_overrides:
                raise ValueError("Direct-supply FP8 cannot be combined with a model quantization_config.")
            num_layers = self.model_config.hf_config.num_hidden_layers
            hf_overrides["quantization_config"] = {
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
                "ignored_layers": [f"model.layers.{layer}.mlp.gate" for layer in range(num_layers)],
            }
            quantization_arg = "fp8"
        hf_quantization_config = getattr(self.model_config.hf_config, "quantization_config", None)
        effective_quantization_config = hf_overrides.get("quantization_config") or hf_quantization_config
        from verl.utils.vllm.vllm_quant_utils import set_default_vllm_quantization_backends

        set_default_vllm_quantization_backends(effective_quantization_config, engine_kwargs)
        if transport_quantization == "fp8" and engine_kwargs.get("linear_backend") == "deep_gemm":
            from vllm.utils.import_utils import has_deep_gemm

            if not has_deep_gemm():
                raise RuntimeError(
                    "Direct-supply FP8 defaults to vLLM's DeepGEMM linear backend, but the "
                    "deep_gemm package is unavailable. Install the pinned DeepGEMM build or set "
                    "the vLLM linear_backend to a supported alternative explicitly."
                )

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
            "enable_nccl_comm_suspend": self.config.enable_nccl_comm_suspend,
            "logprobs_mode": self.config.logprobs_mode,
            "enforce_eager": self.config.enforce_eager,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "disable_log_stats": self.config.disable_log_stats,
            "tensor_parallel_size": self.config.tensor_model_parallel_size,
            "seed": self.replica_rank + (self.config.get("seed") or 0),
            "override_generation_config": json.dumps(override_generation_config),
            "quantization": quantization_arg,
            "hf_overrides": hf_overrides,
            "scheduling_policy": self.config.scheduling_policy,
            "compilation_config": compilation_config,
            # AGENT(VERL): thread runner/task through for pooling model support in PSRL
            "runner": self.config.get("runner", "generate"),
            **engine_kwargs,
        }

        self.kv_cache_manager = await self._build_kv_cache_manager()
        args.update(self.kv_cache_manager.get_engine_kwargs())
        capture_mode = self.config.get("sampler_distribution", "disabled")
        if capture_mode != "disabled":
            from psrl.utils.sampler_distribution import SamplerCapture

            width = SamplerCapture.from_rollout_config(self.config).width
            if args["logprobs_mode"] != "processed_logprobs":
                raise ValueError("Sampler distribution capture requires logprobs_mode=processed_logprobs.")
            configured_max = args.get("max_logprobs", width)
            if configured_max != -1 and configured_max < width:
                raise ValueError(f"vLLM max_logprobs must be at least {width} for sampler capture.")
            args["max_logprobs"] = configured_max

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

        # Select the TMS sleep backend when the torch_memory_saver patches are
        # active. ModelConfig reaches the workers, so setting it here is enough.
        if os.environ.get("PSRL_VLLM_PATCHES", "").strip().startswith("TMS"):
            vllm_config.model_config.sleep_mode_backend = "tms"
            psrl_logger.info("Using vLLM sleep-mode backend: tms")

        fn_args = set(dict(inspect.signature(AsyncLLM.from_vllm_config).parameters).keys())
        kwargs = {}
        if "enable_log_requests" in fn_args:
            kwargs["enable_log_requests"] = engine_args.enable_log_requests
        if "disable_log_stats" in fn_args:
            kwargs["disable_log_stats"] = engine_args.disable_log_stats

        # vLLM dispatches stat-logger factories per engine. Returning the same
        # shared instance for every index preserves the single-collector model.
        self.preemption_queue: asyncio.Queue = asyncio.Queue(maxsize=0)
        self.psrl_preemption_logger = PreemptionStatLogger(
            vllm_config,
            engine_index=0,
            preemption_queue=self.preemption_queue,
        )

        def preemption_logger_factory(vllm_config, engine_index=0):
            return self.psrl_preemption_logger

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

            def dplb_stat_logger_factory(vllm_config, engine_index=0):
                return self.stat_collector

            kwargs["stat_loggers"] = [dplb_stat_logger_factory, preemption_logger_factory]
        else:
            kwargs["stat_loggers"] = [preemption_logger_factory]

        engine_client = AsyncLLM.from_vllm_config(vllm_config=vllm_config, usage_context=usage_context, **kwargs)

        # Every node addresses the keys its own workers stored, so the geometry
        # is resolved here rather than only on the node that serves HTTP.
        self._set_lmcache_geometry(vllm_config)

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

        # The router scores the off-GPU tier from the LMCache event stream, but
        # vLLM does not surface connector placements through its own publisher.
        # Bridge the node's MP-server stream into the same gRPC subscription.
        lmcache_bridge = None
        if (
            self.kv_cache_manager is not None
            and self.kv_cache_manager.enabled
            and self.kv_cache_manager.config.enable_kv_events
            and is_cache_aware_method(self.psrl_config.rollout_coordination.routing_strategy.method)
        ):
            lmcache_bridge = LMCacheKvEventBridge(
                f"http://127.0.0.1:{self.kv_cache_manager.config.http_port}",
                current_version=lambda: self.kv_cache_manager.current_version,
            )

        servicer = PSRLVllmEngineServicer(
            engine_client,
            start_time,
            preemption_queue=self.preemption_queue,
            kv_cache_manager=self.kv_cache_manager,
            kv_transfer_stats_log_interval_s=stats_log_interval_s,
            enable_kv_event_replay=self.psrl_config.rollout_gateway.enable_kv_event_replay,
            lmcache_bridge=lmcache_bridge,
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

    async def weights_resident(self) -> bool:
        """Whether a pull has somewhere to land.

        Not `not is_sleeping()`: vLLM keeps reporting sleeping until *every* tag is
        awake, so an instance woken for weights only -- the direct-supply colocate
        switch, which defers the KV cache until the trainer has left -- would be skipped
        by the sync and then serve whatever its weight memory held.
        """
        return "weights" not in self._asleep_tags

    async def sleep(self, level: int):
        self._asleep_tags = {"weights", "kv_cache"}
        await self.engine.sleep(level)
        await self.engine.reset_encoder_cache()
        if self.psrl_config.tms.range in ["rollout", "all"]:
            # NOTE(linsh): TMS requires an explicit aggressive cache clear.
            aggressive_empty_cache(force_sync=True)

    async def wake_up(self, tags: list[str] | None = None):
        # The TMS backend resumes its "graph" region implicitly, so "graph" is
        # not requested here. Executor.wake_up rejects tags it did not record.
        #
        # `tags` lets a colocate switch under direct supply wake weights while the
        # trainer is still resident (it is the source being read) and the KV cache only
        # after the trainer has left; see `psrl.trainer.ppo.colocate_switch`.
        woken = tags or ["weights", "kv_cache"]
        await self.engine.wake_up(tags=woken)
        self._asleep_tags -= set(woken)

    async def clear_kv_cache(self):
        # vLLM refuses to reset the prefix cache while any block is referenced,
        # so drop PSRL's own pins or old-weight KV stays reachable.
        if self.kv_cache_manager is not None and self.kv_cache_manager.enabled:
            await self.kv_cache_manager.unpin_all_gpu()
        reset = await self.engine.reset_prefix_cache(reset_connector=True)
        if reset is False:
            psrl_logger.error(
                "[LMCache] vLLM refused to reset the prefix cache after a weight update, so KV "
                "produced by the previous weights may still be served. Some blocks are still "
                "referenced by a live request."
            )
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
        # Number of EngineCore instances (data-parallel replicas) this server
        # manages. `engine_ranks_managed` is vLLM's own MPClient attribute.
        return len(self.engine.engine_core.engine_ranks_managed)

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
                # Direct supply derives remote descriptors on the reader; serializing
                # them here would put one descriptor per strided row into every client
                # info, and those infos are what switch-time updates carry.
                has_ps_data_plane(self.psrl_config.ps_mode),
                self._resolve_segment_registration(),
            ),
        )
        self._is_init_nixl_client.set()

    def _resolve_segment_registration(self) -> bool:
        """Whether this engine may register allocator segments instead of allocations.

        `launch_server` enables expandable segments exactly when sleep mode is off; their
        VA range is mapped on demand, so a page-locking registration of the whole range
        is unsafe. The worker cannot observe that setting, so the decision is made here
        and passed down.
        """
        mode = self.psrl_config.nixl.get("segment_registration", "auto")
        if isinstance(mode, bool):
            return mode
        if not self.config.enable_sleep_mode:
            return False
        from psrl.utils.nixl.client import _expandable_segments_enabled  # noqa: PLC0415

        return not _expandable_segments_enabled()

    async def nixl_convert_params(self):
        await self._is_init_model.wait()
        await self.collective_rpc(
            method="nixl_convert_params",
            args=(self.model_config,),
        )

    async def nixl_protocol(self, mode: str = "full"):
        psrl_logger.info(f"vLLM server NIXL protocol (mode={mode!r}): waiting for initialization.")
        await self._is_init_model.wait()
        await self._is_init_nixl_client.wait()
        psrl_logger.info(f"vLLM server NIXL protocol (mode={mode!r}): dispatching to workers.")
        await self.collective_rpc(
            method="nixl_protocol",
            args=(self.model_config, mode),
        )
        psrl_logger.info(f"vLLM server NIXL protocol (mode={mode!r}) done.")

    async def nixl_wake_up(self, tags: list[str] | None = None):
        await self._is_init_nixl_client.wait()
        await self.wake_up(tags)
        # Only the weights are registered with NIXL; waking the KV cache alone changes
        # no address anyone reads.
        if tags is None or "weights" in tags:
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

    async def nixl_wait_for_update_infos(self, info_num: int, timeout: float | None = None):
        await self._is_init_nixl_client.wait()
        await self.collective_rpc(
            method="nixl_wait_for_update_infos",
            args=(info_num, timeout),
        )

    ###### KV Cache Management ######

    def _assert_kv_cache_manager(self) -> KVCacheManager:
        assert self.kv_cache_manager is not None, "KVCacheManager is not initialized. Call launch_server() first."
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

    async def pull_model_for_sync(
        self,
        ps_version: int,
        epoch: int | None = None,
        participants: list[int] | None = None,
    ) -> int:
        data_parallel_ranks = range(self.get_instance_num())
        data_parallel_rank = 0  # take dp 0 as representative for the replica

        if self.curr_rollout_instance_model_version[data_parallel_rank] >= ps_version:
            curr_model_version = self.curr_rollout_instance_model_version[data_parallel_rank]
            self.kv_set_current_version(curr_model_version)
            return curr_model_version

        # C2: the pull defers its version commit so it can ride on the lock release.
        # `_pull_commit_instance_ids` is set by `pull_model` and consumed here.
        self._pull_commit_deferred = True
        self._pull_commit_instance_ids = None

        # E1: announce the read before it starts, so a publisher about to overwrite these
        # weights can wait for it. The version reported is the one *being left behind*,
        # which is what the publisher is about to overwrite; the new version is only
        # assigned by the pull itself.
        #
        # Wrapped in try/finally because a count that never goes back to zero is worse
        # than no count at all: the publisher would wait out its whole budget on every
        # publish from then on, forever, with nothing in the logs to say why.
        ps_manager_handle = self.gen_interface.ps_manager_handle
        reading_version = int(self.curr_rollout_instance_model_version[data_parallel_rank])
        # The same `(worker_id, dp_rank)` pair the version commit below uses to name this
        # reader, so the count and the commit agree on who is reading.
        reader_id = (self.base_worker_id, data_parallel_rank)
        await ps_manager_handle.begin_read.remote(reader_id, reading_version)
        try:
            max_retries = max(2, int(self.psrl_config.nixl.get("preempt_repull_attempts", 2))) + 2
            for attempt in range(max_retries):
                lock = shared_pull_model_context_async(
                    self.gen_interface.ps_manager_handle, reader=reader_id
                )
                async with lock:
                    self._pull_used_epoch = None
                    self._pull_landed_version = None
                    self._pull_repaired = False
                    try:
                        with log_dual_events(
                            "Pull model (partial rollout)", psrl_logger, event_type=EventType.PULL
                        ):
                            await self.pull_model(epoch, participants)
                    except _PullDeferred as deferred:
                        # Release the admission before waiting: the writer's publish must not
                        # wait on this reader's shared count, and the next attempt needs a
                        # fresh epoch anyway.
                        psrl_logger.warning(
                            f"{self.base_worker_id}: pull deferred ({deferred}); waiting for "
                            "the next published version."
                        )
                        lock.commit_instance_ids = None
                        continue
                    lock.commit_instance_ids = self._pull_commit_instance_ids
                    lock.pull_epoch = self._pull_used_epoch
                    lock.landed_version = self._pull_landed_version
                    lock.repaired = self._pull_repaired
                if lock.committed_version is not None:
                    break
                # The commit was rejected: the local bytes may mix versions. Wait for the
                # writer to publish and pull again; exhausting the retries raises, so the
                # replica is quarantined rather than serving a mixture.
                psrl_logger.warning(
                    f"{self.base_worker_id}: pull commit rejected (attempt {attempt + 1}); "
                    "re-pulling the next published version."
                )
                if attempt + 1 >= max_retries:
                    raise WeightRefreshFailed(
                        f"{self.base_worker_id}: pull commit rejected after a write; the instance "
                        "holds bytes from more than one version and must not generate."
                    )
                await self._wait_for_open_version(after=ps_version)
        finally:
            self._pull_commit_deferred = False
            await ps_manager_handle.end_read.remote(reader_id, reading_version)

        # The commit reports the version it assigned, so the steady-state path makes no
        # round trip here. The read remains for the modes that commit no version, rather
        # than being assumed unreachable.
        curr_model_version = (
            lock.committed_version
            if lock.committed_version is not None
            else self._last_pulled_model_version
        )
        if curr_model_version is None:
            GLOBAL_CONTROL_PLANE_METER.record("get_rollout_instance_model_version")
            curr_model_version = (
                await self.gen_interface.ps_manager_handle.get_rollout_instance_model_version.remote(
                    (self.base_worker_id, data_parallel_rank)
                )
            )
        # D6: falling short of the target is a degradation, not a failure. Raising here
        # reached `_finish_gateway_sync`'s except branch, which calls `fail_sync` and
        # pauses the instance -- so the replica went *unavailable*, the opposite of
        # "keep the current version and keep generating". The caller is told the real
        # version instead and decides; D7's whole-DP-group failure lands here, so
        # leaving the assert would make that path quarantine replicas on its first run.
        if curr_model_version < ps_version:
            psrl_logger.warning(
                f"Pull did not reach the target version: have {curr_model_version}, "
                f"wanted {ps_version}. Continuing to generate at {curr_model_version}."
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

    async def sync_with_ps(
        self,
        ps_version: int,
        pause_generation: bool = False,
        epoch: int | None = None,
        participants: list[int] | None = None,
    ):
        if pause_generation:
            await self.pause_for_sync()
        version = await self.pull_model_for_sync(ps_version, epoch, participants)
        if pause_generation:
            await self.resume_after_sync()
        return version

    async def pull_model(
        self,
        epoch: int | None = None,
        participants: list[int] | None = None,
    ):
        # C2: clear before the pull, not after. A version left over from the previous
        # sync would be silently attributed to this one.
        self._last_pulled_model_version = None
        if self.psrl_config.ps_mode == "cpu" or self.psrl_config.ps_mode == "cpu_ref":
            await self.ray_pull_model()
        elif uses_nixl(self.psrl_config.ps_mode):
            await self.nixl_pull_model(epoch, participants)
        else:
            raise NotImplementedError(f"PSRL does not support PS mode '{self.psrl_config.ps_mode}' yet.")
        # Important: the prefix cache needs to be cleared after pulling the model
        await self.clear_kv_cache()
        if self.kv_cache_manager is not None and self.kv_cache_manager.should_clear_on_weight_update:
            await self.kv_cache_manager.clear()
            psrl_logger.debug("Cleared LMCache backend after model weight update.")
        if self.kv_cache_manager is not None and self.kv_cache_manager.enabled:
            # Older-version pins are unreachable now, so they only hold budget.
            await self.kv_cache_manager.release_pin_groups()

    async def nixl_pull_model(
        self,
        epoch: int | None,
        participants: list[int] | None,
    ) -> None:
        """
        Pull through the engine on every worker of this replica.

        A torn pull, or one that read a version the publisher has since declared invalid,
        is retried whole rather than propagated. The retry runs inside the shared pull
        lock, so a push in progress makes it wait and then read what the push published.

        Args:
            epoch (int | None): Sync identifier from the coordinator. None pulls without peers.
            participants (list[int] | None): Replica indices syncing together.
        """
        assert self.gen_interface.ps_manager_handle is not None, "nixl_pull_model requires a PS manager handle"
        assert uses_nixl(self.psrl_config.ps_mode), (
            f"nixl_pull_model needs a NIXL transport, got ps_mode={self.psrl_config.ps_mode!r}"
        )
        ps_manager_handle = self.gen_interface.ps_manager_handle
        if epoch is None:
            # A pull outside a coordinated sync has no peers to wait for, so it reads everything upstream.
            self._solo_pulls = getattr(self, "_solo_pulls", 0) + 1
            epoch, participants = -self._solo_pulls, [self.get_replica_idx()]
        if not self.psrl_config.profile.fix_weight:
            await self._pull_until_consistent(epoch, participants)
        # C2: the version commit is not made here. It rides on the shared-pull lock
        # release (`finish_shared_pull`), which has to happen anyway -- so the commit
        # costs no round trip of its own. `_pull_commit_instance_ids` tells the lock
        # context which instances to commit; a caller holding no lock commits directly.
        data_parallel_ranks = range(self.get_instance_num())
        pulled_instance_ids = [(self.base_worker_id, data_parallel_rank) for data_parallel_rank in data_parallel_ranks]
        if self._pull_commit_deferred:
            self._pull_commit_instance_ids = pulled_instance_ids
        else:
            GLOBAL_CONTROL_PLANE_METER.record("pull_model_state_dict_nixl")
            self._last_pulled_model_version = await ps_manager_handle.pull_model_state_dict_nixl.remote(
                pulled_instance_ids
            )
            # Only meaningful where the metadata behind it is fresh: see
            # `_assert_pulled_the_announced_version`. Counted here rather than read off a
            # worker -- `pull_times` lives on the vLLM worker, across the collective_rpc
            # boundary, and is not an attribute of this server.
            self._nixl_pulls_done = getattr(self, "_nixl_pulls_done", 0) + 1
            if self._nixl_pulls_done == 1:
                await self._assert_pulled_the_announced_version(self._last_pulled_model_version)
        psrl_logger.info("NIXL pull model done.")

    async def _pull_until_consistent(self, epoch: int, participants: list[int] | None) -> None:
        """Pull until the weights hold one live version, or defer/raise.

        Each attempt is a whole pull, so each retry costs a full sync at steady state
        (0.02 to 0.16 s for an 8B model on one node, 2.4 s for 235B across two, FINDINGS
        §5 and §4b). Exhausting the bound raises because the instance holds mixed bytes.

        A retry reads everything upstream itself. Epochs key peer diffusion signals and
        must never be reused, and an invented epoch would differ per replica, so followers
        would wait out the peer timeout before falling back anyway.

        A write-gate preemption that cannot be repaired from holders raises
        `_PullDeferred`: the caller releases its admission, waits for the writer to
        publish, and pulls the next version through a fresh admission. Retrying inside
        the current admission would leave a stale epoch and keep the old reader count
        alive while the writer is trying to open the new version.
        """
        max_attempts = max(1, int(self.psrl_config.nixl.get("max_pull_attempts", 3)))
        ps_manager_handle = self.gen_interface.ps_manager_handle
        attempt_epoch, attempt_participants = epoch, participants
        last_versions: frozenset[int] = frozenset()
        last_reason: str | None = None
        for attempt in range(1, max_attempts + 1):
            # Re-read the state every attempt: the version moved, so the first set was wrong.
            (
                target_version,
                version_holders,
                invalidated,
                writer_epoch,
                write_state,
            ) = await ps_manager_handle.get_nixl_version_state.remote()
            if write_state != "open":
                raise _PullDeferred(f"version closed for a write (state={write_state})")
            # The engine accepts a preemption notice for an epoch newer than the one it
            # starts at. Admission may predate a completed write, so use the current epoch
            # rather than the admission's; None disables the control path when the gate is off.
            engine_epoch = (
                writer_epoch if self.psrl_config.nixl.get("await_write_gate") is True else None
            )
            verdicts = await self.engine.collective_rpc(
                "nixl_pull_model_core",
                args=(attempt_epoch, attempt_participants, version_holders, target_version, engine_epoch),
            )
            self._pull_used_epoch = writer_epoch
            self._pull_repaired = any(v.repaired for v in verdicts)
            if any(not v.landed and v.reason == "preempted" for v in verdicts):
                raise _PullDeferred("pull preempted by a parameter write and not repairable")
            landed = [v for v in verdicts if v.landed]
            if len(landed) == len(verdicts):
                # Ranks landing different versions corrupt one instance just as mixing does.
                landed_versions = frozenset().union(*(v.versions for v in landed))
                # An empty set means the pull read no keys, which is a no-op rather than stale.
                if not landed_versions or (len(landed_versions) == 1 and not landed_versions & invalidated):
                    self._pull_landed_version = next(iter(landed_versions)) if landed_versions else None
                    return
                last_reason = "invalidated" if landed_versions & invalidated else "torn"
                last_versions = landed_versions
            else:
                # A rank that wrote nothing is still consistent. Only `torn` needs a repair.
                torn = [v for v in verdicts if not v.landed and v.reason == "torn"]
                if not torn:
                    # D6/D7: nothing was written, so keeping the current version is correct.
                    psrl_logger.warning(
                        f"{self.base_worker_id}: {sum(not v.landed for v in verdicts)} of "
                        f"{len(verdicts)} rank(s) found no live source for some keys. Keeping "
                        f"the current version."
                    )
                    return
                last_reason = "torn"
                last_versions = frozenset().union(*(v.versions for v in torn))

            if attempt < max_attempts:
                psrl_logger.warning(
                    f"{self.base_worker_id}: pull attempt {attempt}/{max_attempts} did not land "
                    f"({last_reason}, versions={sorted(last_versions)}, announced v{target_version}). "
                    f"Re-pulling without peers."
                )
                self._solo_pulls = getattr(self, "_solo_pulls", 0) + 1
                attempt_epoch, attempt_participants = -self._solo_pulls, [self.get_replica_idx()]

        raise WeightRefreshFailed(
            f"{self.base_worker_id}: {max_attempts} pull attempt(s) did not land a single live "
            f"version (last {last_reason}, versions={sorted(last_versions)}). The weights mix "
            f"versions and must not generate. The next sync will re-pull."
        )

    async def _wait_for_open_version(self, after: int) -> int:
        """Wait until the trainer's version is open and at least ``after``, then return it."""
        ps_manager_handle = self.gen_interface.ps_manager_handle
        timeout = float(self.psrl_config.nixl.get("preempt_repull_timeout_s", 600))
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            version, _, _, _, state = await ps_manager_handle.get_nixl_version_state.remote()
            if state == "open" and version >= after:
                return version
            await asyncio.sleep(0.05)
        raise WeightRefreshFailed(
            f"{self.base_worker_id}: the trainer did not publish a new open version within "
            f"{timeout}s; the instance must not generate while holding mixed bytes."
        )

    async def _assert_pulled_the_announced_version(self, announced: int) -> None:
        """The bytes read must belong to the version the manager just handed out (§3.4).

        The per-shard check inside the pull catches a *torn* read -- shards from different
        versions. It cannot catch a uniformly stale one, where every source still holds
        v-1 and therefore agrees with itself. That is exactly what a missed publish
        produces, and it is the failure mode with no symptom: the pull succeeds, the
        version counter advances, and the engine serves last step's weights.

        Compared here rather than before the pull because a rollout learns its version
        *from* the pull; demanding it up front would reintroduce the round trip C2 removed.

        **Scoped to the first pull, and the scope is a real limit, not a convenience.**
        The version a reader sees lives in cached remote metadata, refreshed only by
        `wait_for_update_infos` -- which production runs at the handshake and on colocate
        switches, not every step. Asserting this on later pulls compares against a stamp
        that predates them and fails on correct runs; making it meaningful per step would
        need a metadata broadcast per step, which costs more than the fault it catches.

        What that leaves covered is bootstrap: that the weights the rollouts first read
        are the version the manager says they are. Steady-state staleness is covered
        instead by the torn-read check inside the pull, which needs no fresh metadata
        because it only compares shards against each other.
        """
        observed = await self.engine.collective_rpc("pulled_version_set")
        seen = set().union(*observed) if observed else set()
        # An empty set means nothing was read -- a no-op pull, not a stale one.
        if not seen or seen == {announced}:
            return
        raise AssertionError(
            f"stale pull: the manager announced v{announced} but the shards read carry "
            f"{sorted(seen)}; the engine would serve weights from the wrong version"
        )

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

    @property
    def max_concurrency(self) -> int:
        """Ray actor concurrency for an HTTP server replica.

        verl's rollout replica base used to provide this and no longer does, while this
        class still passes it to `server_class.options(...)`. Owning it here keeps the
        value next to the only code that consumes it.

        The floor of 1000 is Ray's own default for async actors: dropping below it would
        silently reduce concurrency for small `max_num_seqs`. The margin is for control
        methods (`wake_up`, `sleep`, weight sync), which share the actor's concurrency
        budget with in-flight generation requests -- without it a saturated server cannot
        service the very calls that would drain it.
        """
        control_method_concurrency = 16
        return max(1000, self.config.max_num_seqs + control_method_concurrency)

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
                tag=self.tag,
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
