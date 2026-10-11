import asyncio
import math
import os
import time

# Must precede anything that reaches megatron.bridge. That chain pulls in modelopt ->
# tilelang, which loads its own `libcudart_stub.so` into the process; a later
# `flashinfer.comm` import then resolves cudaDeviceReset against the stub and dies with
# "undefined symbol". vLLM's FusedMoE -> all2all_utils -> nixl_ep path imports
# flashinfer, so the whole harness fails at import time unless the real CUDA runtime is
# bound first. Importing it here is the fix; order, not presence, is what matters.
try:
    import flashinfer.comm  # noqa: F401
except Exception:  # pragma: no cover - flashinfer is optional for CPU-only runs
    pass

import hydra
import ray
import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from omegaconf import DictConfig, OmegaConf
from psrl.utils.common.worker_naming import gen_client_name, train_client_name
from psrl.utils.converter import create_parameter_mapping
from psrl.utils.converter.fsdp_converter import convert_fsdp_inplace
from psrl.utils.converter.megatron_converter import convert_megatron_inplace
from psrl.utils.converter.vllm_converter import convert_vllm_inplace
from psrl.utils.nixl import (
    NIXL_META_SERVER_NAME,
    NIXLClientType,
    NIXLMetaServer,
    NIXLStorageClient,
)
from psrl.workers.ps import (
    PSClassWithInitArgs,
    PSResourcePool,
    PSResourceSpec,
    PSStoragePlan,
    PSStorageWorker,
    PSWorkerGroup,
)
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

try:
    from tests.nixl.coord_judge import (
        assert_state_dict_matches_coord,
        fill_state_dict_by_coord,
    )
except ModuleNotFoundError:
    # Loaded by file path rather than by package name. Run as a script
    # (`python tests/nixl/test_nixl_e2e.py`), sys.path[0] is this file's own directory,
    # so `tests` resolves to whatever else is on the path -- adding the repo root does
    # not help, because the shadowing entry comes first. Loading the sibling module
    # directly sidesteps package resolution entirely and works under pytest too, since
    # the try branch wins there.
    import importlib.util
    from pathlib import Path

    _spec = importlib.util.spec_from_file_location(
        "psrl_e2e_coord_judge", Path(__file__).resolve().parent / "coord_judge.py"
    )
    _coord_judge = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_coord_judge)
    assert_state_dict_matches_coord = _coord_judge.assert_state_dict_matches_coord
    fill_state_dict_by_coord = _coord_judge.fill_state_dict_by_coord
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from transformers import AutoConfig, AutoModelForCausalLM
from verl.models.mcore import hf_to_mcore_config
from verl.utils.fs import copy_to_local
from verl.utils.torch_dtypes import PrecisionType
from verl.workers.engine.megatron.utils import set_random_seed

NUM_GPU_PER_NODE = 8

#: Repository root, shipped to Ray actors so they import this tree rather than the
#: checkout the cluster environment puts on the default path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _fully_shard_transformer(model, mesh) -> None:
    """Apply FSDP2 ``fully_shard`` bottom-up, mirroring verl's wrap policy.

    Each transformer block is sharded as its own unit before the root, which is what
    produces per-parameter DTensors rather than one flat buffer. The order matters:
    `fully_shard` must see children already sharded when it reaches the root.
    """
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is not None:
        for block in layers:
            fully_shard(block, mesh=mesh)
    fully_shard(model, mesh=mesh)


def resolve_torch_dtype(dtype_name: str | None) -> torch.dtype:
    if dtype_name is None:
        return torch.bfloat16
    normalized = str(dtype_name).lower()
    if normalized in ("float32", "fp32"):
        return torch.float32
    if normalized in ("bfloat16", "bf16"):
        return torch.bfloat16
    if normalized in ("float16", "fp16", "half"):
        return torch.float16
    raise ValueError(f"Unsupported dtype {dtype_name!r}.")


def validate_parallel_config(cfg: DictConfig) -> None:
    num_train = cfg.test.num_train
    num_gen = cfg.test.num_gen
    train_engine_type = cfg.test.train_engine_type
    gen_tp = cfg.test.gen.tensor_parallel_size
    gen_pp = cfg.test.gen.pipeline_parallel_size
    gen_ep = cfg.test.gen.get("expert_parallel_size", 1)
    gen_dp = cfg.test.gen.get("data_parallel_size", None)

    assert num_gen % (gen_tp * gen_pp) == 0, (
        f"Gen TP*PP does not divide num_gen. num_gen={num_gen!r}, product={gen_tp * gen_pp!r}."
    )
    computed_dp = num_gen // (gen_tp * gen_pp)
    if gen_dp is not None and gen_dp > 1:
        assert computed_dp == gen_dp, (
            f"gen data_parallel_size={gen_dp} conflicts with computed DP="
            f"{computed_dp} (num_gen={num_gen}, TP={gen_tp}, PP={gen_pp})."
        )
    if gen_ep > 1:
        assert gen_ep == gen_tp, f"vLLM EP must match TP in PSRL. EP={gen_ep!r}, TP={gen_tp!r}."

    if train_engine_type == "fsdp_hybrid":
        ddp_size = cfg.test.fsdp_hybrid.ddp_size
        fsdp_size = cfg.test.fsdp_hybrid.fsdp_size
        assert num_train % (ddp_size * fsdp_size) == 0, (
            f"HSDP DDP*FSDP does not divide num_train. num_train={num_train!r}, product={ddp_size * fsdp_size!r}."
        )
    elif train_engine_type == "megatron":
        megatron = cfg.test.megatron
        train_tp = megatron.tensor_model_parallel_size
        train_pp = megatron.pipeline_model_parallel_size
        train_cp = megatron.get("context_parallel_size", 1)
        model_parallel_size = train_tp * train_pp * train_cp
        assert num_train % model_parallel_size == 0, (
            f"Megatron TP*PP*CP does not divide num_train. num_train={num_train!r}, product={model_parallel_size!r}."
        )


def assert_state_dict_matches_coords(
    state_dict: dict[str, torch.Tensor],
    offsets: dict[str, int],
    actor_label: str,
) -> None:
    """Judge a pulled state dict against the coordinate function it was filled with.

    Replaces the previous all-ones judge (item T1). Filling with 1.0 detects "no data
    arrived" and nothing else: a transposed tensor of ones is still ones, any slice of
    ones is ones, and ones cast to fp8 is still 1.0. Layout errors, shard misplacement,
    precision-conversion errors and MoE expert mix-ups -- the four things this design's
    format conversions can get wrong -- are all invisible to it.

    `assert_state_dict_matches_coord` reports the first offending global coordinate,
    which localises the fault to a dimension rather than just flagging the key.
    """
    assert state_dict, f"{actor_label} state_dict is empty."

    # Only keys the train side actually filled can be judged by coordinate. The two
    # sides do not share a key space: Megatron fuses attention into `linear_qkv` while
    # vLLM exposes `q_proj`/`k_proj`/`v_proj`, and the converters map between them. A
    # key present only on the gen side was produced by that mapping, not copied
    # verbatim, so its elements legitimately do not carry the sender's coordinates.
    shared = {k: v for k, v in state_dict.items() if k in offsets}
    assert shared, (
        f"{actor_label}: none of its {len(state_dict)} keys overlap the "
        f"{len(offsets)} keys the train side filled -- the two sides share no key "
        "space at all, which would make the whole judgement vacuous"
    )
    try:
        assert_state_dict_matches_coord(shared, offsets)
    except AssertionError as exc:
        raise AssertionError(f"{actor_label}: {exc}") from exc
    return len(shared), len(state_dict)


def make_dual_print(log_path, prefix=None):
    with open(log_path, "w") as _:
        pass

    def dual_print(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        if prefix:
            msg = f"[{prefix}] {msg}"
        print(msg, **kwargs)
        with open(log_path, "a") as f:
            print(msg, file=f, **kwargs)

    return dual_print


@ray.remote
class GlobalStore:
    def __init__(self):
        self.train_master_ip = None
        self.gen_master_ip = None
        self.train_master_ip_event = asyncio.Event()
        self.gen_master_ip_event = asyncio.Event()

    async def set_train_master_ip(self, train_master_ip):
        self.train_master_ip = train_master_ip
        self.train_master_ip_event.set()

    async def set_gen_master_ip(self, gen_master_ip):
        self.gen_master_ip = gen_master_ip
        self.gen_master_ip_event.set()

    async def get_train_master_ip(self):
        if self.train_master_ip is None:
            await self.train_master_ip_event.wait()
        return self.train_master_ip

    async def get_gen_master_ip(self):
        if self.gen_master_ip is None:
            await self.gen_master_ip_event.wait()
        return self.gen_master_ip


@ray.remote
class MetaServerActor:
    def __init__(self, server_name, psrl_config, expected_agents, log_dir):
        self.server = NIXLMetaServer(server_name, psrl_config.nixl)
        self.expected_agents = expected_agents
        self.client_name = server_name
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{self.client_name}.log")
        self.print = make_dual_print(log_path, prefix=self.client_name)

    def init_finished(self):
        self.print(f"Server init finished on ip {os.environ.get('LOCAL_IP')}")
        return True

    def protocol(self):
        self.print("step1: wait_for_client_shardings")
        self.server.wait_for_client_shardings(self.expected_agents, timeout=120)
        # self.print(f"client_sharding_dicts: {self.server.client_sharding_dicts}")
        self.print("step2: make_unified_sharding")
        self.server.make_unified_sharding()
        # self.print(f"client_unified_sharding_dicts: {self.server.client_unified_sharding_dicts}")
        self.print("step3: notify_all_client_shardings")
        self.server.notify_all_client_shardings()
        self.print("step4: wait_for_client_infos")
        self.server.wait_for_client_infos(self.expected_agents, timeout=120)
        # self.print(f"client_infos: {self.server.client_infos}")
        self.print("step5: make_comm_plan")
        self.server.make_comm_plan()
        self.print("step6: notify_all_client_infos_and_comm_plan")
        self.server.notify_all_client_infos_and_comm_plan()
        self.print("step7: wait_for_client_temp_mappings")
        self.server.wait_for_client_temp_mappings(self.expected_agents, timeout=120)
        self.print("step8: notify_all_client_temp_mappings")
        self.server.notify_all_client_temp_mappings()
        self.print("protocol done.")
        return self.server.client_unified_sharding_dicts

    def shutdown(self):
        self.server.shutdown()


@ray.remote(num_cpus=1)
class TrainClientActor:
    def __init__(
        self,
        global_store,
        engine_type,
        rank,
        world_size,
        server_name,
        psrl_config,
        backend,
        torch_port,
        log_dir,
        ps_for_push_worker_handles,
        model_path,
        model_config,
        fsdp_hybrid_config=None,
        megatron_config=None,
    ):
        # `fsdp2` is the name the config matrix and `psrl.worker` use; `fsdp` is this
        # harness's older spelling for the same thing, since the branch below calls
        # `fully_shard` either way and FSDP1 is rejected outright by config validation
        # (flat_param cannot give the in-place representation NIXL registration needs).
        # Accepting both keeps one vocabulary from having to be renamed across the matrix.
        assert engine_type in ["fsdp", "fsdp2", "fsdp_hybrid", "megatron"], (
            f"Unsupported engine type: {engine_type!r}."
        )
        if rank == 0:
            train_master_ip = ray.util.get_node_ip_address()
            ray.get(global_store.set_train_master_ip.remote(train_master_ip))
        else:
            train_master_ip = ray.get(global_store.get_train_master_ip.remote())
        os.environ["MASTER_ADDR"] = train_master_ip
        os.environ["MASTER_PORT"] = str(torch_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        self.engine_type = engine_type
        self.rank = rank
        self.world_size = world_size
        self.client_name = train_client_name(rank)
        self.model_config = model_config
        self.trust_remote_code = False
        self.train_dtype = resolve_torch_dtype(model_config.get("train_dtype", "bfloat16"))
        self.fsdp_hybrid_config = fsdp_hybrid_config
        self.megatron_config = megatron_config
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{self.client_name}.log")
        self.print = make_dual_print(log_path, prefix=self.client_name)
        self.print(f"train_master_ip: {train_master_ip}")
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        self.print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '')}")

        # NOTE(lhy): Create the client before loading the model.
        if engine_type == "megatron":
            self._init_megatron_parallel(megatron_config)
        self.client = NIXLStorageClient(
            client_name=self.client_name,
            server_name=server_name,
            use_gpu=True,
            client_type=NIXLClientType.PUSH_SIDE,
            nixl_config=psrl_config.nixl,
            replica_idx=0,
            worker_index=rank,
            logging_path=log_dir,
        )

        if engine_type in ("fsdp", "fsdp2", "fsdp_hybrid"):
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=self.train_dtype,
                trust_remote_code=self.trust_remote_code,
            )
            # Set all model parameters to a deterministic source value.
            for p in model.parameters():
                p.data.fill_(1)
            torch.manual_seed(42)
            # FSDP2 (fully_shard). FSDP1 is rejected by config validation because
            # flat_param cannot guarantee the in-place parameter representation that
            # registering with NIXL requires, so the harness must not build it either.
            if engine_type in ("fsdp", "fsdp2"):
                mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
                _fully_shard_transformer(model, mesh)
                self.model = model
            elif engine_type == "fsdp_hybrid":
                # HSDP: replicate across ddp_size, shard within fsdp_size.
                ddp_size = fsdp_hybrid_config.get("ddp_size", 2)
                fsdp_size = fsdp_hybrid_config.get("fsdp_size", 4)
                assert world_size % (ddp_size * fsdp_size) == 0, (
                    f"DDP*FSDP does not divide world_size. world_size={world_size!r}, "
                    f"product={ddp_size * fsdp_size!r}."
                )
                mesh = init_device_mesh(
                    "cuda",
                    mesh_shape=(ddp_size, fsdp_size),
                    mesh_dim_names=("replicate", "shard"),
                )
                _fully_shard_transformer(model, mesh)
                self.model = model
        elif engine_type == "megatron":
            set_random_seed(42)
            self._init_megatron_model(model_path)
        self.ps_for_push_worker_handles = ps_for_push_worker_handles
        # self._log_env_info()

    def _log_env_info(self):
        for k in sorted(os.environ):
            self.print(f"{k}={os.environ[k]}")

    def _get_replica_id(self):
        if self.engine_type in ("fsdp", "fsdp2"):
            return 0
        elif self.engine_type == "fsdp_hybrid":
            return self.rank // self.fsdp_hybrid_config.get("fsdp_size", 4)
        elif self.engine_type == "megatron":
            assert mpu.is_initialized(), "Megatron parallel is not initialized"
            return mpu.get_data_parallel_rank()

    def _init_megatron_parallel(self, megatron_config):
        """Initialize Megatron parallel state."""
        virtual_pipeline_model_parallel_size = megatron_config.get("virtual_pipeline_model_parallel_size", 1)
        if virtual_pipeline_model_parallel_size == 1:
            virtual_pipeline_model_parallel_size = None
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=megatron_config.get("tensor_model_parallel_size", 4),
            pipeline_model_parallel_size=megatron_config.get("pipeline_model_parallel_size", 2),
            virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
            use_sharp=False,
            context_parallel_size=megatron_config.get("context_parallel_size", 1),
            expert_model_parallel_size=megatron_config.get("expert_model_parallel_size", 1),
            expert_tensor_parallel_size=megatron_config.get("expert_tensor_parallel_size", 1),
            nccl_communicator_config_path=None,
        )
        self.print(f"[Rank {self.rank}] Megatron parallel initialized")

    def _init_megatron_model(self, model_path):
        """Initialize a Megatron-Bridge model like the current train worker."""
        local_model_path = copy_to_local(model_path)
        hf_config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=self.trust_remote_code)

        dtype = PrecisionType.to_dtype(self.train_dtype)
        self.print(f"[Rank {self.rank}] Config loaded: {hf_config.model_type}")

        use_mbridge = True

        if use_mbridge:
            from megatron.bridge import AutoBridge

            bridge = AutoBridge.from_hf_config(hf_config)
            provider = bridge.to_megatron_provider(load_weights=False)
            virtual_pipeline_model_parallel_size = self.megatron_config.get("virtual_pipeline_model_parallel_size", 1)
            if virtual_pipeline_model_parallel_size == 1:
                virtual_pipeline_model_parallel_size = None
            provider_overrides = {
                "tensor_model_parallel_size": self.megatron_config.get("tensor_model_parallel_size", 4),
                "pipeline_model_parallel_size": self.megatron_config.get("pipeline_model_parallel_size", 2),
                "expert_model_parallel_size": self.megatron_config.get("expert_model_parallel_size", 1),
                "expert_tensor_parallel_size": self.megatron_config.get("expert_tensor_parallel_size", 1),
                "virtual_pipeline_model_parallel_size": virtual_pipeline_model_parallel_size,
                "context_parallel_size": self.megatron_config.get("context_parallel_size", 1),
                "sequence_parallel": self.megatron_config.get("sequence_parallel", False),
            }
            provider.apply_overrides_and_finalize(
                dtype=dtype,
                overrides=provider_overrides,
            )
            tf_config = None
        else:
            tf_config = hf_to_mcore_config(hf_config, dtype)

        from verl.utils.megatron_utils import McoreModuleWrapperConfig, make_megatron_module

        wrap_config = McoreModuleWrapperConfig()
        actor_module, updated_tf_config = make_megatron_module(
            wrap_config,
            tf_config,
            hf_config,
            bridge=bridge if use_mbridge else None,
            provider=provider,
        )
        if use_mbridge:
            tf_config = updated_tf_config
        self.model = actor_module

        for name, param in self.model[0].named_parameters():
            param.data.fill_(1)
        for name, param in self.model[0].named_buffers():
            param.data.fill_(1)

        self.print(f"[Rank {self.rank}] Model initialized: {self.model}")

    def init_finished(self):
        self.print(f"Train client init finished on ip {os.environ.get('LOCAL_IP')}")
        return True

    def protocol(self, model_path):
        self.print("step0: convert_fsdp/megatron_inplace")
        if self.engine_type == "fsdp" or self.engine_type == "fsdp_hybrid":
            from transformers import AutoConfig

            model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=self.trust_remote_code)
            parameter_mapping = create_parameter_mapping("FSDP", model_config)
            state_dict, sharding = convert_fsdp_inplace(parameter_mapping, self.model)
        elif self.engine_type == "megatron":
            from transformers import AutoConfig

            model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=self.trust_remote_code)
            parameter_mapping = create_parameter_mapping("Megatron", model_config)
            conversion_result = convert_megatron_inplace(parameter_mapping, self.model)
            state_dict = conversion_result.state_dict
            sharding = conversion_result.sharding_dict
        # Fill with a function of each element's global coordinate rather than with 1.0
        # (item T1). The all-ones fill cannot distinguish a correct transfer from a
        # transposed, misplaced, or wrongly-cast one -- see assert_state_dict_matches_coords.
        # The offsets are keyed by state-dict key and must reach the gen side verbatim,
        # which is why they are returned rather than recomputed there.
        self.coord_offsets = fill_state_dict_by_coord(state_dict, strict=False)
        # self.print(f"state_dict keys: {state_dict.keys()}")
        self.state_dict = state_dict
        self.sharding = sharding
        self.state_dict_keys = list(state_dict.keys())
        self.unified_sharding = None
        self.print("step1: connect_to_server")
        self.client.connect_to_server()
        self.print("step2: send_local_sharding")
        self.client.send_local_sharding(sharding)
        self.print("step3: wait_for_server_sharding")
        self.unified_sharding = self.client.wait_for_server_sharding()
        self.print("step4: register_local_tensors")
        self.client.register_local_tensors(self.state_dict, self.unified_sharding)
        self.print("step5: send_local_info")
        self.client.send_local_info()
        self.print("step6: wait_for_server_info (client infos & comm plan)")
        self.client.wait_for_server_info()
        self.print("step7: send_local_temp_mapping")
        self.client.send_local_temp_mapping()
        self.print("step8: wait_for_server_temp_mappings")
        self.client.wait_for_server_temp_mappings()
        self.print("protocol done.")

    def push_to_ps(self, ps_agent_names, ps_client_names):
        futures = []
        for key in self.state_dict_keys:
            wait_operations = []
            for ps_agent_name, ps_client_name in zip(ps_agent_names, ps_client_names):
                shards_to_transfer = self.client.client_write(
                    ps_agent_name,
                    ps_client_name,
                    key,
                    "train_push",
                    merge_and_cache_xfer=False,
                )
                if len(shards_to_transfer) > 0:
                    wait_operations.append((key, ps_client_name, shards_to_transfer))
                    # self.print(f"Pushing {key} to {ps_client_name}")
            # self.print(f"Waiting for {len(wait_operations)} push operations")
            for key, ps_client_name, shards_to_transfer in wait_operations:
                try:
                    self.client.wait(key, "train_push", "WRITE", target_client=ps_client_name)
                except Exception as e:
                    self.print(f"Wait failed for key {key} to {ps_client_name}. error: {e}")
                    raise e
                futures.append(
                    self.ps_for_push_worker_handles[ps_client_name].transfer_train_to_gen.remote(
                        key, shards_to_transfer
                    )
                )
        ray.get(futures)
        # self.client.merge_and_finish_cached_xfer()

    def advance_version(self, base: int):
        """Refill the weights at a new coordinate base, standing in for a train step.

        A second eval round is only a test if the weights it pulls are *different* from
        the first. Re-filling at the same base reproduces identical bytes, so a rollout
        that received nothing would pass exactly as a rollout that received the new
        version -- which is the confusion the steady-state switch-back check exists to
        resolve (item D4).

        Returns the new offsets because they are what the judge compares against; the
        gen side cannot derive them, since the shift is this call's argument.
        """
        self.coord_offsets = fill_state_dict_by_coord(self.state_dict, strict=False, base=base)
        # NIXL reads the registered tensors directly, so there is nothing to re-register:
        # the addresses are unchanged and only their contents moved. Re-registering here
        # would invalidate every remote descriptor for no reason.
        torch.cuda.synchronize()
        return self.coord_offsets

    def get_coord_offsets(self):
        """The per-key global coordinate offsets this rank filled with.

        The gen side needs exactly these to judge what it pulled: recomputing them
        there would re-derive the same numbers from the same key ordering and so would
        not catch a key-set mismatch between the two sides.
        """
        return self.coord_offsets

    def get_nixl_names(self):
        """This rank's (agent, client) names, for rollouts reading it directly (T2).

        A train actor is its own NIXL agent, so both names are the same string. The gen
        side's pull takes an (agent, client) pair either way, which is why direct supply
        needs no second read path -- only a different pair.
        """
        return self.client_name, self.client_name

    def shutdown(self):
        self.client.shutdown()
        dist.destroy_process_group()


@ray.remote(num_cpus=1)
class GenClientActor:
    def __init__(
        self,
        global_store,
        rank,
        world_size,
        server_name,
        psrl_config,
        backend,
        torch_port,
        log_dir,
        model_path,
        model_config,
        gen_config,
    ):
        from vllm import LLM

        if rank == 0:
            gen_master_ip = ray.util.get_node_ip_address()
            ray.get(global_store.set_gen_master_ip.remote(gen_master_ip))
            """
            os.environ["UCX_LOG_LEVEL"] = "debug"
            os.environ["NIXL_LOG_LEVEL"] = "debug"
            f = open(os.path.join(log_dir, "ucx.log"), "w")
            os.dup2(f.fileno(), sys.stdout.fileno())
            os.dup2(f.fileno(), sys.stderr.fileno())
            """
        else:
            gen_master_ip = ray.get(global_store.get_gen_master_ip.remote())
        os.environ["MASTER_ADDR"] = gen_master_ip
        os.environ["MASTER_PORT"] = str(torch_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        self.tp_size = gen_config.get("tensor_parallel_size", 2)
        self.pp_size = gen_config.get("pipeline_parallel_size", 1)
        self.ep_size = gen_config.get("expert_parallel_size", 1)
        self.rank = rank
        self.tp_rank = rank % self.tp_size
        self.instance_world_size = self.tp_size * self.pp_size
        #: Chunk-wave schedule and chunk->keys map (item D1). Built once and reused
        #: across waves: the schedule derives from the plan, which does not change
        #: between waves, and rebuilding per wave would make the pipeline's own cost
        #: grow with the wave count -- the opposite of the point.
        self._cached_waves = None
        self._chunk_keys = None
        assert world_size % (self.tp_size * self.pp_size) == 0, (
            f"TP*PP does not divide world_size. world_size={world_size!r}, product={self.tp_size * self.pp_size!r}."
        )
        # Use explicit data_parallel_size from config if provided, otherwise compute from world_size
        config_dp_size = gen_config.get("data_parallel_size", None)
        computed_dp_size = world_size // self.instance_world_size
        if config_dp_size is not None and config_dp_size > 1:
            assert computed_dp_size == config_dp_size, (
                f"Computed and configured data parallel sizes differ. computed={computed_dp_size!r}, "
                f"configured={config_dp_size!r}, num_gen={world_size!r}, TP={self.tp_size!r}, PP={self.pp_size!r}."
            )
        self.dp_size = computed_dp_size
        self.model_config = model_config
        # None means "use vLLM's default", which is what disaggregate wants. Colocate
        # sets it, because the card is shared with a train rank.
        self.gpu_memory_utilization = gen_config.get("gpu_memory_utilization", None)
        self.trust_remote_code = False
        self.gen_dtype = model_config.get("gen_dtype", "bfloat16")
        self.client_name = gen_client_name(0, rank)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{self.client_name}.log")
        self.print = make_dual_print(log_path, prefix=self.client_name)
        self.print(f"gen_master_ip: {gen_master_ip}")
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        torch.manual_seed(42)
        os.environ["LOCAL_RANK"] = str(0)
        self.print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '')}")

        # NOTE(lhy): Create the client before loading the model.
        self.client = NIXLStorageClient(
            client_name=self.client_name,
            server_name=server_name,
            use_gpu=True,
            client_type=NIXLClientType.PULL_SIDE,
            nixl_config=psrl_config.nixl,
            replica_idx=0,
            worker_index=rank,
            logging_path=log_dir,
        )

        llm_kwargs = {
            "model": model_path,
            "dtype": self.gen_dtype,
            "tensor_parallel_size": self.tp_size,
            "pipeline_parallel_size": self.pp_size,
            "distributed_executor_backend": "external_launcher",
            "enforce_eager": True,
            "disable_custom_all_reduce": True,
            "trust_remote_code": self.trust_remote_code,
            "enable_expert_parallel": self.ep_size > 1,
            "seed": 42,
        }
        if self.dp_size > 1:
            llm_kwargs["data_parallel_size"] = self.dp_size
        if self.gpu_memory_utilization is not None:
            # Colocate shares the card with a train rank, so vLLM's default (>0.9) is not
            # available: it reserves a fraction of *total* memory and refuses to start
            # when that much is not free. With a trainer already holding its parameters
            # the startup check fails with "Free memory on device ... is less than desired
            # GPU memory utilization", which is the design's memory account showing up as
            # an error rather than a bug.
            #
            # Disaggregate leaves this None and keeps vLLM's default, so the two
            # placements do not share a tuning constant that only one of them needs.
            llm_kwargs["gpu_memory_utilization"] = self.gpu_memory_utilization
        llm = LLM(**llm_kwargs)
        self.model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        self.print(f"local rank {self.rank} model: {self.model}")
        """
        seen_module_prefixes = set()
        for module_prefix, module in self.model.named_modules():
            seen_module_prefixes.add(module_prefix)
        self.print(f"seen_module_prefixes: {seen_module_prefixes}, "
                   f"has lm_head: {'lm_head' in seen_module_prefixes}, "
                   f"has lm_head module: {hasattr(self.model, 'lm_head')}")
        """

    def _get_replica_id(self):
        return self.rank // self.instance_world_size

    def _get_ep_rank(self):
        if self.ep_size <= 1:
            return 0
        try:
            from vllm.distributed.parallel_state import get_ep_group

            return get_ep_group().rank_in_group
        except AssertionError:
            return 0

    def init_finished(self):
        self.print(f"Gen client init finished on ip {os.environ.get('LOCAL_IP')}")
        return True

    def get_state_dict(self):
        return {k: v.cpu() for k, v in self.state_dict.items()}

    def get_nixl_names(self):
        """This client's (agent, client) names, for peers reading it (item D1).

        A gen actor is its own NIXL agent, so both names are the same string -- the same
        shape the train side reports, which is what lets the peer phase reuse the pull
        path unchanged.
        """
        return self.client_name, self.client_name

    def protocol(self, model_path):
        self.print("step0: convert_vllm_inplace")
        from transformers import AutoConfig
        from vllm.compilation.cuda_graph import CUDAGraphWrapper
        from vllm_patches.interfaces import supports_weight_layout

        vllm_model = self.model
        if isinstance(vllm_model, CUDAGraphWrapper):
            vllm_model = vllm_model.unwrap()
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=self.trust_remote_code)
        param_mapping = (
            None if supports_weight_layout(vllm_model) else create_parameter_mapping(type(vllm_model), model_config)
        )
        state_dict, sharding = convert_vllm_inplace(
            vllm_model,
            tp_rank=self.tp_rank,
            ep_rank=self._get_ep_rank(),
            parameter_mapping=param_mapping,
        )
        # self.print(f"state_dict keys: {list(state_dict.keys())}")
        self.state_dict = state_dict
        self.sharding = sharding
        self.state_dict_keys = list(state_dict.keys())
        self.unified_sharding = None
        self.print("step1: connect_to_server")
        self.client.connect_to_server()
        self.print("step2: send_local_sharding")
        self.client.send_local_sharding(sharding)
        self.print("step3: wait_for_server_sharding")
        self.unified_sharding = self.client.wait_for_server_sharding()
        # self.print(f"unified_sharding: {self.unified_sharding}")
        self.print("step4: register_local_tensors")
        self.client.register_local_tensors(self.state_dict, self.unified_sharding)
        self.print("step5: send_local_info")
        self.client.send_local_info()
        self.print("step6: wait_for_server_info (client infos & comm plan)")
        self.client.wait_for_server_info()
        # self.print(f"comm plan: {self.client._comm_plan}")
        self.print("step7: send_local_temp_mapping")
        self.client.send_local_temp_mapping()
        self.print("step8: wait_for_server_temp_mappings")
        self.client.wait_for_server_temp_mappings()
        self.print("protocol done.")

    def pull_from_ps(self, ps_agent_names, ps_client_names, phase="all"):
        """Pull weights. ``phase`` selects which half of the plan to issue.

        D1's peer reads are only valid once the peer has finished its own upstream read,
        so the two need a barrier between them. The driver already supplies one -- it
        `ray.get`s every gen actor -- so running this twice, as ``upstream`` then
        ``peer``, is sufficient; no new synchronisation primitive is required.

        ``all`` keeps the single-phase behaviour, which is what the PS path wants: there
        is nothing to diffuse there and a second round trip through the driver would be
        pure overhead.
        """
        assert phase in ("all", "upstream", "peer") or phase.startswith("wave:"), (
            f"unknown pull phase {phase!r}"
        )
        wait_operations = []
        total_start_time = time.time()
        agent_of = dict(zip(ps_client_names, ps_agent_names))

        # D1: iterate the work the plan assigns rather than probing every
        # (key, source) pair. Against a PS the two are nearly the same -- a couple of
        # workers, each holding every key. Under direct supply the sources are train
        # ranks and pipeline parallelism gives each one a slice of the key space, so
        # probing costs 399 x 24 = 9,576 calls to schedule 399 reads.
        #
        # Falling back to the probe when the plan has no entry keeps this honest: an
        # empty plan means the planner did not cover this client, and silently reading
        # nothing would look like a fast success.
        planned = list(self.client.iter_planned_pull_work())
        if planned:
            work = list(planned)
        else:
            work = [(key, source) for key in self.state_dict_keys for source in ps_client_names]

        # D1: a peer read is only valid once that peer has finished its own upstream
        # read. Issuing it early does not fail -- the registration is valid and the
        # transfer is well formed -- it succeeds against an unfilled buffer and yields
        # silently wrong weights. So the two must be separated by a barrier, which the
        # driver supplies by `ray.get`-ing every actor between the phases.
        #
        # Partition per key, not against a flat set of peer names: a client can be a
        # peer for one key and an upstream source for another. Filtering on `agent_of`
        # first -- the train-client map -- would drop peer reads before this point and
        # report zero of them, a count of nothing rather than evidence that diffusion
        # found no duplication.
        peer_work = [
            (key, src) for key, src in work if src in self.client.peer_sources_for(key)
        ]
        upstream_work = [
            (key, src)
            for key, src in work
            if src not in self.client.peer_sources_for(key) and src in agent_of
        ]
        # Shards a peer will supply are not pulled over the NIC at all -- that omission
        # *is* the traffic claim. Safe only because the peer phase follows the barrier;
        # in single-phase mode the full upstream list is used instead.
        #
        # Exclusion is per *shard*, not per key. A key is only partly peer-covered when
        # the client wants several of its shards and a peer holds some: excluding the
        # whole key would drop the upstream read for the rest, losing those bytes with
        # no error. `client_read` is told exactly which shards to skip.
        peer_covered: dict[str, set] = {}
        for key in {k for k, _ in peer_work}:
            for shard_list in self.client.peer_shards_for(key).values():
                peer_covered.setdefault(key, set()).update(shard_list)
        upstream_without_peer_covered = [
            (key, src)
            for key, src in upstream_work
            if not self._shards_fully_covered(key, src, peer_covered)
        ]

        if phase.startswith("wave:"):
            # D1 chunk pipeline: issue only the reads belonging to one chunk, so a
            # follower's peer read of chunk i overlaps the designate's work on chunk
            # i+1. The driver advances the index and barriers between waves, which is
            # what keeps the per-chunk dependency honest.
            wave_index = int(phase.split(":", 1)[1])
            work = self._wave_work(wave_index)
        elif phase == "upstream":
            work = upstream_without_peer_covered
        elif phase == "peer":
            work = peer_work
        else:
            work = upstream_work
        deferred_peer_reads = peer_work if phase in ("all", "upstream") else []

        # The tag must differ per wave. `wait` keys on (key, tag, op, target), so
        # reusing one tag across waves makes wave i+1's read collide with wave i's
        # already-completed handle for the same key and source -- the second wait would
        # match the wrong transfer.
        tag = f"gen_pull_{phase}"
        # In wave mode the plan is already partitioned: `_wave_work` returns exactly the
        # reads this wave should issue, and a shard appears either as an upstream read or
        # as a peer read, never both. Applying the whole-model `peer_covered` set on top
        # of that skipped shards whose peer read lives in a *different* wave -- 266
        # scheduled reads produced only 101 transfers, and the coordinate judge caught
        # the resulting gap. The wave partition is the authority; nothing is subtracted
        # from it.
        skip_upstream = phase == "upstream"
        for key, source_client in work:
            # Outside wave mode the two lists overlap by construction (the upstream read
            # is kept whole so a caller may skip peer reads), so a peer-covered shard
            # must be excluded from the upstream read -- but never from the peer read
            # itself, which exists precisely to fetch it.
            is_peer_read = source_client in self.client.peer_sources_for(key)
            shards_to_transfer = self.client.client_read(
                agent_of[source_client],
                source_client,
                key,
                tag,
                merge_and_cache_xfer=False,
                skip_shards=(
                    peer_covered.get(key) if (skip_upstream and not is_peer_read) else None
                ),
            )
            if len(shards_to_transfer) > 0:
                wait_operations.append((key, source_client, shards_to_transfer))
        for key, source_client, shards_to_transfer in wait_operations:
            self.client.wait(key, tag, "READ", target_client=source_client)
        self.client.merge_and_finish_cached_xfer()
        total_end_time = time.time()
        # torch.cuda.synchronize()
        self.print(
            f"Total pull from ps done: {total_end_time - total_start_time}s "
            f"({len(work)} scheduled reads, {len(wait_operations)} transfers, "
            f"{len(deferred_peer_reads)} peer reads deferred)"
        )

    def chunk_wave_count(self) -> int:
        """Number of chunk waves this client's plan implies (item D1).

        The driver takes the max across clients so every client is asked for every wave
        index; a client with fewer waves returns no work for the surplus ones rather
        than erroring, which keeps the loop uniform.
        """
        self._wave_work(0)  # builds the cache
        return len(self._cached_waves or [])

    def pull_pipelined(self, source_agent_names, source_client_names, version=1):
        """Run this client's whole pull as one pipelined pass (item D1).

        Unlike `pull_from_ps(phase=...)` this returns only when the client is done, so
        the driver issues one call per actor and no barrier between waves. That is the
        point: a driver barrier is global, and a global barrier makes every client wait
        for the slowest one at every chunk, which is what made the two-phase version add
        cost instead of hiding it.

        The dependency that replaces the barrier is per-chunk and per-source: before
        peer-reading chunk ``i`` this client waits for exactly the designates that serve
        it, and nothing else. A designate publishes chunk ``i`` the moment its own read
        of ``i`` has completed, so a follower proceeds as soon as *its* data is ready
        rather than when everyone's is.

        Both roles run in the same loop because a client is typically both -- designate
        for the shards it reads upstream, follower for the ones it does not. Splitting
        them into two passes would reintroduce the barrier through the back door.
        """
        total_start_time = time.time()
        agent_of = dict(zip(source_client_names, source_agent_names))
        publisher, waiter = self.client.chunk_completion_channel()
        followers_by_chunk = self.client.peer_followers_by_chunk()

        if self._cached_waves is None:
            self._wave_work(0)
        waves = self._cached_waves or []

        # The schedule is per shard; the transport is per (key, source). Coalescing is
        # what keeps a pair from being posted twice -- `client_read` moves all of a
        # pair's shards at once, so a pair spanning waves would repost a live handle.
        from psrl.utils.nixl.chunk_pipeline import coalesce_to_transport_pairs

        wave_pairs = coalesce_to_transport_pairs(waves)
        ready_after = {wave.index: wave.ready_after for wave in waves}

        issued = 0
        transfers = 0
        published = 0
        waited_on = 0
        # Where the time goes, split by cause. Two hypotheses have already been wrong
        # here -- "the wave barrier costs it" (removing the barrier made it *worse*:
        # 2.59x vs 1.99x) and before that "peer reads are slow because of the link"
        # (same-node is 14x faster per byte than cross-node). Both were reasoned from
        # totals. These counters make the next conclusion come from the breakdown.
        t_post_upstream = 0.0
        t_wait_upstream = 0.0
        t_gate = 0.0
        t_post_peer = 0.0
        t_wait_peer = 0.0

        # Posting is separated from waiting. An upstream read depends on nothing, so a
        # wave boundary must not block it; only *peer* reads have a dependency.
        #
        # But posting *all* waves before waiting on any delays the first publish until
        # every read is submitted, and submission is not free: the designate's
        # post_up=1.62s for 653 reads is time its followers spend gated (measured
        # gate=2.42s). So posting is done wave by wave, interleaved with the waiting and
        # publishing below -- wave i's reads are submitted, waited on, and published
        # before wave i+1 is submitted. That keeps the first publish early, which is the
        # only thing a follower actually needs.
        posted_by_wave: dict[int, list[tuple[str, str, str]]] = {}

        def post_upstream_wave(wave_index, pairs):
            nonlocal issued, t_post_upstream
            post_start = time.time()
            for key, source, peer in pairs:
                if peer:
                    continue
                issued += 1
                tag = f"gen_pipelined_up_w{wave_index}"
                shards = self.client.client_read(
                    agent_of.get(source, source),
                    source,
                    key,
                    tag,
                    merge_and_cache_xfer=False,
                )
                if len(shards) > 0:
                    posted_by_wave.setdefault(wave_index, []).append((key, source, tag))
            t_post_upstream += time.time() - post_start

        for wave_index, pairs in wave_pairs:
            post_upstream_wave(wave_index, pairs)
            peer_pairs = [(key, source) for key, source, peer in pairs if peer]

            # Wait for this wave's own upstream transfers before publishing it. Waiting
            # per wave rather than for everything is what keeps the publish points
            # early: a follower gets chunk 0 while chunks 1..n are still in flight.
            wait_start = time.time()
            for key, source_client, tag in posted_by_wave.get(wave_index, []):
                self.client.wait(key, tag, "READ", target_client=source_client)
                transfers += 1
            t_wait_upstream += time.time() - wait_start

            # Publish only after this wave's reads have landed. Publishing earlier is
            # the one way to break ordering: a follower would read a buffer still being
            # filled, and that read succeeds.
            followers = followers_by_chunk.get(wave_index)
            if followers:
                published += publisher.publish(version, wave_index, sorted(followers))

            if peer_pairs:
                # Wait per source, not for "the wave": a chunk can hold shards from
                # several designates, and proceeding once the first reports would read
                # the others' buffers before they are filled -- successfully, which is
                # why this cannot be left to a retry.
                sources = sorted({source for _key, source in peer_pairs})
                depends_on = ready_after.get(wave_index)
                if depends_on is None:
                    depends_on = wave_index - 1
                gate_start = time.time()
                waiter.wait_for(version, depends_on, sources, timeout=600.0)
                t_gate += time.time() - gate_start
                waited_on += len(sources)

                peer_tag = f"gen_pipelined_peer_w{wave_index}"
                peer_waits = []
                peer_post_start = time.time()
                # Peer reads are posted individually. Merging them was tried and is
                # **measurably worse**: post_peer went 4.105s -> 6.270s, because the
                # merge path materialises every descriptor into a Python list and these
                # shards are strided at 4096 descriptors each (399 x 4096 is 1.6M
                # element accesses). The prepped path did engage -- no fallback warning --
                # so the regression is the descriptor marshalling ahead of it, not NIXL.
                #
                # Which means `post_peer` is **not** submission cost, and that is now a
                # measured fact rather than an inference: three attempts to reduce it as
                # if it were (remove the barrier, merge, pre-post all waves) each made it
                # the same or worse. What it is instead has to come from instrumenting
                # inside `client_read`, which is the next step. Leaving it unmerged is
                # the configuration with the best measured number so far (1.93x).
                for key, source in peer_pairs:
                    issued += 1
                    shards = self.client.client_read(
                        agent_of.get(source, source),
                        source,
                        key,
                        peer_tag,
                        merge_and_cache_xfer=False,
                    )
                    if len(shards) > 0:
                        peer_waits.append((key, source))
                t_post_peer += time.time() - peer_post_start
                peer_wait_start = time.time()
                for key, source_client in peer_waits:
                    self.client.wait(key, peer_tag, "READ", target_client=source_client)
                t_wait_peer += time.time() - peer_wait_start
                transfers += len(peer_waits)

        self.client.merge_and_finish_cached_xfer()
        total_end_time = time.time()
        phases = self.client.read_phase_us()
        self.print(
            f"Pipelined pull done: {total_end_time - total_start_time}s "
            f"({len(waves)} waves, {len(wave_pairs)} posting waves, {issued} pairs, "
            f"{transfers} transfers, {published} notifs published, "
            f"{waited_on} source-waits, {waiter.drains} drains) "
            f"[breakdown post_up={t_post_upstream:.3f} wait_up={t_wait_upstream:.3f} "
            f"gate={t_gate:.3f} post_peer={t_post_peer:.3f} wait_peer={t_wait_peer:.3f}] "
            f"[inside client_read, seconds: "
            f"deser={phases['deserialize'] / 1e6:.3f} align={phases['align'] / 1e6:.3f} "
            f"init={phases['initialize'] / 1e6:.3f} xfer={phases['transfer'] / 1e6:.3f}]"
        )
        return {
            "waves": len(waves),
            "posting_waves": len(wave_pairs),
            "issued": issued,
            "transfers": transfers,
            "published": published,
            "waited_on": waited_on,
            "drains": waiter.drains,
            "seconds": total_end_time - total_start_time,
            "post_upstream": t_post_upstream,
            "wait_upstream": t_wait_upstream,
            "gate": t_gate,
            "post_peer": t_post_peer,
            "wait_peer": t_wait_peer,
        }

    def _wave_work(self, wave_index: int) -> list:
        """``(key, source)`` pairs belonging to chunk ``wave_index`` (item D1).

        Derived from the chunk plan rather than recomputed here: the plan already
        decides which shards a chunk carries, and a second definition of that boundary
        is how the two would drift apart.

        An out-of-range index returns nothing rather than raising, so the driver can
        advance past the end without knowing the wave count in advance.
        """
        from psrl.utils.nixl.chunk_pipeline import schedule_pipelined_reads

        if self._cached_waves is None:
            chunks, _groups = self.client.pull_chunks()
            upstream = {
                key: self.client.upstream_sources_for(key) for key in self.state_dict_keys
            }
            peer = {key: self.client.peer_shards_for(key) for key in self.state_dict_keys}
            self._cached_waves = schedule_pipelined_reads(
                chunks,
                {k: v for k, v in upstream.items() if v},
                {k: v for k, v in peer.items() if v},
                self.client_name,
            )
        if wave_index >= len(self._cached_waves):
            return []
        # Take the (key, source) pairs the schedule recorded. Deriving them from chunk
        # membership instead emitted reads nobody planned and dropped planned ones: a
        # chunk holds ~100 keys at this model size, so "every key in this chunk" is not
        # the same set as "the reads placed in this chunk". That mistake produced wrong
        # weights that the coordinate judge caught -- 100 + 269 reads issued against 653
        # planned.
        seen, pairs = set(), []
        for read in self._cached_waves[wave_index].reads:
            pair = (read.key, read.source)
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)
        return pairs

    def _shards_fully_covered(self, key, source, peer_covered) -> bool:
        """Whether every shard ``source`` would serve is already covered by a peer.

        Only then is dropping the upstream read safe. Partial coverage keeps the read
        and excludes just the covered shards, which is why this returns a verdict rather
        than the caller assuming key-level coverage: a key with several shards can be
        half-covered, and dropping it whole would lose the remainder silently.
        """
        covered = peer_covered.get(key)
        if not covered:
            return False
        planned = self.client.upstream_shards_for(key, source)
        return bool(planned) and all(shard in covered for shard in planned)

    def shutdown(self):
        self.client.shutdown()


def create_ps_worker_group(train_engine_type, num_ps, psrl_config, model_config):
    ps_model_config = OmegaConf.create(
        {
            "path": model_config.path,
            "use_shm": False,
            "trust_remote_code": False,
        }
    )
    ray_nodes = ray.nodes()
    ray_nodes_sorted = sorted(ray_nodes, key=lambda n: n["NodeManagerAddress"])
    nodes = [ray_nodes_sorted[-i - 1] for i in range(num_ps)]
    ps_resource_pool = PSResourcePool(
        [
            PSResourceSpec(
                node_ip=node["NodeManagerAddress"],
                node_id=node["NodeID"],
                attached_gpu_id=None,
            )
            for node in nodes
        ]
    )
    storage_plan = PSStoragePlan(
        train_model_dtype=resolve_torch_dtype(model_config.get("train_dtype", "bfloat16")),
        gen_model_dtype=resolve_torch_dtype(model_config.get("gen_dtype", "bfloat16")),
    )
    ps_cls_with_init = PSClassWithInitArgs(
        ray.remote(PSStorageWorker),
        storage_plan,
        ps_model_config,
        psrl_config,
    )
    ps_wg = PSWorkerGroup(ps_resource_pool, ps_cls_with_init)
    return ps_wg


def _assert_actors_run_this_tree() -> None:
    """Fail loudly if Ray actors import a different checkout than the driver.

    The cluster environment puts another `psrl` on the path, so without the PYTHONPATH
    override the actors silently run that one while the driver runs this one -- and the
    run *passes*, proving nothing about the code under test. A one-line check here is
    worth far more than the minutes it costs, because the failure it catches is
    invisible in the result.

    Checked on every node, not just one: a partial PYTHONPATH would be the same bug with
    a smaller blast radius.
    """

    @ray.remote(num_cpus=0)
    def _where() -> str:
        import psrl.utils.nixl.client as module

        return module.__file__

    nodes = [node for node in ray.nodes() if node["Alive"]]
    seen = ray.get(
        [
            _where.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
            ).remote()
            for node in nodes
        ]
    )
    expected = os.path.join(_REPO_ROOT, "psrl", "utils", "nixl", "client.py")
    wrong = sorted({path for path in seen if path != expected})
    assert not wrong, (
        f"Ray actors import a different checkout than the driver.\n"
        f"  expected: {expected}\n"
        f"  actors:   {wrong}\n"
        "The run would pass while measuring code this tree does not contain."
    )
    print(f"[PASS] all {len(seen)} nodes import this tree: {expected}")


@hydra.main(config_path="config", config_name="nixl_e2e", version_base=None)
def test_nixl_e2e(cfg: DictConfig):
    validate_parallel_config(cfg)
    log_dir = cfg.logging.path
    os.makedirs(log_dir, exist_ok=True)
    ray.init(
        ignore_reinit_error=True,
        runtime_env={
            "env_vars": {
                "PSRL_LOGGING_PATH": log_dir,
                # Ray worker processes resolve `psrl` through the cluster's shared
                # environment, which points at a *different* checkout
                # (.../psrl_main/psrl). Without this the driver runs this tree while
                # every actor runs that one, so the harness silently measures code the
                # working tree does not contain -- a passing run that proves nothing
                # about the change under test. Prepending the repo root makes the
                # actors import what the driver does.
                "PYTHONPATH": f"{_REPO_ROOT}:{os.environ.get('PYTHONPATH', '')}".rstrip(":"),
            }
        },
    )
    _assert_actors_run_this_tree()
    listen_ip = cfg.network.listen_ip
    print(f"meta server listen_ip: {listen_ip}")
    server_name = NIXL_META_SERVER_NAME
    backend = cfg.network.backend
    torch_port_train = cfg.network.torch_port_train
    torch_port_gen = cfg.network.torch_port_gen
    # Number of train GPUs
    num_train = cfg.test.num_train
    # Number of gen GPUs
    num_gen = cfg.test.num_gen
    # T2 / D3: with direct supply the rollout reads the train clients and no PS exists.
    # Defaulting to False keeps every existing config on the PS path, so the two can be
    # compared on the same cluster rather than one replacing the other untested.
    direct_supply = bool(cfg.test.get("direct_supply", False))
    # Number of PS Nodes (align with the number of nodes used for gen)
    num_ps = 0 if direct_supply else math.ceil(num_gen / NUM_GPU_PER_NODE)
    ray_nodes = ray.nodes()
    ray_nodes_sorted = sorted(ray_nodes, key=lambda n: n["NodeManagerAddress"])
    # Use the first num_train / NUM_GPU_PER_NODE nodes for train
    train_nnodes = math.ceil(num_train / NUM_GPU_PER_NODE)
    assert len(ray_nodes_sorted) >= train_nnodes, (
        f"Ray has too few train nodes. required={train_nnodes!r}, available={len(ray_nodes_sorted)!r}."
    )
    train_nodes = [ray_nodes_sorted[i] for i in range(train_nnodes)]
    # Use the last num_gen / NUM_GPU_PER_NODE nodes for gen
    gen_nnodes = math.ceil(num_gen / NUM_GPU_PER_NODE)
    assert len(ray_nodes_sorted) >= gen_nnodes, (
        f"Ray has too few generation nodes. required={gen_nnodes!r}, available={len(ray_nodes_sorted)!r}."
    )
    gen_nodes = [ray_nodes_sorted[-i - 1] for i in range(gen_nnodes)]

    # Colocate places train and gen on the *same* GPUs, which is the arrangement every
    # conclusion in design 3.7 is about and the one this harness could not express: with
    # `num_gpus=1` each, Ray guarantees they never share a device, so the timing-exclusion
    # path was unreachable. Taking the first nodes for both and halving the request is
    # what puts a train rank and a gen rank on one card.
    #
    # The fractional request is a *scheduling* statement, not a memory split: each actor
    # still sees the whole device. That is exactly the colocate situation -- two processes
    # on one GPU, which is why the design requires the trainer to offload before rollout
    # allocates its KV cache rather than relying on them fitting together.
    colocate = bool(cfg.test.get("colocate", False))
    colocate_bundles = None
    if colocate:
        assert num_train == num_gen, (
            f"colocate pairs each train rank with a gen rank on one GPU, so the counts "
            f"must match; got num_train={num_train} num_gen={num_gen}"
        )
        gen_nodes = train_nodes
        # A placement group with one bundle per GPU, not a bare fractional request.
        #
        # `num_gpus=0.5` alone lets Ray pack *any* two actors onto a device, and it
        # chose two gen actors: 8 actors landed on 4 devices, so vLLM's TP=2 tried to
        # init two NCCL ranks on one GPU and failed with "invalid usage". The pairing
        # has to be train-with-gen, which means the placement must be expressed rather
        # than left to the scheduler.
        #
        # PACK, not STRICT_SPREAD: the latter demands one bundle per *node*, so asking
        # for 8 bundles on a 5-node cluster never becomes ready and the run dies on a
        # placement-group timeout rather than anything to do with colocate. PACK keeps
        # the bundles together, and one `GPU: 1` bundle per pair is what reserves a whole
        # card for exactly one train rank and one gen rank.
        from ray.util.placement_group import placement_group

        colocate_bundles = placement_group(
            [{"GPU": 1, "CPU": 2} for _ in range(num_train)],
            strategy="PACK",
        )
        ray.get(colocate_bundles.ready(), timeout=300)
    gpus_per_actor = 0.5 if colocate else 1

    def _placement(index: int, node_id: str):
        """Where one actor goes: a colocate bundle, or a node."""
        if colocate_bundles is not None:
            from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

            return PlacementGroupSchedulingStrategy(
                placement_group=colocate_bundles,
                placement_group_bundle_index=index,
            )
        return NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
    # print(f"train_nodes: {train_nodes}, gen_nodes: {gen_nodes}")
    train_engine_type = cfg.test.train_engine_type

    psrl_config = OmegaConf.create(
        {
            "logging_path": log_dir,
            "ps_manager_ip": listen_ip,
            # The whole `nixl` section, not a hand-picked subset. Copying named keys
            # meant a new config option was silently dropped: `peer_diffusion: False`
            # was set in the YAML, never reached the planner, and the run diffused
            # anyway -- with no error, because a missing key just takes its default.
            #
            # Resolved on the way in. The section's values interpolate against siblings
            # (`server_ip: ${network.listen_ip}`), and those siblings do not exist in this
            # smaller config, so copying the node unresolved defers the lookup to a place
            # where it fails: `InterpolationKeyError: network.listen_ip not found`, raised
            # inside the server actor rather than here.
            "nixl": OmegaConf.to_container(cfg.nixl, resolve=True),
            "ps_mode": cfg.ps.mode,
        }
    )

    global_store = GlobalStore.remote()

    start_time = time.time()
    ip_to_node_id = {node["NodeManagerAddress"]: node["NodeID"] for node in ray.nodes()}

    # Create per-node PortScanner actors (required by NIXLStorageClient).
    # The comment above predates the call: every NIXLStorageClient resolves
    # `get_port_scanner(host_ip)` during construction, so without this the first client
    # dies with "Failed to look up actor with name 'psrl_port_scanner_<ip>'". Only alive
    # nodes are passed -- a dead entry would pin a detached actor to a node that cannot
    # run it, and the lookup would hang rather than fail.
    from psrl.utils.nixl.port_scanner import create_port_scanners

    alive_ip_to_node_id = {
        node["NodeManagerAddress"]: node["NodeID"] for node in ray.nodes() if node["Alive"]
    }
    port_scanner_handles = create_port_scanners(alive_ip_to_node_id)
    print(f"[PASS] created {len(port_scanner_handles)} PortScanner actors")

    assert listen_ip in ip_to_node_id, f"listen_ip is absent from Ray nodes: {listen_ip!r}."
    server = MetaServerActor.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=ip_to_node_id[listen_ip], soft=False)
    ).remote(server_name, psrl_config, num_train + num_gen + num_ps, log_dir)
    ray.get(server.init_finished.remote())
    end_time = time.time()
    print(f"[PASS] server init done. time: {end_time - start_time}s")

    start_time = time.time()
    if direct_supply:
        # T2 / D3: no PS at all. The rollout reads the train clients, so there is no
        # intermediate copy to create, register, or wait for. Keeping this as a branch
        # rather than a separate harness is what makes the two arrangements comparable:
        # everything after this point is identical except which names the pull targets.
        ps_wg = None
        ps_agent_names = []
        ps_for_push_names = []
        ps_for_pull_names = []
        ps_for_push_worker_handles = {}
        print("[PASS] direct supply: no PS workers created")
    else:
        ps_wg = create_ps_worker_group(train_engine_type, num_ps, psrl_config, cfg.model)
        ray.get(ps_wg.execute_all_async("init_model"))
        ray.get(ps_wg.execute_all_async("init_nixl_client"))
        ps_agent_names = ray.get(ps_wg.execute_all_async("get_nixl_agent_name"))
        ps_for_push_names = ray.get(ps_wg.execute_all_async("get_nixl_train_storage_client_name"))
        ps_for_pull_names = ray.get(ps_wg.execute_all_async("get_nixl_gen_storage_client_name"))
        print(f"ps_for_push_names: {ps_for_push_names}")
        print(f"ps_for_pull_names: {ps_for_pull_names}")
        ps_for_push_worker_handles = {}
        for ps_for_push_name in ps_for_push_names:
            ps_for_push_worker_handles[ps_for_push_name] = ps_wg.distinguish_worker_by_method(
                lambda worker, name=ps_for_push_name: ray.get(worker.get_nixl_train_storage_client_name.remote())
                == name
            )
        end_time = time.time()
        print(f"[PASS] ps init done. time: {end_time - start_time}s")

    start_time = time.time()
    # Prepare configuration for train actors
    fsdp_hybrid_config = cfg.test.fsdp_hybrid if train_engine_type == "fsdp_hybrid" else None
    megatron_config = cfg.test.megatron if train_engine_type == "megatron" else None

    train_actors = [
        TrainClientActor.options(
            num_gpus=gpus_per_actor,
            scheduling_strategy=_placement(
                rank, train_nodes[rank // NUM_GPU_PER_NODE]["NodeID"]
            ),
        ).remote(
            global_store,
            train_engine_type,
            rank,
            num_train,
            server_name,
            psrl_config,
            backend,
            torch_port_train,
            log_dir,
            ps_for_push_worker_handles,
            cfg.model.path,
            cfg.model,
            fsdp_hybrid_config,
            megatron_config,
        )
        for rank in range(num_train)
    ]
    gen_actors = [
        GenClientActor.options(
            num_gpus=gpus_per_actor,
            scheduling_strategy=_placement(
                rank, gen_nodes[rank // NUM_GPU_PER_NODE]["NodeID"]
            ),
        ).remote(
            global_store,
            rank,
            num_gen,
            server_name,
            psrl_config,
            backend,
            torch_port_gen,
            log_dir,
            cfg.model.path,
            cfg.model,
            cfg.test.gen,
        )
        for rank in range(num_gen)
    ]
    for i in range(num_train):
        ray.get(train_actors[i].init_finished.remote())
    for i in range(num_gen):
        ray.get(gen_actors[i].init_finished.remote())
    end_time = time.time()
    print(f"[PASS] client init done. time: {end_time - start_time}s")

    start_time = time.time()
    # Run protocol for server, clients, and PS workers
    server.protocol.remote()
    futures = []
    for t in train_actors:
        futures.append(t.protocol.remote(cfg.model.path))
    for g in gen_actors:
        futures.append(g.protocol.remote(cfg.model.path))
    if ps_wg is not None:
        futures.extend(ps_wg.execute_all_async("nixl_protocol"))
    ray.get(futures)
    end_time = time.time()
    print(f"[PASS] protocol done. time: {end_time - start_time}s")

    # Warm-up
    """
    futures = []
    for g in gen_actors:
        futures.append(g.pull_from_ps.remote(ps_agent_names, ps_for_pull_names))
    ray.get(futures)
    end_time = time.time()
    print(f"[PASS] Warm-up done. time: {end_time - start_time}s")
    """

    # Each train client pushes to each PS worker. Under direct supply there is no
    # push at all -- that the step disappears rather than becoming a no-op is the
    # point of D3: the intermediate copy is gone, not merely skipped.
    if not direct_supply:
        start_time = time.time()
        futures = []
        for t in train_actors:
            futures.append(t.push_to_ps.remote(ps_agent_names, ps_for_push_names))
        ray.get(futures)
        end_time = time.time()
        print(f"[PASS] train push to all ps done. time: {end_time - start_time}s")

    # Each gen client pulls from each PS worker
    """
    for gen_idx, g in enumerate(gen_actors):
        start_time = time.time()
        ray.get(g.pull_from_ps.remote(ps_agent_names, ps_for_pull_names))
        end_time = time.time()
        print(f"[PASS] gen {gen_idx} pull from all ps done. time: {end_time - start_time}s")
    """
    if direct_supply:
        # The pull takes an (agent, client) pair either way, so direct supply needs a
        # different pair rather than a different read path.
        train_names = ray.get([t.get_nixl_names.remote() for t in train_actors])
        source_agent_names = [agent for agent, _client in train_names]
        source_client_names = [client for _agent, client in train_names]
        label = "train clients"
    else:
        source_agent_names = ps_agent_names
        source_client_names = ps_for_pull_names
        label = "all ps"

    start_time = time.time()
    if direct_supply:
        # D1: two phases with the driver's own `ray.get` as the barrier between them.
        # A peer read is only valid once that peer has completed its upstream read, and
        # reading early *succeeds* against an unfilled buffer rather than failing -- so
        # the ordering is a correctness requirement, not a scheduling preference.
        # Peer reads need the designate's chunk to have landed first, so each wave is
        # separated by a barrier. Driving the loop here keeps the barrier where it has
        # to be while letting wave i+1's upstream read overlap wave i's peer read --
        # which is the whole difference from the two-phase arrangement that measured
        # slower than not diffusing at all.
        gen_names = ray.get([g.get_nixl_names.remote() for g in gen_actors])
        all_agent_names = source_agent_names + [a for a, _c in gen_names]
        all_client_names = source_client_names + [c for _a, c in gen_names]

        # The chunk pipeline is driven by per-chunk completion signalling
        # (`chunk_notify.py`), not by a driver barrier. The difference matters:
        #
        # A shared wave counter cannot express this dependency. Each client's wave list
        # comes from *its own* plan, so a designate with fewer chunks than a follower
        # expects runs out of upstream work while followers are still peer-reading chunks
        # it was supposed to be filling -- the counter advances for everyone regardless.
        # Three cluster runs failed that way, each caught by the coordinate judge: the
        # ordering looks right per client and is wrong globally.
        #
        # A driver barrier per wave would be correct but self-defeating: it is global, so
        # every client waits for the slowest at every chunk. That is what made the
        # two-phase version cost more than not diffusing at all.
        #
        # So the gate is per-chunk and per-source, carried over NIXL notifications
        # designate -> follower. One call per actor, no barrier in between.
        results = ray.get(
            [
                g.pull_pipelined.remote(all_agent_names, all_client_names)
                for g in gen_actors
            ]
        )
        end_time = time.time()
        total_published = sum(r["published"] for r in results)
        total_waits = sum(r["waited_on"] for r in results)
        print(
            f"[PASS] gen pipelined pull done. time: {end_time - start_time}s "
            f"({total_published} notifs published, {total_waits} source-waits across "
            f"{len(results)} clients)"
        )
        # A run where nobody waited is a run with no pipeline in it -- the two-phase
        # fallback and a broken channel both look like this, so it must not pass quietly.
        # With diffusion enabled, a run where nobody waited has no pipeline in it -- the
        # two-phase fallback and a broken channel both look like this, so it must not
        # pass quietly. With diffusion disabled there is deliberately nothing to wait on,
        # and the plan says so, so the assertion follows the plan rather than a constant.
        diffusion_planned = cfg.nixl.get("peer_diffusion", True)
        if diffusion_planned:
            assert total_waits > 0, (
                "no client waited on a peer chunk: diffusion was planned but the pipeline "
                "did not run"
            )
        else:
            assert total_waits == 0, (
                f"{total_waits} peer-chunk waits with diffusion disabled: a staged shard "
                f"has no remote descriptor, so these reads cannot be valid"
            )
        assert total_published == total_waits, (
            f"notification graph is asymmetric: {total_published} published against "
            f"{total_waits} waited on"
        )
        print(f"[PASS] gen pull from {label} done. time: {end_time - start_time}s")

        # Baseline in the *same* run: every client reads everything upstream, no
        # diffusion and no pipeline. Comparing against a number from an earlier run is
        # not comparing like with like -- cluster state, model load order and NIC
        # contention all move between runs, and the pipeline's whole claim is a
        # difference of a few seconds. Measuring both here makes the difference
        # attributable to the code rather than to the day.
        #
        # Runs second so it cannot be credited with warming anything the pipeline then
        # benefits from. It rereads bytes already in place, which is wasteful but
        # harmless: the same shards, from the same sources, into the same buffers.
        if os.environ.get("PSRL_E2E_BASELINE", "1") == "1":
            baseline_start = time.time()
            ray.get(
                [
                    g.pull_from_ps.remote(source_agent_names, source_client_names, phase="all")
                    for g in gen_actors
                ]
            )
            baseline_end = time.time()
            pipelined_seconds = end_time - start_time
            baseline_seconds = baseline_end - baseline_start
            print(
                f"[MEASURE] pipelined {pipelined_seconds:.3f}s vs no-diffusion baseline "
                f"{baseline_seconds:.3f}s "
                f"(ratio {pipelined_seconds / baseline_seconds:.3f})"
            )
    else:
        futures = []
        for g in gen_actors:
            futures.append(g.pull_from_ps.remote(source_agent_names, source_client_names))
        ray.get(futures)
        end_time = time.time()
        print(f"[PASS] gen pull from {label} done. time: {end_time - start_time}s")

    # Fetch and verify gen client state_dicts against the coordinate function.
    print("[CHECK] Verifying GenClientActor state_dicts encode their global coordinates...")
    # Merge offsets from every train rank: under TP/PP each rank holds only part of the
    # key space, so one rank's map would leave most gen keys unjudged. Ranks that do
    # share a key agree on its offset (the fill is driven by the canonical key order),
    # so a plain merge is well defined.
    per_rank_offsets = ray.get([t.get_coord_offsets.remote() for t in train_actors])
    coord_offsets: dict[str, int] = {}
    for rank_offsets in per_rank_offsets:
        coord_offsets.update(rank_offsets)
    print(f"[CHECK] train side filled {len(coord_offsets)} distinct keys")

    gen_state_dicts = ray.get([g.get_state_dict.remote() for g in gen_actors])
    for gen_idx, state_dict in enumerate(gen_state_dicts):
        judged, total = assert_state_dict_matches_coords(
            state_dict, coord_offsets, f"Gen client {gen_idx}"
        )
        print(f"[PASS] Gen client {gen_idx}: {judged}/{total} keys match their coordinates.")
    print("[PASS] All GenClientActor parameters match the coordinate function.")

    # Second round: a new version must actually arrive (item D4).
    #
    # One round cannot distinguish "the transfer works" from "the transfer worked once".
    # The failure this guards is silent: a steady-state switch-back that re-reads the
    # checkpoint, or a rollout that keeps serving a cached version, both produce *valid*
    # weights -- merely the wrong ones -- so nothing raises and the only visible symptom
    # is a loss curve that stops moving.
    #
    # The train side refills at a shifted coordinate base, which is what makes round two
    # a real test: re-filling at the same base would reproduce identical bytes, and a
    # rollout that received nothing would pass exactly as one that received the update.
    if cfg.test.get("rounds", 1) > 1:
        print("[ROUND 2] Advancing the train-side version...")
        version_shift = 1_000_000
        per_rank_offsets_v2 = ray.get(
            [t.advance_version.remote(version_shift) for t in train_actors]
        )
        coord_offsets_v2: dict[str, int] = {}
        for rank_offsets in per_rank_offsets_v2:
            coord_offsets_v2.update(rank_offsets)

        # The shift must have changed something, or the round proves nothing.
        assert coord_offsets_v2 != coord_offsets, (
            "advancing the version left the coordinate offsets unchanged, so round two "
            "cannot tell a fresh pull from a stale one"
        )

        round2_start = time.time()
        if direct_supply:
            ray.get(
                [
                    g.pull_pipelined.remote(all_agent_names, all_client_names)
                    for g in gen_actors
                ]
            )
        else:
            ray.get(
                [
                    g.pull_from_ps.remote(source_agent_names, source_client_names)
                    for g in gen_actors
                ]
            )
        print(f"[PASS] round 2 pull done. time: {time.time() - round2_start}s")

        gen_state_dicts_v2 = ray.get([g.get_state_dict.remote() for g in gen_actors])
        for gen_idx, state_dict in enumerate(gen_state_dicts_v2):
            judged, total = assert_state_dict_matches_coords(
                state_dict, coord_offsets_v2, f"Gen client {gen_idx} (round 2)"
            )
            print(
                f"[PASS] Gen client {gen_idx}: {judged}/{total} keys match the "
                f"*new* version's coordinates."
            )
        print("[PASS] Round 2: every rollout picked up the new version, not a cached one.")

    # Shutdown all actors and Ray
    for t in train_actors:
        ray.get(t.shutdown.remote())
    for g in gen_actors:
        ray.get(g.shutdown.remote())
    if ps_wg is not None:
        ray.get(ps_wg.execute_all_async("shutdown"))
    ray.get(server.shutdown.remote())
    ray.shutdown()


if __name__ == "__main__":
    test_nixl_e2e()
