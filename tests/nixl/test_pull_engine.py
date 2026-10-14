"""
The pull engine moves the right bytes for every train and gen layout, with or without peers.

Real clients register real tensors and the real planner builds the plan. Only the transport
is replaced by host memcpy (`memcpy_agent.py`), so every descriptor the engine posts is
executed and the layout judge checks the result element by element.
"""

from __future__ import annotations

import copy
import ctypes
import threading
from collections import OrderedDict

import pytest
import torch
from omegaconf import OmegaConf
from psrl.utils.common.worker_naming import gen_client_name, model_role_of, train_client_name
from psrl.utils.nixl.block_fp8 import scale_shape
from psrl.utils.nixl.client import NIXLStorageClient
from psrl.utils.nixl.comm_plan import CommunicationPlanner
from psrl.utils.nixl.handshake_notif import NotifKind
from psrl.utils.nixl.nixl_spec import NIXLClientType, NIXLSharding
from psrl.utils.nixl.pull_engine import ChunkUnavailable, NIXLPullEngine
from psrl.utils.nixl.server import NIXLMetaServer

from tests.nixl.coord_judge import check_by_layout, check_fp8_exact, fill_by_layout, fill_fp8_exact
from tests.nixl.memcpy_agent import MemcpyAgent, NotifBus

pytestmark = pytest.mark.cpu_test

HIDDEN, INTER = 128, 256
# key -> (global shape, dim cut by tensor parallelism or None)
MODEL = {
    "layers.0.q_proj.weight": ((INTER, HIDDEN), 0),
    "layers.0.o_proj.weight": ((HIDDEN, INTER), 1),
    "layers.0.down_proj.weight": ((HIDDEN, INTER), 1),
    "layers.0.norm.weight": ((HIDDEN,), None),
}
CONFIG = OmegaConf.create(
    {
        "server_ip": "127.0.0.1",
        "server_port": 0,
        "max_pinned_temp_memory_slots": None,
        "enable_tms_for_temp_buffers": False,
    }
)


def _sharding(dim, tp, rank) -> NIXLSharding:
    if dim is None:
        return NIXLSharding.default()
    return NIXLSharding(shard_mesh=OrderedDict([(dim, tp)]), shard_indices=[(rank,)])


# Keeps the arenas behind `_locals` alive for the life of the test process.
_ARENAS: list[torch.Tensor] = []


def _locals(model: dict, tp: int, dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
    """
    Carve local tensors back to back from one buffer, each over its own storage like a caching allocator.

    Adjacent storages are what tempts a coalescer to merge a descriptor across two registrations.
    """
    shapes = {}
    for key, (shape, dim) in model.items():
        shape = list(shape)
        if dim is not None:
            shape[dim] //= tp
        shapes[key] = shape
    arena = torch.zeros(sum(torch.Size(s).numel() for s in shapes.values()), dtype=dtype)
    itemsize = arena.element_size()
    _ARENAS.append(arena)
    out, offset = {}, 0
    for key, shape in shapes.items():
        n = torch.Size(shape).numel()
        window = (ctypes.c_char * (n * itemsize)).from_address(arena[offset:].data_ptr())
        out[key] = torch.frombuffer(window, dtype=dtype).view(shape)
        offset += n
    return out


class Cluster:
    def __init__(self, train_tp: int, gen_tp: int, instances: int, peer_diffusion: bool, train_dtype=torch.bfloat16):
        self.bus = NotifBus()
        self.declared: dict[str, dict] = {}
        self.states: dict[str, dict] = {}
        kinds = {}
        for r in range(train_tp):
            name = f"NIXLTrainClient_{r}"
            kinds[name] = NIXLClientType.PUSH_SIDE
            self.states[name] = _locals(MODEL, train_tp, train_dtype)
            self.declared[name] = {k: _sharding(d, train_tp, r) for k, (s, d) in MODEL.items()}
        for i in range(instances):
            for r in range(gen_tp):
                name = f"NIXLGenClient_I{i}_R{r}"
                kinds[name] = NIXLClientType.PULL_SIDE
                self.states[name] = _locals(MODEL, gen_tp)
                self.declared[name] = {k: _sharding(d, gen_tp, r) for k, (s, d) in MODEL.items()}
        for name in self.states:
            if name.startswith("NIXLTrainClient"):
                fill_by_layout(self.states[name], self.declared[name], version=7, wire_dtype=torch.bfloat16)

        unified = {name: copy.deepcopy(d) for name, d in self.declared.items()}
        for key in MODEL:
            finest = NIXLSharding.find_finest_shard_mesh([unified[n][key].shard_mesh for n in unified])
            for name in unified:
                unified[name][key].refactor_based_on_finer_shard_mesh(finest)

        self.clients: dict[str, NIXLStorageClient] = {}
        for name, kind in kinds.items():
            client = NIXLStorageClient(
                client_name=name,
                server_name="server",
                use_gpu=False,
                client_type=kind,
                nixl_config=CONFIG,
                binded_agent=MemcpyAgent(name, self.bus),
            )
            if kind is NIXLClientType.PUSH_SIDE:
                client.set_publish_cast(train_dtype, torch.bfloat16)
            client.register_local_tensors(self.states[name], unified[name])
            self.clients[name] = client
        infos = {name: c.local_client_info for name, c in self.clients.items()}
        plan = CommunicationPlanner(peer_diffusion=peer_diffusion).make_comm_plan(infos)
        for client in self.clients.values():
            client._all_client_infos = dict(infos)
            client._comm_plan = plan
        self.plan = plan
        self.gen = [n for n in self.clients if n.startswith("NIXLGenClient")]
        self.engines = {
            n: NIXLPullEngine(self.clients[n], staging_bytes=1 << 16, peer_timeout_s=5.0) for n in self.gen
        }

    def pull(self, epoch: int, participants: set[str] | None = None) -> dict:
        names = sorted(participants) if participants is not None else self.gen
        reports, errors = {}, []

        def run(name):
            try:
                reports[name] = self.engines[name].pull(epoch, participants)
            except Exception as exc:  # surfaced below
                errors.append((name, exc))

        threads = [threading.Thread(target=run, args=(n,)) for n in names]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, errors
        return reports

    def judge(self, names, version: int) -> None:
        for name in names:
            judged, failure = check_by_layout(self.states[name], self.declared[name], list(MODEL), version=version)
            assert failure is None, f"{name}: {failure}"
            assert judged == len(MODEL)


@pytest.mark.parametrize("train_tp,gen_tp", [(2, 2), (4, 2), (2, 4), (1, 4), (4, 1)])
def test_every_layout_pulls_exact_bytes_without_peers(train_tp, gen_tp):
    cluster = Cluster(train_tp, gen_tp, instances=1, peer_diffusion=False)
    reports = cluster.pull(epoch=1)
    cluster.judge(cluster.gen, version=7)
    for report in reports.values():
        assert report.peer_keys == 0 and report.versions == {-1}


def test_fragmented_sources_get_publish_copies_and_readers_one_descriptor_per_shard():
    cluster = Cluster(train_tp=2, gen_tp=4, instances=1, peer_diffusion=False)
    assert all(cluster.clients[f"NIXLTrainClient_{r}"].publish_copy_bytes > 0 for r in range(2))
    reports = cluster.pull(epoch=1)
    cluster.judge(cluster.gen, version=7)
    # q is split on dim 0 on both sides, o and down come from contiguous copies, norm is one run.
    for report in reports.values():
        assert report.descriptors <= len(MODEL), report


def test_fragmented_readers_stage_through_the_pool():
    cluster = Cluster(train_tp=4, gen_tp=1, instances=1, peer_diffusion=False)
    report = cluster.pull(epoch=1)["NIXLGenClient_I0_R0"]
    cluster.judge(cluster.gen, version=7)
    assert report.staged_bytes > 0 and report.descriptors < 64


@pytest.mark.parametrize("train_tp,gen_tp", [(4, 2), (2, 4)])
def test_peers_read_their_designate_and_trainer_egress_is_one_copy_per_group(train_tp, gen_tp):
    cluster = Cluster(train_tp, gen_tp, instances=3, peer_diffusion=True)
    reports = cluster.pull(epoch=1)
    cluster.judge(cluster.gen, version=7)
    upstream = sum(r.upstream_bytes for r in reports.values())
    model_bytes = sum(torch.Size(s).numel() * 2 for s, _ in MODEL.values())
    # Each byte leaves the trainer once per node, whichever instance happens to be its designate.
    assert upstream == model_bytes, (upstream, model_bytes)
    # Upstream work is spread over the instances rather than concentrated on one.
    assert sum(1 for r in reports.values() if r.upstream_keys) > gen_tp
    for report in reports.values():
        assert report.fallback_keys == 0
        assert report.peer_keys == 0 or report.peer_bytes > 0


def test_peer_reads_are_one_descriptor_per_key():
    cluster = Cluster(train_tp=4, gen_tp=2, instances=2, peer_diffusion=True)
    reports = cluster.pull(epoch=1)
    for name, report in reports.items():
        if report.peer_keys:
            peer_only = (
                report.descriptors
                - cluster.engines[name]
                ._phase(
                    cluster.engines[name]._upstream_specs(
                        {
                            k: v
                            for k, v in cluster.plan.rollout_pull_from_ps_plan[name].items()
                            if cluster.plan.get_peer_group(name, k)[0] == name
                        }
                    )
                )
                .descriptors
            )
            assert peer_only == report.peer_keys, (name, report)


def test_a_subset_of_participants_diffuses_among_itself_only():
    cluster = Cluster(train_tp=4, gen_tp=2, instances=3, peer_diffusion=True)
    subset = {"NIXLGenClient_I1_R0", "NIXLGenClient_I1_R1", "NIXLGenClient_I2_R0", "NIXLGenClient_I2_R1"}
    cluster.pull(epoch=1, participants=subset)
    cluster.judge(sorted(subset), version=7)
    # The instance that did not sync was never read into.
    assert all(int(t.abs().sum()) == 0 for t in cluster.states["NIXLGenClient_I0_R0"].values())


def test_a_second_version_replaces_every_byte_and_stale_signals_do_not_leak():
    cluster = Cluster(train_tp=4, gen_tp=2, instances=2, peer_diffusion=True)
    cluster.pull(epoch=1)
    for r in range(4):
        name = f"NIXLTrainClient_{r}"
        fill_by_layout(cluster.states[name], cluster.declared[name], version=8, wire_dtype=torch.bfloat16)
        cluster.clients[name].publish_version(8)
    cluster.pull(epoch=2)
    cluster.judge(cluster.gen, version=8)


def test_a_missing_designate_falls_back_to_upstream():
    cluster = Cluster(train_tp=4, gen_tp=2, instances=2, peer_diffusion=True)
    for engine in cluster.engines.values():
        engine.peer_timeout_s = 0.5
    # Claim both instances participate but run only one, so its designates never answer.
    everyone = set(cluster.gen)
    lone = ["NIXLGenClient_I1_R0", "NIXLGenClient_I1_R1"]
    reports = {}
    threads = [
        threading.Thread(target=lambda n=n: reports.__setitem__(n, cluster.engines[n].pull(1, everyone))) for n in lone
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    cluster.judge(lone, version=7)
    assert sum(r.fallback_keys for r in reports.values()) > 0


def test_descriptors_never_span_two_registered_allocations():
    """Back to back storages on both sides must still be read as separate descriptors."""
    cluster = Cluster(train_tp=2, gen_tp=2, instances=1, peer_diffusion=False)
    reports = cluster.pull(epoch=1)
    cluster.judge(cluster.gen, version=7)
    for report in reports.values():
        assert report.descriptors == len(MODEL), report


@pytest.mark.parametrize("train_tp,gen_tp", [(2, 2), (2, 4), (4, 2)])
def test_a_float32_trainer_publishes_bfloat16_copies(train_tp, gen_tp):
    cluster = Cluster(train_tp, gen_tp, instances=2, peer_diffusion=True, train_dtype=torch.float32)
    for r in range(train_tp):
        client = cluster.clients[f"NIXLTrainClient_{r}"]
        assert client.publish_copy_bytes == sum(t.numel() * 2 for t in cluster.states[client.client_name].values())
    reports = cluster.pull(epoch=1)
    cluster.judge(cluster.gen, version=7)
    for report in reports.values():
        assert report.descriptors <= 2 * len(MODEL), report


def test_past_the_copy_budget_readers_span_read_in_place():
    CONFIG.publish_copy_budget_mb = 0
    try:
        cluster = Cluster(train_tp=2, gen_tp=4, instances=1, peer_diffusion=False)
    finally:
        del CONFIG["publish_copy_budget_mb"]
    assert all(cluster.clients[f"NIXLTrainClient_{r}"].publish_copy_bytes == 0 for r in range(2))
    reports = cluster.pull(epoch=1)
    cluster.judge(cluster.gen, version=7)
    for report in reports.values():
        assert report.staged_bytes > 0 and report.descriptors <= 2 * len(MODEL), report


def test_the_pull_path_makes_no_control_plane_calls():
    """Source selection is a lookup in the plan received at the handshake, never a Ray call per pull."""
    import inspect

    from psrl.utils.nixl import pull_engine
    from psrl.utils.nixl.client import NIXLStorageClient

    source = inspect.getsource(pull_engine) + inspect.getsource(NIXLStorageClient.client_read)
    assert ".remote(" not in source and "ray.get" not in source
    assert "comm_plan" in inspect.getsource(NIXLStorageClient.wait_for_server_info)


BLOCK = (32, 32)
FP8_KEYS = ["layers.0.q_proj.weight", "layers.0.o_proj.weight", "layers.0.down_proj.weight"]


def _fp8_cluster(
    train_tp, gen_tp, instances, peer_diffusion, train_dtype=torch.bfloat16, scale_tile=BLOCK, train_fp8=False
):
    """Build a cluster whose rollouts are FP8, reusing `Cluster` with the gen state swapped before registration."""
    cluster = Cluster.__new__(Cluster)
    cluster.bus = NotifBus()
    cluster.declared, cluster.states = {}, {}
    kinds = {}
    for r in range(train_tp):
        name = f"NIXLTrainClient_{r}"
        kinds[name] = NIXLClientType.PUSH_SIDE
        cluster.states[name] = _locals(MODEL, train_tp, train_dtype)
        cluster.declared[name] = {k: _sharding(d, train_tp, r) for k, (s, d) in MODEL.items()}
    for i in range(instances):
        for r in range(gen_tp):
            name = f"NIXLGenClient_I{i}_R{r}"
            kinds[name] = NIXLClientType.PULL_SIDE
            state = _locals(MODEL, gen_tp)
            declared = {k: _sharding(d, gen_tp, r) for k, (s, d) in MODEL.items()}
            for key in FP8_KEYS:
                state[key] = torch.zeros(state[key].shape, dtype=torch.float8_e4m3fn)
                state[key + "_scale_inv"] = torch.zeros(scale_shape(state[key].shape, scale_tile), dtype=torch.float32)
                declared[key + "_scale_inv"] = copy.deepcopy(declared[key])
                declared[key + "_scale_inv"].scale_tile = scale_tile
            cluster.states[name], cluster.declared[name] = state, declared

    def tile_of(t):
        return (1,) * (t.dim() - 2) + BLOCK

    for r in range(train_tp):
        name = f"NIXLTrainClient_{r}"
        fill_fp8_exact(cluster.states[name], cluster.declared[name], FP8_KEYS, tile_of, version=7, uniform=train_fp8)
        rest = {k: v for k, v in cluster.states[name].items() if k not in FP8_KEYS}
        fill_by_layout(rest, cluster.declared[name], version=7, wire_dtype=torch.bfloat16)
        if train_fp8:
            # A Megatron `fp8_param` trainer: per tensor scaled FP8 bytes, as the converter emits them.
            for key in FP8_KEYS:
                values = cluster.states[name][key]
                scale = torch.tensor(2.0**-7)
                fp8 = (values.float() / scale).to(torch.float8_e4m3fn)
                fp8.psrl_fp8_scale = (lambda scale=scale: scale, train_dtype)
                cluster.states[name][key] = fp8

    # The server derives the trainer's scale shardings, so run its unification rather than a copy of it.
    from psrl.utils.nixl.server import NIXLMetaServer

    server = NIXLMetaServer.__new__(NIXLMetaServer)
    server.client_sharding_dicts = {n: copy.deepcopy(d) for n, d in cluster.declared.items()}
    server.client_unified_sharding_dicts, server.finest_shard_meshes = {}, {}
    from psrl.utils.nixl.handshake_notif import NotifKind

    server._agents_reported = {kind: set() for kind in NotifKind}
    server._agents_reported[NotifKind.SHARDING] = set(cluster.declared)
    server.make_unified_sharding()
    unified = server.client_unified_sharding_dicts

    cluster.clients = {}
    for name, kind in kinds.items():
        client = NIXLStorageClient(
            client_name=name,
            server_name="server",
            use_gpu=False,
            client_type=kind,
            nixl_config=CONFIG,
            binded_agent=MemcpyAgent(name, cluster.bus),
        )
        if kind is NIXLClientType.PUSH_SIDE:
            client.set_publish_cast(train_dtype, torch.bfloat16)
            client.set_publish_fp8(BLOCK)
        client.register_local_tensors(cluster.states[name], unified[name])
        cluster.clients[name] = client
    infos = {name: c.local_client_info for name, c in cluster.clients.items()}
    plan = CommunicationPlanner(peer_diffusion=peer_diffusion).make_comm_plan(infos)
    for client in cluster.clients.values():
        client._all_client_infos = dict(infos)
        client._comm_plan = plan
    cluster.plan = plan
    cluster.gen = [n for n in cluster.clients if n.startswith("NIXLGenClient")]
    cluster.engines = {
        n: NIXLPullEngine(cluster.clients[n], staging_bytes=1 << 16, peer_timeout_s=5.0) for n in cluster.gen
    }
    return cluster


def _judge_fp8(cluster, version, uniform=False):
    for name in cluster.gen:
        judged, failure = check_fp8_exact(
            cluster.states[name], cluster.declared[name], FP8_KEYS, BLOCK, version, uniform=uniform
        )
        assert failure is None, f"{name}: {failure}"
        assert judged == len(FP8_KEYS)
        rest = [k for k in MODEL if k not in FP8_KEYS]
        judged, failure = check_by_layout(cluster.states[name], cluster.declared[name], rest, version=version)
        assert failure is None and judged == len(rest), (name, failure)


@pytest.mark.parametrize("train_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("train_tp,gen_tp", [(2, 2), (4, 2), (2, 4), (1, 2)])
def test_a_trainer_publishes_block_fp8_with_scales(train_dtype, train_tp, gen_tp):
    cluster = _fp8_cluster(train_tp, gen_tp, instances=2, peer_diffusion=True, train_dtype=train_dtype)
    cluster.pull(epoch=1)
    _judge_fp8(cluster, version=7)


@pytest.mark.parametrize("train_tp,gen_tp", [(2, 2), (2, 16)])
def test_a_finer_rollout_scale_grid_gets_each_tile_scale_repeated(train_tp, gen_tp):
    # vLLM refines an FP8 MoE scale grid when a TP cut splits a tile. At gen TP 16 each rank holds
    # 16 of a 32 wide tile, which only the refined 16 x 16 grid can express.
    cluster = _fp8_cluster(train_tp, gen_tp, instances=2, peer_diffusion=True, scale_tile=(16, 16))
    cluster.pull(epoch=1)
    _judge_fp8(cluster, version=7)


@pytest.mark.parametrize("train_tp,gen_tp", [(2, 2), (4, 2), (1, 4)])
def test_an_fp8_trainer_publishes_block_fp8_from_its_per_tensor_bytes(train_tp, gen_tp):
    cluster = _fp8_cluster(train_tp, gen_tp, instances=2, peer_diffusion=True, train_fp8=True)
    for name in cluster.clients:
        if name.startswith("NIXLTrainClient"):
            assert all(cluster.states[name][k].dtype == torch.float8_e4m3fn for k in FP8_KEYS)
    cluster.pull(epoch=1)
    _judge_fp8(cluster, version=7, uniform=True)


def test_a_new_fp8_version_requantises_at_publish():
    cluster = _fp8_cluster(4, 2, instances=2, peer_diffusion=True)
    cluster.pull(epoch=1)

    def tile_of(t):
        return (1,) * (t.dim() - 2) + BLOCK

    for r in range(4):
        name = f"NIXLTrainClient_{r}"
        fill_fp8_exact(cluster.states[name], cluster.declared[name], FP8_KEYS, tile_of, version=9)
        rest = {k: v for k, v in cluster.states[name].items() if k not in FP8_KEYS}
        fill_by_layout(rest, cluster.declared[name], version=9, wire_dtype=torch.bfloat16)
        cluster.clients[name].publish_version(9)
    cluster.pull(epoch=2)
    _judge_fp8(cluster, version=9)


@pytest.mark.parametrize("waves", [1, 2, 7])
def test_any_wave_count_moves_the_same_bytes(waves):
    cluster = Cluster(train_tp=4, gen_tp=2, instances=3, peer_diffusion=True)
    for engine in cluster.engines.values():
        engine.waves = waves
    reports = cluster.pull(epoch=1)
    cluster.judge(cluster.gen, version=7)
    model_bytes = sum(torch.Size(s).numel() * 2 for s, _ in MODEL.values())
    assert sum(r.upstream_bytes for r in reports.values()) == model_bytes


def _replicated_cluster(train_tp, dp, gen_tp, instances, peer_diffusion=False):
    """A cluster whose train ranks come in `dp` identical replicas, as Megatron's DP groups do."""
    cluster = Cluster.__new__(Cluster)
    cluster.bus = NotifBus()
    cluster.declared, cluster.states = {}, {}
    kinds = {}
    for d in range(dp):
        for r in range(train_tp):
            name = f"NIXLTrainClient_{d * train_tp + r}"
            kinds[name] = NIXLClientType.PUSH_SIDE
            cluster.states[name] = _locals(MODEL, train_tp)
            cluster.declared[name] = {k: _sharding(dim, train_tp, r) for k, (s, dim) in MODEL.items()}
            fill_by_layout(cluster.states[name], cluster.declared[name], version=7, wire_dtype=torch.bfloat16)
    for i in range(instances):
        for r in range(gen_tp):
            name = f"NIXLGenClient_I{i}_R{r}"
            kinds[name] = NIXLClientType.PULL_SIDE
            cluster.states[name] = _locals(MODEL, gen_tp)
            cluster.declared[name] = {k: _sharding(dim, gen_tp, r) for k, (s, dim) in MODEL.items()}

    unified = {name: copy.deepcopy(d) for name, d in cluster.declared.items()}
    for key in MODEL:
        finest = NIXLSharding.find_finest_shard_mesh([unified[n][key].shard_mesh for n in unified])
        for name in unified:
            unified[name][key].refactor_based_on_finer_shard_mesh(finest)

    cluster.clients = {}
    for name, kind in kinds.items():
        client = NIXLStorageClient(
            client_name=name,
            server_name="server",
            use_gpu=False,
            client_type=kind,
            nixl_config=CONFIG,
            binded_agent=MemcpyAgent(name, cluster.bus),
        )
        client.register_local_tensors(cluster.states[name], unified[name])
        cluster.clients[name] = client
    infos = {name: c.local_client_info for name, c in cluster.clients.items()}
    plan = CommunicationPlanner(peer_diffusion=peer_diffusion).make_comm_plan(infos)
    for client in cluster.clients.values():
        client._all_client_infos = dict(infos)
        client._comm_plan = plan
    cluster.plan = plan
    cluster.gen = [n for n in cluster.clients if n.startswith("NIXLGenClient")]
    cluster.engines = {
        n: NIXLPullEngine(cluster.clients[n], staging_bytes=1 << 16, peer_timeout_s=2.0, read_timeout_s=2.0)
        for n in cluster.gen
    }
    return cluster


def _planned_source(cluster, train_tp):
    """A train rank some reader is planned to read from, with its DP twin."""
    for reader in cluster.gen:
        for per_source in cluster.plan.rollout_pull_from_ps_plan[reader].values():
            for source in per_source:
                rank = int(source.rsplit("_", 1)[1])
                twin = f"NIXLTrainClient_{(rank + train_tp) % (2 * train_tp)}"
                if twin in cluster.clients:
                    return source, twin
    raise AssertionError("no reader was planned against any train rank")


def _kill(cluster, sources):
    """Make every reader's transport fail for `sources`, as it does once a train rank is gone."""
    for name in cluster.gen:
        cluster.clients[name].agent.dead_sources.update(sources)


def test_a_known_dead_source_is_replaced_before_the_pull_starts():
    cluster = _replicated_cluster(train_tp=2, dp=2, gen_tp=2, instances=1)
    victim, _twin = _planned_source(cluster, train_tp=2)
    dead = {victim}
    _kill(cluster, dead)
    for engine in cluster.engines.values():
        engine.mark_unavailable(dead)
    reports = cluster.pull(epoch=1)
    cluster.judge(cluster.gen, version=7)
    assert all(dead.isdisjoint(r.sources) for r in reports.values()), "a dead source still served bytes"
    assert sum(r.takeover_keys for r in reports.values()) > 0


def test_a_source_that_fails_mid_pull_is_taken_over_by_its_replica():
    cluster = _replicated_cluster(train_tp=2, dp=2, gen_tp=2, instances=1)
    victim, _twin = _planned_source(cluster, train_tp=2)
    _kill(cluster, {victim})  # not announced: the engine must learn it from the read
    reports = cluster.pull(epoch=1)
    cluster.judge(cluster.gen, version=7)
    # Only the readers whose plan named that rank see it fail; they must all recover from a replica.
    hit = [r for r in reports.values() if r.failed_sources]
    assert hit, "no reader was planned against the failed source"
    assert all(r.failed_sources == {victim} and r.takeover_keys > 0 for r in hit)


def test_losing_a_whole_dp_group_is_refused_before_any_byte_moves():
    cluster = _replicated_cluster(train_tp=2, dp=2, gen_tp=2, instances=1)
    # Both replicas of one rank's shards: nothing holds those keys any more.
    victim, twin = _planned_source(cluster, train_tp=2)
    dead = {victim, twin}
    _kill(cluster, dead)
    for engine in cluster.engines.values():
        engine.mark_unavailable(dead)
    reader = next(
        r
        for r in cluster.gen
        if dead.intersection(src for per in cluster.plan.rollout_pull_from_ps_plan[r].values() for src in per)
    )
    before = {k: v.clone() for k, v in cluster.states[reader].items()}
    with pytest.raises(ChunkUnavailable):
        cluster.engines[reader].pull(epoch=1)
    for key, value in before.items():
        assert torch.equal(value, cluster.states[reader][key]), f"{key} was modified by a refused pull"


def test_a_designate_that_dies_sends_its_followers_upstream():
    cluster = _replicated_cluster(train_tp=2, dp=2, gen_tp=2, instances=2, peer_diffusion=True)
    reports = cluster.pull(epoch=1)
    designates = {src for r in reports.values() for src in r.sources if src.startswith("NIXLGenClient")}
    assert designates, "no peer reads were planned"
    cluster.judge(cluster.gen, version=7)
    _kill(cluster, designates)
    reports = cluster.pull(epoch=2)
    cluster.judge(cluster.gen, version=7)
    assert sum(r.fallback_keys for r in reports.values()) > 0


def _multi_role_cluster(roles, train_tp, gen_tp):
    """One cluster carrying several models at once, their keys named identically."""
    cluster = Cluster.__new__(Cluster)
    cluster.bus = NotifBus()
    cluster.declared, cluster.states, cluster.roles = {}, {}, {}
    kinds = {}
    for role in roles:
        for r in range(train_tp):
            name = train_client_name(r, role)
            kinds[name] = NIXLClientType.PUSH_SIDE
            cluster.states[name] = _locals(MODEL, train_tp)
            cluster.declared[name] = {k: _sharding(d, train_tp, r) for k, (s, d) in MODEL.items()}
            # Each model fills at its own version, so bytes of one are never valid for the other.
            fill_by_layout(
                cluster.states[name], cluster.declared[name], version=roles[role], wire_dtype=torch.bfloat16
            )
        for r in range(gen_tp):
            name = gen_client_name(0, r, role)
            kinds[name] = NIXLClientType.PULL_SIDE
            cluster.states[name] = _locals(MODEL, gen_tp)
            cluster.declared[name] = {k: _sharding(d, gen_tp, r) for k, (s, d) in MODEL.items()}
        cluster.roles[role] = [n for n in kinds if model_role_of(n) == role and n.startswith("NIXLGenClient")]

    # The real server unifies and plans, which is what must keep the models apart.
    server = NIXLMetaServer.__new__(NIXLMetaServer)
    server.client_sharding_dicts = {n: copy.deepcopy(d) for n, d in cluster.declared.items()}
    server.client_unified_sharding_dicts, server.finest_shard_meshes = {}, {}
    server._agents_reported = {kind: set() for kind in NotifKind}
    server._agents_reported[NotifKind.SHARDING] = set(cluster.declared)
    server.make_unified_sharding()
    unified = server.client_unified_sharding_dicts

    cluster.clients = {}
    for name, kind in kinds.items():
        client = NIXLStorageClient(
            client_name=name,
            server_name="server",
            use_gpu=False,
            client_type=kind,
            nixl_config=CONFIG,
            binded_agent=MemcpyAgent(name, cluster.bus),
        )
        client.register_local_tensors(cluster.states[name], unified[name])
        cluster.clients[name] = client
    infos = {name: c.local_client_info for name, c in cluster.clients.items()}
    server.client_infos = infos
    server._agents_reported[NotifKind.CLIENT_INFO] = set(infos)
    server._peer_diffusion = False
    server.comm_plan = None
    server.make_comm_plan()
    for client in cluster.clients.values():
        client._all_client_infos = dict(infos)
        client._comm_plan = server.comm_plan
    cluster.plan = server.comm_plan
    cluster.gen = [n for n in cluster.clients if n.startswith("NIXLGenClient")]
    cluster.engines = {
        n: NIXLPullEngine(cluster.clients[n], staging_bytes=1 << 16, peer_timeout_s=2.0) for n in cluster.gen
    }
    return cluster


def test_two_models_publish_and_pull_without_reading_each_other():
    roles = {"actor": 7, "critic": 11}
    cluster = _multi_role_cluster(roles, train_tp=2, gen_tp=2)
    reports = cluster.pull(epoch=1)
    for role, version in roles.items():
        for name in cluster.roles[role]:
            judged, failure = check_by_layout(cluster.states[name], cluster.declared[name], list(MODEL), version)
            assert failure is None, f"{name} ({role}): {failure}"
            assert judged == len(MODEL)
    # Every byte a reader took came from its own model's ranks.
    for name, report in reports.items():
        role = model_role_of(name)
        assert all(model_role_of(src) == role for src in report.sources), f"{name} read {sorted(report.sources)}"


def test_one_model_republishing_does_not_disturb_the_other():
    roles = {"actor": 7, "critic": 11}
    cluster = _multi_role_cluster(roles, train_tp=2, gen_tp=2)
    cluster.pull(epoch=1)
    for r in range(2):
        name = train_client_name(r, "actor")
        fill_by_layout(cluster.states[name], cluster.declared[name], version=9, wire_dtype=torch.bfloat16)
        cluster.clients[name].publish_version(9)
    cluster.pull(epoch=2)
    for role, version in {"actor": 9, "critic": 11}.items():
        for name in cluster.roles[role]:
            judged, failure = check_by_layout(cluster.states[name], cluster.declared[name], list(MODEL), version)
            assert failure is None, f"{name} ({role}) after actor republished: {failure}"
            assert judged == len(MODEL)


def test_an_instance_that_joins_later_pulls_correct_bytes():
    """A rollout started after the handshake is admitted, replanned for, and reads the right bytes."""
    roles = {"actor": 7}
    cluster = _multi_role_cluster(roles, train_tp=2, gen_tp=2)
    cluster.pull(epoch=1)
    incumbents = list(cluster.gen)

    # A second instance registers after everyone else has been planned for.
    joiners = []
    for r in range(2):
        name = gen_client_name(1, r)
        cluster.states[name] = _locals(MODEL, 2)
        cluster.declared[name] = {k: _sharding(d, 2, r) for k, (s, d) in MODEL.items()}
        joiners.append(name)

    server = NIXLMetaServer.__new__(NIXLMetaServer)
    server.client_sharding_dicts = {n: copy.deepcopy(d) for n, d in cluster.declared.items()}
    server.client_unified_sharding_dicts, server.finest_shard_meshes = {}, {}
    server._agents_reported = {kind: set() for kind in NotifKind}
    server._agents_reported[NotifKind.SHARDING] = set(cluster.declared)
    server.make_unified_sharding()
    unified = server.client_unified_sharding_dicts

    for name in joiners:
        client = NIXLStorageClient(
            client_name=name,
            server_name="server",
            use_gpu=False,
            client_type=NIXLClientType.PULL_SIDE,
            nixl_config=CONFIG,
            binded_agent=MemcpyAgent(name, cluster.bus),
        )
        client.register_local_tensors(cluster.states[name], unified[name])
        cluster.clients[name] = client
    infos = {name: c.local_client_info for name, c in cluster.clients.items()}
    server.client_infos = infos
    server._agents_reported[NotifKind.CLIENT_INFO] = set(infos)
    server._peer_diffusion = False
    server.comm_plan = None
    server.make_comm_plan()
    for client in cluster.clients.values():
        client._all_client_infos = dict(infos)
        client._comm_plan = server.comm_plan
    cluster.gen = incumbents + joiners
    cluster.engines.update(
        {n: NIXLPullEngine(cluster.clients[n], staging_bytes=1 << 16, peer_timeout_s=2.0) for n in joiners}
    )

    # The replan must cover the joiners and leave the incumbents able to pull as before.
    assert all(name in server.comm_plan.rollout_pull_from_ps_plan for name in joiners)
    cluster.pull(epoch=2, participants=set(cluster.gen))
    for name in cluster.gen:
        judged, failure = check_by_layout(cluster.states[name], cluster.declared[name], list(MODEL), version=7)
        assert failure is None, f"{name}: {failure}"
        assert judged == len(MODEL)
