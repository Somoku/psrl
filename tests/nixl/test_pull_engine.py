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
from psrl.utils.nixl.client import NIXLStorageClient
from psrl.utils.nixl.comm_plan import CommunicationPlanner
from psrl.utils.nixl.nixl_spec import NIXLClientType, NIXLSharding
from psrl.utils.nixl.pull_engine import NIXLPullEngine

from tests.nixl.coord_judge import check_by_layout, fill_by_layout
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
