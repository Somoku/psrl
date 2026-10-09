"""Collapsing duplicate reads onto same-node peers (implementation item D1).

Parallelism makes rollout clients on one node want byte-identical shards: with gen TP=2
and 8 clients per node, `tp_rank = rank % 2` gives four clients the same tp_rank, so
four of them read the same bytes from upstream. That fourfold duplication is the `N x M`
this item exists to remove -- one client reads upstream, the other three take it over
NVLink.

The tests measure upstream read counts rather than checking that a peer plan exists,
because a plan that redirected nothing would also "exist".
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
from psrl.utils.nixl.comm_plan import CommunicationPlanner, NIXLClientInfo, NIXLClientType
from psrl.utils.nixl.nixl_spec import NIXLSharding, NIXLShardMetaInfo, NIXLTensorInfo

pytestmark = pytest.mark.cpu_test

KEYS = ["w0", "w1"]


def _tensor_info(shard_count: int, owned: list[int]) -> NIXLTensorInfo:
    sharding = NIXLSharding(
        shard_mesh=OrderedDict([(0, shard_count)]),
        shard_indices=[(i,) for i in sorted(owned)],
    )
    meta = NIXLShardMetaInfo(
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        shape=torch.Size((16, 16)),
        stride=(16, 1),
        is_contiguous=True,
    )
    return NIXLTensorInfo(
        desc_bytes_list=[None] * len(owned),
        temp_desc_bytes_list=[None] * len(owned),
        sharding=sharding,
        shard_meta_infos=[meta] * len(owned),
    )


def _client(name, client_type, node_ip, shard_count, owned):
    return NIXLClientInfo(
        name=name,
        node_ip=node_ip,
        node_gpu_id=0,
        type=client_type,
        tensor_infos={key: _tensor_info(shard_count, owned) for key in KEYS},
        meta=b"",
    )


def _cluster(nodes: int, per_node: int, tp: int):
    """Rollouts whose tp_rank repeats within a node, plus trainers holding every shard."""
    clients = {
        f"train{i}": _client(f"train{i}", NIXLClientType.PUSH_SIDE, f"10.0.0.{i}", tp, list(range(tp)))
        for i in range(2)
    }
    for node in range(nodes):
        for slot in range(per_node):
            rank = node * per_node + slot
            clients[f"gen{rank}"] = _client(
                f"gen{rank}", NIXLClientType.PULL_SIDE, f"10.1.0.{node}", tp, [rank % tp]
            )
    return clients


def _upstream_reads(plan, clients) -> int:
    """Upstream reads a diffusion-capable caller would issue.

    Counted as "upstream reads the plan does not cover by a peer", not as everything
    `iter_rollout_pull_work` yields. The iterator deliberately keeps the upstream read
    whole so a caller lacking a barrier can skip peer reads and stay correct, so the
    saving does not show up there -- it shows up in what a caller *with* a barrier is
    allowed to drop.
    """
    total = 0
    for name, info in clients.items():
        if info.type is not NIXLClientType.PULL_SIDE:
            continue
        for key in KEYS:
            covered_by_peer = {
                shard
                for shard_list in plan.get_peer_pull_plan(name, key).values()
                for shard in shard_list
            }
            for shard_list in plan.get_rollout_pull_plan(name, key).values():
                total += sum(1 for shard in shard_list if shard not in covered_by_peer)
    return total


def _peer_reads(plan, clients) -> int:
    return sum(
        len(shard_list)
        for key_plans in plan.peer_pull_plan.values()
        for per_peer in key_plans.values()
        for shard_list in per_peer.values()
    )


class TestDuplicationIsRemoved:
    def test_one_upstream_read_per_node_per_tp_rank(self):
        """The measurable claim: 8 clients per node, TP=2 -> 2 upstream reads, not 8."""
        clients = _cluster(nodes=2, per_node=8, tp=2)
        plan = CommunicationPlanner().make_comm_plan(clients)

        upstream = _upstream_reads(plan, clients)
        expected = 2 * 2 * len(KEYS)  # nodes x tp_ranks x keys
        assert upstream == expected, (
            f"{upstream} upstream reads; one per (node, tp_rank, key) would be {expected}"
        )

    def test_the_followers_read_from_a_peer(self):
        clients = _cluster(nodes=1, per_node=4, tp=2)
        plan = CommunicationPlanner().make_comm_plan(clients)

        assert _peer_reads(plan, clients) > 0

    def test_upstream_reads_stay_flat_as_clients_per_node_grow(self):
        """Without diffusion this scales with client count; with it, it does not."""
        counts = {}
        for per_node in (2, 4, 8):
            clients = _cluster(nodes=2, per_node=per_node, tp=2)
            plan = CommunicationPlanner().make_comm_plan(clients)
            counts[per_node] = _upstream_reads(plan, clients)

        assert len(set(counts.values())) == 1, f"upstream reads moved with client count: {counts}"

    def test_nothing_is_redirected_when_no_two_clients_share_bytes(self):
        """TP == clients per node means every client wants a different shard."""
        clients = _cluster(nodes=1, per_node=4, tp=4)
        plan = CommunicationPlanner().make_comm_plan(clients)

        assert plan.peer_pull_plan == {}
        assert _peer_reads(plan, clients) == 0


class TestNoChainsAndNoCrossNode:
    def test_a_peer_is_always_on_the_same_node(self):
        """A cross-node peer costs the same NIC as upstream, so it buys nothing."""
        clients = _cluster(nodes=3, per_node=4, tp=2)
        plan = CommunicationPlanner().make_comm_plan(clients)

        for follower, key_plans in plan.peer_pull_plan.items():
            for per_peer in key_plans.values():
                for peer in per_peer:
                    assert clients[peer].node_ip == clients[follower].node_ip, (
                        f"{follower} was pointed at off-node peer {peer}"
                    )

    def test_a_designated_reader_is_never_itself_redirected(self):
        """This is what makes the depth one by construction rather than by convention."""
        clients = _cluster(nodes=2, per_node=8, tp=2)
        plan = CommunicationPlanner().make_comm_plan(clients)

        designates = {
            peer
            for key_plans in plan.peer_pull_plan.values()
            for per_peer in key_plans.values()
            for peer in per_peer
        }
        redirected = set(plan.peer_pull_plan)

        assert designates.isdisjoint(redirected), (
            f"these clients both serve and are served, forming a chain: "
            f"{sorted(designates & redirected)}"
        )

    def test_a_client_never_reads_from_itself(self):
        clients = _cluster(nodes=2, per_node=8, tp=2)
        plan = CommunicationPlanner().make_comm_plan(clients)

        for follower, key_plans in plan.peer_pull_plan.items():
            for per_peer in key_plans.values():
                assert follower not in per_peer


class TestEveryByteStillArrives:
    def test_each_client_still_covers_all_its_shards(self):
        """Redirection must move a read, never drop it."""
        clients = _cluster(nodes=2, per_node=8, tp=2)
        plan = CommunicationPlanner().make_comm_plan(clients)

        for name, info in clients.items():
            if info.type is not NIXLClientType.PULL_SIDE:
                continue
            for key in KEYS:
                wanted = set(info.tensor_infos[key].sharding.shard_indices)
                upstream = {
                    shard
                    for shard_list in plan.get_rollout_pull_plan(name, key).values()
                    for shard in shard_list
                }
                via_peer = {
                    shard
                    for shard_list in plan.get_peer_pull_plan(name, key).values()
                    for shard in shard_list
                }
                assert wanted <= (upstream | via_peer), (
                    f"{name} {key}: missing {sorted(wanted - (upstream | via_peer))}"
                )

    def test_the_upstream_read_alone_is_sufficient(self):
        """A caller that skips every peer read must still land all its bytes.

        Peer reads need a barrier -- a peer only holds the bytes after its own upstream
        read finishes, and reading early succeeds against an unfilled buffer rather than
        failing. So a caller without that barrier has to be able to skip them, which
        means the upstream plan must remain complete on its own. Trimming it to what the
        peer plan covers would make the peer read mandatory and lose those shards in
        silence.
        """
        clients = _cluster(nodes=2, per_node=8, tp=2)
        plan = CommunicationPlanner().make_comm_plan(clients)

        for name, info in clients.items():
            if info.type is not NIXLClientType.PULL_SIDE:
                continue
            for key in KEYS:
                wanted = set(info.tensor_infos[key].sharding.shard_indices)
                upstream = {
                    shard
                    for shard_list in plan.get_rollout_pull_plan(name, key).values()
                    for shard in shard_list
                }
                assert wanted <= upstream, (
                    f"{name} {key}: upstream alone misses {sorted(wanted - upstream)}, "
                    "so skipping peer reads would silently lose bytes"
                )

    def test_peer_reads_are_distinguishable_from_upstream(self):
        """A caller must be able to defer exactly the peer reads and nothing else."""
        clients = _cluster(nodes=1, per_node=4, tp=2)
        plan = CommunicationPlanner().make_comm_plan(clients)

        for name, info in clients.items():
            if info.type is not NIXLClientType.PULL_SIDE:
                continue
            for key in KEYS:
                peers = set(plan.get_peer_pull_plan(name, key))
                upstream = set(plan.get_rollout_pull_plan(name, key))
                assert peers.isdisjoint(upstream), (
                    f"{name} {key}: {sorted(peers & upstream)} is both peer and upstream, "
                    "so a caller cannot tell which reads are safe to defer"
                )


class TestDeterminism:
    def test_the_same_cluster_gives_the_same_designates(self):
        """The index predicts holders without communicating, so this cannot vary."""
        clients = _cluster(nodes=2, per_node=8, tp=2)
        first = CommunicationPlanner().make_comm_plan(clients).peer_pull_plan
        second = CommunicationPlanner().make_comm_plan(clients).peer_pull_plan

        assert first == second


class TestPSPathUnaffected:
    def test_a_ps_cluster_also_benefits(self):
        """The duplication is on the rollout side, so it exists with a PS too."""
        clients = _cluster(nodes=1, per_node=4, tp=2)
        clients["ps_pull"] = _client("ps_pull", NIXLClientType.PS_FOR_PULL, "10.0.9.1", 2, [0, 1])
        clients["ps_push"] = _client("ps_push", NIXLClientType.PS_FOR_PUSH, "10.0.9.1", 2, [0, 1])

        plan = CommunicationPlanner().make_comm_plan(clients)

        assert plan.peer_pull_plan, "same-node duplication exists regardless of the source"


class TestSerialization:
    def test_the_peer_plan_survives_the_round_trip(self):
        from psrl.utils.nixl.comm_plan import NIXLCommPlan

        clients = _cluster(nodes=2, per_node=4, tp=2)
        plan = CommunicationPlanner().make_comm_plan(clients)
        restored = NIXLCommPlan.deserialize(plan.serialize())

        assert restored.peer_pull_plan == plan.peer_pull_plan


class TestTheHarnessPartitionsCorrectly:
    """A measurement bug this file exists to prevent recurring.

    The harness first filtered its work list to sources present in the train-client map,
    then split off the peer reads. Peer sources are rollout clients, so they were already
    gone -- and the run reported "0 peer reads deferred", which was a count of nothing
    rather than evidence that diffusion had found no duplication. The planner was working
    the whole time.

    The lesson generalises: a zero in a diagnostic is only meaningful if the thing being
    counted could have been non-zero at that point.
    """

    def test_peers_are_split_off_before_the_train_map_filter(self):
        from pathlib import Path

        import tests.nixl as tests_pkg

        harness = (Path(tests_pkg.__file__).parent / "test_nixl_e2e.py").read_text()
        start = harness.index("planned = list(self.client.iter_planned_pull_work())")
        body = harness[start : start + 1800]

        defer_at = body.index("deferred_peer_reads = [")
        narrow_at = body.index("if planned:")
        assert narrow_at < defer_at

        assert "work = list(planned)" in body, (
            "the work list is narrowed to train clients before peers are split off, so "
            "the deferral count measures nothing"
        )

    def test_the_peer_check_is_per_key(self):
        """A client can be a peer for one key and an upstream source for another."""
        from pathlib import Path

        import tests.nixl as tests_pkg

        harness = (Path(tests_pkg.__file__).parent / "test_nixl_e2e.py").read_text()
        assert "self.client.peer_sources_for(key)" in harness, (
            "a cluster-wide peer-name set would misclassify a client that is a peer for "
            "one key and an upstream source for another"
        )
