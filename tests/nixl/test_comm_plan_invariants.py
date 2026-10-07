"""Invariants the redundancy-free planner must satisfy (pre-implementation verification 3).

Design item D1 replaces `CommunicationPlanner` with a three-stage, peer-aware planner
and deletes the PS client types it currently classifies on. That rewrite needs
executable invariants to land against, otherwise "the new planner works" rests on
inspection. The existing suite has two tests and asserts neither coverage nor
redundancy.

The invariants here are deliberately phrased against the *plan*, not the
implementation, so they survive the rewrite:

1. coverage      -- every shard a target needs is assigned to exactly one source
2. no redundancy -- a source never sends the same shard twice to the same target
3. ownership     -- a source is only asked for shards it actually holds
4. egress        -- total assigned bytes equal the bytes the targets need, not N times
5. determinism   -- the same inputs produce the same plan
6. no chains     -- no client is both a source and a target for the same shard

Invariant 6 is the one the current planner cannot violate (PS sits between push and
pull) but the direct-supply planner can, since peers both read and serve. Design doc
3.5.4 forbids chains because latency would accumulate per hop and a peer overwriting
its buffer would invalidate everything downstream.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from psrl.utils.nixl.comm_plan import (
    NIXLClientInfo,
    NIXLClientType,
    CommunicationPlanner,
)
from psrl.utils.nixl.nixl_spec import NIXLSharding, NIXLShardMetaInfo, NIXLTensorInfo

pytestmark = pytest.mark.cpu_test

KEY = "model.layers.0.mlp.w1"


def _tensor_info(shard_count: int, shard_indices: list[int], shape=(1024, 1024)) -> NIXLTensorInfo:
    sharding = NIXLSharding(
        shard_mesh=OrderedDict([(0, shard_count)]),
        shard_indices=[(i,) for i in sorted(shard_indices)],
    )
    meta = NIXLShardMetaInfo(
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        shape=torch.Size(shape),
        stride=(shape[1], 1),
        is_contiguous=True,
    )
    n = len(shard_indices)
    return NIXLTensorInfo(
        desc_bytes_list=[None] * n,
        temp_desc_bytes_list=[None] * n,
        sharding=sharding,
        shard_meta_infos=[meta] * n,
    )


def _client(
    name: str,
    client_type: NIXLClientType,
    shard_count: int,
    shard_indices: list[int],
    node_ip: str = "10.0.0.1",
    gpu: int = 0,
) -> NIXLClientInfo:
    return NIXLClientInfo(
        name=name,
        node_ip=node_ip,
        node_gpu_id=gpu,
        type=client_type,
        tensor_infos={KEY: _tensor_info(shard_count, shard_indices)},
        meta=b"",
    )


def _train_and_ps(train_shards: dict[str, list[int]], ps_names: list[str], shard_count: int = 4):
    """A push-side fan-out: each train rank holds some shards, each PS holds all."""
    clients: dict[str, NIXLClientInfo] = {}
    for i, (name, shards) in enumerate(train_shards.items()):
        clients[name] = _client(name, NIXLClientType.PUSH_SIDE, shard_count, shards, gpu=i)
    for i, name in enumerate(ps_names):
        clients[name] = _client(
            name, NIXLClientType.PS_FOR_PUSH, shard_count, list(range(shard_count)), gpu=i
        )
    return clients


def _assignments(plan_section: dict) -> list[tuple[str, str, tuple[int, ...]]]:
    """Flatten a plan section into (source_or_target, counterpart, shard) triples."""
    out = []
    for client, keys in plan_section.items():
        for _key, counterparts in keys.items():
            for counterpart, shards in counterparts.items():
                for shard in shards:
                    out.append((client, counterpart, tuple(shard)))
    return out


class TestCoverageAndRedundancy:
    """Invariants 1, 2, 3: the plan must be complete, minimal, and honest."""

    @pytest.fixture
    def plan(self):
        clients = _train_and_ps(
            {"train0": [0, 1], "train1": [2, 3]}, ["ps0", "ps1"], shard_count=4
        )
        return CommunicationPlanner().make_comm_plan(clients), clients

    def test_every_target_shard_is_covered(self, plan):
        """Invariant 1: a missing shard means the PS holds stale bytes for it."""
        comm_plan, _ = plan
        for ps_name in ("ps0", "ps1"):
            covered = set()
            for source, target, shard in _assignments(comm_plan.push_to_ps_plan):
                if target == ps_name:
                    covered.add(shard)
            assert covered == {(0,), (1,), (2,), (3,)}, f"{ps_name} missing shards: {covered}"

    def test_no_shard_is_sent_twice_to_the_same_target(self, plan):
        """Invariant 2: duplicate work, and a race over which write lands last."""
        comm_plan, _ = plan
        seen = set()
        for source, target, shard in _assignments(comm_plan.push_to_ps_plan):
            assert (target, shard) not in seen, f"{shard} sent twice to {target}"
            seen.add((target, shard))

    def test_sources_are_only_asked_for_shards_they_hold(self, plan):
        """Invariant 3: otherwise the source reads out of bounds at transfer time."""
        comm_plan, clients = plan
        for source, target, shard in _assignments(comm_plan.push_to_ps_plan):
            held = set(clients[source].tensor_infos[KEY].sharding.shard_indices)
            assert shard in held, f"{source} asked for {shard} but holds {held}"

    def test_a_source_holding_nothing_is_never_assigned(self):
        clients = _train_and_ps({"train0": [0, 1, 2, 3], "idle": []}, ["ps0"], shard_count=4)
        comm_plan = CommunicationPlanner().make_comm_plan(clients)
        sources = {source for source, _, _ in _assignments(comm_plan.push_to_ps_plan)}
        assert "idle" not in sources


class TestEgressAccounting:
    """Invariant 4: the N-times redundancy the design exists to remove."""

    def test_push_egress_scales_with_ps_count_today(self):
        """Documents the current behaviour that direct supply replaces.

        Each PS worker needs every shard, so every train rank writes its shards to all
        N of them -- total egress N x M. After item D1 the equivalent assertion becomes
        "egress == M", which is why this one is written as a measurement rather than an
        aspiration.
        """
        for ps_count in (1, 2, 4):
            clients = _train_and_ps(
                {"train0": [0, 1], "train1": [2, 3]},
                [f"ps{i}" for i in range(ps_count)],
                shard_count=4,
            )
            comm_plan = CommunicationPlanner().make_comm_plan(clients)
            assignments = _assignments(comm_plan.push_to_ps_plan)
            assert len(assignments) == 4 * ps_count, (
                f"{ps_count} PS workers: expected 4 shards x {ps_count} = "
                f"{4 * ps_count} assignments, got {len(assignments)}"
            )

    def test_pull_egress_is_exactly_what_targets_need(self):
        """Pull is already redundancy-free: each rollout reads only its own shards."""
        clients: dict[str, NIXLClientInfo] = {
            "ps0": _client("ps0", NIXLClientType.PS_FOR_PULL, 4, [0, 1, 2, 3]),
        }
        for i in range(2):
            name = f"gen{i}"
            clients[name] = _client(
                name, NIXLClientType.PULL_SIDE, 4, [2 * i, 2 * i + 1], gpu=i
            )

        comm_plan = CommunicationPlanner().make_comm_plan(clients)
        assignments = _assignments(comm_plan.rollout_pull_from_ps_plan)
        assert len(assignments) == 4, "two rollouts x two shards each, no duplication"


class TestDeterminism:
    """Invariant 5: a non-deterministic plan makes every other invariant untestable."""

    def test_same_inputs_give_identical_plans(self):
        def build():
            clients = _train_and_ps(
                {"train0": [0, 1], "train1": [2, 3]}, ["ps0", "ps1"], shard_count=4
            )
            return CommunicationPlanner().make_comm_plan(clients)

        first, second = build(), build()
        assert sorted(_assignments(first.push_to_ps_plan)) == sorted(
            _assignments(second.push_to_ps_plan)
        )

    def test_plan_survives_serialization(self):
        """The plan crosses a process boundary to each client, so it must round-trip."""
        clients = _train_and_ps({"train0": [0, 1, 2, 3]}, ["ps0"], shard_count=4)
        comm_plan = CommunicationPlanner().make_comm_plan(clients)

        restored = type(comm_plan).deserialize(comm_plan.serialize())
        assert sorted(_assignments(restored.push_to_ps_plan)) == sorted(
            _assignments(comm_plan.push_to_ps_plan)
        )


class TestNoChainedRelays:
    """Invariant 6: forbidden by design doc 3.5.4, and newly possible under D1."""

    def test_no_client_is_both_source_and_target_for_one_shard(self):
        """With PS in the middle this holds trivially; after D1 it must be enforced."""
        clients = _train_and_ps(
            {"train0": [0, 1], "train1": [2, 3]}, ["ps0", "ps1"], shard_count=4
        )
        comm_plan = CommunicationPlanner().make_comm_plan(clients)

        sends: dict[tuple[int, ...], set[str]] = {}
        receives: dict[tuple[int, ...], set[str]] = {}
        for source, target, shard in _assignments(comm_plan.push_to_ps_plan):
            sends.setdefault(shard, set()).add(source)
            receives.setdefault(shard, set()).add(target)

        for shard in sends:
            overlap = sends[shard] & receives.get(shard, set())
            assert not overlap, f"shard {shard}: {overlap} both sends and receives (a relay chain)"


class TestUnevenAndSkewedTopologies:
    """Shapes that a greedy assignment can get wrong."""

    def test_single_source_holding_everything(self):
        clients = _train_and_ps({"train0": [0, 1, 2, 3]}, ["ps0", "ps1"], shard_count=4)
        comm_plan = CommunicationPlanner().make_comm_plan(clients)
        assignments = _assignments(comm_plan.push_to_ps_plan)
        assert len(assignments) == 8
        assert {s for s, _, _ in assignments} == {"train0"}

    def test_replicated_sources_do_not_both_send(self):
        """Two ranks holding identical shards: exactly one must be chosen per target.

        This is the DP-replica case item 8b relies on for failover -- the planner must
        pick one, and the other remains available as a fallback.
        """
        clients = _train_and_ps(
            {"train0": [0, 1, 2, 3], "train1": [0, 1, 2, 3]}, ["ps0"], shard_count=4
        )
        comm_plan = CommunicationPlanner().make_comm_plan(clients)
        assignments = _assignments(comm_plan.push_to_ps_plan)
        assert len(assignments) == 4, "4 shards, each assigned once despite 2 replicas"

    def test_skewed_shard_distribution(self):
        clients = _train_and_ps(
            {"train0": [0], "train1": [1, 2, 3]}, ["ps0"], shard_count=4
        )
        comm_plan = CommunicationPlanner().make_comm_plan(clients)
        by_source: dict[str, int] = {}
        for source, _, _ in _assignments(comm_plan.push_to_ps_plan):
            by_source[source] = by_source.get(source, 0) + 1
        assert by_source == {"train0": 1, "train1": 3}

    def test_missing_key_on_all_sources_is_rejected(self):
        """A target needing a key nobody holds must fail at planning, not transfer."""
        clients = {
            "train0": _client("train0", NIXLClientType.PUSH_SIDE, 4, [0, 1, 2, 3]),
            "ps0": _client("ps0", NIXLClientType.PS_FOR_PUSH, 4, [0, 1, 2, 3]),
        }
        clients["ps0"].tensor_infos["orphan.weight"] = _tensor_info(4, [0, 1, 2, 3])

        with pytest.raises(Exception, match="orphan.weight|No available key"):
            CommunicationPlanner().make_comm_plan(clients)
