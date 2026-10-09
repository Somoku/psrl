"""Rollout reads from the trainer when no PS exists (implementation item D3).

D3's end state is that the PS carries no bytes: rollouts read from train clients
directly. The planner reaches that by source set rather than by mode flag -- a cluster
either has PS clients registered or it does not, and that fact is already in the client
map, so there is nothing to keep in sync.

These tests fix the behaviour before `PSStorageWorker` is deleted, so the deletion can
be judged against a plan that already works rather than against a plan being written at
the same time.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from psrl.utils.nixl.comm_plan import CommunicationPlanner, NIXLClientInfo, NIXLClientType
from psrl.utils.nixl.nixl_spec import NIXLSharding, NIXLShardMetaInfo, NIXLTensorInfo

pytestmark = pytest.mark.cpu_test


def _tensor_info(shard_count: int, shape=(64, 64)) -> NIXLTensorInfo:
    sharding = NIXLSharding(
        shard_mesh=OrderedDict([(0, shard_count)]),
        shard_indices=[(i,) for i in range(shard_count)],
    )
    meta = NIXLShardMetaInfo(
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        shape=torch.Size(shape),
        stride=(shape[1], 1),
        is_contiguous=True,
    )
    return NIXLTensorInfo(
        desc_bytes_list=[None] * shard_count,
        temp_desc_bytes_list=[None] * shard_count,
        sharding=sharding,
        shard_meta_infos=[meta] * shard_count,
    )


def _client(name: str, client_type: NIXLClientType, keys, node_ip="10.0.0.1", gpu=0):
    return NIXLClientInfo(
        name=name,
        node_ip=node_ip,
        node_gpu_id=gpu,
        type=client_type,
        tensor_infos={key: _tensor_info(2) for key in keys},
        meta=b"",
    )


KEYS = ["w0", "w1"]


def _direct_cluster():
    """Trainer and rollout only -- the D3 end state."""
    return {
        "train0": _client("train0", NIXLClientType.PUSH_SIDE, KEYS, node_ip="10.0.0.1"),
        "train1": _client("train1", NIXLClientType.PUSH_SIDE, KEYS, node_ip="10.0.0.2"),
        "gen0": _client("gen0", NIXLClientType.PULL_SIDE, KEYS, node_ip="10.0.0.3"),
        "gen1": _client("gen1", NIXLClientType.PULL_SIDE, KEYS, node_ip="10.0.0.4"),
    }


def _ps_cluster():
    """With a PS present -- the arrangement D3 replaces."""
    clients = _direct_cluster()
    clients["ps0"] = _client("ps0", NIXLClientType.PS_FOR_PULL, KEYS, node_ip="10.0.0.5")
    clients["psw0"] = _client("psw0", NIXLClientType.PS_FOR_PUSH, KEYS, node_ip="10.0.0.5")
    return clients


class TestDirectSupply:
    def test_rollouts_get_a_pull_plan_without_a_ps(self):
        plan = CommunicationPlanner().make_comm_plan(_direct_cluster())

        for gen in ("gen0", "gen1"):
            assert plan.rollout_pull_from_ps_plan[gen], f"{gen} has no plan to pull from"

    def test_every_source_is_a_train_client(self):
        plan = CommunicationPlanner().make_comm_plan(_direct_cluster())

        sources = {
            source
            for key_plans in plan.rollout_pull_from_ps_plan.values()
            for per_key in key_plans.values()
            for source in per_key
        }
        assert sources <= {"train0", "train1"}, f"rollout read from a non-trainer: {sources}"

    def test_every_key_is_covered(self):
        """A missing key is a parameter that silently never updates."""
        plan = CommunicationPlanner().make_comm_plan(_direct_cluster())

        for gen in ("gen0", "gen1"):
            assert set(plan.rollout_pull_from_ps_plan[gen]) == set(KEYS)

    def test_every_shard_is_assigned_exactly_once(self):
        plan = CommunicationPlanner().make_comm_plan(_direct_cluster())

        for key_plans in plan.rollout_pull_from_ps_plan.values():
            for key, per_source in key_plans.items():
                shards = [shard for shard_list in per_source.values() for shard in shard_list]
                assert len(shards) == len(set(shards)), f"{key} has a duplicated shard"
                assert set(shards) == {(0,), (1,)}

    def test_the_chunk_plan_is_still_built(self):
        """B1's pipeline must survive the change of source."""
        plan = CommunicationPlanner().make_comm_plan(_direct_cluster())

        chunks, groups = plan.get_pull_chunks("gen0")
        assert chunks and groups


class TestPSStillWins:
    """While both exist, the PS path must be unchanged."""

    def test_a_ps_cluster_reads_from_the_ps(self):
        plan = CommunicationPlanner().make_comm_plan(_ps_cluster())

        sources = {
            source
            for key_plans in plan.rollout_pull_from_ps_plan.values()
            for per_key in key_plans.values()
            for source in per_key
        }
        assert sources == {"ps0"}, f"expected the PS to serve rollouts, got {sources}"

    def test_the_train_pull_plan_is_unaffected(self):
        plan = CommunicationPlanner().make_comm_plan(_ps_cluster())
        assert plan.train_pull_from_ps_plan["train0"]


class TestNoRolloutsIsNotAnError:
    def test_a_trainer_only_cluster_plans_nothing_to_pull(self):
        """Bootstrap and shutdown both pass through this state."""
        clients = {
            "train0": _client("train0", NIXLClientType.PUSH_SIDE, KEYS),
        }
        plan = CommunicationPlanner().make_comm_plan(clients)

        assert plan.rollout_pull_from_ps_plan == {}
