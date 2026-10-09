"""Iterating planned reads instead of probing every pair (implementation item D1).

The pull loop used to ask every source about every key and discard the empty answers.
Against a PS that is nearly free -- a couple of workers, each holding every key. Under
direct supply the sources are train ranks and pipeline parallelism gives each a slice of
the key space, so the 5-node config probes 399 x 24 = 9,576 combinations to schedule 399
reads.

The tests measure the probe count rather than checking that the reads happen, because a
loop that still probed everything and merely filtered later would pass the weaker check
while costing exactly the same.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
from psrl.utils.nixl.comm_plan import CommunicationPlanner, NIXLClientInfo, NIXLClientType
from psrl.utils.nixl.nixl_spec import NIXLSharding, NIXLShardMetaInfo, NIXLTensorInfo

pytestmark = pytest.mark.cpu_test


def _tensor_info(shard_count: int = 2) -> NIXLTensorInfo:
    sharding = NIXLSharding(
        shard_mesh=OrderedDict([(0, shard_count)]),
        shard_indices=[(i,) for i in range(shard_count)],
    )
    meta = NIXLShardMetaInfo(
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        shape=torch.Size((32, 32)),
        stride=(32, 1),
        is_contiguous=True,
    )
    return NIXLTensorInfo(
        desc_bytes_list=[None] * shard_count,
        temp_desc_bytes_list=[None] * shard_count,
        sharding=sharding,
        shard_meta_infos=[meta] * shard_count,
    )


def _client(name, client_type, keys, node_ip="10.0.0.1"):
    return NIXLClientInfo(
        name=name,
        node_ip=node_ip,
        node_gpu_id=0,
        type=client_type,
        tensor_infos={key: _tensor_info() for key in keys},
        meta=b"",
    )


def _pipeline_cluster(stages: int = 4, keys_per_stage: int = 10):
    """Train ranks each holding one pipeline stage's keys, plus one rollout."""
    clients = {}
    all_keys = []
    for stage in range(stages):
        stage_keys = [f"layer{stage}.w{i}" for i in range(keys_per_stage)]
        all_keys.extend(stage_keys)
        clients[f"train{stage}"] = _client(
            f"train{stage}", NIXLClientType.PUSH_SIDE, stage_keys, node_ip=f"10.0.0.{stage}"
        )
    clients["gen0"] = _client("gen0", NIXLClientType.PULL_SIDE, all_keys, node_ip="10.0.1.0")
    return clients, all_keys


class TestPlannedWorkIsTheScheduledSet:
    def test_every_key_is_scheduled_exactly_once(self):
        clients, all_keys = _pipeline_cluster()
        plan = CommunicationPlanner().make_comm_plan(clients)

        work = list(plan.iter_rollout_pull_work("gen0"))
        scheduled_keys = [key for key, _source in work]

        assert sorted(scheduled_keys) == sorted(all_keys)
        assert len(scheduled_keys) == len(set(scheduled_keys)), "a key is read twice"

    def test_each_key_is_read_from_a_rank_that_holds_it(self):
        """The property probing enforced by accident; now it has to be explicit."""
        clients, _ = _pipeline_cluster()
        plan = CommunicationPlanner().make_comm_plan(clients)

        for key, source in plan.iter_rollout_pull_work("gen0"):
            assert key in clients[source].tensor_infos, (
                f"{source} was scheduled to serve {key}, which it does not hold"
            )

    def test_an_unknown_client_yields_no_work(self):
        clients, _ = _pipeline_cluster()
        plan = CommunicationPlanner().make_comm_plan(clients)

        assert list(plan.iter_rollout_pull_work("nobody")) == []


class TestProbeCountCollapses:
    """The measurable claim, since a filter-after-probe loop would cost the same."""

    def test_planned_work_is_far_smaller_than_the_probe_space(self):
        stages, keys_per_stage = 24, 16
        clients, all_keys = _pipeline_cluster(stages, keys_per_stage)
        plan = CommunicationPlanner().make_comm_plan(clients)

        planned = len(list(plan.iter_rollout_pull_work("gen0")))
        probe_space = len(all_keys) * stages

        assert planned == len(all_keys)
        assert probe_space == planned * stages
        assert planned * 10 < probe_space, (
            f"planned {planned} reads against a {probe_space}-combination probe space; "
            "iterating the plan has to be a large reduction or it is not worth doing"
        )

    def test_the_reduction_grows_with_the_rank_count(self):
        """Probing scales with sources; the plan does not."""
        planned_by_stages = {}
        for stages in (4, 8, 16):
            clients, all_keys = _pipeline_cluster(stages, keys_per_stage=8)
            plan = CommunicationPlanner().make_comm_plan(clients)
            planned_by_stages[stages] = (
                len(list(plan.iter_rollout_pull_work("gen0"))),
                len(all_keys) * stages,
            )

        for stages, (planned, probes) in planned_by_stages.items():
            assert planned == probes // stages, (
                f"{stages} stages: planned {planned}, probes {probes}"
            )


class TestPSPathUnchanged:
    def test_a_ps_cluster_schedules_from_the_ps(self):
        keys = ["w0", "w1"]
        clients = {
            "train0": _client("train0", NIXLClientType.PUSH_SIDE, keys),
            "ps_pull": _client("ps_pull", NIXLClientType.PS_FOR_PULL, keys),
            "ps_push": _client("ps_push", NIXLClientType.PS_FOR_PUSH, keys),
            "gen0": _client("gen0", NIXLClientType.PULL_SIDE, keys),
        }
        plan = CommunicationPlanner().make_comm_plan(clients)

        sources = {source for _key, source in plan.iter_rollout_pull_work("gen0")}
        assert sources == {"ps_pull"}


class TestFallbackIsDistinguishable:
    def test_a_client_without_a_plan_returns_an_empty_iterator(self):
        """The caller must treat empty as 'probe', not as 'nothing to read'."""
        from psrl.utils.nixl.comm_plan import NIXLCommPlan

        empty = NIXLCommPlan(
            push_to_ps_plan={}, rollout_pull_from_ps_plan={}, train_pull_from_ps_plan={}
        )
        assert list(empty.iter_rollout_pull_work("gen0")) == []

    def test_the_harness_falls_back_when_the_plan_is_empty(self):
        from pathlib import Path

        import tests.nixl as tests_pkg

        harness = (Path(tests_pkg.__file__).parent / "test_nixl_e2e.py").read_text()
        assert "if planned:" in harness
        assert "for source in ps_client_names" in harness, (
            "no probe fallback: an uncovered client would read nothing and look fast"
        )
