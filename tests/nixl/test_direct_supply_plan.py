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


class TestTheReadCountAgainstPsIsNowPinned:
    """Why direct supply loses to a PS on time, as an assertion rather than a note.

    An earlier version of D1 was judged on probe counts (9,576 probes collapsing to 653
    reads) and that was the wrong metric: probes against an absent shard are cheap --
    `client_read` returns early on an empty shard list -- so collapsing them saved nothing.
    What costs time is the number of *transfers*, and that is set by how many sources a
    reader must gather one key from.

    The arithmetic under TP=4 x PP=2 with gen TP=2:

      * a train rank holds a quarter of any row-parallel key, so no single rank can serve
        a whole key and a reader gathers from 2 sources on average;
      * a PS worker holds complete shards, so the same reader gathers from 1.

    Two sources instead of one is a structural 2x on transfer count, and it is what the
    measured 2.93s against 1.14s reflects. The way out is not a transport parameter: it is
    making the *gen-side shard* contiguous at registration, which is the sharding-layout
    change the design points at. These tests fix the current number so that change has to
    move it deliberately.
    """

    TOTAL_SHARDS = 8

    def _info(self, name, ctype, shards_per_key, node_ip, gpu):
        sharding = NIXLSharding(
            shard_mesh=OrderedDict([(0, self.TOTAL_SHARDS)]),
            shard_indices=list(shards_per_key),
        )
        meta = NIXLShardMetaInfo(
            dtype=torch.bfloat16,
            device=torch.device("cuda:0"),
            shape=torch.Size([64, 64]),
            stride=(64, 1),
            is_contiguous=True,
        )
        n = len(list(shards_per_key))
        return NIXLClientInfo(
            name=name,
            node_ip=node_ip,
            node_gpu_id=gpu,
            type=ctype,
            tensor_infos={
                key: NIXLTensorInfo(
                    desc_bytes_list=[None] * n,
                    temp_desc_bytes_list=[None] * n,
                    sharding=sharding,
                    shard_meta_infos=[meta] * n,
                )
                for key in KEYS
            },
            meta=b"",
        )

    def _cluster(self, with_ps: bool):
        clients = {}
        # 24 train ranks at TP=4: rank r holds 2 of the 8 shards, so half a key each.
        for r in range(24):
            clients[f"train{r}"] = self._info(
                f"train{r}",
                NIXLClientType.PUSH_SIDE,
                [(r % 4,), (r % 4 + 4,)],
                f"10.0.0.{1 + r // 8}",
                r % 8,
            )
        # 8 rollout replicas on one node, gen TP=2, so four share each shard set.
        for i in range(8):
            clients[f"gen{i}"] = self._info(
                f"gen{i}",
                NIXLClientType.PULL_SIDE,
                [(s,) for s in range(i % 2, self.TOTAL_SHARDS, 2)],
                "10.0.0.1",
                i,
            )
        if with_ps:
            for p in range(2):
                for kind in (NIXLClientType.PS_FOR_PULL, NIXLClientType.PS_FOR_PUSH):
                    clients[f"{kind.name.lower()}{p}"] = self._info(
                        f"{kind.name.lower()}{p}",
                        kind,
                        [(s,) for s in range(self.TOTAL_SHARDS)],
                        f"10.0.2.{p + 1}",
                        0,
                    )
        return clients

    def _sources_per_key(self, clients) -> float:
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)
        counts = []
        for i in range(8):
            for key in KEYS:
                srcs = plan.get_rollout_pull_plan(f"gen{i}", key) or {}
                if srcs:
                    counts.append(len(srcs))
        return sum(counts) / len(counts)

    def test_a_train_rank_cannot_serve_a_whole_key(self):
        """The structural fact behind everything else in this class.

        At TP=4 a rank holds a quarter of the key space; the reader needs the whole key,
        so no plan can hide the fact that the data lives on several ranks.
        """
        direct = self._sources_per_key(self._cluster(with_ps=False))

        assert direct > 1.0, f"expected a split read, got {direct} source(s) per key"

    def test_a_ps_worker_serves_a_whole_key_from_one_source(self):
        """Which is exactly the advantage the PS has, and why deleting it costs time."""
        with_ps = self._sources_per_key(self._cluster(with_ps=True))

        assert with_ps == pytest.approx(1.0)

    def test_direct_costs_more_transfers_than_the_ps_does(self):
        """Pinned numerically so the eventual layout fix has to move it on purpose."""
        direct = self._sources_per_key(self._cluster(with_ps=False))
        with_ps = self._sources_per_key(self._cluster(with_ps=True))

        assert direct == pytest.approx(2.0), (
            f"source count changed to {direct}; if the sharding layout was made "
            f"contiguous this should drop toward the PS's {with_ps}, and this assertion "
            f"is where that improvement gets recorded"
        )


class TestThePairedKnobsAreEnforced:
    """`peer_diffusion` and `shards_must_be_readable` are one decision, not two.

    A designate serves its followers by being read from, which needs a permanent remote
    descriptor; a staged shard has none. So diffusion with staging is not a slower
    configuration, it is a broken one -- and it broke late and far from its cause, as
    "Remote descriptor must be contiguous for client read" on a follower.

    Both keys were also undeclared in `nixl.yaml`, reaching the client only through
    `getattr` defaults, which is what made the pairing invisible to a reader.
    """

    def _config(self, diffusion: bool, readable: bool):
        from omegaconf import OmegaConf

        return OmegaConf.create(
            {"psrl": {"nixl": {"peer_diffusion": diffusion, "shards_must_be_readable": readable}}}
        )

    def test_diffusion_without_readable_shards_is_refused(self):
        from psrl.utils.config import _validate_nixl_peer_diffusion

        with pytest.raises(AssertionError, match="requires shards_must_be_readable"):
            _validate_nixl_peer_diffusion(self._config(diffusion=True, readable=False))

    @pytest.mark.parametrize(
        ("diffusion", "readable"),
        [(True, True), (False, True), (False, False)],
    )
    def test_the_other_three_combinations_are_allowed(self, diffusion, readable):
        """Staging without diffusion is the whole point of having the knob."""
        from psrl.utils.config import _validate_nixl_peer_diffusion

        _validate_nixl_peer_diffusion(self._config(diffusion, readable))

    def test_both_keys_are_declared_in_the_config_file(self):
        """An undeclared key reaches the client as a `getattr` default, so a reader of the
        YAML cannot see that the two are coupled -- or that they exist at all."""
        from pathlib import Path

        text = (
            Path(__file__).resolve().parents[2] / "psrl/trainer/config/psrl/nixl.yaml"
        ).read_text()

        assert "shards_must_be_readable:" in text
        assert "peer_diffusion:" in text

    def test_the_validator_runs_for_direct_supply(self):
        """Direct supply is the mode where diffusion is actually reachable."""
        from pathlib import Path

        source = (Path(__file__).resolve().parents[2] / "psrl/utils/config.py").read_text()
        at = source.index('if config.psrl.ps_mode == "nixl_direct":')

        assert "_validate_nixl_peer_diffusion(config)" in source[at : at + 2500]
