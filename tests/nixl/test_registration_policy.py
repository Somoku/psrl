"""Registration is derived from the plan, not from a deployment flag (design §3.7.3.0).

The property under test is the one that makes the colocate switch cheap: when every
reader of a client is on the client's own node, nothing it publishes can be reached by a
remote NIC, so the RDMA registration it does today is unusable by construction. The
inverse property matters more for correctness: a single remote reader must force
registration, because an unregistered shard that a remote peer tries to read does not
degrade gracefully.

These are plan-level tests, so they run on CPU. The end-to-end count of `register_memory`
calls on a colocate switch is asserted in `test_registration_wiring.py`.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
from psrl.utils.nixl.comm_plan import CommunicationPlanner, NIXLClientInfo, NIXLClientType
from psrl.utils.nixl.nixl_spec import NIXLSharding, NIXLShardMetaInfo, NIXLTensorInfo
from psrl.utils.nixl.registration_policy import (
    keys_read_across_nodes,
    needs_rdma_registration,
    readers_by_source,
    registration_report,
)

pytestmark = pytest.mark.cpu_test

KEYS = ["w0", "w1"]
#: Trainer node and rollout node, distinct so the default cluster is disaggregate.
TRAIN_NODE = "10.0.0.1"
ROLLOUT_NODE = "10.1.0.1"


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


def _client(name, client_type, node_ip, shard_count=2, owned=(0, 1)):
    return NIXLClientInfo(
        name=name,
        node_ip=node_ip,
        node_gpu_id=0,
        type=client_type,
        tensor_infos={key: _tensor_info(shard_count, list(owned)) for key in KEYS},
        meta=b"",
    )


def _direct_supply_cluster(*, rollout_nodes: int, per_node: int = 2, train_node: str = TRAIN_NODE):
    """Trainers holding every shard plus `rollout_nodes` nodes of rolling-out clients.

    No PS clients, so the planner takes the direct-supply branch and names the trainers
    themselves as the rollout sources -- which is the arrangement the criterion has to
    handle for a colocated run.
    """
    clients = {
        "train0": _client("train0", NIXLClientType.PUSH_SIDE, train_node),
    }
    for node in range(rollout_nodes):
        # Same node as the trainer for node 0 when `train_node` is that node's ip.
        node_ip = train_node if node == 0 else f"10.1.0.{node}"
        for slot in range(per_node):
            name = f"gen{node}_{slot}"
            clients[name] = _client(name, NIXLClientType.PULL_SIDE, node_ip)
    return clients


class TestPureColocateNeedsNothing:
    """The headline case: every reader shares the node, so registration is pure waste."""

    def test_a_trainer_read_only_by_its_own_node_registers_nothing(self):
        clients = _direct_supply_cluster(rollout_nodes=1)
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)

        assert needs_rdma_registration(plan, clients, "train0") is False

    def test_the_skipped_keys_are_named_not_just_counted(self):
        """`skippable_keys` is the measurement: what would have been registered and
        cannot be read remotely. A bare False would not survive a plan change."""
        clients = _direct_supply_cluster(rollout_nodes=1)
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)

        report = registration_report(plan, clients, "train0")

        assert report["needs_registration"] is False
        assert report["cross_node_keys"] == 0
        assert report["skippable_keys"] == report["published_keys"] == len(KEYS)

    def test_the_verdict_is_not_vacuous(self):
        """A `False` because nobody reads the client is not the claim. The claim is that
        the readers exist and are all local, so this pins that the trainer really is a
        named source -- without it, an empty plan would make every test above pass."""
        clients = _direct_supply_cluster(rollout_nodes=1)
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)

        readers = readers_by_source(plan)

        assert readers["train0"], "train0 is not a source in the plan"
        assert readers["train0"]["w0"], "no reader of w0"
        assert {clients[r].node_ip for r in readers["train0"]["w0"]} == {TRAIN_NODE}

    def test_peer_diffusion_does_not_change_the_verdict(self):
        """Diffusion reorders who reads whom but adds no cross-node reader, and the
        policy ignores delegation anyway -- so both planner settings must agree."""
        clients = _direct_supply_cluster(rollout_nodes=1, per_node=4)
        off = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)
        on = CommunicationPlanner(peer_diffusion=True).make_comm_plan(clients)

        assert needs_rdma_registration(off, clients, "train0") is False
        assert needs_rdma_registration(on, clients, "train0") is False


class TestOneRemoteReaderForcesRegistration:
    def test_a_remote_rollout_node_is_enough(self):
        clients = _direct_supply_cluster(rollout_nodes=2)
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)

        assert needs_rdma_registration(plan, clients, "train0") is True

    def test_a_local_reader_does_not_cancel_a_remote_one(self):
        """The mixed deployment. Registering here is not waste -- the local reader still
        takes the IPC path, so the extra registration costs only the RDMA MR, while
        skipping it would break the remote reader."""
        clients = _direct_supply_cluster(rollout_nodes=2, per_node=2)
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)

        report = registration_report(plan, clients, "train0")

        assert report["needs_registration"] is True
        assert report["cross_node_keys"] == len(KEYS)
        assert report["skippable_keys"] == 0


class TestUnknownsAnswerRegister:
    """Wrong in this direction costs performance; wrong the other way costs correctness."""

    def test_a_reader_without_a_node_forces_registration(self):
        clients = _direct_supply_cluster(rollout_nodes=1)
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)
        before = needs_rdma_registration(plan, clients, "train0")
        assert before is False
        # A reader the cluster knows about but whose node we cannot see.
        clients["gen0_0"].node_ip = ""

        assert needs_rdma_registration(plan, clients, "train0") is True

    def test_a_source_with_no_known_node_forces_registration(self):
        clients = _direct_supply_cluster(rollout_nodes=1)
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)
        clients["train0"].node_ip = ""

        assert needs_rdma_registration(plan, clients, "train0") is True

    def test_a_reader_absent_from_client_infos_forces_registration(self):
        clients = _direct_supply_cluster(rollout_nodes=1)
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)
        del clients["gen0_0"]

        assert needs_rdma_registration(plan, clients, "train0") is True


class TestNoReaders:
    def test_a_client_nobody_reads_registers_nothing(self):
        """An empty reader set is trivially all-local. Asserting it keeps the policy
        from being written as `not all(...)`, which would answer True here."""
        clients = _direct_supply_cluster(rollout_nodes=1)
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)

        assert needs_rdma_registration(plan, clients, "gen0_0") is False

    def test_an_empty_plan_registers_nothing(self):
        clients = _direct_supply_cluster(rollout_nodes=1)
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)
        plan.rollout_pull_from_ps_plan = {}

        assert needs_rdma_registration(plan, clients, "train0") is False

    def test_a_self_reference_is_not_a_reader(self):
        """A plan naming the source as its own source would otherwise make every client
        look remote to itself, i.e. register everything always."""
        clients = _direct_supply_cluster(rollout_nodes=1)
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)
        plan.rollout_pull_from_ps_plan["train0"] = {"w0": {"train0": [(0,)]}}

        assert needs_rdma_registration(plan, clients, "train0") is False

    def test_an_empty_shard_list_is_not_a_read(self):
        clients = _direct_supply_cluster(rollout_nodes=1)
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)
        plan.rollout_pull_from_ps_plan["train0"] = {"w0": {"train0": []}}

        assert needs_rdma_registration(plan, clients, "train0") is False


class TestInversionCollectsEveryDirection:
    def test_all_three_plan_directions_are_read(self):
        """A trainer appears as a source in `push_to_ps_plan` (to a PS) *and* in
        `rollout_pull_from_ps_plan` (direct supply). Reading only one of them would miss
        readers and answer "no registration" for a client that has remote ones."""
        clients = _direct_supply_cluster(rollout_nodes=1)
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)

        readers = readers_by_source(plan)

        assert "gen0_0" in readers["train0"]["w0"]
        assert "gen0_1" in readers["train0"]["w1"]

    def test_a_ps_read_by_remote_rollouts_registers(self):
        """With PS clients present the planner takes the PS branch; the PS is then the
        source, and it must register because its readers are on other nodes."""
        clients = {
            "train0": _client("train0", NIXLClientType.PUSH_SIDE, TRAIN_NODE),
            "ps0": _client("ps0", NIXLClientType.PS_FOR_PUSH, TRAIN_NODE),
            "ps1": _client("ps1", NIXLClientType.PS_FOR_PULL, TRAIN_NODE),
            "gen0": _client("gen0", NIXLClientType.PULL_SIDE, ROLLOUT_NODE),
        }
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)

        assert needs_rdma_registration(plan, clients, "ps1") is True

    def test_a_ps_read_only_locally_registers_nothing(self):
        clients = {
            "train0": _client("train0", NIXLClientType.PUSH_SIDE, TRAIN_NODE),
            "ps0": _client("ps0", NIXLClientType.PS_FOR_PUSH, TRAIN_NODE),
            "ps1": _client("ps1", NIXLClientType.PS_FOR_PULL, TRAIN_NODE),
            "gen0": _client("gen0", NIXLClientType.PULL_SIDE, TRAIN_NODE),
        }
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)

        assert needs_rdma_registration(plan, clients, "ps1") is False


class TestGranularityIsPerKey:
    def test_only_the_keys_with_remote_readers_are_reported(self):
        """The criterion is stated per `(key, shard)` because a mixed deployment can
        have one key read locally and another remotely. The policy answers at that
        granularity even where the wiring currently uses the coarse verdict."""
        clients = _direct_supply_cluster(rollout_nodes=2, per_node=1)
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)
        # The remote node stops needing w1.
        plan.rollout_pull_from_ps_plan["gen1_0"]["w1"] = {}

        assert keys_read_across_nodes(plan, clients, "train0") == {"w0"}
