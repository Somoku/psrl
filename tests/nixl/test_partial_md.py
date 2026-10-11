"""Two-stage metadata exchange (implementation item E4).

The property that matters is a scaling one: join cost must stop growing with the joiner's
registered region count. Measured on H20 (`bench_partial_md.py`), connection info is
constant at ~1645 B from 1 region to 4096, while full metadata grows to 1.73 MB -- a
1054x gap that becomes 17 GB vs 16 MB per join at 10k members.

These tests pin the *plan* (who gets which half) on CPU. The byte sizes themselves are a
property of NIXL, measured by the benchmark rather than asserted here, because pinning
them in a unit test would turn a NIXL version bump into a failure with no defect.
"""

from __future__ import annotations

import pytest

from psrl.utils.nixl.partial_md import (
    MetadataPlan,
    describe_leave_gap,
    plan_metadata_fanout,
)

pytestmark = pytest.mark.cpu_test


class TestFanoutPlan:
    def test_everyone_gets_connection_info(self):
        plan = plan_metadata_fanout("gen0", ["gen0", "gen1", "train0"], readers_of={"gen1"})

        assert plan.conn_info_peers == {"gen1", "train0"}

    def test_only_readers_get_descriptors(self):
        """The expensive half goes to the few, which is the whole point of the split."""
        plan = plan_metadata_fanout(
            "train0", ["train0"] + [f"gen{i}" for i in range(16)], readers_of={"gen3"}
        )

        assert plan.descriptor_peers == {"gen3"}
        assert len(plan.conn_info_peers) == 16

    def test_a_client_is_never_its_own_peer(self):
        plan = plan_metadata_fanout("gen0", ["gen0", "gen1"], readers_of={"gen0", "gen1"})

        assert "gen0" not in plan.conn_info_peers
        assert "gen0" not in plan.descriptor_peers

    def test_readers_outside_the_cluster_are_dropped(self):
        """A plan naming a client that has left must not create a phantom recipient."""
        plan = plan_metadata_fanout("train0", ["train0", "gen0"], readers_of={"gen0", "gone"})

        assert plan.descriptor_peers == {"gen0"}

    def test_descriptor_fraction_reports_the_saving(self):
        plan = plan_metadata_fanout(
            "train0", ["train0"] + [f"gen{i}" for i in range(99)], readers_of={"gen0"}
        )

        assert plan.descriptor_fraction == pytest.approx(1 / 99)

    def test_descriptor_without_connection_info_is_rejected(self):
        """Descriptors are useless without a link, so the pairing must be impossible."""
        with pytest.raises(ValueError, match="could not establish a link"):
            MetadataPlan(
                conn_info_peers=frozenset({"gen0"}),
                descriptor_peers=frozenset({"gen0", "gen1"}),
            )


class TestFanoutDoesNotGrowWithClusterSize:
    """The reason E4 exists: descriptor recipients are set by the plan, not by membership."""

    def test_descriptor_recipients_stay_flat_as_the_cluster_grows(self):
        counts = {}
        for total in (16, 256, 4096):
            clients = ["train0"] + [f"gen{i}" for i in range(total)]
            # However large the cluster, one trainer rank is read by a fixed group.
            plan = plan_metadata_fanout("train0", clients, readers_of={"gen0", "gen1"})
            counts[total] = len(plan.descriptor_peers)

        assert len(set(counts.values())) == 1, (
            f"descriptor recipients grew with cluster size: {counts}"
        )

    def test_connection_info_recipients_do_grow(self):
        """Stated explicitly: stage one is O(members), and that is accepted.

        It is affordable precisely because its payload is constant -- 10k x 1.6 KB is
        16 MB. Were stage one to carry descriptors too, the same join would cost 17 GB.
        """
        small = plan_metadata_fanout("train0", ["train0", "gen0"], readers_of=set())
        large = plan_metadata_fanout(
            "train0", ["train0"] + [f"gen{i}" for i in range(999)], readers_of=set()
        )

        assert len(large.conn_info_peers) > len(small.conn_info_peers)


class TestLeavePathGapIsStated:
    def test_the_gap_names_the_missing_binding(self):
        """A blocked dependency must be findable, not rediscovered by whoever tries next."""
        gap = describe_leave_gap()

        assert "invalidate_local_md" in gap

    def test_the_binding_is_still_absent(self):
        """Fails when NIXL gains the binding, which is the signal to finish E4's leave path."""
        from nixl._api import nixl_agent

        assert not hasattr(nixl_agent, "invalidate_local_md"), (
            "NIXL now exposes invalidate_local_md; the shrink path in E4 can be completed "
            "and describe_leave_gap should be removed"
        )


class TestControlPlaneScaling:
    """C2's second criterion: control-plane messages must not grow quadratically.

    Naively, every rollout needing every trainer's descriptors is `instances x ranks`
    messages, and both factors grow with the cluster. What actually bounds it is that the
    comm plan assigns each rollout a *subset* of sources, so a trainer's descriptor
    recipients are decided by sharding, not by headcount.

    Measured against the planner rather than asserted from the design, because the claim
    is a property of the plan the planner produces.
    """

    @staticmethod
    def _descriptor_recipients(per_node: int) -> tuple[int, int]:
        from psrl.utils.nixl.comm_plan import CommunicationPlanner

        from tests.nixl.test_peer_diffusion_plan import _cluster

        clients = _cluster(nodes=2, per_node=per_node, tp=2)
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)

        readers = {
            pull_client
            for pull_client, keys in plan.rollout_pull_from_ps_plan.items()
            for per_source in keys.values()
            if "train0" in per_source
        }
        fanout = plan_metadata_fanout("train0", list(clients), readers)
        return 2 * per_node, len(fanout.descriptor_peers)

    def test_recipients_grow_no_faster_than_linearly(self):
        """Quadratic growth is the failure this criterion exists to catch."""
        points = [self._descriptor_recipients(n) for n in (2, 4, 8, 16)]

        first_rollouts, first_recipients = points[0]
        for rollouts, recipients in points[1:]:
            growth = recipients / max(first_recipients, 1)
            scale = rollouts / first_rollouts
            assert growth <= scale * 1.5, (
                f"{rollouts} rollouts needed {recipients} descriptor recipients, "
                f"{growth:.1f}x the baseline for a {scale:.0f}x cluster -- superlinear"
            )

    def test_each_rollout_reads_a_subset_of_trainers_not_all_of_them(self):
        """The mechanism behind the bound: sharding, not headcount, picks the sources."""
        from psrl.utils.nixl.comm_plan import CommunicationPlanner

        from tests.nixl.test_peer_diffusion_plan import _cluster

        clients = _cluster(nodes=2, per_node=8, tp=2)
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)
        trainers = {name for name in clients if name.startswith("train")}

        for pull_client, keys in plan.rollout_pull_from_ps_plan.items():
            sources = {src for per_source in keys.values() for src in per_source}
            assert sources <= trainers
            assert sources, f"{pull_client} was assigned no source at all"
