"""Degrading when trainer ranks fail (implementation items D7 and D8).

D7 and D8 are one change because the response ladder's first rung only exists when DP
replicas do, and whether they do is a property of the backend. The decisive test is the
pair: the same single-rank failure must take DP takeover under HSDP/Megatron and must
return CHUNK_UNAVAILABLE under pure FSDP2.
"""

from __future__ import annotations

import pytest
from psrl.workers.ps.chunk_availability import (
    ChunkAvailabilityIndex,
    ChunkStatus,
    derive_replica_count,
)

pytestmark = pytest.mark.cpu_test


class TestReplicaCountIsDerived:
    """D8: not an assumption that DP replicas exist."""

    def test_megatron_tp_leaves_dp_replicas(self):
        """TP=4 across 32 ranks means 8 identical copies of each shard."""
        assert derive_replica_count({0: 4}, world_size=32) == 8

    def test_pure_fsdp2_has_a_single_replica(self):
        """fsdp_size == world_size cuts across everything; nothing is replicated."""
        assert derive_replica_count({0: 16}, world_size=16) == 1

    def test_hsdp_leaves_the_replicate_degree(self):
        """HSDP shards within a group and replicates across groups."""
        assert derive_replica_count({0: 8}, world_size=16) == 2

    def test_a_two_dimensional_mesh_multiplies_out(self):
        """EP x ETP: 4 x 2 = 8 shards, so 32 ranks give 4 replicas."""
        assert derive_replica_count({0: 4, 1: 2}, world_size=32) == 4

    def test_a_mesh_larger_than_the_world_is_a_config_error(self):
        """Returning 0 would silently make every chunk unavailable."""
        with pytest.raises(ValueError, match="disagree"):
            derive_replica_count({0: 64}, world_size=16)

    def test_a_non_positive_world_is_rejected(self):
        with pytest.raises(ValueError, match="world_size must be positive"):
            derive_replica_count({0: 1}, world_size=0)


class TestSingleRankFailure:
    """Rung one: another rank of the DP group serves the same bytes."""

    def test_a_surviving_replica_takes_over(self):
        index = ChunkAvailabilityIndex()
        index.register(0, ["r0", "r1", "r2"])

        index.report_failure("r0")
        resolution = index.resolve(0, prefer="r0")

        assert resolution.is_available
        assert resolution.source != "r0"

    def test_the_rollout_does_not_see_the_failure(self):
        """Same chunk, still available, just from elsewhere."""
        index = ChunkAvailabilityIndex()
        index.register(0, ["r0", "r1"])

        before = index.resolve(0)
        index.report_failure(before.source)
        after = index.resolve(0)

        assert after.is_available
        assert after.source != before.source

    def test_a_preferred_live_rank_is_honoured(self):
        index = ChunkAvailabilityIndex()
        index.register(0, ["r0", "r1", "r2"])

        assert index.resolve(0, prefer="r2").source == "r2"

    def test_slack_reports_how_many_more_failures_fit(self):
        index = ChunkAvailabilityIndex()
        index.register(0, ["r0", "r1", "r2"])

        assert index.replica_slack(0) == 2
        index.report_failure("r0")
        assert index.replica_slack(0) == 1


class TestWholeGroupFailure:
    """Rung two: an explicit value, never a hang."""

    def test_losing_every_holder_returns_chunk_unavailable(self):
        index = ChunkAvailabilityIndex()
        index.register(0, ["r0", "r1"])

        index.report_failure("r0")
        index.report_failure("r1")

        assert index.resolve(0).status is ChunkStatus.CHUNK_UNAVAILABLE

    def test_an_unavailable_chunk_names_no_source(self):
        """The caller must not be handed an address to read from."""
        index = ChunkAvailabilityIndex()
        index.register(0, ["r0"])
        index.report_failure("r0")

        assert index.resolve(0).source is None

    def test_an_unknown_chunk_is_unavailable_not_an_error(self):
        index = ChunkAvailabilityIndex()
        assert index.resolve(99).status is ChunkStatus.CHUNK_UNAVAILABLE

    def test_other_chunks_stay_available(self):
        """One dead group must not degrade the whole version."""
        index = ChunkAvailabilityIndex()
        index.register(0, ["r0"])
        index.register(1, ["r1"])
        index.report_failure("r0")

        assert index.resolve(0).status is ChunkStatus.CHUNK_UNAVAILABLE
        assert index.resolve(1).is_available


class TestTrainerRestart:
    """Rung three: holders cleared, version retained."""

    def test_clearing_holders_degrades_every_pull(self):
        index = ChunkAvailabilityIndex()
        index.register(0, ["r0"])
        index.register(1, ["r1"])

        index.clear_all_holders()

        assert all(
            index.resolve(chunk).status is ChunkStatus.CHUNK_UNAVAILABLE for chunk in (0, 1)
        )

    def test_re_registering_restores_service(self):
        index = ChunkAvailabilityIndex()
        index.register(0, ["r0"])
        index.clear_all_holders()
        index.register(0, ["r0"])

        assert index.resolve(0).is_available

    def test_a_restored_rank_serves_again(self):
        index = ChunkAvailabilityIndex()
        index.register(0, ["r0"])
        index.report_failure("r0")
        index.restore("r0")

        assert index.resolve(0).is_available


class TestBackendsDivergeOnTheSameFailure:
    """The pair that makes D7 and D8 one change rather than two."""

    @staticmethod
    def _cluster(replica_count: int) -> ChunkAvailabilityIndex:
        index = ChunkAvailabilityIndex()
        index.register(0, [f"r{i}" for i in range(replica_count)])
        return index

    def test_hsdp_takes_dp_takeover(self):
        replicas = derive_replica_count({0: 8}, world_size=16)
        index = self._cluster(replicas)

        index.report_failure("r0")

        assert index.resolve(0).is_available, "HSDP has a replica and must take over"

    def test_pure_fsdp2_degrades_instead(self):
        replicas = derive_replica_count({0: 16}, world_size=16)
        index = self._cluster(replicas)

        index.report_failure("r0")

        assert index.resolve(0).status is ChunkStatus.CHUNK_UNAVAILABLE, (
            "pure FSDP2 has no replica, so attempting DP takeover would be wrong"
        )


class TestVersionVisibility:
    """A partially deliverable version must not be published (design 3.4.6)."""

    def test_a_complete_version_is_deliverable(self):
        index = ChunkAvailabilityIndex()
        for chunk in range(4):
            index.register(chunk, ["r0"])

        assert index.version_is_deliverable(range(4))

    def test_one_missing_chunk_makes_the_version_undeliverable(self):
        """Half an update is worse than the old version, and silent."""
        index = ChunkAvailabilityIndex()
        for chunk in range(4):
            index.register(chunk, [f"r{chunk}"])
        index.report_failure("r2")

        assert not index.version_is_deliverable(range(4))
