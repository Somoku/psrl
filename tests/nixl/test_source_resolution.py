"""Source resolution and the traffic it implies (implementation item D1).

D1's criteria are quantities, not behaviours: the trainer must emit `M` rather than
`N x M`, each rollout node must take in `M / num_nodes`, and no chain `A reads B reads
C` may exist. These tests count what the resolver decides, because "the code does
diffusion" is not the claim -- the byte counts are.
"""

from __future__ import annotations

import pytest
from psrl.utils.nixl.source_resolution import (
    HolderSet,
    Reader,
    SourceKind,
    SourceResolver,
    default_trainer_fanout,
)

pytestmark = pytest.mark.cpu_test


def _cluster(nodes: int, per_node: int) -> tuple[list[Reader], dict[str, str]]:
    readers = [
        Reader(name=f"r{node}_{slot}", node_id=f"node{node}")
        for node in range(nodes)
        for slot in range(per_node)
    ]
    return readers, {reader.name: reader.node_id for reader in readers}


class TestVersionEquality:
    """Reading from an older peer adds staleness the algorithm never agreed to."""

    def test_advancing_the_version_drops_every_holder(self):
        holders = HolderSet(version=3)
        holders.mark_held(0, "r0")
        holders.mark_held(1, "r1")

        holders.advance_to(4)

        assert holders.holders_of(0) == set()
        assert holders.holders_of(1) == set()

    def test_the_version_cannot_go_backwards(self):
        holders = HolderSet(version=5)
        with pytest.raises(ValueError, match="must not go backwards"):
            holders.advance_to(4)

    def test_re_advancing_to_the_same_version_keeps_holders(self):
        """Idempotence matters: a duplicate publish must not discard a whole wave."""
        holders = HolderSet(version=3)
        holders.mark_held(0, "r0")

        holders.advance_to(3)

        assert holders.holders_of(0) == {"r0"}

    def test_after_a_version_bump_everyone_reads_the_trainer_again(self):
        readers, node_of = _cluster(nodes=2, per_node=2)
        holders = HolderSet(version=1)
        for reader in readers:
            holders.mark_held(0, reader.name)
        holders.advance_to(2)

        resolver = SourceResolver(max_trainer_readers=4)
        decisions = resolver.resolve(0, readers, holders, node_of)

        assert all(d.kind is SourceKind.TRAINER for d in decisions)


class TestNoChains:
    """A peer serves only what it got from the trainer."""

    def test_a_second_hand_holder_is_not_offered_as_a_source(self):
        readers, node_of = _cluster(nodes=1, per_node=3)
        holders = HolderSet(version=1)
        # r0_0 read from the trainer; r0_1 read from r0_0 and so holds it second hand.
        holders.mark_held(0, "r0_0")
        holders.mark_held(0, "r0_1", from_peer=True)

        assert holders.servable_holders_of(0) == {"r0_0"}

    def test_the_third_reader_is_served_by_the_first_not_the_second(self):
        readers, node_of = _cluster(nodes=1, per_node=3)
        holders = HolderSet(version=1)
        holders.mark_held(0, "r0_0")
        holders.mark_held(0, "r0_1", from_peer=True)

        resolver = SourceResolver(max_trainer_readers=1)
        decisions = {d.reader: d for d in resolver.resolve(0, readers, holders, node_of)}

        assert decisions["r0_2"].source == "r0_0", (
            "serving from r0_1 would make the chain trainer -> r0_0 -> r0_1 -> r0_2"
        )

    def test_no_resolved_source_is_itself_second_hand(self):
        """The structural check, across a whole wave."""
        readers, node_of = _cluster(nodes=2, per_node=4)
        holders = HolderSet(version=1)
        resolver = SourceResolver(max_trainer_readers=2)

        for _round in range(4):
            decisions = resolver.resolve(0, readers, holders, node_of)
            for decision in decisions:
                if decision.kind is SourceKind.TRAINER:
                    holders.mark_held(0, decision.reader)
                elif decision.kind is not None:
                    assert decision.source in holders.servable_holders_of(0), (
                        f"{decision.reader} was served by second-hand holder {decision.source}"
                    )
                    holders.mark_held(0, decision.reader, from_peer=True)


class TestSourcePriority:
    def test_a_same_node_peer_beats_the_trainer(self):
        """A same-node read costs NVLink, not the NIC -- free in the scarce currency."""
        readers, node_of = _cluster(nodes=1, per_node=2)
        holders = HolderSet(version=1)
        holders.mark_held(0, "r0_0")

        resolver = SourceResolver(max_trainer_readers=8)
        decisions = {d.reader: d for d in resolver.resolve(0, readers, holders, node_of)}

        assert decisions["r0_1"].kind is SourceKind.SAME_NODE_PEER
        assert decisions["r0_1"].source == "r0_0"

    def test_cross_node_peers_are_off_in_phase_one(self):
        readers, node_of = _cluster(nodes=2, per_node=1)
        holders = HolderSet(version=1)
        holders.mark_held(0, "r0_0")

        resolver = SourceResolver(max_trainer_readers=0)
        decisions = {d.reader: d for d in resolver.resolve(0, readers, holders, node_of)}

        assert decisions["r1_0"].is_queued, "phase one must queue rather than cross nodes"

    def test_cross_node_peers_are_used_when_enabled(self):
        readers, node_of = _cluster(nodes=2, per_node=1)
        holders = HolderSet(version=1)
        holders.mark_held(0, "r0_0")

        resolver = SourceResolver(max_trainer_readers=0, allow_cross_node_peers=True)
        decisions = {d.reader: d for d in resolver.resolve(0, readers, holders, node_of)}

        assert decisions["r1_0"].kind is SourceKind.CROSS_NODE_PEER
        assert decisions["r1_0"].source == "r0_0"

    def test_a_reader_is_never_its_own_source(self):
        readers, node_of = _cluster(nodes=1, per_node=2)
        holders = HolderSet(version=1)
        holders.mark_held(0, "r0_0")

        resolver = SourceResolver(max_trainer_readers=8)
        for decision in resolver.resolve(0, readers, holders, node_of):
            assert decision.source != decision.reader


class TestTrainerFanoutBound:
    def test_the_trainer_serves_at_most_w_readers(self):
        readers, node_of = _cluster(nodes=8, per_node=1)
        holders = HolderSet(version=1)

        resolver = SourceResolver(max_trainer_readers=3)
        decisions = resolver.resolve(0, readers, holders, node_of)

        from_trainer = [d for d in decisions if d.kind is SourceKind.TRAINER]
        assert len(from_trainer) == 3

    def test_readers_beyond_w_queue_rather_than_oversubscribe(self):
        readers, node_of = _cluster(nodes=8, per_node=1)
        holders = HolderSet(version=1)

        resolver = SourceResolver(max_trainer_readers=3)
        decisions = resolver.resolve(0, readers, holders, node_of)

        assert sum(1 for d in decisions if d.is_queued) == 5

    def test_in_flight_reads_count_against_the_bound(self):
        """A caller resolving in passes must not oversubscribe across them."""
        readers, node_of = _cluster(nodes=4, per_node=1)
        holders = HolderSet(version=1)

        resolver = SourceResolver(max_trainer_readers=3)
        decisions = resolver.resolve(0, readers, holders, node_of, trainer_in_flight=2)

        assert sum(1 for d in decisions if d.kind is SourceKind.TRAINER) == 1

    def test_default_fanout_is_one_reader_per_node(self):
        assert default_trainer_fanout(nodes_per_instance=4, nic_count=8) == 4
        assert default_trainer_fanout(nodes_per_instance=16, nic_count=8) == 8
        assert default_trainer_fanout(nodes_per_instance=0, nic_count=8) == 1


class TestTrafficClaims:
    """D1's actual criteria, counted."""

    @staticmethod
    def _run_to_completion(resolver, readers, node_of, holders, chunk=0, max_rounds=32):
        """Resolve repeatedly until every reader holds the chunk; tally the sources."""
        tally = {kind: 0 for kind in SourceKind}
        for _ in range(max_rounds):
            decisions = resolver.resolve(chunk, readers, holders, node_of)
            if not decisions:
                break
            progressed = False
            for decision in decisions:
                if decision.is_queued:
                    continue
                tally[decision.kind] += 1
                holders.mark_held(
                    chunk,
                    decision.reader,
                    from_peer=decision.kind is not SourceKind.TRAINER,
                )
                progressed = True
            if not progressed:
                raise AssertionError("resolution stalled: every reader queued")
        assert all(holders.holds(chunk, r.name) for r in readers), "some reader never got the chunk"
        return tally

    def test_trainer_emits_one_copy_per_node_not_per_reader(self):
        """The de-duplication claim: trainer egress is M, not N x M."""
        nodes, per_node = 4, 8
        readers, node_of = _cluster(nodes, per_node)
        holders = HolderSet(version=1)
        resolver = SourceResolver(max_trainer_readers=default_trainer_fanout(nodes, nic_count=8))

        tally = self._run_to_completion(resolver, readers, node_of, holders)

        assert tally[SourceKind.TRAINER] == nodes, (
            f"trainer served {tally[SourceKind.TRAINER]} reads for {nodes} nodes; "
            f"serving all {nodes * per_node} readers would be the N x M case"
        )

    def test_every_other_reader_is_served_within_its_node(self):
        nodes, per_node = 4, 8
        readers, node_of = _cluster(nodes, per_node)
        holders = HolderSet(version=1)
        resolver = SourceResolver(max_trainer_readers=nodes)

        tally = self._run_to_completion(resolver, readers, node_of, holders)

        assert tally[SourceKind.SAME_NODE_PEER] == nodes * (per_node - 1)
        assert tally[SourceKind.CROSS_NODE_PEER] == 0, "phase one must not cross nodes"

    def test_trainer_egress_is_flat_as_readers_per_node_grow(self):
        """Sub-linear scaling, stated as the property rather than a timing."""
        egress = {}
        for per_node in (2, 4, 8, 16):
            readers, node_of = _cluster(nodes=4, per_node=per_node)
            holders = HolderSet(version=1)
            resolver = SourceResolver(max_trainer_readers=4)
            tally = self._run_to_completion(resolver, readers, node_of, holders)
            egress[per_node] = tally[SourceKind.TRAINER]

        assert len(set(egress.values())) == 1, (
            f"trainer egress moved with readers per node: {egress}"
        )

    def test_diffusion_completes_in_logarithmic_rounds(self):
        """Each round doubles the holders, so a node of 16 finishes in ~5 rounds."""
        readers, node_of = _cluster(nodes=1, per_node=16)
        holders = HolderSet(version=1)
        resolver = SourceResolver(max_trainer_readers=1)

        rounds = 0
        while not all(holders.holds(0, r.name) for r in readers):
            decisions = resolver.resolve(0, readers, holders, node_of)
            for decision in decisions:
                if not decision.is_queued:
                    holders.mark_held(
                        0, decision.reader, from_peer=decision.kind is not SourceKind.TRAINER
                    )
            rounds += 1
            assert rounds < 20, "diffusion did not converge"

        assert rounds <= 6, f"took {rounds} rounds for 16 readers; expected logarithmic"


class TestHolderRemoval:
    def test_a_forgotten_reader_stops_being_a_source(self):
        """A crashed holder's pages are gone; reading from it would be silent garbage."""
        readers, node_of = _cluster(nodes=1, per_node=3)
        holders = HolderSet(version=1)
        holders.mark_held(0, "r0_0")

        holders.forget("r0_0")

        assert holders.servable_holders_of(0) == set()

    def test_removal_pushes_readers_back_to_the_trainer(self):
        readers, node_of = _cluster(nodes=1, per_node=2)
        holders = HolderSet(version=1)
        holders.mark_held(0, "r0_0")
        holders.forget("r0_0")

        resolver = SourceResolver(max_trainer_readers=4)
        decisions = {d.reader: d for d in resolver.resolve(0, readers, holders, node_of)}

        assert decisions["r0_1"].kind is SourceKind.TRAINER


class TestResolutionSkipsFinishedReaders:
    def test_a_reader_that_already_holds_the_chunk_gets_no_decision(self):
        """Emitting one would double-count its bytes in the traffic tally."""
        readers, node_of = _cluster(nodes=1, per_node=2)
        holders = HolderSet(version=1)
        holders.mark_held(0, "r0_0")

        resolver = SourceResolver(max_trainer_readers=4)
        decisions = resolver.resolve(0, readers, holders, node_of)

        assert [d.reader for d in decisions] == ["r0_1"]
