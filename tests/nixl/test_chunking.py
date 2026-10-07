"""Chunk and commit-group separation (implementation item B1).

The property under test is that the two granularities are genuinely independent:
changing the chunk size must not change which shards form a commit group, and a group
larger than a chunk must still be published atomically. Design 3.6.1 and verification
item 7 both hinge on this -- half a group reaching the engine is a silent no-op, so
refusing to publish is the only safe behaviour.
"""

from __future__ import annotations

import pytest

from psrl.utils.nixl.chunking import (
    CommitGroup,
    GroupCompletionTracker,
    group_shards_by_key,
    plan_chunks,
)

pytestmark = pytest.mark.cpu_test

MB = 1024 * 1024
GB = 1024 * MB


def _group(name: str, *shards) -> CommitGroup:
    return CommitGroup(name=name, shards=tuple(shards))


class TestChunkPartitioning:
    def test_small_shards_pack_into_one_chunk(self):
        groups = [_group("g", ("a", (0,)), ("b", (0,)))]
        sizes = {("a", (0,)): 100, ("b", (0,)): 100}

        chunks, _resolved = plan_chunks(groups, sizes, chunk_bytes=1000)

        assert len(chunks) == 1
        assert chunks[0].nbytes == 200

    def test_shards_spill_into_successive_chunks(self):
        groups = [_group("g", *[(f"t{i}", (0,)) for i in range(5)])]
        sizes = {(f"t{i}", (0,)): 400 for i in range(5)}

        chunks, _resolved = plan_chunks(groups, sizes, chunk_bytes=1000)

        assert len(chunks) == 2
        assert [chunk.nbytes for chunk in chunks] == [1000, 1000]

    def test_oversized_shard_is_split(self):
        """An 11GB fused MoE key must split; this is a requirement, not an edge case."""
        groups = [_group("moe", ("experts.w13", (0,)))]
        sizes = {("experts.w13", (0,)): 11 * GB}

        chunks, resolved = plan_chunks(groups, sizes, chunk_bytes=512 * MB)

        assert len(chunks) == 22
        assert sum(chunk.nbytes for chunk in chunks) == 11 * GB
        assert len(resolved[0].chunk_indices) == 22, "the group depends on every chunk"

    def test_split_pieces_carry_their_offsets(self):
        """Without offsets the second piece would overwrite the first."""
        groups = [_group("g", ("big", (0,)))]
        sizes = {("big", (0,)): 2500}

        chunks, _resolved = plan_chunks(groups, sizes, chunk_bytes=1000)

        offsets = [piece.offset for chunk in chunks for piece in chunk.pieces]
        assert offsets == [0, 1000, 2000]
        assert [piece.nbytes for chunk in chunks for piece in chunk.pieces] == [1000, 1000, 500]
        assert chunks[0].pieces[0].is_whole_shard
        assert not chunks[1].pieces[0].is_whole_shard

    def test_zero_sized_shards_are_skipped(self):
        """Pipeline stages omit keys, which arrive as empty shards."""
        groups = [_group("g", ("present", (0,)), ("absent", (0,)))]
        sizes = {("present", (0,)): 500, ("absent", (0,)): 0}

        chunks, _resolved = plan_chunks(groups, sizes, chunk_bytes=1000)

        assert len(chunks) == 1
        assert [piece.shard[0] for piece in chunks[0].pieces] == ["present"]

    def test_missing_size_is_an_error(self):
        groups = [_group("g", ("unknown", (0,)))]
        with pytest.raises(KeyError, match="no byte size recorded"):
            plan_chunks(groups, {}, chunk_bytes=1000)

    def test_nonpositive_chunk_size_is_rejected(self):
        with pytest.raises(ValueError, match="chunk_bytes must be positive"):
            plan_chunks([], {}, chunk_bytes=0)


class TestGranularityIndependence:
    """The point of the split: chunk size must not affect group membership."""

    @pytest.mark.parametrize("chunk_bytes", [256, 1000, 4096, 1 << 20])
    def test_group_membership_is_chunk_size_invariant(self, chunk_bytes):
        groups = [
            _group("qkv", ("q", (0,)), ("k", (0,)), ("v", (0,))),
            _group("mlp", ("gate", (0,)), ("up", (0,))),
        ]
        sizes = {shard: 700 for group in groups for shard in group.shards}

        _chunks, resolved = plan_chunks(groups, sizes, chunk_bytes=chunk_bytes)

        assert [group.shards for group in resolved] == [group.shards for group in groups]

    def test_one_chunk_can_hold_several_groups(self):
        groups = [_group("a", ("a1", (0,))), _group("b", ("b1", (0,)))]
        sizes = {("a1", (0,)): 100, ("b1", (0,)): 100}

        chunks, resolved = plan_chunks(groups, sizes, chunk_bytes=1000)

        assert len(chunks) == 1
        assert all(group.chunk_indices == (0,) for group in resolved)

    def test_one_group_can_span_several_chunks(self):
        groups = [_group("fused", ("w13", (0,)), ("w2", (0,)))]
        sizes = {("w13", (0,)): 1500, ("w2", (0,)): 1500}

        chunks, resolved = plan_chunks(groups, sizes, chunk_bytes=1000)

        assert len(chunks) == 3
        assert resolved[0].chunk_indices == (0, 1, 2)

    def test_chunk_indices_are_stable_across_runs(self):
        """Offset staggering predicts who holds which chunk, so indices must be stable."""
        groups = [_group(f"g{i}", (f"t{i}", (0,))) for i in range(6)]
        sizes = {(f"t{i}", (0,)): 400 for i in range(6)}

        first = plan_chunks(groups, sizes, chunk_bytes=1000)[1]
        second = plan_chunks(groups, sizes, chunk_bytes=1000)[1]

        assert [group.chunk_indices for group in first] == [group.chunk_indices for group in second]


class TestAtomicVisibility:
    """Verification item 7: half a group must raise, not publish."""

    @staticmethod
    def _tracker(chunk_bytes: int):
        groups = [
            _group("qkv", ("q", (0,)), ("k", (0,)), ("v", (0,))),
            _group("mlp", ("gate", (0,))),
        ]
        sizes = {("q", (0,)): 600, ("k", (0,)): 600, ("v", (0,)): 600, ("gate", (0,)): 600}
        chunks, resolved = plan_chunks(groups, sizes, chunk_bytes=chunk_bytes)
        return GroupCompletionTracker(resolved), chunks, resolved

    def test_group_completes_only_when_every_chunk_arrives(self):
        tracker, chunks, resolved = self._tracker(chunk_bytes=1000)
        qkv = next(group for group in resolved if group.name == "qkv")
        assert len(qkv.chunk_indices) > 1, "test needs a multi-chunk group"

        for chunk_index in qkv.chunk_indices[:-1]:
            completed = tracker.mark_chunk_complete(chunk_index)
            assert "qkv" not in completed, "group completed before all its chunks arrived"
            assert not tracker.is_complete("qkv")

        completed = tracker.mark_chunk_complete(qkv.chunk_indices[-1])
        assert "qkv" in completed
        assert tracker.is_complete("qkv")

    def test_partial_delivery_refuses_to_publish(self):
        """The assertion verification item 7 calls for."""
        tracker, chunks, _resolved = self._tracker(chunk_bytes=1000)
        tracker.mark_chunk_complete(0)

        with pytest.raises(AssertionError, match="must not be published"):
            tracker.assert_all_complete()

    def test_error_names_the_missing_chunks(self):
        """A count alone would not tell an operator what to retry."""
        tracker, chunks, _resolved = self._tracker(chunk_bytes=1000)
        tracker.mark_chunk_complete(0)

        with pytest.raises(AssertionError) as excinfo:
            tracker.assert_all_complete()
        message = str(excinfo.value)
        assert "missing chunks" in message
        assert "qkv" in message or "mlp" in message

    def test_full_delivery_passes(self):
        tracker, chunks, _resolved = self._tracker(chunk_bytes=1000)
        for chunk in chunks:
            tracker.mark_chunk_complete(chunk.index)
        tracker.assert_all_complete()

    def test_pending_groups_reports_what_is_missing(self):
        tracker, chunks, resolved = self._tracker(chunk_bytes=1000)
        qkv = next(group for group in resolved if group.name == "qkv")
        tracker.mark_chunk_complete(qkv.chunk_indices[0])

        pending = tracker.pending_groups()
        assert "qkv" in pending
        assert qkv.chunk_indices[0] not in pending["qkv"]

    def test_duplicate_chunk_arrival_is_idempotent(self):
        """A retried chunk must not look like progress on a second group."""
        tracker, _chunks, resolved = self._tracker(chunk_bytes=1000)
        qkv = next(group for group in resolved if group.name == "qkv")

        first = tracker.mark_chunk_complete(qkv.chunk_indices[0])
        second = tracker.mark_chunk_complete(qkv.chunk_indices[0])

        assert first == second == [] or first == second
        assert not tracker.is_complete("qkv") or len(qkv.chunk_indices) == 1

    def test_shared_chunk_can_complete_two_groups_at_once(self):
        """Small groups share a chunk, so one arrival may finish several."""
        groups = [_group("a", ("a1", (0,))), _group("b", ("b1", (0,)))]
        sizes = {("a1", (0,)): 100, ("b1", (0,)): 100}
        _chunks, resolved = plan_chunks(groups, sizes, chunk_bytes=1000)
        tracker = GroupCompletionTracker(resolved)

        completed = tracker.mark_chunk_complete(0)

        assert sorted(completed) == ["a", "b"]


class TestShardGrouping:
    def test_groups_by_key_preserving_order(self):
        shards = [("b", (0,)), ("a", (0,)), ("b", (1,))]
        grouped = group_shards_by_key(shards)
        assert list(grouped) == ["b", "a"]
        assert grouped["b"] == [("b", (0,)), ("b", (1,))]
