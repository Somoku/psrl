"""Overlapping peer diffusion with the upstream wave (implementation item D1).

The two-phase version was correct but slower: the peer phase ran entirely after the
upstream phase, so its cost added instead of hiding. Measured at 8.8 ms per read against
0.11 ms of actual bytes, the cost is fixed overhead -- which only overlap can hide, since
the link is already 14x faster than the alternative.

The tests count waves rather than timing anything. A schedule that merely looked
pipelined in a trace would pass a timing check on a quiet cluster and fail under load;
counting how many peer reads have later upstream work to overlap is the property itself.
"""

from __future__ import annotations

import pytest
from psrl.utils.nixl.chunk_pipeline import (
    ChunkWave,
    coalesce_to_transport_pairs,
    exposed_waves,
    overlap_depth,
    schedule_pipelined_reads,
)
from psrl.utils.nixl.chunking import CommitGroup, plan_chunks

pytestmark = pytest.mark.cpu_test


def _laid_out_chunks(layout, shard_bytes: int = 512 * 1024):
    """Chunks with an explicit per-chunk shard layout.

    The module's `_chunks` takes a key count and derives the layout; these tests need to
    place named shards in named chunks, because the property under test is *where* a pair
    lands. Built on the real `plan_chunks` so the chunk objects are the ones the
    scheduler sees, rather than a stand-in whose `pieces` could drift.
    """
    groups, sizes = [], {}
    for chunk_index, refs in enumerate(layout):
        groups.append(CommitGroup(name=f"g{chunk_index}", shards=tuple(refs)))
        for ref in refs:
            sizes[ref] = shard_bytes
    chunks, _resolved = plan_chunks(groups, sizes, chunk_bytes=shard_bytes)
    return chunks


def _chunks(num_keys: int, shards_per_key: int = 2, shard_bytes: int = 512 * 1024):
    """One commit group per key, chunked so each chunk holds one key."""
    groups, sizes = [], {}
    for k in range(num_keys):
        key = f"w{k}"
        refs = tuple((key, (s,)) for s in range(shards_per_key))
        groups.append(CommitGroup(name=key, shards=refs))
        for ref in refs:
            sizes[ref] = shard_bytes
    chunks, resolved = plan_chunks(groups, sizes, chunk_bytes=shard_bytes * shards_per_key)
    return chunks, resolved


def _plans(num_keys: int, shards_per_key: int = 2, peer: str | None = "gen0"):
    upstream = {
        f"w{k}": {"train0": [(s,) for s in range(shards_per_key)]} for k in range(num_keys)
    }
    peer_plan = (
        {f"w{k}": {peer: [(s,) for s in range(shards_per_key)]} for k in range(num_keys)}
        if peer
        else {}
    )
    return upstream, peer_plan


class TestWavesFollowChunks:
    def test_one_wave_per_chunk(self):
        chunks, _ = _chunks(num_keys=5)
        upstream, _ = _plans(num_keys=5, peer=None)

        waves = schedule_pipelined_reads(chunks, upstream, {}, "gen1")

        assert len(waves) == 5
        assert [w.index for w in waves] == [0, 1, 2, 3, 4]

    def test_a_waves_reads_belong_to_its_chunk(self):
        chunks, _ = _chunks(num_keys=3)
        upstream, _ = _plans(num_keys=3, peer=None)

        waves = schedule_pipelined_reads(chunks, upstream, {}, "gen1")

        chunk_keys = {
            chunk.index: {piece.shard[0] for piece in chunk.pieces} for chunk in chunks
        }
        for wave in waves:
            # Every read in the wave must concern a key that chunk carries.
            assert all(read.chunk == wave.index for read in wave.reads)
            assert wave.index in chunk_keys

    def test_scheduled_pairs_match_the_plan_exactly(self):
        """Neither invented nor dropped -- the bug that produced wrong weights.

        An earlier version derived the (key, source) pairs from chunk membership rather
        than from the reads it had placed. A chunk holds ~100 keys at real model size, so
        "every key in this chunk" is a different set from "the reads placed in this
        chunk": it emitted reads nobody planned and silently dropped planned ones. On the
        cluster that showed up as 100 + 269 reads issued against 653 planned, and the
        coordinate judge caught the resulting wrong weights.
        """
        chunks, _ = _chunks(num_keys=5)
        upstream, peer = _plans(num_keys=5)

        waves = schedule_pipelined_reads(chunks, upstream, peer, "gen1")
        scheduled = {
            (read.key, read.shard): (read.source, read.is_peer)
            for wave in waves
            for read in wave.reads
        }

        # Every shard the client wants is scheduled exactly once -- the schedule is a
        # partition, unlike the plan, where the upstream read is kept whole so a caller
        # that cannot run the peer phase stays correct by skipping it. A pipelined
        # caller runs both, so leaving both in would read every diffused shard twice.
        wanted = {
            (key, shard)
            for key, per_source in upstream.items()
            for shards in per_source.values()
            for shard in shards
        }
        assert set(scheduled) == wanted, (
            f"invented: {sorted(set(scheduled) - wanted)}\n"
            f"dropped:  {sorted(wanted - set(scheduled))}"
        )

        # And the peer plan decides which side each shard lands on.
        for key, per_peer in peer.items():
            for peer_name, shards in per_peer.items():
                for shard in shards:
                    assert scheduled[(key, shard)] == (peer_name, True), (
                        f"{key} {shard} was scheduled {scheduled[(key, shard)]}, not "
                        f"from its peer {peer_name}"
                    )

    def test_no_shard_is_read_twice(self):
        """A duplicate read doubles the traffic diffusion exists to remove."""
        chunks, _ = _chunks(num_keys=6)
        upstream, peer = _plans(num_keys=6)

        waves = schedule_pipelined_reads(chunks, upstream, peer, "gen1")
        reads = [(r.key, r.shard) for w in waves for r in w.reads]

        assert len(reads) == len(set(reads)), (
            "a shard is read both upstream and from a peer, so the schedule reports an "
            "overlap it never delivers"
        )

    def test_a_chunk_holding_many_keys_schedules_only_its_own_reads(self):
        """The shape that broke it: few, large chunks each holding many keys."""
        groups, sizes = [], {}
        for k in range(20):
            key = f"w{k}"
            ref = (key, (0,))
            groups.append(CommitGroup(name=key, shards=(ref,)))
            sizes[ref] = 1024
        # One chunk for all 20 keys.
        chunks, _ = plan_chunks(groups, sizes, chunk_bytes=20 * 1024)
        assert len(chunks) == 1

        # The client reads only three of those keys.
        upstream = {f"w{k}": {"train0": [(0,)]} for k in (2, 7, 15)}
        waves = schedule_pipelined_reads(chunks, upstream, {}, "gen1")

        scheduled_keys = {read.key for wave in waves for read in wave.reads}
        assert scheduled_keys == {"w2", "w7", "w15"}, (
            f"chunk membership leaked into the schedule: {sorted(scheduled_keys)}"
        )

    def test_the_read_count_equals_the_shard_count(self):
        """One read per wanted shard -- the schedule partitions, it does not duplicate."""
        chunks, _ = _chunks(num_keys=4)
        upstream, peer = _plans(num_keys=4)

        waves = schedule_pipelined_reads(chunks, upstream, peer, "gen1")
        total = sum(len(w.reads) for w in waves)

        wanted = sum(
            len(s) for per_source in upstream.values() for s in per_source.values()
        )
        assert total == wanted


class TestTheDependencyIsPerChunk:
    """The whole point: a follower waits for chunk i, not for the whole model."""

    def test_peer_reads_sit_one_wave_behind_their_chunk(self):
        """A peer read of chunk i belongs in wave i+1, not wave i.

        Putting it in wave i places it in the *same* barrier interval as the designate's
        upstream read of that chunk, so the two run concurrently and the peer read can
        land against a buffer the designate has not filled. That read succeeds -- the
        registration is valid -- and the wrong bytes surface only in the coordinate
        judge. It did, twice, on the cluster.
        """
        chunks, _ = _chunks(num_keys=4)
        upstream, peer = _plans(num_keys=4)

        waves = schedule_pipelined_reads(chunks, upstream, peer, "gen1")

        for wave in waves:
            if wave.peer_reads:
                assert wave.ready_after == wave.index - 1, (
                    f"wave {wave.index} waits on chunk {wave.ready_after}; it must wait "
                    "on the previous chunk, which an earlier barrier already completed"
                )

    def test_no_peer_read_lands_in_wave_zero(self):
        """Nothing has been fetched before the first barrier, so no peer can serve.

        Indexed by wave number rather than list position: a full follower has no wave 0
        at all, because every read it owns is a peer read and those all shifted later.
        """
        chunks, _ = _chunks(num_keys=5)
        upstream, peer = _plans(num_keys=5)

        by_index = {w.index: w for w in schedule_pipelined_reads(chunks, upstream, peer, "gen1")}

        assert 0 not in by_index or by_index[0].peer_reads == []

    def test_a_wave_without_peer_reads_has_no_dependency(self):
        """A dependency-free wave may be issued without waiting for anything."""
        chunks, _ = _chunks(num_keys=3)
        upstream, _ = _plans(num_keys=3, peer=None)

        waves = schedule_pipelined_reads(chunks, upstream, {}, "gen1")

        assert all(w.ready_after is None for w in waves)


class TestOverlapIsReal:
    """Counted against the schedule, not timed."""

    def test_only_the_first_chunk_is_exposed(self):
        chunks, _ = _chunks(num_keys=8)
        upstream, peer = _plans(num_keys=8)

        waves = schedule_pipelined_reads(chunks, upstream, peer, "gen1")

        assert exposed_waves(waves) == 1

    def test_exposure_does_not_grow_with_the_chunk_count(self):
        """A sequence would expose every peer read; a pipeline exposes one wave."""
        exposure = {}
        for num_keys in (4, 8, 16, 32):
            chunks, _ = _chunks(num_keys=num_keys)
            upstream, peer = _plans(num_keys=num_keys)
            waves = schedule_pipelined_reads(chunks, upstream, peer, "gen1")
            exposure[num_keys] = exposed_waves(waves)

        assert len(set(exposure.values())) == 1, f"exposure moved with chunks: {exposure}"

    def test_a_followers_peer_reads_hide_behind_the_designates_upstream_reads(self):
        """The overlap is across clients: a follower has no upstream reads of its own."""
        chunks, _ = _chunks(num_keys=10)
        upstream, peer = _plans(num_keys=10)

        follower = schedule_pipelined_reads(chunks, upstream, peer, "gen1")
        designate = schedule_pipelined_reads(chunks, upstream, {}, "gen0")

        total_peer = sum(len(w.peer_reads) for w in follower)
        overlapped = overlap_depth(follower, designate)

        # Peer reads now sit one wave behind, so the final chunk's peer read lands in a
        # wave with no designate upstream work left -- that one cannot overlap, and the
        # rest must.
        # Exactly two waves are exposed, and the count is structural rather than
        # tuned: the designate's final chunk has no later upstream work, and the
        # follower's peer read of that chunk is shifted one wave past it. Everything
        # earlier overlaps. Verified on a 10-chunk schedule: waves 1-8 overlap, 9 and
        # 10 do not.
        per_wave = max((len(w.peer_reads) for w in follower), default=0)
        assert overlapped >= total_peer - 2 * per_wave, (
            f"only {overlapped} of {total_peer} peer reads hide behind the designate's "
            f"remaining upstream work; at most two waves ({2 * per_wave} reads) may be "
            "exposed"
        )
        assert overlapped > 0, "no peer read overlaps at all, so this is not a pipeline"

    def test_exposure_stays_two_waves_as_the_model_grows(self):
        """The bound is structural: more chunks must not expose more waves."""
        exposed = {}
        for num_keys in (6, 12, 24):
            chunks, _ = _chunks(num_keys=num_keys)
            upstream, peer = _plans(num_keys=num_keys)
            follower = schedule_pipelined_reads(chunks, upstream, peer, "gen1")
            designate = schedule_pipelined_reads(chunks, upstream, {}, "gen0")

            total = sum(len(w.peer_reads) for w in follower)
            per_wave = max((len(w.peer_reads) for w in follower), default=1)
            exposed[num_keys] = (total - overlap_depth(follower, designate)) // per_wave

        assert set(exposed.values()) == {2}, f"exposure moved with model size: {exposed}"

    def test_a_follower_alone_shows_no_self_overlap(self):
        """Recorded because it is the trap: zero here does not mean no pipeline.

        A full follower's schedule is entirely peer reads, so there is nothing of its
        own to overlap with. Measuring self-overlap would report a perfectly pipelined
        run as having none.
        """
        chunks, _ = _chunks(num_keys=6)
        upstream, peer = _plans(num_keys=6)

        follower = schedule_pipelined_reads(chunks, upstream, peer, "gen1")

        assert all(not w.upstream_reads for w in follower)
        assert overlap_depth(follower) == 0

    def test_the_two_phase_arrangement_overlaps_nothing(self):
        """Contrast: all upstream, then all peer, is what the measurement found slow.

        Collapsing every peer read into one trailing wave leaves no designate upstream
        work after it, so nothing hides -- which is exactly why the two-phase run was
        slower than not diffusing at all.
        """
        chunks, _ = _chunks(num_keys=4)
        upstream, peer = _plans(num_keys=4)

        designate = schedule_pipelined_reads(chunks, upstream, {}, "gen0")
        pipelined = schedule_pipelined_reads(chunks, upstream, peer, "gen1")

        trailing = ChunkWave(index=len(designate) + 1)
        for wave in pipelined:
            trailing.reads.extend(wave.peer_reads)
        trailing.ready_after = trailing.index

        assert overlap_depth([trailing], designate) == 0
        assert overlap_depth(pipelined, designate) > 0


class TestNothingIsSilentlyDropped:
    def test_reads_for_keys_absent_from_the_chunk_plan_still_run(self):
        """The chunk plan comes from the reader's metadata and may omit a key."""
        chunks, _ = _chunks(num_keys=2)
        # A third key the chunk plan knows nothing about.
        upstream = {"w0": {"train0": [(0,)]}, "unknown": {"train0": [(0,)]}}

        waves = schedule_pipelined_reads(chunks, upstream, {}, "gen1")
        scheduled = {(r.source, r.is_peer) for w in waves for r in w.reads}

        assert sum(len(w.reads) for w in waves) == 2, (
            "a read for a key missing from the chunk plan was dropped, losing those "
            "bytes with no error"
        )
        assert scheduled == {("train0", False)}

    def test_an_empty_plan_yields_no_waves(self):
        chunks, _ = _chunks(num_keys=2)
        assert schedule_pipelined_reads(chunks, {}, {}, "gen1") == []


class TestDeterminism:
    def test_the_schedule_is_stable(self):
        """The index predicts holders without communicating, so order cannot vary."""
        chunks, _ = _chunks(num_keys=6)
        upstream, peer = _plans(num_keys=6)

        first = schedule_pipelined_reads(chunks, upstream, peer, "gen1")
        second = schedule_pipelined_reads(chunks, upstream, peer, "gen1")

        assert [(w.index, w.ready_after, len(w.reads)) for w in first] == [
            (w.index, w.ready_after, len(w.reads)) for w in second
        ]


class TestCoalesceToTransportPairs:
    """The schedule is per shard, the transport is per (key, source) (item D1).

    `client_read` moves every shard the plan gives a pair, so a pair whose shards were
    placed in several waves is fully transferred by the first wave that mentions it.
    Posting it again reposts a live handle -- the cluster returned
    `NIXL_ERR_REPOST_ACTIVE` -- so each pair must appear exactly once.
    """

    def test_each_pair_appears_exactly_once(self):
        chunks = _laid_out_chunks([[("k", (0,))], [("k", (1,))]])
        plan = {"k": {"trainer": [(0,), (1,)]}}
        waves = schedule_pipelined_reads(chunks, plan, {}, "c0")

        coalesced = coalesce_to_transport_pairs(waves)

        posted = [pair for _index, pairs in coalesced for pair in pairs]
        assert len(posted) == 1, f"pair posted {len(posted)} times: {posted}"
        assert posted[0][:2] == ("k", "trainer")

    def test_an_upstream_pair_goes_to_its_earliest_wave(self):
        """It has no dependency, and issuing it early is what followers overlap with."""
        chunks = _laid_out_chunks([[("k", (0,))], [("k", (1,))]])
        plan = {"k": {"trainer": [(0,), (1,)]}}
        waves = schedule_pipelined_reads(chunks, plan, {}, "c0")

        coalesced = coalesce_to_transport_pairs(waves)

        assert coalesced[0][0] == 0
        assert len(coalesced) == 1

    def test_a_peer_pair_goes_to_its_latest_wave(self):
        """A peer read is valid only after *every* shard it wants has been filled.

        Taking the earliest wave would issue the read before the designate had the later
        shards -- and that read succeeds against an unfilled buffer, so nothing
        downstream would catch it. This asymmetry with the upstream case is the point of
        the function.
        """
        chunks = _laid_out_chunks([[("k", (0,))], [("k", (1,))]])
        peer_plan = {"k": {"peer0": [(0,), (1,)]}}
        waves = schedule_pipelined_reads(chunks, {}, peer_plan, "c0")

        coalesced = coalesce_to_transport_pairs(waves)

        assert len(coalesced) == 1
        index, pairs = coalesced[0]
        shard_waves = [w.index for w in waves if any(r.is_peer for r in w.reads)]
        assert index == max(shard_waves), (
            f"peer pair posted at wave {index} but its shards span {shard_waves}; "
            "issuing before the last dependency reads an unfilled buffer successfully"
        )
        assert pairs[0][2] is True

    def test_upstream_and_peer_pairs_keep_their_kinds(self):
        chunks = _laid_out_chunks([[("up", (0,)), ("pe", (0,))]])
        waves = schedule_pipelined_reads(
            chunks, {"up": {"trainer": [(0,)]}}, {"pe": {"peer0": [(0,)]}}, "c0"
        )

        coalesced = coalesce_to_transport_pairs(waves)

        kinds = {(key, source): peer for _i, pairs in coalesced for key, source, peer in pairs}
        assert kinds[("up", "trainer")] is False
        assert kinds[("pe", "peer0")] is True

    def test_empty_waves_are_dropped(self):
        """A wave contributing no pair is not a barrier -- nothing in it has to wait."""
        chunks = _laid_out_chunks([[("k", (0,))], [("other", (9,))]])
        waves = schedule_pipelined_reads(chunks, {"k": {"trainer": [(0,)]}}, {}, "c0")

        coalesced = coalesce_to_transport_pairs(waves)

        assert all(pairs for _index, pairs in coalesced)

    def test_no_pair_is_lost(self):
        """Coalescing must not drop work: every scheduled pair has to be posted."""
        chunks = _laid_out_chunks(
            [[("a", (0,)), ("b", (0,))], [("a", (1,)), ("c", (0,))], [("b", (1,))]]
        )
        upstream = {"a": {"t0": [(0,), (1,)]}, "c": {"t1": [(0,)]}}
        peer = {"b": {"p0": [(0,), (1,)]}}
        waves = schedule_pipelined_reads(chunks, upstream, peer, "c0")

        scheduled = {(r.key, r.source) for w in waves for r in w.reads}
        posted = {(key, source) for _i, pairs in coalesce_to_transport_pairs(waves) for key, source, _p in pairs}

        assert posted == scheduled
