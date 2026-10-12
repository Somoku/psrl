"""E1's L2 criterion: the tear is real, and the barrier is what closes it.

The other E1 suites test the parts in isolation -- `test_inflight_reads` pins the counter,
`test_version_barrier` pins the drain policy and which configurations need it. Both would
pass if the hazard they exist for did not exist at all. This file is the one that shows it
does: with the barrier off, a reader mid-pull collects a tensor that is partly ``v`` and
partly ``v+1``; with it on, the same interleaving yields neither.

**Why a chunked copy is a faithful model of the pull.** A NIXL pull of one key issues one
transfer per shard descriptor, so a read is a *sequence* of copies, not an instant. The
optimizer writes the same parameter memory in place. That is the whole hazard: the reader
is only partway through when the bytes under it change. Reducing it to a chunked copy
against an in-place overwrite keeps the property that matters -- non-atomic read, in-place
write -- and drops only the transport.

**Why the oracle is a per-chunk version tag.** This is the coordinate judge's idea at CPU
scale: every chunk carries the version that wrote it, so a tear is visible as more than one
distinct version in the result. It has to be content, not metadata -- `check_pull_versions`
reads the stamp written at publish time, so a shard whose *bytes* were half-overwritten
still reports its own version and passes. That gap is stated in `shard_version`'s docstring
and it is precisely the gap this file covers.

**Why the interleavings are choreographed rather than raced.** The corruption needs the
write to land inside the read window. Left to the scheduler that is rare and the test
would pass for the wrong reason almost every run -- a flaky test asserting a race it did
not actually produce. Each test below forces the interleaving with events, so the ones
that claim a tear produce it every time, and the one that claims the barrier prevents it
had a tear available to prevent.

**What this does not establish.** That the FSDP2 optimizer step and a real NIXL pull
overlap on real hardware. That is a claim about timing on a cluster, and it is why the
barrier's necessity is decided from the configuration (`requires_version_barrier`) rather
than measured. What is established here is that *if* they overlap, the result is a torn
read, and that the barrier removes it.
"""

from __future__ import annotations

import threading
import time

import pytest
from psrl.workers.ps.ps_manager import PSManager
from psrl.workers.ps.version_barrier import PublishOutcome, VersionBarrier

pytestmark = pytest.mark.cpu_test


#: Enough chunks that a tear lands strictly inside the read rather than at its edge.
CHUNKS = 8
#: Generous: a budget that can expire mid-test would make a drain look like an expiry.
AMPLE_BUDGET_MS = 5000.0
#: Holds a read open long enough for a publish to find it, the way a transfer does.
TRANSFER_PER_CHUNK_S = 0.001


class _Registry:
    """The manager's reader registry, with its real lock -- these tests are threaded.

    `test_inflight_reads` bypasses the decorator because it tests arithmetic in one
    thread. Here the publisher polls from one thread while a reader ends from another,
    which is the configuration the lock exists for, so the decorated methods are used.
    """

    def __init__(self):
        self._inflight_reads = {}
        self._state_lock = threading.Lock()

    begin_read = PSManager.begin_read
    end_read = PSManager.end_read
    count_readers = PSManager.count_readers


class _ParameterMemory:
    """Parameter bytes an optimizer overwrites in place and a rollout reads in chunks.

    Every chunk holds the version that last wrote it, so the reader's copy carries its own
    provenance and a tear needs no separate bookkeeping to detect.
    """

    def __init__(self, version: int):
        self.chunks = [version] * CHUNKS

    def overwrite(self, version: int) -> None:
        """What the optimizer step does: in place, over the bytes a reader may hold."""
        for i in range(CHUNKS):
            self.chunks[i] = version

    def read_chunk(self, index: int) -> int:
        return self.chunks[index]


def _versions_in(copy: list[int]) -> set[int]:
    return set(copy)


def _is_torn(copy: list[int]) -> bool:
    """A copy is torn when it holds bytes from more than one version."""
    return len(_versions_in(copy)) > 1


class TestTheHazardIsReal:
    """Without a barrier, an overwrite inside the read window tears the copy."""

    def test_an_overwrite_mid_read_produces_a_tensor_from_no_single_version(self):
        """The result is not stale and not fresh -- it is a policy that never existed.

        This is the distinction that makes the tear worse than staleness: an off-policy
        correction can account for weights from an older step, but there is no step whose
        weights these are.
        """
        memory = _ParameterMemory(version=7)
        copy: list[int] = []
        first_chunk_read = threading.Event()
        overwritten = threading.Event()

        def reader():
            copy.append(memory.read_chunk(0))
            first_chunk_read.set()
            # Without a barrier nothing makes the writer wait, so the overwrite lands
            # here. Waiting for it is what makes the tear happen on every run instead of
            # on the unlucky ones.
            overwritten.wait(timeout=5)
            for i in range(1, CHUNKS):
                copy.append(memory.read_chunk(i))

        def writer():
            first_chunk_read.wait(timeout=5)
            memory.overwrite(version=8)
            overwritten.set()

        threads = [threading.Thread(target=reader), threading.Thread(target=writer)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        assert _is_torn(copy)
        assert _versions_in(copy) == {7, 8}
        assert copy[0] == 7, "the chunk read before the overwrite must be the old version"
        assert copy[-1] == 8, "the chunk read after it must be the new one"

    def test_the_version_stamp_alone_would_not_catch_it(self):
        """Why this file reads content: the stamp is written at publish, not at read.

        A shard whose bytes were half-overwritten still reports the version it was
        published at, so a metadata-only check sees a clean pull. `check_pull_versions`
        catches tears *across* shards and stale reads; this one is inside a shard.
        """
        from psrl.utils.nixl.shard_version import check_pull_versions

        # The reader pulled shard (0,) of "w" and the supplier reports v7 for it -- true,
        # and useless, because the bytes behind it were rewritten during the transfer.
        published = {"train_0": {"w": {(0,): 7}}}

        assert check_pull_versions(7, published) == []


class TestTheBarrierClosesIt:
    """The publisher drains readers first, and the copy survives.

    **The barrier inverts who waits for whom**, which is why the two tests here are
    choreographed differently rather than sharing one sequence. Without it the reader is
    at the writer's mercy, so the reader can safely wait for the overwrite to land -- it
    is going to land regardless. With it the writer is the one that waits, so a reader
    waiting for the overwrite would be waiting on a publish that is itself waiting for
    the reader, and the test would deadlock instead of exercising the drain.

    So the barrier-on reader waits only for the writer to *enter* the publish. That is
    the strongest gate available that does not assume the conclusion: at that moment the
    writer is committed to overwriting, and the only thing between it and the reader's
    bytes is the barrier.
    """

    def test_the_publish_waits_and_the_copy_is_whole(self):
        registry = _Registry()
        barrier = VersionBarrier(
            count_readers=registry.count_readers, budget_ms=AMPLE_BUDGET_MS
        )
        memory = _ParameterMemory(version=7)
        copy: list[int] = []
        first_chunk_read = threading.Event()
        writer_engaged = threading.Event()
        outcome: list = []

        def reader():
            registry.begin_read("rollout_0", 7)
            try:
                copy.append(memory.read_chunk(0))
                first_chunk_read.set()
                # Resume only once the writer is inside the publish, so the remaining
                # chunks are read with an overwrite already pending against them.
                writer_engaged.wait(timeout=5)
                for i in range(1, CHUNKS):
                    copy.append(memory.read_chunk(i))
            finally:
                registry.end_read("rollout_0", 7)

        def writer():
            first_chunk_read.wait(timeout=5)
            writer_engaged.set()
            outcome.append(barrier.publish(previous_version=7))
            memory.overwrite(version=8)

        threads = [threading.Thread(target=reader), threading.Thread(target=writer)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert not _is_torn(copy)
        assert _versions_in(copy) == {7}
        assert outcome[0].outcome is PublishOutcome.DRAINED, (
            "NO_READERS would mean the reader had already finished and the barrier never "
            "had a tear to prevent"
        )
        assert outcome[0].previous_version_valid

    def test_reporting_the_read_without_draining_it_changes_nothing(self):
        """The control: the same registry calls, with nothing polling them.

        Without this, "the copy was whole" could be an artifact of the reader simply
        being quick. Here the reader announces its read exactly as above and the copy
        still tears, which places the protection in the drain rather than in the
        bookkeeping around it.
        """
        registry = _Registry()
        memory = _ParameterMemory(version=7)
        copy: list[int] = []
        first_chunk_read = threading.Event()
        overwritten = threading.Event()

        def reader():
            registry.begin_read("rollout_0", 7)
            try:
                copy.append(memory.read_chunk(0))
                first_chunk_read.set()
                overwritten.wait(timeout=5)
                for i in range(1, CHUNKS):
                    copy.append(memory.read_chunk(i))
            finally:
                registry.end_read("rollout_0", 7)

        def writer():
            first_chunk_read.wait(timeout=5)
            memory.overwrite(version=8)
            overwritten.set()

        threads = [threading.Thread(target=reader), threading.Thread(target=writer)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert registry.count_readers(7) == 0, "the read was reported, start and end"
        assert _is_torn(copy)


class TestWhatHappensWhenTheBudgetIsNotEnough:
    """Correctness comes from invalidation, not from waiting -- so check the invalidation.

    A reader slower than the budget is the case the design accepts a tear in: the publish
    proceeds, and the reader is required to find out and discard. If it is not told, this
    is exactly the silent corruption the barrier was added to prevent, now with a barrier
    in the way making it look handled.
    """

    def test_a_reader_past_the_budget_is_told_its_version_is_gone(self):
        registry = _Registry()
        barrier = VersionBarrier(count_readers=registry.count_readers, budget_ms=5.0)
        registry.begin_read("rollout_0", 7)

        report = barrier.publish(previous_version=7)

        assert report.outcome is PublishOutcome.INVALIDATED
        assert not report.previous_version_valid
        assert barrier.is_invalidated(7), (
            "a reader still on v7 must be able to discover its bytes were overwritten; "
            "without this the torn read is returned as a successful pull"
        )

    def test_a_version_that_drained_is_never_invalidated(self):
        """Invalidating on a clean drain would discard good pulls on every publish."""
        registry = _Registry()
        barrier = VersionBarrier(
            count_readers=registry.count_readers, budget_ms=AMPLE_BUDGET_MS
        )

        barrier.publish(previous_version=7)

        assert not barrier.is_invalidated(7)


class TestItHoldsOverRepeatedPublishes:
    """Many rounds with the scheduler left to interleave them: never a tear.

    Only the barrier-on direction is asserted. "No tear with the barrier" cannot pass by
    luck -- one tear in any iteration fails it. Asserting the barrier-off direction here
    would be the flaky inverse, needing the scheduler to land a write inside a read,
    which is exactly what the choreographed tests above remove the need for.

    Two things keep it from going quietly vacuous, and the outcome assertion below is
    what caught their absence. The writer waits for the read to be registered; and the
    reader sleeps a little per chunk, because eight list reads finish long before the
    writer wakes, so every round reported `NO_READERS` and the drain never ran. The sleep
    is not a fudge for a race -- a transfer genuinely occupies the window, and without it
    there is no in-flight read for a publish to find.
    """

    def test_no_round_tears_and_every_round_actually_drained(self):
        registry = _Registry()
        barrier = VersionBarrier(
            count_readers=registry.count_readers, budget_ms=AMPLE_BUDGET_MS
        )
        rounds = 30
        torn = 0

        for version in range(rounds):
            memory = _ParameterMemory(version=version)
            copy: list[int] = []
            read_registered = threading.Event()

            def reader(version=version, memory=memory, copy=copy, gate=read_registered):
                registry.begin_read("rollout_0", version)
                gate.set()
                try:
                    for i in range(CHUNKS):
                        time.sleep(TRANSFER_PER_CHUNK_S)
                        copy.append(memory.read_chunk(i))
                finally:
                    registry.end_read("rollout_0", version)

            def writer(version=version, memory=memory, gate=read_registered):
                gate.wait(timeout=5)
                barrier.publish(previous_version=version)
                memory.overwrite(version=version + 1)

            threads = [threading.Thread(target=reader), threading.Thread(target=writer)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)

            torn += _is_torn(copy)

        assert torn == 0
        assert len(barrier.history) == rounds
        assert all(
            report.outcome is PublishOutcome.DRAINED for report in barrier.history
        ), (
            "a NO_READERS round means the publish found nothing in flight, so it never "
            "exercised the drain this test exists to stress"
        )
        assert not registry._inflight_reads, "every read must have drained"
