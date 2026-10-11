"""E1's reader registry: the publisher's only view of who is mid-pull.

`VersionBarrier` drains readers of the version it is about to overwrite, and it can only
do that if something counts them. Readers are the only party that knows when a pull
starts and ends, so they report it here and the publisher polls.

The failure this guards is not "it did not drain" -- that is visible. It is a count that
cannot go back to zero, which makes the barrier wait out its whole budget on every
publish from then on, forever, with no error anywhere. Every test below is about the
count returning to zero or refusing to move wrongly.

The lock decorator is bypassed deliberately (`__wrapped__`): `_state_lock` exists to
serialise Ray actor and gRPC threads, and these are single-threaded tests of the
arithmetic. Constructing a real `PSManager` would test the config plumbing instead.
"""

import pytest

from psrl.workers.ps.ps_manager import PSManager

pytestmark = pytest.mark.cpu_test


class _Registry:
    """Stands in for the manager's state, holding only what the methods touch."""

    def __init__(self):
        self._inflight_reads = {}

    begin_read = PSManager.begin_read.__wrapped__
    end_read = PSManager.end_read.__wrapped__
    abort_reads = PSManager.abort_reads.__wrapped__
    count_readers = PSManager.count_readers


class TestTheCountTracksWhatIsActuallyInFlight:
    def test_no_readers_is_zero(self):
        assert _Registry().count_readers(7) == 0

    def test_begin_then_end_returns_to_zero(self):
        """The property the barrier relies on and the one that silently breaks."""
        registry = _Registry()

        registry.begin_read("instance_0", 7)
        assert registry.count_readers(7) == 1

        registry.end_read("instance_0", 7)
        assert registry.count_readers(7) == 0

    def test_readers_of_another_version_are_not_counted(self):
        """Otherwise the barrier waits on the *previous* publish's stragglers."""
        registry = _Registry()
        registry.begin_read("instance_0", 6)

        assert registry.count_readers(7) == 0
        assert registry.count_readers(6) == 1

    def test_two_instances_add_up(self):
        registry = _Registry()

        registry.begin_read("instance_0", 7)
        registry.begin_read("instance_1", 7)

        assert registry.count_readers(7) == 2

    def test_one_instance_reading_twice_counts_twice(self):
        """A reader that overlaps two pulls of the same version must finish both before
        the version is safe to overwrite."""
        registry = _Registry()

        registry.begin_read("instance_0", 7)
        registry.begin_read("instance_0", 7)
        registry.end_read("instance_0", 7)

        assert registry.count_readers(7) == 1


class TestTheCountCannotGetStuckOrGoNegative:
    def test_an_end_without_a_begin_is_ignored(self):
        """A reader that reports an end it never began -- after a restart -- must not
        drive the total negative, which would read as "no readers" while one is live."""
        registry = _Registry()

        registry.end_read("instance_0", 7)

        assert registry.count_readers(7) == 0

    def test_an_end_for_an_unknown_version_is_ignored(self):
        registry = _Registry()
        registry.begin_read("instance_0", 7)

        registry.end_read("instance_0", 6)

        assert registry.count_readers(7) == 1

    def test_abort_drops_one_instance_without_touching_the_others(self):
        """For a reader that will never report its ends -- killed mid-pull. Without this
        the version never drains and the barrier waits out its budget on every publish."""
        registry = _Registry()
        registry.begin_read("instance_0", 7)
        registry.begin_read("instance_1", 7)
        registry.begin_read("instance_1", 6)

        registry.abort_reads("instance_1")

        assert registry.count_readers(7) == 1
        assert registry.count_readers(6) == 0

    def test_abort_is_idempotent(self):
        registry = _Registry()
        registry.begin_read("instance_0", 7)

        registry.abort_reads("instance_0")
        registry.abort_reads("instance_0")

        assert registry.count_readers(7) == 0

    def test_drained_versions_leave_no_entry_behind(self):
        """A long run must not accumulate one dict entry per version published."""
        registry = _Registry()

        for version in range(100):
            registry.begin_read("instance_0", version)
            registry.end_read("instance_0", version)

        assert registry._inflight_reads == {}


class TestTheCountIsSafeToPollFromTheBarrier:
    """`count_readers` is polled in a tight loop by a publisher that holds no lock, while
    readers take the state lock to end their reads. It must therefore not take the lock
    itself -- a locking poll would make a finishing reader wait behind a barrier that is
    waiting for it to finish."""

    def test_count_readers_is_not_lock_decorated(self):
        assert not hasattr(PSManager.count_readers, "__wrapped__"), (
            "count_readers is lock-decorated; the barrier polls it in a loop while "
            "readers need the same lock to end their reads, which deadlocks"
        )

    def test_begin_and_end_are_lock_decorated(self):
        """They mutate, and readers on different threads must not interleave."""
        for method in (PSManager.begin_read, PSManager.end_read, PSManager.abort_reads):
            assert hasattr(method, "__wrapped__"), f"{method.__name__} is not lock-decorated"
