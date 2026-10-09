"""Holder index: role dimension and the four runtime assertions (item D2).

Every failure this guards against is silent. Reading from a non-holder returns bytes
from freed pages, not an error; a critic version overwriting an actor version yields a
correctly-shaped tensor of the wrong model. So these are assertions in the code and
deliberately-violated cases in the tests, rather than documented requirements.
"""

from __future__ import annotations

import pytest
from psrl.workers.ps.holder_index import HolderAdmissionError, HolderIndex

pytestmark = pytest.mark.cpu_test


class TestRoleIsPartOfTheKey:
    """Actor and critic run independent version streams (design 3.10.3)."""

    def test_same_version_number_in_two_roles_stays_separate(self):
        index = HolderIndex()
        index.publish("actor", 3)
        index.publish("critic", 3)
        index.mark_held("actor", 3, chunk=0, holder="a0")
        index.mark_held("critic", 3, chunk=0, holder="c0")

        assert index.servable_holders("actor", 3, 0) == {"a0"}
        assert index.servable_holders("critic", 3, 0) == {"c0"}

    def test_publishing_one_role_does_not_clear_the_other(self):
        """Without the role key, the critic's publish would wipe the actor's holders."""
        index = HolderIndex()
        index.publish("actor", 3)
        index.mark_held("actor", 3, chunk=0, holder="a0")

        index.publish("critic", 3)
        index.publish("critic", 4)

        assert index.servable_holders("actor", 3, 0) == {"a0"}

    def test_roles_advance_independently(self):
        index = HolderIndex()
        index.publish("actor", 7)
        index.publish("critic", 2)

        assert index.current_version("actor") == 7
        assert index.current_version("critic") == 2


class TestVersionLifecycle:
    def test_publishing_a_new_version_drops_the_old_holders(self):
        index = HolderIndex()
        index.publish("actor", 1)
        index.mark_held("actor", 1, chunk=0, holder="a0")

        index.publish("actor", 2)

        assert index.servable_holders("actor", 2, 0) == set()

    def test_versions_do_not_go_backwards(self):
        index = HolderIndex()
        index.publish("actor", 5)
        with pytest.raises(ValueError, match="must not go backwards"):
            index.publish("actor", 4)

    def test_republishing_the_same_version_keeps_holders(self):
        index = HolderIndex()
        index.publish("actor", 1)
        index.mark_held("actor", 1, chunk=0, holder="a0")

        index.publish("actor", 1)

        assert index.servable_holders("actor", 1, 0) == {"a0"}

    def test_a_late_report_for_an_old_version_is_dropped(self):
        """Storing it would make the holder eligible to serve superseded bytes."""
        index = HolderIndex()
        index.publish("actor", 1)
        index.publish("actor", 2)

        index.mark_held("actor", 1, chunk=0, holder="straggler")

        assert index.servable_holders("actor", 1, 0) == set()
        assert index.servable_holders("actor", 2, 0) == set()


class TestPullAdmission:
    """Holder identity *is* the readability check (design 3.4.5)."""

    def test_a_holder_is_admitted(self):
        index = HolderIndex()
        index.publish("actor", 1)
        index.mark_held("actor", 1, chunk=0, holder="a0")

        index.check_pull_admission("actor", 1, 0, "a0")

    def test_a_non_holder_is_refused(self):
        """Admitting it is what lets an RDMA read land on freed pages."""
        index = HolderIndex()
        index.publish("actor", 1)

        with pytest.raises(HolderAdmissionError, match="not a holder"):
            index.check_pull_admission("actor", 1, 0, "ghost")

    def test_a_stale_version_is_refused(self):
        index = HolderIndex()
        index.publish("actor", 2)
        index.mark_held("actor", 2, chunk=0, holder="a0")

        with pytest.raises(HolderAdmissionError, match="current version"):
            index.check_pull_admission("actor", 1, 0, "a0")

    def test_a_second_hand_holder_may_not_serve_on(self):
        index = HolderIndex()
        index.publish("actor", 1)
        index.mark_held("actor", 1, chunk=0, holder="a1", from_peer=True)

        with pytest.raises(HolderAdmissionError, match="no chains"):
            index.check_pull_admission("actor", 1, 0, "a1")

    def test_an_unpublished_role_is_refused(self):
        index = HolderIndex()
        with pytest.raises(HolderAdmissionError, match="no published version"):
            index.check_pull_admission("actor", 1, 0, "a0")

    def test_a_forgotten_holder_is_refused(self):
        index = HolderIndex()
        index.publish("actor", 1)
        index.mark_held("actor", 1, chunk=0, holder="a0")
        index.forget_holder("a0")

        with pytest.raises(HolderAdmissionError, match="not a holder"):
            index.check_pull_admission("actor", 1, 0, "a0")


class TestInFlightReaders:
    """A per-chunk integer, not a version x chunk lease table (design 3.4.2.1)."""

    def test_counts_rise_and_fall(self):
        index = HolderIndex()
        index.begin_read("actor", 0)
        index.begin_read("actor", 0)
        assert index.in_flight("actor", 0) == 2

        index.end_read("actor", 0)
        assert index.in_flight("actor", 0) == 1

    def test_counts_are_per_chunk(self):
        index = HolderIndex()
        index.begin_read("actor", 0)
        assert index.in_flight("actor", 1) == 0

    def test_counts_are_per_role(self):
        index = HolderIndex()
        index.begin_read("actor", 0)
        assert index.in_flight("critic", 0) == 0

    def test_an_unbalanced_end_is_rejected(self):
        """A negative counter would later under-report live readers."""
        index = HolderIndex()
        with pytest.raises(AssertionError, match="no read in flight"):
            index.end_read("actor", 0)

    def test_release_is_refused_while_readers_are_in_flight(self):
        index = HolderIndex()
        index.begin_read("actor", 3)

        with pytest.raises(AssertionError, match="readers in flight"):
            index.assert_no_readers_in_flight("actor", context="colocate sleep")

    def test_release_is_allowed_once_readers_drain(self):
        index = HolderIndex()
        index.begin_read("actor", 3)
        index.end_read("actor", 3)

        index.assert_no_readers_in_flight("actor", context="colocate sleep")

    def test_the_error_names_the_busy_chunks(self):
        """After a stall the question is which chunk, not how many."""
        index = HolderIndex()
        index.begin_read("actor", 2)
        index.begin_read("actor", 5)

        with pytest.raises(AssertionError, match=r"\[2, 5\]"):
            index.assert_no_readers_in_flight("actor", context="optimizer write-back")

    def test_another_roles_readers_do_not_block_release(self):
        index = HolderIndex()
        index.begin_read("critic", 0)

        index.assert_no_readers_in_flight("actor", context="colocate sleep")
