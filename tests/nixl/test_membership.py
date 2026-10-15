"""Elastic join and leave while the cluster serves (implementation item E3).

The design's criterion is that a running cluster accepts a new instance *without
interrupting existing ones*. Two properties carry that:

* a join must not disturb members already serving -- it adds a replan, not a pause;
* a leave must not release memory while a reader is still mid-transfer, because that read
  lands on freed pages and succeeds;
* a leave must not release memory while any peer can still *address* it, which is the
  same failure reached a different way -- the peer need not be mid-transfer, only
  holding descriptors it has not been told to forget.

Admission itself is A7's job (`classify_join`); these tests cover when its answer is acted
on and what the rest of the cluster owes as a result.
"""

from __future__ import annotations

import pytest

from psrl.utils.nixl.membership import Membership, MembershipEvent

pytestmark = pytest.mark.cpu_test


def _always_admit(_client: str):
    return True, False, "fast_path"


def _admit_with_reshard(_client: str):
    return True, True, "warn_reshard"


def _always_refuse(_client: str):
    return False, False, "mesh does not divide tensor dimension"


class TestJoin:
    def test_a_join_adds_a_member_and_requires_a_replan(self):
        m = Membership(admit=_always_admit, members={"gen0"})

        change = m.join("gen1")

        assert change.event is MembershipEvent.JOINED
        assert change.requires_replan
        assert m.members == {"gen0", "gen1"}

    def test_a_refused_join_leaves_the_cluster_unchanged(self):
        m = Membership(admit=_always_refuse, members={"gen0"})

        change = m.join("gen1")

        assert change.event is MembershipEvent.REFUSED
        assert not change.requires_replan
        assert m.members == {"gen0"}
        assert "divide" in change.reason

    def test_a_reshard_join_also_requires_an_epoch_bump(self):
        """A7's WARN_RESHARD: addresses may be unchanged but their meaning is not."""
        m = Membership(admit=_admit_with_reshard, members={"gen0"})

        change = m.join("gen1")

        assert change.requires_epoch_bump

    def test_a_fast_path_join_needs_no_epoch_bump(self):
        m = Membership(admit=_always_admit, members={"gen0"})

        assert not m.join("gen1").requires_epoch_bump

    def test_rejoining_is_allowed_and_replans(self):
        """A reconnect after a blip is routine; treating it as a conflict kills the instance."""
        m = Membership(admit=_always_admit, members={"gen0"})

        change = m.join("gen0")

        assert change.event is MembershipEvent.JOINED
        assert change.requires_replan, "the rejoiner's descriptors are new"

    def test_existing_members_keep_serving_across_a_join(self):
        """The criterion: a join is additive, not a pause."""
        m = Membership(admit=_always_admit, members={"gen0", "gen1"})
        before = set(m.serving)

        m.join("gen2")

        assert before <= m.serving


class TestLeaveIsTwoPhase:
    def test_draining_removes_a_member_from_serving_but_not_from_members(self):
        m = Membership(admit=_always_admit, members={"gen0", "gen1"})

        m.begin_leave("gen1")

        assert "gen1" not in m.serving
        assert "gen1" in m.members, "it still holds memory until its readers drain"

    def test_completing_a_leave_with_readers_in_flight_is_refused(self):
        """Releasing now lets an in-flight read land on freed pages -- and succeed."""
        m = Membership(admit=_always_admit, members={"gen0", "gen1"})
        m.begin_leave("gen1")

        with pytest.raises(RuntimeError, match="in flight"):
            m.complete_leave("gen1", in_flight=2)

        assert "gen1" in m.members

    def test_completing_a_leave_after_draining_removes_it(self):
        m = Membership(admit=_always_admit, members={"gen0", "gen1"})
        m.begin_leave("gen1")
        m.record_revocation("gen1", "gen0")

        change = m.complete_leave("gen1", in_flight=0)

        assert change.event is MembershipEvent.LEFT
        assert change.requires_replan
        assert m.members == {"gen0"}

    def test_removing_without_draining_first_is_refused(self):
        """Skipping the announce means readers were never told to stop."""
        m = Membership(admit=_always_admit, members={"gen0", "gen1"})

        with pytest.raises(KeyError, match="begin_leave"):
            m.complete_leave("gen1", in_flight=0)

    def test_draining_a_non_member_is_refused(self):
        m = Membership(admit=_always_admit, members={"gen0"})

        with pytest.raises(KeyError, match="not a member"):
            m.begin_leave("gen1")

    def test_other_members_keep_serving_throughout_a_leave(self):
        m = Membership(admit=_always_admit, members={"gen0", "gen1", "gen2"})

        m.begin_leave("gen1")
        m.record_revocation("gen1", "gen0")
        m.record_revocation("gen1", "gen2")
        m.complete_leave("gen1", in_flight=0)

        assert m.serving == {"gen0", "gen2"}


class TestLeaveRequiresRevocation:
    """E4 step three: nobody may still hold usable descriptors when the memory goes.

    Measured on this NIXL build, a holder that has *not* been told to forget keeps a
    working descriptor, while one that has gets NIXL_ERR_NOT_FOUND -- so the difference
    between these two states is the difference between a loud failure and a silent read
    of freed pages. That makes it a precondition, not a best effort.
    """

    def test_completing_a_leave_without_any_revocation_is_refused(self):
        m = Membership(admit=_always_admit, members={"gen0", "gen1"})
        m.begin_leave("gen1")

        with pytest.raises(RuntimeError, match="has not been revoked by"):
            m.complete_leave("gen1", in_flight=0)

        assert "gen1" in m.members

    def test_a_partial_revocation_is_refused_and_names_the_holdout(self):
        """The holdout's name is the point: one unconfirmed peer is the whole risk."""
        m = Membership(admit=_always_admit, members={"gen0", "gen1", "gen2"})
        m.begin_leave("gen2")
        m.record_revocation("gen2", "gen0")

        with pytest.raises(RuntimeError, match="gen1"):
            m.complete_leave("gen2", in_flight=0)

    def test_the_required_set_can_be_overridden_for_agent_granularity(self):
        """One `remove_remote_agent` covers every client an agent hosts, so the server
        confirms per agent; demanding a per-client ack would wait forever."""
        m = Membership(admit=_always_admit, members={"gen0", "gen1", "gen2"})
        m.begin_leave("gen2")
        m.record_revocation("gen2", "agent-a")

        change = m.complete_leave("gen2", in_flight=0, required_revokers={"agent-a"})

        assert change.event is MembershipEvent.LEFT

    def test_a_member_nobody_holds_can_leave_immediately(self):
        m = Membership(admit=_always_admit, members={"gen0"})
        m.begin_leave("gen0")

        assert m.complete_leave("gen0", in_flight=0).event is MembershipEvent.LEFT

    def test_in_flight_readers_are_reported_before_missing_revocations(self):
        """Both are fatal; the drain is the earlier step, so it is the actionable one."""
        m = Membership(admit=_always_admit, members={"gen0", "gen1"})
        m.begin_leave("gen1")

        with pytest.raises(RuntimeError, match="in flight"):
            m.complete_leave("gen1", in_flight=1)

    def test_an_early_ack_still_counts(self):
        """Acks can outrun the bookkeeping; dropping them would deadlock the shrink."""
        m = Membership(admit=_always_admit, members={"gen0", "gen1"})
        m.record_revocation("gen1", "gen0")
        m.begin_leave("gen1")

        assert m.revoked_by("gen1") == set()
        m.record_revocation("gen1", "gen0")
        assert m.complete_leave("gen1", in_flight=0).event is MembershipEvent.LEFT

    def test_restarting_a_leave_discards_the_previous_confirmations(self):
        """A re-leave follows a re-registration, so old acks describe gone descriptors."""
        m = Membership(admit=_always_admit, members={"gen0", "gen1"})
        m.begin_leave("gen1")
        m.record_revocation("gen1", "gen0")
        m.begin_leave("gen1")

        with pytest.raises(RuntimeError, match="has not been revoked by"):
            m.complete_leave("gen1", in_flight=0)

    def test_revocations_do_not_outlive_the_member(self):
        m = Membership(admit=_always_admit, members={"gen0", "gen1"})
        m.begin_leave("gen1")
        m.record_revocation("gen1", "gen0")
        m.complete_leave("gen1", in_flight=0)

        assert m.revoked_by("gen1") == set()


class TestReplanAccounting:
    def test_only_membership_changes_trigger_replans(self):
        """A refused join must not cost a replan; that is work for nothing."""
        m = Membership(admit=_always_refuse, members={"gen0"})

        for _ in range(5):
            m.join("gen1")

        assert m.replan_count == 0

    def test_each_accepted_change_counts_once(self):
        m = Membership(admit=_always_admit, members={"gen0"})

        m.join("gen1")
        m.begin_leave("gen1")
        m.record_revocation("gen1", "gen0")
        m.complete_leave("gen1", in_flight=0)

        assert m.replan_count == 2
