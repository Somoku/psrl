"""Revocation primitives for the shrink path (item E4, lower half).

These cover the pieces that have no cluster in them: the two message encodings, the
eviction wrapper's idempotence, and the outcome object the server raises on. The
round trip lives in `test_release_member.py`, and the claim that eviction actually
invalidates a descriptor -- the reason any of this exists -- is measured against a real
agent in `test_revoke_descriptors_gpu.py`.
"""

from __future__ import annotations

import pytest
from nixl._bindings import nixlNotFoundError
from psrl.utils.nixl.handshake_notif import NotifKind, decode
from psrl.utils.nixl.leave_protocol import (
    RevokeOutcome,
    encode_leave_ack,
    encode_member_leave,
    leaver_agent_of,
    revoke_descriptors,
)

pytestmark = pytest.mark.cpu_test


class TestMessages:
    def test_a_leave_carries_the_agent_to_forget(self):
        """The receiver must not have to look it up: the leaver may already be gone."""
        notif = decode(encode_member_leave("gen3", "agent-gen3"))

        assert notif.kind is NotifKind.MEMBER_LEAVE
        assert leaver_agent_of(notif.entries, "gen3") == "agent-gen3"

    def test_an_ack_names_the_leaver_not_the_sender(self):
        """`get_new_notifs` keys by sending agent, so repeating it would be a second,
        disagreeing source of the revoker's identity."""
        notif = decode(encode_leave_ack("gen3", held=True))

        assert notif.kind is NotifKind.LEAVE_ACK
        assert set(notif.entries) == {"gen3"}
        assert notif.entries["gen3"]["held"] is True

    def test_an_ack_can_report_that_there_was_nothing_to_forget(self):
        assert decode(encode_leave_ack("gen3", held=False)).entries["gen3"]["held"] is False

    def test_membership_messages_are_distinguishable_from_the_rest(self):
        """They share the agent stream with the handshake and with chunk signals."""
        kinds = {
            decode(encode_member_leave("g", "a")).kind,
            decode(encode_leave_ack("g", held=True)).kind,
        }

        assert kinds == {NotifKind.MEMBER_LEAVE, NotifKind.LEAVE_ACK}

    def test_a_foreign_payload_is_not_mistaken_for_one(self):
        assert decode(b"psrl-chunk-done:whatever") is None


class _Agent:
    def __init__(self, known: set[str]):
        self.known = set(known)
        self.removed: list[str] = []

    def remove_remote_agent(self, name: str) -> None:
        self.removed.append(name)
        if name not in self.known:
            raise nixlNotFoundError("NIXL_ERR_NOT_FOUND")
        self.known.discard(name)


class TestRevokeIsIdempotent:
    def test_a_known_peer_is_evicted(self):
        agent = _Agent(known={"agent-gen3"})

        revoke_descriptors(agent, "agent-gen3")

        assert agent.known == set()

    def test_an_unknown_peer_is_not_an_error(self):
        """The post-condition is "cannot address it", and never having loaded it
        satisfies that. A raise here would make the normal retry -- after a lost ack --
        report a failure that does not exist.

        The stub raises not-found; the installed NIXL does not (measured: it returns
        normally even for a name that was never loaded). Both must be benign, which is
        why the guard exists even though this build never exercises it.
        """
        agent = _Agent(known=set())

        revoke_descriptors(agent, "agent-gen3")

    def test_revoking_twice_is_benign(self):
        agent = _Agent(known={"agent-gen3"})

        revoke_descriptors(agent, "agent-gen3")
        revoke_descriptors(agent, "agent-gen3")

        assert agent.removed == ["agent-gen3", "agent-gen3"]

    def test_it_reports_nothing_because_nothing_is_knowable(self):
        """Guards the signature. "True if it had anything to forget" is the obvious
        shape and is not implementable on this build, so a future edit that adds it
        back would be putting an unbacked value into the ack."""
        assert revoke_descriptors(_Agent(known={"agent-gen3"}), "agent-gen3") is None

    def test_other_failures_still_propagate(self):
        """Only not-found is benign; a backend fault must not look like a clean leave."""

        class _Broken:
            def remove_remote_agent(self, name):
                raise RuntimeError("backend exploded")

        with pytest.raises(RuntimeError, match="exploded"):
            revoke_descriptors(_Broken(), "agent-gen3")


class TestOutcome:
    def test_a_full_set_of_acks_is_complete(self):
        outcome = RevokeOutcome("gen3", frozenset({"a", "b"}), frozenset({"a", "b"}))

        assert outcome.complete
        assert outcome.missing == frozenset()
        assert "confirmed" in outcome.describe()

    def test_an_incomplete_outcome_names_who_is_missing(self):
        """Which holder is unconfirmed decides what the operator does next, so the
        message carries names rather than a count."""
        outcome = RevokeOutcome("gen3", frozenset({"a", "b", "c"}), frozenset({"a"}))

        assert not outcome.complete
        assert outcome.missing == frozenset({"b", "c"})
        assert "'b'" in outcome.describe() and "'c'" in outcome.describe()

    def test_a_member_nobody_holds_is_trivially_complete(self):
        assert RevokeOutcome("gen3", frozenset(), frozenset()).complete

    def test_extra_acks_do_not_make_it_incomplete(self):
        """An agent can ack a leaver it was never required to forget -- it simply had
        the descriptors -- and that is strictly good news."""
        outcome = RevokeOutcome("gen3", frozenset({"a"}), frozenset({"a", "b"}))

        assert outcome.complete
