"""Letting a running cluster accept and release members (item E3).

The handshake is a straight line: `wait_for_client_shardings` and `wait_for_client_infos`
each set a flag and, once set, return immediately on every later call
(`server.py`, marked `TODO(lhy): Support elastic clients after initial registration`).
A new instance arriving afterwards has no entry point -- not because it is rejected, but
because nobody is listening.

This module is the membership state the event-driven loop needs, kept separate from the
server's notification plumbing for the reason the rest of this design keeps policy out of
transport: a state machine that decides *when a replan is required* can be tested by
driving it through joins and leaves, whereas the same logic inside the notification loop
could only be tested by standing up a cluster.

**Admission reuses A7's three-level verdict.** `classify_join` already answers whether a
joiner's mesh divides the current LCM; elasticity does not need a second notion of
"allowed". What this adds is *when* the answer is acted on, and what the rest of the
cluster must do about it.

**Why leaves need a quiesce step and joins do not.** A joiner has nothing anyone reads
yet, so it can be added the moment its metadata lands. A leaver may be mid-service: the
design's shrink order is drop holder status, drain in-flight readers, invalidate
descriptors, then release memory. All four are represented here -- one, two and four as
the `begin_leave`/`complete_leave` split, three as the revocations `complete_leave`
refuses to proceed without (`leave_protocol` carries out the eviction itself).

**Revocation is tracked, not assumed.** `complete_leave` already refused to run with
readers in flight; it now refuses equally when a holder has not confirmed that it
forgot the leaver's descriptors. The two failures are the same failure -- memory freed
while something can still address it -- so they are reported the same way.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum


class MembershipEvent(Enum):
    JOINED = "joined"
    LEFT = "left"
    #: A join that A7 refused; the cluster is unchanged.
    REFUSED = "refused"


@dataclass(frozen=True)
class MembershipChange:
    event: MembershipEvent
    client: str
    #: Whether the comm plan must be recompiled before the next pull.
    requires_replan: bool
    #: Whether existing members must re-fetch descriptors (A7's WARN_RESHARD case).
    requires_epoch_bump: bool
    reason: str = ""


@dataclass
class Membership:
    """Who is in the cluster, and what a change obliges the rest of it to do.

    ``admit`` is injected -- `classify_join` in production -- so this can be driven
    through admission outcomes without constructing meshes, and so there is exactly one
    place that decides admissibility.
    """

    #: Returns (allowed, needs_reshard, reason) for a prospective joiner.
    admit: Callable[[str], tuple[bool, bool, str]]
    members: set[str] = field(default_factory=set)
    #: Members that have been told to stop serving but may still have readers in flight.
    draining: set[str] = field(default_factory=set)
    #: leaver -> holders that confirmed they evicted its descriptors (E4 step three).
    #: Keyed by whatever identity the caller uses for a holder: the server records
    #: *agent* names, because `remove_remote_agent` evicts per agent, not per client.
    revocations: dict[str, set[str]] = field(default_factory=dict)
    history: list[MembershipChange] = field(default_factory=list)

    def join(self, client: str) -> MembershipChange:
        """Admit ``client`` if A7 allows it.

        A re-join of an existing member is not an error: a client that reconnects after a
        transient failure is the common case, and treating it as a conflict would turn a
        recoverable blip into a dead instance. It does require a replan, because its
        descriptors are new.
        """
        allowed, needs_reshard, reason = self.admit(client)
        if not allowed:
            change = MembershipChange(
                MembershipEvent.REFUSED,
                client,
                requires_replan=False,
                requires_epoch_bump=False,
                reason=reason,
            )
            self.history.append(change)
            return change

        self.members.add(client)
        self.draining.discard(client)
        change = MembershipChange(
            MembershipEvent.JOINED,
            client,
            requires_replan=True,
            requires_epoch_bump=needs_reshard,
            reason=reason,
        )
        self.history.append(change)
        return change

    def begin_leave(self, client: str) -> None:
        """Stop routing reads to ``client`` while letting in-flight ones finish.

        Separate from `complete_leave` because the two are separated in time by however
        long the slowest reader takes. Collapsing them would free memory under an active
        read, and that read lands on freed pages and *succeeds*.
        """
        if client not in self.members:
            raise KeyError(f"{client} is not a member; nothing to drain")
        self.draining.add(client)
        # A re-leave starts its confirmations over. Acks from a previous attempt refer
        # to descriptors the member has since re-published, so counting them would clear
        # the shrink on evidence about a registration that no longer exists.
        self.revocations.pop(client, None)

    def record_revocation(self, client: str, holder: str) -> None:
        """Note that ``holder`` has evicted ``client``'s descriptors.

        Accepted before `begin_leave` as well as after: the ack is a statement about the
        holder's own state, and dropping an early one would deadlock a shrink whose acks
        outran its bookkeeping.
        """
        self.revocations.setdefault(client, set()).add(holder)

    def revoked_by(self, client: str) -> set[str]:
        """Holders that have confirmed eviction of ``client``."""
        return set(self.revocations.get(client, set()))

    def complete_leave(
        self,
        client: str,
        in_flight: int,
        required_revokers: set[str] | None = None,
    ) -> MembershipChange:
        """Remove ``client`` once its readers have drained and its holders have forgotten it.

        Refuses while readers remain rather than removing anyway: the caller's job is to
        wait, and silently dropping the member here would leave those readers pointed at
        memory about to be released.

        ``required_revokers`` names who must have confirmed step three. It defaults to
        the other members, which is the right answer when holders are identified by
        client name; the server overrides it with agent names, since one eviction covers
        every client an agent hosts. Pass an empty set only for a member that never
        published anything -- the check exists because a holder that still has the
        descriptors can read freed memory *successfully*, which is the silent corruption
        §3.3.1d catalogues.
        """
        if client not in self.draining:
            raise KeyError(
                f"{client} was removed without begin_leave; readers were never told to "
                f"stop, so some may still be mid-transfer"
            )
        if in_flight > 0:
            raise RuntimeError(
                f"{client} still has {in_flight} reader(s) in flight; releasing its "
                f"memory now would let those reads land on freed pages"
            )
        required = self.members - {client} if required_revokers is None else set(required_revokers)
        missing = required - self.revoked_by(client)
        if missing:
            raise RuntimeError(
                f"{client} has not been revoked by {sorted(missing)}; those holders can "
                f"still address its descriptors, so releasing its memory now would let "
                f"their reads land on freed pages"
            )

        self.members.discard(client)
        self.draining.discard(client)
        self.revocations.pop(client, None)
        change = MembershipChange(
            MembershipEvent.LEFT,
            client,
            requires_replan=True,
            requires_epoch_bump=False,
            reason="drained",
        )
        self.history.append(change)
        return change

    @property
    def serving(self) -> set[str]:
        """Members eligible to serve reads: everyone not draining."""
        return self.members - self.draining

    @property
    def replan_count(self) -> int:
        return sum(1 for change in self.history if change.requires_replan)
