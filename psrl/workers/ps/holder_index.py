"""The holder index: who may serve which chunk of which version (item D2).

Under trainer-direct supply the PS stops carrying bytes and becomes an index. What it
must answer is "may this reader pull this chunk, and from whom" -- and get it right,
because the failure modes here are silent rather than loud.

**Role is part of the key, not a label on it.** Actor and critic run independent version
streams (design 3.10.3), so a single ``version -> holders`` map lets the critic's v3
overwrite the actor's v3. Nothing downstream notices: the reader gets a well-formed
tensor of the right shape belonging to a different model. Keying on ``(role, version)``
makes that unrepresentable instead of merely discouraged.

**Holder identity is the readability check.** Design 3.4.5 collapses "weights registered
and not mid-update" into "is a holder", so admitting a pull from a non-holder is what
would let an RDMA read land on freed pages. That is why `check_pull_admission` refuses
rather than warns.

**In-flight readers are a per-chunk integer, not a lease table.** Per 3.4.2.1 the count
exists to stop the optimizer overwriting parameter memory a reader is still reading; a
``version x chunk`` lease table would carry state proportional to both, for a question
that only needs "is anyone reading this chunk right now".
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


class HolderAdmissionError(RuntimeError):
    """A pull was refused: the proposed source cannot legally serve it."""


@dataclass
class RoleVersionHolders:
    """Holders for one ``(role, version)`` pair."""

    role: str
    version: int
    #: chunk -> holders that finished it
    chunks: dict[int, set[str]] = field(default_factory=lambda: defaultdict(set))
    #: chunk -> holders that got it from a peer and so may not serve it on
    second_hand: dict[int, set[str]] = field(default_factory=lambda: defaultdict(set))


@dataclass
class HolderIndex:
    """Holder bookkeeping keyed by ``(role, version, chunk)``.

    Only the current version of each role is retained. Keeping older versions would
    invite reads from them, which is the staleness the version-equality rule exists to
    forbid (3.5.2); dropping them also bounds the structure's size by role count rather
    than by run length.
    """

    _current: dict[str, RoleVersionHolders] = field(default_factory=dict)
    #: (role, chunk) -> number of reads currently in flight
    _in_flight: dict[tuple[str, int], int] = field(default_factory=lambda: defaultdict(int))

    def publish(self, role: str, version: int) -> None:
        """Start a new version for ``role``, dropping the previous one's holders."""
        existing = self._current.get(role)
        if existing is not None:
            if version < existing.version:
                raise ValueError(
                    f"{role}: version must not go backwards, index holds "
                    f"{existing.version} and was given {version}"
                )
            if version == existing.version:
                return
        self._current[role] = RoleVersionHolders(role=role, version=version)

    def current_version(self, role: str) -> int | None:
        entry = self._current.get(role)
        return entry.version if entry is not None else None

    def mark_held(
        self, role: str, version: int, chunk: int, holder: str, *, from_peer: bool = False
    ) -> None:
        """Record that ``holder`` finished ``chunk``.

        A report for a version that is no longer current is dropped rather than stored:
        it describes bytes that have already been superseded, and keeping it would make
        the holder eligible to serve a stale read.
        """
        entry = self._current.get(role)
        if entry is None or entry.version != version:
            return
        entry.chunks[chunk].add(holder)
        if from_peer:
            entry.second_hand[chunk].add(holder)

    def servable_holders(self, role: str, version: int, chunk: int) -> set[str]:
        """Holders that may serve ``chunk`` on: current version, not second-hand."""
        entry = self._current.get(role)
        if entry is None or entry.version != version:
            return set()
        return set(entry.chunks.get(chunk, set())) - set(entry.second_hand.get(chunk, set()))

    def forget_holder(self, holder: str) -> None:
        """Remove a holder everywhere -- it crashed, or left.

        Removal must be total and cheap: a stale entry means a reader is handed an
        address whose pages are gone, which reads as corruption rather than as an error.
        """
        for entry in self._current.values():
            for holders in entry.chunks.values():
                holders.discard(holder)
            for holders in entry.second_hand.values():
                holders.discard(holder)

    def check_pull_admission(self, role: str, version: int, chunk: int, source: str) -> None:
        """Raise unless ``source`` may legally serve ``(role, version, chunk)``.

        Four refusals, each corresponding to a way the read would otherwise land on
        memory that is not what the reader believes it is.
        """
        entry = self._current.get(role)
        if entry is None:
            raise HolderAdmissionError(f"no published version for role {role!r}")
        if version != entry.version:
            raise HolderAdmissionError(
                f"{role}: pull asks for version {version} but the current version is "
                f"{entry.version}; a same-version source is required"
            )
        if source not in entry.chunks.get(chunk, set()):
            raise HolderAdmissionError(
                f"{role} v{version} chunk {chunk}: {source!r} is not a holder, so its "
                "registration may already be invalid"
            )
        if source in entry.second_hand.get(chunk, set()):
            raise HolderAdmissionError(
                f"{role} v{version} chunk {chunk}: {source!r} holds it second hand and "
                "must not serve it on (no chains)"
            )

    # ---- in-flight readers (design 3.4.2.1) ----

    def begin_read(self, role: str, chunk: int) -> None:
        self._in_flight[(role, chunk)] += 1

    def end_read(self, role: str, chunk: int) -> None:
        key = (role, chunk)
        if self._in_flight[key] <= 0:
            raise AssertionError(
                f"{role} chunk {chunk}: end_read with no read in flight; the counter "
                "would go negative and later under-report live readers"
            )
        self._in_flight[key] -= 1

    def in_flight(self, role: str, chunk: int) -> int:
        return self._in_flight[(role, chunk)]

    def total_in_flight(self, role: str) -> int:
        return sum(count for (r, _chunk), count in self._in_flight.items() if r == role)

    def assert_no_readers_in_flight(self, role: str, context: str) -> None:
        """Raise if any read is still in flight for ``role``.

        Called before releasing or overwriting the memory those reads address -- a
        colocate sleep, or an optimizer write-back. Proceeding regardless is the
        "RDMA reads a freed page" failure, which produces wrong numbers and no error.
        """
        busy = {
            chunk: count
            for (r, chunk), count in self._in_flight.items()
            if r == role and count > 0
        }
        if busy:
            raise AssertionError(
                f"{context}: {role} still has readers in flight on chunks "
                f"{sorted(busy)}; releasing now would read freed pages"
            )
