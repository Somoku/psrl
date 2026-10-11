"""Telling apart the two ways a pull can be out of date (item E2).

A rollout holds two pieces of state it did not compute: the *descriptors* that say where a
source's bytes live, and the *chunk plan* that says which version it is assembling. Either
can go stale, and the recoveries are opposite:

* **`DESC_STALE`** -- the addresses are wrong. The source re-registered memory, or a
  non-divisible join forced a re-shard. Retrying the read reaches the same wrong place;
  the client must re-fetch descriptors. Reading anyway lands on freed or re-purposed pages
  and **succeeds**, which is why this is a refusal rather than a warning.
* **`CHUNK_STALE`** -- the addresses are fine, the version is not. The source no longer
  holds what was asked for. Re-fetching descriptors changes nothing; the client must pick
  a different source, or re-pull at the newer version.

Conflating them produces a retry loop that cannot converge: a client that re-fetches
descriptors for a version nobody holds will keep getting valid descriptors for the wrong
bytes, and a client that switches source on a re-registration will find every source
equally wrong.

**Why the epoch must advance even when addresses do not.** A non-divisible join re-shards:
the same buffer, cut differently. Every byte is where it was, so an address check passes --
but the *meaning* of each descriptor changed, and a read returns someone else's slice. The
registration epoch is therefore bumped on re-shard as well as on re-registration (design
9c-a2), and it is the epoch, not the address, that the staleness check compares.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Staleness(Enum):
    """Why a pull request cannot be served as asked."""

    FRESH = "fresh"
    #: Descriptors are out of date; re-fetch them, keep the source.
    DESC_STALE = "desc_stale"
    #: The version is out of date or absent here; change source or version, keep descriptors.
    CHUNK_STALE = "chunk_stale"


class Recovery(Enum):
    """What the client should do about it."""

    PROCEED = "proceed"
    REFETCH_DESCRIPTORS = "refetch_descriptors"
    RESELECT_SOURCE = "reselect_source"


#: The mapping is one-to-one and total, so a new staleness kind cannot be added without
#: deciding its recovery. A dict lookup would let an unmapped member fall through.
_RECOVERY: dict[Staleness, Recovery] = {
    Staleness.FRESH: Recovery.PROCEED,
    Staleness.DESC_STALE: Recovery.REFETCH_DESCRIPTORS,
    Staleness.CHUNK_STALE: Recovery.RESELECT_SOURCE,
}


def recovery_for(staleness: Staleness) -> Recovery:
    try:
        return _RECOVERY[staleness]
    except KeyError:  # pragma: no cover - guards a future enum member
        raise NotImplementedError(
            f"{staleness} has no recovery path; a staleness kind without one would make "
            f"the client retry blindly"
        ) from None


@dataclass(frozen=True)
class SourceState:
    """What a source currently offers, as the index knows it."""

    registration_epoch: int
    versions_held: frozenset[int]


@dataclass(frozen=True)
class PullRequest:
    """What a client is asking for, with the epoch its descriptors came from."""

    version: int
    chunk: int
    descriptor_epoch: int


def classify(request: PullRequest, source: SourceState) -> Staleness:
    """Classify a pull against what the source actually has.

    **Epoch is checked first, and the order matters.** If both are stale, the descriptors
    are the blocking problem: re-selecting a source while holding descriptors from a
    previous registration just moves the bad addresses to a new peer. Reporting
    `DESC_STALE` first makes the client fix the thing that would otherwise invalidate its
    next attempt too.
    """
    if request.descriptor_epoch != source.registration_epoch:
        return Staleness.DESC_STALE
    if request.version not in source.versions_held:
        return Staleness.CHUNK_STALE
    return Staleness.FRESH


@dataclass
class RegistrationEpoch:
    """A counter that advances on anything invalidating remote descriptors.

    Two causes, and the second is the one that gets forgotten: re-registration moves
    addresses, and a re-shard keeps them while changing what each descriptor covers. Both
    make a remote's cached descriptors wrong, so both advance the epoch.
    """

    value: int = 0

    def note_reregistered(self) -> int:
        """Memory was registered again, so addresses may have moved."""
        self.value += 1
        return self.value

    def note_resharded(self) -> int:
        """Same buffer, different cuts -- addresses unchanged, meaning changed.

        Advancing here is what the design calls out: an address-equality check would pass
        and the read would return a neighbouring shard's bytes without error.
        """
        self.value += 1
        return self.value
