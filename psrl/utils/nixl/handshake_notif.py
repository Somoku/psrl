"""Tagged handshake notifications, so the server can tell them apart (item E3).

The three things a client sends during the handshake -- its sharding, its client info,
and its temporary mappings -- were all pickled as ``{client_name: payload}``. They are
*structurally identical*: every one of them is a dict keyed by client name, so
``isinstance(obj, dict)`` accepts all three and nothing in the bytes says which is which.

The server got away with that only because the handshake is lock-step. Each
``wait_for_*`` method drains notifications until it has heard from every agent, and the
phases never overlap, so whatever arrives during a phase must belong to it. Position in
time *was* the type tag.

**That is what blocks elastic membership.** A client joining after the initial handshake
sends its sharding whenever it happens to be ready -- which is, by definition, out of
phase. Two things then go wrong, and neither reports:

* If some other ``wait_for_*`` loop is running, it consumes that notification (the stream
  is drained destructively) and files it in the **wrong registry**, because the dict looks
  exactly like its own. A sharding dict lands in ``client_infos``.
* If no loop is running, nothing ever drains it and the join is simply lost.

So the envelope carries its kind explicitly, and the server routes on that rather than on
what phase it believes it is in. This is also what makes a single demultiplexing drain
possible: with a tag, one reader can serve all three registries, which it must, because
``get_new_notifs`` hands each notification to exactly one caller.

Payloads that are not handshake notifications -- peer-diffusion chunk notices share the
same agent-level stream -- decode to ``None`` and are left for their own consumer rather
than raising.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from enum import Enum
from typing import Any

#: Marks a payload as a handshake envelope. Present so that a notification from another
#: subsystem is *recognised as foreign* rather than guessed at -- the failure this module
#: exists to remove is a payload being accepted by the wrong reader.
_MAGIC = "psrl.nixl.handshake"


class NotifKind(Enum):
    """Which registry a handshake notification belongs to."""

    SHARDING = "sharding"
    CLIENT_INFO = "client_info"
    TEMP_MAPPING = "temp_mapping"
    #: Info *and* temp mapping together, re-sent after a resharding. A separate kind
    #: because its payload is `{"info": ..., "temp_mapping": ...}` rather than a bare
    #: one, and because its reader rejects anything else by raising -- an untagged
    #: handshake notification landing there took the run down rather than being skipped.
    UPDATE_INFO = "update_info"
    #: Server -> peers: forget a departing member's descriptors (E4's shrink step three).
    #: It rides this channel rather than NIXL's own `invalidateLocalMD` socket because
    #: the revocation has to be *confirmed* before the leaver frees memory, and the
    #: socket path is fire-and-forget (`nixl_p2p_metadata_backend.cpp` submits the INVL
    #: to a worker queue and only logs failures). See `leave_protocol`.
    MEMBER_LEAVE = "member_leave"
    #: Peer -> server: this agent has executed the revocation. The half that makes the
    #: shrink safe; without it the server cannot distinguish "revoked" from "unreachable".
    LEAVE_ACK = "leave_ack"
    #: Server -> reader: the trainer is about to overwrite the parameters this pull is
    #: reading. The reader stops committing, acknowledges, and repairs the unfinished
    #: keys from the holders listed in the payload (item E5, write gate).
    PULL_PREEMPT = "pull_preempt"
    #: Reader -> server: the boundary is recorded; pre-acknowledgement reads are safe.
    PULL_PREEMPT_ACK = "pull_preempt_ack"


@dataclass(frozen=True)
class HandshakeNotif:
    """One notification: a kind, and the per-client payloads it carries.

    Several clients at once because a multi-client agent reports for all of its clients
    in one notification; a single client sends a one-entry mapping.
    """

    kind: NotifKind
    entries: dict[str, Any]


def encode(kind: NotifKind, entries: dict[str, Any]) -> bytes:
    """Serialise ``entries`` tagged with ``kind``."""
    return pickle.dumps((_MAGIC, kind.value, entries))


def decode(raw: bytes) -> HandshakeNotif | None:
    """Parse a handshake notification, or return None if it is not one.

    Returns None rather than raising for anything foreign or malformed: this is called on
    every notification the agent delivers, including ones belonging to other subsystems,
    and a raise would turn an unrelated message into a handshake failure.
    """
    try:
        payload = pickle.loads(raw)
    except Exception:
        return None

    if not isinstance(payload, tuple) or len(payload) != 3:
        return None
    magic, kind_value, entries = payload
    if magic != _MAGIC or not isinstance(entries, dict):
        return None
    try:
        kind = NotifKind(kind_value)
    except ValueError:
        return None
    return HandshakeNotif(kind=kind, entries=entries)
