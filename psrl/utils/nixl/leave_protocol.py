"""Revoking a departing member's descriptors, with confirmation (item E4, lower half).

The design's shrink is four steps: drop holder status, drain in-flight readers, revoke
the descriptors the member handed out, then release its memory. Step three was recorded
for several rounds as "blocked, NIXL has no `invalidateLocalMD` binding". **That was
wrong twice over.** The binding is `invalidate_local_metadata` (the probe asked for
`invalidate_local_md`), and PSRL's own `NIXLMetaServer.shutdown` has been calling both it
and `remove_remote_agent` all along.

**What revocation actually is.** Reading the P2P backend settles what the two calls do:

    invalidateLocal(ip, port)  ->  sendToPeer(ip, port, "NIXLCOMM:INVL" + myName)
    peer listener receives INVL ->  ctx_.invalidateRemoteMD(myName)

So `invalidate_local_metadata` is nothing but a *remote trigger* for the peer to run
`remove_remote_agent(me)`. The eviction itself is always local to the holder of the
descriptors. Three consequences decide the protocol below:

* **It is not a broadcast.** The broadcast form (`ip_addr=""`) routes to a central
  metadata store; PSRL has none -- metadata travels out-of-band through the server's
  notification channel and is loaded with `add_remote_agent(bytes)`. The P2P form needs
  one call per peer, each carrying that peer's listen address, which `NIXLClientInfo`
  does not even publish.
* **It cannot be confirmed.** `invalidateLocal` submits to a worker queue and returns;
  failures are logged, not reported. Step four (free the memory) is only safe once every
  holder has *actually* forgotten, and a fire-and-forget call cannot establish that.
  This, not a missing binding, was the real gap.
* **It needs no new transport.** Making a peer call `remove_remote_agent` is something
  PSRL already does on the notification channel: `wait_for_update_infos` evicts a peer
  that re-registered. A leave is the same eviction with no replacement metadata behind
  it, so it rides the same channel and can be acknowledged on it.

**Measured, not assumed: eviction does invalidate descriptors already handed out.**
Two agents, one region, a successful READ, then `remove_remote_agent` on the reader:

    make_prepped_xfer from the pre-eviction dlist  ->  NIXL_ERR_NOT_FOUND
    repost of the pre-eviction request handle      ->  NIXL_ERR_NOT_FOUND

NIXL weakly references the remote section (`nixl_agent.cpp` checks
`remoteSection.expired()` in `makeXferReq`, `postXferReq` and `getXferStatus`), so a
stale descriptor fails loudly instead of reading freed pages. That is exactly the
property §3.4.5 wanted step three for, and it means PSRL needs no epoch of its own here
-- only the guarantee that the eviction ran everywhere before the memory goes.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from nixl._bindings import nixlNotFoundError

from .handshake_notif import NotifKind, encode

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))


def encode_member_leave(leaver: str, leaver_agent: str) -> bytes:
    """Server -> peer: forget ``leaver``.

    The agent name travels in the payload rather than being looked up by the receiver,
    because a peer may be told to forget a client whose info it never cached -- and the
    whole point of the message is that the leaver may already be gone.
    """
    return encode(NotifKind.MEMBER_LEAVE, {leaver: {"agent": leaver_agent}})


def encode_leave_ack(leaver: str, held: bool) -> bytes:
    """Peer -> server: the eviction for ``leaver`` has run here.

    No sender identity: `get_new_notifs` already keys deliveries by the sending agent,
    and the *agent* is the right granularity -- `remove_remote_agent` evicts for every
    client the agent hosts at once, so a per-client ack would claim more than one call
    can promise.

    ``held`` says whether this peer had the leaver's metadata loaded, taken from PSRL's
    own connection cache rather than from NIXL (see `revoke_descriptors`: NIXL cannot
    answer it). False is still an ack -- "nothing of the leaver is addressable here" is
    the state the shrink wants, and under a comm plan most pairs never load each other.
    """
    return encode(NotifKind.LEAVE_ACK, {leaver: {"held": bool(held)}})


def leaver_agent_of(entries: dict[str, Any], leaver: str) -> str:
    """Agent name carried by a MEMBER_LEAVE entry."""
    return str(entries[leaver].get("agent", ""))


def revoke_descriptors(agent, remote_agent_name: str) -> None:
    """Make this agent forget ``remote_agent_name``.

    Returns nothing on purpose. The obvious signature is "True if it had anything to
    forget", and it is not implementable: measured on this build, `remove_remote_agent`
    returns normally for a peer that was never loaded and for one evicted twice already,
    so there is no answer to report. Returning a fabricated boolean would put a value in
    the ack that nothing backs. Whether a peer *was* held is PSRL-side knowledge --
    the client's connection cache -- and that is where the ack takes it from.

    A not-found remote would in any case be success, not failure: the post-condition is
    "this agent cannot use that peer's descriptors", which never having loaded them
    satisfies. The guard below keeps that true on a build whose bindings do raise, so a
    retried leave -- the normal case after a lost ack -- stays benign.
    """
    try:
        agent.remove_remote_agent(remote_agent_name)
    except nixlNotFoundError:
        pass


@dataclass(frozen=True)
class RevokeOutcome:
    """Which holders confirmed the eviction, and which did not.

    Carries the unconfirmed set rather than a bare boolean because the caller's choice
    depends on *who* is missing: an unreachable peer that is itself being torn down is
    not the same risk as a live reader that never answered.
    """

    leaver: str
    required: frozenset[str]
    acked: frozenset[str]

    @property
    def missing(self) -> frozenset[str]:
        return frozenset(self.required - self.acked)

    @property
    def complete(self) -> bool:
        return not self.missing

    def describe(self) -> str:
        if self.complete:
            return f"{self.leaver}: all {len(self.required)} holder(s) confirmed revocation"
        return (
            f"{self.leaver}: {len(self.missing)} of {len(self.required)} holder(s) did not "
            f"confirm revocation ({sorted(self.missing)}); their descriptors may still be "
            f"loaded, so releasing the leaver's memory is not yet safe"
        )
