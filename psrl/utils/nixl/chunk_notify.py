"""Per-chunk completion signalling, so the peer-diffusion pipeline can actually run (D1).

`chunk_pipeline` computes a schedule in which a follower's peer read of chunk ``i`` overlaps
the designate's upstream read of chunk ``i+1``. Nothing could drive that schedule: the
dependency is *cross-client* -- "my designate has finished chunk i" -- and a wave counter
local to one actor cannot express it. The two-phase fallback is the degenerate case of
waiting for the last chunk, which is why its cost added instead of hiding.

This module is the missing edge. The signal travels designate -> follower, which is the
**reverse** of the data flow (the follower pulls bytes from the designate). That direction
is what keeps the mechanism P2P: no coordinator is asked "is chunk i ready", so the
control plane does not grow a per-chunk round trip -- the thing C2 existed to remove.

Ordering is established by the designate, not by the network. The designate publishes only
after its own completion check for chunk ``i`` has returned, so by the time a follower can
observe the notification the bytes are already in the designate's buffer. A notification
that overtakes data is therefore not possible, and the follower needs no fence of its own.

Three hazards shape the API; each one silently corrupts rather than erroring:

1. **The notification stream is shared.** ``get_new_notifs()`` drains *everything* addressed
   to the agent -- handshake payloads, client-info broadcasts, these chunk signals. A
   consumer that drops what it does not recognise destroys another subsystem's message.
   So foreign payloads are handed to ``stash_foreign`` rather than discarded.

2. **Identification must not be by trial deserialisation.** The existing handshake waiters
   do `pickle.loads` in a `try` and `continue` on failure, which means any payload that
   happens to unpickle into a dict is *misread* rather than rejected. Chunk signals carry a
   byte prefix and are classified before anything is deserialised.

3. **Delivery is batched and unordered.** One drain can return chunks ``i``, ``i+1`` and
   ``i+2`` together, and a waiter that matches only the chunk it currently wants throws the
   others away and then blocks forever on them. Signals accumulate into a set; waiting is a
   membership test over what has arrived, never a test of what arrived *last*.

The version travels with every signal because a peer source is only valid at the same
version (design 3.5.2). Keying on ``(version, chunk, sender)`` makes a stale signal unable
to satisfy a current wait, instead of making it a judgement call at the call site.
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))

# Chunk signals are identified by this prefix before any deserialisation. See hazard 2.
_PREFIX = b"psrl-chunk-done:"


@dataclass(frozen=True)
class ChunkDone:
    """One designate reporting that one chunk of one version has landed in its buffer."""

    version: int
    chunk: int
    sender: str


def encode_chunk_done(version: int, chunk: int, sender: str) -> bytes:
    """Serialise a completion signal, prefixed so receivers can classify without unpickling."""
    return _PREFIX + pickle.dumps((version, chunk, sender), protocol=pickle.HIGHEST_PROTOCOL)


def decode_chunk_done(payload: bytes) -> ChunkDone | None:
    """Return the signal, or None when the payload belongs to another subsystem.

    None is not an error: it is the common case on a shared stream. The caller is expected
    to hand those payloads back rather than drop them.
    """
    if not isinstance(payload, (bytes, bytearray)) or not payload.startswith(_PREFIX):
        return None
    try:
        version, chunk, sender = pickle.loads(bytes(payload[len(_PREFIX) :]))
    except Exception:
        # Prefix matched but the body did not parse. This is a genuine fault in our own
        # encoding rather than a foreign message, so it must be visible, not stashed.
        psrl_logger.error("chunk-done payload carried our prefix but failed to parse")
        return None
    return ChunkDone(version=int(version), chunk=int(chunk), sender=str(sender))


class ChunkCompletionPublisher:
    """Designate side: announce that a chunk has landed.

    The follower set comes from the comm plan, so a designate with no followers publishes
    nothing and pays nothing -- the common case for a rollout that is nobody's source.
    """

    def __init__(self, sender_name: str, send: Callable[[str, bytes], None]) -> None:
        self._sender_name = sender_name
        self._send = send
        self._published: list[tuple[int, int, str]] = []

    @property
    def published(self) -> list[tuple[int, int, str]]:
        """``(version, chunk, follower)`` triples sent so far, for assertions and tracing."""
        return list(self._published)

    def publish(self, version: int, chunk: int, followers: Sequence[str]) -> int:
        """Tell each follower that ``chunk`` of ``version`` is readable. Returns notif count.

        Must be called only after the local completion check for ``chunk`` has returned;
        publishing early is the one way to break the ordering guarantee this module rests
        on, and it would surface as a follower reading an unfilled buffer *successfully*.
        """
        if not followers:
            return 0
        payload = encode_chunk_done(version, chunk, self._sender_name)
        sent = 0
        for follower in followers:
            if follower == self._sender_name:
                continue
            self._send(follower, payload)
            self._published.append((version, chunk, follower))
            sent += 1
        return sent


class ChunkCompletionWaiter:
    """Follower side: block until named designates report a chunk.

    Signals accumulate, so arrival order and batching do not matter (hazard 3). Anything
    not ours goes to ``stash_foreign`` (hazard 1); when no stash is supplied the waiter
    refuses to run rather than quietly dropping other subsystems' messages.
    """

    def __init__(
        self,
        drain: Callable[[], dict[str, list[bytes]]],
        stash_foreign: Callable[[str, bytes], None],
        poll_interval: float = 0.0005,
    ) -> None:
        self._drain = drain
        self._stash_foreign = stash_foreign
        self._poll_interval = poll_interval
        self._seen: set[tuple[int, int, str]] = set()
        self._drains = 0

    @property
    def drains(self) -> int:
        """Number of drain calls, so a test can tell a real wait from a free one."""
        return self._drains

    def seen(self, version: int, chunk: int, sender: str) -> bool:
        return (version, chunk, sender) in self._seen

    def absorb(self) -> int:
        """Drain once, keeping our signals and handing the rest back. Returns new signals."""
        self._drains += 1
        gained = 0
        for agent_name, payloads in self._drain().items():
            for payload in payloads:
                signal = decode_chunk_done(payload)
                if signal is None:
                    self._stash_foreign(agent_name, payload)
                    continue
                if (signal.version, signal.chunk, signal.sender) not in self._seen:
                    self._seen.add((signal.version, signal.chunk, signal.sender))
                    gained += 1
        return gained

    def wait_for(
        self,
        version: int,
        chunk: int,
        sources: Iterable[str],
        timeout: float = 600.0,
    ) -> None:
        """Block until every source has reported ``(version, chunk)``.

        ``sources`` is every designate this follower peer-reads ``chunk`` from -- a chunk
        can hold keys served by more than one designate, and proceeding once the first has
        reported would read the others' buffers before they are filled.

        A timeout names the exact triples that never arrived. That matters more than the
        failure itself: the symptom of a missing edge in the notification graph is a hang,
        and without the names there is nothing to distinguish it from a slow designate.
        """
        pending = {source for source in sources if not self.seen(version, chunk, source)}
        if not pending:
            return
        start = time.time()
        while pending:
            self.absorb()
            pending = {source for source in pending if not self.seen(version, chunk, source)}
            if not pending:
                return
            if time.time() - start > timeout:
                raise TimeoutError(
                    f"timed out after {timeout}s waiting for chunk {chunk} of version "
                    f"{version} from {sorted(pending)}; received "
                    f"{sorted(s for s in self._seen if s[0] == version and s[1] == chunk)}"
                )
            time.sleep(self._poll_interval)

    def forget_versions_before(self, version: int) -> int:
        """Drop signals for older versions. Returns how many were dropped.

        Without this the set grows by ``chunks x designates`` every step for the life of the
        process. Pruning by version rather than by age is what keeps it safe: a signal is
        useless exactly when its version can no longer be pulled.
        """
        stale = {signal for signal in self._seen if signal[0] < version}
        self._seen -= stale
        return len(stale)
