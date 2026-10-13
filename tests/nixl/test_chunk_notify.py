"""Tests for per-chunk completion signalling, and for the pipeline it drives.

The point of these tests is to fail when the pipeline only *looks* driven. Three of them
exist specifically because an earlier version of this work passed while the pipeline was
not running at all:

- `test_wait_blocks_until_published` -- a waiter that never blocks is indistinguishable
  from no dependency, which is what the two-phase fallback is.
- `test_foreign_payloads_are_not_dropped` -- the stream is shared with the handshake, so a
  channel that passes its own tests can still break connection setup.
- `test_batched_out_of_order_delivery` -- one drain can return three chunks; a waiter that
  only matches the one it wants discards the rest and then hangs on them.
"""

from __future__ import annotations

import threading

import pytest
from psrl.utils.nixl.chunk_notify import (
    ChunkCompletionPublisher,
    ChunkCompletionWaiter,
    ChunkDone,
    decode_chunk_done,
    encode_chunk_done,
)


class FakeFabric:
    """A notification fabric with no ordering guarantees beyond per-pair FIFO.

    NIXL delivers into a per-agent mailbox that the receiver drains in bulk, so a test
    fabric that hands messages over one at a time in send order would hide hazard 3. This
    one returns everything queued, grouped by sender, exactly as `get_new_notifs` does.
    """

    def __init__(self) -> None:
        self.mailboxes: dict[str, list[tuple[str, bytes]]] = {}
        self.lock = threading.Lock()

    def send(self, dst: str, payload: bytes, src: str = "sender") -> None:
        with self.lock:
            self.mailboxes.setdefault(dst, []).append((src, payload))

    def drain_for(self, agent: str):
        def drain() -> dict[str, list[bytes]]:
            with self.lock:
                queued = self.mailboxes.pop(agent, [])
            grouped: dict[str, list[bytes]] = {}
            for src, payload in queued:
                grouped.setdefault(src, []).append(payload)
            return grouped

        return drain


def make_waiter(fabric: FakeFabric, agent: str):
    foreign: list[tuple[str, bytes]] = []
    waiter = ChunkCompletionWaiter(
        drain=fabric.drain_for(agent),
        stash_foreign=lambda name, payload: foreign.append((name, payload)),
        poll_interval=0.0001,
    )
    return waiter, foreign


def test_encode_decode_roundtrip():
    payload = encode_chunk_done(7, 3, "designate_0")
    assert decode_chunk_done(payload) == ChunkDone(version=7, chunk=3, sender="designate_0")


def test_foreign_payloads_are_classified_not_parsed():
    """A foreign payload that happens to unpickle must still be rejected.

    The existing handshake waiters classify by trial `pickle.loads`, so a dict-shaped
    message from another subsystem is *misread* rather than skipped. Prefix classification
    is what makes this impossible, and this test pins it: the payload below unpickles
    cleanly into a dict, and must still be recognised as not-ours.
    """
    import pickle

    assert decode_chunk_done(pickle.dumps({"client_infos": {"a": 1}})) is None
    assert decode_chunk_done(b"") is None
    assert decode_chunk_done(b"psrl-chunk-done-but-not-really") is None


def test_foreign_payloads_are_not_dropped():
    fabric = FakeFabric()
    waiter, foreign = make_waiter(fabric, "follower")
    fabric.send("follower", b"handshake-payload", src="server")
    fabric.send("follower", encode_chunk_done(1, 0, "designate"), src="designate")

    gained = waiter.absorb()

    assert gained == 1
    assert waiter.seen(1, 0, "designate")
    # The handshake payload must be handed back, not consumed. Dropping it is how this
    # channel would break connection setup while passing all of its own tests.
    assert foreign == [("server", b"handshake-payload")]


def test_batched_out_of_order_delivery():
    """Signals accumulate, so a single drain carrying several chunks loses none."""
    fabric = FakeFabric()
    waiter, _ = make_waiter(fabric, "follower")
    for chunk in (2, 0, 1):
        fabric.send("follower", encode_chunk_done(5, chunk, "designate"), src="designate")

    waiter.absorb()

    for chunk in (0, 1, 2):
        assert waiter.seen(5, chunk, "designate")
    # Waiting for an already-arrived chunk must not drain again: that is the difference
    # between a pipeline that reuses signals and one that re-polls per wave.
    before = waiter.drains
    waiter.wait_for(5, 0, ["designate"])
    assert waiter.drains == before


def test_stale_version_cannot_satisfy_a_wait():
    """A peer source is only valid at the same version, so version is part of the key."""
    fabric = FakeFabric()
    waiter, _ = make_waiter(fabric, "follower")
    fabric.send("follower", encode_chunk_done(4, 0, "designate"), src="designate")
    waiter.absorb()

    assert waiter.seen(4, 0, "designate")
    assert not waiter.seen(5, 0, "designate")
    with pytest.raises(TimeoutError):
        waiter.wait_for(5, 0, ["designate"], timeout=0.05)


def test_wait_blocks_until_published():
    """The wait must really block, otherwise there is no dependency being expressed."""
    fabric = FakeFabric()
    waiter, _ = make_waiter(fabric, "follower")
    publisher = ChunkCompletionPublisher("designate", lambda dst, p: fabric.send(dst, p, "designate"))
    released = threading.Event()

    def follower():
        waiter.wait_for(9, 0, ["designate"], timeout=10.0)
        released.set()

    thread = threading.Thread(target=follower, daemon=True)
    thread.start()
    assert not released.wait(timeout=0.2), "waiter returned before the designate published"

    publisher.publish(9, 0, ["follower"])
    assert released.wait(timeout=5.0), "waiter did not wake after publication"
    thread.join(timeout=5.0)


def test_wait_requires_every_source():
    """A chunk served by two designates must wait for both.

    Proceeding on the first report would read the second designate's buffer before it is
    filled -- and that read *succeeds*, so nothing downstream would catch it.
    """
    fabric = FakeFabric()
    waiter, _ = make_waiter(fabric, "follower")
    fabric.send("follower", encode_chunk_done(1, 0, "designate_a"), src="designate_a")

    with pytest.raises(TimeoutError) as excinfo:
        waiter.wait_for(1, 0, ["designate_a", "designate_b"], timeout=0.05)
    # The message must name what is missing: a missing notification edge presents as a
    # hang, and without the name it is indistinguishable from a slow peer.
    assert "designate_b" in str(excinfo.value)
    assert "designate_a" not in str(excinfo.value).split(";")[0]


def test_publisher_skips_itself_and_empty_follower_sets():
    sent: list[tuple[str, bytes]] = []
    publisher = ChunkCompletionPublisher("designate", lambda dst, p: sent.append((dst, p)))

    assert publisher.publish(1, 0, []) == 0
    assert publisher.publish(1, 0, ["designate"]) == 0
    assert publisher.publish(1, 0, ["f0", "f1"]) == 2
    assert [dst for dst, _ in sent] == ["f0", "f1"]


def test_forget_versions_before_prunes_only_older():
    fabric = FakeFabric()
    waiter, _ = make_waiter(fabric, "follower")
    for version in (3, 4, 5):
        fabric.send("follower", encode_chunk_done(version, 0, "designate"), src="designate")
    waiter.absorb()

    dropped = waiter.forget_versions_before(5)

    assert dropped == 2
    assert waiter.seen(5, 0, "designate")
    assert not waiter.seen(4, 0, "designate")


def test_pipeline_deadlocks_are_reported_not_hung():
    """A follower whose designate never publishes must time out with the chunk named."""
    fabric = FakeFabric()
    waiter, _ = make_waiter(fabric, "follower")
    with pytest.raises(TimeoutError) as excinfo:
        waiter.wait_for(1, 2, ["designate"], timeout=0.05)
    assert "chunk 2" in str(excinfo.value)
