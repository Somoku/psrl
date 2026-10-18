"""Write-gate preemption in the pull engine: acknowledge, repair, or refuse.

The writer preempts a pull that is reading the parameters it is about to overwrite. The
reader must:

* acknowledge over the notification channel (a control round trip, not a data drain);
* finish by re-reading the keys it had not completed from same-node holders of the same
  version, which lands one version and keeps the replica available;
* refuse to land anything when no holder can supply every suspect key, so the server
  waits for the next version instead of serving a mixture.

These tests use the repository's CPU cluster (real clients, planner, engine, and memcpy
transport) with the notification bus standing in for NIXL notifications.
"""

from __future__ import annotations

import pytest
from psrl.utils.nixl.handshake_notif import NotifKind, decode, encode
from psrl.utils.nixl.pull_engine import PullPreempted

from tests.nixl.test_pull_engine import Cluster

pytestmark = pytest.mark.cpu_test

VERSION = 7
READER = "NIXLGenClient_I0_R0"
HOLDER = "NIXLGenClient_I1_R0"


def _ack_fn(cluster: Cluster, reader: str, acked: list[int]):
    """The worker's acknowledgement: record locally and send over the notif channel."""

    def ack(epoch: int) -> None:
        acked.append(epoch)
        cluster.clients[reader].send_pull_preempt_ack(epoch)

    return ack


def _preempt(cluster: Cluster, reader: str, epoch: int, holders: list[str], version: int = VERSION) -> None:
    cluster.bus.send(
        "server",
        reader,
        encode(
            NotifKind.PULL_PREEMPT,
            {reader: {"epoch": epoch, "version": version, "holders": holders}},
        ),
    )


def _acks(cluster: Cluster) -> list[int]:
    """Epochs acknowledged by readers, drained from the notification bus."""
    seen: list[int] = []
    for _sender, payloads in cluster.bus.drain("server").items():
        for raw in payloads:
            notif = decode(raw)
            if notif is not None and notif.kind is NotifKind.PULL_PREEMPT_ACK:
                seen.extend(int(entry["epoch"]) for entry in notif.entries.values())
    return seen


def test_a_preempted_pull_repairs_its_unfinished_keys_from_a_holder():
    cluster = Cluster(train_tp=4, gen_tp=1, instances=2, peer_diffusion=True)
    cluster.pull(epoch=1)  # both instances hold VERSION before the write
    cluster.judge(cluster.gen, version=VERSION)

    # The next pull starts at epoch 0; the write announces epoch 1.
    _preempt(cluster, READER, epoch=1, holders=[HOLDER])
    acked: list[int] = []
    report = cluster.engines[READER].pull(
        epoch=2,
        participants=None,
        pull_epoch=0,
        on_preempt_ack=_ack_fn(cluster, READER, acked),
    )

    assert acked == [1], "the reader must acknowledge the preemption it observed"
    assert _acks(cluster) == [1], "the acknowledgement must go over the notification channel"
    assert report.repaired, "the pull should have been completed from a certified holder"
    assert report.versions == {VERSION}
    cluster.judge([READER, HOLDER], version=VERSION)


def test_a_preempted_pull_without_a_holder_refuses_to_land():
    cluster = Cluster(train_tp=4, gen_tp=1, instances=2, peer_diffusion=True)
    cluster.pull(epoch=1)
    cluster.judge(cluster.gen, version=VERSION)

    _preempt(cluster, READER, epoch=1, holders=[])
    acked: list[int] = []
    with pytest.raises(PullPreempted) as raised:
        cluster.engines[READER].pull(
            epoch=2,
            participants=None,
            pull_epoch=0,
            on_preempt_ack=_ack_fn(cluster, READER, acked),
        )
    assert acked == [1]
    assert not raised.value.repaired
    assert raised.value.suspect_keys, "every unfinished key must be reported suspect"


def test_a_preemption_for_an_older_epoch_is_ignored():
    """A late notice from a previous write must not abort the current pull."""
    cluster = Cluster(train_tp=4, gen_tp=1, instances=2, peer_diffusion=True)
    cluster.pull(epoch=1)

    _preempt(cluster, READER, epoch=0, holders=[HOLDER])
    acked: list[int] = []
    report = cluster.engines[READER].pull(
        epoch=2,
        participants=None,
        pull_epoch=1,
        on_preempt_ack=_ack_fn(cluster, READER, acked),
    )
    assert acked == []
    assert not report.repaired
    cluster.judge([READER, HOLDER], version=VERSION)


def test_a_reader_with_nothing_in_flight_is_not_acknowledged():
    """The writer preempts only admitted pulls; a notice arriving outside one is stale."""
    cluster = Cluster(train_tp=4, gen_tp=1, instances=2, peer_diffusion=True)
    cluster.pull(epoch=1)

    _preempt(cluster, READER, epoch=1, holders=[HOLDER])
    # No pull is running, so nothing drains the notice; the next pull with a newer
    # admission epoch still sees it, but its start epoch makes it stale.
    acked: list[int] = []
    cluster.engines[READER].pull(
        epoch=2,
        participants=None,
        pull_epoch=1,
        on_preempt_ack=_ack_fn(cluster, READER, acked),
    )
    assert acked == []
