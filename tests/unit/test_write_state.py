"""Optimistic commit validation: the v2 write gate's state machine.

These tests drive `WriteState` directly. They pin the property the cluster tests cannot
enumerate: for every interleaving of a pull and an optimizer write, the pull either
commits a version it read entirely before the write, or is rejected (and the caller
re-pulls). A mixed read can never be committed.
"""

from __future__ import annotations

import itertools

import pytest
from psrl.workers.ps.write_state import (
    CLOSING,
    WRITING,
    WriteState,
)

pytestmark = pytest.mark.cpu_test

A = (0, 0)
B = (1, 0)
C = (2, 0)


def _admit(state: WriteState, instance=A, version=1):
    admission = state.begin_pull(instance, version)
    assert admission.admitted, admission.reason
    return admission.token


def test_a_pull_with_no_write_commits_its_version():
    state = WriteState(version=1)
    _admit(state, version=1)
    decision = state.finish_pull(A)
    assert decision.accepted and decision.version == 1
    assert state.holders(1) == frozenset({A})


def test_a_write_between_start_and_commit_rejects_the_pull():
    state = WriteState(version=1)
    _admit(state, version=1)
    state.begin_write()
    state.start_writing()
    state.publish(2)
    decision = state.finish_pull(A)
    assert not decision.accepted and decision.reason == "epoch_moved"


def test_an_acknowledged_and_repaired_pull_commits_the_old_version():
    state = WriteState(version=1)
    _admit(state, version=1)
    batch = state.begin_write()
    assert state.ack(A, batch.epoch)
    state.start_writing()
    # The repair finishes while the write is still open; the reader keeps its old version.
    decision = state.finish_pull(A, repaired=True)
    assert decision.accepted and decision.version == 1
    state.publish(2)
    assert state.holders(1) == frozenset({A})


def test_repair_without_an_acknowledgement_is_rejected():
    state = WriteState(version=1)
    _admit(state, version=1)
    state.begin_write()
    state.start_writing()
    assert not state.finish_pull(A, repaired=True).accepted


def test_a_stale_acknowledgement_is_ignored():
    state = WriteState(version=1)
    _admit(state, version=1)
    batch = state.begin_write()
    assert not state.ack(A, batch.epoch + 1)
    assert not state.ack(B, batch.epoch)  # B never started a pull


def test_admission_is_closed_while_the_writer_holds_the_version():
    state = WriteState(version=1)
    state.begin_write()
    for phase in (CLOSING, WRITING):
        state.state = phase
        assert not state.begin_pull(B, 1).admitted
    state.start_writing()
    state.publish(2)
    assert state.begin_pull(B, 2).admitted


def test_holders_are_pinned_for_the_write_and_released_on_publish():
    state = WriteState(version=1)
    _admit(state, A, 1)
    state.finish_pull(A)
    state.publish(2)
    _admit(state, B, 2)
    state.finish_pull(B)
    # A pull is in flight, so the write must enumerate and pin the version's holders as
    # repair sources; without a reader the gate skips that work entirely.
    _admit(state, C, 2)
    batch = state.begin_write()
    assert state.pinned() == frozenset({B})
    assert not state.begin_pull((3, 0), 2).admitted  # the state is closed
    assert B in batch.holders and A not in batch.holders
    state.publish(3)
    assert state.pinned() == frozenset()


def test_a_pull_retried_after_the_write_reports_the_new_epoch():
    """An admission may predate a completed write; the attempt that reads afterwards can
    still commit, because it reports the epoch it actually read at."""
    state = WriteState(version=1)
    token = _admit(state, version=1)
    assert token.start_epoch == 0
    state.begin_write()
    state.start_writing()
    state.publish(2)
    decision = state.finish_pull(A, pull_epoch=state.epoch, landed_version=2)
    assert decision.accepted and decision.version == 2
    assert state.holders(2) == frozenset({A})


def test_a_retry_after_publish_records_the_published_version():
    """The engine's shard stamps lag the coordinator; the commit must not reject on them.

    After a write publishes, a retry reads the new version but may still report the old
    per-shard stamp because rollout-side stamps are only refreshed on re-registration.
    The epoch is the authority: unchanged epoch means the published version is what was
    read, and that is what the staleness ledger records.
    """
    state = WriteState(version=1)
    _admit(state, version=1)
    state.begin_write()
    state.start_writing()
    state.publish(2)
    decision = state.finish_pull(A, pull_epoch=state.epoch, landed_version=1)
    assert decision.accepted and decision.version == 2
    assert state.holders(2) == frozenset({A})


def test_a_second_write_invalidates_an_earlier_repairs_certificate():
    state = WriteState(version=1)
    _admit(state, version=1)
    first = state.begin_write()
    assert state.ack(A, first.epoch)
    state.start_writing()
    state.publish(2)
    # New pull of v2, then another write. The certificate still names epoch 1, which is no
    # longer current, so it cannot be used to commit after the second write began.
    _admit(state, version=2)
    second = state.begin_write()
    assert second.epoch == first.epoch + 1
    assert not state.finish_pull(A, repaired=True, pull_epoch=first.epoch).accepted


def test_aborted_write_keeps_the_version_closed():
    state = WriteState(version=1)
    _admit(state, version=1)
    state.begin_write()
    state.start_writing()
    state.abort_write()
    assert state.state == WRITING
    assert not state.begin_pull(B, 1).admitted
    # Recovery: the next update bumps the epoch again and closes normally.
    batch = state.begin_write()
    assert batch.epoch == 2


def test_abandoned_pulls_are_forgotten():
    state = WriteState(version=1)
    _admit(state, version=1)
    state.abort_pull(A)
    assert state.inflight() == frozenset()
    assert not state.finish_pull(A).accepted


@pytest.mark.parametrize(
    "pull_starts_before_write,acks,repaired",
    list(itertools.product([True, False], [True, False], [True, False])),
)
def test_every_interleaving_commits_only_a_believable_version(
    pull_starts_before_write: bool,
    acks: bool,
    repaired: bool,
):
    """Exhaustive truth table for the commit decision.

    ``pull_starts_before_write`` selects whether the pull was admitted before the epoch
    bumped. ``acks`` and ``repaired`` are what the reader reports. The commit is accepted
    exactly when the pull is provably single-version: it read at the current epoch, or it
    was acknowledged before the write and repaired from certified holders.
    """
    state = WriteState(version=1)
    if pull_starts_before_write:
        _admit(state, version=1)
        batch = state.begin_write()
        if acks:
            assert state.ack(A, batch.epoch)
        state.start_writing()
        decision = state.finish_pull(A, repaired=repaired, pull_epoch=0, landed_version=1)
        if acks and repaired:
            assert decision.accepted and decision.version == 1
        else:
            assert not decision.accepted
        return

    batch = state.begin_write()
    state.start_writing()
    state.publish(2)
    _admit(state, version=2)
    if acks:
        state.ack(A, batch.epoch)  # stale: it belongs to the previous write
    decision = state.finish_pull(A, repaired=repaired, pull_epoch=state.epoch, landed_version=2)
    assert decision.accepted and decision.version == 2


def test_holders_query_tracks_the_latest_committed_version():
    state = WriteState(version=0)
    for version in (1, 2, 3):
        state.publish(version)
        _admit(state, A, version)
        assert state.finish_pull(A).accepted
    assert state.holders(3) == frozenset({A})
    assert state.holders(2) == frozenset()


def test_a_steady_state_write_does_not_enumerate_the_fleet():
    """With no reader in flight there is nothing to preempt, so the write is O(1).

    This is the property that lets 10k-instance training steps pay nothing for the gate:
    holder enumeration and notification fanout are skipped entirely.
    """
    state = WriteState(version=1)
    for i in range(10_000):
        state.record_published((i, 0), 1)
    batch = state.begin_write()
    assert batch.readers == ()
    assert batch.holders == ()
    assert state.pinned() == frozenset()
