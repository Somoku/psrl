"""Distinguishing stale descriptors from a stale version (implementation item E2).

The design's criterion is that the two trigger *different* recovery paths, and the reason
is that each recovery is a no-op for the other failure:

* re-fetching descriptors when the version is gone yields correct addresses for bytes
  nobody has;
* switching source when descriptors are stale carries the wrong addresses to a new peer.

Conflating them gives a retry loop that cannot converge, which presents as a hang rather
than an error.
"""

from __future__ import annotations

import pytest

from psrl.utils.nixl.staleness import (
    PullRequest,
    Recovery,
    RegistrationEpoch,
    SourceState,
    Staleness,
    classify,
    recovery_for,
)

pytestmark = pytest.mark.cpu_test


def _source(epoch: int = 1, versions=(5,)) -> SourceState:
    return SourceState(registration_epoch=epoch, versions_held=frozenset(versions))


class TestClassification:
    def test_matching_epoch_and_held_version_is_fresh(self):
        request = PullRequest(version=5, chunk=0, descriptor_epoch=1)

        assert classify(request, _source()) is Staleness.FRESH

    def test_old_descriptor_epoch_is_desc_stale(self):
        request = PullRequest(version=5, chunk=0, descriptor_epoch=0)

        assert classify(request, _source(epoch=1)) is Staleness.DESC_STALE

    def test_unheld_version_is_chunk_stale(self):
        request = PullRequest(version=4, chunk=0, descriptor_epoch=1)

        assert classify(request, _source(epoch=1, versions=(5,))) is Staleness.CHUNK_STALE

    def test_descriptor_staleness_is_reported_first_when_both_are_stale(self):
        """Fixing the source first would carry the bad addresses to the new peer."""
        request = PullRequest(version=4, chunk=0, descriptor_epoch=0)

        assert classify(request, _source(epoch=2, versions=(5,))) is Staleness.DESC_STALE

    def test_a_newer_descriptor_epoch_is_also_stale(self):
        """Mismatch, not ordering: a client ahead of the source is equally unserviceable."""
        request = PullRequest(version=5, chunk=0, descriptor_epoch=9)

        assert classify(request, _source(epoch=1)) is Staleness.DESC_STALE


class TestRecoveryPathsDiffer:
    """The design's actual criterion for this item."""

    def test_the_two_staleness_kinds_recover_differently(self):
        assert recovery_for(Staleness.DESC_STALE) != recovery_for(Staleness.CHUNK_STALE)

    def test_desc_stale_refetches_rather_than_switching_source(self):
        """Switching source leaves the same wrong addresses in hand."""
        assert recovery_for(Staleness.DESC_STALE) is Recovery.REFETCH_DESCRIPTORS

    def test_chunk_stale_switches_source_rather_than_refetching(self):
        """Re-fetching yields valid addresses for bytes this source does not have."""
        assert recovery_for(Staleness.CHUNK_STALE) is Recovery.RESELECT_SOURCE

    def test_fresh_proceeds(self):
        assert recovery_for(Staleness.FRESH) is Recovery.PROCEED

    def test_every_staleness_kind_has_a_recovery(self):
        """A new kind must not fall through to a blind retry."""
        for kind in Staleness:
            assert isinstance(recovery_for(kind), Recovery)


class TestEpochAdvancesOnReshardNotJustReregistration:
    """The case the design singles out, because an address check cannot see it."""

    def test_reregistration_advances_the_epoch(self):
        epoch = RegistrationEpoch()

        assert epoch.note_reregistered() == 1

    def test_resharding_advances_the_epoch_although_addresses_are_unchanged(self):
        """Same bytes, different cuts: a remote's cached descriptors now mean something else.

        Without this, an address-equality check passes and the read returns a neighbouring
        shard's bytes -- successfully, with no error anywhere.
        """
        epoch = RegistrationEpoch(value=3)

        assert epoch.note_resharded() == 4

    def test_a_request_from_before_a_reshard_is_rejected(self):
        """End to end: the reshard must make an in-flight client's descriptors stale."""
        epoch = RegistrationEpoch(value=1)
        request = PullRequest(version=5, chunk=0, descriptor_epoch=epoch.value)
        assert classify(request, _source(epoch=epoch.value)) is Staleness.FRESH

        epoch.note_resharded()

        assert classify(request, _source(epoch=epoch.value)) is Staleness.DESC_STALE
