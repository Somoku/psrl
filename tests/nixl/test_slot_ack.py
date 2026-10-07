"""Wire-buffer slot reclamation (implementation item B2).

The cast path's wire buffer is cyclic and only ``2 x chunk``, so the next cast can
overwrite a slot a slow reader is still reading. These tests pin the two properties
that make that impossible:

* a slot is never handed out while a reader still owes an acknowledgement -- `acquire`
  raises instead of overwriting, because overwriting is silent
* a reader that disappears cannot hold a slot forever -- `forget_reader` converges,
  which is the specific advantage of a bitmap over the in-flight counter used for the
  same-precision path

Memory is deliberately *not* the remedy: the slow-reader test uses two slots and a
reader that never acknowledges, and asserts the table refuses rather than growing.
"""

from __future__ import annotations

import pytest

from psrl.utils.nixl.slot_ack import SlotAckTable, SlotBusyError

pytestmark = pytest.mark.cpu_test

VERSION = 7


class TestAcquireAndRelease:
    def test_acquire_uses_a_free_slot(self):
        table = SlotAckTable(slot_count=2)
        index = table.acquire(chunk_id=0, version=VERSION, expected_readers={1, 2})
        assert index in (0, 1)
        assert table.chunk_in_slot(index) == 0
        assert table.version_in_slot(index) == VERSION

    def test_slot_frees_when_every_reader_acknowledges(self):
        table = SlotAckTable(slot_count=2)
        table.acquire(chunk_id=0, version=VERSION, expected_readers={1, 2})

        assert table.acknowledge(0, reader_id=1) is False, "freed before all readers acked"
        assert table.acknowledge(0, reader_id=2) is True
        assert len(table.free_slots()) == 2
        assert table.reclaim_count == 1

    def test_partial_acknowledgement_keeps_the_slot(self):
        table = SlotAckTable(slot_count=2)
        table.acquire(chunk_id=0, version=VERSION, expected_readers={1, 2, 3})
        table.acknowledge(0, reader_id=1)

        assert table.outstanding_readers(0) == {2, 3}
        assert len(table.free_slots()) == 1

    def test_reacquiring_a_live_chunk_is_rejected(self):
        table = SlotAckTable(slot_count=2)
        table.acquire(chunk_id=0, version=VERSION, expected_readers={1})
        with pytest.raises(SlotBusyError, match="already occupies slot"):
            table.acquire(chunk_id=0, version=VERSION, expected_readers={1})

    def test_explicit_release_frees_regardless(self):
        """The timeout path (CHUNK_STALE) releases without full acknowledgement."""
        table = SlotAckTable(slot_count=1)
        table.acquire(chunk_id=0, version=VERSION, expected_readers={1, 2})
        table.release(0)
        assert table.free_slots() == [0]

    def test_releasing_an_unknown_chunk_is_a_no_op(self):
        table = SlotAckTable(slot_count=1)
        table.release(99)
        assert table.free_slots() == [0]

    def test_slot_count_must_be_positive(self):
        with pytest.raises(ValueError, match="at least 1"):
            SlotAckTable(slot_count=0)


class TestSlowReaderDoesNotCorruptData:
    """The correctness property: refuse the slot rather than overwrite it."""

    def test_exhausted_slots_raise_instead_of_overwriting(self):
        table = SlotAckTable(slot_count=2)
        table.acquire(chunk_id=0, version=VERSION, expected_readers={1})
        table.acquire(chunk_id=1, version=VERSION, expected_readers={1})

        with pytest.raises(SlotBusyError, match="no free slot"):
            table.acquire(chunk_id=2, version=VERSION, expected_readers={1})

    def test_error_names_the_readers_being_waited_on(self):
        """An operator needs to know which instance is stalling the pipeline."""
        table = SlotAckTable(slot_count=1)
        table.acquire(chunk_id=0, version=VERSION, expected_readers={3, 4})

        with pytest.raises(SlotBusyError) as excinfo:
            table.acquire(chunk_id=1, version=VERSION, expected_readers={3})
        message = str(excinfo.value)
        assert "3" in message and "4" in message
        assert "partly overwritten" in message, "the message should say why waiting is right"

    def test_one_slow_reader_does_not_block_a_different_chunk(self):
        """Other slots keep cycling while one is stuck."""
        table = SlotAckTable(slot_count=2)
        table.acquire(chunk_id=0, version=VERSION, expected_readers={1, 2})
        table.acknowledge(0, reader_id=1)  # reader 2 never acknowledges

        slot = table.acquire(chunk_id=1, version=VERSION, expected_readers={1})
        assert table.acknowledge(1, reader_id=1) is True
        assert slot in table.free_slots()
        assert table.outstanding_readers(0) == {2}, "the stuck chunk is still waiting"

    def test_deepening_is_not_required_for_many_readers(self):
        """Two slots serve any number of readers, as long as they acknowledge.

        This is the claim that keeps the wire buffer at 2 x chunk instead of W x chunk.
        """
        table = SlotAckTable(slot_count=2)
        readers = set(range(16))

        for chunk_id in range(10):
            table.acquire(chunk_id=chunk_id, version=VERSION, expected_readers=readers)
            for reader_id in readers:
                table.acknowledge(chunk_id, reader_id)

        assert table.slot_count == 2
        assert table.reclaim_count == 10


class TestReaderLossConverges:
    """A counter would leak here; the bitmap is chosen precisely for this."""

    def test_forgetting_a_reader_can_satisfy_a_slot(self):
        table = SlotAckTable(slot_count=2)
        table.acquire(chunk_id=0, version=VERSION, expected_readers={1, 2})
        table.acknowledge(0, reader_id=1)

        satisfied = table.forget_reader(2)

        assert satisfied == [0]
        assert len(table.free_slots()) == 2

    def test_forgetting_the_only_reader_frees_the_slot(self):
        table = SlotAckTable(slot_count=1)
        table.acquire(chunk_id=0, version=VERSION, expected_readers={5})

        assert table.forget_reader(5) == [0]
        assert table.free_slots() == [0]

    def test_forgetting_an_unrelated_reader_changes_nothing(self):
        table = SlotAckTable(slot_count=1)
        table.acquire(chunk_id=0, version=VERSION, expected_readers={1})

        assert table.forget_reader(99) == []
        assert table.outstanding_readers(0) == {1}

    def test_forgetting_frees_several_slots_at_once(self):
        """A dead instance may be the last holdout on multiple chunks."""
        table = SlotAckTable(slot_count=3)
        for chunk_id in range(3):
            table.acquire(chunk_id=chunk_id, version=VERSION, expected_readers={1, 2})
            table.acknowledge(chunk_id, reader_id=1)

        satisfied = table.forget_reader(2)

        assert sorted(satisfied) == [0, 1, 2]
        assert len(table.free_slots()) == 3


class TestLateAndDuplicateAcknowledgements:
    """Retries and late notifications are normal and must not corrupt accounting."""

    def test_duplicate_acknowledgement_is_harmless(self):
        table = SlotAckTable(slot_count=1)
        table.acquire(chunk_id=0, version=VERSION, expected_readers={1, 2})

        table.acknowledge(0, reader_id=1)
        table.acknowledge(0, reader_id=1)

        assert table.outstanding_readers(0) == {2}

    def test_acknowledgement_after_release_is_counted_not_fatal(self):
        table = SlotAckTable(slot_count=1)
        table.acquire(chunk_id=0, version=VERSION, expected_readers={1})
        table.acknowledge(0, reader_id=1)

        assert table.acknowledge(0, reader_id=1) is False
        assert table.stale_acks == 1

    def test_unexpected_reader_acknowledgement_still_completes_the_slot(self):
        """A reader not in the plan acking is odd but must not wedge the slot."""
        table = SlotAckTable(slot_count=1)
        table.acquire(chunk_id=0, version=VERSION, expected_readers={1})

        table.acknowledge(0, reader_id=99)
        assert not table.free_slots(), "an unexpected ack must not satisfy the slot"
        assert table.acknowledge(0, reader_id=1) is True

    def test_empty_expectation_frees_immediately_on_acquire_cycle(self):
        """A chunk nobody reads -- possible when a plan assigns it zero readers."""
        table = SlotAckTable(slot_count=1)
        table.acquire(chunk_id=0, version=VERSION, expected_readers=set())
        # Satisfied from the start; an explicit release models the publish step.
        assert table.outstanding_readers(0) == set()
        table.release(0)
        assert table.free_slots() == [0]


class TestSteadyState:
    def test_pipeline_cycles_without_growth(self):
        """20 chunks through 2 slots: the invariant the memory accounting rests on."""
        table = SlotAckTable(slot_count=2)
        readers = {1, 2, 3}
        in_flight: list[int] = []

        for chunk_id in range(20):
            if len(table.free_slots()) == 0:
                oldest = in_flight.pop(0)
                for reader_id in readers:
                    table.acknowledge(oldest, reader_id)
            table.acquire(chunk_id=chunk_id, version=VERSION, expected_readers=readers)
            in_flight.append(chunk_id)

        assert table.slot_count == 2
        assert table.reclaim_count == 18
