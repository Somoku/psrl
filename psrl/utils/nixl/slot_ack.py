"""Wire-buffer slot reclamation by reader acknowledgement (implementation item B2).

The cast path writes each chunk into a wire buffer of only ``2 x chunk`` and hands
readers a descriptor into it. That buffer is cyclic, so the *next* cast overwrites a
slot some reader may still be reading -- a second writer that is not the optimizer, and
therefore not covered by the same-precision write barrier. Rollout then receives a
chunk that is half old and half new, silently, and only when readers drift apart.

Deepening the buffer does not fix it. Depth `D` masks a lag of at most `D` chunks, and W
independent instances have no bounded lag: one stalling on KV-cache pressure can fall
arbitrarily behind. Spending memory on an unbounded quantity is dressing a correctness
problem as a capacity problem.

What the three reference systems do instead is make reclamation explicit:

* verl's mooncake engine has the reader RDMA-write a 4-byte magic value into the
  sender's slot, and the sender polls it (`mooncake_checkpoint_engine.py`)
* verl's NCCL engine swaps buffers only after `wait_for_complete()`
* miles blocks on a free-queue, back-pressuring the gather itself (`delta.py:133`)

None of those transfer to a one-to-many pull directly -- verl's relay has exactly one
downstream per slot, miles knows its recipients because it pushes, and here W readers
each initiate one-sided reads with the trainer outside the data path. But the shared
shape does: an explicit return signal, and blocking when nothing is free.

The adaptation this module implements is to take the expected reader set from the plan
rather than discovering it at runtime. `expected` is known at handshake, so reclamation
asks "has everyone who was going to read this finished?" -- answerable with one bitmap
per slot, folded into the completion notification readers already send. A counter would
need two round trips and would leak permanently if a reader died; a bitmap converges
because `forget_reader` clears the bit.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class SlotBusyError(RuntimeError):
    """Raised when a slot is reused before its readers have acknowledged it."""


@dataclass
class _SlotState:
    """Who still owes an acknowledgement for the chunk currently in this slot."""

    chunk_id: int | None = None
    version: int | None = None
    expected: set[int] = field(default_factory=set)
    acked: set[int] = field(default_factory=set)

    @property
    def is_free(self) -> bool:
        return self.chunk_id is None

    @property
    def is_satisfied(self) -> bool:
        return not self.is_free and self.expected <= self.acked


class SlotAckTable:
    """Reader acknowledgements for a fixed set of wire-buffer slots.

    Usage mirrors the pipeline: `acquire` a slot for a chunk, transfer, collect
    `acknowledge` calls as readers finish, and `release` when satisfied. `acquire`
    raises rather than overwriting, because overwriting is the silent corruption this
    exists to prevent.
    """

    def __init__(self, slot_count: int):
        if slot_count < 1:
            raise ValueError(f"slot_count must be at least 1, got {slot_count}")
        self._slots = [_SlotState() for _ in range(slot_count)]
        #: chunk_id -> slot index, so acknowledgements need not know the slot.
        self._slot_of_chunk: dict[int, int] = {}
        self.reclaim_count = 0
        self.stale_acks = 0

    @property
    def slot_count(self) -> int:
        return len(self._slots)

    def free_slots(self) -> list[int]:
        return [index for index, slot in enumerate(self._slots) if slot.is_free]

    def acquire(self, chunk_id: int, version: int, expected_readers: set[int]) -> int:
        """Claim a free slot for ``chunk_id``; return its index.

        Raises `SlotBusyError` when nothing is free. The caller is expected to wait for
        an acknowledgement or time out into `CHUNK_STALE` (design 3.4.3) rather than
        proceeding -- which is why this does not silently pick the least-recently-used
        slot.
        """
        if chunk_id in self._slot_of_chunk:
            raise SlotBusyError(
                f"chunk {chunk_id} already occupies slot {self._slot_of_chunk[chunk_id]}"
            )
        for index, slot in enumerate(self._slots):
            if slot.is_free:
                slot.chunk_id = chunk_id
                slot.version = version
                slot.expected = set(expected_readers)
                slot.acked = set()
                self._slot_of_chunk[chunk_id] = index
                return index

        outstanding = {
            index: sorted(slot.expected - slot.acked) for index, slot in enumerate(self._slots)
        }
        raise SlotBusyError(
            f"no free slot for chunk {chunk_id}; every slot is awaiting readers "
            f"{outstanding}. Waiting is correct here -- reusing a slot would hand a "
            f"reader a chunk that is partly overwritten."
        )

    def acknowledge(self, chunk_id: int, reader_id: int) -> bool:
        """Record that ``reader_id`` finished ``chunk_id``. True if the slot is now free.

        Acknowledgements for a chunk that has already been released are counted and
        ignored: a retry or a late notification is normal, not an error.
        """
        index = self._slot_of_chunk.get(chunk_id)
        if index is None:
            self.stale_acks += 1
            return False

        slot = self._slots[index]
        slot.acked.add(reader_id)
        if slot.is_satisfied:
            self.release(chunk_id)
            return True
        return False

    def release(self, chunk_id: int) -> None:
        """Return a slot to the free list."""
        index = self._slot_of_chunk.pop(chunk_id, None)
        if index is None:
            return
        self._slots[index] = _SlotState()
        self.reclaim_count += 1

    def forget_reader(self, reader_id: int) -> list[int]:
        """Drop a reader from every expectation; return chunks that became satisfied.

        Called when a holder is removed (design 3.5.7's passive eviction). This is why
        the bitmap converges where a counter would not: a reader that dies mid-transfer
        would leave a counter permanently above zero and the slot permanently busy.
        """
        satisfied = []
        for slot in self._slots:
            if slot.is_free:
                continue
            slot.expected.discard(reader_id)
            slot.acked.discard(reader_id)
            if slot.is_satisfied:
                satisfied.append(slot.chunk_id)
        for chunk_id in satisfied:
            self.release(chunk_id)
        return satisfied

    def outstanding_readers(self, chunk_id: int) -> set[int]:
        """Readers still owing an acknowledgement for ``chunk_id``."""
        index = self._slot_of_chunk.get(chunk_id)
        if index is None:
            return set()
        slot = self._slots[index]
        return slot.expected - slot.acked

    def chunk_in_slot(self, slot_index: int) -> int | None:
        return self._slots[slot_index].chunk_id

    def version_in_slot(self, slot_index: int) -> int | None:
        return self._slots[slot_index].version
