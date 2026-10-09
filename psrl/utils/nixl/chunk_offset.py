"""Staggering where each instance starts reading (implementation item D5).

``W`` lets only ``W`` instances read the trainer in the first wave while the rest queue.
If those ``W`` all start at chunk 0 and advance in the same order, they hold chunk 0 at
the same moment, then chunk 1 at the same moment, and a queued instance has to wait for
one of them to finish an entire round before it can find a peer for anything.

Starting instance ``i`` at ``i x num_chunks / R`` and wrapping fixes it for free: the
first wave holds disjoint chunk sets at every instant, so a released queuer finds a
same-version peer for most of what it needs immediately, and diffusion begins *within*
the first wave rather than after it.

Three constraints from design 3.5.2.3, all of which this module enforces rather than
documents:

* **Offsets align to commit-group boundaries.** Starting mid-group would make that
  group's visibility span the ring wrap, so the completion check stops being "have all
  my chunks arrived" and becomes a wrap-aware special case.
* **Offsets are deterministic**, derived from the instance ordinal. The index predicts
  who holds what without communicating, which is what keeps source selection off the
  control plane.
* **Only meaningful when ``R > W``.** With ``R <= W`` everyone reads the trainer directly
  and there is nothing to diffuse, so staggering would add scatter for no benefit.
"""

from __future__ import annotations

from collections.abc import Sequence


def group_start_chunks(group_chunk_indices: Sequence[Sequence[int]]) -> list[int]:
    """The chunk each commit group starts at, sorted and de-duplicated.

    These are the only legal ring starting points: beginning anywhere else splits a
    group across the wrap.
    """
    starts = {min(indices) for indices in group_chunk_indices if len(indices) > 0}
    return sorted(starts)


def start_chunk_for_instance(
    instance_index: int,
    num_chunks: int,
    num_instances: int,
    group_starts: Sequence[int] | None = None,
) -> int:
    """Where instance ``instance_index`` begins its ring.

    With ``group_starts`` the raw offset is snapped down to the nearest group boundary,
    which keeps every group contiguous in read order. Snapping *down* rather than to the
    nearest boundary keeps the mapping monotonic in the instance index, so two adjacent
    instances never swap places -- the index relies on that to predict holders.
    """
    if num_chunks <= 0:
        raise ValueError(f"num_chunks must be positive, got {num_chunks}")
    if num_instances <= 0:
        raise ValueError(f"num_instances must be positive, got {num_instances}")

    raw = (instance_index * num_chunks // num_instances) % num_chunks
    if not group_starts:
        return raw

    candidates = [start for start in group_starts if start <= raw]
    return candidates[-1] if candidates else group_starts[0]


def read_order(start_chunk: int, num_chunks: int) -> list[int]:
    """The ring order an instance reads in, beginning at ``start_chunk``."""
    if num_chunks <= 0:
        raise ValueError(f"num_chunks must be positive, got {num_chunks}")
    start = start_chunk % num_chunks
    return [(start + offset) % num_chunks for offset in range(num_chunks)]


def should_stagger(num_instances: int, trainer_fanout: int) -> bool:
    """Whether staggering helps: only when some instance has to queue."""
    return num_instances > trainer_fanout


def plan_read_orders(
    num_chunks: int,
    num_instances: int,
    trainer_fanout: int,
    group_chunk_indices: Sequence[Sequence[int]] | None = None,
) -> list[list[int]]:
    """Read order for every instance, staggered when that helps and plain when not.

    Returning plain order below the threshold is deliberate: with ``R <= W`` everyone
    reads the trainer anyway, and scattering the access pattern would cost locality for
    a diffusion that never happens.
    """
    if not should_stagger(num_instances, trainer_fanout):
        return [list(range(num_chunks)) for _ in range(num_instances)]

    group_starts = group_start_chunks(group_chunk_indices) if group_chunk_indices else None
    return [
        read_order(
            start_chunk_for_instance(index, num_chunks, num_instances, group_starts),
            num_chunks,
        )
        for index in range(num_instances)
    ]


def first_wave_overlap(read_orders: Sequence[Sequence[int]], wave: int) -> int:
    """How many instances share their most-held chunk after ``wave`` reads.

    The quantity staggering exists to reduce: with every instance starting at chunk 0
    this equals the instance count, and a queued reader has no peer to turn to.
    """
    if wave <= 0:
        return 0
    counts: dict[int, int] = {}
    for order in read_orders:
        for chunk in order[:wave]:
            counts[chunk] = counts.get(chunk, 0) + 1
    return max(counts.values()) if counts else 0


def distinct_chunks_held(read_orders: Sequence[Sequence[int]], wave: int) -> int:
    """Distinct chunks the group holds after ``wave`` reads each.

    The other side of the same coin: a released queuer can be served for exactly these,
    so more is better and the unstaggered case gives the minimum.
    """
    if wave <= 0:
        return 0
    held: set[int] = set()
    for order in read_orders:
        held.update(order[:wave])
    return len(held)
