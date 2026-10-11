"""Deciding whether shipping only the changed bytes is worth it (item F1).

Delta transfer sends `(position, value)` pairs instead of whole tensors. It pays off only
when two things hold at once, and both are properties of the deployment rather than of
this code:

* **The link is slow enough that bytes dominate.** On the measured cluster a pull is
  0.619 s for ~16 GB at roughly 103 GB/s aggregate per node, so transfer is already cheap
  and the diff would cost more than it saves. Across a datacentre boundary the ratio
  inverts.
* **The weights actually change little between versions.** Early in training nearly every
  element moves; late in a run, or with a small learning rate, the changed fraction is
  small. A delta of 90% changed elements is worse than sending the tensor, because each
  changed element also carries its index.

**The constraint-(b) note in the design needs correcting, and the correction is what makes
this item viable.** It says a diff base is "an extra CPU copy of the weights", so delta can
only ever be an opt-in exception. Reading verl's implementation shows that is not how the
snapshot is held: each rank pins a snapshot of **only its own shard** and diffs locally
(`delta_checkpoint_engine.py`: "no rank ever holds a full-model snapshot"). The cost is
therefore `M / ranks` per rank and scales down as the cluster grows -- the *distributed*
CPU residency the requirements explicitly permit, not the centralised copy they forbid.

That does not make it free, so this module computes the trade rather than assuming it. The
decision is a planning-time one: both inputs are known before any bytes move.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeltaEstimate:
    """What a delta would cost against sending the shard whole."""

    changed_fraction: float
    delta_bytes: int
    full_bytes: int
    snapshot_bytes_per_rank: int

    @property
    def is_worthwhile(self) -> bool:
        return self.delta_bytes < self.full_bytes

    @property
    def byte_ratio(self) -> float:
        return self.delta_bytes / self.full_bytes if self.full_bytes else 1.0


def estimate_delta(
    *,
    shard_bytes: int,
    element_bytes: int,
    changed_fraction: float,
    index_bytes: int = 4,
) -> DeltaEstimate:
    """Bytes a delta would move for one shard.

    Each changed element costs its value *plus its index*, which is the term that decides
    the break-even: with 4-byte indices and bf16 values a changed element costs 3x a
    resent one, so the delta stops paying once about a third of the elements move. Omitting
    the index term would make delta look worthwhile at any change rate below 100%.
    """
    if not 0.0 <= changed_fraction <= 1.0:
        raise ValueError(f"changed_fraction must be in [0, 1], got {changed_fraction}")
    if element_bytes <= 0:
        raise ValueError(f"element_bytes must be positive, got {element_bytes}")

    elements = shard_bytes // element_bytes
    changed = int(elements * changed_fraction)
    delta_bytes = changed * (element_bytes + index_bytes)

    return DeltaEstimate(
        changed_fraction=changed_fraction,
        delta_bytes=delta_bytes,
        full_bytes=shard_bytes,
        # The diff base is this rank's shard only, following verl: no rank holds the
        # whole model, so this scales down as the cluster grows.
        snapshot_bytes_per_rank=shard_bytes,
    )


def break_even_fraction(element_bytes: int, index_bytes: int = 4) -> float:
    """Changed fraction above which a delta costs more than resending.

    Worth having as a function rather than a constant because it moves with dtype: fp8
    halves the value term while the index stays 4 bytes, so delta breaks even *earlier*
    for quantised wires -- the opposite of the intuition that smaller elements favour
    deltas.
    """
    return element_bytes / (element_bytes + index_bytes)


def should_use_delta(
    *,
    shard_bytes: int,
    element_bytes: int,
    changed_fraction: float,
    link_gbps: float,
    diff_gbps: float = 400.0,
    index_bytes: int = 4,
) -> tuple[bool, str]:
    """Whether to send a delta, with the reason.

    ``diff_gbps`` is the local bandwidth for computing the byte diff -- a full pass over
    the shard, so on a fast link the diff alone can exceed the transfer it replaces. That
    is the measured regime on this cluster, which is why the answer here is normally no
    and the item stays opt-in.

    Returns the reason alongside the verdict because "delta was not used" is otherwise
    indistinguishable from "delta was not configured".
    """
    estimate = estimate_delta(
        shard_bytes=shard_bytes,
        element_bytes=element_bytes,
        changed_fraction=changed_fraction,
        index_bytes=index_bytes,
    )
    if not estimate.is_worthwhile:
        threshold = break_even_fraction(element_bytes, index_bytes)
        return False, (
            f"{changed_fraction:.0%} of elements changed, above the {threshold:.0%} "
            f"break-even for {element_bytes}-byte values with {index_bytes}-byte indices"
        )

    diff_ms = shard_bytes / (diff_gbps * 1e9) * 1e3
    saved_ms = (estimate.full_bytes - estimate.delta_bytes) / (link_gbps * 1e9) * 1e3
    if diff_ms >= saved_ms:
        return False, (
            f"diffing costs {diff_ms:.2f}ms against {saved_ms:.2f}ms saved at "
            f"{link_gbps:.0f} GB/s; the link is too fast for a delta to pay"
        )

    return True, (
        f"{estimate.byte_ratio:.1%} of the bytes, saving {saved_ms:.1f}ms against "
        f"{diff_ms:.2f}ms of diffing"
    )
