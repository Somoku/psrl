"""Per-shard version stamps: the cheap half of what a PS used to tell you.

Direct supply removes an observation point. With a PS in the middle you could compare "the
version in the PS" against "the version in the rollout" and separate a transport fault
from a training fault; `_debug_log_ps_info` exists for that. Reading train ranks directly
deletes the middle term.

The design's own answer is a **version equality assertion**, not a content hash:
§3.4's invariant table lists "版本相等性 -- index 指派 peer 时，校验 `version(peer) == v`"
among the four that must be runtime assertions, and says the metadata-only ones are
constant-cost and **should be on by default** (「开销是常数级，应当常开」).

Content hashing cannot be that. Digesting a Qwen3-8B model costs ~61 s against a measured
1.4 s pull -- 40x the thing it checks -- so it could only ever run sampled and off by
default, which means the fault it is meant to catch happens on the unsampled keys. It also
scales with model size and dtype, so every new model re-opens the question.

A version stamp is O(1) per shard regardless of model, dtype or quantisation. What it
buys, precisely:

* **stale reads** -- the dominant real fault, and the one a missed publish produces. A
  reader that pulled before the supplier republished sees `version(shard) < v` and says so.
* **torn reads** -- shards of one key at different versions, which is what a half-finished
  publish looks like.
* **misdirected reads** -- a shard arriving from a supplier that never held it.

What it does *not* catch is bit-level corruption of correctly-versioned bytes. That is a
deliberate trade, and it is the right one here: RDMA is checksummed by the NIC, so silent
bit corruption is already improbable, while a stale or torn read is a *logic* error that
version stamps catch exactly and cheaply. The expensive check would have been off by
default anyway.
"""

from __future__ import annotations

from dataclasses import dataclass


#: A shard that was registered but never published at any version.
UNVERSIONED = -1


@dataclass(frozen=True)
class ShardStaleness:
    """One shard whose version did not match what the reader expected."""

    key: str
    shard: tuple[int, ...]
    source: str
    #: Version the reader expected -- the one it was told to pull.
    expected: int
    #: Version the supplier actually had for this shard, or None when it had no stamp.
    actual: int | None

    @property
    def is_stale(self) -> bool:
        """Older than expected: the supplier had not yet republished."""
        return self.actual is not None and self.actual < self.expected

    @property
    def is_ahead(self) -> bool:
        """Newer than expected, which means the reader raced a publish.

        Distinguished from stale because the remedies differ: behind means the pull was
        too early, ahead means the version this reader recorded is already wrong.
        """
        return self.actual is not None and self.actual > self.expected

    def __str__(self) -> str:
        if self.actual is None:
            state = "unpublished"
        elif self.is_stale:
            state = f"stale by {self.expected - self.actual}"
        else:
            state = f"ahead by {self.actual - self.expected}"
        return (
            f"{self.key} shard {self.shard} from {self.source}: "
            f"expected v{self.expected}, got v{self.actual} ({state})"
        )


def check_pull_versions(
    expected: int,
    published: dict[str, dict[str, dict[tuple[int, ...], int]]],
) -> list[ShardStaleness]:
    """Find every shard a pull read at a version other than `expected`.

    `published` is ``{source_client: {key: {shard: version}}}`` -- what each supplier
    reports it held for the shards this reader actually pulled from it.

    Reports all mismatches rather than the first, because the pattern is the diagnosis:
    one shard of one key is a transport fault, every shard of one source is that rank
    failing to publish, and the same shard index across all sources is a layout bug.
    Stopping early discards exactly the information that separates them.
    """
    stale: list[ShardStaleness] = []

    for source in sorted(published):
        for key in sorted(published[source]):
            for shard, version in sorted(published[source][key].items()):
                if version == expected:
                    continue
                stale.append(
                    ShardStaleness(
                        key=key,
                        shard=shard,
                        source=source,
                        expected=expected,
                        actual=None if version == UNVERSIONED else version,
                    )
                )
    return stale


def describe(stale: list[ShardStaleness], limit: int = 5) -> str:
    """A bounded, diagnosis-shaped summary for an assertion message.

    Bounded because a layout bug produces one entry per shard of the model, and an
    assertion that prints all of them buries the shape of the failure in its own output.
    The counts come first for that reason: "all shards of one source" and "one shard of
    one key" are different bugs and should not need counting by eye.
    """
    if not stale:
        return "no version mismatches"

    sources = {item.source for item in stale}
    keys = {item.key for item in stale}
    behind = sum(1 for item in stale if item.is_stale)
    ahead = sum(1 for item in stale if item.is_ahead)
    unpublished = sum(1 for item in stale if item.actual is None)

    head = (
        f"{len(stale)} shard(s) at the wrong version across {len(sources)} source(s) "
        f"and {len(keys)} key(s) -- {behind} behind, {ahead} ahead, "
        f"{unpublished} unpublished"
    )
    shown = "; ".join(str(item) for item in stale[:limit])
    suffix = f"; ... and {len(stale) - limit} more" if len(stale) > limit else ""
    return f"{head}: {shown}{suffix}"
