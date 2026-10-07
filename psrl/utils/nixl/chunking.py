"""Chunks and commit groups (implementation item B1).

Today one ``bucket`` carries six responsibilities at once: plan partitioning, submit and
wait granularity, version visibility, release granularity, holder-bitmap bit, and wire
buffer reuse unit. Convenient, but any change to one granularity drags the other five
along -- the reason "a larger W makes bucket turnover slower" was ever a sentence.

There are really two independent concepts (design 3.6.1):

=============  ====================  =========================  ==============
concept        serves                boundary set by            splits a key?
=============  ====================  =========================  ==============
``chunk``      pipelining, bandwidth a byte budget              **yes** -- an
                                                                11GB MoE key
                                                                must split
``CommitGroup`` correctness          the engine loader's        **no** -- that
                                     atomicity requirement      is its definition
=============  ====================  =========================  ==============

They are many-to-many: one chunk may hold several small groups, and one large group (a
fused MoE ``w13_weight``) spans several chunks. Separating them removes the awkward
rule the earlier design needed -- "let a fused tensor cross buckets but share a group
marker" -- because crossing is simply what the two concepts permit.

The atomicity constraint is not negotiable. vLLM's loader concatenates several HF
tensors into one engine parameter, so a half-delivered group is a silent no-op or an
assertion deep in the engine (miles' `atomic_groups.py` records the same lesson). This
module therefore refuses to mark a group visible until every one of its chunks has
landed, and `pending_groups` names what is missing rather than reporting a count.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

#: A shard of one tensor: the unit the plan assigns and the transport moves.
ShardRef = tuple[str, tuple[int, ...]]


@dataclass(frozen=True)
class ShardPiece:
    """One shard's bytes, possibly a slice of it when the shard spans chunks."""

    shard: ShardRef
    offset: int
    nbytes: int

    @property
    def is_whole_shard(self) -> bool:
        return self.offset == 0


@dataclass
class Chunk:
    """A transfer unit: a byte budget's worth of shard pieces."""

    index: int
    pieces: list[ShardPiece] = field(default_factory=list)

    @property
    def nbytes(self) -> int:
        return sum(piece.nbytes for piece in self.pieces)


@dataclass
class CommitGroup:
    """A visibility unit: shards the engine loader requires together."""

    name: str
    shards: tuple[ShardRef, ...]
    #: Chunks carrying any part of this group. Populated by `plan_chunks`.
    chunk_indices: tuple[int, ...] = ()


class GroupCompletionTracker:
    """Tracks which commit groups are fully delivered.

    A group becomes visible only when every chunk carrying any of its shards has
    arrived. Partial delivery is the failure this exists to prevent, so
    `mark_chunk_complete` never returns a partially satisfied group and
    `assert_all_complete` names the missing chunks.
    """

    def __init__(self, groups: list[CommitGroup]):
        self._groups = {group.name: group for group in groups}
        self._needed: dict[str, set[int]] = {
            group.name: set(group.chunk_indices) for group in groups
        }
        self._arrived: dict[str, set[int]] = {group.name: set() for group in groups}
        self._groups_by_chunk: dict[int, list[str]] = {}
        for group in groups:
            for chunk_index in group.chunk_indices:
                self._groups_by_chunk.setdefault(chunk_index, []).append(group.name)

    def mark_chunk_complete(self, chunk_index: int) -> list[str]:
        """Record a chunk's arrival; return the groups this completed, if any."""
        completed = []
        for name in self._groups_by_chunk.get(chunk_index, ()):
            self._arrived[name].add(chunk_index)
            if self._arrived[name] == self._needed[name]:
                completed.append(name)
        return completed

    def is_complete(self, group_name: str) -> bool:
        return self._arrived[group_name] == self._needed[group_name]

    def pending_groups(self) -> dict[str, set[int]]:
        """Groups still missing chunks, mapped to which chunks are missing."""
        return {
            name: self._needed[name] - self._arrived[name]
            for name in self._groups
            if self._arrived[name] != self._needed[name]
        }

    def assert_all_complete(self) -> None:
        """Raise unless every group is fully delivered.

        Called before a version is published. Half a group reaching the engine is worse
        than nothing arriving, because the engine will not complain.
        """
        pending = self.pending_groups()
        if pending:
            detail = ", ".join(
                f"{name} missing chunks {sorted(chunks)}" for name, chunks in sorted(pending.items())
            )
            raise AssertionError(
                f"{len(pending)} commit group(s) incomplete, so the version must not be "
                f"published: {detail}"
            )


def plan_chunks(
    groups: list[CommitGroup],
    shard_bytes: dict[ShardRef, int],
    chunk_bytes: int,
) -> tuple[list[Chunk], list[CommitGroup]]:
    """Partition a version's shards into chunks of at most ``chunk_bytes``.

    A shard larger than the budget is split across consecutive chunks: a Kimi-K2 fused
    ``w13_weight`` is ~11GB at fp8, far above any sensible chunk size, so the ability to
    split one shard is a requirement rather than an edge case.

    Group membership is recomputed rather than assumed, so the returned groups carry the
    chunk indices they actually depend on -- which is what the completion tracker needs
    to refuse a partial publish.
    """
    if chunk_bytes <= 0:
        raise ValueError(f"chunk_bytes must be positive, got {chunk_bytes}")

    chunks: list[Chunk] = []
    current = Chunk(index=0)
    chunks_by_shard: dict[ShardRef, set[int]] = {}

    def close_current() -> None:
        nonlocal current
        if current.pieces:
            chunks.append(current)
            current = Chunk(index=len(chunks))

    # Preserve the caller's group order: it reflects the plan, and a stable order keeps
    # chunk indices reproducible, which the design relies on for offset staggering.
    for group in groups:
        for shard in group.shards:
            if shard not in shard_bytes:
                raise KeyError(f"no byte size recorded for shard {shard}")
            remaining = shard_bytes[shard]
            if remaining == 0:
                continue
            offset = 0
            while remaining > 0:
                room = chunk_bytes - current.nbytes
                if room == 0:
                    close_current()
                    room = chunk_bytes
                take = min(room, remaining)
                current.pieces.append(ShardPiece(shard=shard, offset=offset, nbytes=take))
                chunks_by_shard.setdefault(shard, set()).add(current.index)
                offset += take
                remaining -= take
    close_current()

    resolved = [
        CommitGroup(
            name=group.name,
            shards=group.shards,
            chunk_indices=tuple(
                sorted(
                    {
                        index
                        for shard in group.shards
                        for index in chunks_by_shard.get(shard, ())
                    }
                )
            ),
        )
        for group in groups
    ]
    return chunks, resolved


def group_shards_by_key(shards: list[ShardRef]) -> "OrderedDict[str, list[ShardRef]]":
    """Group shard references by tensor key, preserving first-seen order."""
    grouped: OrderedDict[str, list[ShardRef]] = OrderedDict()
    for shard in shards:
        grouped.setdefault(shard[0], []).append(shard)
    return grouped
