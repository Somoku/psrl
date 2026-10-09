"""Degrading when trainer ranks fail (implementation items D7 and D8).

Direct supply removes the CPU replica, so at first glance there is nothing to fall back
to. There is, and it is free: the train side's sharding declaration carries no DP
dimension, so the same ``(key, shard)`` exists identically on every rank of a DP group.
What normally spreads egress across those ranks is, on failure, the redundancy.

**But that is backend-dependent, which is why D7 and D8 are one change.** Pure FSDP2
with ``fsdp_size >= world_size`` has a one-dimensional mesh and exactly one replica, so
the single-rank path does not exist there. Writing "a DP replica is always available"
and fixing it later would mean reworking the response ladder, so the replica count is
derived from `NIXLSharding` at runtime and ``replica_count == 1`` simply skips a rung.

The ladder (design 3.5.7):

* one rank fails -> drop it, another rank of its DP group serves the same bytes, the
  affected pull re-resolves. The rollout never notices.
* a whole DP group fails -> no holder for that chunk -> ``CHUNK_UNAVAILABLE``, an
  explicit value rather than a timeout. The rollout keeps its current version and keeps
  generating; staleness it already has is visible to the algorithm, whereas a hang or a
  half-updated model is not.
* the trainer restarts -> holders cleared, version retained, every pull degrades at once.

Peers are not the fallback. They only ever cover the version already read, and the
holder bitmap is cleared wholesale when a version is published -- so peers help not at
all with a new version's first wave, which is exactly when a failure may strike. DP
replicas are structural redundancy; peers are a bandwidth optimisation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum


class ChunkStatus(Enum):
    """Outcome of asking the index where to read a chunk."""

    AVAILABLE = "available"
    #: No holder survives for this chunk. An explicit value, never a timeout: the
    #: rollout must be able to tell "give up this round" from "wait longer".
    CHUNK_UNAVAILABLE = "chunk_unavailable"


@dataclass(frozen=True)
class ChunkResolution:
    status: ChunkStatus
    source: str | None = None
    #: Ranks that could also have served it, for reporting how much slack remains.
    alternatives: tuple[str, ...] = ()

    @property
    def is_available(self) -> bool:
        return self.status is ChunkStatus.AVAILABLE


def derive_replica_count(shard_mesh: dict[int, int], world_size: int) -> int:
    """Replicas of each shard, derived rather than assumed (item D8).

    The mesh describes how a tensor is cut; whatever ranks remain after accounting for
    every cut dimension hold identical copies. Megatron declares only TP/ETP, so its DP
    degree falls out here as replication. Pure FSDP2 cuts across the whole world and
    leaves one.

    A mesh claiming more shards than there are ranks is a configuration error rather
    than a zero-replica cluster, so it raises instead of returning 0 and letting the
    caller treat every chunk as unavailable.
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")

    sharded = 1
    for count in shard_mesh.values():
        sharded *= count
    if sharded <= 0:
        raise ValueError(f"shard mesh has a non-positive dimension: {shard_mesh}")
    if sharded > world_size:
        raise ValueError(
            f"shard mesh {shard_mesh} implies {sharded} shards but the world has only "
            f"{world_size} ranks; the mesh and the world size disagree"
        )
    return world_size // sharded


@dataclass
class ChunkAvailabilityIndex:
    """Which ranks can serve each chunk, and what to do when they cannot.

    Failure detection rides on the existing error path: a rollout whose `client_read`
    fails reports the holder it failed against, and that rank is dropped. A failed rank
    does not announce itself, and a heartbeat would be a second mechanism for something
    the error path already observes.
    """

    #: chunk -> ranks holding it
    _holders: dict[int, set[str]] = field(default_factory=dict)
    _failed: set[str] = field(default_factory=set)

    def register(self, chunk: int, ranks: Iterable[str]) -> None:
        self._holders.setdefault(chunk, set()).update(ranks)

    def report_failure(self, rank: str) -> None:
        """Drop a rank, on the evidence of a transfer error against it."""
        self._failed.add(rank)

    def restore(self, rank: str) -> None:
        """Re-admit a rank after it re-registers."""
        self._failed.discard(rank)

    def clear_all_holders(self) -> None:
        """Trainer-wide restart: every holder is gone, the version is not.

        Version continuity matters -- `maybe_delete_buffer` and the staleness buffers
        index by ``v-1``, so skipping a number leaks them (design 3.2).
        """
        self._holders.clear()

    def live_holders(self, chunk: int) -> set[str]:
        return self._holders.get(chunk, set()) - self._failed

    def replica_slack(self, chunk: int) -> int:
        """Surviving holders beyond the one in use -- how many more failures it takes."""
        return max(0, len(self.live_holders(chunk)) - 1)

    def resolve(self, chunk: int, prefer: str | None = None) -> ChunkResolution:
        """Pick a surviving holder, or report the chunk unavailable.

        ``prefer`` is honoured only if it is still alive, so a caller retrying after a
        failure does not have to know which rank it just lost.
        """
        live = self.live_holders(chunk)
        if not live:
            return ChunkResolution(ChunkStatus.CHUNK_UNAVAILABLE)

        chosen = prefer if prefer in live else sorted(live)[0]
        return ChunkResolution(
            status=ChunkStatus.AVAILABLE,
            source=chosen,
            alternatives=tuple(sorted(live - {chosen})),
        )

    def resolve_all(self, chunks: Iterable[int]) -> dict[int, ChunkResolution]:
        return {chunk: self.resolve(chunk) for chunk in chunks}

    def version_is_deliverable(self, chunks: Iterable[int]) -> bool:
        """Whether every chunk of a version can still be served.

        A version missing any chunk must not be marked visible: a partially updated
        model is worse than the old one, and the engine will not complain about it
        (design 3.4.6).
        """
        return all(self.resolve(chunk).is_available for chunk in chunks)
