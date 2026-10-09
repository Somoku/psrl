"""Choosing where each chunk is read from (implementation item D1).

Under trainer-direct supply there is no PS data plane, so every rollout reads either
from the trainer or from a peer that already holds the same version. Picking which is
the whole of D1's traffic claim: the trainer must emit ``M`` bytes rather than ``N x M``,
and each rollout node must take in ``M / num_nodes``.

Three rules, in descending order of how badly violating them hurts:

**Only a same-version peer may be a source.** Reading from a peer that holds ``v-1``
silently adds one unit of staleness that the algorithm never agreed to, and a mix of
``v`` and ``v-1`` chunks produces a policy that never existed -- worse than staleness,
because no correction term can compensate for it (design 3.5.2). The reward is that
holder state collapses to one bitmap for the current version, cleared wholesale when the
version advances; per-chunk version tracking is unnecessary.

**No chains.** A peer may serve only what it read from the trainer, never what it read
from another peer. Chains make completion time depend on an unbounded path length and
turn one slow reader into a stall for everyone behind it (3.5.4).

**Same-node peers first, trainer second, cross-node peers last.** A same-node peer costs
NVLink/PCIe rather than NIC bandwidth, so it is free in the currency that is actually
scarce. The trainer comes next, bounded by ``W`` concurrent readers. Cross-node peers are
phase two, behind a flag: the candidate filter and the ordering weight are the only
difference, so enabling it is not a second mechanism.

``W`` is not a rate limiter. It is the number of readers that share the trainer's single
emission in the first wave -- the de-duplication group size (3.5.2.1). The default,
one representative per rollout node, is what makes cross-node traffic exactly one copy
per node.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum


class SourceKind(Enum):
    """Where a chunk is read from, ordered by how little NIC it costs."""

    SAME_NODE_PEER = "same_node_peer"
    TRAINER = "trainer"
    CROSS_NODE_PEER = "cross_node_peer"


@dataclass(frozen=True)
class Reader:
    """A rollout instance that needs a chunk."""

    name: str
    node_id: str


@dataclass(frozen=True)
class SourceDecision:
    """The resolved source for one reader, or a queued verdict."""

    reader: str
    kind: SourceKind | None
    source: str | None

    @property
    def is_queued(self) -> bool:
        """No source was available; the reader retries once one frees up."""
        return self.kind is None


@dataclass
class HolderSet:
    """Who holds the current version, and for which chunks.

    Scoped to a single version on purpose. `advance_to` clears everything rather than
    merging, because a holder of the previous version is not a legal source and keeping
    it would require per-chunk version tracking to avoid reading from it.
    """

    version: int
    #: chunk index -> readers that have finished it for `version`
    _holders: dict[int, set[str]] = field(default_factory=dict)
    #: readers that read a chunk from a peer rather than the trainer, per chunk. They
    #: must not serve it on: that would be a chain.
    _second_hand: dict[int, set[str]] = field(default_factory=dict)

    def advance_to(self, version: int) -> None:
        """Move to a new version, dropping every holder of the old one."""
        if version < self.version:
            raise ValueError(
                f"version must not go backwards: holder set is at {self.version}, got {version}"
            )
        if version == self.version:
            return
        self.version = version
        self._holders.clear()
        self._second_hand.clear()

    def mark_held(self, chunk: int, reader: str, *, from_peer: bool = False) -> None:
        """Record that ``reader`` finished ``chunk`` for the current version."""
        self._holders.setdefault(chunk, set()).add(reader)
        if from_peer:
            self._second_hand.setdefault(chunk, set()).add(reader)

    def forget(self, reader: str) -> None:
        """Drop a reader entirely -- it crashed, or left the cluster."""
        for holders in self._holders.values():
            holders.discard(reader)
        for second_hand in self._second_hand.values():
            second_hand.discard(reader)

    def holders_of(self, chunk: int) -> set[str]:
        return set(self._holders.get(chunk, ()))

    def servable_holders_of(self, chunk: int) -> set[str]:
        """Holders that may serve ``chunk`` on: those that got it from the trainer.

        Excluding second-hand holders is what makes the no-chain rule structural rather
        than a convention the planner has to remember.
        """
        return self.holders_of(chunk) - self._second_hand.get(chunk, set())

    def holds(self, chunk: int, reader: str) -> bool:
        return reader in self._holders.get(chunk, ())


@dataclass
class SourceResolver:
    """Resolves, per (reader, chunk), where the bytes should come from.

    The resolver is pure: it is told who holds what and how many reads the trainer has
    in flight, and returns decisions. Keeping the transport out of it is what lets the
    traffic claims be tested without a cluster.
    """

    #: Concurrent readers the trainer serves in the first wave. Not a rate limiter --
    #: the de-duplication group size (design 3.5.2.1).
    max_trainer_readers: int
    trainer_name: str = "trainer"
    #: Phase two. Off by default, which leaves same-node peers and the trainer, exactly
    #: the phase-one design.
    allow_cross_node_peers: bool = False

    def resolve(
        self,
        chunk: int,
        readers: Iterable[Reader],
        holders: HolderSet,
        node_of: dict[str, str],
        trainer_in_flight: int = 0,
    ) -> list[SourceDecision]:
        """Resolve a source for each reader that still needs ``chunk``.

        ``trainer_in_flight`` is the number of reads already issued against the trainer
        for this chunk, so a caller resolving in several passes does not oversubscribe
        it. Readers that already hold the chunk are not returned at all -- they have
        nothing to do, and emitting a decision for them would double-count traffic.
        """
        decisions: list[SourceDecision] = []
        in_flight = trainer_in_flight

        pending: list[Reader] = []
        for reader in readers:
            if holders.holds(chunk, reader.name):
                continue

            same_node = self._same_node_source(chunk, reader, holders, node_of)
            if same_node is not None:
                decisions.append(
                    SourceDecision(reader.name, SourceKind.SAME_NODE_PEER, same_node)
                )
            else:
                pending.append(reader)

        # Trainer slots go to one reader per *node*, not to the first W readers in
        # whatever order they arrive. That ordering is the entire de-duplication
        # property: a node that gets one copy can diffuse it over NVLink, whereas
        # spending several slots on one node leaves other nodes with nothing to diffuse
        # from, and they come back to the trainer next round. Measured on a 4-node x
        # 8-reader cluster, list order made the trainer serve 16 reads instead of 4.
        served_nodes = {
            node_of[name]
            for name in holders.holders_of(chunk)
            if name in node_of
        }
        first_per_node: list[Reader] = []
        rest: list[Reader] = []
        claimed: set[str] = set(served_nodes)
        for reader in pending:
            if reader.node_id not in claimed:
                claimed.add(reader.node_id)
                first_per_node.append(reader)
            else:
                rest.append(reader)

        for reader in first_per_node + rest:
            if in_flight < self.max_trainer_readers:
                in_flight += 1
                decisions.append(
                    SourceDecision(reader.name, SourceKind.TRAINER, self.trainer_name)
                )
                continue

            cross_node = (
                self._cross_node_source(chunk, reader, holders, node_of)
                if self.allow_cross_node_peers
                else None
            )
            if cross_node is not None:
                decisions.append(
                    SourceDecision(reader.name, SourceKind.CROSS_NODE_PEER, cross_node)
                )
                continue

            # Queue rather than fall back to the trainer: exceeding W is what the
            # bound exists to prevent, and a peer will exist once the wave completes.
            decisions.append(SourceDecision(reader.name, None, None))

        return decisions

    def _same_node_source(
        self,
        chunk: int,
        reader: Reader,
        holders: HolderSet,
        node_of: dict[str, str],
    ) -> str | None:
        candidates = sorted(
            name
            for name in holders.servable_holders_of(chunk)
            if name != reader.name and node_of.get(name) == reader.node_id
        )
        return candidates[0] if candidates else None

    def _cross_node_source(
        self,
        chunk: int,
        reader: Reader,
        holders: HolderSet,
        node_of: dict[str, str],
    ) -> str | None:
        candidates = sorted(
            name
            for name in holders.servable_holders_of(chunk)
            if name != reader.name and node_of.get(name) != reader.node_id
        )
        return candidates[0] if candidates else None


def default_trainer_fanout(nodes_per_instance: int, nic_count: int) -> int:
    """``W``'s initial value: one reader per rollout node, capped by the NIC count.

    Design 3.5.2.1 derives this as ``min(NIC_count, nodes_per_instance)``. The physical
    reading is what makes it the right default rather than a tuned constant: one
    representative per node means cross-node traffic is exactly one copy per node, which
    is the de-duplication property the whole phase exists for.
    """
    return max(1, min(nic_count, nodes_per_instance))
