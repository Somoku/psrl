"""Who may serve a version, and where a cold start gets its bytes (item D4).

The rest of the transport assumes the trainer is the one authority: `SourceKind` has a
`TRAINER` member, and a rollout is only ever a *peer* that passes along what the trainer
emitted. That assumption breaks in the configuration the design calls for, where a run
starts with every GPU doing rollout and no trainer exists yet (design 3.3.1a). Something
has to hold ``v0``, and it cannot be a process that is not running.

So holders are symmetric: a holder is identified by *what it holds and where*, not by the
role of the process holding it. A rollout representative that read ``v0`` from the
checkpoint is as valid a source as a trainer rank, and the planner's preference order
(same node, then remote) applies to both without a special case.

**The cold-start asymmetry that remains, and why it is not an inconsistency.** When the
trainer first wakes it loads from the checkpoint rather than pulling from a rollout, even
though a rollout holds the current version. Under the cast path the rollout's bytes are
fp8/fp4: dequantising them is lossy, and the optimizer needs fp32 master params that were
never transmitted. The rollout holds *a* copy of the weights, not a copy a trainer can
resume from. `BootstrapPolicy` encodes that as a rule rather than leaving it to a comment.

**Why first-vs-steady needs its own flag.** The obvious test -- "are we in rollout mode?"
-- is wrong, because `is_rollout_mode_in_actor` flips back and forth on every colocate
switch, so it cannot distinguish "first time here" from "here again". Reading the
checkpoint on a steady-state switch-back is silent data loss: training continues from the
last *saved* version, discarding every step since, and nothing errors. The marker is
therefore one-way and explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class HolderKind(Enum):
    """Where a version's bytes live.

    Ordered by how little scarce bandwidth a read costs, which is the same order the
    planner uses. The names say *where*, not *which role* -- `ROLLOUT_GPU` is a valid
    source precisely because the design stops treating the trainer as the only authority.
    """

    #: This process already holds it; no transfer at all.
    SELF = "self"
    #: A rollout GPU, which may be the authority during a rollout-only phase.
    ROLLOUT_GPU = "rollout_gpu"
    #: A trainer GPU holding its own parameters.
    TRAIN_GPU = "train_gpu"


@dataclass(frozen=True)
class Holder:
    """One process holding one version, with the node it sits on."""

    name: str
    kind: HolderKind
    node_id: str
    version: int


class BootstrapSource(Enum):
    """Where a process gets weights when it has none."""

    #: Read the checkpoint from storage.
    CHECKPOINT = "checkpoint"
    #: Pull from a peer that already holds the version.
    PEER = "peer"
    #: Restore the copy this process parked in host memory before yielding its GPU.
    HOST_RELOAD = "host_reload"


@dataclass
class BootstrapPolicy:
    """Decides where a process's weights come from, given what it is and what it has.

    State is a single one-way flag per process, `has_ever_held_real_weights`. It is set
    once real weights land and never cleared, which is what separates "first arrival here"
    from "returning here". Deriving it from the colocate mode instead would make every
    switch-back look like a first arrival.
    """

    #: Processes that have held real weights at some point in this run.
    _ever_held: set[str] = field(default_factory=set)
    #: Processes whose weights are parked in host memory right now.
    _parked: set[str] = field(default_factory=set)

    def note_holds_real_weights(self, process: str) -> None:
        """Record that ``process`` now holds real weights. Idempotent and one-way."""
        self._ever_held.add(process)

    def note_parked_to_host(self, process: str) -> None:
        """Record that ``process`` copied its weights to host memory and freed the GPU."""
        if process not in self._ever_held:
            raise ValueError(
                f"{process} parked weights it never held; the park/restore pair is out of "
                f"order, and restoring would hand back uninitialised memory"
            )
        self._parked.add(process)

    def note_restored_from_host(self, process: str) -> None:
        self._parked.discard(process)

    def has_ever_held_real_weights(self, process: str) -> bool:
        return process in self._ever_held

    def source_for(
        self,
        process: str,
        kind: HolderKind,
        *,
        peers_hold_version: bool,
        wire_is_quantised: bool,
    ) -> BootstrapSource:
        """Where ``process`` should get weights from right now.

        ``wire_is_quantised`` is what makes the trainer's answer differ from a rollout's:
        under a cast path the bytes on the wire are fp8/fp4, which a trainer cannot resume
        from (lossy to invert, and no fp32 master params). A rollout consumes exactly those
        bytes, so for it a peer is always fine.
        """
        if process in self._parked:
            # Steady-state switch-back. Reading the checkpoint here would silently roll
            # training back to the last save.
            return BootstrapSource.HOST_RELOAD

        if kind is HolderKind.TRAIN_GPU:
            if wire_is_quantised or not peers_hold_version:
                return BootstrapSource.CHECKPOINT
            return BootstrapSource.PEER

        # Rollout: any holder of the version will do, and only falls back to storage when
        # nobody has it yet -- the cold start where one representative per node reads.
        return BootstrapSource.PEER if peers_hold_version else BootstrapSource.CHECKPOINT


def choose_checkpoint_readers(
    rollouts: list[Holder],
    *,
    per_node: int = 1,
) -> list[str]:
    """Pick the rollouts that read ``v0`` from storage; the rest take it from them.

    One reader per node by default, which is the same de-duplication the transfer path
    uses: storage bandwidth is shared per node, so N readers on a node take N times as
    long to deliver the same bytes. The others get it over NVLink.

    Deterministic by name so every process computes the same answer without agreeing on
    one -- the planner relies on this to predict holders without a round trip.
    """
    if per_node < 1:
        raise ValueError(f"per_node must be at least 1, got {per_node}")

    by_node: dict[str, list[str]] = {}
    for holder in rollouts:
        by_node.setdefault(holder.node_id, []).append(holder.name)

    readers: list[str] = []
    for node in sorted(by_node):
        readers.extend(sorted(by_node[node])[:per_node])
    return readers
