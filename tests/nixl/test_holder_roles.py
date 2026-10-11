"""Holder symmetrisation and bootstrap routing (implementation item D4).

Two failures shape these tests, and both are silent:

* **Reading the checkpoint on a steady-state switch-back** rolls training back to the
  last save and discards every step since. Nothing errors -- the weights are valid, just
  old -- so the only defence is that the first-vs-steady test cannot be derived from a
  flag that flips on every switch.
* **Letting a trainer resume from a rollout's quantised bytes** gives it weights that
  cannot be inverted losslessly and no fp32 master params. It trains, badly.

The third property, holder symmetry, is what makes a rollout-only start possible at all:
if the trainer is the only valid authority then `v0` has no source before the trainer
exists.
"""

from __future__ import annotations

import pytest

from psrl.workers.ps.holder_roles import (
    BootstrapPolicy,
    BootstrapSource,
    Holder,
    HolderKind,
    choose_checkpoint_readers,
)

pytestmark = pytest.mark.cpu_test


def _rollouts(nodes: int, per_node: int, version: int = 0) -> list[Holder]:
    return [
        Holder(
            name=f"gen{node * per_node + slot}",
            kind=HolderKind.ROLLOUT_GPU,
            node_id=f"node{node}",
            version=version,
        )
        for node in range(nodes)
        for slot in range(per_node)
    ]


class TestHoldersAreSymmetric:
    """A rollout can be an authority, which is what a trainer-less start requires."""

    def test_a_rollout_is_a_valid_holder_kind(self):
        holder = Holder("gen0", HolderKind.ROLLOUT_GPU, "node0", version=0)
        assert holder.kind is HolderKind.ROLLOUT_GPU

    def test_self_is_distinct_from_a_remote_rollout(self):
        """`SELF` means no transfer at all, not 'a rollout that happens to be local'."""
        assert HolderKind.SELF is not HolderKind.ROLLOUT_GPU
        assert len({k.value for k in HolderKind}) == len(list(HolderKind))


class TestCheckpointReaderSelection:
    def test_one_reader_per_node_by_default(self):
        readers = choose_checkpoint_readers(_rollouts(nodes=3, per_node=8))

        assert len(readers) == 3
        assert len({r for r in readers}) == 3

    def test_readers_are_spread_across_nodes_not_concentrated(self):
        """The point is per-node storage bandwidth, so one node taking all readers is the bug."""
        rollouts = _rollouts(nodes=3, per_node=8)
        readers = set(choose_checkpoint_readers(rollouts))
        nodes_covered = {h.node_id for h in rollouts if h.name in readers}

        assert nodes_covered == {"node0", "node1", "node2"}

    def test_reader_count_does_not_grow_with_rollouts_per_node(self):
        """Storage reads stay at one per node however many rollouts share it."""
        counts = {
            per_node: len(choose_checkpoint_readers(_rollouts(nodes=2, per_node=per_node)))
            for per_node in (2, 4, 16)
        }

        assert len(set(counts.values())) == 1, f"reader count moved with density: {counts}"

    def test_selection_is_deterministic(self):
        """Every process must compute the same answer without communicating."""
        rollouts = _rollouts(nodes=4, per_node=4)
        shuffled = list(reversed(rollouts))

        assert choose_checkpoint_readers(rollouts) == choose_checkpoint_readers(shuffled)

    def test_per_node_above_one_is_honoured(self):
        readers = choose_checkpoint_readers(_rollouts(nodes=2, per_node=8), per_node=3)
        assert len(readers) == 6

    def test_zero_readers_is_rejected(self):
        with pytest.raises(ValueError, match="at least 1"):
            choose_checkpoint_readers(_rollouts(nodes=1, per_node=2), per_node=0)


class TestBootstrapRouting:
    def test_cold_rollout_with_no_holders_reads_the_checkpoint(self):
        policy = BootstrapPolicy()

        source = policy.source_for(
            "gen0", HolderKind.ROLLOUT_GPU, peers_hold_version=False, wire_is_quantised=True
        )

        assert source is BootstrapSource.CHECKPOINT

    def test_rollout_prefers_a_peer_once_one_holds_the_version(self):
        policy = BootstrapPolicy()

        source = policy.source_for(
            "gen1", HolderKind.ROLLOUT_GPU, peers_hold_version=True, wire_is_quantised=True
        )

        assert source is BootstrapSource.PEER

    def test_trainer_reads_the_checkpoint_when_the_wire_is_quantised(self):
        """A rollout's fp8 bytes are not a resumable training state.

        Inverting the quantisation is lossy and the optimizer's fp32 master params were
        never on the wire, so "a peer has the weights" does not mean the trainer can use
        them.
        """
        policy = BootstrapPolicy()

        source = policy.source_for(
            "train0", HolderKind.TRAIN_GPU, peers_hold_version=True, wire_is_quantised=True
        )

        assert source is BootstrapSource.CHECKPOINT

    def test_trainer_may_pull_from_a_peer_on_a_same_precision_wire(self):
        """Without a cast the bytes are the trainer's own format, so the rule relaxes."""
        policy = BootstrapPolicy()

        source = policy.source_for(
            "train0", HolderKind.TRAIN_GPU, peers_hold_version=True, wire_is_quantised=False
        )

        assert source is BootstrapSource.PEER


class TestFirstVersusSteadyIsNotDerivedFromColocateMode:
    """The marker must be one-way, or a switch-back silently rolls training back."""

    def test_a_parked_process_restores_from_host_not_storage(self):
        policy = BootstrapPolicy()
        policy.note_holds_real_weights("train0")
        policy.note_parked_to_host("train0")

        source = policy.source_for(
            "train0", HolderKind.TRAIN_GPU, peers_hold_version=True, wire_is_quantised=True
        )

        assert source is BootstrapSource.HOST_RELOAD, (
            "a steady-state switch-back read storage; training would silently resume from "
            "the last save and lose every step since"
        )

    def test_the_marker_survives_repeated_switches(self):
        """`is_rollout_mode_in_actor` flips every switch; this must not."""
        policy = BootstrapPolicy()
        policy.note_holds_real_weights("train0")

        for _ in range(3):
            policy.note_parked_to_host("train0")
            policy.note_restored_from_host("train0")

        assert policy.has_ever_held_real_weights("train0")

    def test_after_restoring_the_process_is_no_longer_parked(self):
        policy = BootstrapPolicy()
        policy.note_holds_real_weights("train0")
        policy.note_parked_to_host("train0")
        policy.note_restored_from_host("train0")

        source = policy.source_for(
            "train0", HolderKind.TRAIN_GPU, peers_hold_version=True, wire_is_quantised=True
        )

        assert source is BootstrapSource.CHECKPOINT

    def test_parking_weights_never_held_is_rejected(self):
        """Restoring them would hand back uninitialised memory, and that read succeeds."""
        policy = BootstrapPolicy()

        with pytest.raises(ValueError, match="never held"):
            policy.note_parked_to_host("train0")

    def test_a_fresh_process_has_not_ever_held_weights(self):
        assert not BootstrapPolicy().has_ever_held_real_weights("train0")
