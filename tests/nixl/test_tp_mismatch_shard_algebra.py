"""Shard algebra under train/rollout parallelism mismatch (verification item 19).

The design claims layout alignment needs no per-model code: `find_finest_shard_mesh`
takes the per-dimension LCM of both sides' shard counts, each side refactors onto that
mesh, and the bytes line up. `test_shard_algebra_properties.py` proves the algebra's
abstract invariants (coverage, disjointness, LCM identity). This file pins the concrete
claim item 19 actually asks for: run **train TP=8 <-> rollout TP=4 and train TP=4 <->
rollout TP=8**, and compare tensor contents element by element.

Why a separate file rather than more parameters on the property sweep: the property
tests assert structural facts about one side's shards. Item 19 is about two sides
agreeing, so the test has to materialise a global tensor, cut it both ways, and check
that what the reader assembles equals what the writer held. A mesh that is structurally
valid on each side independently can still misalign between them, and only this shape of
test catches that.

Both directions matter and are not symmetric in the code path:

* train TP=8 -> rollout TP=4: finest mesh is 8, so the **rollout** side refactors
  (one reader slice spans two writer shards -- a gather).
* train TP=4 -> rollout TP=8: finest mesh is 8, so the **train** side refactors
  (one writer shard feeds two reader slices -- a scatter).

The LCM reconciliation is what makes both work with the same code, which is the claim.

CPU-only: this is index arithmetic over small tensors, no NIXL and no GPU.
"""

from __future__ import annotations

from collections import OrderedDict
from math import gcd

import pytest
import torch

from psrl.utils.nixl.nixl_spec import NIXLSharding

pytestmark = pytest.mark.cpu_test


# Hidden sizes that divide cleanly by 8 (so both TP=4 and TP=8 are expressible) plus one
# that only divides by 4, to confirm the finest-mesh rejection path is reached rather
# than silently producing a wrong cut.
DIVISIBLE_BY_EIGHT = [128, 4096, 1536]


def _sharding(shard_count: int, index: int, dim: int = 0) -> NIXLSharding:
    return NIXLSharding(
        shard_mesh=OrderedDict([(dim, shard_count)]),
        shard_indices=[(index,)],
    )


def _global_tensor(dim_size: int, width: int = 3) -> torch.Tensor:
    """A tensor whose every element encodes its own global row index.

    Row `i` is filled with `i * width + column`, so a shard that lands at the wrong
    offset holds detectably wrong values rather than plausible-looking ones. An
    all-ones or all-zeros tensor would make a misaligned cut invisible.
    """
    return torch.arange(dim_size * width, dtype=torch.int64).reshape(dim_size, width)


def _local_slice(global_tensor: torch.Tensor, shard_count: int, index: int) -> torch.Tensor:
    """The contiguous slice rank `index` owns under a `shard_count`-way split."""
    dim_size = global_tensor.shape[0]
    assert dim_size % shard_count == 0, "caller must ensure even division"
    chunk = dim_size // shard_count
    return global_tensor[index * chunk : (index + 1) * chunk]


def _refactor_to_finest(
    train_count: int, rollout_count: int, holder_count: int, holder_index: int
) -> NIXLSharding:
    """Refactor one side's sharding onto the mesh both sides agree on."""
    finest = NIXLSharding.find_finest_shard_mesh(
        [OrderedDict([(0, train_count)]), OrderedDict([(0, rollout_count)])]
    )
    sharding = _sharding(holder_count, holder_index)
    sharding.refactor_based_on_finer_shard_mesh(finest)
    return sharding


class TestFinestMeshReconcilesBothDirections:
    """The LCM must be the same regardless of which side is finer."""

    @pytest.mark.parametrize("train_tp,rollout_tp", [(8, 4), (4, 8)])
    def test_finest_mesh_is_the_lcm(self, train_tp, rollout_tp):
        finest = NIXLSharding.find_finest_shard_mesh(
            [OrderedDict([(0, train_tp)]), OrderedDict([(0, rollout_tp)])]
        )
        expected = train_tp * rollout_tp // gcd(train_tp, rollout_tp)
        assert finest[0] == expected, (
            f"finest mesh {finest[0]} != lcm({train_tp}, {rollout_tp}) = {expected}; "
            "the two sides would cut the tensor on different boundaries"
        )

    def test_finest_mesh_is_direction_independent(self):
        """Swapping which side is train must not change the agreed mesh.

        If it did, the mesh would depend on handshake arrival order, which is exactly the
        nondeterminism the LCM reconciliation exists to remove.
        """
        forward = NIXLSharding.find_finest_shard_mesh(
            [OrderedDict([(0, 8)]), OrderedDict([(0, 4)])]
        )
        backward = NIXLSharding.find_finest_shard_mesh(
            [OrderedDict([(0, 4)]), OrderedDict([(0, 8)])]
        )
        assert forward == backward


class TestTrainEightToRolloutFour:
    """train TP=8 -> rollout TP=4: the reader's slice spans two writer shards."""

    @pytest.mark.parametrize("dim_size", DIVISIBLE_BY_EIGHT)
    def test_reader_slice_reassembles_from_writer_shards(self, dim_size):
        """Each rollout rank's slice must equal the concatenation of its train shards.

        This is the gather direction: with finest mesh 8 and rollout TP=4, rollout rank
        `r` covers train shards `2r` and `2r+1`. Asserting on content rather than on
        shard counts is what makes an off-by-one offset fail here.
        """
        global_tensor = _global_tensor(dim_size)
        rollout_tp, train_tp = 4, 8

        for rollout_rank in range(rollout_tp):
            expected = _local_slice(global_tensor, rollout_tp, rollout_rank)

            sharding = _refactor_to_finest(train_tp, rollout_tp, rollout_tp, rollout_rank)
            shards = sharding.get_local_sharded_tensors(expected)

            assert torch.equal(torch.cat(shards, dim=0), expected), (
                f"rollout rank {rollout_rank} reassembled the wrong bytes at "
                f"dim_size={dim_size}"
            )

    @pytest.mark.parametrize("dim_size", DIVISIBLE_BY_EIGHT)
    def test_union_of_all_rollout_ranks_is_the_global_tensor(self, dim_size):
        """No row may be lost or claimed twice across the whole rollout group."""
        global_tensor = _global_tensor(dim_size)
        rollout_tp, train_tp = 4, 8

        assembled = []
        for rollout_rank in range(rollout_tp):
            local = _local_slice(global_tensor, rollout_tp, rollout_rank)
            sharding = _refactor_to_finest(train_tp, rollout_tp, rollout_tp, rollout_rank)
            assembled.extend(sharding.get_local_sharded_tensors(local))

        assert torch.equal(torch.cat(assembled, dim=0), global_tensor), (
            "the rollout group's shards do not tile the global tensor exactly"
        )

    @pytest.mark.parametrize("dim_size", DIVISIBLE_BY_EIGHT)
    def test_each_reader_slice_splits_into_two_shards(self, dim_size):
        """The finest mesh is twice the rollout count, so each slice must split in two.

        A single shard would mean the refactor silently no-op'd and the rollout side is
        still cutting on its own boundaries rather than the agreed ones.
        """
        global_tensor = _global_tensor(dim_size)
        rollout_tp, train_tp = 4, 8

        sharding = _refactor_to_finest(train_tp, rollout_tp, rollout_tp, 0)
        shards = sharding.get_local_sharded_tensors(
            _local_slice(global_tensor, rollout_tp, 0)
        )
        assert len(shards) == train_tp // rollout_tp, (
            f"expected {train_tp // rollout_tp} shards per rollout rank, got {len(shards)}"
        )


class TestTrainFourToRolloutEight:
    """train TP=4 -> rollout TP=8: one writer shard feeds two reader slices."""

    @pytest.mark.parametrize("dim_size", DIVISIBLE_BY_EIGHT)
    def test_writer_slice_reassembles_from_its_shards(self, dim_size):
        """Each train rank's slice must equal the concatenation of its refactored shards.

        The scatter direction. Here the *train* side is coarser, so it refactors onto
        mesh 8 and each train rank yields two shards -- one per rollout reader.
        """
        global_tensor = _global_tensor(dim_size)
        train_tp, rollout_tp = 4, 8

        for train_rank in range(train_tp):
            expected = _local_slice(global_tensor, train_tp, train_rank)

            sharding = _refactor_to_finest(train_tp, rollout_tp, train_tp, train_rank)
            shards = sharding.get_local_sharded_tensors(expected)

            assert torch.equal(torch.cat(shards, dim=0), expected), (
                f"train rank {train_rank} produced the wrong bytes at dim_size={dim_size}"
            )

    @pytest.mark.parametrize("dim_size", DIVISIBLE_BY_EIGHT)
    def test_rollout_rank_content_matches_what_train_produced(self, dim_size):
        """The end-to-end claim: what a rollout rank should hold equals what train sent.

        Built by reconstructing the global tensor from the train side's refactored shards
        in global order, then checking each rollout rank's expected slice against it. A
        mesh that misaligned the two sides would show up as a content mismatch even
        though both sides' own invariants hold.
        """
        global_tensor = _global_tensor(dim_size)
        train_tp, rollout_tp = 4, 8

        # What the train group collectively puts on the wire, in global row order.
        on_the_wire = []
        for train_rank in range(train_tp):
            local = _local_slice(global_tensor, train_tp, train_rank)
            sharding = _refactor_to_finest(train_tp, rollout_tp, train_tp, train_rank)
            on_the_wire.extend(sharding.get_local_sharded_tensors(local))
        delivered = torch.cat(on_the_wire, dim=0)

        # Every rollout rank's slice must be present at the right offset.
        for rollout_rank in range(rollout_tp):
            expected = _local_slice(global_tensor, rollout_tp, rollout_rank)
            chunk = dim_size // rollout_tp
            got = delivered[rollout_rank * chunk : (rollout_rank + 1) * chunk]
            assert torch.equal(got, expected), (
                f"rollout rank {rollout_rank} would read the wrong rows at "
                f"dim_size={dim_size}: the two sides disagree on the cut"
            )

    @pytest.mark.parametrize("dim_size", DIVISIBLE_BY_EIGHT)
    def test_each_writer_slice_splits_into_two_shards(self, dim_size):
        global_tensor = _global_tensor(dim_size)
        train_tp, rollout_tp = 4, 8

        sharding = _refactor_to_finest(train_tp, rollout_tp, train_tp, 0)
        shards = sharding.get_local_sharded_tensors(
            _local_slice(global_tensor, train_tp, 0)
        )
        assert len(shards) == rollout_tp // train_tp


class TestRoundTripAcrossTheMismatch:
    """A full write-then-read cycle must be byte-exact in both directions."""

    @pytest.mark.parametrize("train_tp,rollout_tp", [(8, 4), (4, 8)])
    @pytest.mark.parametrize("dim_size", DIVISIBLE_BY_EIGHT)
    def test_write_then_read_is_lossless(self, train_tp, rollout_tp, dim_size):
        """Cut by train, reassemble by rollout: the result must equal the original.

        This is the single assertion item 19 reduces to. It holds only if the finest mesh
        reconciliation is correct in both directions, which is why the test runs both.
        """
        global_tensor = _global_tensor(dim_size)

        # Train side cuts and sends.
        sent: list[torch.Tensor] = []
        for train_rank in range(train_tp):
            local = _local_slice(global_tensor, train_tp, train_rank)
            sharding = _refactor_to_finest(train_tp, rollout_tp, train_tp, train_rank)
            sent.extend(sharding.get_local_sharded_tensors(local))

        # Rollout side cuts the same global tensor its own way and receives.
        received: list[torch.Tensor] = []
        for rollout_rank in range(rollout_tp):
            local = _local_slice(global_tensor, rollout_tp, rollout_rank)
            sharding = _refactor_to_finest(train_tp, rollout_tp, rollout_tp, rollout_rank)
            received.extend(sharding.get_local_sharded_tensors(local))

        assert torch.equal(torch.cat(sent, dim=0), torch.cat(received, dim=0)), (
            f"train TP={train_tp} -> rollout TP={rollout_tp} at dim_size={dim_size}: "
            "the wire bytes and the reader's expectation disagree"
        )

    @pytest.mark.parametrize("train_tp,rollout_tp", [(8, 4), (4, 8)])
    def test_shard_count_is_the_finest_mesh_either_way(self, train_tp, rollout_tp):
        """Both sides must produce the same total shard count: the finest mesh size."""
        global_tensor = _global_tensor(4096)
        finest = train_tp * rollout_tp // gcd(train_tp, rollout_tp)

        for count in (train_tp, rollout_tp):
            total = 0
            for rank in range(count):
                local = _local_slice(global_tensor, count, rank)
                sharding = _refactor_to_finest(train_tp, rollout_tp, count, rank)
                total += len(sharding.get_local_sharded_tensors(local))
            assert total == finest, (
                f"side with TP={count} produced {total} shards, expected finest mesh {finest}"
            )
