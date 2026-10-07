"""Exhaustive property tests for the shard algebra (pre-implementation verification 1).

`NIXLSharding` is the foundation every transfer rests on: `find_finest_shard_mesh`
decides the global LCM, `refactor_based_on_finer_shard_mesh` rewrites each client's
view onto it, and `get_local_sharded_tensors` cuts the actual bytes. A fault in any
of the three produces a transfer that completes and delivers wrong data.

The join-admission predicate (item A7) was initially wrong because hand-picked cases
missed the distinction between the global mesh and a client's local shard count. These
sweeps enumerate the combination space instead -- deterministically, since hypothesis
is not available here -- so the same class of mistake surfaces as a failure.

The invariants asserted:

1. coverage      -- refactoring then splitting reproduces exactly the original bytes
2. disjointness  -- no element is claimed by two shards
3. idempotence   -- refactoring to the mesh you already have is a no-op
4. LCM identity  -- find_finest_shard_mesh returns the per-dimension LCM
5. monotonicity  -- a divisor mesh never raises the finest mesh
6. agreement     -- when refactor+split raises, it is exactly when A7 says REJECT
"""

from __future__ import annotations

import itertools
from collections import OrderedDict
from math import gcd

import pytest
import torch

from psrl.utils.nixl.join_admission import JoinVerdict, classify_join
from psrl.utils.nixl.nixl_spec import NIXLSharding


def _lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


def _sharding(shard_count: int, index: int, dim: int = 0) -> NIXLSharding:
    return NIXLSharding(
        shard_mesh=OrderedDict([(dim, shard_count)]),
        shard_indices=[(index,)],
    )


# Dimension sizes chosen to span the interesting divisibility cases: powers of two,
# a highly composite value, and sizes that defeat common shard counts.
DIM_SIZES = [128, 1536, 4096, 100, 768]
SHARD_COUNTS = [1, 2, 3, 4, 6, 8, 16]


class TestFindFinestShardMesh:
    """Invariant 4 and 5: the finest mesh is the per-dimension LCM, and nothing else."""

    @pytest.mark.parametrize("a,b", list(itertools.product(SHARD_COUNTS, SHARD_COUNTS)))
    def test_is_per_dimension_lcm(self, a, b):
        finest = NIXLSharding.find_finest_shard_mesh(
            [OrderedDict([(0, a)]), OrderedDict([(0, b)])]
        )
        assert finest[0] == _lcm(a, b)

    @pytest.mark.parametrize("counts", list(itertools.combinations(SHARD_COUNTS, 3)))
    def test_associative_over_three_clients(self, counts):
        """Order of clients must not change the result, or handshake becomes racy."""
        meshes = [OrderedDict([(0, c)]) for c in counts]
        reference = NIXLSharding.find_finest_shard_mesh(meshes)
        for permutation in itertools.permutations(meshes):
            assert NIXLSharding.find_finest_shard_mesh(list(permutation)) == reference

    @pytest.mark.parametrize("base", [4, 8, 16])
    @pytest.mark.parametrize("divisor_factor", [1, 2, 4])
    def test_divisor_mesh_does_not_raise_finest(self, base, divisor_factor):
        """Invariant 5: this is exactly A7's FAST_PATH condition."""
        if base % divisor_factor != 0:
            pytest.skip("not a divisor")
        finest = NIXLSharding.find_finest_shard_mesh(
            [OrderedDict([(0, base)]), OrderedDict([(0, base // divisor_factor)])]
        )
        assert finest[0] == base

    def test_multi_dimensional_lcm_is_independent_per_dimension(self):
        """MoE after grouping is {0: EP, 1: ETP}; the dimensions must not interact."""
        finest = NIXLSharding.find_finest_shard_mesh(
            [OrderedDict([(0, 8), (1, 2)]), OrderedDict([(0, 3), (1, 4)])]
        )
        assert finest == OrderedDict([(0, 24), (1, 4)])


class TestRefactorPreservesBytes:
    """Invariants 1-3: refactoring must be a pure re-description of the same bytes."""

    @staticmethod
    def _refactor_and_split(dim_size: int, old_count: int, new_count: int, holder: int):
        """Return the local shards a holder ends up with, or None if it raises."""
        sharding = _sharding(old_count, holder)
        try:
            sharding.refactor_based_on_finer_shard_mesh(OrderedDict([(0, new_count)]))
        except Exception:
            return None
        if dim_size % old_count != 0:
            return None
        local = torch.arange(dim_size, dtype=torch.int32).reshape(dim_size, 1)
        chunk = dim_size // old_count
        local_slice = local[holder * chunk : (holder + 1) * chunk]
        try:
            return sharding.get_local_sharded_tensors(local_slice)
        except ValueError:
            return None

    @pytest.mark.parametrize("dim_size", DIM_SIZES)
    @pytest.mark.parametrize("old_count", [2, 4, 8])
    @pytest.mark.parametrize("growth", [1, 2, 3, 4])
    def test_shards_cover_the_local_slice_exactly(self, dim_size, old_count, growth):
        """Invariant 1 + 2: concatenating the shards rebuilds the slice, in order."""
        if dim_size % old_count != 0:
            pytest.skip("establishing mesh already uneven")
        new_count = old_count * growth
        holder = old_count // 2

        shards = self._refactor_and_split(dim_size, old_count, new_count, holder)
        if shards is None:
            pytest.skip("inexpressible; covered by the agreement test")

        chunk = dim_size // old_count
        expected = torch.arange(holder * chunk, (holder + 1) * chunk, dtype=torch.int32).reshape(-1, 1)
        assert torch.equal(torch.cat(shards, dim=0), expected)
        assert sum(s.shape[0] for s in shards) == chunk

    @pytest.mark.parametrize("dim_size", DIM_SIZES)
    @pytest.mark.parametrize("count", [1, 2, 4, 8])
    def test_refactor_to_same_mesh_is_identity(self, dim_size, count):
        """Invariant 3: a FAST_PATH join must not perturb existing descriptors."""
        if dim_size % count != 0:
            pytest.skip("uneven")
        sharding = _sharding(count, 0)
        before_mesh = OrderedDict(sharding.shard_mesh)
        before_indices = list(sharding.shard_indices)

        sharding.refactor_based_on_finer_shard_mesh(OrderedDict([(0, count)]))

        assert OrderedDict(sharding.shard_mesh) == before_mesh
        assert sharding.shard_indices == before_indices

    @pytest.mark.parametrize("old_count", [2, 4])
    @pytest.mark.parametrize("growth", [2, 3])
    def test_all_holders_together_cover_the_global_tensor(self, old_count, growth):
        """Across holders, the union must be the whole tensor with no overlap."""
        dim_size = 4096 if (4096 % (old_count * growth) == 0) else 1536
        if dim_size % (old_count * growth) != 0:
            pytest.skip("inexpressible")
        new_count = old_count * growth

        seen: list[int] = []
        for holder in range(old_count):
            shards = self._refactor_and_split(dim_size, old_count, new_count, holder)
            assert shards is not None
            for shard in shards:
                seen.extend(shard.flatten().tolist())

        assert sorted(seen) == list(range(dim_size)), "shards must partition the tensor"

    @pytest.mark.parametrize("count", [2, 4, 8])
    def test_global_shard_indices_are_unique_across_holders(self, count):
        """Two holders claiming the same global index would double-transfer it."""
        claimed: set[tuple[int, ...]] = set()
        for holder in range(count):
            sharding = _sharding(count, holder)
            sharding.refactor_based_on_finer_shard_mesh(OrderedDict([(0, count * 2)]))
            for index in sharding.shard_indices:
                assert index not in claimed, f"index {index} claimed twice"
                claimed.add(index)
        assert len(claimed) == count * 2


class TestAgreementWithJoinAdmission:
    """Invariant 6: A7's predicate and the runtime must agree on every combination."""

    @pytest.mark.parametrize("dim_size", DIM_SIZES)
    @pytest.mark.parametrize("old_count", [2, 4, 8])
    @pytest.mark.parametrize("new_degree", SHARD_COUNTS)
    def test_reject_coincides_with_runtime_failure(self, dim_size, old_count, new_degree):
        if dim_size % old_count != 0:
            pytest.skip("establishing mesh already uneven")

        decision = classify_join(
            OrderedDict([(0, old_count)]), {0: new_degree}, {"w": (dim_size, 4)}
        )
        new_count = decision.new_finest[0]
        if new_count == old_count:
            assert decision.verdict is JoinVerdict.FAST_PATH
            return

        runtime_ok = (
            TestRefactorPreservesBytes._refactor_and_split(dim_size, old_count, new_count, 0)
            is not None
        )
        assert runtime_ok == (decision.verdict is not JoinVerdict.REJECT), (
            f"dim={dim_size} {old_count}->{new_count}: verdict={decision.verdict.value} "
            f"but runtime {'succeeded' if runtime_ok else 'failed'}"
        )


class TestMoEExpertSharding:
    """Two-dimensional meshes, which MoE grouping (item C1) introduces."""

    @pytest.mark.parametrize("num_experts,ep,etp", [(128, 8, 1), (128, 8, 2), (64, 4, 2), (64, 8, 8)])
    def test_expert_and_projection_dims_split_independently(self, num_experts, ep, etp):
        """shard_dim=0 cuts experts, shard_dim=1 cuts the projection."""
        intermediate = 768
        if num_experts % ep != 0 or intermediate % etp != 0:
            pytest.skip("uneven")

        sharding = NIXLSharding(
            shard_mesh=OrderedDict([(0, ep), (1, etp)]),
            shard_indices=[(0, 0)],
        )
        local = torch.zeros(num_experts // ep, intermediate // etp, dtype=torch.uint8)
        shards = sharding.get_local_sharded_tensors(local)

        assert len(shards) == 1, "a single-index holder owns exactly one shard"
        assert shards[0].shape == local.shape

    def test_expert_dim_refactor_preserves_expert_boundaries(self):
        """Splitting experts must never cut an individual expert in half."""
        num_experts, ep = 128, 8
        sharding = _sharding(ep, 0)
        sharding.refactor_based_on_finer_shard_mesh(OrderedDict([(0, ep * 2)]))

        local = torch.arange(num_experts // ep, dtype=torch.int32).reshape(-1, 1)
        shards = sharding.get_local_sharded_tensors(local)

        assert len(shards) == 2
        assert all(s.shape[0] == (num_experts // ep) // 2 for s in shards)
        assert torch.equal(torch.cat(shards, dim=0), local)
