"""Verify the coordinate judge catches what the all-ones judge cannot (item T1).

This is a test *of a test*, and it is worth the effort: if the judge is weak, every
e2e result built on it is meaningless. Each test here injects a specific fault and
asserts two things -- that the all-ones judge passes (demonstrating the gap) and that
the coordinate judge fails (demonstrating the fix).

The fault list comes straight from design doc 7.0's table of what "all ones" misses:
transposition, shard displacement, dtype conversion error, and MoE expert mix-up.
"""

from __future__ import annotations

import pytest
import torch

from tests.nixl.coord_judge import (
    CoordinateCapacityError,
    assert_dtype_capacity,
    assert_matches_global_coord,
    assert_state_dict_matches_coord,
    encode_coord,
    exact_int_limit,
    fill_by_global_coord,
    fill_state_dict_by_coord,
)


def _all_ones_judge_passes(tensor: torch.Tensor) -> bool:
    """The judge being replaced: fill with ones, assert ones."""
    return bool(torch.all(tensor == 1.0))


class TestDetectsFaultsAllOnesMisses:
    """Each fault must be invisible to all-ones and visible to the coordinate judge."""

    def test_transposition(self):
        ones = torch.ones(4, 8)
        assert _all_ones_judge_passes(ones.t().contiguous()), "premise: ones survive transpose"

        tensor = fill_by_global_coord(torch.zeros(4, 8, dtype=torch.float32))
        transposed = tensor.t().contiguous()
        with pytest.raises(AssertionError, match="coordinate mismatch"):
            assert_matches_global_coord(transposed, key="w")

    def test_shard_displacement(self):
        """Reading a neighbour's shard: all-ones cannot see it, coordinates can."""
        ones = torch.ones(64)
        assert _all_ones_judge_passes(ones[16:32]), "premise: any slice of ones is ones"

        full = fill_by_global_coord(torch.zeros(64, dtype=torch.float32))
        wrong_shard = full[16:32]  # claimed to be shard 0 but actually shard 1
        with pytest.raises(AssertionError, match="coordinate mismatch"):
            assert_matches_global_coord(wrong_shard, global_offset=0, key="w")

    def test_correct_shard_passes(self):
        """The judge must not cry wolf on a correctly placed shard."""
        full = fill_by_global_coord(torch.zeros(64, dtype=torch.float32))
        assert_matches_global_coord(full[16:32], global_offset=16, key="w")

    def test_reshape_that_preserves_order_is_accepted(self):
        """A reshape is legal as long as the flat order is unchanged."""
        tensor = fill_by_global_coord(torch.zeros(4, 8, dtype=torch.float32))
        assert_matches_global_coord(tensor.reshape(8, 4), key="w")

    def test_dtype_truncation(self):
        """A value silently rounded by a narrow dtype must be reported."""
        ones_fp8 = torch.ones(16, dtype=torch.float32).to(torch.float8_e4m3fn)
        assert _all_ones_judge_passes(ones_fp8.to(torch.float32)), "premise: 1.0 survives fp8"

        tensor = fill_by_global_coord(torch.zeros(256, dtype=torch.float32))
        round_tripped = tensor.to(torch.float8_e4m3fn).to(torch.float32)
        with pytest.raises(AssertionError, match="coordinate mismatch"):
            assert_matches_global_coord(round_tripped, key="w")

    def test_expert_permutation(self):
        """MoE expert mix-up: swap two experts' rows."""
        experts = fill_by_global_coord(torch.zeros(8, 16, dtype=torch.float32))
        permuted = experts.clone()
        permuted[[2, 5]] = permuted[[5, 2]]

        assert _all_ones_judge_passes(torch.ones(8, 16)[[5, 2]].contiguous())
        with pytest.raises(AssertionError, match="coordinate mismatch"):
            assert_matches_global_coord(permuted, key="experts.w13")

    def test_zeroed_region(self):
        """A chunk that never arrived reads as zeros, which must not pass."""
        tensor = fill_by_global_coord(torch.zeros(64, dtype=torch.float32))
        tensor[32:48] = 0
        with pytest.raises(AssertionError, match="coordinate mismatch"):
            assert_matches_global_coord(tensor, key="w")


class TestErrorMessageIsActionable:
    """The message must localise the fault, or debugging a 1TB run is hopeless."""

    def test_reports_first_offending_index_and_count(self):
        tensor = fill_by_global_coord(torch.zeros(64, dtype=torch.float32))
        tensor[10] = 999
        with pytest.raises(AssertionError) as excinfo:
            assert_matches_global_coord(tensor, key="layers.0.mlp.w1")
        message = str(excinfo.value)
        assert "layers.0.mlp.w1" in message
        assert "flat index 10" in message
        assert "1 of 64" in message

    def test_reports_global_index_under_offset(self):
        """With an offset the global position is what identifies the shard."""
        tensor = fill_by_global_coord(torch.zeros(16, dtype=torch.float32), global_offset=48)
        tensor[2] = 999
        with pytest.raises(AssertionError, match=r"global 50"):
            assert_matches_global_coord(tensor, global_offset=48, key="w")

    def test_names_the_three_failure_shapes(self):
        tensor = fill_by_global_coord(torch.zeros(16, dtype=torch.float32))
        tensor[0] = 999
        with pytest.raises(AssertionError) as excinfo:
            assert_matches_global_coord(tensor, key="w")
        message = str(excinfo.value)
        assert "shard-offset" in message
        assert "transpose" in message
        assert "precision" in message


class TestDtypeCapacity:
    """Constraint 1: refuse to run rather than silently saturate."""

    @pytest.mark.parametrize(
        "dtype,limit",
        [
            (torch.float8_e4m3fn, 16),
            (torch.float8_e5m2, 8),
            (torch.bfloat16, 256),
            (torch.float32, 2**24),
        ],
    )
    def test_recorded_limits_are_actually_exact(self, dtype, limit):
        """Every integer up to the limit must survive a round trip; limit+1 need not."""
        values = torch.arange(0, limit + 1, dtype=torch.float32)
        round_tripped = values.to(dtype).to(torch.float32)
        assert torch.equal(values, round_tripped), f"{dtype} does not represent 0..{limit} exactly"

    def test_oversized_shape_is_refused_for_fp8(self):
        with pytest.raises(CoordinateCapacityError, match="distinct values"):
            assert_dtype_capacity((1024,), torch.float8_e4m3fn)

    def test_strict_fill_refuses_fp8_beyond_capacity(self):
        with pytest.raises(CoordinateCapacityError):
            fill_by_global_coord(torch.zeros(1024, dtype=torch.float8_e4m3fn))

    def test_non_strict_fp8_fill_round_trips(self):
        """With the modulus, fp8 values survive exactly -- that is the whole point."""
        tensor = fill_by_global_coord(
            torch.zeros(1024, dtype=torch.float8_e4m3fn), strict=False
        )
        assert_matches_global_coord(tensor, strict=False, key="w")

    def test_non_strict_fp8_still_detects_local_displacement(self):
        """A shift smaller than the modulus must still be caught."""
        tensor = fill_by_global_coord(
            torch.zeros(1024, dtype=torch.float8_e4m3fn), strict=False
        )
        shifted = torch.roll(tensor.to(torch.float32), 3).to(torch.float8_e4m3fn)
        with pytest.raises(AssertionError):
            assert_matches_global_coord(shifted, strict=False, key="w")

    def test_unknown_dtype_is_refused(self):
        with pytest.raises(CoordinateCapacityError, match="no exact-integer limit"):
            exact_int_limit(torch.complex64)


class TestInjectivity:
    """Constraint 2: distinct coordinates must get distinct values."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int32])
    def test_encoding_is_injective_within_capacity(self, dtype):
        limit = min(exact_int_limit(dtype), 4096)
        values = [encode_coord(i, dtype) for i in range(limit)]
        assert len(set(values)) == len(values)

    def test_swapping_two_elements_is_detected(self):
        """Injectivity's purpose: a permutation must not be invisible."""
        tensor = fill_by_global_coord(torch.zeros(32, dtype=torch.float32))
        tensor[[7, 19]] = tensor[[19, 7]]
        with pytest.raises(AssertionError):
            assert_matches_global_coord(tensor, key="w")


class TestStateDictLevel:
    """Cross-key mix-ups need disjoint coordinate ranges per key."""

    def test_offsets_are_disjoint_and_ordered(self):
        state = {
            "b.weight": torch.zeros(8, dtype=torch.float32),
            "a.weight": torch.zeros(16, dtype=torch.float32),
        }
        offsets = fill_state_dict_by_coord(state)
        assert offsets == {"a.weight": 0, "b.weight": 16}
        assert_state_dict_matches_coord(state, offsets)

    def test_swapped_keys_are_detected(self):
        """Two same-shaped tensors delivered to each other's key."""
        state = {
            "a.weight": torch.zeros(8, dtype=torch.float32),
            "b.weight": torch.zeros(8, dtype=torch.float32),
        }
        offsets = fill_state_dict_by_coord(state)
        state["a.weight"], state["b.weight"] = state["b.weight"], state["a.weight"]
        with pytest.raises(AssertionError, match="coordinate mismatch"):
            assert_state_dict_matches_coord(state, offsets)

    def test_missing_offset_is_reported(self):
        state = {"a.weight": torch.zeros(4, dtype=torch.float32)}
        offsets = fill_state_dict_by_coord(state)
        state["surprise.weight"] = torch.zeros(4, dtype=torch.float32)
        with pytest.raises(AssertionError, match="no recorded global offset"):
            assert_state_dict_matches_coord(state, offsets)


class TestToleranceMode:
    """Precision paths need a tolerance that still rejects displacement."""

    def test_rtol_accepts_bf16_round_trip(self):
        tensor = fill_by_global_coord(torch.zeros(128, dtype=torch.float32))
        round_tripped = tensor.to(torch.bfloat16).to(torch.float32)
        assert_matches_global_coord(round_tripped, key="w", rtol=1e-2)

    def test_rtol_still_rejects_displacement(self):
        """A tolerance for precision must not become a tolerance for wrong data."""
        tensor = fill_by_global_coord(torch.zeros(128, dtype=torch.float32))
        displaced = torch.roll(tensor, 1)
        with pytest.raises(AssertionError):
            assert_matches_global_coord(displaced, key="w", rtol=1e-2)


class TestStridedAndGradTargets:
    """Fill must work on what the converters actually hand over, not just plain tensors.

    Both cases below came out of the 5-node e2e run; the unit tests had only ever
    exercised freshly-allocated contiguous tensors, so neither was visible before.
    """

    def test_fill_writes_through_a_leaf_parameter(self):
        """Model parameters are leaf Variables requiring grad.

        An in-place write through a view of one is rejected by autograd, which is why
        the fill goes through `.data` inside `no_grad` -- the same thing ParamSyncPlan
        does in production.
        """
        param = torch.nn.Parameter(torch.zeros(32, 8))
        fill_by_global_coord(param, 0, strict=False)
        assert_matches_global_coord(param, 0, key="param", strict=False)

    def test_fill_is_in_place_on_a_strided_view(self):
        """QKV and MoE slices are strided views of a fused buffer, not contiguous.

        `.view(-1)` raises on these, and `.reshape(-1)` would silently produce a copy --
        the fill would then land nowhere and the judge would compare against zeros. The
        assertions check both that the base buffer changed and that nothing outside the
        view did.
        """
        base = torch.nn.Parameter(torch.zeros(4, 96, 16))
        view = base.data.narrow(1, 0, 64)
        assert not view.is_contiguous(), "test needs a non-contiguous target"

        fill_by_global_coord(view, 7, strict=False)
        assert_matches_global_coord(view, 7, key="strided", strict=False)

        assert base.data.narrow(1, 0, 64).ne(0).any(), "fill did not reach the base buffer"
        assert base.data.narrow(1, 64, 32).eq(0).all(), "fill spilled outside the view"
