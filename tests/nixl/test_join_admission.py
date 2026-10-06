"""Unit tests for join admission (implementation item A7, design doc 3.3.1d).

The third verdict is the load-bearing one: if a join raises the finest mesh to a
value that no longer divides some tensor dimension, admitting it makes *every*
already-registered client raise `ValueError` inside
`NIXLSharding.get_local_sharded_tensors` at re-registration time. That turns a
scale-up into a cluster-wide outage, which is why it is refused at admission rather
than warned about.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest

from psrl.utils.nixl.join_admission import (
    JoinVerdict,
    classify_join,
    merge_finest_mesh,
    suggest_divisor_degrees,
)
from psrl.utils.nixl.nixl_spec import NIXLSharding

# A realistic starting point: train TP=8 meeting rollout TP=4 -> finest = lcm(8,4) = 8.
ESTABLISHED = OrderedDict([(0, 8)])
# Qwen-class hidden size; divisible by 8/16 but not by 24.
SHAPES = {"layers.0.attn.qkv.weight": (4096, 4096)}


class TestFastPath:
    """A joiner whose mesh divides the current LCM must change nothing."""

    @pytest.mark.parametrize("degree", [1, 2, 4, 8])
    def test_divisor_degrees_keep_mesh_unchanged(self, degree):
        d = classify_join(ESTABLISHED, {0: degree}, SHAPES)
        assert d.verdict is JoinVerdict.FAST_PATH
        assert d.new_finest == dict(ESTABLISHED), "FAST_PATH must leave the finest mesh identical"
        assert not d.requires_reshard
        assert d.admitted

    def test_first_client_is_always_fast_path(self):
        """With no established mesh there is nothing to invalidate or re-shard.

        Note the joiner's mesh need not divide its own tensors here -- that is
        `get_sharding_for_param`'s declaration-time assertion (item A5), not an
        admission question. Admission only answers "does this break existing clients".
        """
        d = classify_join({}, {0: 7}, SHAPES)
        assert d.verdict is JoinVerdict.FAST_PATH
        assert d.new_finest == {0: 7}

    def test_dp_scale_up_is_fast_path(self):
        """Train-side sharding carries no DP dimension, so DP scale-up adds no mesh."""
        d = classify_join(ESTABLISHED, {}, SHAPES)
        assert d.verdict is JoinVerdict.FAST_PATH
        assert d.new_finest == dict(ESTABLISHED)


class TestReshard:
    """LCM rises but stays expressible: legal, warn, proceed."""

    def test_tp16_against_tp8_reshards(self):
        """lcm(8,16)=16 and 4096 % 16 == 0 -> expressible, so admit with a warning."""
        d = classify_join(ESTABLISHED, {0: 16}, SHAPES)
        assert d.verdict is JoinVerdict.RESHARD
        assert d.new_finest == {0: 16}
        assert d.requires_reshard
        assert d.admitted, "a re-shard is a cost, not an error"

    def test_tp3_is_expressible_when_dim_allows(self):
        """hidden=1536: lcm(8,3)=24 and 1536 % 24 == 0, so TP=3 is admissible here."""
        d = classify_join(ESTABLISHED, {0: 3}, {"w": (1536, 1536)})
        assert d.verdict is JoinVerdict.RESHARD

    def test_describe_offers_an_alternative(self):
        """The warning is only actionable if it names a safe degree."""
        d = classify_join(ESTABLISHED, {0: 16}, SHAPES)
        assert d.suggested_degrees
        assert all(max(ESTABLISHED.values()) % s == 0 for s in d.suggested_degrees)
        assert "DP dimension" in d.describe()


class TestReject:
    """The correctness case: inexpressible sharding must not be admitted."""

    def test_tp3_against_tp8_with_4096_hidden_is_rejected(self):
        """lcm(8,3)=24, and 4096 % 24 == 64/3 -> not an integer. Must reject."""
        d = classify_join(ESTABLISHED, {0: 3}, SHAPES)
        assert d.verdict is JoinVerdict.REJECT
        assert not d.admitted
        assert d.blocking_dims, "REJECT must name the offending dimensions"
        key, dim, size, count = d.blocking_dims[0]
        assert (dim, size, count) == (0, 4096, 24)

    def test_rejection_leaves_current_mesh_untouched(self):
        """The whole point: existing clients keep serving after a refused join."""
        before = dict(ESTABLISHED)
        d = classify_join(ESTABLISHED, {0: 3}, SHAPES)
        assert d.verdict is JoinVerdict.REJECT
        assert d.current_finest == before
        assert dict(ESTABLISHED) == before, "classify_join must not mutate its input"

    def test_rejection_message_distinguishes_itself_from_a_cost_warning(self):
        """A user who reads 'slow' and retries will keep failing; say 'inexpressible'."""
        text = classify_join(ESTABLISHED, {0: 3}, SHAPES).describe()
        assert "inexpressible" in text
        assert "ValueError" in text

    def test_one_bad_tensor_among_many_is_enough(self):
        """Divisibility must hold for every registered tensor, not most of them."""
        shapes = {"ok": (1536, 1536), "bad": (4096, 4096)}
        d = classify_join(ESTABLISHED, {0: 3}, shapes)
        assert d.verdict is JoinVerdict.REJECT
        assert [k for k, *_ in d.blocking_dims] == ["bad"]


class TestAgreementWithShardingRuntime:
    """The predicate must match what actually raises at re-registration time.

    This is the test that makes A7 meaningful, and it is the one that caught the
    first implementation being wrong: it checked ``global_dim % new_count``, whereas
    the runtime refactors each client's *local* slice and checks divisibility against
    the local shard count. The two disagree in both directions.
    """

    @staticmethod
    def _runtime_raises(dim_size: int, old_count: int, new_count: int) -> bool:
        """Replay refactor + local split exactly as re-registration would."""
        import torch

        sharding = NIXLSharding(
            shard_mesh=OrderedDict([(0, old_count)]),
            shard_indices=[(0,)],  # this client holds 1 of old_count shards
        )
        try:
            sharding.refactor_based_on_finer_shard_mesh(OrderedDict([(0, new_count)]))
        except Exception:
            return True
        if dim_size % old_count != 0:
            return True
        local = torch.zeros(dim_size // old_count, 4, dtype=torch.uint8)
        try:
            sharding.get_local_sharded_tensors(local)
        except ValueError:
            return True
        return False

    @pytest.mark.parametrize("dim_size", [4096, 1536, 512, 100])
    @pytest.mark.parametrize("old_count", [4, 8])
    @pytest.mark.parametrize("new_degree", [3, 4, 6, 8, 12, 16, 24])
    def test_verdict_matches_runtime_exhaustively(self, dim_size, old_count, new_degree):
        established = OrderedDict([(0, old_count)])
        shapes = {"w": (dim_size, 4)}
        decision = classify_join(established, {0: new_degree}, shapes)

        new_count = decision.new_finest[0]
        if new_count == old_count:
            assert decision.verdict is JoinVerdict.FAST_PATH
            return

        expected_reject = self._runtime_raises(dim_size, old_count, new_count)
        got_reject = decision.verdict is JoinVerdict.REJECT
        assert got_reject == expected_reject, (
            f"dim={dim_size} {old_count}->{new_count}: verdict={decision.verdict.value} "
            f"but runtime {'raises' if expected_reject else 'succeeds'}"
        )


class TestHelpers:
    def test_merge_is_per_dimension_lcm(self):
        assert merge_finest_mesh({0: 8}, {0: 4}) == {0: 8}
        assert merge_finest_mesh({0: 8}, {0: 3}) == {0: 24}
        assert merge_finest_mesh({0: 8}, {1: 4}) == {0: 8, 1: 4}

    def test_merge_does_not_mutate_input(self):
        current = {0: 8}
        merge_finest_mesh(current, {0: 3})
        assert current == {0: 8}

    def test_suggestions_all_divide_the_current_mesh(self):
        for requested in (3, 5, 7, 24):
            for s in suggest_divisor_degrees({0: 8}, requested):
                assert 8 % s == 0

    def test_suggestions_are_ordered_by_closeness(self):
        """Nearest safe degree first; ties break toward more parallelism."""
        assert suggest_divisor_degrees({0: 8}, 3)[0] == 4  # 2 and 4 tie, prefer 4
        assert suggest_divisor_degrees({0: 8}, 7)[0] == 8
        assert suggest_divisor_degrees({0: 8}, 5)[0] == 4

    def test_no_suggestions_without_an_established_mesh(self):
        assert suggest_divisor_degrees({}, 4) == []


class TestMultiDimensional:
    """MoE after grouping has {0: EP, 1: ETP}; both dimensions must be checked."""

    def test_expert_dim_rejection(self):
        established = OrderedDict([(0, 8), (1, 2)])
        shapes = {"experts.w13": (8, 4096, 4096)}  # dim0=8 experts
        d = classify_join(established, {0: 3}, shapes)
        assert d.verdict is JoinVerdict.REJECT  # lcm(8,3)=24 on dim0, 8 % 24 != 0

    def test_second_dim_can_be_the_offender(self):
        established = OrderedDict([(0, 2), (1, 8)])
        shapes = {"experts.w13": (8, 4096, 100)}
        d = classify_join(established, {1: 3}, shapes)
        assert d.verdict is JoinVerdict.REJECT
        assert d.blocking_dims[0][1] == 1, "must attribute the failure to dim 1"
