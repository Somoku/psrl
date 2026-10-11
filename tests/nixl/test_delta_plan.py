"""When delta transfer is worth its diff (implementation item F1).

F1 is explicitly opt-in and orthogonal to the main path, so what matters is that the
decision is *computed* rather than assumed. Two results drive these tests:

* The link speed alone does not refuse it. I expected the cluster's measured ~103 GB/s
  aggregate to make diffing not worth it; the arithmetic says the opposite, and so does
  same-node NVLink at 343 GB/s. What keeps F1 opt-in is the *change rate*, which nothing
  here can predict -- early in training nearly every element moves.
* The index term is what sets the break-even. Each changed element ships its position as
  well as its value, so with bf16 values and 4-byte indices a changed element costs 3x a
  resent one -- delta stops paying at roughly a third changed, not at 100%.
"""

from __future__ import annotations

import pytest

from psrl.utils.nixl.delta_plan import (
    break_even_fraction,
    estimate_delta,
    should_use_delta,
)

pytestmark = pytest.mark.cpu_test

GB = 1 << 30
BF16 = 2
FP8 = 1


class TestByteEstimate:
    def test_a_small_change_moves_few_bytes(self):
        estimate = estimate_delta(shard_bytes=GB, element_bytes=BF16, changed_fraction=0.01)

        assert estimate.byte_ratio < 0.05
        assert estimate.is_worthwhile

    def test_the_index_term_is_counted(self):
        """Ignoring indices would make delta look worthwhile at any fraction below 1.0."""
        estimate = estimate_delta(shard_bytes=GB, element_bytes=BF16, changed_fraction=0.5)

        # 50% changed at 2-byte values + 4-byte indices = 1.5x the full shard.
        assert estimate.byte_ratio > 1.0
        assert not estimate.is_worthwhile

    def test_the_snapshot_is_one_shard_not_the_model(self):
        """Following verl: no rank holds a full-model diff base.

        This is the correction that makes F1 compatible with the no-centralised-CPU-copy
        constraint -- the residency is `M / ranks` and shrinks as the cluster grows.
        """
        estimate = estimate_delta(shard_bytes=GB, element_bytes=BF16, changed_fraction=0.1)

        assert estimate.snapshot_bytes_per_rank == GB

    def test_an_out_of_range_fraction_is_rejected(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            estimate_delta(shard_bytes=GB, element_bytes=BF16, changed_fraction=1.5)


class TestBreakEven:
    def test_bf16_breaks_even_around_a_third(self):
        assert break_even_fraction(BF16) == pytest.approx(2 / 6)

    def test_quantised_wires_break_even_earlier_not_later(self):
        """Counter-intuitive and worth pinning: smaller values do not favour deltas.

        The index stays 4 bytes while the value halves, so each changed element becomes
        relatively more expensive and the break-even fraction falls.
        """
        assert break_even_fraction(FP8) < break_even_fraction(BF16)

    def test_larger_values_favour_deltas(self):
        assert break_even_fraction(4) > break_even_fraction(BF16)


class TestLinkSpeedDecidesIt:
    def test_a_link_faster_than_the_diff_makes_it_not_worth_it(self):
        """The real boundary: a delta loses once the link outruns the diff pass.

        I expected the cluster's 103 GB/s to be past that boundary and it is not -- a 1GB
        shard diffs in 2.7ms and saves 9.8ms there. Nor is same-node NVLink at 343 GB/s
        (saves 2.9ms). The crossover sits above local copy bandwidth, which is to say a
        delta is byte-wise worthwhile on every link this cluster has.
        """
        use, reason = should_use_delta(
            shard_bytes=GB,
            element_bytes=BF16,
            changed_fraction=0.02,
            link_gbps=1000.0,
            diff_gbps=400.0,
        )

        assert not use
        assert "too fast" in reason

    def test_same_node_nvlink_still_favours_a_delta_on_bytes(self):
        """Pinned because it bounds how far the 'link too fast' refusal reaches."""
        use, _reason = should_use_delta(
            shard_bytes=GB,
            element_bytes=BF16,
            changed_fraction=0.02,
            link_gbps=343.0,
            diff_gbps=400.0,
        )

        assert use

    def test_a_cluster_speed_link_would_pay_if_the_change_rate_were_low(self):
        """Recorded because it contradicts the design's framing of F1 as cross-DC only.

        At the measured 103 GB/s a 2%-changed shard is a win on bytes. What keeps F1
        opt-in is not the link but the change rate: early in training nearly every element
        moves, and nothing here measures that -- it has to be observed per run.
        """
        use, _reason = should_use_delta(
            shard_bytes=GB,
            element_bytes=BF16,
            changed_fraction=0.02,
            link_gbps=103.0,
        )

        assert use

    def test_a_slow_cross_dc_link_makes_it_worth_it(self):
        """The case F1 exists for: bytes dominate once the link is slow enough."""
        use, reason = should_use_delta(
            shard_bytes=GB,
            element_bytes=BF16,
            changed_fraction=0.02,
            link_gbps=1.0,
        )

        assert use
        assert "%" in reason

    def test_a_large_change_is_refused_even_on_a_slow_link(self):
        """Byte count decides first: no link is slow enough to make 1.5x bytes a win."""
        use, reason = should_use_delta(
            shard_bytes=GB,
            element_bytes=BF16,
            changed_fraction=0.5,
            link_gbps=1.0,
        )

        assert not use
        assert "break-even" in reason

    def test_the_reason_distinguishes_refusal_from_absence(self):
        """"Delta was not used" must not be ambiguous with "delta was not configured"."""
        _use, reason = should_use_delta(
            shard_bytes=GB, element_bytes=BF16, changed_fraction=0.9, link_gbps=1.0
        )

        assert reason, "a refusal with no reason is indistinguishable from not asking"
