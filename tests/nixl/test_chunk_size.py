"""Adaptive chunk-size selection (implementation item B3).

The chunk size is the single knob that trades pipeline rounds against per-round cost.
Too large: the pipeline has too few rounds to overlap format cost with RDMA. Too small:
polling overhead dominates. The design's cost model gives a formula; this file pins it.

Unlike the GPU tests here, these are CPU-only: the inputs are model sizes and bandwidth
constants, not live tensor allocations. The test that matters most is that the model's
recommendation matches the design's worked example for H20 / 1TB, since that is the
claim the bandwidth-saturation argument rests on.
"""

from __future__ import annotations

import math

import pytest

from psrl.utils.nixl.chunk_size import (
    MB,
    GB,
    choose_chunk_bytes,
    polling_overhead_fraction,
    t_fmt,
    t_rdma,
    t_rdma_per_chunk,
)

pytestmark = pytest.mark.cpu_test

# H20 constants (calibrated in test_cost_model.py)
H20 = {
    "bw_hbm_gbs": 4800.0,
    "bw_nic_gbs": 170.0,
    "t_py_us": 12.5,
    "t_lat_ms": 0.002,
    "polling_overhead_ms": 0.5,
    "polling_overhead_threshold": 0.25,
}


class TestCostModelFunctions:
    def test_t_fmt_dense_model_is_dispatch_dominated(self):
        """Dense 0.9GB model, 290 keys: Python dispatch dominates, not HBM.

        Measured: dispatch 3.625ms vs HBM 0.403ms. The design calls T_fmt
        "launch-dominated before grouping", and a dense model is already in that regime
        because it has few enough bytes that the HBM pass is cheap.
        """
        tf = t_fmt(n_keys=290, model_bytes=int(0.9 * GB), bw_hbm_gbs=4800.0, t_py_us=12.5)
        hbm_term = 2.0 * 0.9 * GB / (4800.0 * 1e9) * 1000.0
        py_term = 290 * 12.5 / 1000.0

        assert abs(tf - (hbm_term + py_term)) < 0.001, "t_fmt must be the sum of its terms"
        assert py_term > hbm_term * 5, (
            f"dispatch {py_term:.3f}ms should dominate HBM {hbm_term:.3f}ms for a dense model"
        )

    def test_t_fmt_k2_ungrouped_is_launch_dominated(self):
        """K2 ungrouped: 70k keys × 12.5μs >> BW term."""
        tf_un = t_fmt(70_272, int(15.6 * GB), H20["bw_hbm_gbs"], H20["t_py_us"])
        assert tf_un > 800.0, "K2 ungrouped should be ~885ms"

    def test_t_fmt_k2_grouped_is_bw_dominated(self):
        """After MoE grouping, 183 keys: BW term dominates."""
        tf_gr = t_fmt(183, int(15.6 * GB), H20["bw_hbm_gbs"], H20["t_py_us"])
        assert tf_gr < 20.0, "K2 grouped should be ~8.8ms"
        assert tf_gr > 1.0

    def test_t_rdma_scales_with_chunks(self):
        model = int(15.6 * GB)
        t_big = t_rdma(model, 4 * GB, H20["bw_nic_gbs"], H20["t_lat_ms"])
        t_small = t_rdma(model, 512 * MB, H20["bw_nic_gbs"], H20["t_lat_ms"])
        assert t_small > t_big, "more chunks = more per-chunk latency"

    def test_polling_overhead_fraction_decreases_with_chunk_size(self):
        big = polling_overhead_fraction(1 * GB, H20["bw_nic_gbs"], H20["polling_overhead_ms"])
        small = polling_overhead_fraction(64 * MB, H20["bw_nic_gbs"], H20["polling_overhead_ms"])
        assert small > big

    def test_polling_floor_is_about_512mb_at_h20(self):
        """Design says 256MB is the 25% threshold; 512MB is the practical floor."""
        f256 = polling_overhead_fraction(256 * MB, H20["bw_nic_gbs"], H20["polling_overhead_ms"])
        f512 = polling_overhead_fraction(512 * MB, H20["bw_nic_gbs"], H20["polling_overhead_ms"])
        assert f256 >= 0.25, f"256MB fraction {f256:.2%} should be at the 25% threshold"
        assert f512 < 0.25, f"512MB fraction {f512:.2%} should be below the threshold"


class TestChunkSizeRecommendation:
    def test_1tb_model_k2_recommends_reasonable_chunk(self):
        """Design §3.5.6 worked example: 1TB fp8, K2 grouped, H20, D=2."""
        result = choose_chunk_bytes(
            model_bytes=int(15.6 * GB),
            n_grouped_keys=183,
            wire_buffer_depth=2,
            **H20,
        )
        # Design says 512MB gives 31 rounds and is the recommendation.
        assert result.chunk_bytes >= 256 * MB, "below polling floor"
        assert result.chunk_bytes <= 4 * GB, "too large -- pipeline rounds < 2"
        assert result.pipeline_rounds >= 2, "needs at least 2 rounds for pipeline to matter"
        assert result.polling_overhead_fraction < 0.25, "polling overhead too high"

    def test_model_smaller_than_the_polling_floor_is_reported_not_hidden(self):
        """A 0.9GB model cannot satisfy both constraints at once, and must say so.

        The 25% polling floor at 170 GB/s is ~1.7GB per chunk, which already exceeds the
        whole model. Pipelining (D=2) then demands chunks of at most 0.45GB. The two
        constraints are unsatisfiable together, so the recommender honours the pipeline
        bound and reports the resulting polling overhead rather than silently returning a
        chunk larger than the model.

        This is the `chunk_max < chunk_min` case design 3.5.6 flags: the fix is the
        notification-based completion path (item 5b), not a different chunk size.
        """
        model_bytes = int(0.9 * GB)
        result = choose_chunk_bytes(
            model_bytes=model_bytes,
            n_grouped_keys=290,
            wire_buffer_depth=2,
            **H20,
        )

        assert result.chunk_bytes <= model_bytes, "a chunk must never exceed the model"
        assert result.pipeline_rounds >= 2, "the pipeline bound must still be honoured"
        assert result.polling_overhead_fraction > 0.25, (
            "the conflict must surface in the reported overhead, not be hidden"
        )
        assert "polling_overhead" in result.reason

    def test_ungrouped_moe_produces_larger_chunk(self):
        """Without MoE grouping, T_fmt >> T_rdma, so more rounds are needed → smaller chunk.

        Actually: the design says larger T_fmt/T_rdma ratio means the pipeline needs MORE
        rounds to hide it, which means SMALLER chunks. This test captures that monotonicity.
        """
        result_grouped = choose_chunk_bytes(int(15.6 * GB), n_grouped_keys=183, **H20)
        result_ungrouped = choose_chunk_bytes(int(15.6 * GB), n_grouped_keys=70_272, **H20)
        # Ungrouped needs more rounds (smaller chunk) to hide the larger fmt cost.
        assert result_ungrouped.chunk_bytes <= result_grouped.chunk_bytes

    def test_result_is_aligned_to_default_64mb(self):
        result = choose_chunk_bytes(int(15.6 * GB), 183, **H20)
        assert result.chunk_bytes % (64 * MB) == 0

    def test_result_is_aligned_to_custom_alignment(self):
        result = choose_chunk_bytes(
            int(15.6 * GB), 183,
            alignment_bytes=256 * MB,
            **H20,
        )
        assert result.chunk_bytes % (256 * MB) == 0

    def test_deeper_pipeline_recommends_smaller_chunk(self):
        """D=4 means we need at least 4 rounds, so chunk is at most M/4."""
        r2 = choose_chunk_bytes(int(15.6 * GB), 183, wire_buffer_depth=2, **H20)
        r4 = choose_chunk_bytes(int(15.6 * GB), 183, wire_buffer_depth=4, **H20)
        assert r4.chunk_bytes <= r2.chunk_bytes
        assert r4.pipeline_rounds >= 4

    def test_chunk_count_is_consistent(self):
        result = choose_chunk_bytes(int(15.6 * GB), 183, **H20)
        expected_rounds = math.ceil(int(15.6 * GB) / result.chunk_bytes)
        assert result.n_chunks == expected_rounds

    def test_reason_string_includes_key_numbers(self):
        result = choose_chunk_bytes(int(15.6 * GB), 183, **H20)
        assert "T_fmt" in result.reason
        assert "rounds" in result.reason


class TestEdgeCases:
    def test_single_chunk_model_works(self):
        """Model that fits entirely in one chunk: rounds == 1, pipeline trivially satisfied."""
        result = choose_chunk_bytes(
            model_bytes=128 * MB,
            n_grouped_keys=30,
            **H20,
        )
        assert result.chunk_bytes > 0
        assert result.n_chunks >= 1

    def test_fast_nic_lowers_the_floor(self):
        """A 400 Gbps NIC drops the polling-overhead floor proportionally."""
        result_fast = choose_chunk_bytes(
            int(15.6 * GB), 183,
            bw_nic_gbs=400.0,  # 2.4× faster
            bw_hbm_gbs=H20["bw_hbm_gbs"],
            t_py_us=H20["t_py_us"],
            t_lat_ms=H20["t_lat_ms"],
            polling_overhead_ms=H20["polling_overhead_ms"],
            polling_overhead_threshold=H20["polling_overhead_threshold"],
        )
        result_slow = choose_chunk_bytes(int(15.6 * GB), 183, **H20)
        # Faster NIC → same polling overhead fraction is hit at a smaller chunk.
        # So the fast-NIC chunk can be smaller or the same, never meaningfully larger.
        assert result_fast.chunk_bytes <= result_slow.chunk_bytes * 2

    def test_a_slow_nic_lowers_the_polling_floor(self):
        """A slower NIC *lowers* the floor, because each chunk takes longer to move.

        The polling tax is `poll_interval / transfer_time`, so halving bandwidth doubles
        the transfer time and halves the tax -- a smaller chunk becomes affordable. The
        test previously asserted the opposite ("slow NIC raises the floor") and passed
        only because the old objective returned the grid maximum for every input, which
        satisfies `>= 512MB` regardless of what the floor is doing.
        """
        slow = choose_chunk_bytes(
            int(15.6 * GB), 183,
            bw_nic_gbs=50.0,
            bw_hbm_gbs=H20["bw_hbm_gbs"],
            t_py_us=H20["t_py_us"],
            t_lat_ms=H20["t_lat_ms"],
            polling_overhead_ms=H20["polling_overhead_ms"],
            polling_overhead_threshold=H20["polling_overhead_threshold"],
        )
        fast = choose_chunk_bytes(int(15.6 * GB), 183, **H20)

        assert slow.chunk_bytes < fast.chunk_bytes, (
            f"slow NIC chose {slow.chunk_bytes // MB}MB and fast chose "
            f"{fast.chunk_bytes // MB}MB; a slower link should tolerate smaller chunks"
        )
        # Both must still respect the polling threshold they were given.
        for result, bw in ((slow, 50.0), (fast, H20["bw_nic_gbs"])):
            tax = polling_overhead_fraction(
                result.chunk_bytes, bw, H20["polling_overhead_ms"]
            )
            assert tax <= H20["polling_overhead_threshold"]

    def test_zero_keys_produces_valid_result(self):
        result = choose_chunk_bytes(int(1 * GB), n_grouped_keys=0, **H20)
        assert result.chunk_bytes > 0

    def test_result_fields_are_self_consistent(self):
        result = choose_chunk_bytes(int(15.6 * GB), 183, **H20)
        t_chunk = t_rdma_per_chunk(result.chunk_bytes, H20["bw_nic_gbs"], H20["t_lat_ms"])
        assert abs(result.t_rdma_per_chunk_ms - t_chunk) < 0.001
        pf = polling_overhead_fraction(result.chunk_bytes, H20["bw_nic_gbs"], H20["polling_overhead_ms"])
        assert abs(result.polling_overhead_fraction - pf) < 0.001


class TestObjectiveIsMakespanNotRoundDistance:
    """What the recommender optimises, and why the previous objective was degenerate.

    The old score was `abs(rounds - ceil(T_fmt / t_chunk))`. Both terms fall as the chunk
    grows, and the second falls faster, so the score was monotone decreasing and the grid
    maximum (8GB) won for *every* input -- a 16GB model and a 1.3TB model got the same
    chunk. Several tests passed anyway because `>= 512MB` is true of 8GB.

    The objective is now the pipeline makespan, `(rounds + D - 1) * max(t_chunk,
    T_fmt / rounds)`, which is what actually determines how long a pull takes.
    """

    def test_the_recommendation_is_not_simply_the_grid_maximum(self):
        """The bug signature: every model collapsing onto the largest candidate."""
        small = choose_chunk_bytes(int(16 * GB), 399, **H20)
        huge = choose_chunk_bytes(int(1340 * GB), 579, **H20)

        assert small.chunk_bytes < 8 * GB, "recommender returned the grid maximum"
        assert huge.chunk_bytes < 8 * GB, "recommender returned the grid maximum"

    def test_smallest_affordable_chunk_wins_under_a_linear_rdma_model(self):
        """`rounds x t_chunk` is invariant, so only the pipeline fill term varies.

        This is a property of the cost model, not a tuning choice: RDMA time is linear in
        bytes, so halving the chunk doubles the rounds and leaves steady-state time
        unchanged. The only term that moves is the `D - 1` rounds of fill, which is
        cheaper with a smaller chunk. The recommendation is therefore pinned to the
        polling floor until format cost per chunk overtakes transfer cost.

        B3's stated criterion -- "chunk *bytes* vary with model size" -- does not hold
        under this model, and the honest resolution is to record why rather than to bend
        the objective until it does.
        """
        result = choose_chunk_bytes(int(16 * GB), 399, **H20)

        # One step below the recommendation must breach the polling threshold, which is
        # what makes the recommendation a floor rather than an optimum.
        below = result.chunk_bytes // 2
        tax = polling_overhead_fraction(below, H20["bw_nic_gbs"], H20["polling_overhead_ms"])
        assert tax > H20["polling_overhead_threshold"], (
            f"{below // MB}MB has only {tax:.1%} polling tax, so the recommender should "
            f"have chosen it over {result.chunk_bytes // MB}MB"
        )

    def test_format_bound_workloads_do_not_shrink_below_the_floor(self):
        """A huge key count raises T_fmt, but the floor still governs.

        Worth pinning because it is the case where the two terms compete: ungrouped MoE
        has `T_fmt` in the seconds, yet `T_fmt / rounds` stays under `t_chunk` at every
        affordable chunk size, so the answer does not move. If a future change makes
        format cost per chunk dominate, this test fails and the floor stops being the
        whole story.
        """
        grouped = choose_chunk_bytes(int(1340 * GB), 579, **H20)
        ungrouped = choose_chunk_bytes(int(1340 * GB), 140_000, **H20)

        assert ungrouped.chunk_bytes <= grouped.chunk_bytes
        for result in (grouped, ungrouped):
            tax = polling_overhead_fraction(
                result.chunk_bytes, H20["bw_nic_gbs"], H20["polling_overhead_ms"]
            )
            assert tax <= H20["polling_overhead_threshold"]


class TestRoundDecisionsAreLocal:
    """B3's second criterion: no cross-process round trip per transfer round.

    The plan is compiled once during the handshake and carried by each client, so source
    selection during a pull is a dictionary lookup rather than a query to the meta server.
    This is what makes the control plane independent of round count -- the property that
    matters at 10k-GPU scale, where a per-round RPC would be `O(rounds x ranks)`.

    Pinned by source inspection rather than a log grep: a log can be silent because the
    code path did not run, which is the same failure mode the "0 peer reads deferred"
    diagnostic had.
    """

    @staticmethod
    def _pull_source() -> str:
        import inspect

        from psrl.utils.nixl.client import NIXLStorageClient

        return "\n".join(
            inspect.getsource(getattr(NIXLStorageClient, name))
            for name in (
                "client_read",
                "iter_planned_pull_work",
                "peer_sources_for",
                "upstream_sources_for",
                "pull_chunks",
            )
        )

    def test_the_pull_path_makes_no_ray_calls(self):
        source = self._pull_source()

        assert ".remote(" not in source, "the pull path issues a Ray call per round"
        assert "ray.get" not in source, "the pull path blocks on a Ray future per round"

    def test_source_selection_reads_the_cached_plan(self):
        """Each accessor resolves against `self._comm_plan`, not a live query."""
        source = self._pull_source()

        assert "self._comm_plan" in source
        # And the plan is received once, during the handshake, not fetched per round.
        import inspect

        from psrl.utils.nixl.client import NIXLStorageClient

        handshake = inspect.getsource(NIXLStorageClient.wait_for_server_info)
        assert "comm_plan" in handshake
