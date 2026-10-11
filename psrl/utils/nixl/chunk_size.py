"""Adaptive wire-transfer chunk size from the design's cost model (item B3).

Design §3.5.6 derives two quantities:

    T_fmt(N_keys, M_v) = N_keys × T_py + 2 × M_v / BW_HBM   [ms]
    T_rdma(M_v, chunk) = M_v / BW_NIC + ceil(M_v/chunk) × T_lat  [ms]

The chunk size must satisfy two things simultaneously:

* **Enough pipeline rounds to hide format cost.** The design requires
  ceil(T_fmt / T_rdma_per_chunk) >= D (the wire-buffer depth, default 2), which means
  chunk must be no larger than M_v / (D × ceil(T_fmt / (M_v / BW_NIC))).

* **Chunk large enough that polling overhead is tolerable.** `check_xfer_state` polls
  at 1ms intervals; a chunk that takes less than 5ms to transfer pays >20% polling tax.
  At 170 GB/s that floor is about 512MB (the design quotes 256MB for the 25%-threshold
  crossover, then recommends 512MB for headroom).

The caller can pass hardware constants directly (for tests and custom hardware) or omit
them to get the defaults calibrated against H20 in `tests/nixl/test_cost_model.py`.

There is no closed-form inversion for the ceil(), so the function searches across a
discrete candidate grid; the grid is the set of power-of-two multiples of 64MB from
64MB to 8GB, coarse enough to stay O(1) and fine enough to not miss a factor-2
improvement. The result is snapped to a multiple of the `alignment` argument so the
caller can require, say, 256MB alignment to match registered-region granularity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

MB = 1 << 20
GB = 1 << 30

#: Calibrated against H20 in test_cost_model.py.
_DEFAULT = {
    "bw_hbm_gbs": 4800.0,
    "bw_nic_gbs": 170.0,
    "t_py_us": 12.5,
    "t_lat_ms": 0.002,
    "polling_overhead_ms": 0.5,
    "polling_overhead_threshold": 0.25,
}


@dataclass
class ChunkSizeResult:
    chunk_bytes: int
    n_chunks: int
    t_fmt_ms: float
    t_rdma_per_chunk_ms: float
    pipeline_rounds: int
    polling_overhead_fraction: float
    reason: str


def t_fmt(n_keys: int, model_bytes: int, bw_hbm_gbs: float, t_py_us: float) -> float:
    """Format cost: Python dispatch + HBM read pass.

    Returns milliseconds. Units: bw_hbm_gbs in GB/s, model_bytes in bytes.
    """
    return n_keys * t_py_us / 1_000.0 + 2.0 * model_bytes / (bw_hbm_gbs * 1e9) * 1_000.0


def t_rdma(model_bytes: int, chunk_bytes: int, bw_nic_gbs: float, t_lat_ms: float) -> float:
    """RDMA transfer cost for a whole version. Returns milliseconds."""
    n_chunks = math.ceil(model_bytes / chunk_bytes)
    return model_bytes / (bw_nic_gbs * 1e9) * 1_000.0 + n_chunks * t_lat_ms


def t_rdma_per_chunk(chunk_bytes: int, bw_nic_gbs: float, t_lat_ms: float) -> float:
    """Time to transfer one chunk. Returns milliseconds."""
    return chunk_bytes / (bw_nic_gbs * 1e9) * 1_000.0 + t_lat_ms


def polling_overhead_fraction(chunk_bytes: int, bw_nic_gbs: float, polling_overhead_ms: float) -> float:
    """Fraction of per-chunk transfer time spent in polling backoff."""
    t_xfer = chunk_bytes / (bw_nic_gbs * 1e9) * 1_000.0
    if t_xfer <= 0:
        return float("inf")
    return polling_overhead_ms / t_xfer


def choose_chunk_bytes(
    model_bytes: int,
    n_grouped_keys: int,
    *,
    wire_buffer_depth: int = 2,
    alignment_bytes: int = 64 * MB,
    bw_hbm_gbs: float | None = None,
    bw_nic_gbs: float | None = None,
    t_py_us: float | None = None,
    t_lat_ms: float | None = None,
    polling_overhead_ms: float | None = None,
    polling_overhead_threshold: float | None = None,
) -> ChunkSizeResult:
    """Choose the optimal transfer chunk size for this model and hardware.

    Parameters
    ----------
    model_bytes:
        Total bytes to transfer per version (M_v in the design).
    n_grouped_keys:
        Key count *after* MoE grouping (§3.5.5). Using the ungrouped count gives a
        spuriously large T_fmt and an unnecessarily small chunk -- which is valid but
        suboptimal.
    wire_buffer_depth:
        Number of pipeline stages (D). Default 2.
    alignment_bytes:
        Output is snapped to a multiple of this.
    """
    hw = _DEFAULT.copy()
    for key, val in {
        "bw_hbm_gbs": bw_hbm_gbs,
        "bw_nic_gbs": bw_nic_gbs,
        "t_py_us": t_py_us,
        "t_lat_ms": t_lat_ms,
        "polling_overhead_ms": polling_overhead_ms,
        "polling_overhead_threshold": polling_overhead_threshold,
    }.items():
        if val is not None:
            hw[key] = val

    tf = t_fmt(n_grouped_keys, model_bytes, hw["bw_hbm_gbs"], hw["t_py_us"])
    t_rdma_total = t_rdma(model_bytes, max(model_bytes, 1), hw["bw_nic_gbs"], hw["t_lat_ms"])

    # Chunk ceiling: large enough that we need at least D rounds.
    # chunk_max = M_v / D (the round count must be at least D for the pipeline to fill)
    chunk_max = model_bytes // max(wire_buffer_depth, 1)

    # Chunk floor from polling overhead.
    # polling_overhead / (chunk_bytes / bw_nic) <= threshold
    # => chunk_bytes >= polling_overhead * bw_nic / threshold
    polling_floor_bytes = int(
        hw["polling_overhead_ms"] * hw["bw_nic_gbs"] * 1e3 / hw["polling_overhead_threshold"]
    )

    # Candidate grid: power-of-two multiples of 64MB up to 8GB.
    candidates = [64 * MB * (2 ** i) for i in range(8)]  # 64MB .. 8GB
    candidates = [
        c for c in candidates if polling_floor_bytes <= c <= max(chunk_max, polling_floor_bytes)
    ]
    if not candidates:
        candidates = [polling_floor_bytes]

    best = None
    best_score = float("inf")

    for candidate in candidates:
        aligned = max(
            ((candidate + alignment_bytes - 1) // alignment_bytes) * alignment_bytes,
            alignment_bytes,
        )
        aligned = max(aligned, polling_floor_bytes)

        pf = polling_overhead_fraction(aligned, hw["bw_nic_gbs"], hw["polling_overhead_ms"])
        if pf > hw["polling_overhead_threshold"]:
            continue

        t_chunk = t_rdma_per_chunk(aligned, hw["bw_nic_gbs"], hw["t_lat_ms"])
        rounds = math.ceil(model_bytes / aligned)

        # Minimise makespan, not the distance to a round count.
        #
        # The previous objective scored `abs(rounds - rounds_needed)` where
        # `rounds_needed = ceil(T_fmt / t_chunk)`. That is monotone in chunk size --
        # a bigger chunk raises `t_chunk`, so `rounds_needed` falls faster than `rounds`
        # does -- and the score therefore always bottomed out at the grid's largest
        # entry. Every model from 16GB to 1.3TB got the same 8GB chunk, which is the one
        # thing B3's own criterion says must not happen ("chunk *bytes* vary with the
        # model, not a fixed value").
        #
        # The quantity that actually matters is how long the pipeline takes. With two
        # stages (format, then RDMA) the steady state runs at the slower stage, and
        # `D - 1` extra rounds fill and drain it:
        #
        #     makespan = (rounds + D - 1) x max(t_chunk, T_fmt / rounds)
        #
        # Small chunks make `t_chunk` cheap but spread `T_fmt` thin over many rounds;
        # large chunks do the reverse. The minimum sits where the two stages balance,
        # which is exactly the pipeline-design intent -- and it moves with the model,
        # because `T_fmt` grows with key count and bytes while `t_chunk` does not.
        t_fmt_per_chunk = tf / max(rounds, 1)
        makespan = (rounds + max(wire_buffer_depth, 1) - 1) * max(t_chunk, t_fmt_per_chunk)

        # Polling tax is already bounded by the `continue` above; adding it here breaks
        # ties toward the cheaper-to-poll candidate without competing with makespan,
        # since `pf <= threshold < 1` and makespan is in milliseconds.
        score = makespan + pf

        if score < best_score:
            best_score = score
            best = aligned

    if best is None:
        # No candidate satisfies the polling threshold; use the floor anyway.
        best = max(
            ((polling_floor_bytes + alignment_bytes - 1) // alignment_bytes) * alignment_bytes,
            alignment_bytes,
        )

    final_t_chunk = t_rdma_per_chunk(best, hw["bw_nic_gbs"], hw["t_lat_ms"])
    final_rounds = math.ceil(model_bytes / best)
    final_pf = polling_overhead_fraction(best, hw["bw_nic_gbs"], hw["polling_overhead_ms"])

    reason_parts = [
        f"T_fmt={tf:.1f}ms",
        f"T_rdma_chunk={final_t_chunk:.3f}ms",
        f"rounds={final_rounds}",
        f"polling_overhead={final_pf:.1%}",
    ]
    reason = "; ".join(reason_parts)

    return ChunkSizeResult(
        chunk_bytes=best,
        n_chunks=final_rounds,
        t_fmt_ms=tf,
        t_rdma_per_chunk_ms=final_t_chunk,
        pipeline_rounds=final_rounds,
        polling_overhead_fraction=final_pf,
        reason=reason,
    )
