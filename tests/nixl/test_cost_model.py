"""Calibrate and pin the weight-transport cost model (pre-implementation verification 4).

Design doc 3.6.5 replaces the earlier "measure it later" placeholders with closed-form
estimates for T_fmt, T_rdma, T_cast, W and chunk size. Those formulas carry one fitted
constant -- `T_py_overhead`, the per-key Python/weight_loader dispatch cost -- derived
from a single published number (miles' 884ms for Kimi-K2). A constant fitted to one
data point is a liability, so these tests do two things:

1. Ground the `N_keys` term in real checkpoints from /jizhicfs/johnnyslin/models rather
   than an estimate. Qwen3-30B-A3B turns out to have 18,867 HF keys of which 18,432
   (97.7%) are expert keys -- which is the MoE-grouping premise measured, not assumed.
2. Measure per-key dispatch overhead locally and assert the fitted constant is in the
   right order of magnitude. If a future vLLM or Python version moves it, this fails
   and the formula gets re-fitted rather than silently mispredicting.

What these tests deliberately do NOT do is assert absolute millisecond predictions.
T_fmt depends on the inference engine's loader, which is not exercised here; the
formula's *shape* and its inputs are what can be checked on one machine.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pytest
import torch

from tests.nixl.conftest import MODEL_ROOT

# Hardware constants for this node, from NVIDIA specs and ibstat (design doc 3.6.5).
BW_HBM_GBS = 4800.0  # H20 HBM3
NIC_COUNT = 8
NIC_LINE_GBPS = 200.0  # RoCE, from ibstat
NIC_EFFICIENCY = 0.85  # large-message RDMA efficiency
BW_NIC_GBS = NIC_COUNT * NIC_LINE_GBPS * NIC_EFFICIENCY / 8

# The fitted constant under test.
T_PY_OVERHEAD_US = 12.5


def _hf_keys(model_dir: Path) -> list[str]:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        return list(json.loads(index.read_text())["weight_map"])
    from safetensors import safe_open

    with safe_open(str(model_dir / "model.safetensors"), "pt") as handle:
        return list(handle.keys())


def t_fmt_ms(n_keys: int, m_v_gb: float, t_py_us: float = T_PY_OVERHEAD_US) -> float:
    """Design doc 3.6.5. Two regimes: launch-bound at many keys, bandwidth-bound after
    grouping collapses them. See `test_t_fmt_has_two_regimes`."""
    return n_keys * t_py_us / 1e3 + 2 * m_v_gb / BW_HBM_GBS * 1e3


def t_rdma_ms(m_v_gb: float, chunk_gb: float, bw_gbs: float = BW_NIC_GBS) -> float:
    """Design doc 3.6.5: LogGP, latency term negligible at large chunk sizes."""
    rounds = math.ceil(m_v_gb / chunk_gb)
    return m_v_gb / bw_gbs * 1e3 + rounds * 2e-3


def t_cast_ms(m_v_gb: float, bytes_per_src: float = 3.008) -> float:
    """Design doc 3.6.5: Roofline, memory-bandwidth bound (BF16->FP8 default)."""
    return m_v_gb * bytes_per_src / BW_HBM_GBS * 1e3


class TestRealCheckpointKeyCounts:
    """Ground the N_keys term in checkpoints rather than estimates."""

    @pytest.mark.parametrize(
        "model,min_keys,moe",
        [
            ("Qwen2.5-0.5B-Instruct", 200, False),
            ("Moonlight-16B-A3B", 5000, True),
            ("Qwen3-30B-A3B-Instruct-2507", 18000, True),
        ],
    )
    def test_key_counts_are_as_the_model_predicts(self, model, min_keys, moe):
        model_dir = MODEL_ROOT / model
        if not model_dir.exists():
            pytest.skip(f"{model} not present")

        keys = _hf_keys(model_dir)
        assert len(keys) >= min_keys

        expert_keys = [k for k in keys if "expert" in k.lower()]
        if moe:
            fraction = len(expert_keys) / len(keys)
            assert fraction > 0.9, (
                f"{model}: expert keys are {fraction:.1%} of the total; MoE grouping "
                f"(design item C1) is premised on this dominating"
            )
        else:
            assert not expert_keys

    def test_moe_grouping_reduces_keys_by_two_orders_of_magnitude(self):
        """The C1 premise, measured: collapse per-expert keys into one per projection."""
        model_dir = MODEL_ROOT / "Qwen3-30B-A3B-Instruct-2507"
        if not model_dir.exists():
            pytest.skip("Qwen3-30B-A3B not present")

        keys = _hf_keys(model_dir)
        # Grouping replaces experts.{i}.{proj} with experts.{proj}, so the grouped count
        # is the number of distinct (layer, projection) pairs plus the non-expert keys.
        grouped = set()
        for key in keys:
            if ".experts." in key:
                prefix, _, suffix = key.partition(".experts.")
                _, _, projection = suffix.partition(".")
                grouped.add(f"{prefix}.experts.{projection}")
            else:
                grouped.add(key)

        reduction = len(keys) / len(grouped)
        # 128 experts give ~32x here; Kimi-K2's 384 experts would give ~384x. The
        # compression scales with expert count, so the floor is deliberately low.
        assert reduction > 25, f"expected >25x key reduction, got {reduction:.1f}x"
        assert len(grouped) < 1000, f"grouped key count {len(grouped)} should be in the hundreds"

    def test_t_fmt_prediction_flips_the_bottleneck_after_grouping(self):
        """Design doc 3.6.0's two-phase conclusion, evaluated on a real key count."""
        model_dir = MODEL_ROOT / "Qwen3-30B-A3B-Instruct-2507"
        if not model_dir.exists():
            pytest.skip("Qwen3-30B-A3B not present")

        keys = _hf_keys(model_dir)
        grouped_keys = 579  # measured for this checkpoint; see the reduction test
        m_v_gb = 30.0 * 0.5 / 8  # 30B at fp8 over 8 nodes

        ungrouped = t_fmt_ms(len(keys), m_v_gb)
        grouped = t_fmt_ms(grouped_keys, m_v_gb)
        rdma = t_rdma_ms(m_v_gb, 0.512)

        assert ungrouped > rdma, "before grouping, formatting must dominate"
        assert grouped < rdma, "after grouping, RDMA must dominate"


class TestPerKeyDispatchOverhead:
    """Keep the fitted T_py_overhead honest against a locally measured floor."""

    def test_measured_dispatch_cost_is_within_an_order_of_magnitude(self):
        """Measure a minimal per-key loader step: dtype check, cast, reshape, copy.

        This is a floor, not vLLM's real `weight_loader` -- that additionally resolves
        name->param, computes shard dims, narrows, and for MoE parses expert ids. So
        the assertion is a range whose purpose is to catch the fitted constant being
        wrong by an order of magnitude, not to re-derive it.

        Warm-up matters more than it looks: the same loop measures ~17us/key over 2k
        iterations and ~6us/key over 10k, because allocator and dispatch caches are
        still filling early on. Measuring the cold phase would make the floor appear
        to exceed the fitted constant.
        """
        n_keys = 10_000
        warmup = 2_000
        tensors = [torch.zeros(256, dtype=torch.float32) for _ in range(n_keys)]
        destinations = [torch.zeros(256, dtype=torch.bfloat16) for _ in range(n_keys)]

        def run(pairs):
            for source, destination in pairs:
                staged = source.to(destination.dtype) if source.dtype != destination.dtype else source
                destination.view(-1).copy_(staged.reshape(-1))

        run(list(zip(tensors[:warmup], destinations[:warmup])))

        measured = list(zip(tensors[warmup:], destinations[warmup:]))
        start = time.perf_counter()
        run(measured)
        per_key_us = (time.perf_counter() - start) / len(measured) * 1e6

        assert 0.5 < per_key_us < T_PY_OVERHEAD_US, (
            f"measured {per_key_us:.1f}us/key for a minimal loader step; the fitted "
            f"{T_PY_OVERHEAD_US}us/key should exceed it (it covers vLLM's full "
            f"weight_loader) but stay within an order of magnitude"
        )

    def test_fitted_constant_reproduces_the_published_measurement(self):
        """miles measured 884ms for Kimi-K2's post_load_weights.

        The key count must come from K2's architecture, not an estimate:
        384 experts x 61 layers x 3 projections = 70,272 HF keys, which is the figure
        design doc 3.5.5 derives. An earlier version of the cost model fitted against
        14,000 keys -- a tenth of 140k, a transcription slip -- and so carried a
        T_py_overhead 5x too large.
        """
        k2_keys = 384 * 61 * 3
        assert k2_keys == 70_272

        predicted = t_fmt_ms(n_keys=k2_keys, m_v_gb=15.6)
        assert 800 < predicted < 970, (
            f"T_fmt({k2_keys} keys) predicts {predicted:.0f}ms; miles measured 884ms. "
            f"A large divergence means T_py_overhead needs re-fitting."
        )


class TestCostModelShape:
    """The formulas' structural properties, which must hold on any hardware."""

    def test_t_fmt_is_linear_in_key_count(self):
        a, b = t_fmt_ms(1000, 10.0), t_fmt_ms(2000, 10.0)
        launch_only = lambda n: n * T_PY_OVERHEAD_US / 1e3
        assert math.isclose(b - a, launch_only(1000), rel_tol=1e-6)

    def test_t_fmt_has_two_regimes(self):
        """Launch dominates before grouping; bandwidth dominates after.

        The design doc originally called T_fmt simply "launch-dominated". That holds
        only at large key counts: at a few hundred keys the per-key term falls below
        the read+write bandwidth term, so the formula changes character. Treating it as
        launch-bound everywhere would predict grouping gains that do not exist.
        """
        bandwidth = 2 * 15.6 / BW_HBM_GBS * 1e3

        ungrouped_launch = 70_272 * T_PY_OVERHEAD_US / 1e3
        assert ungrouped_launch > bandwidth * 50, "before grouping: launch regime"

        grouped_launch = 183 * T_PY_OVERHEAD_US / 1e3
        assert grouped_launch < bandwidth, "after grouping: bandwidth regime"

    def test_t_rdma_bandwidth_term_is_inversely_proportional(self):
        rounds = math.ceil(16.0 / 0.512)
        latency = rounds * 2e-3
        slow = t_rdma_ms(16.0, 0.512, bw_gbs=170.0) - latency
        fast = t_rdma_ms(16.0, 0.512, bw_gbs=340.0) - latency
        assert math.isclose(slow, 2 * fast, rel_tol=1e-9)

    def test_t_rdma_latency_term_is_negligible_at_512mb(self):
        m_v, chunk = 15.6, 0.512
        bandwidth_term = m_v / BW_NIC_GBS * 1e3
        latency_term = math.ceil(m_v / chunk) * 2e-3
        assert latency_term < bandwidth_term / 1000

    def test_cast_is_always_faster_than_rdma_on_this_node(self):
        """Why constraint (5) on W never binds: cast outruns the NIC by ~10x."""
        for m_v in (1.0, 15.6, 62.5):
            assert t_cast_ms(m_v) < t_rdma_ms(m_v, 0.512) / 5

    @pytest.mark.parametrize("nodes,expected_dominant", [(8, "rdma"), (32, "rdma"), (128, "rdma")])
    def test_rdma_dominates_at_every_scale_after_grouping(self, nodes, expected_dominant):
        """Scale sweep from design doc 3.6.5's table."""
        m_v = 500.0 / nodes
        fmt = t_fmt_ms(200, m_v)
        rdma = t_rdma_ms(m_v, 0.512)
        dominant = "fmt" if fmt > rdma else "rdma"
        assert dominant == expected_dominant


class TestChunkSizeRule:
    """Design doc 3.5.6: bounds come from pipeline depth and completion granularity."""

    def test_chunk_max_uses_per_node_bytes_not_total(self):
        """The corrected rule: the denominator is M_v, not M."""
        m_total, nodes = 500.0, 32
        m_v = m_total / nodes
        depth = 2
        ratio = t_fmt_ms(200, m_v) / t_rdma_ms(m_v, 0.512)
        n_min = math.ceil(max(ratio, 1) * depth)
        chunk_max_gb = m_v / n_min

        assert chunk_max_gb > 1.0, "with M_v the bound is loose; with M it would be 30x tighter"
        assert 0.512 < chunk_max_gb

    def test_polling_floor_makes_sub_256mb_chunks_uneconomic(self):
        """Design doc 3.6.4: the 1ms backoff bounds how small a chunk can be."""
        poll_overhead_ms = 0.5  # expected value of a 0-1ms backoff
        measured = {}
        for chunk_gb in (1.0, 0.512, 0.256, 0.128, 0.032):
            rdma_ms = chunk_gb / BW_NIC_GBS * 1e3
            measured[chunk_gb] = poll_overhead_ms / rdma_ms

        # On a 170 GB/s node the design's 512MB default already pays 17%, and 256MB
        # pays 33% -- so the doc's "~256MB floor" is optimistic by one step. The
        # notification-based completion path (item 5b) is what removes this.
        assert measured[1.0] < 0.10
        assert 0.15 < measured[0.512] < 0.20
        assert measured[0.256] > 0.30, "256MB is already past a 25% overhead budget"
        assert measured[0.032] > 1.0, "32MB spends more time polling than transferring"


class TestHardwareConstants:
    """Pin the node's constants so a different machine fails loudly."""

    def test_nic_bandwidth_matches_ibstat(self):
        assert math.isclose(BW_NIC_GBS, 170.0, rel_tol=0.01)

    def test_gpu_is_h20(self):
        if not torch.cuda.is_available():
            pytest.skip("no CUDA")
        name = torch.cuda.get_device_name(0)
        assert "H20" in name, f"cost model constants are for H20, found {name}"

    def test_hbm_bandwidth_is_plausible_for_the_device(self):
        if not torch.cuda.is_available():
            pytest.skip("no CUDA")
        size = 256 * 1024 * 1024 // 4
        source = torch.zeros(size, dtype=torch.float32, device="cuda")
        destination = torch.empty_like(source)

        for _ in range(3):
            destination.copy_(source)
        torch.cuda.synchronize()

        start = time.perf_counter()
        iterations = 10
        for _ in range(iterations):
            destination.copy_(source)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        moved_gb = 2 * source.numel() * 4 / 1e9 * iterations
        achieved = moved_gb / elapsed
        assert achieved > BW_HBM_GBS * 0.25, (
            f"device-to-device copy achieved {achieved:.0f} GB/s against a "
            f"{BW_HBM_GBS:.0f} GB/s spec; the cast model would overpredict"
        )
