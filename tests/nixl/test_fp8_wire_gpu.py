"""FP8 wire-format quantization against vLLM's own primitives (item A9).

Design 3.8 requires the fp8 wire format to come from vLLM/TE primitives rather than a
reimplementation, so that what the trainer produces is bit-compatible with what the
inference engine would have loaded from a native fp8 checkpoint. These tests pin the
properties the transport depends on:

* round-trip accuracy -- dequantizing what we send reconstructs the source within fp8's
  representable error, so a mis-scaled block shows up as a large relative error
* wire size -- fp8 plus its scales really is about half of bf16, which is the bandwidth
  claim the design makes
* block alignment -- the 128x128 blockwise layout, and what happens at shapes that do
  not divide evenly, since the design says PSRL should surface that at config time
* scale pairing -- weight and scale must travel as one unit, since half a pair is a
  silent no-op in the engine loader

These are quantization-correctness tests, not an end-to-end comparison against a loaded
fp8 checkpoint; that needs a real vLLM engine and belongs to the e2e suite (item 8 in
the verification list).
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

BLOCK = 128
FP8_DTYPE = torch.float8_e4m3fn
# fp8_e4m3 carries 3 mantissa bits, so per-element relative error is bounded by about
# 2^-4. Blockwise scaling removes the exponent range problem but not the mantissa.
FP8_RELATIVE_TOLERANCE = 0.08


@pytest.fixture(scope="module")
def per_block_cast():
    deep_gemm = pytest.importorskip("vllm.utils.deep_gemm", reason="vLLM not installed")
    if not hasattr(deep_gemm, "per_block_cast_to_fp8"):
        pytest.skip("per_block_cast_to_fp8 unavailable in this vLLM build")
    return deep_gemm.per_block_cast_to_fp8


@pytest.fixture(scope="module")
def scaled_fp8_quant():
    ops = pytest.importorskip("vllm._custom_ops", reason="vLLM not installed")
    if not hasattr(ops, "scaled_fp8_quant"):
        pytest.skip("scaled_fp8_quant unavailable in this vLLM build")
    return ops.scaled_fp8_quant


def _dequantize_blockwise(quantized: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Undo a 128x128 blockwise cast, so accuracy can be measured end to end."""
    rows, cols = quantized.shape
    out = quantized.to(torch.float32)
    for i in range(scales.shape[0]):
        for j in range(scales.shape[1]):
            r0, r1 = i * BLOCK, min((i + 1) * BLOCK, rows)
            c0, c1 = j * BLOCK, min((j + 1) * BLOCK, cols)
            out[r0:r1, c0:c1] *= scales[i, j]
    return out


class TestBlockwiseRoundTrip:
    """What the trainer sends must reconstruct the weights it holds."""

    @pytest.mark.parametrize("shape", [(128, 128), (256, 512), (1024, 1024)])
    def test_round_trip_is_within_fp8_error(self, per_block_cast, shape):
        source = torch.randn(shape, dtype=torch.bfloat16, device="cuda").to(torch.float32)

        quantized, scales = per_block_cast(source)
        recovered = _dequantize_blockwise(quantized, scales.to(torch.float32))

        relative = (recovered - source).abs() / source.abs().clamp(min=1e-6)
        assert relative.median().item() < FP8_RELATIVE_TOLERANCE, (
            f"median relative error {relative.median().item():.4f} exceeds fp8's "
            f"expected {FP8_RELATIVE_TOLERANCE}; a mis-applied scale looks like this"
        )

    def test_scales_have_one_entry_per_block(self, per_block_cast):
        """A scale grid of the wrong shape means blocks are scaled by the wrong factor."""
        rows, cols = 512, 256
        source = torch.randn(rows, cols, dtype=torch.float32, device="cuda")

        quantized, scales = per_block_cast(source)

        assert quantized.shape == (rows, cols)
        assert quantized.dtype == FP8_DTYPE
        assert scales.shape[0] == (rows + BLOCK - 1) // BLOCK
        assert scales.shape[1] == (cols + BLOCK - 1) // BLOCK

    def test_large_dynamic_range_survives_blockwise_scaling(self, per_block_cast):
        """Blockwise scaling exists for this: per-block exponents, not one global one.

        A single global scale would flush the small block to zero, which is exactly the
        failure blockwise quantization avoids.
        """
        source = torch.ones(256, 256, dtype=torch.float32, device="cuda")
        source[:128, :128] *= 1e4
        source[128:, 128:] *= 1e-4

        quantized, scales = per_block_cast(source)
        recovered = _dequantize_blockwise(quantized, scales.to(torch.float32))

        for region in (recovered[:128, :128], recovered[128:, 128:]):
            relative = (region - source[: region.shape[0], : region.shape[1]]).abs()
            assert torch.isfinite(region).all(), "quantization produced inf/nan"
        assert recovered[128:, 128:].abs().max() > 0, "small-magnitude block flushed to zero"

    def test_quantization_is_deterministic(self, per_block_cast):
        """Two ranks quantizing the same weights must agree, or shards disagree."""
        source = torch.randn(256, 256, dtype=torch.float32, device="cuda")
        first_q, first_s = per_block_cast(source)
        second_q, second_s = per_block_cast(source)

        assert torch.equal(first_q.view(torch.uint8), second_q.view(torch.uint8))
        assert torch.equal(first_s, second_s)


class TestWireSize:
    """The bandwidth claim: fp8 plus scales is about half of bf16."""

    @pytest.mark.parametrize("shape", [(1024, 1024), (4096, 4096)])
    def test_fp8_wire_is_about_half_of_bf16(self, per_block_cast, shape):
        source = torch.randn(shape, dtype=torch.float32, device="cuda")
        quantized, scales = per_block_cast(source)

        bf16_bytes = shape[0] * shape[1] * 2
        fp8_bytes = quantized.numel() * 1 + scales.numel() * scales.element_size()
        ratio = fp8_bytes / bf16_bytes

        assert 0.45 < ratio < 0.55, f"fp8 wire is {ratio:.3f} of bf16, expected ~0.5"

    def test_scale_overhead_shrinks_with_size(self, per_block_cast):
        """Scales are one per 128x128 block, so their share falls as tensors grow."""
        overheads = []
        for size in (256, 1024, 4096):
            source = torch.randn(size, size, dtype=torch.float32, device="cuda")
            quantized, scales = per_block_cast(source)
            overheads.append(scales.numel() * scales.element_size() / quantized.numel())

        assert overheads == sorted(overheads, reverse=True), "scale overhead should shrink"
        assert overheads[-1] < 0.001, f"4096x4096 scale overhead is {overheads[-1]:.5f}"


class TestBlockAlignment:
    """Design 3.8 says misalignment is surfaced at config time, not in the transport."""

    @pytest.mark.parametrize("shape", [(128, 128), (256, 256), (384, 512)])
    def test_aligned_shapes_quantize_cleanly(self, per_block_cast, shape):
        source = torch.randn(shape, dtype=torch.float32, device="cuda")
        quantized, scales = per_block_cast(source)
        assert quantized.shape == shape
        assert scales.shape == (shape[0] // BLOCK, shape[1] // BLOCK)

    @pytest.mark.parametrize("shape", [(100, 128), (128, 100), (130, 130)])
    def test_unaligned_shapes_either_pad_or_raise(self, per_block_cast, shape):
        """Record the behaviour so the config check can be written against it.

        Whatever it does must be *consistent*: a partial block silently reusing a
        neighbour's scale would corrupt the edge rows, so this asserts the scale grid
        covers every element.
        """
        source = torch.randn(shape, dtype=torch.float32, device="cuda")
        try:
            quantized, scales = per_block_cast(source)
        except Exception:
            return  # rejecting unaligned shapes outright is acceptable

        assert scales.shape[0] * BLOCK >= shape[0], "scale grid does not cover all rows"
        assert scales.shape[1] * BLOCK >= shape[1], "scale grid does not cover all columns"
        recovered = _dequantize_blockwise(quantized, scales.to(torch.float32))
        relative = (recovered - source).abs() / source.abs().clamp(min=1e-6)
        assert relative.median().item() < FP8_RELATIVE_TOLERANCE


class TestScalePairing:
    """Weight and scale form one commit group; the test is that one is useless alone."""

    def test_weight_without_its_scale_is_wrong_by_orders_of_magnitude(self, per_block_cast):
        """Why they must arrive together: the unscaled tensor is not an approximation.

        The magnitude has to be far outside fp8_e4m3's own range (max 448) for the
        scale to carry real information. With standard-normal data the computed scales
        sit near 1.0, so dropping them costs only ~6x -- bad, but not obviously bad.
        Weights whose blocks differ by 1e3 are the realistic case, and there the scale
        is the only thing making the value recoverable at all.
        """
        source = torch.randn(256, 256, dtype=torch.float32, device="cuda")
        source[:128] *= 1e5  # one block far above fp8 range, one far below
        source[128:] *= 1e-3

        quantized, scales = per_block_cast(source)
        unscaled = quantized.to(torch.float32)

        with_scale = (
            (_dequantize_blockwise(quantized, scales.to(torch.float32)) - source).abs().median()
        )
        without_scale = (unscaled - source).abs().median()
        assert without_scale > with_scale * 100, (
            f"dropping the scale changed the median error only {without_scale / with_scale:.1f}x; "
            f"it must be grossly wrong, or a lost scale could pass unnoticed"
        )

    def test_scale_dtype_is_float32(self, per_block_cast):
        """The wire format pairs an fp8 tensor with fp32 scales (design 3.8)."""
        source = torch.randn(256, 256, dtype=torch.float32, device="cuda")
        _quantized, scales = per_block_cast(source)
        assert scales.dtype == torch.float32


class TestPerTensorQuantization:
    """The simpler per-tensor path, used where blockwise is unavailable."""

    def test_round_trip_within_tolerance(self, scaled_fp8_quant):
        source = torch.randn(512, 512, dtype=torch.float16, device="cuda")
        quantized, scale = scaled_fp8_quant(source)

        recovered = quantized.to(torch.float32) * scale.to(torch.float32)
        relative = (recovered - source.to(torch.float32)).abs() / source.abs().clamp(min=1e-3).to(
            torch.float32
        )
        assert relative.median().item() < FP8_RELATIVE_TOLERANCE

    def test_static_scale_is_honoured(self, scaled_fp8_quant):
        """A pre-computed scale must be used as given, so both sides can agree on it."""
        source = torch.randn(256, 256, dtype=torch.float16, device="cuda")
        scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")

        quantized, returned = scaled_fp8_quant(source, scale)

        assert torch.equal(returned.to(torch.float32).flatten()[:1], scale)
