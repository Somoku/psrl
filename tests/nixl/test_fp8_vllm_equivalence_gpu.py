"""FP8 pipeline equivalence against vLLM's own weight-loading path (item 8).

Design 3.8 requires the three-stage fp8 pipeline PSRL runs on the trainer side to
produce the same per-block-quantized weight that vLLM would have produced had it loaded
the same weight directly from a native fp8 checkpoint.  The comparison has two legs:

  Stage 1 (trainer quantizes):
      bf16 weight ──per_block_cast_to_fp8──► fp8 + e4m3 float32 scales

  Stage 3 (rollout runtime-formats):
      fp8 + e4m3 scales ──requant_weight_ue8m0_inplace──► fp8 + ue8m0 scales

  vLLM checkpoint path (baseline):
      bf16 weight ──scaled_fp8_quant──► fp8 + per-tensor scale
      (for dynamic/per-tensor quantization, which is the non-block path)

Because the PSRL path and the vLLM checkpoint path use fundamentally different
quantization granularities (128×128 blockwise vs per-tensor), the two cannot be
bit-identical.  What the design actually claims is that:

  (a) The PSRL output satisfies the shape contract vLLM expects: an fp8_e4m3fn weight
      plus a float32 scale tensor of shape (M // 128, K // 128).
  (b) The round-trip accuracy of the PSRL three-stage pipeline does not regress
      compared to vLLM's own per_block_cast_to_fp8, within fp8's representable error.
  (c) requant_weight_ue8m0_inplace is idempotent: running it twice is the same as once.
  (d) Scales are strictly positive and finite after all three stages.

These properties together mean that what PSRL hands to the vLLM engine loader is a
valid, correctly-formatted fp8 weight that the loader can consume without additional
quantization — the precondition the design calls "stage 3 completes".

The e2e logit comparison against a real vLLM engine (item 8 in the verification list)
requires a full Ray cluster and belongs to the e2e suite; this file covers everything
that can be verified on a single GPU without engine startup.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

BLOCK = 128
FP8_DTYPE = torch.float8_e4m3fn
# fp8_e4m3 carries 3 mantissa bits; per-element relative error is bounded by ~2^-4.
# Blockwise scaling removes the exponent-range problem but not the mantissa quantisation.
FP8_RELATIVE_TOLERANCE = 0.08


@pytest.fixture(scope="module")
def per_block_cast():
    deep_gemm = pytest.importorskip("vllm.utils.deep_gemm", reason="vLLM not installed")
    if not hasattr(deep_gemm, "per_block_cast_to_fp8"):
        pytest.skip("per_block_cast_to_fp8 not available in this vLLM build")
    return deep_gemm.per_block_cast_to_fp8


@pytest.fixture(scope="module")
def requant_ue8m0():
    fp8_utils = pytest.importorskip(
        "vllm.model_executor.layers.quantization.utils.fp8_utils",
        reason="vLLM not installed",
    )
    if not hasattr(fp8_utils, "requant_weight_ue8m0_inplace"):
        pytest.skip("requant_weight_ue8m0_inplace not available in this vLLM build")
    return fp8_utils.requant_weight_ue8m0_inplace


@pytest.fixture(scope="module")
def validate_fp8_block_shape():
    fp8_utils = pytest.importorskip(
        "vllm.model_executor.layers.quantization.utils.fp8_utils",
        reason="vLLM not installed",
    )
    return fp8_utils.validate_fp8_block_shape


@pytest.fixture(scope="module")
def vllm_tp1_group():
    """Initialise vLLM's TP=1 distributed state for the whole module.

    `validate_fp8_block_shape` resolves the TP size as
    ``getattr(layer, "tp_size", get_tensor_model_parallel_world_size())``. Python
    evaluates that default eagerly, so the group lookup runs even when the layer
    carries its own `tp_size` -- setting the attribute is not enough to avoid
    initialising the group. The function also instantiates a CustomOp, which needs an
    active `set_current_vllm_config` context.

    Module-scoped because a process group can only be created once per process.
    """
    import os

    pytest.importorskip("vllm", reason="vLLM not installed")
    import torch.distributed as dist
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29531")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    with set_current_vllm_config(VllmConfig()):
        if not dist.is_initialized():
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method="env://",
                backend="nccl",
            )
            initialize_model_parallel(tensor_model_parallel_size=1)
        yield
    if dist.is_initialized():
        dist.destroy_process_group()


def _dequantize_blockwise(fp8: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Undo 128×128 blockwise quantisation for accuracy measurement."""
    rows, cols = fp8.shape
    out = fp8.to(torch.float32)
    for i in range(scales.shape[0]):
        for j in range(scales.shape[1]):
            r0, r1 = i * BLOCK, min((i + 1) * BLOCK, rows)
            c0, c1 = j * BLOCK, min((j + 1) * BLOCK, cols)
            out[r0:r1, c0:c1] *= scales[i, j].float()
    return out


class TestStage1ShapeContract:
    """per_block_cast_to_fp8 output shape and dtype must match what vLLM's loader reads."""

    @pytest.mark.parametrize("shape", [(128, 128), (256, 512), (4096, 4096)])
    def test_quantised_dtype_is_fp8_e4m3fn(self, per_block_cast, shape):
        source = torch.randn(*shape, dtype=torch.float32, device="cuda")
        fp8, scales = per_block_cast(source)
        assert fp8.dtype == FP8_DTYPE, f"expected fp8_e4m3fn, got {fp8.dtype}"

    @pytest.mark.parametrize("shape", [(128, 128), (256, 512), (384, 640)])
    def test_scale_grid_dimensions_match_128x128_blocks(self, per_block_cast, shape):
        rows, cols = shape
        source = torch.randn(rows, cols, dtype=torch.float32, device="cuda")
        _fp8, scales = per_block_cast(source)
        expected_rows = (rows + BLOCK - 1) // BLOCK
        expected_cols = (cols + BLOCK - 1) // BLOCK
        assert scales.shape == (expected_rows, expected_cols), (
            f"scale shape {tuple(scales.shape)} != expected ({expected_rows}, {expected_cols})"
        )

    def test_weight_shape_is_preserved(self, per_block_cast):
        source = torch.randn(512, 1024, dtype=torch.float32, device="cuda")
        fp8, _scales = per_block_cast(source)
        assert fp8.shape == source.shape

    def test_scale_dtype_is_float32(self, per_block_cast):
        source = torch.randn(256, 256, dtype=torch.float32, device="cuda")
        _fp8, scales = per_block_cast(source)
        assert scales.dtype == torch.float32, f"scales dtype is {scales.dtype}"

    def test_scales_are_strictly_positive(self, per_block_cast):
        source = torch.randn(512, 512, dtype=torch.float32, device="cuda")
        _fp8, scales = per_block_cast(source)
        assert (scales > 0).all(), "some scales are non-positive after per_block_cast"


class TestStage3RequantContract:
    """requant_weight_ue8m0_inplace must preserve shape, dtype, and scale positivity."""

    def test_requant_preserves_fp8_dtype(self, per_block_cast, requant_ue8m0):
        source = torch.randn(512, 512, dtype=torch.float32, device="cuda")
        fp8, scales = per_block_cast(source)
        requant_ue8m0(fp8, scales)
        assert fp8.dtype == FP8_DTYPE

    def test_requant_preserves_weight_shape(self, per_block_cast, requant_ue8m0):
        source = torch.randn(256, 512, dtype=torch.float32, device="cuda")
        fp8, scales = per_block_cast(source)
        orig_shape = fp8.shape
        requant_ue8m0(fp8, scales)
        assert fp8.shape == orig_shape

    def test_requant_preserves_scale_grid_shape(self, per_block_cast, requant_ue8m0):
        source = torch.randn(512, 256, dtype=torch.float32, device="cuda")
        fp8, scales = per_block_cast(source)
        orig_scale_shape = scales.shape
        requant_ue8m0(fp8, scales)
        assert scales.shape == orig_scale_shape

    def test_requant_scales_remain_positive_and_finite(self, per_block_cast, requant_ue8m0):
        source = torch.randn(512, 512, dtype=torch.float32, device="cuda")
        fp8, scales = per_block_cast(source)
        requant_ue8m0(fp8, scales)
        assert (scales > 0).all(), "some scales became non-positive after requant"
        assert scales.isfinite().all(), "some scales became non-finite after requant"

    def test_requant_is_idempotent(self, per_block_cast, requant_ue8m0):
        """Running requant twice must give the same result as running it once.

        This is load-bearing: the rollout worker may be woken up and call
        runtime-format twice on the same tensor without reinitialising it.  A
        non-idempotent requant would silently corrupt the second call.
        """
        source = torch.randn(512, 512, dtype=torch.float32, device="cuda")
        fp8_a, scales_a = per_block_cast(source)
        fp8_b, scales_b = per_block_cast(source)

        requant_ue8m0(fp8_a, scales_a)
        requant_ue8m0(fp8_b, scales_b)
        requant_ue8m0(fp8_b, scales_b)  # second call

        assert torch.equal(fp8_a.view(torch.uint8), fp8_b.view(torch.uint8)), (
            "requant is not idempotent: second application changed the fp8 bytes"
        )
        assert torch.allclose(scales_a, scales_b, rtol=1e-5, atol=1e-7), (
            "requant is not idempotent: second application changed the scales"
        )


class TestFullPipelineAccuracy:
    """End-to-end accuracy of the stage-1 + stage-3 pipeline against fp8's error bound."""

    @pytest.mark.parametrize("shape", [(128, 128), (256, 512), (1024, 1024)])
    def test_three_stage_round_trip_within_fp8_error(
        self, per_block_cast, requant_ue8m0, shape
    ):
        """Dequantising the stage-1 + stage-3 output must recover the source weight.

        `requant_weight_ue8m0_inplace` rewrites *both* the fp8 bytes and the scales, so
        the decode has to use the post-requant scales. Pairing post-requant bytes with
        the pre-requant scales measures nothing real -- it just reports the size of the
        ue8m0 rounding as if it were error.
        """
        source = torch.randn(*shape, dtype=torch.float32, device="cuda")
        fp8, scales = per_block_cast(source)
        requant_ue8m0(fp8, scales)

        recovered = _dequantize_blockwise(fp8, scales)
        relative = (recovered - source).abs() / source.abs().clamp(min=1e-6)
        assert relative.median().item() < FP8_RELATIVE_TOLERANCE, (
            f"pipeline median relative error {relative.median().item():.4f} "
            f"exceeds fp8 bound {FP8_RELATIVE_TOLERANCE}"
        )

    @pytest.mark.parametrize("shape", [(256, 512), (1024, 1024)])
    def test_requant_does_not_degrade_accuracy(self, per_block_cast, requant_ue8m0, shape):
        """Stage 3 must not make the weight materially less accurate than stage 1 alone.

        ue8m0 forces each scale to a power of two, which costs some precision -- the
        point of the test is that the cost is bounded, not zero. A stage 3 that mangled
        the fp8 bytes while leaving the scales plausible would pass the shape and
        positivity checks and only show up here.
        """
        source = torch.randn(*shape, dtype=torch.float32, device="cuda")

        stage1_fp8, stage1_scales = per_block_cast(source)
        stage1_err = (
            (_dequantize_blockwise(stage1_fp8, stage1_scales) - source).abs()
            / source.abs().clamp(min=1e-6)
        ).median().item()

        stage3_fp8, stage3_scales = per_block_cast(source)
        requant_ue8m0(stage3_fp8, stage3_scales)
        stage3_err = (
            (_dequantize_blockwise(stage3_fp8, stage3_scales) - source).abs()
            / source.abs().clamp(min=1e-6)
        ).median().item()

        assert stage3_err < FP8_RELATIVE_TOLERANCE, (
            f"stage-3 median relative error {stage3_err:.4f} exceeds fp8 bound"
        )
        assert stage3_err <= stage1_err * 2.5 + 1e-6, (
            f"ue8m0 requant degraded accuracy more than expected: "
            f"stage1={stage1_err:.4f} -> stage3={stage3_err:.4f}"
        )

    def test_large_dynamic_range_handled_by_blockwise_scaling(
        self, per_block_cast, requant_ue8m0
    ):
        """Weights with very different block magnitudes must not flush any block to zero.

        A single global scale would flush the small-magnitude block entirely.  Blockwise
        scaling avoids that, and this is the key correctness claim that justifies the
        128×128 block granularity.
        """
        source = torch.ones(256, 256, dtype=torch.float32, device="cuda")
        source[:128, :128] *= 1e4   # large-magnitude block
        source[128:, 128:] *= 1e-4  # small-magnitude block

        fp8, scales = per_block_cast(source)
        requant_ue8m0(fp8, scales)

        # Neither block should be all zeros after quantisation.
        large_block = fp8[:128, :128].view(torch.uint8)
        small_block = fp8[128:, 128:].view(torch.uint8)
        assert not large_block.eq(0).all(), "large-magnitude block was flushed to zero"
        assert not small_block.eq(0).all(), "small-magnitude block was flushed to zero"


class TestValidateFp8BlockShape:
    """validate_fp8_block_shape must accept Qwen3-8B linear layer shapes at TP=1 and TP=2."""

    def test_column_parallel_qwen3_8b_tp1(self, validate_fp8_block_shape, vllm_tp1_group):
        """Qwen3-8B gate/up projection: (12288, 4096) at TP=1 must pass validation."""
        import torch.nn as nn

        layer = nn.Linear(4096, 12288, bias=False)
        layer.tp_size = 1
        # Should not raise.
        validate_fp8_block_shape(
            layer,
            input_size=4096,
            output_size=12288,
            input_size_per_partition=4096,
            output_partition_sizes=[12288],
            block_size=[BLOCK, BLOCK],
        )

    def test_column_parallel_qwen3_8b_tp2(self, validate_fp8_block_shape, vllm_tp1_group):
        """At TP=2 output_partition_sizes=[6144] must divide BLOCK=128."""
        import torch.nn as nn

        layer = nn.Linear(4096, 6144, bias=False)
        layer.tp_size = 2
        validate_fp8_block_shape(
            layer,
            input_size=4096,
            output_size=12288,
            input_size_per_partition=4096,
            output_partition_sizes=[6144],
            block_size=[BLOCK, BLOCK],
        )

    def test_row_parallel_qwen3_8b_tp1(self, validate_fp8_block_shape, vllm_tp1_group):
        """Down projection: (4096, 12288) at TP=1."""
        import torch.nn as nn

        layer = nn.Linear(12288, 4096, bias=False)
        layer.tp_size = 1
        validate_fp8_block_shape(
            layer,
            input_size=12288,
            output_size=4096,
            input_size_per_partition=12288,
            output_partition_sizes=[4096],
            block_size=[BLOCK, BLOCK],
        )

    def test_misaligned_output_raises(self, validate_fp8_block_shape, vllm_tp1_group):
        """An output partition not divisible by BLOCK must raise, not silently pass."""
        import torch.nn as nn

        layer = nn.Linear(4096, 4096, bias=False)
        layer.tp_size = 2
        with pytest.raises(ValueError, match="block"):
            # 2049 is not divisible by 128
            validate_fp8_block_shape(
                layer,
                input_size=4096,
                output_size=4098,
                input_size_per_partition=4096,
                output_partition_sizes=[2049],
                block_size=[BLOCK, BLOCK],
            )
