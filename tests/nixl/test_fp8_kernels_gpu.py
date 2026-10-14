"""The fused publish quantiser against the reference, bit for bit, and against the memory roofline."""

import time

import pytest
import torch
from psrl.utils.nixl import fp8_kernels
from psrl.utils.nixl.block_fp8 import quantize_blockwise

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel needs CUDA")

TILE = (128, 128)


def _reference(x):
    return quantize_blockwise(x.contiguous(), (1,) * (x.dim() - 2) + TILE)


def _input(shape, dist):
    x = torch.randn(shape, device="cuda")
    if dist == "wide":
        x = x * torch.logspace(-8, 8, shape[-1], device="cuda")
    elif dist == "tiny":
        x = x * 1e-7
    elif dist == "zeros":
        x = torch.zeros(shape, device="cuda")
    return x.to(torch.bfloat16)


@pytest.mark.parametrize("shape", [(128, 128), (1536, 4096), (4096, 1536), (8, 256, 384), (512, 4096)])
@pytest.mark.parametrize("dist", ["randn", "wide", "tiny", "zeros"])
def test_bits_and_scales_match_the_reference(shape, dist):
    torch.manual_seed(0)
    x = _input(shape, dist)
    q_ref, s_ref = _reference(x)
    q = torch.empty(shape, dtype=torch.float8_e4m3fn, device="cuda")
    s = torch.empty(s_ref.shape, dtype=torch.float32, device="cuda")
    fp8_kernels.quantize_into(x, q, s, TILE)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.equal(s, s_ref)


def test_strided_input_and_output_views():
    # A column cell of a train shard, written into a column view of a wider copy.
    x = torch.randn(1024, 2048, device="cuda", dtype=torch.bfloat16)[:, 1024:]
    q_ref, s_ref = _reference(x)
    wide = torch.empty(1024, 2048, dtype=torch.float8_e4m3fn, device="cuda")
    scale_wide = torch.empty(8, 16, dtype=torch.float32, device="cuda")
    fp8_kernels.quantize_into(x, wide[:, :1024], scale_wide[:, 8:], TILE)
    assert torch.equal(wide[:, :1024].contiguous().view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.equal(scale_wide[:, 8:], s_ref)


def test_runs_near_the_memory_roofline():
    x = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = torch.empty(64, 64, device="cuda")
    y = torch.empty_like(x)

    def timed(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) / 20

    quant = timed(lambda: fp8_kernels.quantize_into(x, q, s, TILE))
    copy = timed(lambda: y.copy_(x))
    # Quantising moves 3 bytes per element, a bf16 copy 4, so it should take at most as long.
    assert quant <= copy * 1.1, f"quantise {quant * 1e6:.0f} us vs bf16 copy {copy * 1e6:.0f} us"
