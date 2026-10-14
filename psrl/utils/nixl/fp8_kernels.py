"""
Fused block FP8 quantisation for the publish point.

`block_fp8.quantize_blockwise` reads the input four times (abs, amax, scale, cast) and
allocates its outputs, about 131 GB/s on an H20. Quantisation at publish is bound by
memory, 2 bytes in and 1 byte out per element, so one pass that reads each tile once and
writes the FP8 bytes and the scale in place is what the hardware allows.

The kernel reproduces `per_block_cast_to_fp8` bit for bit, which fixes the arithmetic:
the amax is taken in float32 and clamped at 1e-4. PyTorch's CUDA `tensor / scalar`
multiplies by the float32 reciprocal of the scalar rather than dividing, so the scale is
`amax * float32(1 / fp8_max)`. The elements are multiplied by the correctly rounded
reciprocal of the scale (`div_rn`, since Triton's float32 `/` is approximate) and cast
with round to nearest even.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU only environments
    triton = None


if triton is not None:

    @triton.jit
    def _quantize_tiles(
        x_ptr,
        q_ptr,
        s_ptr,
        rows,
        cols,
        stride_xr,
        stride_xc,
        stride_qr,
        stride_qc,
        stride_sr,
        stride_sc,
        inv_fp8_max,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        tile_r = tl.program_id(0)
        tile_c = tl.program_id(1)
        r = tile_r * BLOCK_R + tl.arange(0, BLOCK_R)[:, None]
        c = tile_c * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
        mask = (r < rows) & (c < cols)
        x = tl.load(x_ptr + r * stride_xr + c * stride_xc, mask=mask, other=0.0).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(x)), 1e-4)
        scale = amax * inv_fp8_max
        inv = tl.math.div_rn(1.0, scale)
        q = (x * inv).to(tl.float8e4nv)
        tl.store(q_ptr + r * stride_qr + c * stride_qc, q, mask=mask)
        tl.store(s_ptr + tile_r * stride_sr + tile_c * stride_sc, scale)


# float32(1 / 448), the reciprocal PyTorch multiplies by. Passed as a float32 kernel argument.
_INV_FP8_MAX = float(torch.tensor(1.0, dtype=torch.float32) / torch.tensor(torch.finfo(torch.float8_e4m3fn).max))


def available(x: torch.Tensor) -> bool:
    return triton is not None and x.is_cuda and x.dim() >= 2


def quantize_into(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor, tile: tuple[int, int]) -> None:
    """
    Block quantise `x` into `q` (FP8, same shape) and `s` (float32, one entry per tile).

    Leading dims of a 3D or larger tensor have tile 1 and are folded into rows. All three
    tensors may be strided views, which is what lets the copies of a train shard be written
    without an intermediate tensor.
    """
    block_r, block_c = tile
    rows = x.numel() // x.shape[-1]
    cols = x.shape[-1]
    assert rows % block_r == 0 and cols % block_c == 0, f"{tuple(x.shape)} is not tiled whole by {tile}."
    x2, q2 = x.reshape(rows, cols), q.view(rows, cols)
    s2 = s.view(rows // block_r, cols // block_c)
    grid = (rows // block_r, cols // block_c)
    _quantize_tiles[grid](
        x2,
        q2,
        s2,
        rows,
        cols,
        x2.stride(0),
        x2.stride(1),
        q2.stride(0),
        q2.stride(1),
        s2.stride(0),
        s2.stride(1),
        _INV_FP8_MAX,
        BLOCK_R=block_r,
        BLOCK_C=block_c,
        num_warps=8,
    )
