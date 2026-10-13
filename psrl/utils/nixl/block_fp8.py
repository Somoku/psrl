"""
Block scaled FP8 weights on the wire: scale geometry, scale discovery and quantisation.

A block scaled weight carries one scale per tile, and vLLM requires every rank's tiles to
be whole. The scale grid therefore shards exactly like its weight, so a scale travels as
an ordinary key whose sharding is the weight's. The rollout declares which keys are
scales of which weights, read from its own parameters, and the trainer produces them.
"""

from __future__ import annotations

import torch

from psrl.utils.nixl.moe_grouping import GroupedShards

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)
SCALE_SUFFIX = "_scale_inv"


def scale_key_of(weight_key: str) -> str:
    return weight_key + SCALE_SUFFIX


def scale_view(weight_view: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    The view of `scale` holding the tiles of `weight_view`, a strided view of `weight`.

    Each view dim walks one dim of `weight`. A walk in steps of whole tiles keeps its extent
    and steps through scale tiles. A unit step walk spans whole tiles and shrinks by the tile.
    """
    assert weight.is_contiguous() and scale.is_contiguous(), "Fused weight and scale must be dense."
    block = [w // s for w, s in zip(weight.shape, scale.shape)]
    assert all(w == b * s for w, b, s in zip(weight.shape, block, scale.shape)), (
        f"Scale {tuple(scale.shape)} does not tile weight {tuple(weight.shape)}."
    )
    item = weight.element_size()
    offset = (weight_view.data_ptr() - weight.data_ptr()) // item
    origin = []
    for stride in weight.stride():
        origin.append(offset // stride)
        offset %= stride
    assert all(o % b == 0 for o, b in zip(origin, block)), f"View origin {origin} is not tile aligned to {block}."
    scale_offset = sum((o // b) * s for o, b, s in zip(origin, block, scale.stride()))

    shape, strides = [], []
    for extent, stride in zip(weight_view.shape, weight_view.stride()):
        if extent == 1:
            shape.append(1)
            strides.append(0)
            continue
        dim = next(d for d, s in enumerate(weight.stride()) if stride % s == 0 and stride // s < weight.shape[d])
        steps = stride // weight.stride(dim)
        if steps % block[dim] == 0:
            shape.append(extent)
            strides.append(steps // block[dim] * scale.stride(dim))
        else:
            assert steps == 1 and extent % block[dim] == 0, (
                f"A walk of {extent} in steps of {steps} along dim {dim} does not cover whole tiles of {block[dim]}."
            )
            shape.append(extent // block[dim])
            strides.append(scale.stride(dim))
    return torch.as_strided(scale, shape, strides, scale_offset)


def _first(entry) -> torch.Tensor:
    return entry.shards[0] if isinstance(entry, GroupedShards) else entry


def block_scales_of(state_dict: dict) -> dict[str, tuple[str, tuple[int, ...]]]:
    """
    `{scale_key: (weight_key, tile)}` for every FP8 weight in `state_dict` that has a scale.

    A grouped expert key has a leading expert dim, whose tile is 1.
    """
    found = {}
    for key, entry in state_dict.items():
        if not key.endswith(SCALE_SUFFIX):
            continue
        base = key[: -len(SCALE_SUFFIX)]
        if base not in state_dict or _first(state_dict[base]).dtype != FP8_DTYPE:
            continue
        weight, scale = _first(state_dict[base]), _first(entry)
        tile = tuple(w // s for w, s in zip(weight.shape, scale.shape))
        if isinstance(entry, GroupedShards):
            tile = (1, *tile)
        found[key] = (base, tile)
    return found


def scale_shape(shape, tile) -> tuple[int, ...]:
    assert all(n % t == 0 for n, t in zip(shape, tile)), f"Shape {tuple(shape)} is not tiled whole by {tile}."
    return tuple(n // t for n, t in zip(shape, tile))


def quantize_blockwise(x: torch.Tensor, tile: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantise `x` to FP8 with one float32 scale per tile, bit for bit as vLLM's `per_block_cast_to_fp8`.

    Returns the FP8 tensor and the scales, which are the dequantisation multipliers.
    """
    blocks = scale_shape(x.shape, tile)
    split = [d for pair in zip(blocks, tile) for d in pair]
    tiles = x.reshape(split)
    amax = tiles.abs().float().amax(dim=tuple(range(1, 2 * len(tile), 2)), keepdim=True).clamp(1e-4)
    scale = amax / FP8_MAX
    quantized = (tiles * (1.0 / scale)).to(FP8_DTYPE)
    return quantized.reshape(x.shape), scale.reshape(blocks)
