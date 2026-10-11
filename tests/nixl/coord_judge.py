"""Coordinate-function correctness judge for weight-transport tests (item T1).

The existing e2e judge fills every parameter with 1.0 and asserts 1.0 arrives. That
detects "no data" and nothing else -- a transposed tensor of ones is still ones, any
slice of ones is ones, and ones cast to fp8 is still one. The format conversions and
precision paths this design rests on are exactly what such a judge cannot see.

This module fills each element with a deterministic function of its *global
coordinate*, so misplacement is detectable: a transpose, an off-by-one shard offset,
a swapped expert, or a wrong dtype all change values at known positions, and the
assertion reports the first offending coordinate.

Two constraints shape the encoding, both from the design doc (7.0):

1. The value must be exactly representable in the destination dtype. fp8_e4m3 carries
   3 mantissa bits, so only small integers survive; a float hash would not.
2. The map from coordinate to value must be injective within a tensor, or a swap of
   two positions is invisible.

`encode_coord` therefore packs small integers by bit position and keeps the result
inside the dtype's exactly-representable integer range, which `assert_dtype_capacity`
checks up front rather than letting a silent rounding pass as success.
"""

from __future__ import annotations

import torch

# Largest integer N such that every integer in [0, N] is exactly representable.
# For a float with m mantissa bits that is 2**(m+1); beyond it, values round.
_EXACT_INT_LIMIT: dict[torch.dtype, int] = {
    torch.float8_e4m3fn: 2**4,  # 3 mantissa bits -> integers up to 16
    torch.float8_e5m2: 2**3,  # 2 mantissa bits -> integers up to 8
    torch.bfloat16: 2**8,  # 7 mantissa bits
    torch.float16: 2**11,  # 10 mantissa bits
    torch.float32: 2**24,
    torch.float64: 2**53,
    torch.int8: 2**7 - 1,
    torch.int32: 2**31 - 1,
    torch.int64: 2**62,
}


class CoordinateCapacityError(AssertionError):
    """Raised when a tensor's coordinate space cannot be encoded in its dtype."""


def exact_int_limit(dtype: torch.dtype) -> int:
    if dtype not in _EXACT_INT_LIMIT:
        raise CoordinateCapacityError(f"no exact-integer limit recorded for {dtype}")
    return _EXACT_INT_LIMIT[dtype]


def coord_space_size(shape: tuple[int, ...]) -> int:
    """Number of distinct values the encoding needs for a tensor of this shape."""
    total = 1
    for dim in shape:
        total *= max(dim, 1)
    return total


def assert_dtype_capacity(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    """Fail loudly when the dtype cannot hold one distinct value per coordinate.

    Without this the encoding silently saturates and distinct positions compare
    equal, which would make the judge report success for a misplaced shard -- the
    precise failure the judge exists to catch.
    """
    needed = coord_space_size(shape)
    limit = exact_int_limit(dtype)
    if needed > limit:
        raise CoordinateCapacityError(
            f"shape {shape} needs {needed} distinct values but {dtype} represents only "
            f"integers up to {limit} exactly. Use a smaller test tensor, or verify this "
            f"dtype with a tiled encoding rather than a global one."
        )


def encode_coord(
    flat_index: int | torch.Tensor,
    dtype: torch.dtype,
    *,
    modulus: int | None = None,
) -> int | torch.Tensor:
    """Map a flat global index to a value exactly representable in ``dtype``.

    ``modulus`` trades injectivity for dtype range: with it, the encoding repeats
    every ``modulus`` elements, so it detects *local* displacement but not a shift by
    an exact multiple of the modulus. fp8 needs this (its exact range is 16 values),
    and `fill_by_global_coord` combines it with a per-shard offset check so a
    whole-shard displacement is still caught.
    """
    limit = exact_int_limit(dtype)
    span = min(modulus, limit) if modulus else limit
    return flat_index % span


def fill_by_global_coord(
    tensor: torch.Tensor,
    global_offset: int = 0,
    *,
    strict: bool = True,
) -> torch.Tensor:
    """Fill ``tensor`` in place so each element encodes its global flat position.

    ``global_offset`` is the element index this tensor's first element occupies in the
    full logical tensor, which is what makes a misplaced shard detectable: shard k
    holds a different value sequence than shard k+1.

    With ``strict`` the dtype must distinguish every coordinate. Set it False only for
    fp8, where the modulus is unavoidable.
    """
    numel = tensor.numel()
    if strict:
        assert_dtype_capacity((numel,), tensor.dtype)
        modulus = None
    else:
        modulus = exact_int_limit(tensor.dtype)

    indices = torch.arange(
        global_offset, global_offset + numel, dtype=torch.int64, device=tensor.device
    )
    values = encode_coord(indices, tensor.dtype, modulus=modulus)
    # `no_grad` + `.data`: the e2e harness fills real model parameters, which are leaf
    # Variables requiring grad, and autograd rejects an in-place write through a view of
    # one. This mirrors how the production sync hooks write into parameters
    # (ParamSyncPlan wraps every action in torch.no_grad()).
    #
    # The fill must stay in place and must not assume contiguity. The converters hand
    # back strided views of a fused buffer (QKV slices, MoE expert slices), for which
    # `.view(-1)` raises "spans across two contiguous subspaces"; `.reshape(-1)` would
    # silently return a *copy* for those, so the fill would land nowhere. Writing
    # through a same-shape view of the value buffer keeps the write in place for both
    # contiguous and strided targets.
    with torch.no_grad():
        target = tensor.data
        target.copy_(values.to(tensor.dtype).view(target.shape))
    return tensor


def assert_matches_global_coord(
    tensor: torch.Tensor,
    global_offset: int = 0,
    *,
    key: str = "<tensor>",
    strict: bool = True,
    rtol: float = 0.0,
) -> None:
    """Assert every element still encodes its global coordinate.

    Reports the first mismatching flat index and both values, which localises the
    fault to a dimension: a constant offset means a shard-index error, a stride-shaped
    pattern means a transpose or reshape, and scattered noise means a precision issue.
    """
    numel = tensor.numel()
    modulus = None if strict else exact_int_limit(tensor.dtype)
    indices = torch.arange(global_offset, global_offset + numel, dtype=torch.int64)
    expected = encode_coord(indices, tensor.dtype, modulus=modulus).to(torch.float64)
    actual = tensor.detach().to(torch.float64).reshape(-1).cpu()

    if rtol == 0.0:
        mismatches = (actual != expected).nonzero()
    else:
        mismatches = (~torch.isclose(actual, expected, rtol=rtol, atol=0.0)).nonzero()

    if mismatches.numel() == 0:
        return

    first = int(mismatches[0].item())
    raise AssertionError(
        f"{key}: coordinate mismatch at flat index {first} "
        f"(global {global_offset + first}): expected {expected[first].item()}, "
        f"got {actual[first].item()}. {mismatches.numel()} of {numel} elements differ. "
        f"A constant delta indicates a shard-offset error; a periodic pattern "
        f"indicates a transpose or reshape; scattered differences indicate precision."
    )


def fill_state_dict_by_coord(
    state_dict: dict[str, torch.Tensor],
    *,
    strict: bool = True,
    base: int = 0,
    shardings: dict | None = None,
) -> dict[str, int]:
    """Fill every tensor in a state dict, giving each key a disjoint coordinate range.

    Returns the per-key global offsets, which the matching assertion needs. Disjoint
    ranges are what make a *cross-key* mix-up detectable -- two keys filled from zero
    would be interchangeable.

    ``base`` shifts every coordinate, which is how a second version is made genuinely
    different from the first. Without it a re-fill produces identical bytes, and a reader
    that received nothing at all would be indistinguishable from one that received the
    new version -- the exact confusion a steady-state switch-back test has to avoid.
    """
    offsets: dict[str, int] = {}
    cursor = base
    for key in sorted(state_dict):
        tensor = state_dict[key]
        # `cursor` walks the *global* key space, so it stays comparable across ranks --
        # the assertion is given the same offsets on every rank and checks the whole
        # logical tensor. `start` is where this rank's shard sits inside that range.
        #
        # Without `shardings` this is the unsharded case and `start` is zero. Threading
        # it here rather than leaving each rank to fill from zero is what makes a
        # misdelivered shard visible: 8 ranks each filling their own 16 elements from
        # zero all produce identical bytes, so the check passes whatever the transport
        # did -- which is exactly the failure the FSDP2 cell reported.
        offsets[key] = cursor
        start = (
            shard_global_offset(tensor, shardings.get(key))
            if shardings and key in shardings
            else 0
        )
        fill_by_global_coord(tensor, cursor + start, strict=strict)
        cursor += tensor.numel()
    return offsets


def shard_global_offset(
    tensor: torch.Tensor,
    sharding,
    *,
    global_shape: tuple[int, ...] | None = None,
) -> int:
    """Where this local shard starts in the global tensor's flat element order.

    Filling by coordinate only detects misplacement if each rank writes *its own* range.
    `fill_state_dict_by_coord` walks the local state dict from zero, so under any sharded
    backend every rank writes the same values and the check cannot see a shard that went
    to the wrong place -- which is the one thing it exists to catch. That is not
    hypothetical: it is what the FSDP2 cell reported as

        q_norm.weight: coordinate mismatch at flat index 16: expected 32.0, got 16.0
        112 of 128 elements differ

    which is `offset + (i mod local_numel)` -- every one of the 8 ranks having filled its
    16-element shard with shard 0's coordinates.

    ``sharding`` is a `NIXLSharding`: ``shard_mesh`` gives each dimension's shard count
    and ``shard_indices`` the coordinates this tensor owns. Only the leading dimension is
    sliced (row-parallel), which is what the converters produce, so the offset is a
    multiple of the local row count.
    """
    if sharding is None or not sharding.shard_indices:
        return 0

    mesh = sharding.shard_mesh
    dims = sorted(mesh)
    if not dims:
        return 0

    row_dim = dims[0]
    local_rows = tensor.shape[row_dim]
    # Row-major, so everything before the first sliced dimension multiplies by the
    # trailing extent. For the 1-D and 2-D row-parallel cases the converters emit, this
    # reduces to (shard row index) x local_rows x trailing_extent.
    trailing = 1
    for extent in tensor.shape[row_dim + 1 :]:
        trailing *= extent
    leading = 1
    for extent in tensor.shape[:row_dim]:
        leading *= extent

    start = 0
    for index in sharding.shard_indices:
        start += index[row_dim] * local_rows * trailing
    return start * leading


def assert_state_dict_matches_coord(
    state_dict: dict[str, torch.Tensor],
    offsets: dict[str, int],
    *,
    strict: bool = True,
    rtol: float = 0.0,
) -> None:
    """Assert every tensor in a state dict still encodes its coordinates."""
    for key, tensor in state_dict.items():
        if key not in offsets:
            raise AssertionError(f"{key}: no recorded global offset; was it added after fill?")
        assert_matches_global_coord(tensor, offsets[key], key=key, strict=strict, rtol=rtol)
