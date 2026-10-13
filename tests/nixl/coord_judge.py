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

    indices = torch.arange(global_offset, global_offset + numel, dtype=torch.int64, device=tensor.device)
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
        start = shard_global_offset(tensor, shardings.get(key)) if shardings and key in shardings else 0
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


# Layout judge. The transport copies bytes without interpreting them, so each element's raw bits carry
# a hash of (key, version, global coordinate). The value judge above keeps only `flat mod 256` under bf16.
_BITS_DTYPE = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _key_salt(key: str) -> int:
    import zlib

    return zlib.crc32(key.encode()) & 0xFFFFFFFF


def _coord_hash(global_flat: torch.Tensor, key: str, version: int, itemsize: int) -> torch.Tensor:
    """Mix (key, version, global flat index) into ``8 * itemsize`` bits."""
    salt = (_key_salt(key) ^ (version * 0x9E3779B1)) & 0xFFFFFFFF
    h = (global_flat * 0x9E3779B1 + salt) & 0xFFFFFFFF
    h = h ^ (h >> 15)
    h = (h * 0x85EBCA77) & 0xFFFFFFFF
    h = h ^ (h >> 13)
    bits = 8 * min(itemsize, 4)
    h = h & ((1 << bits) - 1)
    if itemsize == 2:
        h = torch.where(h >= 1 << 15, h - (1 << 16), h)  # into int16 range
    elif itemsize == 4:
        h = torch.where(h >= 1 << 31, h - (1 << 32), h)
    return h


def _cells_with_origin(entry, sharding):
    """
    Yield `(cell, origin, global_shape)` for every declared cell of `entry`, with origin `index * extent`.
    """
    from psrl.utils.nixl.moe_grouping import local_shards_of

    if sharding is None or not sharding.shard_indices:
        if isinstance(entry, torch.Tensor):
            yield entry, (0,) * entry.dim(), tuple(entry.shape)
        return
    from psrl.utils.nixl.moe_grouping import GroupedShards

    mesh = list(sharding.shard_mesh.items())
    cells = local_shards_of(entry, sharding)
    # Grouped cells lack the expert axis that the mesh numbers as dim 0.
    if isinstance(entry, GroupedShards):
        cells = [cell.unsqueeze(0) for cell in cells]
    for index, cell in zip(sharding.shard_indices, cells):
        origin = [0] * cell.dim()
        global_shape = list(cell.shape)
        for pos, (dim, count) in enumerate(mesh):
            if dim >= cell.dim():
                continue
            origin[dim] = index[pos] * cell.shape[dim]
            global_shape[dim] = count * cell.shape[dim]
        yield cell, tuple(origin), tuple(global_shape)


# Exponent masks of floating dtypes, so a hashed pattern is never NaN or infinite and survives a cast.
_EXPONENT = {
    torch.bfloat16: (0x7F80, 0x4000),
    torch.float16: (0x7C00, 0x4000),
    torch.float32: (0x7F800000, 0x40000000),
    torch.float8_e4m3fn: (0x7F, 0x01),
}


def _finite(h: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if dtype not in _EXPONENT:
        return h
    mask, flip = _EXPONENT[dtype]
    return torch.where((h & mask) == mask, h ^ flip, h)


def _expected_bits(
    cell: torch.Tensor, origin, global_shape, key: str, version: int, hash_dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Expected bits of `cell` in `hash_dtype`, the dtype the reader holds, which defaults to the cell's."""
    hash_dtype = hash_dtype or cell.dtype
    itemsize = torch.empty(0, dtype=hash_dtype).element_size()
    device = cell.device if cell.device.type != "meta" else "cpu"
    flat = torch.zeros(cell.shape, dtype=torch.int64, device=device)
    stride = 1
    for dim in reversed(range(cell.dim())):
        coords = torch.arange(origin[dim], origin[dim] + cell.shape[dim], dtype=torch.int64, device=device)
        shape = [1] * cell.dim()
        shape[dim] = cell.shape[dim]
        flat += coords.view(shape) * stride
        stride *= global_shape[dim]
    bits = _finite(_coord_hash(flat, key, version, itemsize), hash_dtype)
    return bits.to(_BITS_DTYPE[itemsize])


# Expectations are int64, so judge in blocks beside a vLLM engine that holds most of the card.
_JUDGE_BLOCK_ELEMS = 1 << 24


def _row_blocks(cell: torch.Tensor, origin):
    """Split ``cell`` along dim 0 into views of at most ``_JUDGE_BLOCK_ELEMS``."""
    if cell.dim() == 0 or cell.numel() <= _JUDGE_BLOCK_ELEMS:
        yield cell, origin
        return
    per_row = max(cell.numel() // cell.shape[0], 1)
    rows = max(_JUDGE_BLOCK_ELEMS // per_row, 1)
    for start in range(0, cell.shape[0], rows):
        piece = cell[start : start + rows]
        yield piece, (origin[0] + start, *origin[1:])


def fill_by_layout(
    state_dict: dict, shardings: dict | None, version: int = 0, wire_dtype: torch.dtype | None = None
) -> list[str]:
    """
    Fill every declared cell in place with its coordinate hash and return the filled keys.

    With `wire_dtype` the hash is drawn in that dtype and stored through an exact upcast, so a
    reader holding `wire_dtype` sees the hash bits after the transport casts back down.
    """
    filled = []
    with torch.no_grad():
        for key in sorted(state_dict):
            entry = state_dict[key]
            sharding = (shardings or {}).get(key)
            for cell, origin, global_shape in _cells_with_origin(entry, sharding):
                if cell.numel() == 0:
                    continue
                for piece, piece_origin in _row_blocks(cell, origin):
                    cast = wire_dtype is not None and wire_dtype != piece.dtype and piece.is_floating_point()
                    bits = _expected_bits(piece, piece_origin, global_shape, key, version, wire_dtype if cast else None)
                    if cast:
                        piece.data.copy_(bits.view(wire_dtype).to(piece.dtype))
                    else:
                        piece.data.view(bits.dtype).copy_(bits)
            filled.append(key)
    return filled


def check_by_layout(state_dict: dict, shardings: dict | None, keys, version: int = 0) -> tuple[int, str | None]:
    """
    Judge `keys` of a pulled state dict and return `(judged, first_failure)`.
    """
    judged = 0
    for key in sorted(set(keys) & set(state_dict)):
        entry = state_dict[key]
        sharding = (shardings or {}).get(key)
        for cell, origin, global_shape in _cells_with_origin(entry, sharding):
            if cell.numel() == 0:
                continue
            n_bad, first = 0, None
            for piece, piece_origin in _row_blocks(cell, origin):
                expected = _expected_bits(piece, piece_origin, global_shape, key, version)
                actual = piece.detach().view(expected.dtype)
                if actual.device != expected.device:
                    actual = actual.to(expected.device)
                bad = (actual != expected).nonzero()
                if bad.numel():
                    n_bad += bad.shape[0]
                    if first is None:
                        first = tuple(o + int(c) for o, c in zip(piece_origin, bad[0]))
            if n_bad:
                return judged, (
                    f"{key}: {n_bad} of {cell.numel()} elements of the cell at "
                    f"origin {origin} differ, first at global coordinate {first}."
                )
        judged += 1
    return judged, None
