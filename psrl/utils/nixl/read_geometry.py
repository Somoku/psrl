"""
Byte geometry of a read: which contiguous runs move from where to where.

A transfer between two views of the same logical block is a list of `(local, remote,
length)` segments. Its cost is dominated by the segment count, since NIXL charges a few
microseconds per descriptor while bytes move at tens of GB/s. Everything here works on
whole arrays so a plan of millions of runs is built in milliseconds, once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RunList:
    """
    Equal length contiguous runs that cover a view in row major element order.
    """

    starts: np.ndarray  # int64 byte addresses, one per run
    run_bytes: int

    @property
    def count(self) -> int:
        return int(self.starts.shape[0])

    @property
    def total_bytes(self) -> int:
        return self.count * self.run_bytes


def view_runs(address: int, shape: tuple[int, ...], strides: tuple[int, ...], itemsize: int) -> RunList:
    """
    Decompose a strided view into its contiguous runs.

    Args:
        address (int): Byte address of element zero of the view.
        shape (tuple[int, ...]): Extent per dim.
        strides (tuple[int, ...]): Element stride per dim.
        itemsize (int): Bytes per element.
    """
    assert len(shape) == len(strides), f"Shape {shape} and strides {strides} disagree in rank."
    dims = [(int(n), int(s)) for n, s in zip(shape, strides) if n != 1]
    if any(n == 0 for n, _ in dims):
        return RunList(np.zeros(0, dtype=np.int64), 0)
    # The run is the longest suffix whose dims tile memory without gaps.
    run_elems = 1
    while dims and dims[-1][1] == run_elems:
        run_elems *= dims.pop()[0]
    starts = np.array([address], dtype=np.int64)
    for extent, stride in dims:
        offsets = np.arange(extent, dtype=np.int64) * (stride * itemsize)
        starts = (starts[:, None] + offsets[None, :]).reshape(-1)
    return RunList(starts, run_elems * itemsize)


def split_runs(runs: RunList, piece_bytes: int) -> RunList:
    """
    Cut every run into pieces of `piece_bytes`, which must divide the run length.
    """
    if piece_bytes == runs.run_bytes or runs.count == 0:
        return runs
    assert runs.run_bytes % piece_bytes == 0, f"Piece of {piece_bytes} bytes does not divide {runs.run_bytes}."
    offsets = np.arange(runs.run_bytes // piece_bytes, dtype=np.int64) * piece_bytes
    return RunList((runs.starts[:, None] + offsets[None, :]).reshape(-1), piece_bytes)


def pair_runs(local: RunList, remote: RunList) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Pair two run lists that cover the same bytes in the same element order.

    Both run lengths are products of trailing extents of one shape, so the shorter divides
    the longer and every segment has the shorter length.

    Returns:
        tuple[np.ndarray, np.ndarray, int]: Local starts, remote starts, segment bytes.
    """
    assert local.total_bytes == remote.total_bytes, (
        f"Local covers {local.total_bytes} bytes but remote covers {remote.total_bytes}."
    )
    if local.count == 0:
        return local.starts, remote.starts, 0
    piece = min(local.run_bytes, remote.run_bytes)
    return split_runs(local, piece).starts, split_runs(remote, piece).starts, piece


def coalesce(
    local: np.ndarray,
    remote: np.ndarray,
    lengths: np.ndarray,
    regions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge segments that are adjacent on both sides into single descriptors.

    Segments are ordered by local address first, so runs of one row that belong to
    different cells (for example the column halves of one tensor) become neighbours.
    `regions` labels each segment with its pair of registered allocations. Separately
    registered allocations can sit back to back in memory, and a descriptor spanning two
    of them is rejected by the transport, so segments only merge within one label.
    """
    if local.shape[0] <= 1:
        return local, remote, lengths
    order = np.lexsort((remote, local))
    local, remote, lengths = local[order], remote[order], lengths[order]
    ends_local = local[:-1] + lengths[:-1]
    ends_remote = remote[:-1] + lengths[:-1]
    joins = (ends_local == local[1:]) & (ends_remote == remote[1:])
    if regions is not None:
        regions = regions[order]
        joins &= regions[:-1] == regions[1:]
    heads = np.concatenate(([True], ~joins))
    head_index = np.flatnonzero(heads)
    merged_lengths = np.add.reduceat(lengths, head_index)
    return local[head_index], remote[head_index], merged_lengths


def run_count(shape: tuple[int, ...], strides: tuple[int, ...]) -> int:
    """
    Number of contiguous runs `view_runs` would return, without building them.
    """
    dims = [(int(n), int(s)) for n, s in zip(shape, strides) if n != 1]
    if any(n == 0 for n, _ in dims):
        return 0
    run_elems = 1
    while dims and dims[-1][1] == run_elems:
        run_elems *= dims.pop()[0]
    count = 1
    for extent, _ in dims:
        count *= extent
    return count


def contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """
    Row major element strides of a dense tensor of `shape`.
    """
    strides = []
    acc = 1
    for extent in reversed(shape):
        strides.append(acc)
        acc *= int(extent)
    return tuple(reversed(strides))
