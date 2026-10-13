"""Read geometry: runs, pairing and coalescing agree with an element by element copy."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from psrl.utils.nixl.read_geometry import coalesce, pair_runs, view_runs

pytestmark = pytest.mark.cpu_test


def _runs_of(view: torch.Tensor, base: torch.Tensor):
    offset = (view.data_ptr() - base.data_ptr()) // view.element_size()
    return view_runs(offset * view.element_size(), tuple(view.shape), view.stride(), view.element_size())


def _apply(src_base: torch.Tensor, dst_base: torch.Tensor, local, remote, lengths) -> None:
    """Execute segments as byte copies from `src_base` into `dst_base`."""
    src = src_base.reshape(-1).view(torch.uint8)
    dst = dst_base.view(-1).view(torch.uint8)
    for l, r, n in zip(local.tolist(), remote.tolist(), lengths.tolist()):
        dst[l : l + n] = src[r : r + n]


@pytest.mark.parametrize(
    "make",
    [
        lambda t: t,
        lambda t: t[:, 2:6],
        lambda t: t[1:5],
        lambda t: t.view(4, 2, 8)[:, 1, :],
        lambda t: t.t(),
    ],
)
def test_runs_cover_exactly_the_view(make):
    base = torch.arange(64, dtype=torch.int16).reshape(8, 8)
    view = make(base)
    runs = _runs_of(view, base)
    flat = base.view(-1).view(torch.uint8)
    gathered = torch.cat([flat[s : s + runs.run_bytes] for s in runs.starts.tolist()]).view(torch.int16)
    assert torch.equal(gathered, view.reshape(-1))


def test_contiguous_view_is_one_run():
    base = torch.zeros(16, 32, dtype=torch.bfloat16)
    assert _runs_of(base[4:8], base).count == 1
    assert _runs_of(base[:, :16], base).count == 16


@pytest.mark.parametrize("cut", [(slice(None), slice(0, 4)), (slice(2, 6), slice(None)), (slice(1, 7), slice(3, 5))])
def test_pairing_strided_against_contiguous_copies_the_block(cut):
    src = torch.arange(96, dtype=torch.int16).reshape(8, 12)
    block = src[cut]
    dst = torch.zeros_like(block).contiguous()
    lr = view_runs(0, tuple(dst.shape), dst.stride(), 2)
    rr = _runs_of(block, src)
    local, remote, piece = pair_runs(lr, rr)
    local, remote, lengths = coalesce(local, remote, np.full(local.shape, piece, dtype=np.int64))
    _apply(src, dst, local, remote, lengths)
    assert torch.equal(dst, block)
    assert lengths.shape[0] == rr.count


def test_identical_layouts_coalesce_to_one_descriptor_across_cells():
    """Column cells of one tensor read from an identical tensor: the peer read case."""
    src = torch.arange(4096, dtype=torch.int16).reshape(64, 64)
    dst = torch.zeros_like(src)
    locals_, remotes, lengths = [], [], []
    for j in range(4):
        cols = slice(16 * j, 16 * (j + 1))
        l, r, piece = pair_runs(_runs_of(dst[:, cols], dst), _runs_of(src[:, cols], src))
        locals_.append(l)
        remotes.append(r)
        lengths.append(np.full(l.shape, piece, dtype=np.int64))
    local, remote, length = coalesce(np.concatenate(locals_), np.concatenate(remotes), np.concatenate(lengths))
    assert local.shape[0] == 1 and length[0] == src.numel() * 2
    _apply(src, dst, local, remote, length)
    assert torch.equal(dst, src)


def test_mismatched_rows_do_not_merge():
    src = torch.arange(256, dtype=torch.int16).reshape(16, 16)
    dst = torch.zeros(16, 8, dtype=torch.int16)
    l, r, piece = pair_runs(_runs_of(dst, dst), _runs_of(src[:, 4:12], src))
    local, remote, lengths = coalesce(l, r, np.full(l.shape, piece, dtype=np.int64))
    assert local.shape[0] == 16
    _apply(src, dst, local, remote, lengths)
    assert torch.equal(dst, src[:, 4:12])
