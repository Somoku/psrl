"""Descriptor-count alignment between strided and contiguous sides (item A2).

NIXL pairs transfer descriptors element-wise. A strided shard expands to one descriptor
per row, while the other side of the transfer -- a PS buffer or a rollout parameter --
is contiguous and therefore a single descriptor. NIXL does not split the contiguous
side automatically; it rejects the whole transfer with NIXL_ERR_INVALID_PARAM.

That is what `_align_desc_counts` exists to prevent, and it is not a hypothetical: the
5-node run produced exactly this failure on `q_proj.weight` once strided shards started
taking the in-place path.

The correspondence being relied on: the strided rows all carry the same `row_bytes`,
and the destination buffer is dense, so source row i maps to destination bytes
`[i*row_bytes, (i+1)*row_bytes)`. Cutting the contiguous descriptor into that many
equal consecutive pieces reproduces the pairing a staging copy would have produced.
"""

from __future__ import annotations

import pytest

nixlBind = pytest.importorskip("nixl._bindings", reason="NIXL not installed")

pytestmark = pytest.mark.cpu_test

SEG = nixlBind.VRAM_SEG


@pytest.fixture
def client():
    """A client built without __init__: only `_align_desc_counts` is under test."""
    from psrl.utils.nixl.client import NIXLStorageClient

    instance = NIXLStorageClient.__new__(NIXLStorageClient)
    instance.client_name = "pytest-client"
    return instance


def _descs(rows):
    return nixlBind.nixlXferDList(SEG, rows)


def _rows_of(desc):
    return [desc[i] for i in range(desc.descCount())]


class TestAlignment:
    def test_contiguous_remote_is_split_to_match_strided_local(self, client):
        strided = _descs([(0x20000 + i * 8192, 512, 0) for i in range(8)])
        contiguous = _descs([(0x10000, 4096, 0)])

        local, remote = client._align_desc_counts(strided, contiguous)

        assert local.descCount() == remote.descCount() == 8

    def test_split_pieces_are_consecutive_and_equal(self, client):
        """The split must tile the original range in order, with no gaps or overlap.

        A split that merely produced the right *count* would satisfy NIXL and silently
        scatter the bytes, so the addresses are what this checks.
        """
        strided = _descs([(0x20000 + i * 8192, 512, 0) for i in range(8)])
        contiguous = _descs([(0x10000, 4096, 0)])

        _local, remote = client._align_desc_counts(strided, contiguous)

        rows = _rows_of(remote)
        assert all(length == 512 for _addr, length, _dev in rows)
        assert [addr for addr, _l, _d in rows] == [0x10000 + i * 512 for i in range(8)]

    def test_split_preserves_total_bytes(self, client):
        strided = _descs([(0x20000 + i * 8192, 512, 0) for i in range(8)])
        contiguous = _descs([(0x10000, 4096, 0)])

        _local, remote = client._align_desc_counts(strided, contiguous)

        assert sum(length for _a, length, _d in _rows_of(remote)) == 4096

    def test_alignment_works_in_the_other_direction(self, client):
        """Either side can be the single descriptor: READ and WRITE differ in which."""
        contiguous = _descs([(0x10000, 4096, 0)])
        strided = _descs([(0x20000 + i * 8192, 512, 0) for i in range(8)])

        local, remote = client._align_desc_counts(contiguous, strided)

        assert local.descCount() == remote.descCount() == 8

    def test_equal_counts_are_returned_unchanged(self, client):
        """The contiguous-to-contiguous path must keep its single-descriptor fast case."""
        a = _descs([(0x10000, 4096, 0)])
        b = _descs([(0x20000, 4096, 0)])

        local, remote = client._align_desc_counts(a, b)

        assert local is a and remote is b

    def test_indivisible_length_is_rejected(self, client):
        """Refuse rather than round: a truncated split would silently drop bytes."""
        strided = _descs([(0x20000 + i * 8192, 512, 0) for i in range(8)])
        contiguous = _descs([(0x10000, 100, 0)])

        with pytest.raises(RuntimeError, match="does not divide"):
            client._align_desc_counts(strided, contiguous)

    def test_two_multi_descriptor_sides_are_rejected(self, client):
        """Neither side is a single descriptor, so there is nothing to split."""
        a = _descs([(0x10000 + i * 1024, 256, 0) for i in range(4)])
        b = _descs([(0x20000 + i * 1024, 256, 0) for i in range(8)])

        with pytest.raises(RuntimeError, match="neither side is a single"):
            client._align_desc_counts(a, b)
