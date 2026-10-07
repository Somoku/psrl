"""Data-correctness tests for the prepped and strided descriptor paths (items A1, A2).

The micro-benchmark established that `make_prepped_xfer` is 6-28x cheaper than
`initialize_xfer` and that one Nx5 strided descriptor replaces 16384 plain ones. Speed
is not the risk, though -- the risk is moving the wrong bytes. These tests perform real
RDMA loopback transfers and verify the destination with the coordinate judge, so a
descriptor that is fast but misaddressed fails here rather than in a training run.

Specifically:

* A1 -- a prepped handle submitted with an int32 index array must move the same bytes
  as `initialize_xfer`, including under reordered and partial index sets, which is what
  the design's chunk-offset staggering relies on.
* A2 -- a strided descriptor must be byte-identical to the equivalent list of plain
  descriptors. That equivalence is the precondition for deleting the pinned staging
  path, since nothing else would catch a stride computed one element off.

Agents are session-scoped (see `conftest_agents.py`): construction costs ~60s here.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.nixl.conftest_agents import (  # noqa: F401  (fixtures)
    agent_pair,
    cuda_available,
    registered_buffer,
    xfer_descs,
)

nixlBind = pytest.importorskip("nixl._bindings", reason="NIXL not installed")

SLICE_BYTES = 64 * 1024
SLICE_COUNT = 64
TOTAL_BYTES = SLICE_BYTES * SLICE_COUNT


def _fill_pattern(buffer: torch.Tensor) -> None:
    """Write a position-dependent byte pattern so displacement is detectable.

    uint8 wraps every 256 bytes, so this detects any shift that is not a multiple of
    256. Slices here are 64KiB, far larger, so a whole-slice displacement always lands
    on a different value sequence.
    """
    length = buffer.numel()
    indices = torch.arange(length, dtype=torch.int64, device=buffer.device)
    buffer.copy_((indices % 251).to(torch.uint8))


def _assert_slice_matches(
    destination: torch.Tensor,
    source: torch.Tensor,
    offset: int,
    length: int,
    label: str,
) -> None:
    got = destination[offset : offset + length]
    want = source[offset : offset + length]
    if not torch.equal(got, want):
        differing = (got != want).nonzero()
        first = int(differing[0].item())
        raise AssertionError(
            f"{label}: byte mismatch at offset {offset + first} "
            f"(expected {int(want[first])}, got {int(got[first])}); "
            f"{differing.numel()} of {length} bytes differ"
        )


def _transfer(local, op, local_descs, remote_descs, peer_name, tag=b"t"):
    """Post one transfer and wait for completion."""
    handle = local.initialize_xfer(op, local_descs, remote_descs, peer_name, tag)
    assert handle, "initialize_xfer returned no handle"
    try:
        state = local.transfer(handle)
        assert state != "ERR", "transfer submission failed"
        for _ in range(200_000):
            state = local.check_xfer_state(handle)
            assert state != "ERR", "transfer reported ERR"
            if state == "DONE":
                return
        raise AssertionError("transfer did not complete")
    finally:
        local.release_xfer_handle(handle)


class TestPreppedDescriptorCorrectness:
    """A1: the prepped path must be equivalent to initialize_xfer, not merely faster."""

    @pytest.fixture
    def buffers(self, registered_buffer):
        source = registered_buffer(TOTAL_BYTES)
        destination = registered_buffer(TOTAL_BYTES)
        _fill_pattern(source)
        destination.zero_()
        return source, destination

    def test_prepped_transfer_moves_correct_bytes(self, agent_pair, buffers):
        local, peer = agent_pair
        source, destination = buffers
        slices = [(i * SLICE_BYTES, SLICE_BYTES) for i in range(SLICE_COUNT)]

        local_handle = local.prep_xfer_dlist(
            "NIXL_INIT_AGENT", xfer_descs(local, destination, slices), mem_type="VRAM"
        )
        remote_handle = local.prep_xfer_dlist(
            "pytest-peer", xfer_descs(local, source, slices), mem_type="VRAM"
        )
        try:
            indices = np.arange(SLICE_COUNT, dtype=np.int32)
            handle = local.make_prepped_xfer(
                "READ", local_handle, indices, remote_handle, indices, b"prepped"
            )
            assert handle
            try:
                assert local.transfer(handle) != "ERR"
                for _ in range(200_000):
                    state = local.check_xfer_state(handle)
                    assert state != "ERR"
                    if state == "DONE":
                        break
                else:
                    raise AssertionError("prepped transfer did not complete")
            finally:
                local.release_xfer_handle(handle)
        finally:
            local.release_dlist_handle(local_handle)
            local.release_dlist_handle(remote_handle)

        assert torch.equal(destination, source), "prepped transfer delivered wrong bytes"

    def test_reordered_indices_move_the_same_bytes(self, agent_pair, buffers):
        """Chunk-offset staggering submits rotated index sets over one handle."""
        local, peer = agent_pair
        source, destination = buffers
        slices = [(i * SLICE_BYTES, SLICE_BYTES) for i in range(SLICE_COUNT)]

        local_handle = local.prep_xfer_dlist(
            "NIXL_INIT_AGENT", xfer_descs(local, destination, slices), mem_type="VRAM"
        )
        remote_handle = local.prep_xfer_dlist(
            "pytest-peer", xfer_descs(local, source, slices), mem_type="VRAM"
        )
        try:
            rotated = np.roll(np.arange(SLICE_COUNT, dtype=np.int32), -SLICE_COUNT // 3)
            handle = local.make_prepped_xfer(
                "READ", local_handle, rotated, remote_handle, rotated, b"rotated"
            )
            try:
                assert local.transfer(handle) != "ERR"
                for _ in range(200_000):
                    if local.check_xfer_state(handle) == "DONE":
                        break
            finally:
                local.release_xfer_handle(handle)
        finally:
            local.release_dlist_handle(local_handle)
            local.release_dlist_handle(remote_handle)

        # Rotating both sides identically is a permutation of the pairing, so the
        # result must still be a faithful copy.
        assert torch.equal(destination, source), "rotated indices delivered wrong bytes"

    def test_subset_indices_touch_only_their_slices(self, agent_pair, buffers):
        """Commit-group submission sends a subset; the rest must stay untouched."""
        local, peer = agent_pair
        source, destination = buffers
        slices = [(i * SLICE_BYTES, SLICE_BYTES) for i in range(SLICE_COUNT)]

        local_handle = local.prep_xfer_dlist(
            "NIXL_INIT_AGENT", xfer_descs(local, destination, slices), mem_type="VRAM"
        )
        remote_handle = local.prep_xfer_dlist(
            "pytest-peer", xfer_descs(local, source, slices), mem_type="VRAM"
        )
        try:
            subset = np.array([0, 1, 2, 3], dtype=np.int32)
            handle = local.make_prepped_xfer(
                "READ", local_handle, subset, remote_handle, subset, b"subset"
            )
            try:
                assert local.transfer(handle) != "ERR"
                for _ in range(200_000):
                    if local.check_xfer_state(handle) == "DONE":
                        break
            finally:
                local.release_xfer_handle(handle)
        finally:
            local.release_dlist_handle(local_handle)
            local.release_dlist_handle(remote_handle)

        for i in subset:
            _assert_slice_matches(
                destination, source, int(i) * SLICE_BYTES, SLICE_BYTES, f"slice {i}"
            )
        untouched = destination[4 * SLICE_BYTES :]
        assert torch.all(untouched == 0), "transfer wrote outside the submitted subset"

    def test_int32_indices_are_required(self, agent_pair, buffers):
        """int64 indices must not silently produce a wrong transfer.

        The benchmark showed a Python list is accepted but ~15x slower. What matters
        for correctness is that a wrong dtype does not quietly misaddress: either the
        binding rejects it, or it converts and the bytes still land correctly.
        """
        local, _peer = agent_pair
        source, destination = buffers
        slices = [(i * SLICE_BYTES, SLICE_BYTES) for i in range(4)]

        local_handle = local.prep_xfer_dlist(
            "NIXL_INIT_AGENT", xfer_descs(local, destination, slices), mem_type="VRAM"
        )
        remote_handle = local.prep_xfer_dlist(
            "pytest-peer", xfer_descs(local, source, slices), mem_type="VRAM"
        )
        try:
            wrong_dtype = np.arange(4, dtype=np.int64)
            try:
                handle = local.make_prepped_xfer(
                    "READ", local_handle, wrong_dtype, remote_handle, wrong_dtype, b"i64"
                )
            except Exception:
                return  # rejected outright, which is the safe outcome
            try:
                assert local.transfer(handle) != "ERR"
                for _ in range(200_000):
                    if local.check_xfer_state(handle) == "DONE":
                        break
            finally:
                local.release_xfer_handle(handle)
        finally:
            local.release_dlist_handle(local_handle)
            local.release_dlist_handle(remote_handle)

        for i in range(4):
            _assert_slice_matches(
                destination, source, i * SLICE_BYTES, SLICE_BYTES, f"int64 slice {i}"
            )


class TestStridedDescriptorEquivalence:
    """A2: a strided descriptor must be byte-identical to the plain list it replaces."""

    @pytest.fixture
    def buffers(self, registered_buffer):
        source = registered_buffer(TOTAL_BYTES)
        plain_destination = registered_buffer(TOTAL_BYTES)
        strided_destination = registered_buffer(TOTAL_BYTES)
        _fill_pattern(source)
        plain_destination.zero_()
        strided_destination.zero_()
        return source, plain_destination, strided_destination

    @staticmethod
    def _strided_descs(agent, buffer, slice_bytes, stride, count):
        array = np.zeros((1, 5), dtype=np.uint64)
        array[0] = (buffer.data_ptr(), slice_bytes, buffer.device.index, stride, count)
        return array

    def test_contiguous_stride_matches_plain_descriptors(self, agent_pair, buffers):
        local, _peer = agent_pair
        source, plain_destination, strided_destination = buffers
        slices = [(i * SLICE_BYTES, SLICE_BYTES) for i in range(SLICE_COUNT)]

        # Plain path.
        _transfer(
            local,
            "READ",
            xfer_descs(local, plain_destination, slices),
            xfer_descs(local, source, slices),
            "pytest-peer",
            b"plain",
        )

        # Strided path: one descriptor for the whole set.
        local_handle = local.prep_xfer_dlist(
            "NIXL_INIT_AGENT",
            self._strided_descs(local, strided_destination, SLICE_BYTES, SLICE_BYTES, SLICE_COUNT),
            mem_type="VRAM",
        )
        remote_handle = local.prep_xfer_dlist(
            "pytest-peer",
            self._strided_descs(local, source, SLICE_BYTES, SLICE_BYTES, SLICE_COUNT),
            mem_type="VRAM",
        )
        try:
            # A strided row expands to `count` indexable entries inside the prepped
            # handle -- it compresses the *descriptor list*, not the submission index
            # set. Submitting a single index transfers only the first slice, silently
            # and with a DONE status, which is why this is asserted byte-wise.
            indices = np.arange(SLICE_COUNT, dtype=np.int32)
            handle = local.make_prepped_xfer(
                "READ", local_handle, indices, remote_handle, indices, b"strided"
            )
            try:
                assert local.transfer(handle) != "ERR"
                for _ in range(200_000):
                    if local.check_xfer_state(handle) == "DONE":
                        break
            finally:
                local.release_xfer_handle(handle)
        finally:
            local.release_dlist_handle(local_handle)
            local.release_dlist_handle(remote_handle)

        assert torch.equal(plain_destination, source), "plain path is wrong; test invalid"
        assert torch.equal(strided_destination, plain_destination), (
            "strided descriptor is not byte-equivalent to the plain descriptor list"
        )

    def test_gapped_stride_skips_the_gaps(self, agent_pair, registered_buffer):
        """A stride larger than the slice must leave the gaps untouched.

        This is the shape MoE expert sharding produces, and an off-by-one stride would
        silently shift every expert -- invisible to a judge that only checks arrival.
        """
        local, _peer = agent_pair
        count, slice_bytes, stride = 16, 4096, 8192
        span = stride * count
        source = registered_buffer(span)
        destination = registered_buffer(span)
        _fill_pattern(source)
        destination.zero_()

        local_handle = local.prep_xfer_dlist(
            "NIXL_INIT_AGENT",
            self._strided_descs(local, destination, slice_bytes, stride, count),
            mem_type="VRAM",
        )
        remote_handle = local.prep_xfer_dlist(
            "pytest-peer",
            self._strided_descs(local, source, slice_bytes, stride, count),
            mem_type="VRAM",
        )
        try:
            indices = np.arange(count, dtype=np.int32)
            handle = local.make_prepped_xfer(
                "READ", local_handle, indices, remote_handle, indices, b"gapped"
            )
            try:
                assert local.transfer(handle) != "ERR"
                for _ in range(200_000):
                    if local.check_xfer_state(handle) == "DONE":
                        break
            finally:
                local.release_xfer_handle(handle)
        finally:
            local.release_dlist_handle(local_handle)
            local.release_dlist_handle(remote_handle)

        for i in range(count):
            offset = i * stride
            _assert_slice_matches(destination, source, offset, slice_bytes, f"strided slice {i}")
            gap = destination[offset + slice_bytes : offset + stride]
            assert torch.all(gap == 0), f"slice {i}: transfer wrote into the gap"

    def test_strided_rejects_nx3_shape(self, agent_pair, registered_buffer):
        """nixlXferDList is Nx3-only in every build; strided must go via prep.

        Recording the asymmetry so it is not mistaken for a missing feature later.
        """
        local, _peer = agent_pair
        buffer = registered_buffer(TOTAL_BYTES)
        strided = self._strided_descs(local, buffer, SLICE_BYTES, SLICE_BYTES, SLICE_COUNT)

        with pytest.raises(Exception, match="Nx3|shape"):
            nixlBind.nixlXferDList(nixlBind.VRAM_SEG, strided)
