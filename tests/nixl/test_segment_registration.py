"""Segment-granularity registration: one NIXL region per allocator segment.

A vLLM instance holds one CUDA allocation per weight (and one per FP8 scale), and NIXL
registration costs roughly a fixed budget per region. Registering the containing
caching-allocator segment once covers every tensor in it without moving bytes, changing
the data path, or keeping extra memory. The invariants are:

* read_bases / IPC handles stay per **allocation** (a segment may hold several, each
  with its own IPC handle);
* the registered extent is the segment, deduplicated once;
* an allocation the snapshot cannot place (CuMem pool, publish copies allocated after
  the snapshot) falls back to its own extent;
* expandable segments disable the mode, because their pages are mapped on demand and
  must not be registered as one page-locked range.
"""

from __future__ import annotations

import pytest
import torch
from psrl.utils.nixl.client import NIXLStorageClient, _expandable_segments_enabled

pytestmark = pytest.mark.cpu_test


def _client(segment_registration: bool = True) -> NIXLStorageClient:
    client = NIXLStorageClient.__new__(NIXLStorageClient)
    client.client_name = "test-client"
    client.device = torch.device("cpu")
    client._segment_registration = segment_registration
    client._segment_extents = []
    client._storage_bases = set()
    client._reg_regions = set()
    client._mtype_to_reg_region_lists = {}
    client._ipc_regions = {}
    return client


def _storage(tensor: torch.Tensor) -> tuple[int, int]:
    storage = tensor.untyped_storage()
    return storage.data_ptr(), storage.nbytes()


class TestExtentLookup:
    def test_an_allocation_inside_a_segment_maps_to_it(self):
        client = _client()
        client._segment_extents = [(1000, 500), (2000, 800)]

        assert client._registration_extent(1200, 100) == (1000, 500)
        assert client._registration_extent(1000, 500) == (1000, 500)

    def test_an_allocation_across_a_segment_boundary_keeps_its_own_extent(self):
        """Registering a range that runs past the segment would cover unmapped VA."""
        client = _client()
        client._segment_extents = [(1000, 500)]

        assert client._registration_extent(1400, 200) == (1400, 200)

    def test_an_allocation_outside_every_segment_keeps_its_own_extent(self):
        """CuMem-pool and post-snapshot allocations are not in the snapshot."""
        client = _client()
        client._segment_extents = [(1000, 500)]

        assert client._registration_extent(5000, 128) == (5000, 128)

    def test_without_segments_every_allocation_is_its_own_region(self):
        client = _client(segment_registration=False)

        assert client._registration_extent(1000, 128) == (1000, 128)


class TestRecording:
    def test_two_allocations_in_one_segment_register_once(self):
        a = torch.empty(64, dtype=torch.uint8)
        b = torch.empty(64, dtype=torch.uint8)
        a_base, a_bytes = _storage(a)
        b_base, b_bytes = _storage(b)
        low = min(a_base, b_base)
        span = max(a_base + a_bytes, b_base + b_bytes) - low

        client = _client()
        client._segment_extents = [(low, span)]
        returned_a = client._record_region_registration(a)
        returned_b = client._record_region_registration(b)

        assert list(client._reg_regions) == [(low, span, 0, "cpu")]
        assert client._mtype_to_reg_region_lists["cpu"] == [(low, span, 0, "")]
        # The caller-facing key is the allocation, not the segment: read_bases and IPC
        # handles are per allocation.
        assert returned_a == (a_base, a_bytes, 0, "cpu")
        assert returned_b == (b_base, b_bytes, 0, "cpu")
        assert client._storage_bases == {a_base, b_base}

    def test_without_segments_each_allocation_registers(self):
        a = torch.empty(64, dtype=torch.uint8)
        b = torch.empty(64, dtype=torch.uint8)
        client = _client(segment_registration=False)

        client._record_region_registration(a)
        client._record_region_registration(b)

        assert len(client._reg_regions) == 2


class TestExpandableSegments:
    def test_the_flag_is_detected(self, monkeypatch):
        monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:20,expandable_segments:True")
        assert _expandable_segments_enabled() is True

    def test_other_options_do_not_disable_segment_registration(self, monkeypatch):
        monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:20")
        assert _expandable_segments_enabled() is False

    def test_the_newer_variable_is_honoured(self, monkeypatch):
        monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
        monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        assert _expandable_segments_enabled() is True
