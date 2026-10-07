"""End-to-end verification that the prepped hot path moves correct bytes (item A1).

`test_prepped_xfer_cache.py` proves the cache's bookkeeping against a fake agent, and
`test_descriptor_correctness_gpu.py` proves NIXL's prepped API moves bytes correctly.
This file closes the gap between them: it drives `NIXLStorageClient.post_cached_xfer`
with a real agent and real GPU memory, then checks the destination byte-for-byte.

That combination is what the A1 criterion actually requires. Counting
`initialize_xfer` calls proves which path ran; comparing bytes proves the path is
right. Either alone would miss the failure mode where the fast path runs and delivers
the wrong data.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.nixl.conftest_agents import (  # noqa: F401  (fixtures)
    agent_pair,
    cuda_available,
    registered_buffer,
)

nixlBind = pytest.importorskip("nixl._bindings", reason="NIXL not installed")

SLICE_BYTES = 64 * 1024
SLICE_COUNT = 32
TOTAL_BYTES = SLICE_BYTES * SLICE_COUNT


class _Desc:
    """The minimal descriptor-list interface `post_cached_xfer` consumes."""

    def __init__(self, items):
        self._items = items

    def descCount(self):
        return len(self._items)

    def getType(self):
        return nixlBind.VRAM_SEG

    def __getitem__(self, index):
        return self._items[index]


def _client(agent, name: str):
    from psrl.utils.nixl.client import NIXLStorageClient
    from psrl.utils.nixl.prepped_xfer import PreppedXferCache

    client = NIXLStorageClient.__new__(NIXLStorageClient)
    client.agent = agent
    client.client_name = name
    client._cached_xfer_descs = []
    client._prepped_cache = PreppedXferCache(agent, name)
    return client


def _fill(buffer: torch.Tensor) -> None:
    indices = torch.arange(buffer.numel(), dtype=torch.int64, device=buffer.device)
    buffer.copy_((indices % 251).to(torch.uint8))


def _queue_read(client, destination, source, slice_indices, tag: str, target: str):
    """Queue one merged READ group, as `client_read` does with merging enabled."""
    for i in slice_indices:
        offset = i * SLICE_BYTES
        local = _Desc([(destination.data_ptr() + offset, SLICE_BYTES, destination.device.index)])
        remote = _Desc([(source.data_ptr() + offset, SLICE_BYTES, source.device.index)])
        client._cached_xfer_descs.append(("READ", local, remote, "pytest-peer", tag, target))


class TestPreppedHotPathDeliversCorrectBytes:
    @pytest.fixture
    def buffers(self, registered_buffer):
        source = registered_buffer(TOTAL_BYTES)
        destination = registered_buffer(TOTAL_BYTES)
        _fill(source)
        destination.zero_()
        return source, destination

    def test_full_round_is_byte_exact(self, agent_pair, buffers):
        local_agent, _peer = agent_pair
        source, destination = buffers
        client = _client(local_agent, "pytest-local")

        _queue_read(client, destination, source, range(SLICE_COUNT), "chunk0", "pytest-peer")
        posted = client.post_cached_xfer()
        client.wait_posted_xfer(posted, timeout=120.0)

        assert torch.equal(destination, source), "prepped hot path delivered wrong bytes"
        assert client._prepped_cache is not None, "prepped path fell back unexpectedly"
        assert client._prepped_cache.submit_count == 1

    def test_three_rounds_reuse_one_prepped_dlist(self, agent_pair, buffers):
        """prep once, submit many -- and every round must still be correct."""
        local_agent, _peer = agent_pair
        source, destination = buffers
        client = _client(local_agent, "pytest-local")

        thirds = [range(0, 10), range(10, 20), range(20, SLICE_COUNT)]
        for round_index, slice_indices in enumerate(thirds):
            _queue_read(
                client, destination, source, slice_indices, f"chunk{round_index}", "pytest-peer"
            )
            posted = client.post_cached_xfer()
            client.wait_posted_xfer(posted, timeout=120.0)

        assert torch.equal(destination, source), "multi-round transfer lost or misplaced bytes"
        cache = client._prepped_cache
        assert cache.submit_count == 3
        # Each round covers a different third of the buffer, so the dlist grows to the
        # union as rounds arrive: at most one re-prep per side per new round, and the
        # steady state afterwards is zero. What must not happen is a fresh prep per
        # submission, which would make this O(rounds) with no benefit over
        # initialize_xfer.
        assert cache.prep_count <= 6, (
            f"re-prepped {cache.prep_count} times across 3 rounds; the union dlist "
            f"should converge rather than growing without bound"
        )

        # Replaying the first round must now be a pure cache hit.
        preps_before_replay = cache.prep_count
        _queue_read(client, destination, source, thirds[0], "replay", "pytest-peer")
        client.wait_posted_xfer(client.post_cached_xfer(), timeout=120.0)
        assert cache.prep_count == preps_before_replay, "replaying a round re-prepped"

    def test_subset_round_leaves_other_slices_untouched(self, agent_pair, buffers):
        """A commit group transfers part of the tensor; the rest must not move."""
        local_agent, _peer = agent_pair
        source, destination = buffers
        client = _client(local_agent, "pytest-local")

        _queue_read(client, destination, source, range(4), "subset", "pytest-peer")
        client.wait_posted_xfer(client.post_cached_xfer(), timeout=120.0)

        transferred = 4 * SLICE_BYTES
        assert torch.equal(destination[:transferred], source[:transferred])
        assert torch.all(destination[transferred:] == 0), "wrote outside the submitted subset"

    def test_no_initialize_xfer_on_the_hot_path(self, agent_pair, buffers, monkeypatch):
        """The A1 criterion, measured against the real agent rather than a fake."""
        local_agent, _peer = agent_pair
        source, destination = buffers
        client = _client(local_agent, "pytest-local")

        calls: list[str] = []
        original = type(local_agent).initialize_xfer

        def tracked(self, *args, **kwargs):
            calls.append("initialize_xfer")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(type(local_agent), "initialize_xfer", tracked)

        for round_index in range(3):
            _queue_read(
                client, destination, source, range(SLICE_COUNT), f"r{round_index}", "pytest-peer"
            )
            client.wait_posted_xfer(client.post_cached_xfer(), timeout=120.0)

        assert calls == [], f"hot path called initialize_xfer {len(calls)} times, expected 0"
        assert torch.equal(destination, source)

    def test_reregistration_repreps_and_stays_correct(self, agent_pair, registered_buffer):
        """A moved buffer must invalidate the cached dlist, not read the old address.

        This is the local half of the hazard `registration_epoch` covers remotely: a
        prepped handle embeds addresses, so reusing it after re-registration would read
        whatever now occupies the old pages.
        """
        local_agent, _peer = agent_pair
        client = _client(local_agent, "pytest-local")

        first_source = registered_buffer(TOTAL_BYTES)
        first_destination = registered_buffer(TOTAL_BYTES)
        _fill(first_source)
        first_destination.zero_()
        _queue_read(client, first_destination, first_source, range(8), "a", "pytest-peer")
        client.wait_posted_xfer(client.post_cached_xfer(), timeout=120.0)
        assert torch.equal(first_destination[: 8 * SLICE_BYTES], first_source[: 8 * SLICE_BYTES])
        preps_after_first = client._prepped_cache.prep_count

        # New allocations: different addresses, so the cached dlist cannot serve them.
        second_source = registered_buffer(TOTAL_BYTES)
        second_destination = registered_buffer(TOTAL_BYTES)
        second_source.copy_(torch.full_like(second_source, 7))
        second_destination.zero_()
        _queue_read(client, second_destination, second_source, range(8), "b", "pytest-peer")
        client.wait_posted_xfer(client.post_cached_xfer(), timeout=120.0)

        assert torch.all(second_destination[: 8 * SLICE_BYTES] == 7), (
            "after re-registration the transfer read the wrong memory"
        )
        assert client._prepped_cache.prep_count > preps_after_first, "stale dlist was reused"


class TestPreppedSubmissionIsCheaper:
    """Where the prepped path actually wins, measured through PSRL's own cache.

    The benefit is asymptotic: `initialize_xfer` re-serialises N descriptors per call,
    while a prepped submission ships a cached index array. Below some crossover the
    fixed Python overhead of the cache lookup dominates and prepped is *slower* --
    measured 0.55x at 32 descriptors. That is fine, because the regime the design cares
    about is MoE scale (thousands of descriptors per group), but it has to be measured
    rather than assumed, and the crossover has to be known.
    """

    @staticmethod
    def _median_submit_times(local_agent, registered_buffer, count: int):
        """(prepped, initialize_xfer) median submission cost at ``count`` descriptors."""
        import time

        slice_bytes = 4096
        total = slice_bytes * count
        source = registered_buffer(total)
        destination = registered_buffer(total)

        local_descs = [
            (destination.data_ptr() + i * slice_bytes, slice_bytes, destination.device.index)
            for i in range(count)
        ]
        remote_descs = [
            (source.data_ptr() + i * slice_bytes, slice_bytes, source.device.index)
            for i in range(count)
        ]

        from psrl.utils.nixl.prepped_xfer import PreppedXferCache

        cache = PreppedXferCache(local_agent, "bench")
        # Warm: the first submission prepares the dlist and the index array.
        for _ in range(2):
            handle = cache.make_xfer(
                "READ", "pytest-peer", nixlBind.VRAM_SEG, local_descs,
                nixlBind.VRAM_SEG, remote_descs, b"warm",
            )
            local_agent.release_xfer_handle(handle)

        prepped = []
        for _ in range(30):
            start = time.perf_counter()
            handle = cache.make_xfer(
                "READ", "pytest-peer", nixlBind.VRAM_SEG, local_descs,
                nixlBind.VRAM_SEG, remote_descs, b"bench",
            )
            prepped.append(time.perf_counter() - start)
            local_agent.release_xfer_handle(handle)
        cache.clear()

        local_list = nixlBind.nixlXferDList(nixlBind.VRAM_SEG, local_descs)
        remote_list = nixlBind.nixlXferDList(nixlBind.VRAM_SEG, remote_descs)
        baseline = []
        for _ in range(30):
            start = time.perf_counter()
            handle = local_agent.initialize_xfer(
                "READ", local_list, remote_list, "pytest-peer", b"bench"
            )
            baseline.append(time.perf_counter() - start)
            local_agent.release_xfer_handle(handle)

        return sorted(prepped)[len(prepped) // 2], sorted(baseline)[len(baseline) // 2]

    def test_prepped_wins_at_moe_descriptor_counts(self, agent_pair, registered_buffer):
        """At the scale the design targets, prepped submission must be cheaper."""
        local_agent, _peer = agent_pair
        prepped, baseline = self._median_submit_times(local_agent, registered_buffer, 2048)
        speedup = baseline / prepped

        assert speedup > 2.0, (
            f"prepped submission is only {speedup:.2f}x cheaper at 2048 descriptors "
            f"({prepped * 1e3:.4f}ms vs {baseline * 1e3:.4f}ms); the whole point of the "
            f"prepped path is that it stops scaling with descriptor count"
        )

    def test_advantage_grows_with_descriptor_count(self, agent_pair, registered_buffer):
        """The shape of the win, which is what justifies the cache's complexity.

        Also records the crossover: prepped loses below a few hundred descriptors, so a
        caller batching tiny groups would be better off on the old path. The design's
        commit groups are far larger, so this is documentation rather than a problem.
        """
        local_agent, _peer = agent_pair
        speedups = {}
        for count in (64, 512, 4096):
            prepped, baseline = self._median_submit_times(local_agent, registered_buffer, count)
            speedups[count] = baseline / prepped

        assert speedups[4096] > speedups[512] > speedups[64], (
            f"prepped advantage must grow with descriptor count, got {speedups}"
        )
        assert speedups[4096] > 3.0, f"expected a clear win at 4096, got {speedups[4096]:.2f}x"
