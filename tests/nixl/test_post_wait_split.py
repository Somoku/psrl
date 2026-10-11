"""Tests for the post/wait split of merged transfers (implementation item A3).

The property under test is structural, not numerical: `post_cached_xfer` must hand
control back while transfers are still in flight, so a caller can post round i+1
before round i has landed. Without that, `post(i) -> wait(i) -> post(i+1)` is forced
and the design's depth-2 double buffer cannot overlap anything (design doc 3.6.4).

These tests drive a fake agent rather than real NIXL: the ordering guarantee lives in
PSRL's control flow, so a stub that records call order verifies it exactly, and it
runs anywhere. Bandwidth and real completion latency belong to the multi-node tests.
"""

from __future__ import annotations

import nixl._bindings as nixlBind
import pytest

from psrl.utils.nixl.client import NIXLStorageClient, PostedXfer

# The merged-transfer path builds a real nixlXferDList from the cached descriptors, so
# the memory type must be a genuine nixl_mem_t even though the agent itself is faked.
VRAM = nixlBind.VRAM_SEG


class FakeDescList:
    """Stands in for nixlXferDList: one descriptor, a memory type, a payload."""

    def __init__(self, mem_type=VRAM, descs=None):
        self._mem_type = mem_type
        self._descs = descs if descs is not None else [(0x1000, 4096, 0)]

    def descCount(self) -> int:
        return len(self._descs)

    def getType(self):
        return self._mem_type

    def __getitem__(self, i):
        return self._descs[i]


class FakeAgent:
    """Records the order of initialize/transfer/check so ordering can be asserted.

    ``completions_needed`` controls how many ``check_xfer_state`` calls a handle takes
    to report DONE, which lets a test keep a transfer "in flight" deterministically.
    """

    def __init__(self, completions_needed: int = 1):
        self.calls: list[tuple[str, object]] = []
        self.completions_needed = completions_needed
        self._checks: dict[object, int] = {}
        self._next_handle = 0

    def initialize_xfer(self, op_type, local, remote, agent, tag):
        self._next_handle += 1
        handle = f"handle-{self._next_handle}"
        self.calls.append(("initialize", handle))
        return handle

    def transfer(self, handle):
        self.calls.append(("transfer", handle))
        return "PROC"

    def check_xfer_state(self, handle):
        self._checks[handle] = self._checks.get(handle, 0) + 1
        self.calls.append(("check", handle))
        return "DONE" if self._checks[handle] >= self.completions_needed else "PROC"


def _client(agent: FakeAgent) -> NIXLStorageClient:
    """A client with just the attributes the merged-transfer path touches."""
    client = NIXLStorageClient.__new__(NIXLStorageClient)
    client.agent = agent
    client.client_name = "test-client"
    client._cached_xfer_descs = []
    return client


def _cache_one(client, tag="chunk0", op_type="READ", target="peer"):
    client._cached_xfer_descs.append(
        (op_type, FakeDescList(), FakeDescList(), "peer-agent", tag, target)
    )


class TestPostReturnsWithoutWaiting:
    """The core A3 guarantee."""

    def test_post_does_not_poll_for_completion(self):
        agent = FakeAgent()
        client = _client(agent)
        _cache_one(client)

        client.post_cached_xfer()

        kinds = [kind for kind, _ in agent.calls]
        assert "check" not in kinds, (
            "post_cached_xfer polled for completion; it must return while the "
            "transfer is still in flight or no pipelining is possible"
        )
        assert kinds == ["initialize", "transfer"]

    def test_second_post_precedes_first_wait(self):
        """The design's double buffer requires exactly this interleaving."""
        agent = FakeAgent()
        client = _client(agent)

        _cache_one(client, tag="chunk0")
        first = client.post_cached_xfer()
        _cache_one(client, tag="chunk1")
        second = client.post_cached_xfer()

        # Both are in flight before anything is awaited.
        posted_before_any_check = [k for k, _ in agent.calls if k == "transfer"]
        assert len(posted_before_any_check) == 2
        assert "check" not in [k for k, _ in agent.calls]

        client.wait_posted_xfer(first)
        client.wait_posted_xfer(second)
        assert [k for k, _ in agent.calls].count("check") == 2

    def test_post_returns_handles_for_every_group(self):
        agent = FakeAgent()
        client = _client(agent)
        _cache_one(client, tag="chunk0")
        _cache_one(client, tag="chunk1")

        posted = client.post_cached_xfer()

        assert len(posted) == 2
        assert all(isinstance(entry, PostedXfer) for entry in posted)
        assert {entry.tag for entry in posted} == {"chunk0", "chunk1"}

    def test_cache_is_drained_by_post(self):
        """Re-posting the same descriptors would duplicate the transfer."""
        agent = FakeAgent()
        client = _client(agent)
        _cache_one(client)

        client.post_cached_xfer()
        assert client._cached_xfer_descs == []
        assert client.post_cached_xfer() == []


class TestMerging:
    """Descriptors sharing (op_type, target, tag) must collapse into one NIXL op."""

    def test_same_tag_merges_into_one_transfer(self):
        agent = FakeAgent()
        client = _client(agent)
        for _ in range(4):
            _cache_one(client, tag="chunk0")

        posted = client.post_cached_xfer()

        assert len(posted) == 1
        assert posted[0].desc_count == 4
        assert [k for k, _ in agent.calls].count("initialize") == 1

    def test_different_op_types_stay_separate(self):
        agent = FakeAgent()
        client = _client(agent)
        _cache_one(client, tag="t", op_type="READ")
        _cache_one(client, tag="t", op_type="WRITE")

        posted = client.post_cached_xfer()

        assert {entry.op_type for entry in posted} == {"READ", "WRITE"}

    def test_different_targets_stay_separate(self):
        agent = FakeAgent()
        client = _client(agent)
        _cache_one(client, tag="t", target="peer-a")
        _cache_one(client, tag="t", target="peer-b")

        posted = client.post_cached_xfer()

        assert {entry.target_client for entry in posted} == {"peer-a", "peer-b"}


class TestWaitSemantics:
    def test_wait_polls_until_done(self):
        agent = FakeAgent(completions_needed=3)
        client = _client(agent)
        _cache_one(client)

        client.wait_posted_xfer(client.post_cached_xfer())

        assert [k for k, _ in agent.calls].count("check") == 3

    def test_wait_on_empty_list_is_a_noop(self):
        agent = FakeAgent()
        client = _client(agent)
        client.wait_posted_xfer([])
        assert agent.calls == []

    def test_transfer_error_is_raised_at_post(self):
        agent = FakeAgent()
        agent.transfer = lambda handle: "ERR"
        client = _client(agent)
        _cache_one(client)

        with pytest.raises(RuntimeError, match="posting client"):
            client.post_cached_xfer()

    def test_completion_error_is_raised_at_wait(self):
        agent = FakeAgent()
        client = _client(agent)
        _cache_one(client)
        posted = client.post_cached_xfer()
        agent.check_xfer_state = lambda handle: "ERR"

        with pytest.raises(RuntimeError, match="Merged transfer error"):
            client.wait_posted_xfer(posted)

    def test_timeout_names_the_transfer(self):
        agent = FakeAgent()
        client = _client(agent)
        _cache_one(client, tag="chunk7")
        posted = client.post_cached_xfer()
        agent.check_xfer_state = lambda handle: "PROC"

        with pytest.raises(TimeoutError, match="chunk7"):
            client.wait_posted_xfer(posted, timeout=0.0)


class TestBackwardCompatibleComposition:
    """`merge_and_finish_cached_xfer` must keep working for non-pipelined callers."""

    def test_combined_helper_posts_and_waits(self):
        agent = FakeAgent()
        client = _client(agent)
        _cache_one(client)

        client.merge_and_finish_cached_xfer()

        kinds = [k for k, _ in agent.calls]
        assert kinds == ["initialize", "transfer", "check"]
        assert client._cached_xfer_descs == []

    def test_combined_helper_tolerates_empty_cache(self):
        agent = FakeAgent()
        client = _client(agent)
        client.merge_and_finish_cached_xfer()
        assert agent.calls == []

    def test_works_when_cache_attribute_is_absent(self):
        """client_read only creates _cached_xfer_descs when merging is requested."""
        agent = FakeAgent()
        client = NIXLStorageClient.__new__(NIXLStorageClient)
        client.agent = agent
        client.client_name = "c"

        client.merge_and_finish_cached_xfer()
        assert agent.calls == []


class TestStridedShardsCanBeMerged:
    """Multi-descriptor entries must merge, not be rejected (items A1/A2 + D1).

    A strided shard expands to one descriptor per row (item A2), so a cached entry
    legitimately holds thousands of descriptors -- the 5-node Qwen3-8B run registers
    "144 strided shard(s) in place" at 4096 descriptors each. The merge path used to
    assert `descCount() == 1`, which was not a NIXL limit but a guard against the
    accumulation silently keeping only `desc[0]`.

    Why it matters: without merging, a client whose shards are strided can only submit
    one read at a time, and the cluster breakdown put *all* the remaining cost there --
    post_peer 4.105s against wait_peer 0.007s. The bytes are free; the submissions are
    the bill.
    """

    def _cache_strided(self, client, count, tag="chunk0", base=0x1000):
        descs = [(base + i * 4096, 4096, 0) for i in range(count)]
        client._cached_xfer_descs.append(
            (
                "READ",
                FakeDescList(descs=list(descs)),
                FakeDescList(descs=list(descs)),
                "peer-agent",
                tag,
                "peer",
            )
        )

    def test_a_multi_descriptor_entry_is_accepted(self):
        agent = FakeAgent()
        client = _client(agent)
        self._cache_strided(client, 4096)

        posted = client.post_cached_xfer()

        assert len(posted) == 1
        assert posted[0].desc_count == 4096

    def test_every_descriptor_survives_the_merge(self):
        """The old accumulation took `desc[0]` only; dropping rows loses bytes silently."""
        agent = FakeAgent()
        client = _client(agent)
        self._cache_strided(client, 8, tag="chunk0", base=0x1000)
        self._cache_strided(client, 8, tag="chunk0", base=0x9000)

        posted = client.post_cached_xfer()

        assert len(posted) == 1
        # 16, not 2: a merge that kept one descriptor per entry would transfer 2 rows
        # and silently leave 14 unwritten.
        assert posted[0].desc_count == 16

    def test_many_strided_reads_become_one_submission(self):
        """This is the whole point: N posts collapse to one."""
        agent = FakeAgent()
        client = _client(agent)
        for i in range(32):
            self._cache_strided(client, 128, tag="chunk0", base=0x1000 + i * 0x100000)

        posted = client.post_cached_xfer()

        assert len(posted) == 1
        assert posted[0].desc_count == 32 * 128
        assert [k for k, _ in agent.calls].count("initialize") == 1

    def test_mismatched_counts_are_still_rejected(self):
        """NIXL pairs descriptors element-wise, so unequal sides cannot be merged.

        `_align_desc_counts` equalises them at cache time, so reaching here unequal
        means that invariant broke -- which must surface rather than transfer a prefix.
        """
        agent = FakeAgent()
        client = _client(agent)
        client._cached_xfer_descs.append(
            (
                "READ",
                FakeDescList(descs=[(0x1000, 4096, 0), (0x2000, 4096, 0)]),
                FakeDescList(descs=[(0x1000, 4096, 0)]),
                "peer-agent",
                "chunk0",
                "peer",
            )
        )

        with pytest.raises(AssertionError, match="descriptor counts must match"):
            client.post_cached_xfer()
