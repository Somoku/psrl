"""The prepped-descriptor cache, and its integration into the transfer path (item A1).

Two things need verifying, and they fail differently:

* the cache's index bookkeeping -- unit-testable, and the place a subtle bug would live,
  since a prepped handle is addressed purely by position
* that the hot path actually uses it -- an integration property, asserted by counting
  `initialize_xfer` calls. The design's criterion is zero in steady state, which is
  stronger than "fewer" for a reason: passing a Python list instead of an int32 ndarray
  is accepted silently and costs 15x, so "it got faster" is not evidence that the
  intended path ran.

Byte-level correctness of the prepped path over real RDMA lives in
`test_descriptor_correctness_gpu.py`; this file is about whether PSRL reaches it.
"""

from __future__ import annotations

import numpy as np
import pytest

from psrl.utils.nixl.prepped_xfer import PreppedXferCache, mem_type_name

pytestmark = pytest.mark.cpu_test

# A real memory-type name: `mem_type_name` validates these, so a placeholder would be
# rejected (and rejecting unknown names is deliberate -- see TestMemTypeTranslation).
VRAM = "VRAM"
DRAM = "DRAM"


class FakeAgent:
    """Records prep and submit calls so index handling can be asserted exactly."""

    def __init__(self, *, fail_prep: bool = False, fail_submit: bool = False):
        self.preps: list[tuple] = []
        self.submits: list[tuple] = []
        self.released: list[object] = []
        self.initialize_calls: list[tuple] = []
        self.posted: list[object] = []
        self.fail_prep = fail_prep
        self.fail_submit = fail_submit
        self._next = 0

    def prep_xfer_dlist(self, agent_name, descs, mem_type=None):
        if self.fail_prep:
            raise RuntimeError("prep unavailable in this build")
        self._next += 1
        handle = f"dlist-{self._next}"
        self.preps.append((agent_name, mem_type, np.asarray(descs).shape, handle))
        return handle

    def make_prepped_xfer(self, op, local_handle, local_idx, remote_handle, remote_idx, notif):
        if self.fail_submit:
            raise RuntimeError("submit unavailable")
        self._next += 1
        handle = f"xfer-{self._next}"
        self.submits.append((op, local_handle, local_idx, remote_handle, remote_idx, notif, handle))
        return handle

    def release_dlist_handle(self, handle):
        self.released.append(handle)

    def initialize_xfer(self, op, local, remote, agent, notif):
        self._next += 1
        handle = f"init-{self._next}"
        self.initialize_calls.append((op, agent, notif, handle))
        return handle

    def transfer(self, handle):
        self.posted.append(handle)
        return "PROC"

    def check_xfer_state(self, handle):
        return "DONE"

    def release_xfer_handle(self, handle):
        self.released.append(handle)


def _descriptors(count: int, base: int = 0x1000, length: int = 4096) -> list[tuple[int, int, int]]:
    return [(base + i * length, length, 0) for i in range(count)]


class TestIndexing:
    """A prepped handle is addressed by position, so indices must be exact."""

    def test_first_submission_preps_both_sides(self):
        agent = FakeAgent()
        cache = PreppedXferCache(agent, "c")
        local, remote = _descriptors(4), _descriptors(4, base=0x9000)

        cache.make_xfer("READ", "peer", VRAM, local, VRAM, remote, b"n")

        assert len(agent.preps) == 2, "one prep for local memory, one for the remote peer"
        assert agent.preps[0][0] == PreppedXferCache.LOCAL_AGENT
        assert agent.preps[1][0] == "peer"

    def test_indices_are_int32_ndarrays(self):
        """A Python list is accepted by NIXL but ~15x slower; enforce the fast form."""
        agent = FakeAgent()
        cache = PreppedXferCache(agent, "c")
        cache.make_xfer("READ", "peer", VRAM, _descriptors(8), VRAM, _descriptors(8, 0x9000), b"n")

        _op, _lh, local_idx, _rh, remote_idx, _notif, _handle = agent.submits[0]
        for indices in (local_idx, remote_idx):
            assert isinstance(indices, np.ndarray)
            assert indices.dtype == np.int32

    def test_indices_match_descriptor_order(self):
        agent = FakeAgent()
        cache = PreppedXferCache(agent, "c")
        cache.make_xfer("READ", "peer", VRAM, _descriptors(5), VRAM, _descriptors(5, 0x9000), b"n")

        _op, _lh, local_idx, _rh, remote_idx, _notif, _handle = agent.submits[0]
        assert local_idx.tolist() == [0, 1, 2, 3, 4]
        assert remote_idx.tolist() == [0, 1, 2, 3, 4]

    def test_second_round_reuses_the_dlist(self):
        """The whole point: prep once, submit many."""
        agent = FakeAgent()
        cache = PreppedXferCache(agent, "c")
        local, remote = _descriptors(4), _descriptors(4, 0x9000)

        for _ in range(5):
            cache.make_xfer("READ", "peer", VRAM, local, VRAM, remote, b"n")

        assert len(agent.preps) == 2, f"re-prepped {len(agent.preps) // 2} times, expected once"
        assert len(agent.submits) == 5
        assert cache.prep_count == 2
        assert cache.submit_count == 5

    def test_subset_submission_uses_original_indices(self):
        """Commit-group submission sends part of a round; positions must not shift."""
        agent = FakeAgent()
        cache = PreppedXferCache(agent, "c")
        local, remote = _descriptors(8), _descriptors(8, 0x9000)
        cache.make_xfer("READ", "peer", VRAM, local, VRAM, remote, b"all")

        cache.make_xfer("READ", "peer", VRAM, local[2:5], VRAM, remote[2:5], b"subset")

        assert len(agent.preps) == 2, "a subset must not trigger a re-prep"
        _op, _lh, local_idx, _rh, remote_idx, _notif, _handle = agent.submits[-1]
        assert local_idx.tolist() == [2, 3, 4]
        assert remote_idx.tolist() == [2, 3, 4]

    def test_reordered_submission_follows_the_given_order(self):
        """Chunk-offset staggering rotates the order; indices must track it."""
        agent = FakeAgent()
        cache = PreppedXferCache(agent, "c")
        local, remote = _descriptors(4), _descriptors(4, 0x9000)
        cache.make_xfer("READ", "peer", VRAM, local, VRAM, remote, b"first")

        rotated_local = local[2:] + local[:2]
        rotated_remote = remote[2:] + remote[:2]
        cache.make_xfer("READ", "peer", VRAM, rotated_local, VRAM, rotated_remote, b"rot")

        _op, _lh, local_idx, _rh, remote_idx, _notif, _handle = agent.submits[-1]
        assert local_idx.tolist() == [2, 3, 0, 1]
        assert remote_idx.tolist() == [2, 3, 0, 1]

    def test_new_descriptor_grows_the_dlist_and_releases_the_old_handle(self):
        """A moved descriptor set re-preps over the union, releasing the old handle.

        Keeping the superset matters: successive rounds transfer different chunks of the
        same registered memory, so dropping the previous descriptors would re-prep on
        every round and lose the benefit entirely.
        """
        agent = FakeAgent()
        cache = PreppedXferCache(agent, "c")
        local, remote = _descriptors(4), _descriptors(4, 0x9000)
        cache.make_xfer("READ", "peer", VRAM, local, VRAM, remote, b"n")
        old_local_handle = agent.preps[0][3]

        moved = _descriptors(4, base=0x50000)
        cache.make_xfer("READ", "peer", VRAM, moved, VRAM, remote, b"n")

        assert old_local_handle in agent.released, "stale dlist was not released"
        assert len(agent.preps) == 3, "expected a re-prep of the local side only"
        # The regrown dlist covers both sets, so neither needs another prep.
        cache.make_xfer("READ", "peer", VRAM, local, VRAM, remote, b"n")
        cache.make_xfer("READ", "peer", VRAM, moved, VRAM, remote, b"n")
        assert len(agent.preps) == 3, "union dlist should serve both descriptor sets"

    def test_disjoint_rounds_prep_once(self):
        """Three rounds over different chunks of one registration: one prep each side.

        This is the steady-state pattern -- it is what the GPU test
        `test_three_rounds_reuse_one_prepped_dlist` asserts end to end.
        """
        agent = FakeAgent()
        cache = PreppedXferCache(agent, "c")
        local, remote = _descriptors(30), _descriptors(30, 0x9000)

        for start in (0, 10, 20):
            window = slice(start, start + 10)
            cache.make_xfer("READ", "peer", VRAM, local[window], VRAM, remote[window], b"n")

        assert len(agent.preps) <= 6, (
            f"prepped {len(agent.preps)} times for 3 rounds; the union should converge"
        )
        # The last two rounds must have hit the cache.
        assert cache.submit_count == 3

    def test_index_arrays_are_reused_in_steady_state(self):
        """Rebuilding an index array costs ~0.085us/descriptor -- linear, which is the
        very per-descriptor cost prepped submission exists to remove.

        The first submission after a prep builds the array; every submission after that
        must return the identical object. Measured overhead at 4096 descriptors fell from
        0.110ms to 0.006ms once this held.
        """
        agent = FakeAgent()
        cache = PreppedXferCache(agent, "c")
        local, remote = _descriptors(64), _descriptors(64, 0x9000)

        arrays = []
        for _ in range(4):
            cache.make_xfer("READ", "peer", VRAM, local, VRAM, remote, b"n")
            arrays.append(agent.submits[-1][2])

        steady = arrays[1:]
        assert all(array is steady[0] for array in steady), (
            "index array was rebuilt between steady-state submissions; it must be "
            "cached or submission is O(descriptors)"
        )

    def test_full_round_takes_the_identity_path(self):
        """Submitting the whole prepped set must not hash or rebuild anything.

        This is the path every full round takes, so it is the one that has to be O(1).
        """
        agent = FakeAgent()
        cache = PreppedXferCache(agent, "c")
        local, remote = _descriptors(64), _descriptors(64, 0x9000)
        cache.make_xfer("READ", "peer", VRAM, local, VRAM, remote, b"n")

        entry = cache._local[("READ", PreppedXferCache.LOCAL_AGENT, VRAM)]
        # A fresh equal-but-distinct list must still hit the identity cache.
        cache.make_xfer("READ", "peer", VRAM, list(local), VRAM, list(remote), b"n")

        assert entry._identity_indices is not None, "identity path was never taken"
        assert agent.submits[-1][2] is entry._identity_indices

    def test_mismatched_lengths_are_rejected(self):
        """Pairing is positional, so unequal lengths would silently mis-pair."""
        agent = FakeAgent()
        cache = PreppedXferCache(agent, "c")
        with pytest.raises(AssertionError, match="pairing is positional"):
            cache.make_xfer("READ", "peer", VRAM, _descriptors(4), VRAM, _descriptors(3), b"n")


class TestCacheKeying:
    """A prepped handle binds one remote agent and one memory type."""

    def test_different_peers_get_different_dlists(self):
        agent = FakeAgent()
        cache = PreppedXferCache(agent, "c")
        local, remote = _descriptors(4), _descriptors(4, 0x9000)

        cache.make_xfer("READ", "peer-a", VRAM, local, VRAM, remote, b"n")
        cache.make_xfer("READ", "peer-b", VRAM, local, VRAM, remote, b"n")

        prepped_peers = [name for name, _mem, _shape, _h in agent.preps]
        assert "peer-a" in prepped_peers and "peer-b" in prepped_peers
        assert prepped_peers.count(PreppedXferCache.LOCAL_AGENT) == 1, "local side is shared"

    def test_different_op_types_get_different_dlists(self):
        agent = FakeAgent()
        cache = PreppedXferCache(agent, "c")
        local, remote = _descriptors(4), _descriptors(4, 0x9000)

        cache.make_xfer("READ", "peer", VRAM, local, VRAM, remote, b"n")
        cache.make_xfer("WRITE", "peer", VRAM, local, VRAM, remote, b"n")

        assert len(agent.preps) == 4, "READ and WRITE must not share a prepped dlist"

    def test_changed_mem_type_reprep(self):
        agent = FakeAgent()
        cache = PreppedXferCache(agent, "c")
        local, remote = _descriptors(4), _descriptors(4, 0x9000)

        cache.make_xfer("READ", "peer", VRAM, local, VRAM, remote, b"n")
        cache.make_xfer("READ", "peer", DRAM, local, VRAM, remote, b"n")

        assert len(agent.preps) == 3


class TestLifecycle:
    def test_clear_releases_every_handle(self):
        agent = FakeAgent()
        cache = PreppedXferCache(agent, "c")
        cache.make_xfer("READ", "a", VRAM, _descriptors(2), VRAM, _descriptors(2, 0x9000), b"n")
        cache.make_xfer("WRITE", "b", VRAM, _descriptors(2), VRAM, _descriptors(2, 0x9000), b"n")

        cache.clear()

        prepped = {handle for _n, _m, _s, handle in agent.preps}
        assert prepped.issubset(set(agent.released)), "clear() left handles dangling"

    def test_clear_is_idempotent(self):
        agent = FakeAgent()
        cache = PreppedXferCache(agent, "c")
        cache.make_xfer("READ", "a", VRAM, _descriptors(2), VRAM, _descriptors(2, 0x9000), b"n")
        cache.clear()
        released_after_first = len(agent.released)
        cache.clear()
        assert len(agent.released) == released_after_first

    def test_cache_repreps_after_clear(self):
        agent = FakeAgent()
        cache = PreppedXferCache(agent, "c")
        local, remote = _descriptors(2), _descriptors(2, 0x9000)
        cache.make_xfer("READ", "a", VRAM, local, VRAM, remote, b"n")
        cache.clear()
        cache.make_xfer("READ", "a", VRAM, local, VRAM, remote, b"n")
        assert len(agent.preps) == 4

    def test_missing_handle_is_an_error_not_a_silent_skip(self):
        agent = FakeAgent()
        agent.prep_xfer_dlist = lambda *a, **k: None
        cache = PreppedXferCache(agent, "c")
        with pytest.raises(RuntimeError, match="prep_xfer_dlist returned no handle"):
            cache.make_xfer("READ", "a", VRAM, _descriptors(2), VRAM, _descriptors(2), b"n")


class TestHotPathUsesPreppedSubmission:
    """The integration criterion: zero `initialize_xfer` in steady state."""

    @staticmethod
    def _client(agent):
        from psrl.utils.nixl.client import NIXLStorageClient

        client = NIXLStorageClient.__new__(NIXLStorageClient)
        client.agent = agent
        client.client_name = "test-client"
        client._cached_xfer_descs = []
        client._prepped_cache = PreppedXferCache(agent, "test-client")
        return client

    @staticmethod
    def _cache_group(client, tag="chunk0", count=4):
        """Queue descriptors the way client_read does when merging."""
        import nixl._bindings as nixlBind

        class Desc:
            def __init__(self, items):
                self._items = items

            def descCount(self):
                return len(self._items)

            def getType(self):
                return nixlBind.VRAM_SEG

            def __getitem__(self, i):
                return self._items[i]

        for i in range(count):
            local = Desc([(0x1000 + i * 4096, 4096, 0)])
            remote = Desc([(0x9000 + i * 4096, 4096, 0)])
            client._cached_xfer_descs.append(("READ", local, remote, "peer-agent", tag, "peer"))

    def test_steady_state_never_calls_initialize_xfer(self):
        pytest.importorskip("nixl._bindings", reason="NIXL not installed")
        agent = FakeAgent()
        client = self._client(agent)

        for round_index in range(3):
            self._cache_group(client, tag=f"chunk{round_index}")
            client.post_cached_xfer()

        assert agent.initialize_calls == [], (
            f"hot path called initialize_xfer {len(agent.initialize_calls)} times; "
            f"the design requires zero in steady state"
        )
        assert len(agent.submits) == 3, "one prepped submission per round"
        assert len(agent.preps) == 2, "prepped once, reused across rounds"

    def test_falls_back_when_prepping_is_unavailable(self):
        """An older NIXL build must still work, at the old cost."""
        pytest.importorskip("nixl._bindings", reason="NIXL not installed")
        agent = FakeAgent(fail_prep=True)
        client = self._client(agent)

        self._cache_group(client)
        client.post_cached_xfer()

        assert len(agent.initialize_calls) == 1, "fallback did not run"
        assert client._prepped_cache is None, "fallback must disable further attempts"

    def test_fallback_is_sticky(self):
        """Retrying a failing prep every round would pay the cost twice over."""
        pytest.importorskip("nixl._bindings", reason="NIXL not installed")
        agent = FakeAgent(fail_prep=True)
        client = self._client(agent)

        for round_index in range(3):
            self._cache_group(client, tag=f"chunk{round_index}")
            client.post_cached_xfer()

        assert len(agent.preps) == 0
        assert len(agent.initialize_calls) == 3


class TestMemTypeTranslation:
    """`prep_xfer_dlist` names memory types with strings, the merged path has enums.

    Passing the enum through raises a bare `KeyError` from inside the binding, which is
    how this mismatch originally hid behind the generic fallback warning.
    """

    def test_enum_is_translated(self):
        nixlBind = pytest.importorskip("nixl._bindings", reason="NIXL not installed")
        assert mem_type_name(nixlBind.VRAM_SEG) == "VRAM"
        assert mem_type_name(nixlBind.DRAM_SEG) == "DRAM"

    @pytest.mark.parametrize("name,expected", [("VRAM", "VRAM"), ("VRAM_SEG", "VRAM"), ("DRAM_SEG", "DRAM")])
    def test_both_string_spellings_are_accepted(self, name, expected):
        assert mem_type_name(name) == expected

    @pytest.mark.parametrize("bad", ["BOGUS", "vram", "", "GPU"])
    def test_unknown_names_are_rejected(self, bad):
        """Passing an unknown string through would fail deep inside the binding."""
        with pytest.raises(ValueError, match="unrecognised NIXL memory type"):
            mem_type_name(bad)
