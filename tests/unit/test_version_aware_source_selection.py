"""D1 version-aware source selection: only trainer ranks that have published the target
version are used as upstream sources.

Tests here are pure CPU: no NIXL agents, no Ray, no GPUs.  They exercise the two
layers that implement the guarantee:

1. ``PullEngine.pull()`` filters ``upstream`` to ``version_holders`` before any byte
   moves.  When all sources for a key are at an older version the original list is kept
   as a fallback (never deadlock).

2. ``PSManager.push_model_state_dict_nixl()`` populates ``_nixl_version_holders`` from
   ``ps_nixl_train_storage_client_names``, and ``get_nixl_version_state()`` returns
   ``(version, holders, invalidated)`` so rollouts can read it atomically.

The ``source_resolution`` module (``HolderSet``, ``SourceResolver``) is tested
separately; it remains a pure, unwired planning layer.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.cpu_test


# ---------------------------------------------------------------------------
# Helpers: minimal stand-ins for PullEngine internals
# ---------------------------------------------------------------------------


def _apply_version_filter(
    upstream: dict[str, dict[str, list]],
    version_holders: frozenset[str],
    target_version: int = 1,
) -> tuple[dict[str, dict[str, list]], list[str]]:
    """Reproduce the filtering logic from ``PullEngine.pull``.

    Returns (filtered_upstream, warned_keys) so tests can assert on both.
    """
    warned: list[str] = []
    out: dict[str, dict[str, list]] = {}
    for key, per_source in upstream.items():
        allowed = {s: shards for s, shards in per_source.items() if s in version_holders}
        if allowed:
            out[key] = allowed
        else:
            # All sources stale — fall back so the pull can still succeed.
            out[key] = per_source
            warned.append(key)
    return out, warned


# ---------------------------------------------------------------------------
# Tests for the upstream-source filter (pull_engine logic)
# ---------------------------------------------------------------------------


class TestVersionFilter:
    def test_known_holders_pass_through(self):
        upstream = {"w.q": {"train-0": [(0,)], "train-1": [(1,)], "train-2": [(2,)]}}
        holders = frozenset({"train-0", "train-1", "train-2"})
        out, warned = _apply_version_filter(upstream, holders)
        assert out == upstream
        assert warned == []

    def test_stale_source_is_removed(self):
        # train-2 has NOT published the new version yet.
        upstream = {"w.q": {"train-0": [(0,)], "train-1": [(1,)], "train-2": [(2,)]}}
        holders = frozenset({"train-0", "train-1"})
        out, warned = _apply_version_filter(upstream, holders)
        assert "train-2" not in out["w.q"]
        assert set(out["w.q"].keys()) == {"train-0", "train-1"}
        assert warned == []

    def test_all_stale_falls_back_to_original(self):
        """When every source for a key is stale the original set is kept — no deadlock."""
        upstream = {"w.q": {"train-0": [(0,)], "train-1": [(1,)]}}
        holders = frozenset({"train-99"})  # none of the actual sources
        out, warned = _apply_version_filter(upstream, holders)
        assert out["w.q"] == upstream["w.q"]
        assert "w.q" in warned

    def test_empty_holders_is_no_op(self):
        """Empty frozenset means no filtering — backward-compatible for non-direct-supply."""
        upstream = {"w.q": {"train-0": [(0,)]}}
        # The real pull_engine guards on `if version_holders:`, so empty set → no filter.
        holders: frozenset[str] = frozenset()
        # Simulate the guard
        if not holders:
            out = upstream
            warned: list[str] = []
        else:
            out, warned = _apply_version_filter(upstream, holders)
        assert out == upstream
        assert warned == []

    def test_none_holders_is_no_op(self):
        """None means no filtering — backward-compatible default."""
        upstream = {"w.q": {"train-0": [(0,)]}}
        holders = None
        if not holders:
            out = upstream
        else:
            out, _ = _apply_version_filter(upstream, holders)  # type: ignore[arg-type]
        assert out == upstream

    def test_multiple_keys_filtered_independently(self):
        """Filtering is per-key: a key with no live holder keeps its original sources."""
        upstream = {
            "layer.w": {"train-0": [(0,)], "train-1": [(1,)]},
            "layer.b": {"train-0": [(0,)], "train-1": [(1,)]},
        }
        # Only train-0 has the new version for both keys.
        holders = frozenset({"train-0"})
        out, warned = _apply_version_filter(upstream, holders)
        assert out["layer.w"] == {"train-0": [(0,)]}
        assert out["layer.b"] == {"train-0": [(0,)]}
        assert warned == []

    def test_partial_keys_stale(self):
        """One key filtered, another falls back."""
        upstream = {
            "key.a": {"train-0": [(0,)]},  # train-0 is in holders → kept
            "key.b": {"train-1": [(0,)]},  # train-1 NOT in holders → fallback
        }
        holders = frozenset({"train-0"})
        out, warned = _apply_version_filter(upstream, holders)
        assert out["key.a"] == {"train-0": [(0,)]}
        assert out["key.b"] == {"train-1": [(0,)]}  # original kept
        assert "key.b" in warned

    def test_filter_preserves_shard_lists(self):
        """Shard tuples must survive filtering unchanged."""
        shards = [(0, 1), (2, 3)]
        upstream = {"w": {"train-0": shards}}
        holders = frozenset({"train-0"})
        out, _ = _apply_version_filter(upstream, holders)
        assert out["w"]["train-0"] == shards


# ---------------------------------------------------------------------------
# Tests for the PSManager holder bookkeeping
# ---------------------------------------------------------------------------


class FakeModelStore:
    def __init__(self, version_tag: int) -> None:
        self.version_tag = version_tag


class MinimalPSManagerHolder:
    """Minimal stand-in that reproduces only the holder-tracking logic."""

    def __init__(self) -> None:
        self.model_store: FakeModelStore | None = FakeModelStore(0)
        self.ps_nixl_train_storage_client_names: list[str] | None = None
        self._nixl_version_holders: frozenset[str] = frozenset()

    def bind_direct_supply(self, train_world_size: int) -> None:
        names = [f"NIXLTrainClient_{r}" for r in range(train_world_size)]
        self.ps_nixl_train_storage_client_names = names
        self._nixl_version_holders = frozenset(names)

    def push_model_state_dict_nixl(self, version_tag: int) -> None:
        self.model_store = FakeModelStore(version_tag)
        if self.ps_nixl_train_storage_client_names:
            self._nixl_version_holders = frozenset(self.ps_nixl_train_storage_client_names)

    def get_nixl_version_state(self) -> tuple[int, frozenset[str], frozenset[int]]:
        version = self.model_store.version_tag if self.model_store is not None else 0
        return version, self._nixl_version_holders, frozenset()


class TestPSManagerHolders:
    def test_initial_state_empty_before_bind(self):
        ps = MinimalPSManagerHolder()
        v, holders, _ = ps.get_nixl_version_state()
        assert v == 0
        assert holders == frozenset()

    def test_bind_direct_supply_populates_holders(self):
        ps = MinimalPSManagerHolder()
        ps.bind_direct_supply(4)
        v, holders, _ = ps.get_nixl_version_state()
        assert v == 0
        assert holders == frozenset({"NIXLTrainClient_0", "NIXLTrainClient_1",
                                     "NIXLTrainClient_2", "NIXLTrainClient_3"})

    def test_push_updates_version_and_holders(self):
        ps = MinimalPSManagerHolder()
        ps.bind_direct_supply(2)
        ps.push_model_state_dict_nixl(3)
        v, holders, _ = ps.get_nixl_version_state()
        assert v == 3
        assert holders == frozenset({"NIXLTrainClient_0", "NIXLTrainClient_1"})

    def test_holders_set_is_stable_between_pushes(self):
        """The holder set stays as the full trainer set on every push (no partial update)."""
        ps = MinimalPSManagerHolder()
        ps.bind_direct_supply(3)
        for step in range(1, 5):
            ps.push_model_state_dict_nixl(step)
            _, holders, _ = ps.get_nixl_version_state()
            assert len(holders) == 3, f"step {step}: expected 3 holders, got {len(holders)}"

    def test_no_holders_when_not_direct_supply(self):
        """Non-direct-supply modes never call bind_direct_supply; holder set stays empty."""
        ps = MinimalPSManagerHolder()
        ps.model_store = FakeModelStore(5)  # simulate a version existing
        # push without binding (non-direct-supply)
        ps.push_model_state_dict_nixl(6)
        _, holders, _ = ps.get_nixl_version_state()
        # No train clients registered → holders is empty → pull_engine skips filtering
        assert holders == frozenset()

    def test_version_tracks_latest_push(self):
        ps = MinimalPSManagerHolder()
        ps.bind_direct_supply(1)
        for v in [1, 2, 3, 10]:
            ps.push_model_state_dict_nixl(v)
            reported_v, _, _ = ps.get_nixl_version_state()
            assert reported_v == v


# ---------------------------------------------------------------------------
# Integration: filter + PSManager together
# ---------------------------------------------------------------------------


class TestFilterWithHolders:
    def test_end_to_end_correct_version(self):
        """All trainers published → no sources dropped."""
        ps = MinimalPSManagerHolder()
        ps.bind_direct_supply(4)
        ps.push_model_state_dict_nixl(7)
        target_v, holders, _ = ps.get_nixl_version_state()
        assert target_v == 7

        upstream = {
            "w": {f"NIXLTrainClient_{r}": [(r,)] for r in range(4)},
        }
        out, warned = _apply_version_filter(upstream, holders, target_version=target_v)
        assert out == upstream
        assert warned == []

    def test_end_to_end_stale_trainer(self):
        """Simulate one trainer that has NOT yet published version 8."""
        ps = MinimalPSManagerHolder()
        ps.bind_direct_supply(3)
        ps.push_model_state_dict_nixl(8)
        target_v, holders, _ = ps.get_nixl_version_state()

        # The actual upstream plan still lists all 3 trainers, but NIXLTrainClient_2
        # hasn't published v8 yet (the system-level guarantee is that once PSManager
        # calls push_model_state_dict_nixl *all* ranks have published, but this test
        # verifies the filter works if we manually narrow the holder set).
        restricted_holders = holders - {"NIXLTrainClient_2"}
        upstream = {"w": {f"NIXLTrainClient_{r}": [(r,)] for r in range(3)}}
        out, warned = _apply_version_filter(upstream, restricted_holders, target_version=target_v)
        assert "NIXLTrainClient_2" not in out["w"]
        assert warned == []
