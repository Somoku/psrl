"""Unit tests for `make_xfer_tag` (implementation item A4).

`make_xfer_tag` runs once per (key, shard) descriptor, so at MoE scale it executes
~140k times per weight sync. It used to call ``pickle.dumps``; these tests pin both
the injectivity it must preserve and the speedup that motivated the change.

Injectivity matters because the tag is the key of ``NIXLStorageClient.xfer_handles``:
two distinct transfers colliding on one tag would silently make one of them wait on
the other's handle.
"""

from __future__ import annotations

import time

import pytest

from psrl.utils.nixl.client import make_xfer_tag

# Descriptor count for a Kimi-K2-class MoE after HF expansion (design doc 3.5.5).
MOE_DESCRIPTOR_COUNT = 140_000


def test_distinct_inputs_give_distinct_tags():
    """Every field must participate in the tag, and no two inputs may collide."""
    cases = [
        ("t", "a", "b", "k"),
        ("t2", "a", "b", "k"),  # tag differs
        ("t", "a2", "b", "k"),  # src differs
        ("t", "a", "b2", "k"),  # target differs
        ("t", "a", "b", "k2"),  # key differs
    ]
    tags = {make_xfer_tag(*c) for c in cases}
    assert len(tags) == len(cases)


def test_shard_index_tuples_do_not_collide():
    """(1, 2) and (12,) must differ -- a naive concat without a separator would not."""
    assert make_xfer_tag("t", "a", "b", "k", (1, 2)) != make_xfer_tag("t", "a", "b", "k", (12,))
    assert make_xfer_tag("t", "a", "b", "k", (0, 1)) != make_xfer_tag("t", "a", "b", "k", (1, 0))
    # Absent vs present-but-zero are different transfers.
    assert make_xfer_tag("t", "a", "b", "k") != make_xfer_tag("t", "a", "b", "k", (0,))


def test_none_target_client_does_not_collide_with_empty_string():
    """`wait()` types target_client as `str | None`, so None is reachable here."""
    assert make_xfer_tag("t", "a", None, "k") != make_xfer_tag("t", "a", "", "k")


def test_single_and_multi_shard_paths_agree_on_uniqueness():
    """The len==1 fast path must not collide with the general join path."""
    seen: dict[bytes, tuple] = {}
    for key in ("layers.0.mlp.w1", "layers.0.mlp.w13", "merged_xfer_for_t"):
        for shard in (None, (0,), (1,), (10,), (0, 1), (1, 0), (1, 0, 1)):
            for target in ("c_b", None, ""):
                args = ("train_push_3", "c_a", target, key, shard)
                tag = make_xfer_tag(*args)
                assert tag not in seen, f"collision: {args} vs {seen[tag]}"
                seen[tag] = args


def test_returns_bytes():
    """NIXL's notif API takes bytes; returning str would fail deep in the C++ layer."""
    assert isinstance(make_xfer_tag("t", "a", "b", "k", (0,)), bytes)


@pytest.mark.parametrize("shard", [None, (3,), (3, 4)])
def test_stable_across_calls(shard):
    """The tag is used to look a handle back up, so it must be deterministic."""
    assert make_xfer_tag("t", "a", "b", "k", shard) == make_xfer_tag("t", "a", "b", "k", shard)


def test_moe_scale_construction_cost():
    """Tag construction must not be a visible cost at MoE descriptor counts.

    Budget is 40ms for 140k tags. The pickle implementation measured ~55ms and this
    one ~20ms on an H20 node; 40ms leaves headroom for slower CPUs while still
    failing if someone reintroduces a per-call serializer.
    """
    keys = [f"layers.{i % 61}.experts.{i}.w1" for i in range(MOE_DESCRIPTOR_COUNT)]
    shards = [(i % 8,) for i in range(MOE_DESCRIPTOR_COUNT)]

    start = time.perf_counter()
    for i in range(MOE_DESCRIPTOR_COUNT):
        make_xfer_tag("train_push_3", "client_a", "client_b", keys[i], shards[i])
    elapsed_ms = (time.perf_counter() - start) * 1e3

    assert elapsed_ms < 40.0, f"{MOE_DESCRIPTOR_COUNT} tags took {elapsed_ms:.0f}ms, budget 40ms"
