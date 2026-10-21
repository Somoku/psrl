"""Conversion of LMCache placement events into SMG's tiered KV-event index."""

import pytest
from psrl.workers.gen.lmcache_kv_events import LMCacheKvEventBridge, _stable_block_hash


def _key(chunk_hash: str, model: str = "model", kv_rank: int = 1) -> dict:
    return {
        "chunk_hash_hex": chunk_hash,
        "model_name": model,
        "kv_rank": kv_rank,
        "object_group_id": 0,
        "cache_salt": "",
        "tags": [],
    }


def _store_batch(entries: list[dict], event_type: str = "store") -> dict:
    return {
        "batches": [
            {
                "instance_id": "node",
                "incarnation": 1,
                "seq": 1,
                "event_type": event_type,
                "tier": "l1",
                "backend": "dram",
                "entries": entries,
                "shared": False,
                "ts": 0.0,
            }
        ]
    }


def _entry(chunk_hash: str, offset: int, tokens: int = 256, model: str = "model", version: str | None = None) -> dict:
    key = _key(chunk_hash, model=model)
    if version is not None:
        key["tags"] = [["lmcache.tag.model_version", version]]
    return {
        "key": key,
        "size_bytes": 1024,
        "token_ids": list(range(tokens)),
        "token_offset": offset,
    }


def test_store_chunks_land_at_their_own_positions():
    bridge = LMCacheKvEventBridge("http://127.0.0.1:1")
    events, next_id = bridge.convert(_store_batch([_entry("aa" * 32, 0), _entry("bb" * 32, 256)]), 0)

    assert next_id == 1
    assert len(events) == 1
    stored = events[0].stored
    assert len(stored.blocks) == 2
    assert [block.block_size for block in stored.blocks] == [256, 256]
    assert [block.token_ids[0] for block in stored.blocks] == [0, 0]
    assert all(block.cache_level == 1 for block in stored.blocks)
    assert not stored.HasField("parent_block_hash")


def test_a_later_turn_chains_onto_the_stored_prefix():
    bridge = LMCacheKvEventBridge("http://127.0.0.1:1")
    bridge.convert(_store_batch([_entry("aa" * 32, 0), _entry("bb" * 32, 256)]), 0)
    events, _ = bridge.convert(_store_batch([_entry("cc" * 32, 512)]), 2)

    stored = events[0].stored
    assert stored.parent_block_hash == _stable_block_hash("bb" * 32)
    assert stored.blocks[0].block_hash == _stable_block_hash("cc" * 32)


def test_a_delete_maps_the_chunk_back_to_the_stored_block():
    bridge = LMCacheKvEventBridge("http://127.0.0.1:1")
    bridge.convert(_store_batch([_entry("aa" * 32, 0)]), 0)
    events, _ = bridge.convert(_store_batch([{"key": _key("aa" * 32)}], event_type="delete"), 1)

    removed = events[0].removed
    assert list(removed.block_hashes) == [_stable_block_hash("aa" * 32)]
    assert removed.cache_level == 1


def test_entries_without_tokens_are_skipped():
    bridge = LMCacheKvEventBridge("http://127.0.0.1:1")
    entry = _entry("aa" * 32, 0)
    entry["token_ids"] = []
    events, next_id = bridge.convert(_store_batch([entry]), 0)

    assert events == []
    assert next_id == 0


def test_a_gap_in_offsets_splits_the_parent_chain():
    bridge = LMCacheKvEventBridge("http://127.0.0.1:1")
    events, _ = bridge.convert(_store_batch([_entry("aa" * 32, 0), _entry("cc" * 32, 768)]), 0)

    assert len(events) == 2
    assert not events[0].stored.HasField("parent_block_hash")
    assert not events[1].stored.HasField("parent_block_hash")


def test_config_and_access_batches_carry_no_placement():
    bridge = LMCacheKvEventBridge("http://127.0.0.1:1")
    assert bridge.convert({"batches": [{"event_type": "config", "entries": []}]}, 0) == ([], 0)
    assert bridge.convert({"batches": [{"event_type": "access", "entries": [_entry("aa" * 32, 0)]}]}, 0) == ([], 0)


@pytest.mark.unit
def test_block_hash_is_stable_and_positive():
    value = _stable_block_hash("aa" * 32)
    assert value > 0
    assert value < 1 << 63
    assert value == _stable_block_hash("aa" * 32)


def test_entries_from_another_model_version_are_not_indexed():
    bridge = LMCacheKvEventBridge("http://127.0.0.1:1", current_version=lambda: 3)
    events, _ = bridge.convert(
        _store_batch([_entry("aa" * 32, 0, version="2"), _entry("bb" * 32, 256, version="3")]),
        0,
    )

    blocks = events[0].stored.blocks
    assert [block.block_hash for block in blocks] == [_stable_block_hash("bb" * 32)]
    # The stale chunk was not remembered, so the indexed chunk starts the chain.
    assert not events[0].stored.HasField("parent_block_hash")


def test_a_version_change_resets_the_placement_chain():
    version = 3
    bridge = LMCacheKvEventBridge("http://127.0.0.1:1", current_version=lambda: version)
    bridge.convert(_store_batch([_entry("aa" * 32, 0, version="3"), _entry("bb" * 32, 256, version="3")]), 0)
    version = 4
    events, _ = bridge.convert(_store_batch([_entry("cc" * 32, 512, version="4")]), 2)

    assert not events[0].stored.HasField("parent_block_hash")

