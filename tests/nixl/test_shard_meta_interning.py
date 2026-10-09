"""Shard metadata is interned before it goes on the wire (part of item C1).

Grouping collapses 128 expert keys into one key with 128 shards. That removes 128 key
entries but leaves 128 shard-metadata objects, and those are equal rather than
identical -- experts of one projection have the same dtype, shape and stride, differing
only in address, which lives in the descriptor instead.

Pickle memoises by identity, so equal-but-distinct objects serialise in full every
time. Interning them is what takes the handshake reduction from 2.5x to 10.7x on
Qwen3-30B-A3B; without it the key-count win is mostly given back in shard metadata.

These tests exercise the interning rule directly rather than through a live NIXL agent,
so they stay CPU-only; `test_nixl_e2e.py` covers the registered path.
"""

from __future__ import annotations

import pickle
from collections import OrderedDict

import pytest
import torch

from psrl.utils.nixl.nixl_spec import NIXLShardMetaInfo

pytestmark = pytest.mark.cpu_test


def _intern(interner: dict, meta: NIXLShardMetaInfo) -> NIXLShardMetaInfo:
    """The rule under test, mirroring `NIXLStorageClient._intern_shard_meta`."""
    cache_key = (
        meta.dtype,
        str(meta.device),
        tuple(meta.shape),
        meta.stride,
        meta.is_contiguous,
        meta.is_strided_expressible,
    )
    return interner.setdefault(cache_key, meta)


def _meta(**overrides) -> NIXLShardMetaInfo:
    fields = dict(
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        shape=torch.Size((768, 2048)),
        stride=(2048, 1),
        is_contiguous=True,
    )
    fields.update(overrides)
    return NIXLShardMetaInfo(**fields)


class TestInterningIsNecessary:
    def test_equal_metadata_is_not_identical_by_default(self):
        """The premise: pickle cannot collapse these on its own."""
        first, second = _meta(), _meta()
        assert first == second
        assert first is not second

    def test_distinct_objects_cost_proportionally_more(self):
        distinct = [_meta() for _ in range(128)]
        shared = [_meta()] * 128

        assert len(pickle.dumps(distinct)) > 10 * len(pickle.dumps(shared)), (
            "interning has to be worth doing, or the key-count win is given back here"
        )


class TestInterningRule:
    def test_equal_metadata_collapses_to_one_object(self):
        interner: dict = {}
        shards = [_intern(interner, _meta()) for _ in range(128)]

        assert len(interner) == 1
        assert all(shard is shards[0] for shard in shards)

    def test_a_different_shape_stays_separate(self):
        interner: dict = {}
        a = _intern(interner, _meta())
        b = _intern(interner, _meta(shape=torch.Size((768, 4096))))

        assert a is not b

    @pytest.mark.parametrize(
        "overrides",
        [
            {"dtype": torch.float32},
            {"device": torch.device("cuda:1")},
            {"stride": (1, 768)},
            {"is_contiguous": False},
        ],
        ids=["dtype", "device", "stride", "contiguity"],
    )
    def test_every_field_participates_in_identity(self, overrides):
        """A field left out of the key would silently merge genuinely different shards."""
        interner: dict = {}
        base = _intern(interner, _meta())
        other = _intern(interner, _meta(**overrides))

        assert base is not other, f"{list(overrides)} did not affect interning identity"

    def test_strided_expressibility_participates(self):
        """It decides whether the peer stages a copy, so it cannot be merged away."""
        interner: dict = {}
        a = _intern(interner, _meta(is_contiguous=False))
        b = _meta(is_contiguous=False)
        b.is_strided_expressible = True
        b = _intern(interner, b)

        assert a is not b

    def test_interning_preserves_value(self):
        """Sharing an object must not change what any shard reports."""
        interner: dict = {}
        original = _meta()
        interned = _intern(interner, original)
        again = _intern(interner, _meta())

        assert interned == original
        assert again == original


class TestInternedMetadataSurvivesTheWire:
    def test_identity_is_preserved_through_a_round_trip(self):
        """Interning is only useful if one object stays one object after unpickling."""
        interner: dict = {}
        shards = [_intern(interner, _meta()) for _ in range(8)]

        restored = pickle.loads(pickle.dumps(shards))

        assert all(item is restored[0] for item in restored)
        assert restored[0] == shards[0]

    def test_a_grouped_key_payload_stays_small(self):
        """The property the design's 'hundreds of KB' claim rests on."""
        from psrl.utils.nixl.nixl_spec import NIXLSharding, NIXLTensorInfo

        num_experts = 128
        interner: dict = {}
        metas = [_intern(interner, _meta()) for _ in range(num_experts)]
        info = NIXLTensorInfo(
            desc_bytes_list=[b"x" * 406] * num_experts,
            temp_desc_bytes_list=[None] * num_experts,
            sharding=NIXLSharding(
                shard_mesh=OrderedDict([(0, num_experts)]),
                shard_indices=[(i,) for i in range(num_experts)],
            ),
            shard_meta_infos=metas,
        )

        payload = len(info.serialize())
        assert payload < 2500, (
            f"a 128-expert grouped key serialised to {payload} bytes; the shard metadata "
            "is no longer being shared"
        )
