"""Derived FP8 block scales must share one registration, not one per weight.

NIXL registers memory regions, so a scale tensor per weight means one UCX MR per
weight. A block-FP8 Qwen3-8B train rank owns ~250 of them; measured on the cluster,
registering 257 regions took 8.5-9.2 s on every trainer wake while the rest of the
switch stayed under 0.1 s. Pooling every scale shard of a device into one allocation
turns those into contiguous views of a single region, so the wake cost tracks one
registration.

The test exercises the shard algebra without NIXL: the property that matters is that
the derived scales are distinct, correctly shaped windows into one storage.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
from psrl.utils.nixl.block_fp8 import SCALE_SUFFIX
from psrl.utils.nixl.client import NIXLStorageClient
from psrl.utils.nixl.nixl_spec import NIXLSharding

pytestmark = pytest.mark.cpu_test

SHAPES = {
    "model.layers.0.self_attn.q_proj.weight": (256, 256),
    "model.layers.1.mlp.up_proj.weight": (384, 512),
}


class _Client:
    """Just the parts `_derive_block_scales` reads."""

    _fp8_block = (128, 128)
    _tile_of = NIXLStorageClient._tile_of
    _assert_whole_tiles = staticmethod(NIXLStorageClient._assert_whole_tiles)
    _derive_block_scales = NIXLStorageClient._derive_block_scales


def _sharding() -> NIXLSharding:
    return NIXLSharding(shard_mesh=OrderedDict([(0, 1)]), shard_indices=[(0,)])


def test_every_scale_shard_is_a_view_of_one_allocation():
    state = {key: torch.empty(shape) for key, shape in SHAPES.items()}
    sharding = {key: _sharding() for key in SHAPES}
    sharding.update({key + SCALE_SUFFIX: _sharding() for key in SHAPES})

    derived = _Client()._derive_block_scales(state, sharding)

    scales = [derived[key + SCALE_SUFFIX] for key in SHAPES]
    assert [tuple(s.shape) for s in scales] == [(2, 2), (3, 4)]
    assert len({s.untyped_storage().data_ptr() for s in scales}) == 1, (
        "the derived scales do not share one allocation, so each becomes its own NIXL region"
    )


def test_the_windows_do_not_overlap():
    """A view that aliased another would make two weights share scale entries, which
    corrupts both dequantisation paths at once."""
    state = {key: torch.empty(shape) for key, shape in SHAPES.items()}
    sharding = {key: _sharding() for key in SHAPES}
    sharding.update({key + SCALE_SUFFIX: _sharding() for key in SHAPES})

    derived = _Client()._derive_block_scales(state, sharding)
    first = derived[next(iter(SHAPES)) + SCALE_SUFFIX]
    second = derived[list(SHAPES)[1] + SCALE_SUFFIX]
    first.fill_(1.0)
    second.fill_(2.0)

    assert torch.all(first == 1.0) and torch.all(second == 2.0)
