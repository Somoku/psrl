"""The layout judge sees the misplacements the mod-256 bf16 judge cannot (T1 follow-up).

The value judge encodes `flat_index mod 256` under bf16, so a row or a column block that
lands one 256-multiple away from where it belongs passes. These tests pin that the
layout judge catches exactly those, on the TP-mismatch geometry the e2e harness runs
(train TP=4 cells reassembled into a gen TP=2 shard), and that it passes the correct
assembly -- a judge that rejects everything is as useless as one that accepts it.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
from psrl.utils.nixl.nixl_spec import NIXLSharding

from tests.nixl.coord_judge import assert_matches_global_coord, check_by_layout, fill_by_global_coord, fill_by_layout

pytestmark = pytest.mark.cpu_test

KEY = "model.layers.0.self_attn.o_proj.weight"


def _col_sharding(count: int, index: int) -> NIXLSharding:
    return NIXLSharding(shard_mesh=OrderedDict([(1, count)]), shard_indices=[(index,)])


def _train_cells(dtype=torch.bfloat16):
    """Four train TP ranks of a [512, 1024] row-parallel weight, filled by layout."""
    cells = []
    for r in range(4):
        t = torch.empty(512, 256, dtype=dtype)
        fill_by_layout({KEY: t}, {KEY: _col_sharding(4, r)})
        cells.append(t)
    return cells


def _judge_gen(tensor, gen_rank=0):
    return check_by_layout({KEY: tensor}, {KEY: _col_sharding(2, gen_rank)}, [KEY])


class TestTheOldJudgeIsBlindHere:
    """Pinned so the reason for the second judge is not lost."""

    def test_swapped_column_halves_pass_the_mod_256_judge(self):
        g = torch.empty(4096, 2048, dtype=torch.bfloat16)
        fill_by_global_coord(g, 0, strict=False)
        swapped = torch.cat([g[:, 1024:], g[:, :1024]], dim=1).contiguous()
        assert_matches_global_coord(swapped, 0, strict=False)  # passes: that is the bug


class TestLayoutJudge:
    def test_correct_reassembly_passes(self):
        c = _train_cells()
        assert _judge_gen(torch.cat([c[0], c[1]], dim=1)) == (1, None)
        assert _judge_gen(torch.cat([c[2], c[3]], dim=1), gen_rank=1) == (1, None)

    def test_swapped_cells_are_caught(self):
        c = _train_cells()
        judged, failure = _judge_gen(torch.cat([c[1], c[0]], dim=1))
        assert failure and "origin (0, 0)" in failure

    def test_wrong_rank_cell_is_caught(self):
        c = _train_cells()
        _, failure = _judge_gen(torch.cat([c[0], c[2]], dim=1))
        assert failure

    def test_row_shift_is_caught(self):
        c = _train_cells()
        g = torch.cat([c[0], c[1]], dim=1)
        _, failure = _judge_gen(torch.roll(g, shifts=1, dims=0))
        assert failure

    def test_stale_version_is_caught(self):
        g = torch.empty(512, 512, dtype=torch.bfloat16)
        fill_by_layout({KEY: g}, {KEY: _col_sharding(2, 0)}, version=1)
        _, failure = check_by_layout({KEY: g}, {KEY: _col_sharding(2, 0)}, [KEY], version=2)
        assert failure

    def test_fill_stays_in_place_on_a_strided_view(self):
        base = torch.zeros(512, 1024, dtype=torch.bfloat16)
        view = base[:, 256:512]
        fill_by_layout({KEY: view}, {KEY: _col_sharding(4, 1)})
        assert base[:, 256:512].view(torch.int16).abs().sum() > 0
        assert base[:, :256].view(torch.int16).abs().sum() == 0

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float8_e4m3fn, torch.float16])
    def test_other_widths(self, dtype):
        c = _train_cells(dtype)
        assert _judge_gen(torch.cat([c[0], c[1]], dim=1)) == (1, None)
        _, failure = _judge_gen(torch.cat([c[1], c[0]], dim=1))
        assert failure

    def test_finest_mesh_cells_on_the_gen_side(self):
        """The gen tensor judged through its *unified* sharding gives the same verdict."""
        c = _train_cells()
        g = torch.cat([c[0], c[1]], dim=1)
        unified = _col_sharding(2, 0)
        unified.refactor_based_on_finer_shard_mesh(OrderedDict([(1, 4)]))
        assert check_by_layout({KEY: g}, {KEY: unified}, [KEY]) == (1, None)


def test_blocked_judgement_matches_whole(monkeypatch):
    """Splitting into row blocks must not change the verdict (it bounds memory only)."""
    import tests.nixl.coord_judge as cj

    c = _train_cells()
    g = torch.cat([c[0], c[1]], dim=1)
    monkeypatch.setattr(cj, "_JUDGE_BLOCK_ELEMS", 1000)
    assert _judge_gen(g) == (1, None)
    _, failure = _judge_gen(torch.roll(g, shifts=1, dims=0))
    assert failure
    t = torch.empty(512, 256, dtype=torch.bfloat16)
    fill_by_layout({KEY: t}, {KEY: _col_sharding(4, 0)})
    assert torch.equal(t.view(torch.int16), c[0].view(torch.int16))  # bits: some patterns are NaN


def test_grouped_expert_keys_agree_across_ep_and_tp():
    """Train EP (whole experts) and gen TP (split inside experts) must judge alike."""
    from psrl.utils.nixl.moe_grouping import group_expert_tensors

    key = "model.layers.0.mlp.experts.gate_proj.weight"
    train = {
        f"model.layers.0.mlp.experts.{e}.gate_proj.weight": torch.empty(8, 6, dtype=torch.bfloat16) for e in (2, 3)
    }
    g = group_expert_tensors(train, {k: NIXLSharding.default() for k in train}, num_experts=4)
    fill_by_layout(g.state_dict, g.sharding_dict)

    # gen: all 4 experts, each split in two along per-expert dim 0 (grouped dim 1)
    gen = {
        f"model.layers.0.mlp.experts.{e}.gate_proj.weight": torch.empty(4, 6, dtype=torch.bfloat16) for e in range(4)
    }
    gsh = {k: NIXLSharding(shard_mesh=OrderedDict([(0, 2)]), shard_indices=[(0,)]) for k in gen}
    gg = group_expert_tensors(gen, gsh, num_experts=4)
    # "transfer": copy expert 2/3 rows 0..3 from train into gen
    for e in (2, 3):
        gen[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"].copy_(
            train[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"][:4]
        )
    judged_only_present = {key: gg.state_dict[key]}
    # experts 0,1 were never filled, so the judge must fail on them and pass on 2,3
    _, failure = check_by_layout(judged_only_present, gg.sharding_dict, [key])
    assert failure and "origin (0," in failure
    for e in (0, 1):
        gen[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"].copy_(
            gen[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"]
        )
    sub = NIXLSharding(shard_mesh=OrderedDict([(0, 4), (1, 2)]), shard_indices=[(2, 0), (3, 0)])
    from psrl.utils.nixl.moe_grouping import GroupedShards

    present = GroupedShards(
        shards=tuple(gen[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"] for e in (2, 3)),
        shard_indices=((2, 0), (3, 0)),
        shard_mesh=((0, 4), (1, 2)),
    )
    assert check_by_layout({key: present}, {key: sub}, [key]) == (1, None)


@pytest.mark.parametrize("train,wire", [(torch.float32, torch.bfloat16), (torch.float32, torch.float16)])
def test_a_cast_fill_survives_the_cast(train, wire):
    """The train side fills through an exact upcast, so casting back yields the reader's expected bits."""
    t = torch.empty(512, 256, dtype=train)
    fill_by_layout({KEY: t}, {KEY: _col_sharding(4, 1)}, version=3, wire_dtype=wire)
    received = t.to(wire)
    assert check_by_layout({KEY: received}, {KEY: _col_sharding(4, 1)}, [KEY], version=3) == (1, None)
    assert torch.isfinite(received.float()).all()
    _, failure = check_by_layout({KEY: received}, {KEY: _col_sharding(4, 2)}, [KEY], version=3)
    assert failure
