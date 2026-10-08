"""CPU tests for the compact-capture -> training-contract adapters.

These exercise the multi-turn case (masked environment rows interleaved with assistant rows), which
is the shape that broke the original ``to_dense_topk`` broadcast.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from psrl.utils.sampler_adapter import to_compact_support, to_dense_support, to_dense_topk  # noqa: E402
from psrl.utils.sampler_distribution import SamplerDistribution  # noqa: E402


def _distribution(heads, complete=None):
    """Build a compact distribution from a list of per-row head id lists."""
    offsets = [0]
    token_ids = []
    log_probs = []
    for head in heads:
        token_ids.extend(head)
        log_probs.extend([-0.1 * (i + 1) for i in range(len(head))])
        offsets.append(len(token_ids))
    if complete is None:
        complete = [False] * len(heads)
    return SamplerDistribution(
        offsets=np.asarray(offsets, dtype=np.int64),
        token_ids=np.asarray(token_ids, dtype=np.int32),
        log_probs=np.asarray(log_probs, dtype=np.float32),
        complete=np.asarray(complete, dtype=np.bool_),
    )


def test_dense_topk_places_heads_and_leaves_env_rows_empty():
    # Interleaved environment row at response index 1.
    distribution = _distribution([[10, 11], [], [12, 13], [14, 15]])
    out = to_dense_topk(distribution, [1, 0, 1, 1], prompt_length=3)
    ids = out["rollout_topk_ids"]
    log_probs = out["rollout_topk_log_probs"]

    # Full sequence has prompt_length + response_length rows.
    assert ids.shape == (3 + 4, 2) == log_probs.shape

    # Response token r is predicted by the row at index prompt_length-1+r.
    assert ids[2].tolist() == [10, 11]
    assert ids[4].tolist() == [12, 13]
    assert ids[5].tolist() == [14, 15]

    # The env row and every prompt/pad row carry the sentinel head.
    assert ids[3].tolist() == [-1, -1]
    for row in (0, 1, 6):
        assert ids[row].tolist() == [-1, -1]
    assert bool(torch.isneginf(log_probs[3]).all())


def test_dense_topk_rejects_ragged_widths():
    distribution = _distribution([[10, 11], [12]])
    with pytest.raises(ValueError, match="fixed width"):
        to_dense_topk(distribution, [1, 1], prompt_length=1)


def test_dense_topk_rejects_env_rows_with_entries():
    distribution = _distribution([[10, 11], [12, 13]])
    with pytest.raises(ValueError, match="environment/tool rows"):
        to_dense_topk(distribution, [1, 0], prompt_length=1)


def test_dense_support_pads_rows_and_keeps_env_rows_empty():
    distribution = _distribution([[10, 11, 12], [], [20, 21]])
    out = to_dense_support(distribution, [1, 0, 1], width=3, prompt_length=2)
    ids = out["rollout_topk_ids"]
    assert ids.shape == (2 + 3, 3)
    assert ids[1].tolist() == [10, 11, 12]
    assert ids[3].tolist() == [20, 21, -1]  # short row right-padded with the sentinel
    assert ids[2].tolist() == [-1, -1, -1]  # env row


def test_compact_support_ships_lengths_flat_ids_and_flat_log_probs():
    distribution = _distribution([[10, 11, 12], [], [20, 21]])
    out = to_compact_support(distribution, [1, 0, 1], width=3, prompt_length=2)
    assert set(out) == {
        "rollout_topk_lengths",
        "rollout_support_ids",
        "rollout_support_log_probs",
    }
    lengths = out["rollout_topk_lengths"]
    assert lengths.shape == (2 + 3,)
    # One length per sequence row: the leading prompt row and the trailing final-position row are
    # zero, and the response rows start at prompt_length-1.
    assert lengths.tolist() == [0, 3, 0, 2, 0]
    assert out["rollout_support_ids"].tolist() == [10, 11, 12, 20, 21]
    # Log-probs are flat (only the realized support), not a dense [T, width] array.
    assert out["rollout_support_log_probs"].shape == (5,)


def test_dense_support_vectorized_padding_over_many_ragged_rows():
    # Six rows with mixed counts and empty rows, to guard the vectorized ragged scatter.
    heads = [[10, 11, 12], [], [20], [30, 31, 32, 33], [], [40, 41]]
    distribution = _distribution(heads)
    out = to_dense_support(distribution, [1, 0, 1, 1, 0, 1], width=4, prompt_length=2)
    ids = out["rollout_topk_ids"]
    assert ids.shape == (2 + 6, 4)
    assert ids[0].tolist() == [-1, -1, -1, -1]  # prompt
    assert ids[1].tolist() == [10, 11, 12, -1]  # response row 0
    assert ids[2].tolist() == [-1, -1, -1, -1]  # env row
    assert ids[3].tolist() == [20, -1, -1, -1]
    assert ids[4].tolist() == [30, 31, 32, 33]
    assert ids[5].tolist() == [-1, -1, -1, -1]  # env row
    assert ids[6].tolist() == [40, 41, -1, -1]
    assert ids[7].tolist() == [-1, -1, -1, -1]  # final position
