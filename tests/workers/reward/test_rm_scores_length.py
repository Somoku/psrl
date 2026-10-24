"""`rm_scores` is built from each trajectory's own response length.

A session can produce several trajectories of different lengths. One length shared across them
gave the longer trajectories a reward tensor that disagreed with their `response_mask`, and the
batch then padded the two fields apart. These tests pin that every row follows its own
trajectory.
"""

import pytest

torch = pytest.importorskip("torch")

from psrl.workers.reward.rm_scores import rm_scores_from_lengths  # noqa: E402

pytestmark = pytest.mark.cpu_test


def test_each_row_follows_its_own_trajectory_length() -> None:
    """The regression: one session, three trajectories, three different lengths."""
    rows = rm_scores_from_lengths([39086, 39705, 279], [1.0, 0.0, 0.5])

    assert [row.numel() for row in rows] == [39086, 39705, 279]
    assert [float(row[-1]) for row in rows] == [1.0, 0.0, 0.5]


def test_the_reward_sits_on_the_last_token() -> None:
    (row,) = rm_scores_from_lengths([4], [0.75])

    assert row.tolist() == [0.0, 0.0, 0.0, 0.75]


def test_a_zero_length_response_gets_an_empty_row() -> None:
    (row,) = rm_scores_from_lengths([0], [1.0])

    assert row.numel() == 0


def test_mismatched_sizes_are_rejected() -> None:
    with pytest.raises(ValueError, match="response lengths"):
        rm_scores_from_lengths([3, 4], [1.0])
