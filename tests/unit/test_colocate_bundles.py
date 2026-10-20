"""GPU-share fractions that make train, rollout and validate share one node.

The fractions are Ray scheduling shares, not memory splits. Their values decide whether
each card hosts one worker of each role or two of one role, so they are pinned here
rather than left to a comment in the script.
"""

from __future__ import annotations

import pytest
from psrl.trainer.main_ppo import colocate_bundle_fractions


def test_validate_only_colocation_keeps_rollout_whole():
    assert colocate_bundle_fractions({"colocate_validate_and_train": True}) == pytest.approx((0.9, 1.0, 0.1))


def test_no_colocation_requests_whole_gpus():
    assert colocate_bundle_fractions({"colocate_validate_and_train": False}) == (1.0, 1.0, 0.0)


def test_full_colocation_fits_each_card():
    train, rollout, validate = colocate_bundle_fractions(
        {
            "colocate_validate_and_train": True,
            "colocate_rollout_and_train": True,
            "colocate_train_fraction": 0.6,
            "colocate_rollout_fraction": 0.3,
            "colocate_validate_fraction": 0.09,
        }
    )

    assert (train, rollout, validate) == (0.6, 0.3, 0.09)
    assert train + rollout + validate <= 1.0
    # At most one worker of each role per card; two of any role cannot fit.
    assert 2 * train > 1.0
    assert train + 2 * rollout > 1.0
    assert train + rollout + 2 * validate > 1.0


def test_full_colocation_requires_validate_colocation():
    with pytest.raises(ValueError, match="requires colocate_validate_and_train"):
        colocate_bundle_fractions({"colocate_rollout_and_train": True})


def test_fractions_above_one_gpu_are_refused():
    with pytest.raises(ValueError, match="at most 1 GPU"):
        colocate_bundle_fractions(
            {
                "colocate_validate_and_train": True,
                "colocate_rollout_and_train": True,
                "colocate_train_fraction": 0.6,
                "colocate_rollout_fraction": 0.4,
                "colocate_validate_fraction": 0.1,
            }
        )
