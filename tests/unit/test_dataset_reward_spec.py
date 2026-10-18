"""A dataset row must carry the reward specification into the rollout request.

The agent loop reads `reward_model_dicts` from the batch and passes it to the reward
manager; an empty list resolves to no manager and the reward is the sum over an empty
list, i.e. zero for every sample. Datasets ship `reward_model.ground_truth` only, so the
configured specification has to be attached when the row is built. Without this, a run
looks healthy -- coherent generations, finite loss -- while every reward is zero.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from psrl.utils.dataset.rl_dataset import _attach_reward_model_dicts

pytestmark = pytest.mark.cpu_test

SPEC = [{"reward_loop_type": "naive", "reward_fn": "default", "reward_coef": 1.0}]


def test_a_plain_dataset_row_gets_the_configured_specification():
    row = {"data_source": "openai/gsm8k", "reward_model": {"ground_truth": "10", "style": "rule"}}
    config = OmegaConf.create({"reward_model_dicts": SPEC})

    attached = _attach_reward_model_dicts(row, config)

    assert attached["reward_model_dicts"] == SPEC
    assert isinstance(attached["reward_model_dicts"], list)
    assert isinstance(attached["reward_model_dicts"][0], dict)


def test_a_row_that_already_specifies_its_reward_keeps_it():
    per_sample = [{"reward_loop_type": "dapo", "reward_fn": "compute_score", "reward_coef": 1.0}]
    row = {"reward_model_dicts": per_sample}
    config = OmegaConf.create({"reward_model_dicts": SPEC})

    assert _attach_reward_model_dicts(row, config)["reward_model_dicts"] == per_sample


def test_a_dataset_without_a_specification_is_left_alone():
    """The reward manager then resolves through its own fallback; silently inventing a
    specification here would hide a dataset-config error."""
    row = {"data_source": "openai/gsm8k"}
    config = OmegaConf.create({"reward_model_dicts": None})

    assert "reward_model_dicts" not in _attach_reward_model_dicts(row, config)
