"""CPU test for the actor-side bypass/score-centering config mirror.

The actor worker selects its loss function once in `init_model` from its own config snapshot, so
bypass mode and score centering only take effect if the actor's static `policy_loss` carries
`loss_mode="bypass_mode"` and the mirrored `rollout_correction`.
"""

import pytest
from omegaconf import OmegaConf
from psrl.utils.config import mirror_bypass_config_to_actor


def _config(bypass_mode: bool):
    return OmegaConf.create(
        {
            "train_actor_rollout_ref": {"actor": {"policy_loss": {"loss_mode": "vanilla"}}},
            "algorithm": {
                "rollout_correction": {
                    "rollout_is": "token",
                    "rollout_is_threshold": 2.0,
                    "rollout_rs": None,
                    "rollout_rs_threshold": None,
                    "bypass_mode": bypass_mode,
                    "loss_type": "reinforce",
                    "rollout_is_batch_normalize": False,
                    "score_centering": bypass_mode,
                }
            },
        }
    )


def test_mirror_sets_actor_policy_loss_for_bypass():
    config = _config(bypass_mode=True)
    mirror_bypass_config_to_actor(config)
    policy_loss = config.train_actor_rollout_ref.actor.policy_loss
    assert policy_loss["loss_mode"] == "bypass_mode"
    assert policy_loss["rollout_correction"]["score_centering"] is True
    assert policy_loss["rollout_correction"]["loss_type"] == "reinforce"
    assert policy_loss["rollout_correction"]["rollout_is"] == "token"


def test_mirror_is_a_noop_without_bypass():
    config = _config(bypass_mode=False)
    mirror_bypass_config_to_actor(config)
    policy_loss = config.train_actor_rollout_ref.actor.policy_loss
    assert policy_loss["loss_mode"] == "vanilla"
    assert "rollout_correction" not in policy_loss


def test_mirror_copies_so_algorithm_edits_do_not_alias():
    config = _config(bypass_mode=True)
    mirror_bypass_config_to_actor(config)
    config.algorithm.rollout_correction.rollout_is = "sequence"
    assert config.train_actor_rollout_ref.actor.policy_loss["rollout_correction"]["rollout_is"] == "token"


def test_mirror_rejects_a_conflicting_static_loss_mode():
    config = _config(bypass_mode=True)
    config.train_actor_rollout_ref.actor.policy_loss.loss_mode = "gspo"
    with pytest.raises(ValueError, match="loss_mode"):
        mirror_bypass_config_to_actor(config)
