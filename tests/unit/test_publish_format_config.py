"""The published weight format is one resolved plan shared by every participant.

The plan names the wire format, so the publisher, the converter and both rollout roles
cannot disagree about scale names or block geometry. It is resolved once at config
validation and serialized into `psrl.nixl.publish_quantization`.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf
from psrl.utils.config import _resolve_publish_cast

pytestmark = pytest.mark.cpu_test


def _config(scheme: str = "none", **weight_quantization):
    return OmegaConf.create(
        {
            "psrl": {
                "colocate_validate_and_train": True,
                "nixl": {
                    "weight_quantization": {"scheme": scheme, **weight_quantization},
                },
            },
            "gen_actor_rollout_ref": {"rollout": {"dtype": "bfloat16", "quantization": None}},
            "train_actor_rollout_ref": {
                "actor": {"strategy": "megatron"},
                "rollout": {"quantization": None},
            },
        }
    )


def test_an_fp8_plan_publishes_block_fp8():
    config = _config("fp8")
    _resolve_publish_cast(config)

    plan = config.psrl.nixl.publish_quantization
    assert plan.scheme == "fp8"
    assert list(plan.block) == [128, 128]
    assert plan.scale_suffix == "_scale_inv"


def test_a_none_plan_publishes_the_trainer_dtype():
    config = _config()
    _resolve_publish_cast(config)

    assert config.psrl.nixl.publish_quantization.scheme == "none"


def test_a_reserved_scheme_is_rejected_at_validation():
    with pytest.raises(ValueError, match="reserved for Phase 2"):
        _resolve_publish_cast(_config("mxfp8"))


def test_an_unknown_scheme_is_rejected_at_validation():
    with pytest.raises(ValueError, match="Unknown psrl.nixl.weight_quantization.scheme"):
        _resolve_publish_cast(_config("int4"))


def test_trainer_side_qat_is_refused_under_direct_supply():
    config = _config()
    config.gen_actor_rollout_ref.rollout.quantization = {"linear": "FP8_BLOCK"}
    with pytest.raises(ValueError, match="trainer-side QAT path"):
        _resolve_publish_cast(config)
