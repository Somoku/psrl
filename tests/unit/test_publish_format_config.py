"""The published weight format is shared by every rollout engine that pulls it.

Generation engines read `gen_actor_rollout_ref.rollout`; validate engines are built from
`train_actor_rollout_ref.rollout`. Under direct supply the trainer publishes one format
for both. If only one side is configured for block FP8, the other registers bf16 tensors
and its first pull of a published version dies on a dtype assertion -- during a validation
wake, long after startup looked healthy. The mismatch is refused at config validation.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf
from psrl.utils.config import _resolve_publish_cast

pytestmark = pytest.mark.cpu_test


def _config(gen_quantization=None, validate_quantization=None, colocate: bool = True):
    return OmegaConf.create(
        {
            "psrl": {
                "colocate_validate_and_train": colocate,
                "nixl": {"publish_fp8_block": None},
            },
            "gen_actor_rollout_ref": {
                "rollout": {"dtype": "bfloat16", "quantization": gen_quantization}
            },
            "train_actor_rollout_ref": {
                "actor": {"strategy": "megatron"},
                "rollout": {"quantization": validate_quantization},
            },
        }
    )


def test_an_fp8_rollout_publishes_block_fp8():
    config = _config("fp8", "fp8")
    _resolve_publish_cast(config)

    assert config.psrl.nixl.publish_fp8_block == [128, 128]


def test_matching_bf16_rollouts_publish_no_fp8_copy():
    _resolve_publish_cast(_config())

    # No assertion raised; the block stays null.


def test_a_bf16_validate_engine_next_to_an_fp8_rollout_is_refused():
    with pytest.raises(AssertionError, match="must agree on quantization"):
        _resolve_publish_cast(_config(gen_quantization="fp8", validate_quantization=None))


def test_disagreeing_formats_are_allowed_without_validate_instances():
    """Without a separate validate fleet, validation runs on the generation engines, so
    the two config blocks cannot describe different formats."""
    config = _config(gen_quantization="fp8", validate_quantization=None, colocate=False)

    _resolve_publish_cast(config)

    assert config.psrl.nixl.publish_fp8_block == [128, 128]
