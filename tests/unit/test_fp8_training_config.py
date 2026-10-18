"""Persistent FP8 parameters are refused without the model-init context they need.

`fp8_param` only produces Transformer Engine `QuantizedTensor` weights when the model
is built inside `fp8_model_init`. This engine builds it directly, so the flag leaves
bf16 parameters while changing the backward's weight-gradient accounting: a run looks
healthy for one step, then entropy and gradient norm jump by orders of magnitude. The
supported FP8 training path is hybrid autocast.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf
from psrl.utils.config import _validate_megatron_fp8_param

pytestmark = pytest.mark.cpu_test


def _config(overrides: dict, strategy: str = "megatron"):
    return OmegaConf.create(
        {
            "train_actor_rollout_ref": {
                "actor": {
                    "strategy": strategy,
                    "megatron": {"override_transformer_config": overrides},
                }
            }
        }
    )


def test_hybrid_autocast_is_accepted():
    _validate_megatron_fp8_param(
        _config({"fp8": "hybrid", "fp8_recipe": "tensorwise", "fp8_param": False})
    )


def test_persistent_fp8_parameters_are_refused():
    with pytest.raises(ValueError, match="fp8_model_init"):
        _validate_megatron_fp8_param(_config({"fp8": "hybrid", "fp8_param": True}))


def test_the_check_is_megatron_only():
    """`override_transformer_config` is a Megatron field; an FSDP config that happens to
    carry the key is not making this claim."""
    _validate_megatron_fp8_param(_config({"fp8_param": True}, strategy="fsdp"))
