"""Weight-format plans resolve once and reject every unsupported option early."""

from __future__ import annotations

import pytest
import torch
from psrl.utils.quantization import (
    DEFAULT_FP8_BLOCK,
    RESERVED_SCHEMES,
    QuantizationPlan,
    plan_dtype,
    resolve_quantization_plan,
)

pytestmark = pytest.mark.cpu_test


def test_the_default_plan_is_passthrough():
    plan = resolve_quantization_plan(None)
    assert plan.scheme == "none"
    assert not plan.is_quantized
    assert plan.scale_suffix is None


def test_an_fp8_plan_resolves_the_vllm_serialized_layout():
    plan = resolve_quantization_plan({"scheme": "fp8"})
    assert plan.block == DEFAULT_FP8_BLOCK
    assert plan.scale_suffix == "_scale_inv"
    assert plan_dtype(plan, "weight_dtype") is torch.float8_e4m3fn
    assert plan_dtype(plan, "scale_dtype") is torch.float32


def test_the_scale_key_round_trips():
    plan = resolve_quantization_plan({"scheme": "fp8"})
    assert plan.scale_key_of("model.layers.0.self_attn.q_proj.weight") == (
        "model.layers.0.self_attn.q_proj.weight_scale_inv"
    )
    assert plan.base_of_scale_key("model.layers.0.self_attn.q_proj.weight_scale_inv") == (
        "model.layers.0.self_attn.q_proj.weight"
    )
    assert plan.base_of_scale_key("model.layers.0.self_attn.q_proj.weight") is None


@pytest.mark.parametrize("scheme", sorted(RESERVED_SCHEMES))
def test_each_reserved_scheme_names_its_phase(scheme):
    with pytest.raises(ValueError, match=RESERVED_SCHEMES[scheme]):
        resolve_quantization_plan({"scheme": scheme})


def test_an_unknown_scheme_lists_the_choices():
    with pytest.raises(ValueError, match="Unknown psrl.nixl.weight_quantization.scheme"):
        resolve_quantization_plan({"scheme": "int4"})


def test_fp8_rejects_a_non_default_block():
    with pytest.raises(ValueError, match="requires block"):
        resolve_quantization_plan({"scheme": "fp8", "block": [64, 64]})


def test_fp8_rejects_compressed_tensors_options():
    with pytest.raises(ValueError, match="compressed-tensors"):
        resolve_quantization_plan({"scheme": "fp8", "preset": "FP8_BLOCK"})


def test_passthrough_rejects_quantization_options():
    with pytest.raises(ValueError, match="scheme=none"):
        resolve_quantization_plan({"scheme": "none", "ignore": ["re:.*lm_head$"]})


def test_the_config_round_trip_preserves_every_field():
    plan = resolve_quantization_plan({"scheme": "fp8"})
    rebuilt = QuantizationPlan.from_config(plan.to_config())
    assert rebuilt == plan
    assert not rebuilt.is_aux_key("model.embed_tokens.weight")
