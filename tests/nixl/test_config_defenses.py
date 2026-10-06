"""Config-time defenses for the NIXL weight transport (implementation item A5).

Every check here exists because the corresponding failure is otherwise *silent* or
surfaces far from its cause:

- `model_dtype` ignored  -> buffers sized for the wrong dtype, wire format misread
- `param_offload`        -> registered GPU pointers dangle between steps
- `offload_policy`       -> parameters live on the host, nothing stable to register
- uneven FSDP2 sharding  -> ValueError mid-transfer instead of at startup
- un-merged LoRA         -> base model synced, reward never moves, no error at all

Design doc 3.8.0, 3.9.2.1, 3.9.3c, 3.11.
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from psrl.utils.config import (
    _validate_nixl_lora,
    _validate_nixl_offload_flags,
    resolve_train_model_dtype,
)


def _config(**overrides) -> OmegaConf:
    """Minimal config carrying only the fields these validators read."""
    base = {
        "train_actor_rollout_ref": {
            "actor": {
                "strategy": "fsdp2",
                "param_offload": False,
                "engine": {"model_dtype": "fp32", "offload_policy": False},
            },
            "model": {"lora": {}, "lora_rank": 0},
        }
    }
    cfg = OmegaConf.create(base)
    for path, value in overrides.items():
        OmegaConf.update(cfg, path, value, merge=True)
    return cfg


class TestResolveTrainModelDtype:
    """A5(a): read model_dtype, do not infer it from the backend."""

    def test_megatron_is_bf16_regardless_of_model_dtype(self):
        """Megatron keeps bf16 model params next to fp32 master params."""
        cfg = _config(**{"train_actor_rollout_ref.actor.strategy": "megatron"})
        assert resolve_train_model_dtype(cfg) == "bf16"

    def test_fsdp2_defaults_to_fp32(self):
        assert resolve_train_model_dtype(_config()) == "fp32"

    def test_fsdp2_honours_bf16_model_dtype(self):
        """The case the old backend-branch got wrong; used by 8 verl production recipes."""
        cfg = _config(**{"train_actor_rollout_ref.actor.engine.model_dtype": "bf16"})
        assert resolve_train_model_dtype(cfg) == "bf16"

    def test_resolved_dtype_is_convertible_to_torch(self):
        """The result feeds PrecisionType.to_dtype, so it must be a known name."""
        from verl.utils.torch_dtypes import PrecisionType

        for model_dtype, expected in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            cfg = _config(**{"train_actor_rollout_ref.actor.engine.model_dtype": model_dtype})
            assert PrecisionType.to_dtype(resolve_train_model_dtype(cfg)) is expected

    def test_missing_engine_section_falls_back_to_fp32(self):
        cfg = OmegaConf.create({"train_actor_rollout_ref": {"actor": {"strategy": "fsdp2"}}})
        assert resolve_train_model_dtype(cfg) == "fp32"


class TestOffloadFlags:
    """A5(b): both flags must be rejected, independently."""

    def test_clean_config_passes(self):
        _validate_nixl_offload_flags(_config())

    def test_param_offload_is_rejected(self):
        cfg = _config(**{"train_actor_rollout_ref.actor.param_offload": True})
        with pytest.raises(ValueError, match="param_offload"):
            _validate_nixl_offload_flags(cfg)

    def test_offload_policy_is_rejected(self):
        cfg = _config(**{"train_actor_rollout_ref.actor.engine.offload_policy": True})
        with pytest.raises(ValueError, match="offload_policy"):
            _validate_nixl_offload_flags(cfg)

    def test_offload_policy_is_caught_even_when_param_offload_is_false(self):
        """verl forces _is_offload_param=False when offload_policy is set, so a
        single combined check would miss this exact combination."""
        cfg = _config(
            **{
                "train_actor_rollout_ref.actor.param_offload": False,
                "train_actor_rollout_ref.actor.engine.offload_policy": True,
            }
        )
        with pytest.raises(ValueError, match="offload_policy"):
            _validate_nixl_offload_flags(cfg)

    def test_optimizer_offload_is_allowed(self):
        """It is the only way to free optimizer state across a colocate switch."""
        cfg = _config(**{"train_actor_rollout_ref.actor.optimizer_offload": True})
        _validate_nixl_offload_flags(cfg)

    def test_error_names_the_surviving_alternative(self):
        cfg = _config(**{"train_actor_rollout_ref.actor.param_offload": True})
        with pytest.raises(ValueError, match="optimizer_offload may stay enabled"):
            _validate_nixl_offload_flags(cfg)


class TestLoRA:
    """A5(e): un-merged LoRA must be refused, because it fails silently."""

    def test_no_lora_passes(self):
        _validate_nixl_lora(_config())

    def test_unmerged_lora_rank_is_rejected(self):
        cfg = _config(**{"train_actor_rollout_ref.model.lora": {"rank": 16, "merge": False}})
        with pytest.raises(ValueError, match="LoRA adapter synchronisation is not supported"):
            _validate_nixl_lora(cfg)

    def test_merged_lora_is_allowed(self):
        """Merged weights have full-fine-tuning shape, so the transport expresses them."""
        cfg = _config(**{"train_actor_rollout_ref.model.lora": {"rank": 16, "merge": True}})
        _validate_nixl_lora(cfg)

    def test_legacy_lora_rank_field_is_honoured(self):
        """`model.lora_rank` predates `model.lora.rank`; both are read elsewhere."""
        cfg = _config(**{"train_actor_rollout_ref.model.lora_rank": 8})
        with pytest.raises(ValueError, match="LoRA"):
            _validate_nixl_lora(cfg)

    def test_adapter_path_without_rank_is_rejected(self):
        """A pre-trained adapter is LoRA even when rank is left at 0."""
        cfg = _config(**{"train_actor_rollout_ref.model.lora_adapter_path": "/some/adapter"})
        with pytest.raises(ValueError, match="LoRA"):
            _validate_nixl_lora(cfg)

    def test_error_names_the_merge_escape_hatch(self):
        cfg = _config(**{"train_actor_rollout_ref.model.lora": {"rank": 16, "merge": False}})
        with pytest.raises(ValueError, match="merge=true"):
            _validate_nixl_lora(cfg)

    def test_error_states_the_silent_failure_mode(self):
        """A user who does not know it fails silently will not understand the refusal."""
        cfg = _config(**{"train_actor_rollout_ref.model.lora": {"rank": 16, "merge": False}})
        with pytest.raises(ValueError, match="silently sync the base model"):
            _validate_nixl_lora(cfg)


class TestEvenShardingAssertion:
    """A5(c): uneven FSDP2 sharding must fail at declaration, not mid-transfer.

    `fsdp_converter` imports `verl.utils.model`, which imports `transformers.AutoModel`
    -- broken in environments where numpy 2 was pulled in. The assertion itself is a
    pure shape check, so skip rather than fail when the import chain is unavailable.
    """

    @staticmethod
    def _assert_even_sharding():
        pytest.importorskip("psrl.utils.converter.fsdp_converter", reason="transformers import chain")
        from psrl.utils.converter.fsdp_converter import _assert_even_sharding

        return _assert_even_sharding

    def test_even_sharding_passes(self):
        self._assert_even_sharding()("w", torch.zeros(4096, 128), shard_dim=0, shard_count=8)

    def test_uneven_sharding_raises_with_actionable_message(self):
        with pytest.raises(ValueError, match="does not divide evenly"):
            self._assert_even_sharding()("layers.0.mlp.w1", torch.zeros(100, 128), shard_dim=0, shard_count=8)

    def test_message_names_the_parameter_and_the_fix(self):
        with pytest.raises(ValueError) as excinfo:
            self._assert_even_sharding()("layers.0.mlp.w1", torch.zeros(100, 128), shard_dim=0, shard_count=8)
        message = str(excinfo.value)
        assert "layers.0.mlp.w1" in message
        assert "fsdp_size" in message

    def test_second_dimension_is_checked(self):
        with pytest.raises(ValueError, match="dim1"):
            self._assert_even_sharding()("w", torch.zeros(4096, 100), shard_dim=1, shard_count=8)
