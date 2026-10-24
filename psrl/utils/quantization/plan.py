"""Weight-transport quantization plans.

The plan is the one description of the bytes the trainer publishes. The generation
engine, the validation engine, the vLLM converter and the NIXL publisher all rebuild
the same plan, so a format change cannot reach only one side.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import torch

# Schemes the transport can produce today.
SUPPORTED_SCHEMES = ("none", "fp8")
# Recognized names whose publisher and engine paths arrive in a later phase.
RESERVED_SCHEMES = {
    "mxfp8": "Phase 2",
    "nvfp4": "Phase 3",
    "compressed_tensors": "Phase 2",
}
DEFAULT_FP8_BLOCK = (128, 128)
FP8_SCALE_SUFFIX = "_scale_inv"
FP8_WEIGHT_DTYPE = "float8_e4m3fn"
FP8_SCALE_DTYPE = "float32"


@dataclass(frozen=True)
class QuantizationPlan:
    """Serializable description of the published weight format.

    Args:
        scheme (str): `none` or `fp8` in this phase.
        block (tuple[int, int] | None): Quantization tile on the last two dimensions.
        scale_suffix (str | None): Suffix the rollout gives its scale parameter.
        scale_dtype (str | None): Wire dtype of the scale parameter.
        weight_dtype (str | None): Wire dtype of the quantized weight.
        preset (str | None): Compressed-tensors preset, reserved for later phases.
        ignore (tuple[str, ...]): Module patterns kept high precision, reserved.
        config_file (str | None): Compressed-tensors config file, reserved.
    """

    scheme: str = "none"
    block: tuple[int, int] | None = None
    scale_suffix: str | None = None
    scale_dtype: str | None = None
    weight_dtype: str | None = None
    preset: str | None = None
    ignore: tuple[str, ...] = field(default_factory=tuple)
    config_file: str | None = None

    @property
    def is_quantized(self) -> bool:
        """Whether any auxiliary parameter travels with the weight."""
        return self.scheme != "none"

    @property
    def block_shape(self) -> tuple[int, ...] | None:
        """The tile applied to a 2D weight."""
        return None if self.block is None else tuple(self.block)

    def scale_key_of(self, weight_key: str) -> str | None:
        """The rollout parameter that holds this weight's scale."""
        return None if self.scale_suffix is None else weight_key + self.scale_suffix

    def base_of_scale_key(self, key: str) -> str | None:
        """The weight key for a scale key, or None when `key` is not a scale."""
        if self.scale_suffix is None or not key.endswith(self.scale_suffix):
            return None
        return key[: -len(self.scale_suffix)]

    def is_aux_key(self, key: str) -> bool:
        """Whether `key` is an auxiliary parameter rather than a weight."""
        return self.base_of_scale_key(key) is not None

    def to_config(self) -> dict[str, Any]:
        """A plain mapping that survives a Hydra/OmegaConf round trip."""
        return {
            "scheme": self.scheme,
            "block": None if self.block is None else list(self.block),
            "scale_suffix": self.scale_suffix,
            "scale_dtype": self.scale_dtype,
            "weight_dtype": self.weight_dtype,
            "preset": self.preset,
            "ignore": list(self.ignore),
            "config_file": self.config_file,
        }

    @classmethod
    def from_config(cls, raw: Mapping[str, Any] | None) -> QuantizationPlan:
        """Rebuild a resolved plan from `psrl.nixl.publish_quantization`."""
        if not raw:
            return cls()
        known = {field_name for field_name in cls.__dataclass_fields__}
        values = {key: value for key, value in dict(raw).items() if key in known}
        block = values.get("block")
        if block is not None:
            values["block"] = tuple(int(value) for value in block)
        ignore = values.get("ignore")
        if ignore is not None:
            values["ignore"] = tuple(str(value) for value in ignore)
        return cls(**values)

    def replace(self, **changes) -> QuantizationPlan:
        """Return a copy with selected fields replaced."""
        return replace(self, **changes)


def resolve_quantization_plan(raw: Mapping[str, Any] | None) -> QuantizationPlan:
    """Validate the configured wire format and resolve its defaults.

    Args:
        raw (Mapping | None): The `psrl.nixl.weight_quantization` node.

    Returns:
        QuantizationPlan: A fully resolved plan.

    Raises:
        ValueError: If the scheme is unknown, reserved for a later phase, or carries
            options this phase does not implement.
    """
    values = dict(raw or {})
    scheme = str(values.get("scheme") or "none").lower()
    if scheme == "none":
        if values.get("block") or values.get("preset") or values.get("config_file") or values.get("ignore"):
            raise ValueError(
                "psrl.nixl.weight_quantization.scheme=none does not accept block, preset, ignore "
                "or config_file."
            )
        return QuantizationPlan()
    if scheme in RESERVED_SCHEMES:
        raise ValueError(
            f"psrl.nixl.weight_quantization.scheme={scheme!r} is reserved for {RESERVED_SCHEMES[scheme]}. "
            "It is recognized but not implemented yet. Use scheme=fp8 or scheme=none."
        )
    if scheme not in SUPPORTED_SCHEMES:
        raise ValueError(
            f"Unknown psrl.nixl.weight_quantization.scheme={scheme!r}. "
            f"Choose from {sorted(SUPPORTED_SCHEMES)} or a reserved name {sorted(RESERVED_SCHEMES)}."
        )

    block = values.get("block") or DEFAULT_FP8_BLOCK
    block = tuple(int(value) for value in block)
    if len(block) != 2 or block != DEFAULT_FP8_BLOCK:
        raise ValueError(
            f"psrl.nixl.weight_quantization.scheme=fp8 requires block={list(DEFAULT_FP8_BLOCK)}, "
            f"got {list(block)}."
        )
    if values.get("preset") or values.get("config_file") or values.get("ignore"):
        raise ValueError(
            "preset, ignore and config_file belong to the compressed-tensors schemes and are not "
            "implemented for scheme=fp8."
        )
    return QuantizationPlan(
        scheme="fp8",
        block=block,
        scale_suffix=FP8_SCALE_SUFFIX,
        scale_dtype=FP8_SCALE_DTYPE,
        weight_dtype=FP8_WEIGHT_DTYPE,
    )


def plan_dtype(plan: QuantizationPlan, field_name: str) -> torch.dtype | None:
    """Resolve a plan dtype string to a torch dtype."""
    name = getattr(plan, field_name)
    return None if name is None else getattr(torch, name)
