"""
SciAccel-RL Runtime Configuration for PSRL.

Dataclass-based config for the Harbor-based SciAccel integration. Controls
Harbor Job parameters and episode timeouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import OmegaConf


@dataclass
class HarborConfig:
    """
    Harbor Job execution settings.
    """

    jobs_dir: str = "/tmp/sciaccel_jobs"
    agent_name: str = "terminus-2"
    override_gpus: int | None = None
    gpu_compose_override: str = ""


@dataclass
class SciAccelRuntimeConfig:
    """
    Top-level config for the SciAccel-RL PSRL integration.
    """

    harbor: HarborConfig = field(default_factory=HarborConfig)
    # Agent budget, enforced by Harbor from outside the container via
    # `AgentConfig.override_timeout_sec`. This is the clock that actually stops a
    # slow episode. PSRL's `asyncio.wait_for` is only a backstop above it.
    task_timeout_sec: float = 3600.0
    # Verifier budget. Only used to size that backstop, so it must be >= the task's
    # own `[verifier] timeout_sec` or the outer guard can pre-empt grading.
    verifier_timeout_sec: float = 900.0


def build_runtime_config(yaml_kwargs: dict[str, Any]) -> SciAccelRuntimeConfig:
    """
    Build config by merging YAML kwargs onto the structured schema.
    """
    raw = OmegaConf.to_container(OmegaConf.create(yaml_kwargs), resolve=True)
    if not isinstance(raw, dict):
        raw = {}
    raw.pop("name", None)
    raw.pop("_target_", None)

    schema = OmegaConf.structured(SciAccelRuntimeConfig)
    merged = OmegaConf.merge(schema, OmegaConf.create(raw))
    return OmegaConf.to_object(merged)
