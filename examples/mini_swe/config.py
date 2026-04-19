"""
mini-SWE-agent Runtime Configuration for PSRL.

Dataclass-based config for the mini-SWE-agent integration. These configs
control the PSRL-side orchestration (sandbox timeouts, parallelism, templates).
mini-swe-agent's own components (`DockerEnvironment`, `DefaultAgent`,
`LitellmTextbasedModel`) are configured directly via their Python APIs
in the agent loop -- no YAML generation is needed.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from omegaconf import DictConfig, OmegaConf

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


# ---------------------------------------------------------------------------
# Nested structured dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MiniEnvironmentConfig:
    """
    Docker environment settings passed to `DockerEnvironment(**kwargs)`.
    """

    image: str = "python:3.11-slim"
    cwd: str = "/testbed"
    env: dict = field(
        default_factory=lambda: {
            "PAGER": "cat",
            "MANPAGER": "cat",
            "LESS": "-R",
            "PIP_PROGRESS_BAR": "off",
            "TQDM_DISABLE": "1",
        }
    )
    run_args: list = field(
        default_factory=lambda: [
            "--rm",
            "--memory=8g",
            "--network",
            "host",
            "--add-host",
            "host.docker.internal:host-gateway",
        ]
    )
    container_timeout: str = "2h"


@dataclass
class MiniSandboxConfig:
    """
    PSRL-side orchestration settings (not passed to mini-swe-agent directly).
    """

    max_parallel_tasks_per_worker: int = 0
    environment: MiniEnvironmentConfig = field(default_factory=MiniEnvironmentConfig)


@dataclass
class MiniAgentConfig:
    """
    Agent settings passed to `DefaultAgent(**kwargs)`.

    `system_template` and `problem_template` are required by mini-swe-agent's
    `AgentConfig` (no defaults in upstream). `problem_template` maps to
    mini-swe-agent's `instance_template` kwarg.

    Both templates should be provided via mini_swe_agent_config.yaml; the
    empty-string defaults here are intentional -- `build_runtime_config` will
    raise if they remain unset after YAML merge.
    """

    cost_limit: float = 0.0
    system_template: str = ""
    problem_template: str = ""


@dataclass
class MiniSWEAgentRuntimeConfig:
    """
    Top-level config for the mini-SWE-agent PSRL integration.
    """

    sandbox_config: MiniSandboxConfig = field(default_factory=MiniSandboxConfig)
    agent: MiniAgentConfig = field(default_factory=MiniAgentConfig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dict(val: Any) -> dict:
    """
    Coerce `val` to a plain dict, handling JSON strings and `OmegaConf`.
    """
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return {}
    if isinstance(val, DictConfig):
        result = OmegaConf.to_container(val, resolve=True)
        return result if isinstance(result, dict) else {}
    if isinstance(val, dict):
        return val
    return {}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_runtime_config(yaml_kwargs: dict[str, Any]) -> MiniSWEAgentRuntimeConfig:
    """
    Build config by merging YAML kwargs onto the `OmegaConf` structured schema.
    """
    raw = OmegaConf.to_container(OmegaConf.create(yaml_kwargs), resolve=True)
    if not isinstance(raw, dict):
        raw = {}
    raw.pop("name", None)
    raw.pop("_target_", None)

    schema = OmegaConf.structured(MiniSWEAgentRuntimeConfig)
    merged = OmegaConf.merge(schema, OmegaConf.create(raw))
    cfg: MiniSWEAgentRuntimeConfig = OmegaConf.to_object(merged)  # type: ignore[assignment]

    if not cfg.agent.system_template or not cfg.agent.problem_template:
        raise ValueError(
            "agent.system_template and agent.problem_template must be provided "
            "in mini_swe_agent_config.yaml (they have no hardcoded defaults)."
        )

    return cfg


# ---------------------------------------------------------------------------
# Per-instance overrides
# ---------------------------------------------------------------------------

_SANDBOX_FIELDS = frozenset(MiniSandboxConfig.__dataclass_fields__)
_AGENT_OVERRIDE_FIELDS = frozenset(("cost_limit", "system_template", "problem_template"))


def apply_data_overrides(
    base: MiniSWEAgentRuntimeConfig,
    extra_info: dict[str, Any],
) -> MiniSWEAgentRuntimeConfig:
    """
    Per-instance copy of `base` with data-affine overrides. `base` is not mutated.
    """
    sandbox_ov = _ensure_dict(extra_info.get("sandbox_overrides", {}))
    agent_ov = _ensure_dict(extra_info.get("agent_overrides", {}))
    if not sandbox_ov and not agent_ov:
        return base

    patch: dict[str, Any] = {}
    if sandbox_ov:
        sandbox_patch = {k: v for k, v in sandbox_ov.items() if k in _SANDBOX_FIELDS}
        if sandbox_patch:
            patch["sandbox_config"] = sandbox_patch
    if agent_ov:
        agent_patch = {k: v for k, v in agent_ov.items() if k in _AGENT_OVERRIDE_FIELDS}
        if agent_patch:
            patch["agent"] = agent_patch

    if not patch:
        return base

    base_cfg = OmegaConf.structured(base)
    return OmegaConf.to_object(OmegaConf.merge(base_cfg, OmegaConf.create(patch)))  # type: ignore[return-value]
