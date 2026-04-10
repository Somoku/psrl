"""
mini-SWE-Agent Runtime Configuration & CLI YAML Builder.

Nested OmegaConf structured configs mirror the YAML layout so that
``build_runtime_config`` is just ``OmegaConf.merge(schema, yaml)``.
All defaults live in the dataclasses -- no module-level constant dicts.

Key differences from swe_agent config:
- Model API base lives under ``model.model_kwargs.api_base``.
- Step limit is ``agent.step_limit`` (not ``agent.model.per_instance_call_limit``).
- Docker labeling is via ``environment.run_args`` (not ``env.deployment.docker_args``).
- No ``env.repo`` or ``env.deployment`` structure; uses ``environment`` top-level key.
- Submission via ``echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` (no ``submit`` tool).
- Environment selection uses ``environment_class`` key (mini-swe-agent v2 API).
"""

from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


# ---------------------------------------------------------------------------
# Nested structured dataclasses — mirror the YAML layout
# ---------------------------------------------------------------------------


@dataclass
class ProxyConfig:
    port: int = 0
    max_port_retries: int = 1000
    timeout: int = 600


@dataclass
class MiniEnvironmentConfig:
    environment_class: str = "docker"
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
    swe_agent_timeout: int = 1800
    output_dir: str = ""
    max_parallel_tasks_per_worker: int = 0
    max_model_calls_per_instance: int = 15
    environment: MiniEnvironmentConfig = field(default_factory=MiniEnvironmentConfig)


# ---------------------------------------------------------------------------
# Default prompt templates for mini-SWE-agent
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_TEMPLATE = (
    "You are an expert software engineer working on a repository to fix issues.\n\n"
    "Your response must contain exactly ONE bash code block with ONE command "
    "(or commands connected with && or ||).\n"
    "Include a THOUGHT section before your command where you explain your reasoning.\n"
    "Format your response as shown in <format_example>.\n\n"
    "<format_example>\n"
    "Your reasoning and analysis here. Explain why you want to perform the action.\n\n"
    "```mswea_bash_command\n"
    "your_command_here\n"
    "```\n"
    "</format_example>\n\n"
    "Failure to follow this format will cause your response to be rejected.\n\n"
    "IMPORTANT RULES:\n"
    "- Make targeted, minimal changes to fix the issue.\n"
    "- Do NOT modify test files.\n"
    "- When you are done, submit by running ONLY this command:\n"
    "    echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n"
    "  Do NOT combine it with any other command.\n"
    "- You MUST submit before running out of steps."
)

DEFAULT_INSTANCE_TEMPLATE = (
    "Please solve this issue: {{ task }}\n\n"
    "You can execute bash commands and edit files to implement the necessary changes.\n\n"
    "## Recommended Workflow\n\n"
    "1. Explore the repository structure and find relevant files\n"
    "2. Identify the root cause of the issue\n"
    "3. Make the minimal necessary changes\n"
    "4. Verify your changes with tests if possible\n"
    "5. Submit: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`\n"
    "   Do not combine it with any other command.\n\n"
    "## Important Rules\n\n"
    "1. Every response must contain exactly one action in triple backticks.\n"
    "2. Directory or environment variable changes are not persistent. "
    "Every action runs in a new subshell.\n"
    "   Use `cd /path && ...` to work in a specific directory.\n\n"
    "<system_information>\n"
    "{{ system }} {{ release }} {{ version }} {{ machine }}\n"
    "</system_information>\n\n"
    "## Useful command examples\n\n"
    "### Create a new file:\n\n"
    "```mswea_bash_command\n"
    "cat <<'EOF' > newfile.py\n"
    "import numpy as np\n"
    "hello = \"world\"\n"
    "print(hello)\n"
    "EOF\n"
    "```\n\n"
    "### Edit files with sed:\n\n"
    "```mswea_bash_command\n"
    "sed -i 's/old_string/new_string/g' filename.py\n"
    "```\n\n"
    "### View file content:\n\n"
    "```mswea_bash_command\n"
    "nl -ba filename.py | sed -n '10,20p'\n"
    "```"
)


@dataclass
class MiniAgentConfig:
    step_limit: int = 15
    cost_limit: float = 0.0
    system_template: str = field(default_factory=lambda: DEFAULT_SYSTEM_TEMPLATE)
    instance_template: str = field(default_factory=lambda: DEFAULT_INSTANCE_TEMPLATE)


@dataclass
class MiniModelConfig:
    model_name: str = "openai/verl-model"
    model_class: str = "litellm_textbased"
    cost_tracking: str = "ignore_errors"
    model_kwargs: dict = field(
        default_factory=lambda: {
            "api_key": "verl-mini-swe-agent-key",
            "temperature": 0.0,
            "drop_params": True,
        }
    )


@dataclass
class MiniSWEAgentRuntimeConfig:
    """
    Top-level config. Nesting matches the YAML structure.
    """

    proxy_config: ProxyConfig = field(default_factory=ProxyConfig)
    sandbox_config: MiniSandboxConfig = field(default_factory=MiniSandboxConfig)
    agent: MiniAgentConfig = field(default_factory=MiniAgentConfig)
    model: MiniModelConfig = field(default_factory=MiniModelConfig)


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

    out = cfg.sandbox_config.output_dir or os.path.join(os.getcwd(), "mini_swe_agent_outputs")
    cfg.sandbox_config.output_dir = os.path.abspath(os.path.expanduser(out))
    os.makedirs(cfg.sandbox_config.output_dir, exist_ok=True)
    return cfg


# ---------------------------------------------------------------------------
# Per-instance overrides
# ---------------------------------------------------------------------------

_SANDBOX_FIELDS = frozenset(MiniSandboxConfig.__dataclass_fields__)
_AGENT_OVERRIDE_FIELDS = frozenset(("step_limit", "cost_limit", "system_template", "instance_template"))


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


# ---------------------------------------------------------------------------
# mini-SWE-Agent CLI YAML Builder
# ---------------------------------------------------------------------------


def build_mini_sweagent_yaml(
    cfg: MiniSWEAgentRuntimeConfig,
    *,
    swe_problem_id: str,
    repo_path: str,
    model_proxy_port: int,
    max_input_tokens: int = 0,
    repo_type: str = "preexisting",
    preexisting_repo_name: str = "",
) -> str:
    """
    Generate the YAML string consumed by ``mini-swe-agent -c <yaml>``.

    Args:
        cfg (MiniSWEAgentRuntimeConfig): Runtime configuration.
        swe_problem_id (str): Unique SWE problem identifier (used for Docker label).
        repo_path (str): For ``repo_type="local"``, path on host to mount as /testbed.
            For ``repo_type="preexisting"``, ignored (repo already in image).
        model_proxy_port (int): Port where `ModelProxy` is listening.
        max_input_tokens (int): If >0, added as ``max_tokens`` in ``model_kwargs``.
        repo_type (str): ``"preexisting"`` (repo in Docker image) or ``"local"`` (mount).
        preexisting_repo_name (str): Name of the pre-baked repo directory inside the
            Docker image (e.g. ``"train_0"``). When non-empty and ``repo_type`` is
            ``"preexisting"``, overrides ``environment.cwd`` to ``"/{name}"``.
    """
    sb = cfg.sandbox_config
    ag = cfg.agent
    mod = cfg.model
    env = sb.environment

    # Build run_args -- start from config base, then add instance label.
    run_args = list(env.run_args)

    # Add instance label for cleanup.
    run_args.extend(["--label", f"psrl.swe_problem_id={swe_problem_id}"])

    # For local repo, mount it into /testbed.
    if repo_type == "local" and repo_path:
        run_args.extend(["--volume", f"{repo_path}:/testbed"])

    # For preexisting repo, override cwd to the baked repo directory.
    effective_cwd = env.cwd
    if repo_type == "preexisting" and preexisting_repo_name:
        effective_cwd = f"/{preexisting_repo_name}"

    # Build model_kwargs.
    model_kwargs: dict[str, Any] = deepcopy(mod.model_kwargs if isinstance(mod.model_kwargs, dict) else {})
    model_kwargs["api_base"] = f"http://127.0.0.1:{model_proxy_port}/v1"
    model_kwargs.setdefault("api_key", "verl-mini-swe-agent-key")
    model_kwargs.setdefault("temperature", 0.0)
    if max_input_tokens > 0:
        model_kwargs["max_tokens"] = max_input_tokens

    config = {
        "agent": {
            "step_limit": ag.step_limit,
            "cost_limit": ag.cost_limit,
            "system_template": ag.system_template,
            "instance_template": ag.instance_template,
        },
        "model": {
            "model_name": mod.model_name,
            "model_class": mod.model_class,
            "model_kwargs": model_kwargs,
            "cost_tracking": mod.cost_tracking,
        },
        "environment": {
            "environment_class": env.environment_class,
            "image": env.image,
            "cwd": effective_cwd,
            "env": env.env if isinstance(env.env, dict) else {},
            "run_args": run_args,
            "container_timeout": env.container_timeout,
        },
    }

    return yaml.dump(config, default_flow_style=False, allow_unicode=True)
