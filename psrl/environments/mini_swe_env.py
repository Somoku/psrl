"""
mini-SWE-agent Environment for PSRL.

This environment adapts the mini-SWE-agent integration to PSRL's `Environment`
interface. It handles:
- `reset()`: Parse `DataProto` task data, build per-instance config, create workspace.
- `close()`: Clean up temporary directories and safety-net Docker container cleanup.

It does NOT use `step()` because mini-swe-agent manages its own tool execution
loop internally via `DefaultAgent.run()`.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
import numpy as np

import ray
from examples.mini_swe.config import (
    MiniSWEAgentRuntimeConfig,
    apply_data_overrides,
    build_runtime_config,
)
from omegaconf import DictConfig
from verl import DataProto

from psrl.environments.base import Environment, EnvStepOutput
from psrl.utils.common.docker_utils import cleanup_containers_by_label

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@Environment.register("mini_swe_env")
class MiniSWEEnvironment(Environment[dict, None]):
    """
    Environment adapter for mini-SWE-agent in-process episodes.

    Prepares per-episode workspaces, applies per-instance config overrides,
    and provides safety-net Docker cleanup. Docker container lifecycle is
    primarily managed by mini-swe-agent's `DockerEnvironment` internally.
    """

    def __init__(
        self,
        config: DictConfig,
        reward_manager: ray.actor.ActorHandle,
        runtime_config: MiniSWEAgentRuntimeConfig | None = None,
    ):
        """
        Initialize the mini-SWE-agent environment.

        Args:
            config (DictConfig): PSRL trainer configuration.
            reward_manager (ray.actor.ActorHandle): Ray actor handle for computing rewards.
            runtime_config (MiniSWEAgentRuntimeConfig | None): Pre-built runtime config
                from the agent loop. If None, a default config is built from scratch.
        """
        super().__init__(config, reward_manager)
        self._base_runtime_config = runtime_config
        self._swe_problem_id: str = ""
        self._runtime_config: MiniSWEAgentRuntimeConfig | None = None

    async def reset(self, task: DataProto, **kwargs) -> tuple[dict, dict]:
        """
        Parse task data and prepare per-episode workspace.

        Args:
            task (DataProto): `DataProto` containing the SWE task in `non_tensor_batch`.

        Returns:
            Tuple of (observation_dict, info_dict).
        """
        self.task = task

        # Log available keys for debugging data pipeline issues.
        ntb_keys = list(task.non_tensor_batch.keys()) if task.non_tensor_batch else []
        psrl_logger.info(f"MiniSWEEnvironment reset: non_tensor_batch keys={ntb_keys}.")

        # Extract extra_info if available (may be absent when the data pipeline
        # does not copy it to gen_batch — the reward_manager holds the original).
        extra_info_raw = task.non_tensor_batch.get("extra_info", [{}])[0]
        if isinstance(extra_info_raw, str):
            try:
                extra_info = json.loads(extra_info_raw)
            except json.JSONDecodeError:
                extra_info = {}
        elif isinstance(extra_info_raw, dict):
            extra_info = extra_info_raw
        else:
            extra_info = {}

        # Fall back to raw_prompt for problem_statement if extra_info is empty.
        problem_statement = extra_info.get("problem_statement", "") or ""
        if not problem_statement:
            raw_prompt = task.non_tensor_batch.get("raw_prompt", [None])[0]
            if raw_prompt is not None:
                if isinstance(raw_prompt, (list, np.ndarray)) and len(raw_prompt) > 0:
                    first_msg = raw_prompt[0] if isinstance(raw_prompt[0], dict) else {}
                    problem_statement = str(first_msg.get("content", ""))
                elif isinstance(raw_prompt, str):
                    problem_statement = raw_prompt

        repo_path = extra_info.get("repo_path", None)
        problem_swe_id = str(extra_info.get("instance_id", "") or "")
        sandbox_overrides = extra_info.get("sandbox_overrides", {}) or {}

        # Build runtime config from pre-built base or defaults.
        base_config = self._base_runtime_config or build_runtime_config({})
        self._runtime_config = apply_data_overrides(base_config, extra_info)

        # Generate unique swe_problem_id.
        self._swe_problem_id = (
            problem_swe_id
            if problem_swe_id
            else f"{uuid.uuid4().hex[:12]}-{int(time.time())}"
        )

        sb = self._runtime_config.sandbox_config

        # Determine repo type and preexisting repo name.
        use_preexisting_repo = bool(sandbox_overrides.get("use_preexisting_repo", False))
        preexisting_repo_name = str(sandbox_overrides.get("preexisting_repo_name", "") or "")
        if not use_preexisting_repo and not repo_path:
            use_preexisting_repo = True

        observation = {
            "problem_statement": problem_statement,
            "swe_problem_id": self._swe_problem_id,
            "runtime_config": self._runtime_config,
            "repo_path": repo_path,
            "use_preexisting_repo": use_preexisting_repo,
            "preexisting_repo_name": preexisting_repo_name,
        }

        _problem_short = f"{problem_statement[:80]}..." if len(problem_statement) > 80 else problem_statement
        psrl_logger.info(
            f"[mini-SWE-agent, id={self._swe_problem_id}] MiniSWEEnvironment reset: "
            f"problem={_problem_short!r}, "
            f"repo={preexisting_repo_name!r}, preexisting={use_preexisting_repo}."
        )

        return observation, {}

    async def step(self, action: None) -> EnvStepOutput:
        """
        Not used. mini-SWE-agent manages its own step loop internally.
        """
        raise NotImplementedError(
            "MiniSWEEnvironment does not support step(). "
            "mini-SWE-agent manages its own tool execution loop in-process."
        )

    async def close(self) -> None:
        """
        Safety-net cleanup for Docker containers and temporary directories.

        Primary Docker cleanup is handled by mini-swe-agent's
        `DockerEnvironment.cleanup()`. This method provides a fallback
        via label-based container cleanup in case the primary path fails.
        """
        if self._swe_problem_id:
            await cleanup_containers_by_label("psrl.swe_problem_id", self._swe_problem_id)

        psrl_logger.info(f"[mini-SWE-agent, id={self._swe_problem_id}] MiniSWEEnvironment closed.")

    @property
    def state(self) -> dict:
        """
        Return current environment state for debugging.
        """
        return {
            "swe_problem_id": self._swe_problem_id,
            "runtime_config": self._runtime_config,
        }
