"""
mini-SWE-Agent Environment for PSRL.

This environment adapts the mini-SWE-agent subprocess lifecycle to PSRL's
`Environment` interface. It does NOT use the traditional `step()` loop
because mini-SWE-agent is a black box that manages its own Docker containers
and tool execution internally.

Responsibilities:
- `reset()`: Parse task data, generate swe_problem_id, create workspace.
- `extract_patch()`: Extract patch from subprocess output.
- `close()`: Cleanup Docker containers and temp files.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from typing import Any

import ray
from examples.mini_swe.config import (
    MiniSWEAgentRuntimeConfig,
    apply_data_overrides,
    build_runtime_config,
)
from psrl.utils.common.patch_extractor import PatchExtractor
from omegaconf import DictConfig
from verl import DataProto

from psrl.environments.base import Environment, EnvStepOutput
from psrl.utils.common.docker_utils import cleanup_containers_by_label

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@Environment.register("mini_swe_env")
class MiniSWEEnvironment(Environment[dict, None]):
    """
    Environment adapter for mini-SWE-agent subprocess-based episodes.

    This environment prepares per-episode workspaces, extracts patches
    after completion, and cleans up Docker containers. It does not implement
    `step()` because mini-SWE-agent manages its own tool execution loop.
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
        self._output_dir: str = ""
        self._exec_dir: str = ""
        self._trajectory_json_path: str = ""
        self._patch: str | None = None
        self._runtime_config: MiniSWEAgentRuntimeConfig | None = None

    async def reset(self, task: DataProto, **kwargs) -> tuple[dict, dict]:
        """
        Parse task data and prepare per-episode workspace.

        Extracts problem_statement, repo_path, and swe_problem_id from the
        `DataProto`'s `non_tensor_batch`. Creates output and execution directories.

        Args:
            task (DataProto): `DataProto` containing the SWE task in `non_tensor_batch`.

        Returns:
            Tuple of (observation_dict, info_dict).
        """
        self.task = task
        self._patch = None

        # Extract extra_info from task.
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

        problem_statement = extra_info.get("problem_statement", "") or ""
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

        # Create workspace directories.
        sb = self._runtime_config.sandbox_config
        self._output_dir = os.path.join(sb.output_dir, self._swe_problem_id)
        os.makedirs(self._output_dir, exist_ok=True)
        self._exec_dir = tempfile.mkdtemp(prefix=f"mini_swe_exec_{self._swe_problem_id}_")
        self._trajectory_json_path = os.path.join(
            self._output_dir, f"{self._swe_problem_id}.traj.json",
        )

        # Determine repo type and preexisting repo name.
        use_preexisting_repo = bool(sandbox_overrides.get("use_preexisting_repo", False))
        preexisting_repo_name = str(sandbox_overrides.get("preexisting_repo_name", "") or "")
        if not use_preexisting_repo and not repo_path:
            use_preexisting_repo = True

        observation = {
            "problem_statement": problem_statement,
            "swe_problem_id": self._swe_problem_id,
            "runtime_config": self._runtime_config,
            "output_dir": self._output_dir,
            "repo_path": repo_path,
            "use_preexisting_repo": use_preexisting_repo,
            "preexisting_repo_name": preexisting_repo_name,
        }

        psrl_logger.info(
            f"[{self._swe_problem_id}] MiniSWEEnvironment reset: "
            f"problem={problem_statement[:80]!r}..."
        )

        return observation, {}

    async def step(self, action: None) -> EnvStepOutput:
        """
        Not used. mini-SWE-agent manages its own step loop internally.
        """
        raise NotImplementedError(
            "MiniSWEEnvironment does not support step(). "
            "mini-SWE-agent manages its own tool execution loop as a subprocess."
        )

    async def extract_patch(self) -> str | None:
        """
        Extract the generated patch from mini-SWE-agent output.

        Tries trajectory JSON first, then falls back to git diff.

        Returns:
            Patch string or None if no patch was produced.
        """
        repo_path = ""
        if self.task is not None:
            extra_info_raw = self.task.non_tensor_batch.get("extra_info", [{}])[0]
            if isinstance(extra_info_raw, dict):
                repo_path = extra_info_raw.get("repo_path", "") or ""

        extractor = PatchExtractor(
            output_dir=self._output_dir,
            swe_problem_id=self._swe_problem_id,
            repo_path=repo_path,
            trajectory_json_path=self._trajectory_json_path or None,
        )
        self._patch = await extractor.extract()
        return self._patch

    async def close(self) -> None:
        """
        Cleanup Docker containers and temporary directories.
        """
        if self._swe_problem_id:
            await cleanup_containers_by_label("psrl.swe_problem_id", self._swe_problem_id)

        if self._exec_dir and os.path.isdir(self._exec_dir):
            shutil.rmtree(self._exec_dir, ignore_errors=True)

        psrl_logger.info(f"[{self._swe_problem_id}] MiniSWEEnvironment closed.")

    @property
    def state(self) -> dict:
        """
        Return current environment state for debugging.
        """
        return {
            "swe_problem_id": self._swe_problem_id,
            "output_dir": self._output_dir,
            "patch": self._patch,
            "runtime_config": self._runtime_config,
        }
