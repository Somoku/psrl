"""
MLGym environment backed by a PSRL sandbox lease.

MLGym's `BaseAgent` is used unmodified. It reaches its environment only through
`MLGymEnv.communicate()` and `MLGymEnv.step()`, both of which funnel into the
low-level `_communicate`. Subclassing lets us replace the container and its shell
with a leased sandbox while every other MLGym behavior, including workspace setup,
conda activation, baseline scoring, and submission grading, is inherited intact.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

import yaml
from examples.airs_bench.config import AirsBenchRuntimeConfig, build_runtime_config
from omegaconf import DictConfig

from psrl.environments.base import Environment, EnvStepOutput
from psrl.sandbox import (
    MountSpec,
    ResourceSpec,
    SandboxSource,
    SandboxSpec,
    SyncSandboxSession,
)
from psrl.sandbox.capacity import parse_memory_mb

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

REQUIRED_HISTORY_PROCESSOR = "DefaultHistoryProcessor"
AGENT_WORKSPACE = "/home/agent/workspace"


def assert_default_history_processor(agent_config_path: str) -> None:
    """
    Fail loudly unless an MLGym agent config pins the safe history processor.

    Any processor that rewrites earlier messages, such as `Last5Observations`,
    invalidates TITO's message prefix hashes. The corruption is silent, producing
    misaligned training tokens rather than an error, so it is checked up front.

    Args:
        agent_config_path (str): Path to the MLGym agent config yaml.

    Raises:
        ValueError: If the key is absent or set to anything else.
    """
    payload = yaml.safe_load(Path(agent_config_path).read_text()) or {}
    processor = payload.get("history_processor")
    if processor is None:
        raise ValueError(
            f"Agent config {agent_config_path!r} does not set history_processor, "
            f"which must be {REQUIRED_HISTORY_PROCESSOR} for TITO correctness."
        )
    if processor != REQUIRED_HISTORY_PROCESSOR:
        raise ValueError(
            f"Agent config {agent_config_path!r} sets history_processor to {processor!r}. "
            f"RL training requires {REQUIRED_HISTORY_PROCESSOR} because any processor that "
            "rewrites earlier messages silently corrupts TITO training data."
        )


def build_sandbox_spec(
    runtime_config: AirsBenchRuntimeConfig,
    dataset_data_path: str,
    labels: dict[str, str],
    *,
    episode_id: str,
) -> SandboxSpec:
    """
    Build the portable sandbox request for one AIRS-Bench episode.

    Task data is mounted read only so the agent cannot reach the hidden test labels
    by editing them, while still letting the in-sandbox `evaluate.py` read them.

    `airs_bench` is a policy profile rather than a spec field, because host
    networking and loopback proxy rewriting are node policy and not a portable
    property of the workload.

    Args:
        runtime_config (AirsBenchRuntimeConfig): Recipe runtime settings.
        dataset_data_path (str): Host path to this task's prepared data.
        labels (dict[str, str]): Container labels for cleanup and diagnostics.
        episode_id (str): Episode identity, which also scopes the sandbox lease.

    Returns:
        SandboxSpec: Spec ready for the sandbox manager.
    """
    proxy_env = {
        key: os.environ[key]
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy", "NO_PROXY")
        if key in os.environ
    }
    return SandboxSpec(
        source=SandboxSource.image(runtime_config.image),
        resources=ResourceSpec(
            cpu_count=runtime_config.sandbox_cpus,
            memory_mb=parse_memory_mb(runtime_config.sandbox_memory),
        ),
        mounts=(MountSpec(source=dataset_data_path, target=f"{AGENT_WORKSPACE}/data", read_only=True),),
        env=proxy_env,
        metadata=labels,
        policy_profile="airs_bench",
        # One sandbox per episode, so the episode is the workflow phase that the
        # reservation protects.
        workflow_id=episode_id,
        idempotency_key=f"{episode_id}:airs",
        # An AIRS-Bench episode is long and its cap is the wall clock, not idleness.
        lifetime_timeout_s=runtime_config.episode_timeout_s,
    )


def build_psrl_mlgym_env_class() -> type:
    """
    Build the `PSRLMLGymEnv` class, importing MLGym lazily.

    MLGym lives outside this repo and is imported by path, so the subclass is
    created on demand rather than at module import time. This keeps `psrl` importable
    on machines where MLGym is absent, which matters for CPU-only unit tests.

    Returns:
        type: A subclass of MLGym's `MLGymEnv`.
    """
    from mlgym.environment.env import MLGymEnv

    class PSRLMLGymEnv(MLGymEnv):  # type: ignore[misc]
        """MLGym environment whose container lives in a leased PSRL sandbox."""

        def __init__(
            self,
            args: Any,
            session: SyncSandboxSession,
            per_action_timeout_s: float,
            **kwargs: Any,
        ):
            self._session = session
            self._per_action_timeout_s = per_action_timeout_s
            super().__init__(args, devices=["cpu_0"], **kwargs)

        def _init_container(self, cached_image: str | None = None) -> None:
            """
            Adopt the already-running sandbox instead of starting a local container.

            The lease is taken by the agent loop before the agent starts, so there is
            nothing to launch here. `container` and `container_obj` are set to None
            because every consumer of them is routed through `_communicate` in this
            subclass.
            """
            self.container = None
            self.container_obj = None
            self.container_name = self._session.ref.sandbox_id
            psrl_logger.info(f"Adopted sandbox {self._session.ref.sandbox_id!r} for MLGym.")

        def _communicate(
            self,
            input: str,
            timeout_duration: float = 25,
            no_output_timeout_duration: float = 25,
        ) -> str:
            """
            Execute one command in the leased sandbox.

            Overriding this single method reroutes every MLGym code path that talks to
            the container, because `communicate`, `step`, workspace setup, and grading
            all funnel through here. The sync facade schedules onto the loop that owns
            the sandbox and blocks this thread, which is what makes a synchronous agent
            usable against an asynchronous session.
            """
            timeout_s = min(float(timeout_duration), self._per_action_timeout_s)
            try:
                result = self._session.exec(
                    input,
                    timeout_s=timeout_s,
                    silence_timeout_s=float(no_output_timeout_duration),
                )
            except TimeoutError as exc:
                # MLGym's own contract: a training command that overruns raises here and
                # the agent sees a timeout observation rather than losing the episode.
                raise TimeoutError(f"Sandbox command exceeded {timeout_s}s: {input[:120]!r}.") from exc
            self.returncode = result.exit_code
            # MLGym reads one combined stream, so a provider that separates the two is
            # joined here rather than losing the diagnostics.
            return result.stdout + result.stderr

        def close(self) -> None:
            """Report closure. The lease owns container teardown."""
            psrl_logger.info(f"Closing MLGym env for sandbox {self._session.ref.sandbox_id!r}.")

    return PSRLMLGymEnv


@Environment.register("mlgym_env")
class MLGymEnvironment(Environment[dict, None]):
    """
    PSRL environment adapter for AIRS-Bench tasks.

    Like `MiniSWEEnvironment`, this prepares per-episode state and hands it to the
    agent loop. `ActType` is None because MLGym's own agent owns the step loop, so
    `step` is never called.
    """

    def __init__(
        self,
        config: DictConfig,
        reward_manager: Any,
        tokenizer: Any = None,
        processor: Any = None,
        dataset_cls: Any = None,
        runtime_config: AirsBenchRuntimeConfig | None = None,
        **kwargs: Any,
    ):
        super().__init__(config, reward_manager, tokenizer, processor, dataset_cls)
        self._runtime_config = runtime_config or build_runtime_config(None)
        self._task_id: str | None = None
        self._episode_id: str | None = None

    async def reset(self, task: dict, **kwargs: Any) -> tuple[dict, dict]:
        """
        Prepare one AIRS-Bench episode.

        Args:
            task (dict): Dataset row, whose `extra_info` carries the task identity and
                its normalization constants.
            **kwargs (Any): Ignored framework extras.

        Returns:
            tuple[dict, dict]: Observation for the agent loop, plus an empty info dict.
        """
        extra_info = task.get("extra_info") or {}
        self._task_id = extra_info.get("airs_task_id")
        self._episode_id = f"{uuid.uuid4().hex[:12]}-{int(time.time())}"

        if not self._task_id:
            raise ValueError("Dataset row is missing extra_info.airs_task_id.")
        assert_default_history_processor(self._runtime_config.agent_config_path)

        observation = {
            "airs_task_id": self._task_id,
            "episode_id": self._episode_id,
            "task_config_path": extra_info["task_config_path"],
            "dataset_data_path": extra_info["dataset_data_path"],
            "metric": extra_info["metric"],
            "runtime_config": self._runtime_config,
        }
        return observation, {}

    async def step(self, action: None, **kwargs: Any) -> EnvStepOutput:
        """Never called. MLGym's own agent owns the step loop."""
        raise NotImplementedError("MLGymEnvironment does not implement step because MLGym's agent owns the loop.")

    async def close(self) -> None:
        """Force-remove any container still labelled with this episode id."""
        if self._episode_id is None:
            return
        from psrl.sandbox.backends.docker.cli import force_remove_containers_by_label

        await asyncio.to_thread(force_remove_containers_by_label, "psrl.airs_episode_id", self._episode_id)

    @property
    def state(self) -> Any:
        """Return the current episode identity."""
        return {"airs_task_id": self._task_id, "episode_id": self._episode_id}
