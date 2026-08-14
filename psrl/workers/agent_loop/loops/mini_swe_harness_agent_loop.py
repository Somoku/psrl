"""Mini-SWE task hooks for the generic harness agent loop."""

import asyncio
import contextlib
import logging
import os
import time
from dataclasses import asdict, dataclass

from examples.mini_swe.config import MINI_SWE_SLOT_PREFIX, MiniSWEAgentRuntimeConfig, build_runtime_config
from examples.mini_swe.harness_task import build_harness_prompt, collect_git_patch
from examples.mini_swe.runner import build_grader_spec, build_sandbox_spec, grade_patch

from psrl.environments import Environment
from psrl.sandbox import SandboxSession, SnapshotRef, SyncSandboxManager
from psrl.utils.concurrency import SlotManager
from psrl.workers.agent_loop.context import AgentLoopContext
from psrl.workers.agent_loop.harness import HarnessRuntime, HarnessTaskContext
from psrl.workers.agent_loop.loops.harness_agent_loop import HarnessAgentLoop
from psrl.workers.agent_loop.loops.utils import register

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@dataclass(frozen=True)
class MiniSWEHarnessTaskState:
    """
    Store Mini-SWE resources and grader payload owned by the task hook.
    """

    environment: Environment
    payload: dict
    run_slot: tuple[int, int] | None


@register("mini_swe_harness")
class MiniSWEHarnessAgentLoop(HarnessAgentLoop):
    """
    Supply Mini-SWE environment, patch, grader, and reward hooks.
    """

    def __init__(self, context: AgentLoopContext, **kwargs) -> None:
        if "harness" not in kwargs:
            raise ValueError("mini-SWE harness loop requires a harness configuration.")
        super().__init__(context=context, harness=kwargs["harness"])
        runtime_kwargs = {key: kwargs[key] for key in ("sandbox_config", "agent", "model") if key in kwargs}
        self.runtime_config: MiniSWEAgentRuntimeConfig = build_runtime_config(
            runtime_kwargs,
            require_agent_templates=False,
        )

    async def prepare_harness_task(self, request: dict) -> HarnessTaskContext[MiniSWEHarnessTaskState]:
        """
        Prepare Mini-SWE task data without taking ownership of generic resources.
        """
        env_class = request.get("env_class", self.config.gen_actor_rollout_ref.rollout.agent.env.name)
        environment = Environment.get_environment(
            env_class,
            self.config,
            self.reward_manager,
            tokenizer=self.tokenizer,
            processor=self.processor,
            dataset_cls=self.dataset_cls,
            runtime_config=self.runtime_config,
        )
        run_slot: tuple[int, int] | None = None
        try:
            observation, _ = await environment.reset(task=request, seed=request.get("seed"))
            runtime_config = observation["runtime_config"]
            run_slot = await self._acquire_run_slot(runtime_config)
            payload = self._build_task_payload(observation, runtime_config)
            sandbox_spec = build_sandbox_spec(payload)
            grader_spec = build_grader_spec(payload)
            state = MiniSWEHarnessTaskState(
                environment=environment,
                payload=payload,
                run_slot=run_slot,
            )
            return HarnessTaskContext(
                state=state,
                prompt=build_harness_prompt(observation.get("problem_statement", "")),
                sandbox_spec=sandbox_spec,
                backend=runtime_config.sandbox_config.backend,
                clean_sandbox_spec=grader_spec,
                collect_resource_metrics=runtime_config.sandbox_config.collect_resource_metrics,
            )
        except BaseException:
            SlotManager.release(run_slot)
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.shield(environment.close())
            raise

    async def collect_harness_artifact(
        self,
        task: HarnessTaskContext[MiniSWEHarnessTaskState],
        sandbox: SandboxSession,
        runtime: HarnessRuntime,
    ) -> str:
        """
        Capture the repository patch before the rollout sandbox is destroyed.
        """
        return await collect_git_patch(sandbox, runtime.workdir)

    async def finalize_harness_task(
        self,
        task: HarnessTaskContext[MiniSWEHarnessTaskState],
        artifact: str,
        clean_snapshot: SnapshotRef | None,
        timing: dict[str, float],
    ) -> dict:
        """
        Grade the Mini-SWE patch and return reward-specific metadata.
        """
        grading_started = time.perf_counter()
        grader_result = await self._grade_patch(task, artifact, clean_snapshot)
        timing["grading_s"] = time.perf_counter() - grading_started
        result = grader_result or {}
        return {
            "patch": artifact or None,
            "alignment_failed": False,
            "alignment_failure_reason": "",
            "grader_result": result,
            "acc": float(bool(result.get("resolved", False))),
        }

    async def close_harness_task(self, task: HarnessTaskContext[MiniSWEHarnessTaskState]) -> None:
        """
        Close the Mini-SWE environment and release its concurrency slot.
        """
        try:
            await task.state.environment.close()
        finally:
            SlotManager.release(task.state.run_slot)

    async def _acquire_run_slot(self, runtime_config: MiniSWEAgentRuntimeConfig) -> tuple[int, int] | None:
        parallelism = runtime_config.sandbox_config.max_parallel_tasks_per_worker
        if parallelism <= 0:
            return None
        namespace = os.path.join(
            str(self.config.trainer.project_name),
            str(self.config.trainer.experiment_name),
        )
        return await SlotManager.acquire(parallelism, namespace, prefix=MINI_SWE_SLOT_PREFIX)

    def _build_task_payload(
        self,
        observation: dict,
        runtime_config: MiniSWEAgentRuntimeConfig,
    ) -> dict:
        return {
            "observation": {key: value for key, value in observation.items() if key != "runtime_config"},
            "runtime_config": {"sandbox_config": asdict(runtime_config.sandbox_config)},
        }

    async def _grade_patch(
        self,
        task: HarnessTaskContext[MiniSWEHarnessTaskState],
        patch: str,
        clean_snapshot: SnapshotRef | None,
    ) -> dict | None:
        if not patch:
            return None
        sync_sandbox: SyncSandboxManager = self.sandbox_manager.sync(backend=task.backend)
        grader_task = asyncio.create_task(
            asyncio.to_thread(
                grade_patch,
                task.state.payload,
                patch,
                sync_sandbox,
                clean_snapshot,
                task.clean_sandbox_spec,
            )
        )
        try:
            return await asyncio.shield(grader_task)
        except asyncio.CancelledError:
            await sync_sandbox.aclose()
            with contextlib.suppress(Exception):
                await asyncio.shield(grader_task)
            raise
        finally:
            await sync_sandbox.aclose()
