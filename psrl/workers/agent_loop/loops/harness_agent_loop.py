"""Generic task lifecycle for sandboxed coding harness training."""

import asyncio
import logging
import os
import time
from abc import abstractmethod
from collections.abc import Awaitable, Mapping
from typing import Any

from omegaconf import DictConfig

from psrl.sandbox import SandboxFeature, SandboxLease, SandboxSession, SnapshotKind, SnapshotRef
from psrl.workers.agent_loop.context import AgentLoopContext
from psrl.workers.agent_loop.harness import (
    Harness,
    HarnessConfig,
    HarnessResult,
    HarnessRuntime,
    HarnessTaskContext,
    clean_snapshot_compatible,
    create_harness,
)
from psrl.workers.agent_loop.loops.session_agent_loop import SessionAgentLoop
from psrl.workers.agent_loop.loops.utils import TerminateReason
from psrl.workers.gen.utils import TokenOutput

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class HarnessAgentLoop(SessionAgentLoop):
    """
    Run one task-scoped harness and turn its TITO session into training data.

    Subclasses define task preparation and may override artifact collection,
    post-rollout finalization, and task cleanup. This class remains the sole
    owner of the TITO session, sandbox lease, harness process, and snapshot.
    """

    def __init__(
        self,
        context: AgentLoopContext,
        harness: HarnessConfig | DictConfig | Mapping[str, Any],
    ) -> None:
        super().__init__(context=context)
        self.harness_config = HarnessConfig.from_value(harness)
        multi_turn = context.config.gen_actor_rollout_ref.rollout.multi_turn
        if not getattr(multi_turn, "enable", False):
            raise ValueError("Harness training requires rollout.multi_turn.enable=True.")
        if self.trajectory_id_strategy != "auto":
            raise ValueError(
                "Harness training requires psrl.rollout_gateway.trajectory_id_strategy=auto so TITO can assign "
                "multi-agent and compaction branches from its prefix tree."
            )
        if self.sandbox_manager is None:
            raise ValueError("Harness training requires rollout.agent.sandbox.default_backend.")
        if context.config.gen_actor_rollout_ref.rollout.agent.traj_reward_mode != "traj":
            raise ValueError("Harness training supports only agent.traj_reward_mode=traj.")

    @abstractmethod
    async def prepare_harness_task(self, request: dict) -> HarnessTaskContext:
        """
        Build task state, prompt, and sandbox specifications.

        A subclass that acquires partial resources before returning must release
        them if preparation raises. Once returned, `close_harness_task` is always
        called by the generic lifecycle.
        """

    async def collect_harness_artifact(
        self,
        task: HarnessTaskContext,
        sandbox: SandboxSession,
        runtime: HarnessRuntime,
    ) -> Any:
        """
        Collect an optional task artifact before the agent sandbox is destroyed.
        """
        return None

    async def finalize_harness_task(
        self,
        task: HarnessTaskContext,
        artifact: Any,
        clean_snapshot: SnapshotRef | None,
        timing: dict[str, float],
    ) -> dict:
        """
        Run optional task evaluation and return task-specific reward fields.
        """
        return {}

    async def close_harness_task(self, task: HarnessTaskContext) -> None:
        """
        Release task-specific state after generic resources are cleaned up.
        """
        return None

    async def run(
        self,
        request: dict,
    ) -> tuple[TokenOutput | list[TokenOutput] | None, TerminateReason]:
        """
        Execute the generic harness lifecycle for one training task.
        """
        task: HarnessTaskContext | None = None
        session_id: str | None = None
        lease: SandboxLease | None = None
        harness: Harness | None = None
        clean_snapshot: SnapshotRef | None = None
        run_start = time.perf_counter()
        timing = {"prep_s": 0.0, "assistant_s": 0.0, "env_s": 0.0, "elapsed_s": 0.0}
        try:
            task = await self.prepare_harness_task(request)
            timing["task_prepare_s"] = time.perf_counter() - run_start
            session_id = await self.create_session(request)
            sandbox_started = time.perf_counter()
            # Retain lease ownership if cancellation races sandbox creation.
            acquire_task = asyncio.create_task(
                self.sandbox_manager.acquire(task.sandbox_spec, backend=task.backend),
            )
            try:
                lease = await asyncio.shield(acquire_task)
            except asyncio.CancelledError:
                lease = await acquire_task
                raise
            timing["sandbox_create_s"] = time.perf_counter() - sandbox_started
            clean_snapshot = await self._try_snapshot_clean_sandbox(task, lease)

            session_root_url = self.session_root_url(session_id, self.harness_config.callback_base_url)
            session_root_url = lease.session.resolve_callback_url(session_root_url)
            harness = create_harness(self.harness_config, lease.session)
            harness_runtime = HarnessRuntime(
                session_id=session_id,
                session_root_url=session_root_url,
                workdir=task.sandbox_spec.workdir or "/",
                model=str(self.model_config.path),
            )
            await harness.prepare(harness_runtime)
            timing["prep_s"] = time.perf_counter() - run_start

            harness_started = time.perf_counter()
            try:
                harness_result = await harness.run(task.prompt, harness_runtime)
            except TimeoutError:
                return None, TerminateReason.TRAJECTORY_TIMEOUT
            timing["assistant_s"] = time.perf_counter() - harness_started

            artifact = await self.collect_harness_artifact(task, lease.session, harness_runtime)
            await self._capture_resource_metrics(task, lease, timing)
            training_data = await self.get_training_data(session_id)

            await lease.release()
            lease = None
            await self.delete_session(session_id)
            session_id = None

            if not training_data or any(item["num_turns"] <= 0 or not item["response_ids"] for item in training_data):
                psrl_logger.error(
                    f"{self.harness_config.kind} produced no usable TITO trajectory "
                    f"(exit_code={harness_result.exit_code!r}, stderr_tail={harness_result.stderr_tail[-2000:]!r})."
                )
                return None, TerminateReason.ROLLOUT_ERROR

            task_reward_info = await self.finalize_harness_task(
                task,
                artifact,
                clean_snapshot,
                timing,
            )
            timing["elapsed_s"] = time.perf_counter() - run_start
            reward_info = self._build_reward_info(
                task_reward_info,
                training_data,
                timing,
                harness_result,
            )
            outputs = [self._build_capped_output(item) for item in training_data]
            for output in outputs:
                output.agent_reward_info = dict(reward_info)
            output_value: TokenOutput | list[TokenOutput] = outputs[0] if len(outputs) == 1 else outputs
            scored_output = await self.compute_reward_score(output_value, **request)
            if scored_output is None:
                return None, TerminateReason.ABORTED
            return scored_output, self.get_harness_terminate_reason(training_data)
        finally:
            await self._cleanup_harness_run(task, session_id, lease, harness, clean_snapshot)

    async def _try_snapshot_clean_sandbox(
        self,
        task: HarnessTaskContext,
        lease: SandboxLease,
    ) -> SnapshotRef | None:
        if not (
            clean_snapshot_compatible(task)
            and lease.session.capabilities.supports(SandboxFeature.FULL_STATE_SNAPSHOT)
            and lease.session.capabilities.supports(SandboxFeature.RESTORE)
        ):
            return None
        try:
            return await self.sandbox_manager.checkpoint(
                lease.session,
                SnapshotKind.FULL_STATE,
                task.sandbox_spec.state_policy,
            )
        except Exception:
            psrl_logger.warning(
                "Could not create a safe clean-sandbox snapshot; task finalization will provision a fresh sandbox.",
                exc_info=True,
            )
            return None

    async def _cleanup_harness_run(
        self,
        task: HarnessTaskContext | None,
        session_id: str | None,
        lease: SandboxLease | None,
        harness: Harness | None,
        clean_snapshot: SnapshotRef | None,
    ) -> None:
        cleanup_task = asyncio.create_task(
            self._cleanup_harness_resources(task, session_id, lease, harness, clean_snapshot),
        )
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            await cleanup_task
            raise

    async def _cleanup_harness_resources(
        self,
        task: HarnessTaskContext | None,
        session_id: str | None,
        lease: SandboxLease | None,
        harness: Harness | None,
        clean_snapshot: SnapshotRef | None,
    ) -> None:
        if harness is not None:
            await self._run_cleanup_operation("harness abort", harness.abort())
        if lease is not None:
            await self._run_cleanup_operation("sandbox release", lease.release())
        if session_id is not None:
            await self._run_cleanup_operation("session deletion", self.delete_session(session_id))
        if clean_snapshot is not None:
            await self._run_cleanup_operation(
                "clean snapshot deletion",
                self.sandbox_manager.delete_snapshot(clean_snapshot),
            )
        if task is not None:
            await self._run_cleanup_operation("task cleanup", self.close_harness_task(task))

    @staticmethod
    async def _run_cleanup_operation(name: str, operation: Awaitable[None]) -> None:
        """
        Isolate one cleanup failure so subsequent resources are still released.
        """
        try:
            await operation
        except Exception as exc:
            psrl_logger.warning(f"Harness cleanup operation {name!r} failed: {exc!r}.")

    @staticmethod
    async def _capture_resource_metrics(
        task: HarnessTaskContext,
        lease: SandboxLease,
        timing: dict[str, float],
    ) -> None:
        if not task.collect_resource_metrics:
            return
        try:
            usage = await lease.session.stats()
            timing["sandbox_memory_mib"] = usage.memory_bytes / (1024 * 1024)
            timing["sandbox_peak_memory_mib"] = usage.peak_memory_bytes / (1024 * 1024)
            timing["sandbox_cpu_total_s"] = usage.cpu_total_ns / 1_000_000_000
        except Exception:
            psrl_logger.warning("Could not collect harness sandbox resource metrics.", exc_info=True)

    def _build_reward_info(
        self,
        task_reward_info: dict,
        training_data: list[dict],
        timing: dict[str, float],
        harness_result: HarnessResult,
    ) -> dict:
        num_turns = max(item["num_turns"] for item in training_data)
        return {
            **task_reward_info,
            "num_turns": num_turns,
            "actual_num_turns": num_turns,
            "timing": timing,
            "harness": self.harness_config.kind,
            "harness_exit_code": harness_result.exit_code,
            "harness_stderr_tail": harness_result.stderr_tail if harness_result.exit_code != 0 else "",
        }

    def _build_capped_output(self, training_data: dict) -> TokenOutput:
        output = self.build_token_output(training_data)
        response_length = int(self.rollout_config.response_length)
        output.response_ids = output.response_ids[:response_length]
        output.response_mask = output.response_mask[:response_length]
        if output.response_log_probs is not None:
            output.response_log_probs = output.response_log_probs[:response_length]
        if output.routed_experts is not None:
            output.routed_experts = output.routed_experts[: len(output.prompt_ids) + response_length]
        return output

    def get_harness_terminate_reason(self, training_data: list[dict]) -> TerminateReason:
        """
        Map captured TITO data to the framework's successful stop reasons.
        """
        if any(len(item["response_ids"]) >= int(self.rollout_config.response_length) for item in training_data):
            return TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED
        if max(item["num_turns"] for item in training_data) >= self.max_turns:
            return TerminateReason.MAX_TURNS_EXCEEDED
        return TerminateReason.FINISHED
