"""
mini-SWE-Agent Loop -- External control multi-turn interaction mode.

Intercepts mini-SWE-agent model calls through `ModelProxy`, allowing PSRL to
control generation and collect training trajectories.

Delegated responsibilities:
- Config merge / dataclass: `examples.mini_swe.config`.
- mini-SWE-agent CLI YAML generation: `examples.mini_swe.config.build_mini_sweagent_yaml`.
- Subprocess lifecycle + Docker cleanup: `examples.mini_swe.subprocess_runner`.
- Data conversion: `MiniSWEAgentData`.
- Environment lifecycle: `MiniSWEEnvironment`.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import logging
import os
import shutil
import tempfile
import time
import uuid

import numpy as np
from examples.mini_swe.config import (
    MiniSWEAgentRuntimeConfig,
    build_mini_sweagent_yaml,
    build_runtime_config,
)
from examples.mini_swe.model_proxy import ModelProxy
from examples.mini_swe.subprocess_runner import cleanup_instance_containers, execute_mini_swe_agent
from omegaconf import DictConfig, OmegaConf
from verl import DataProto

from psrl.environments.mini_swe_env import MiniSWEEnvironment
from psrl.workers.agent_loop.agent_data.mini_swe_agent_data import (
    MiniSWEAgentData,
    normalize_openai_messages,
)
from psrl.workers.agent_loop.gateway_client import RolloutGatewayClient
from psrl.workers.agent_loop.loops.base_agent_loop import AgentLoopBase
from psrl.workers.agent_loop.loops.utils import TerminateReason, register
from psrl.workers.agent_loop.sticky_session import StickySession

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

_MIN_GEN_TOKENS = 256


@register("mini_swe_agent")
class MiniSWEAgentLoop(AgentLoopBase):
    """
    mini-SWE-Agent loop — external control multi-turn interaction mode.

    Orchestrates the full episode lifecycle:
    1. Environment setup (workspace, Docker config)
    2. ModelProxy start (OpenAI-compatible HTTP server)
    3. mini-SWE-agent subprocess launch
    4. Interaction loop (intercept model calls, generate via PSRL rollout)
    5. Patch extraction, trajectory reconstruction, reward computation
    6. Cleanup (Docker containers, temp files, ModelProxy)
    """

    @classmethod
    def init_class(cls, config: DictConfig, **kwargs) -> None:
        """
        Perform heavy initialization work shared across all instances.
        """
        if cls._class_initialized:
            return
        cls._class_initialized = True

        cls.prompt_length = config.gen_actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.gen_actor_rollout_ref.rollout.response_length

        # Build runtime config from kwargs (YAML-loaded fields).
        effective_kwargs = kwargs
        if "sandbox_config" not in kwargs:
            effective_kwargs = cls._reload_yaml_config(config, kwargs)
        cls.runtime_config: MiniSWEAgentRuntimeConfig = build_runtime_config(
            yaml_kwargs=effective_kwargs,
        )

    @staticmethod
    def _reload_yaml_config(config: DictConfig, kwargs: dict) -> dict:
        """
        Reload agent loop config from YAML when registry entry was stripped.
        """
        try:
            rollout = config.gen_actor_rollout_ref.rollout
            yaml_path = rollout.agent.agent_loop_config_path
            if yaml_path:
                configs = OmegaConf.load(yaml_path)
                for c in configs:
                    if getattr(c, "name", None) == "mini_swe_agent":
                        merged = OmegaConf.to_container(c, resolve=True)
                        merged.update(kwargs)
                        psrl_logger.info(
                            f"Reloaded YAML config for mini_swe_agent from {yaml_path!r}."
                        )
                        return merged
        except Exception as e:
            psrl_logger.warning(f"Failed to reload YAML config: {e}.")
        return kwargs

    # --- Run slot management (cross-process concurrency control) ---

    @classmethod
    def _slot_lock_dir(cls, output_dir: str) -> str:
        """
        Return lock directory for cross-process run-slot coordination.
        """
        digest = hashlib.sha1(os.path.abspath(output_dir).encode("utf-8")).hexdigest()[:12]
        return os.path.join(tempfile.gettempdir(), f"psrl_mini_swe_agent_slots_{digest}")

    @classmethod
    async def _acquire_run_slot(
        cls,
        max_parallel_tasks_per_worker: int,
        output_dir: str,
    ) -> tuple[int, int] | None:
        """
        Acquire one cross-process run slot via fcntl file lock.
        """
        if max_parallel_tasks_per_worker <= 0:
            return None

        lock_dir = cls._slot_lock_dir(output_dir)
        os.makedirs(lock_dir, exist_ok=True)

        while True:
            for slot_idx in range(max_parallel_tasks_per_worker):
                lock_path = os.path.join(lock_dir, f"slot_{slot_idx}.lock")
                fd = os.open(
                    lock_path,
                    os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0),
                    0o666,
                )
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    os.ftruncate(fd, 0)
                    os.write(fd, f"pid={os.getpid()}\n".encode())
                    return fd, slot_idx
                except BlockingIOError:
                    os.close(fd)

            await asyncio.sleep(0.2)

    @staticmethod
    def _release_run_slot(run_slot: tuple[int, int] | None) -> None:
        """
        Release a previously acquired run slot.
        """
        if run_slot is None:
            return
        fd, _ = run_slot
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    # --- Main loop ---

    async def run(self, request: DataProto) -> tuple[DataProto | None, TerminateReason]:
        """
        Run one mini-SWE-Agent episode and return the trajectory.

        Args:
            request: Single input DataProto request.

        Returns:
            Tuple of (finalized DataProto, termination reason).
        """
        run_start_time = time.time()
        agent_task: asyncio.Task | None = None
        model_proxy: ModelProxy | None = None
        run_slot: tuple[int, int] | None = None

        # Initialize environment and agent data.
        env = MiniSWEEnvironment(
            self.config, self.reward_manager,
            runtime_config=self.__class__.runtime_config,
        )
        observation, info = await env.reset(task=request)

        agent_data = MiniSWEAgentData(
            self.config, self.reward_manager, self.tokenizer, env,
        )
        agent_data.reset()
        agent_data.init_trajectory(request)

        runtime_config = observation["runtime_config"]
        sb = runtime_config.sandbox_config
        pc = runtime_config.proxy_config
        swe_problem_id = observation["swe_problem_id"]

        try:
            # Acquire run slot if configured.
            if sb.max_parallel_tasks_per_worker > 0:
                run_slot = await self._acquire_run_slot(
                    sb.max_parallel_tasks_per_worker, sb.output_dir,
                )
                psrl_logger.info(
                    f"[{swe_problem_id}] Acquired run slot "
                    f"(slot={run_slot[1] if run_slot else 'n/a'})."
                )

            # Start ModelProxy.
            model_proxy = ModelProxy(port=pc.port)
            await model_proxy.start_server(max_retries=pc.max_port_retries)
            psrl_logger.info(
                f"[{swe_problem_id}] ModelProxy started on port {model_proxy.port}."
            )

            # Launch mini-SWE-Agent subprocess.
            agent_task = asyncio.create_task(
                self._launch_agent(
                    observation=observation,
                    runtime_config=runtime_config,
                    model_proxy_port=model_proxy.port,
                )
            )

            # Interaction loop.
            session_id = swe_problem_id if swe_problem_id else str(uuid.uuid4())
            patch, num_turns = await self._interaction_loop(
                agent_task=agent_task,
                agent_data=agent_data,
                request=request,
                max_model_calls_per_instance=sb.max_model_calls_per_instance,
                request_timeout=pc.timeout,
                model_proxy=model_proxy,
                session_id=session_id,
            )

            # Drain agent task.
            if agent_task is not None and not agent_task.done():
                drain_patch = await self._drain_agent_task(
                    agent_task, num_turns >= sb.max_model_calls_per_instance,
                )
                if drain_patch and not patch:
                    patch = drain_patch

            # Extract patch from environment if not already obtained.
            if not patch:
                patch = await env.extract_patch()

            agent_data.set_patch(patch)

            total_elapsed = time.time() - run_start_time
            psrl_logger.info(
                f"[{swe_problem_id}] Episode completed: {num_turns} turns, "
                f"patch={'yes' if patch else 'no'}, total={total_elapsed:.1f}s."
            )

            # Finalize trajectory.
            finalized = await agent_data.finalize_output(request)

            # Determine termination reason.
            if num_turns >= sb.max_model_calls_per_instance:
                terminate_reason = TerminateReason.MAX_TURNS_EXCEEDED
            elif num_turns == 0:
                terminate_reason = TerminateReason.UNKNOWN
            else:
                terminate_reason = TerminateReason.FINISHED

            return finalized, terminate_reason

        finally:
            if agent_task is not None and not agent_task.done():
                agent_task.cancel()
                try:
                    await asyncio.wait_for(agent_task, timeout=10.0)
                except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                    pass
            if model_proxy is not None:
                await model_proxy.stop_server()
            await env.close()
            if run_slot is not None:
                self._release_run_slot(run_slot)

    # --- Interaction loop ---

    async def _interaction_loop(
        self,
        agent_task: asyncio.Task,
        agent_data: MiniSWEAgentData,
        request: DataProto,
        max_model_calls_per_instance: int,
        request_timeout: float,
        model_proxy: ModelProxy,
        session_id: str,
    ) -> tuple[str | None, int]:
        """
        Run the main turn-by-turn interaction with mini-SWE-Agent via ModelProxy.

        Returns:
            Tuple of (patch_or_none, num_turns).
        """
        num_turns = 0
        patch: str | None = None

        # Compute token budget.
        rollout_cfg = self.config.gen_actor_rollout_ref.rollout
        cfg_prompt_len = int(getattr(rollout_cfg, "prompt_length", 0) or 0)
        cfg_response_len = int(getattr(rollout_cfg, "response_length", 4096) or 4096)
        max_model_len = int(getattr(rollout_cfg, "max_model_len", 0) or 0)
        vllm_budget = cfg_prompt_len + cfg_response_len if cfg_prompt_len else max_model_len
        effective_limit = (
            min(max_model_len, vllm_budget)
            if max_model_len and vllm_budget
            else (max_model_len or vllm_budget)
        )

        while True:
            # Pre-check: agent already done?
            if agent_task.done():
                try:
                    patch = await agent_task
                except Exception as e:
                    psrl_logger.exception(f"mini-SWE-Agent task failed: {e}.")
                break

            # Race: model request vs. agent completion.
            request_coro = asyncio.create_task(model_proxy.get_request())
            done, pending = await asyncio.wait(
                {request_coro, agent_task},
                timeout=request_timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if not done:
                psrl_logger.error(
                    f"Both request and agent tasks timed out after {request_timeout}s."
                )
                request_coro.cancel()
                try:
                    await request_coro
                except (asyncio.CancelledError, Exception):
                    pass
                break

            if agent_task in done:
                if request_coro in pending:
                    request_coro.cancel()
                    try:
                        await request_coro
                    except (asyncio.CancelledError, Exception):
                        pass
                try:
                    patch = await agent_task
                except Exception as e:
                    psrl_logger.exception(f"mini-SWE-Agent task failed: {e}.")
                break

            # Process the model request.
            try:
                model_request = request_coro.result()
            except Exception as e:
                psrl_logger.exception(f"Error getting model request: {e}.")
                continue

            messages = normalize_openai_messages(model_request.messages)
            prompt_ids = agent_data.encode_messages(messages, add_generation_prompt=True)

            # Early-stop if prompt leaves insufficient generation room.
            remaining = max(
                (effective_limit - len(prompt_ids)) if effective_limit else cfg_response_len,
                0,
            )
            if effective_limit and remaining < _MIN_GEN_TOKENS:
                psrl_logger.warning(
                    f"Turn {num_turns + 1}: remaining budget {remaining} < {_MIN_GEN_TOKENS} "
                    f"(prompt_len={len(prompt_ids)}, limit={effective_limit}), stopping."
                )
                await model_proxy.send_response(
                    "", request=model_request, finish_reason="length",
                )
                break

            # Build generation request.
            gen_request = self._build_gen_request(request, prompt_ids)

            # Generate via PSRL rollout (with sticky session for KV cache reuse).
            request_id = request.non_tensor_batch["uid"][0]
            async with StickySession(self.rollout_router, request_id):
                if self.config.psrl.server_rollout.enable:
                    gateway_client = RolloutGatewayClient.from_config(self.config)
                    output = await gateway_client.generate_async(gen_request)
                else:
                    output = await self.rollout_router.generate_async.remote(gen_request)

            if output is None:
                psrl_logger.warning(f"Turn {num_turns + 1}: rollout returned None, stopping.")
                await model_proxy.send_response(
                    "", request=model_request, finish_reason="stop",
                )
                break

            # Extract training signals.
            response_ids = list(output.non_tensor_batch["raw_response_ids"][0])
            response_logprobs_raw = output.non_tensor_batch.get("rollout_log_probs", [None])[0]
            response_logprobs = (
                list(response_logprobs_raw) if response_logprobs_raw is not None
                else [0.0] * len(response_ids)
            )
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            # Record turn.
            agent_data.record_turn(
                turn_index=num_turns,
                messages=messages,
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_text=response_text,
                response_logprobs=response_logprobs,
            )

            num_turns += 1

            # Send response back to mini-SWE-agent.
            await model_proxy.send_response(response_text, request=model_request)

            psrl_logger.info(f"Turn {num_turns}: {len(response_ids)} model tokens.")

            if num_turns >= max_model_calls_per_instance:
                psrl_logger.warning(
                    f"Max model calls reached ({num_turns}/{max_model_calls_per_instance})."
                )
                break

        return patch, num_turns

    # --- Agent launch ---

    async def _launch_agent(
        self,
        observation: dict,
        runtime_config: MiniSWEAgentRuntimeConfig,
        model_proxy_port: int,
    ) -> str | None:
        """
        Generate config, run mini-SWE-Agent subprocess, return patch.
        """
        swe_problem_id = observation["swe_problem_id"]
        output_dir = observation["output_dir"]
        repo_path = observation.get("repo_path") or ""
        use_preexisting_repo = observation.get("use_preexisting_repo", True)
        preexisting_repo_name = observation.get("preexisting_repo_name", "")

        exec_dir = tempfile.mkdtemp(prefix=f"mini_swe_exec_{swe_problem_id}_")
        output_json_path = os.path.join(output_dir, f"{swe_problem_id}.traj.json")

        # Build YAML config for mini-SWE-Agent CLI.
        rollout_cfg = self.config.gen_actor_rollout_ref.rollout
        max_input_tokens = int(getattr(rollout_cfg, "max_model_len", 0) or 0)

        repo_type = "preexisting" if use_preexisting_repo else "local"

        yaml_str = build_mini_sweagent_yaml(
            runtime_config,
            swe_problem_id=swe_problem_id,
            repo_path=repo_path,
            model_proxy_port=model_proxy_port,
            max_input_tokens=max_input_tokens,
            repo_type=repo_type,
            preexisting_repo_name=preexisting_repo_name,
        )
        config_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=f"_mini_swe_config_{swe_problem_id}.yaml",
            delete=False,
            encoding="utf-8",
        )
        config_file.write(yaml_str)
        config_file.close()
        config_path = config_file.name

        problem_statement = observation.get("problem_statement", "")

        try:
            patch = await execute_mini_swe_agent(
                config_path=config_path,
                problem_statement=problem_statement,
                swe_problem_id=swe_problem_id,
                output_dir=output_dir,
                output_json_path=output_json_path,
                repo_path=repo_path,
                exec_dir=exec_dir,
                swe_agent_timeout=runtime_config.sandbox_config.swe_agent_timeout,
                proxy_port=model_proxy_port,
            )
            return patch
        except Exception as e:
            psrl_logger.exception(
                f"[{swe_problem_id}] mini-SWE-Agent execution failed: {e}."
            )
            return None
        finally:
            await cleanup_instance_containers(swe_problem_id)
            try:
                os.unlink(config_path)
            except OSError:
                pass
            shutil.rmtree(exec_dir, ignore_errors=True)

    # --- Helpers ---

    def _build_gen_request(self, request: DataProto, prompt_ids: list[int]) -> DataProto:
        """
        Build a generation request ``DataProto`` from prompt token IDs.
        """
        non_tensor_batch = request.non_tensor_batch.copy()
        non_tensor_batch["raw_prompt_ids"] = np.array([prompt_ids])

        return DataProto(
            non_tensor_batch=non_tensor_batch,
            meta_info=request.meta_info,
        )

    @staticmethod
    async def _drain_agent_task(
        agent_task: asyncio.Task,
        max_model_calls_reached: bool,
    ) -> str | None:
        """
        Wait for / cancel the mini-SWE-Agent background task.
        """
        if max_model_calls_reached:
            try:
                return await asyncio.wait_for(agent_task, timeout=30.0)
            except asyncio.TimeoutError:
                psrl_logger.warning(
                    "Cancelling mini-SWE-Agent task due to max_model_calls limit."
                )
                agent_task.cancel()
                try:
                    return await asyncio.wait_for(agent_task, timeout=15.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    return None
        else:
            try:
                return await asyncio.wait_for(agent_task, timeout=60.0)
            except asyncio.TimeoutError:
                psrl_logger.warning("Timeout waiting for agent task completion.")
                return None
