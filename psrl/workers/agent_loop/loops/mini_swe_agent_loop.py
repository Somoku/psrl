"""
mini-SWE-agent Loop -- In-process library mode.

Uses mini-swe-agent v2 as a Python library instead of a subprocess. A custom
`_PSRLModel` inherits `LitellmTextbasedModel` and overrides `query()` to
bridge between mini-swe-agent's synchronous agent loop (running in a worker
thread) and PSRL's async rollout engine (running in the event loop).

Communication uses a pair of `queue.Queue` objects:
- `_PSRLModel.query()` puts messages into `request_queue`, blocks on `response_queue`.
- The async generation loop consumes `request_queue`, calls vLLM, records the turn,
  and puts the response text into `response_queue`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import time

# Suppress mini-swe-agent startup banner before importing minisweagent.
os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")

from examples.mini_swe.config import (  # noqa: E402
    MiniSWEAgentRuntimeConfig,
    build_runtime_config,
)
from litellm import ModelResponse  # noqa: E402
from minisweagent.agents.default import DefaultAgent  # noqa: E402
from minisweagent.environments.docker import DockerEnvironment  # noqa: E402
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from verl import DataProto  # noqa: E402

from psrl.environments.mini_swe_env import MiniSWEEnvironment  # noqa: E402
from psrl.utils.common.docker_utils import cleanup_containers_by_label  # noqa: E402
from psrl.utils.concurrency import SlotManager  # noqa: E402
from psrl.utils.profiling.collector import TurnProfilingCollector  # noqa: E402
from psrl.workers.agent_loop.agent_data.mini_swe_agent_data import (  # noqa: E402
    MiniSWEAgentData,
    normalize_openai_messages,
)
from psrl.workers.agent_loop.gateway_client import RolloutGatewayClient  # noqa: E402
from psrl.workers.agent_loop.loops.base_agent_loop import AgentLoopBase  # noqa: E402
from psrl.workers.agent_loop.loops.utils import TerminateReason, register  # noqa: E402
from psrl.workers.agent_loop.sticky_session import StickySession  # noqa: E402

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

_MIN_GEN_TOKENS = 256
_QUEUE_POLL_TIMEOUT = 2.0

# Dedicated thread pool for running mini-swe-agent sync code.
# Separating from the default pool prevents deadlocks when
# `asyncio.to_thread` (used in the generation loop) competes with
# `run_in_executor` (used for the agent thread).
import concurrent.futures

_AGENT_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=32, thread_name_prefix="mini-swe-agent",
)


# ---------------------------------------------------------------------------
# In-process model bridging mini-swe-agent -> PSRL rollout
# ---------------------------------------------------------------------------


class _PSRLModel(LitellmTextbasedModel):
    """
    Bridge between mini-swe-agent's synchronous `Model` protocol and PSRL's
    async rollout engine.

    Inherits `LitellmTextbasedModel` to reuse:
    - `format_message()` (system/user message formatting).
    - `format_observation_messages()` (observation template rendering).
    - `_parse_actions()` (regex extraction of ``mswea_bash_command`` blocks).
    - `get_template_vars()` / `serialize()` (template vars and trajectory serialization).
    - `config` (`LitellmTextbasedModelConfig` with `observation_template`,
      `format_error_template`, `action_regex`).

    Only `query()` is overridden to route generation through PSRL instead of HTTP.
    """

    def __init__(
        self,
        request_queue: queue.Queue,
        response_queue: queue.Queue,
        **model_kwargs,
    ):
        """
        Initialize the PSRL model bridge.

        Args:
            request_queue (queue.Queue): Messages from mini-swe-agent -> PSRL async side.
            response_queue (queue.Queue): Generated text from PSRL async side -> mini-swe-agent.
            **model_kwargs: Forwarded to `LitellmTextbasedModel` (must include `model_name`).
        """
        super().__init__(**model_kwargs)
        self._req_q = request_queue
        self._res_q = response_queue

    def query(self, messages: list[dict], **kwargs) -> dict:
        """
        Override `LitellmModel.query()` to route through PSRL rollout via queues.

        Called synchronously by `DefaultAgent` in a worker thread.
        Blocks until the async generation loop puts a response.
        """
        psrl_logger.info(f"_PSRLModel.query() called with {len(messages)} messages, waiting for generation...")
        self._req_q.put(messages)
        text = self._res_q.get(timeout=600)
        psrl_logger.info(f"_PSRLModel.query() received response: {len(text)} chars.")

        response = ModelResponse(
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            model="psrl-rollout",
        )
        actions = self._parse_actions(response)
        message = response.choices[0].message.model_dump()
        message["extra"] = {
            "actions": actions,
            "cost": 0.0,
            "timestamp": time.time(),
        }
        return message


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


@register("mini_swe_agent")
class MiniSWEAgentLoop(AgentLoopBase):
    """
    mini-SWE-agent loop -- in-process library mode.

    Orchestrates the full episode lifecycle:
    1. Environment setup (parse `DataProto`, build workspace).
    2. Create `_PSRLModel` + `DockerEnvironment` + `DefaultAgent` (mini-swe-agent).
    3. Run the agent in a worker thread; consume model requests in the async loop.
    4. Record turns, extract patch, compute reward.
    5. Cleanup (Docker containers, temp dirs).
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

        effective_kwargs = kwargs
        if "sandbox_config" not in kwargs:
            effective_kwargs = cls._reload_yaml_config(config, kwargs)
        cls.runtime_config: MiniSWEAgentRuntimeConfig = build_runtime_config(
            yaml_kwargs=effective_kwargs,
        )

        multi_turn = config.gen_actor_rollout_ref.rollout.multi_turn
        if not getattr(multi_turn, "enable", False):
            raise ValueError(
                "mini-SWE-agent requires rollout.multi_turn.enable=True. "
                "Set gen_actor_rollout_ref.rollout.multi_turn.enable=True in your config."
            )
        if getattr(multi_turn, "max_turns", None) is None:
            raise ValueError(
                "mini-SWE-agent requires rollout.multi_turn.max_turns to be set. "
                "Set gen_actor_rollout_ref.rollout.multi_turn.max_turns=<int> in your config."
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

    # --- Main loop ---

    async def run(
        self,
        request: DataProto,
        profiling_collector: TurnProfilingCollector | None = None,
    ) -> tuple[DataProto | None, TerminateReason]:
        """
        Run one mini-SWE-agent episode and return the trajectory.

        Args:
            request (DataProto): Single input `DataProto` request.
            profiling_collector: Per-trajectory profiling collector, or None if disabled.

        Returns:
            Tuple of (finalized DataProto, termination reason).
        """
        run_start_time = time.time()
        run_slot: tuple[int, int] | None = None
        swe_problem_id = ""

        # Initialize environment and agent data.
        env = MiniSWEEnvironment(
            self.config, self.reward_manager,
            runtime_config=self.__class__.runtime_config,
        )
        observation, info = await env.reset(task=request)

        swe_problem_id = observation["swe_problem_id"]
        log_prefix = f"[mini-SWE-agent, id={swe_problem_id}]" if swe_problem_id else "[mini-SWE-agent]"
        psrl_logger.info(f"{log_prefix} Initializing agent data...")

        agent_data = MiniSWEAgentData(
            self.config, self.reward_manager, self.tokenizer, env,
        )
        agent_data.reset()
        agent_data.init_trajectory(request)

        runtime_config = observation["runtime_config"]
        sb = runtime_config.sandbox_config

        psrl_logger.info(
            f"{log_prefix} Agent data initialized. "
            f"max_parallel={sb.max_parallel_tasks_per_worker}, "
            f"traj_output={self.traj_writer.output_dir!r} (enable={self.traj_writer.enable})."
        )

        try:
            # Acquire run slot if configured.
            if sb.max_parallel_tasks_per_worker > 0:
                psrl_logger.info(
                    f"{log_prefix} Waiting for run slot "
                    f"(max_parallel={sb.max_parallel_tasks_per_worker})..."
                )
                run_slot = await SlotManager.acquire(
                    sb.max_parallel_tasks_per_worker, self.traj_writer.output_dir,
                    prefix="psrl_mini_swe_agent_slots",
                )
                psrl_logger.info(
                    f"{log_prefix} Acquired run slot "
                    f"(slot={run_slot[1] if run_slot else 'n/a'})."
                )
            else:
                psrl_logger.info(f"{log_prefix} No slot limit, proceeding directly.")

            # Read max_turns from rollout.multi_turn (validated in init_class).
            max_turns = int(self.config.gen_actor_rollout_ref.rollout.multi_turn.max_turns)

            # Build per-instance environment kwargs.
            env_cfg = sb.environment
            preexisting_repo_name = observation.get("preexisting_repo_name", "")
            use_preexisting = observation.get("use_preexisting_repo", True)
            effective_cwd = env_cfg.cwd
            if use_preexisting and preexisting_repo_name:
                effective_cwd = f"/{preexisting_repo_name}"

            run_args = list(env_cfg.run_args)
            run_args.extend(["--label", f"psrl.swe_problem_id={swe_problem_id}"])
            if not use_preexisting and observation.get("repo_path"):
                run_args.extend(["--volume", f"{observation['repo_path']}:/testbed"])

            docker_env_kwargs = {
                "image": env_cfg.image,
                "cwd": effective_cwd,
                "env": env_cfg.env if isinstance(env_cfg.env, dict) else {},
                "run_args": run_args,
                "container_timeout": env_cfg.container_timeout,
            }

            # Build agent config kwargs.
            ag_cfg = runtime_config.agent
            agent_kwargs = {
                "system_template": ag_cfg.system_template,
                "instance_template": ag_cfg.problem_template,
                "step_limit": max_turns,
                "cost_limit": ag_cfg.cost_limit,
                "output_path": None,
            }

            # Create queues for sync/async bridging.
            req_q: queue.Queue = queue.Queue()
            res_q: queue.Queue = queue.Queue()

            model = _PSRLModel(
                req_q, res_q,
                model_name="psrl/rollout",
                cost_tracking="ignore_errors",
            )

            problem_statement = observation.get("problem_statement", "")

            _problem_short = f"{problem_statement[:60]}..." if len(problem_statement) > 60 else problem_statement
            psrl_logger.info(
                f"{log_prefix} Starting episode: problem={_problem_short!r}, "
                f"cwd={effective_cwd!r}, image={env_cfg.image!r}, "
                f"max_turns={max_turns}."
            )

            # Run the agent in a dedicated worker thread (not the default pool,
            # which would deadlock with asyncio.to_thread used in _generation_loop).
            loop = asyncio.get_running_loop()
            psrl_logger.info(f"{log_prefix} Launching agent in worker thread...")
            agent_future = loop.run_in_executor(
                _AGENT_THREAD_POOL,
                self._run_agent_sync,
                model, docker_env_kwargs, agent_kwargs, problem_statement, swe_problem_id,
            )

            # Async generation loop: consume requests and generate via PSRL.
            # trajectory_output_path is resolved lazily inside the loop from the first output's version_tag.
            psrl_logger.info(f"{log_prefix} Entering generation loop...")
            num_turns, patch, trajectory_output_path = await self._generation_loop(
                agent_future=agent_future,
                agent_data=agent_data,
                request=request,
                req_q=req_q,
                res_q=res_q,
                max_turns=max_turns,
                swe_problem_id=swe_problem_id,
                profiling_collector=profiling_collector,
            )

            # Wait for agent thread to finish.
            try:
                result = await asyncio.wait_for(agent_future, timeout=60.0)
                thread_patch = result.get("submission", "") if isinstance(result, dict) else ""
                if thread_patch and not patch:
                    patch = thread_patch
            except (asyncio.TimeoutError, Exception) as e:
                psrl_logger.warning(f"{log_prefix} Agent thread did not finish cleanly: {e}.")

            agent_data.set_patch(patch)

            total_elapsed = time.time() - run_start_time
            psrl_logger.info(
                f"{log_prefix} Episode completed: {num_turns} turns, "
                f"patch={'yes' if patch else 'no'}, total={total_elapsed:.1f}s."
            )

            summary_text = ""
            if patch:
                summary_text += f"=== Submission ===\n{patch}\n\n"
            summary_text += (
                f"=== Summary ===\n"
                f"turns: {num_turns}, patch: {'yes' if patch else 'no'}, "
                f"elapsed: {total_elapsed:.1f}s\n"
            )
            self.traj_writer.append(trajectory_output_path, summary_text)

            finalized = await agent_data.finalize_output(request)

            if num_turns >= max_turns:
                terminate_reason = TerminateReason.MAX_TURNS_EXCEEDED
            elif num_turns == 0:
                terminate_reason = TerminateReason.UNKNOWN
            else:
                terminate_reason = TerminateReason.FINISHED

            return finalized, terminate_reason

        finally:
            # Safety-net Docker cleanup via labels.
            if swe_problem_id:
                await cleanup_containers_by_label("psrl.swe_problem_id", swe_problem_id)
            await env.close()
            if run_slot is not None:
                SlotManager.release(run_slot)

    # --- Worker thread entry point ---

    @staticmethod
    def _run_agent_sync(
        model: _PSRLModel,
        docker_env_kwargs: dict,
        agent_kwargs: dict,
        task: str,
        swe_problem_id: str = "",
    ) -> dict:
        """
        Run mini-swe-agent synchronously in a worker thread.

        Creates `DockerEnvironment` and `DefaultAgent` in-thread (Docker
        container starts immediately on construction), runs the agent, then
        cleans up the container.
        """
        log_prefix = f"[mini-SWE-agent, id={swe_problem_id}]" if swe_problem_id else "[mini-SWE-agent]"
        docker_env = None
        try:
            psrl_logger.info(
                f"{log_prefix} Creating DockerEnvironment: image={docker_env_kwargs.get('image')!r}, "
                f"cwd={docker_env_kwargs.get('cwd')!r}."
            )
            docker_env = DockerEnvironment(**docker_env_kwargs)
            psrl_logger.info(
                f"{log_prefix} DockerEnvironment created, container_id={docker_env.container_id!r}."
            )

            # Validate that the cwd exists inside the container.
            # Run from "/" to avoid docker exec failing before our check command runs.
            cwd = docker_env_kwargs.get("cwd", "")
            if cwd:
                check = docker_env.execute(
                    {"command": f"test -d {cwd} && echo EXISTS || echo MISSING"},
                    cwd="/",
                )
                if "MISSING" in check.get("output", ""):
                    raise RuntimeError(
                        f"{log_prefix} cwd {cwd!r} does not exist inside Docker container "
                        f"{docker_env.container_id!r}. Check that the image was prepared "
                        "with 'bake_simple_repos.sh' and the correct preexisting_repo_name."
                    )
                psrl_logger.info(f"{log_prefix} cwd={cwd!r} confirmed to exist inside container.")

            psrl_logger.info(f"{log_prefix} Creating DefaultAgent with step_limit={agent_kwargs.get('step_limit')}.")
            agent = DefaultAgent(model, docker_env, **agent_kwargs)

            _task_short = f"{task[:60]}..." if len(task) > 60 else task
            psrl_logger.info(f"{log_prefix} Running agent with task={_task_short!r}.")
            result = agent.run(task)

            submission = result.get("submission", "") if isinstance(result, dict) else ""
            exit_status = result.get("exit_status", "") if isinstance(result, dict) else ""
            psrl_logger.info(
                f"{log_prefix} Agent finished: exit_status={exit_status!r}, "
                f"submission_len={len(submission)}, n_calls={agent.n_calls}."
            )
            return result
        except Exception as e:
            psrl_logger.exception(f"{log_prefix} Agent thread failed: {e}.")
            return {"exit_status": "error", "submission": ""}
        finally:
            if docker_env is not None:
                psrl_logger.info(f"{log_prefix} Cleaning up DockerEnvironment...")
                docker_env.cleanup()

    # --- Async generation loop ---

    async def _generation_loop(
        self,
        agent_future: asyncio.Future,
        agent_data: MiniSWEAgentData,
        request: DataProto,
        req_q: queue.Queue,
        res_q: queue.Queue,
        max_turns: int,
        swe_problem_id: str,
        profiling_collector: TurnProfilingCollector | None = None,
    ) -> tuple[int, str | None, str]:
        """
        Consume model requests from `req_q`, generate via PSRL rollout,
        record turns, and put responses into `res_q`.

        The full trajectory is accumulated in a string buffer during the loop.
        After the loop ends the real `version_tag` (assigned by the router on
        the first generation call) is used to determine the versioned output
        directory, and the buffer is written atomically to disk via
        `self.traj_writer`.

        Returns:
            Tuple of (num_turns, patch_or_none, trajectory_path).
        """
        num_turns = 0
        trajectory_path = ""
        resolved_version: int | None = None

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

        log_prefix = f"[mini-SWE-agent, id={swe_problem_id}]" if swe_problem_id else "[mini-SWE-agent]"

        psrl_logger.info(
            f"{log_prefix} Generation loop started: "
            f"effective_limit={effective_limit}, max_turns={max_turns}."
        )

        # Accumulate trajectory text throughout the loop; written to disk at the end.
        traj_text: list[str] = [f"{log_prefix}\n\n"]

        while not agent_future.done():
            try:
                messages = req_q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.1)
                continue

            psrl_logger.info(
                f"{log_prefix} Generation loop received request "
                f"(turn {num_turns + 1}): {len(messages)} messages."
            )

            # Capture the observation before generating (messages[-1] is the latest
            # user/tool message: problem statement on turn 1, Docker output thereafter).
            observation = messages[-1].get("content", "") if messages else ""

            normalized = normalize_openai_messages(messages)
            prompt_ids = agent_data.encode_messages(normalized, add_generation_prompt=True)

            # Token budget check.
            remaining = max(
                (effective_limit - len(prompt_ids)) if effective_limit else cfg_response_len,
                0,
            )
            if effective_limit and remaining < _MIN_GEN_TOKENS:
                psrl_logger.warning(
                    f"{log_prefix} Turn {num_turns + 1}: remaining budget "
                    f"{remaining} < {_MIN_GEN_TOKENS}, sending empty response."
                )
                res_q.put("")
                break

            # Build generation request (routing metadata managed by agent_data).
            gen_request = agent_data.prepare_generation_request(request, prompt_ids=prompt_ids)
            request_id = request.non_tensor_batch["uid"][0]
            async with StickySession(self.rollout_router, request_id):
                if profiling_collector is not None:
                    profiling_collector.on_turn_submit()
                if self.config.psrl.server_rollout.enable:
                    gateway_client = RolloutGatewayClient.from_config(self.config)
                    output = await gateway_client.generate_async(gen_request)
                else:
                    output = await self.rollout_router.generate_async.remote(gen_request)

            if output is None:
                psrl_logger.warning(
                    f"{log_prefix} Turn {num_turns + 1}: rollout returned None."
                )
                res_q.put("")
                break

            if profiling_collector is not None:
                profiling_collector.on_turn_complete(output)

            response_ids = list(output.non_tensor_batch["raw_response_ids"][0])
            response_logprobs_raw = output.non_tensor_batch.get("rollout_log_probs", [None])[0]
            response_logprobs = (
                list(response_logprobs_raw) if response_logprobs_raw is not None
                else [0.0] * len(response_ids)
            )
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            # Capture version_tag from the first real output (initial request carries -1).
            if resolved_version is None:
                resolved_version = int(output.non_tensor_batch["version_tag"][0])

            # Record turn and update routing state (instance_id / version_tag).
            agent_data.update_from_external_turn(
                turn_index=num_turns,
                messages=normalized,
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_text=response_text,
                response_logprobs=response_logprobs,
                output=output,
            )
            num_turns += 1

            # Send response text to the worker thread.
            res_q.put(response_text)

            # Append turn (observation captured before generate + response) to buffer.
            traj_text.append(
                f"=== Turn {num_turns} ===\n"
                f"--- observation ---\n{observation}\n\n"
                f"--- assistant ---\n{response_text}\n\n"
            )

            psrl_logger.info(
                f"{log_prefix} Turn {num_turns}: {len(response_ids)} model tokens."
            )

            if num_turns >= max_turns:
                psrl_logger.warning(
                    f"{log_prefix} Max turns reached "
                    f"({num_turns}/{max_turns})."
                )
                break

        # Write the accumulated buffer via the shared TrajectoryWriter.
        # Use uid as the file stem for uniqueness across rollout_n > 1 runs of the same problem.
        trajectory_path = ""
        if resolved_version is not None and num_turns > 0:
            traj_id = str(request.non_tensor_batch["uid"][0])
            trajectory_path = self.traj_writer.write(resolved_version, traj_id, "".join(traj_text))

        return num_turns, None, trajectory_path
