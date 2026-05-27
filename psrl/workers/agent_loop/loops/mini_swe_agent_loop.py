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
import re
import time
from dataclasses import dataclass

# Suppress mini-swe-agent startup banner before importing minisweagent.
os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")

from examples.mini_swe.config import (  # noqa: E402
    MiniSWEAgentRuntimeConfig,
    build_runtime_config,
)
from litellm import ModelResponse  # noqa: E402
from minisweagent.agents.default import DefaultAgent  # noqa: E402
from minisweagent.environments.docker import DockerEnvironment  # noqa: E402
from minisweagent.exceptions import FormatError, InterruptAgentFlow  # noqa: E402
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from verl import DataProto  # noqa: E402

from psrl.environments.mini_swe_env import MiniSWEEnvironment  # noqa: E402
from psrl.utils.common.docker_utils import force_remove_containers_by_label  # noqa: E402
from psrl.utils.concurrency import SlotManager  # noqa: E402
from psrl.utils.profiling.collector import TurnProfilingCollector  # noqa: E402
from psrl.workers.agent_loop.agent_data.conversation_agent_data import (  # noqa: E402
    normalize_openai_messages,
)
from psrl.workers.agent_loop.agent_data.mini_swe_agent_data import MiniSWEAgentData  # noqa: E402
from psrl.workers.agent_loop.gateway_client import RolloutGatewayClient  # noqa: E402
from psrl.workers.agent_loop.loops.base_agent_loop import AgentLoopBase  # noqa: E402
from psrl.workers.agent_loop.loops.utils import TerminateReason, register  # noqa: E402
from psrl.workers.agent_loop.sticky_session import maybe_sticky_session  # noqa: E402

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

# Suppress verbose debug logs from minisweagent's docker environment.
logging.getLogger("minisweagent.environment").setLevel(logging.WARNING)

_MIN_GEN_TOKENS = 256
_QUEUE_POLL_TIMEOUT = 2.0

# Sentinel value for action_regex that triggers the XML function-calling
# parser (psrl.tools.tool_parser.xml_fc_tool_parser) instead of the default
# single-regex approach.
_XML_FC_SENTINEL = "__XML_FUNCTION_CALLING__"


def _check_token_budget(
    token_count: int,
    effective_limit: int,
    log_prefix: str,
    turn: int,
    label: str = "token count",
) -> bool:
    """
    Return True (and log a warning) when the token count leaves fewer than
    ``_MIN_GEN_TOKENS`` tokens for the model to generate.

    Args:
        token_count: Number of tokens already consumed (prompt + response so far).
        effective_limit: The hard token ceiling (0 means no limit).
        log_prefix: Log prefix string for the current task.
        turn: Current turn number (1-based, for logging).
        label: Short description of what was counted (for the log message).

    Returns:
        True if the budget is exhausted, False otherwise.
    """
    if not effective_limit:
        return False
    remaining = effective_limit - token_count
    if remaining < _MIN_GEN_TOKENS:
        psrl_logger.warning(
            f"{log_prefix} Turn {turn}: {label} {token_count} leaves only "
            f"{remaining} tokens remaining (limit={effective_limit}, "
            f"min_gen={_MIN_GEN_TOKENS}), terminating agent."
        )
        return True
    return False

# Fallback wall-clock budget for one episode when container_timeout cannot be parsed.
_DEFAULT_EPISODE_TIMEOUT_SECS = 7200  # 2 h


def _parse_timeout_secs(container_timeout: str) -> float:
    """
    Parse a human-readable duration string (e.g. ``"2h"``, ``"90m"``, ``"3600s"``,
    ``"3600"``) into seconds.  Falls back to ``_DEFAULT_EPISODE_TIMEOUT_SECS`` on
    any parse error so the loop always has a finite upper bound.
    """
    s = str(container_timeout).strip().lower()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([hms]?)", s)
    if not m:
        return _DEFAULT_EPISODE_TIMEOUT_SECS
    value, unit = float(m.group(1)), m.group(2)
    multipliers = {"h": 3600.0, "m": 60.0, "s": 1.0, "": 1.0}
    return value * multipliers[unit]

# Dedicated thread pool for running mini-swe-agent sync code.
# Separating from the default pool prevents deadlocks when
# `asyncio.to_thread` (used in the generation loop) competes with
# `run_in_executor` (used for the agent thread).
import concurrent.futures

_AGENT_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=512, thread_name_prefix="mini-swe-agent",
)

# Dedicated thread pool for post-rollout fresh-container grading.
# Kept separate from _AGENT_THREAD_POOL to prevent grading jobs from
# starving rollout threads (and vice versa).
_GRADER_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=16, thread_name_prefix="swebench-grader",
)


# ---------------------------------------------------------------------------
# In-process model bridging mini-swe-agent -> PSRL rollout
# ---------------------------------------------------------------------------


@dataclass
class _TerminateSignal:
    """
    Sentinel put into `res_q` by the generation loop when PSRL decides to abort
    the current episode (e.g. prompt budget exhausted, rollout returned None).

    `_PSRLModel.query()` detects this and raises `InterruptAgentFlow` with a
    ``role="exit"`` message so `DefaultAgent.run()` terminates immediately,
    instead of treating an empty string as a model response and looping until
    `step_limit` via `FormatError` retries.

    Recognised ``exit_status`` values (used by ``_classify_mini_swe_termination``):

    - ``"PromptBudgetExhausted"`` -- the next turn would not fit in the model
      context window.
    - ``"EpisodeTimeout"``        -- the per-episode wall-clock budget elapsed.
    - ``"RolloutReturnedNone"``   -- the rollout returned ``None`` (i.e. PSRL
      aborted this trajectory; classified as ``ABORTED``).
    - ``"RolloutError"``          -- the rollout raised; the generation loop
      re-raises the original exception, but signals the agent thread first so
      it can exit cleanly instead of blocking on ``res_q.get`` for 600s.
    """

    exit_status: str
    content: str = ""


def _classify_mini_swe_termination(
    thread_exit_status: str, num_turns: int, max_turns: int
) -> TerminateReason:
    """
    Map the agent thread's exit_status (plus turn count) to a ``TerminateReason``.

    Centralises the previously duplicated ``stop_reason`` / ``terminate_reason``
    logic in ``run()``. The summary "stop:" line now reads
    ``terminate_reason.value`` directly, so naming changes propagate everywhere.

    Args:
        thread_exit_status: ``exit_status`` field from the agent thread's
            result dict, or one of the ``_TerminateSignal`` labels above when
            PSRL initiated the stop.
        num_turns: Number of generation turns completed.
        max_turns: Configured ``rollout.multi_turn.max_turns``.
    """
    if thread_exit_status == "PromptBudgetExhausted":
        return TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED
    if thread_exit_status == "EpisodeTimeout":
        # Wall-clock cutoff inside _generation_loop -- this is a timeout,
        # not a generic error (which is what the old code mistakenly produced).
        return TerminateReason.TRAJECTORY_TIMEOUT
    if thread_exit_status == "RolloutReturnedNone":
        # Router returned None *without* raising, which now exclusively means
        # the request was intentionally aborted (e.g. staleness filter, sibling
        # group failure). Real rollout errors raise and bypass this helper.
        return TerminateReason.ABORTED
    if thread_exit_status == "RolloutError":
        # Defensive: the rollout-exception path re-raises before reaching
        # classification, so this branch should not normally fire. Kept so the
        # mapping is explicit if future code paths leave this label in place.
        return TerminateReason.ROLLOUT_ERROR
    if num_turns >= max_turns:
        return TerminateReason.MAX_TURNS_EXCEEDED
    if num_turns == 0:
        return TerminateReason.UNKNOWN
    return TerminateReason.FINISHED


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
        query_timeout: int = 600,
        **model_kwargs,
    ):
        """
        Initialize the PSRL model bridge.

        Args:
            request_queue (queue.Queue): Messages from mini-swe-agent -> PSRL async side.
            response_queue (queue.Queue): Generated text from PSRL async side -> mini-swe-agent.
            query_timeout (int): Seconds to wait for a response from the generation loop.
                Should be greater than the generation loop's ``rollout_turn_timeout`` so
                the loop always signals the agent before this fires.
            **model_kwargs: Forwarded to `LitellmTextbasedModel` (must include `model_name`).
        """
        super().__init__(**model_kwargs)
        self._req_q = request_queue
        self._res_q = response_queue
        self._query_timeout = query_timeout

    def query(self, messages: list[dict], **kwargs) -> dict:
        """
        Override `LitellmModel.query()` to route through PSRL rollout via queues.

        Called synchronously by `DefaultAgent` in a worker thread.
        Blocks until the async generation loop puts a response.
        """
        psrl_logger.debug(f"_PSRLModel.query() called with {len(messages)} messages, waiting for generation...")
        self._req_q.put(messages)
        item = self._res_q.get(timeout=self._query_timeout)

        # PSRL-initiated abort: exit the agent on the same turn by raising
        # InterruptAgentFlow with a role="exit" message. `DefaultAgent.run()`
        # catches it, extends messages, sees the exit role and breaks.
        if isinstance(item, _TerminateSignal):
            psrl_logger.debug(
                f"_PSRLModel.query() received terminate signal: exit_status={item.exit_status!r}."
            )
            raise InterruptAgentFlow(
                {
                    "role": "exit",
                    "content": item.content or item.exit_status,
                    "extra": {
                        "exit_status": item.exit_status,
                        "submission": "",
                        "cost": 0.0,
                        "timestamp": time.time(),
                    },
                }
            )

        text = item
        psrl_logger.debug(f"_PSRLModel.query() received response: {len(text)} chars.")

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

    # ------------------------------------------------------------------
    # Multi-format action parser for xml_function_calling models
    # ------------------------------------------------------------------

    def _parse_actions(self, response) -> list[dict]:
        """
        Override base ``_parse_actions`` to support multi-format XML function
        calling when ``action_regex`` is the sentinel ``__XML_FUNCTION_CALLING__``.

        Falls back to the base class regex parser for standard configs.
        """
        if self.config.action_regex != _XML_FC_SENTINEL:
            return super()._parse_actions(response)

        content = response.choices[0].message.content or ""
        return self._parse_xml_fc_actions(content)

    def _parse_xml_fc_actions(self, content: str) -> list[dict]:
        """
        Extract a bash command from xml_function_calling model output.

        Delegates to ``psrl.tools.tool_parser.xml_fc_tool_parser.parse_xml_fc_to_bash``
        which handles:
        - ``<function=bash>`` — direct passthrough
        - ``<function=submit>`` — SWE-bench submission command
        - ``<function=str_replace>`` / ``<function=str_replace_editor>`` — translated
          to equivalent bash commands (create, view, str_replace, insert, undo_edit)
        - Degraded patterns (``<bash>``, markdown fences)

        Raises:
            FormatError: When no valid action is found.
        """
        from psrl.tools.tool_parser.xml_fc_tool_parser import parse_xml_fc_to_bash

        cmd = parse_xml_fc_to_bash(content)
        if cmd is not None:
            return [{"command": cmd}]

        # No valid action found.
        raise FormatError(
            {
                "role": "user",
                "content": self.config.format_error_template,
                "extra": {
                    "interrupt_type": "FormatError",
                    "n_actions": 0,
                    "model_response": content,
                },
            }
        )


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
                        psrl_logger.debug(
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
        swe_task_id = ""

        # Initialize environment and agent data.
        env = MiniSWEEnvironment(
            self.config, self.reward_manager,
            runtime_config=self.__class__.runtime_config,
        )
        observation, info = await env.reset(task=request)

        swe_task_id = observation["swe_task_id"]
        log_prefix = f"[mini-SWE-agent, task_id={swe_task_id}]" if swe_task_id else "[mini-SWE-agent]"
        psrl_logger.debug(f"{log_prefix} Initializing agent data...")

        agent_data = MiniSWEAgentData(
            self.config, self.reward_manager, self.tokenizer, env,
        )
        agent_data.reset()
        agent_data.init_trajectory(request)

        runtime_config = observation["runtime_config"]
        sb = runtime_config.sandbox_config

        psrl_logger.debug(
            f"{log_prefix} Agent data initialized. "
            f"max_parallel={sb.max_parallel_tasks_per_worker}, "
            f"traj_output={self.traj_writer.output_dir!r} (enable={self.traj_writer.enable})."
        )

        try:
            # Acquire run slot if configured.
            if sb.max_parallel_tasks_per_worker > 0:
                psrl_logger.debug(
                    f"{log_prefix} Waiting for run slot "
                    f"(max_parallel={sb.max_parallel_tasks_per_worker})..."
                )
                run_slot = await SlotManager.acquire(
                    sb.max_parallel_tasks_per_worker, self.traj_writer.output_dir,
                    prefix="psrl_mini_swe_agent_slots",
                )
                psrl_logger.debug(
                    f"{log_prefix} Acquired run slot "
                    f"(slot={run_slot[1] if run_slot else 'n/a'})."
                )
            else:
                psrl_logger.debug(f"{log_prefix} No slot limit, proceeding directly.")

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
            run_args.extend(["--label", f"psrl.swe_task_id={swe_task_id}"])
            # Per-actor label consumed by the reaper sidecar in
            # psrl.utils.common.docker_utils. Lets the reaper force-remove
            # exactly this actor's containers when the actor process dies
            # under SIGKILL / OOM / Ray restart, instead of waiting for the
            # 2 h container_timeout fallback.
            _actor_id = os.environ.get("PSRL_ACTOR_ID", "")
            if _actor_id:
                run_args.extend(["--label", f"psrl.actor_id={_actor_id}"])
            if not use_preexisting and observation.get("repo_path"):
                run_args.extend(["--volume", f"{observation['repo_path']}:/testbed"])

            # Forward proxy env vars so pip/apt inside agent containers can
            # reach external package indexes through the corporate proxy.
            _PROXY_ENV_KEYS = [
                "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
                "no_proxy", "NO_PROXY",
            ]

            docker_env_kwargs = {
                "image": env_cfg.image,
                "cwd": effective_cwd,
                "env": env_cfg.env if isinstance(env_cfg.env, dict) else {},
                "forward_env": _PROXY_ENV_KEYS,
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

            m_cfg = runtime_config.model
            model = _PSRLModel(
                req_q, res_q,
                query_timeout=sb.query_timeout,
                model_name="psrl/rollout",
                cost_tracking="ignore_errors",
                action_regex=m_cfg.action_regex,
                observation_template=m_cfg.observation_template,
                format_error_template=m_cfg.format_error_template,
            )

            problem_statement = observation.get("problem_statement", "")

            _problem_short = f"{problem_statement[:60]}..." if len(problem_statement) > 60 else problem_statement
            psrl_logger.debug(
                f"{log_prefix} Starting episode: problem={_problem_short!r}, "
                f"cwd={effective_cwd!r}, image={env_cfg.image!r}, "
                f"max_turns={max_turns}."
            )

            # Run the agent in a dedicated worker thread (not the default pool,
            # which would deadlock with asyncio.to_thread used in _generation_loop).
            loop = asyncio.get_running_loop()
            psrl_logger.debug(f"{log_prefix} Launching agent in worker thread...")
            agent_future = loop.run_in_executor(
                _AGENT_THREAD_POOL,
                self._run_agent_sync,
                model, docker_env_kwargs, agent_kwargs, problem_statement, swe_task_id,
            )

            # Async generation loop: consume requests and generate via PSRL.
            # trajectory_output_path is resolved lazily inside the loop from the first output's version_tag.
            psrl_logger.debug(f"{log_prefix} Entering generation loop...")
            num_turns, patch, trajectory_output_path = await self._generation_loop(
                agent_future=agent_future,
                agent_data=agent_data,
                request=request,
                req_q=req_q,
                res_q=res_q,
                max_turns=max_turns,
                swe_problem_id=swe_task_id,
                container_timeout=env_cfg.container_timeout,
                rollout_turn_timeout=sb.rollout_turn_timeout,
                profiling_collector=profiling_collector,
            )

            # Wait for agent thread to finish.
            thread_exit_status = ""
            try:
                result = await asyncio.wait_for(agent_future, timeout=60.0)
                thread_patch = result.get("submission", "") if isinstance(result, dict) else ""
                thread_exit_status = result.get("exit_status", "") if isinstance(result, dict) else ""
                if thread_patch and not patch:
                    patch = thread_patch
            except (asyncio.TimeoutError, Exception) as e:
                psrl_logger.warning(f"{log_prefix} Agent thread did not finish cleanly: {e}.")

            agent_data.set_patch(patch)

            # --- Post-rollout grading (SWE-bench / SWE-smith instances) ---
            # Spawn a fresh container from the same image to grade the patch.
            # This is intentionally done after set_patch() and before
            # finalize_output() so that set_grader_result() can attach the
            # result to agent_reward_info before the reward function is called.
            grading_s = 0.0
            grader_kind_str = str(observation.get("swe_grader", "") or "")
            psrl_logger.warning(
                f"{log_prefix} GRADING CHECK: patch={bool(patch)}, "
                f"grader={grader_kind_str!r}, "
                f"swe_image={bool(observation.get('swe_problem_image', ''))}, "
                f"swe_problem={bool(observation.get('swe_problem', {}))}, "
                f"thread_exit_status={thread_exit_status!r}, "
                f"num_turns={num_turns}."
            )
            if patch and grader_kind_str == "swebench_fresh_container":
                swe_problem = observation.get("swe_problem", {})
                swe_image = str(observation.get("swe_problem_image", "") or "")
                swe_restore_tests = bool(observation.get("swe_restore_tests", False))
                grader_kind = (
                    "smith" if swe_restore_tests
                    else "gym" if isinstance(swe_problem, dict) and swe_problem.get("eval_script")
                    else "verified"
                )

                if swe_image and swe_problem:
                    from examples.mini_swe.swebench_grader import grade_fresh_container

                    loop = asyncio.get_running_loop()
                    psrl_logger.debug(
                        f"{log_prefix} Starting fresh-container grading: "
                        f"grader_kind={grader_kind!r}, image={swe_image!r}."
                    )
                    _grading_start = time.time()
                    try:
                        grader_result = await loop.run_in_executor(
                            _GRADER_THREAD_POOL,
                            grade_fresh_container,
                            swe_problem,
                            patch,
                            grader_kind,
                            swe_image,
                            900,
                            swe_task_id,
                            runtime_config.sandbox_config.environment.grader_memory,
                        )
                        agent_data.set_grader_result(grader_result)
                        grading_s = time.time() - _grading_start
                        psrl_logger.debug(
                            f"{log_prefix} Grading done: resolved={grader_result.get('resolved')}, "
                            f"elapsed={grading_s:.1f}s."
                        )
                    except Exception as grading_exc:
                        grading_s = time.time() - _grading_start
                        psrl_logger.error(
                            f"{log_prefix} Grading raised an unexpected error: {grading_exc}."
                        )
                        # FIX: Set a failure result instead of leaving it empty
                        f2p = swe_problem.get("FAIL_TO_PASS", [])
                        p2p = swe_problem.get("PASS_TO_PASS", [])
                        agent_data.set_grader_result({
                            "policy_violated": False,
                            "policy_reasons": [],
                            "resolved": False,
                            "apply_ok": False,
                            "f2p_pass": 0,
                            "f2p_total": len(f2p),
                            "p2p_pass": 0,
                            "p2p_total": len(p2p),
                            "timeout": False,
                            "error": str(grading_exc),
                            "elapsed_s": grading_s,
                            "output_tail": "",
                            "resolved_by": "grader_exception",
                        })
                else:
                    psrl_logger.warning(
                        f"{log_prefix} swe_grader={grader_kind_str!r} but swe_problem_image or "
                        f"swe_problem missing, skipping grading."
                    )

            total_elapsed = time.time() - run_start_time
            psrl_logger.debug(
                f"{log_prefix} Episode completed: {num_turns} turns, "
                f"patch={'yes' if patch else 'no'}, total={total_elapsed:.1f}s."
            )

            # Single source of truth for the termination classification: drives
            # both the human-readable "stop:" line in the summary and the
            # TerminateReason returned to the worker.
            terminate_reason = _classify_mini_swe_termination(
                thread_exit_status, num_turns, max_turns,
            )

            # Compute token-count breakdown from the trajectory for debugging.
            traj = agent_data.trajectory
            n_prompt = len(traj.prompt_ids)
            n_assistant = traj.response_mask.count(1)
            n_env = traj.response_mask.count(0)
            token_counts = (
                f"\n[Token Counts] prompt: {n_prompt}"
                f" | assistant: {n_assistant}"
                f" | env: {n_env}"
                f" | total: {n_prompt + n_assistant + n_env}"
            )

            # Build optional timing breakdown line when profiling data is available.
            timing_detail = ""
            if profiling_collector is not None:
                ts = profiling_collector.trajectory_start_ts
                if ts > 0:
                    bd = profiling_collector.get_timing_breakdown()
                    prep_s = max(ts - run_start_time, 0.0)
                    parts = [
                        f"prep: {prep_s:.1f}s",
                        f"assistant: {bd['assistant_s']:.1f}s",
                        f"env: {bd['env_s']:.1f}s",
                    ]
                    if grading_s > 0:
                        parts.append(f"grading: {grading_s:.1f}s")
                    timing_detail = "\n[Time Breakdown] " + " | ".join(parts)

            summary_text = ""
            if patch:
                summary_text += f"=== Submission ===\n{patch}\n\n"
            summary_text += (
                f"=== Summary ===\n"
                f"turns: {num_turns}, patch: {'yes' if patch else 'no'}, "
                f"stop: {terminate_reason.value}, elapsed: {total_elapsed:.1f}s"
                f"{token_counts}{timing_detail}\n"
            )
            self.traj_writer.append(trajectory_output_path, summary_text)

            finalized = await agent_data.finalize_output(request)

            if finalized is None:
                # finalize_output returns None when compute_score was aborted
                # (the PS manager already called notify_group_failed for this group).
                # Propagate as ABORTED so the worker drops this trajectory cleanly
                # without double-notifying the manager.
                return None, TerminateReason.ABORTED

            return finalized, terminate_reason

        finally:
            # Safety-net Docker cleanup via labels. Use ``docker rm -f`` (not
            # ``docker stop``) so a hung sleep PID 1 with active ``docker
            # exec`` sessions is reaped immediately rather than relying on the
            # 10s SIGTERM/SIGKILL grace window, which has been observed to
            # silently succeed without actually killing the container.
            if swe_task_id:
                await asyncio.to_thread(
                    force_remove_containers_by_label,
                    "psrl.swe_task_id",
                    swe_task_id,
                )
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
        swe_task_id: str = "",
    ) -> dict:
        """
        Run mini-swe-agent synchronously in a worker thread.

        Creates `DockerEnvironment` and `DefaultAgent` in-thread (Docker
        container starts immediately on construction), runs the agent, then
        cleans up the container.
        """
        log_prefix = f"[mini-SWE-agent, task_id={swe_task_id}]" if swe_task_id else "[mini-SWE-agent]"
        docker_env = None
        try:
            psrl_logger.debug(
                f"{log_prefix} Creating DockerEnvironment: image={docker_env_kwargs.get('image')!r}, "
                f"cwd={docker_env_kwargs.get('cwd')!r}."
            )
            # Pass a WARNING-level logger explicitly so docker startup noise is always suppressed,
            # regardless of how the root logger is configured in the Ray worker process.
            _docker_logger = logging.getLogger("minisweagent.environment")
            _docker_logger.setLevel(logging.WARNING)
            docker_env = DockerEnvironment(**docker_env_kwargs, logger=_docker_logger)
            psrl_logger.debug(
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
                psrl_logger.debug(f"{log_prefix} cwd={cwd!r} confirmed to exist inside container.")

            psrl_logger.debug(f"{log_prefix} Creating DefaultAgent with step_limit={agent_kwargs.get('step_limit')}.")
            agent = DefaultAgent(model, docker_env, **agent_kwargs)

            _task_short = f"{task[:60]}..." if len(task) > 60 else task
            psrl_logger.debug(f"{log_prefix} Running agent with task={_task_short!r}.")
            result = agent.run(task)

            submission = result.get("submission", "") if isinstance(result, dict) else ""
            exit_status = result.get("exit_status", "") if isinstance(result, dict) else ""
            psrl_logger.debug(
                f"{log_prefix} Agent finished: exit_status={exit_status!r}, "
                f"submission_len={len(submission)}, n_calls={agent.n_calls}."
            )
            return result
        except Exception as e:
            psrl_logger.exception(f"{log_prefix} Agent thread failed: {e}.")
            return {"exit_status": "error", "submission": ""}
        finally:
            if docker_env is not None:
                psrl_logger.debug(f"{log_prefix} Cleaning up DockerEnvironment...")
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
        container_timeout: str = "2h",
        rollout_turn_timeout: int = 480,
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

        log_prefix = f"[mini-SWE-agent, task_id={swe_problem_id}]" if swe_problem_id else "[mini-SWE-agent]"

        # Wall-clock budget: parsed from container_timeout with a 20 % safety
        # margin so PSRL always times out before the Docker daemon kills the
        # container.  This is the last line of defence against a trajectory
        # that hangs forever (e.g. a `pip install` that never returns, an
        # infinite loop that doesn't consume CPU, etc.).
        episode_timeout_secs = _parse_timeout_secs(container_timeout) * 0.8
        episode_start = time.time()

        psrl_logger.debug(
            f"{log_prefix} Generation loop started: "
            f"effective_limit={effective_limit}, max_turns={max_turns}, "
            f"episode_timeout={episode_timeout_secs:.0f}s."
        )

        # Accumulate trajectory text throughout the loop; written to disk at the end.
        traj_text: list[str] = [f"{log_prefix}\n\n"]

        while not agent_future.done():
            # Hard wall-clock cutoff — fires if the agent thread hangs without
            # ever putting a new message into req_q (e.g. a Docker command that
            # blocks forever).  We signal the agent thread via res_q so it can
            # clean up its DockerEnvironment before the outer finally-block runs.
            elapsed = time.time() - episode_start
            if elapsed > episode_timeout_secs:
                psrl_logger.warning(
                    f"{log_prefix} Episode wall-clock timeout "
                    f"({elapsed:.0f}s > {episode_timeout_secs:.0f}s), terminating agent."
                )
                res_q.put(_TerminateSignal("EpisodeTimeout"))
                break

            try:
                messages = req_q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.1)
                continue

            psrl_logger.debug(
                f"{log_prefix} Generation loop received request "
                f"(turn {num_turns + 1}): {len(messages)} messages."
            )

            # Capture the text of the latest user/tool message for the trajectory log.
            observation = messages[-1].get("content", "") if messages else ""

            normalized = normalize_openai_messages(messages)

            # Pre-update budget check: re-encode the full conversation with the
            # chat template to get an approximate token count before we update
            # the trajectory. This catches the common case early (no update needed).
            prompt_ids = agent_data.encode_messages(normalized, add_generation_prompt=True)
            if _check_token_budget(len(prompt_ids), effective_limit, log_prefix, num_turns + 1,
                                   label="re-encoded prompt length"):
                res_q.put(_TerminateSignal("PromptBudgetExhausted"))
                break

            # Drive the unified env protocol.
            # Turn 0: encode the full initial observation → sets trajectory.prompt_ids.
            # Turn N>0: encode only the new user/tool message via fixed-base delta
            #            → appends user delta to trajectory.response_ids (mask=0).
            if num_turns == 0:
                await agent_data.update_from_env(normalized, 0, False, {})
            else:
                await agent_data.update_from_env([normalized[-1]], 0, False, {})

            # Post-update budget check using the exact trajectory token count.
            # `encode_messages` above is approximate: the actual tokens sent to vLLM
            # are `trajectory.prompt_ids + trajectory.response_ids`, which accumulate
            # incrementally via delta encoding and may diverge from the re-encoded
            # count (e.g. due to BPE round-trip differences over many turns).
            traj_token_count = (
                len(agent_data.trajectory.prompt_ids)
                + len(agent_data.trajectory.response_ids)
            )
            if _check_token_budget(traj_token_count, effective_limit, log_prefix, num_turns + 1,
                                   label="trajectory token count"):
                res_q.put(_TerminateSignal("PromptBudgetExhausted"))
                break

            # Build generation request from trajectory state (prompt_ids + response_ids).
            gen_request = agent_data.prepare_generation_request(request)
            request_id = request.non_tensor_batch["uid"][0]
            try:
                async with maybe_sticky_session(
                    self.rollout_router,
                    request_id,
                    self.config.psrl.agentic_rl.sticky_session,
                ):
                    if profiling_collector is not None:
                        profiling_collector.on_turn_submit()
                    if self.config.psrl.server_rollout.enable:
                        gateway_client = RolloutGatewayClient.from_config(self.config)
                        output = await asyncio.wait_for(
                            gateway_client.generate_async(gen_request),
                            timeout=rollout_turn_timeout,
                        )
                    else:
                        output = await asyncio.wait_for(
                            self.rollout_router.generate_async.remote(gen_request),
                            timeout=rollout_turn_timeout,
                        )
            except asyncio.TimeoutError:
                # The rollout system did not produce a result within the per-turn
                # budget. Most often this means the routing layer silently lost
                # the request (e.g. result_future was never set due to an
                # unhandled exception in `_route_single_request`). We turn this
                # into a clean ROLLOUT_ERROR rather than leaving the agent thread
                # to time out on `_res_q.get(query_timeout)` later.
                psrl_logger.error(
                    f"{log_prefix} Turn {num_turns + 1}: rollout did not return "
                    f"within {rollout_turn_timeout}s, treating as ROLLOUT_ERROR."
                )
                res_q.put(_TerminateSignal("RolloutError"))
                break
            except Exception as rollout_exc:
                # Real rollout failure: propagate so run_with_termination_handling
                # maps it to TerminateReason.ROLLOUT_ERROR. We still signal the
                # agent thread first so it doesn't block on res_q.get for 600s
                # while the outer finally tears down the container/env.
                psrl_logger.error(
                    f"{log_prefix} Turn {num_turns + 1}: rollout raised an exception: {rollout_exc}. "
                    "Signalling agent thread and re-raising as ROLLOUT_ERROR.",
                    exc_info=True,
                )
                res_q.put(_TerminateSignal("RolloutError"))
                raise

            if output is None:
                psrl_logger.warning(
                    f"{log_prefix} Turn {num_turns + 1}: rollout returned None, terminating agent."
                )
                res_q.put(_TerminateSignal("RolloutReturnedNone"))
                break

            if profiling_collector is not None:
                profiling_collector.on_turn_complete(output)

            # Read token count before update_from_model_token_ids consumes raw_response_ids.
            n_response_tokens = len(output.non_tensor_batch["raw_response_ids"][0])

            # Capture version_tag from the first real output (initial request carries -1).
            if resolved_version is None:
                resolved_version = int(output.non_tensor_batch["version_tag"][0])

            # Append assistant response to trajectory (mask=1); decodes text into step.
            await agent_data.update_from_model_token_ids(output)
            response_text = agent_data.get_current_step().model_response
            num_turns += 1

            # Send response text back to the agent worker thread.
            res_q.put(response_text)

            # Append turn to trajectory text buffer.
            traj_text.append(
                f"=== Turn {num_turns} ===\n"
                f"--- observation ---\n{observation}\n\n"
                f"--- assistant ---\n{response_text}\n\n"
            )

            psrl_logger.debug(
                f"{log_prefix} Turn {num_turns}: {n_response_tokens} model tokens."
            )

            if num_turns >= max_turns:
                # DefaultAgent.query() checks step_limit at the start of the
                # next turn and raises LimitsExceeded before pushing to req_q,
                # so no terminate signal is needed here.
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
