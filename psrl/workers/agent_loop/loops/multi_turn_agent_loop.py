import asyncio
import logging
import os
import time
from contextlib import contextmanager

from omegaconf import DictConfig
from verl import DataProto

from psrl.environments.base import Environment
from psrl.utils.profiling.collector import TurnProfilingCollector
from psrl.utils.rollout.rollout_trace import rollout_trace_op
from psrl.workers.agent_loop.agent_data import AgentData
from psrl.workers.agent_loop.loops.base_agent_loop import AgentLoopBase
from psrl.workers.agent_loop.loops.utils import TerminateReason, register
from psrl.workers.agent_loop.sticky_session import maybe_sticky_session

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@contextmanager
def _append_timer(timings_list: list):
    """Context manager that records elapsed time and appends to a list."""
    t0 = time.perf_counter()
    yield
    timings_list.append(time.perf_counter() - t0)


def _set_multi_turn_aggregates(
    multi_turn_metrics: dict,
    generate_times: list,
    env_step_times: list,
) -> None:
    """Write max/min/avg for each phase into multi_turn_metrics."""
    for name, times in [
        ("generate", generate_times),
        ("env_step", env_step_times),
    ]:
        if times:
            multi_turn_metrics[f"multi_turn/{name}_per_turn"] = sum(times) / len(times)
            multi_turn_metrics[f"multi_turn/{name}_max_per_turn"] = max(times)
            multi_turn_metrics[f"multi_turn/{name}_min_per_turn"] = min(times)
            multi_turn_metrics[f"multi_turn/{name}_all_turns"] = sum(times)
            multi_turn_metrics[f"multi_turn/max_{name}_all_turns"] = sum(times)
            multi_turn_metrics[f"multi_turn/min_{name}_all_turns"] = sum(times)
        else:
            multi_turn_metrics[f"multi_turn/{name}_per_turn"] = 0
            multi_turn_metrics[f"multi_turn/{name}_max_per_turn"] = 0
            multi_turn_metrics[f"multi_turn/{name}_min_per_turn"] = 0
            multi_turn_metrics[f"multi_turn/{name}_all_turns"] = 0
            multi_turn_metrics[f"multi_turn/max_{name}_all_turns"] = 0
            multi_turn_metrics[f"multi_turn/min_{name}_all_turns"] = 0


def _format_observation(observation: object) -> str:
    """
    Format an environment observation for human-readable trajectory output.

    Handles `ConversationType` (list of role/content dicts) by rendering each
    message as `[role]: content`.  Falls back to `str()` for any other type.
    """
    if isinstance(observation, list) and observation and isinstance(observation[0], dict):
        parts = []
        for msg in observation:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            parts.append(f"[{role}]: {content}")
        return "\n\n".join(parts)
    return str(observation)


@register("multi_turn_agent")
class MultiTurnAgentLoop(AgentLoopBase):
    """
    Agent loop that performs multi-turn generation in streaming mode.

    This loop interacts with an environment that supports multiple turns of interaction.
    1. Initializes the environment and agent data for multi-turn interactions.
    2. Resets the environment and prepares the initial observation.
    3. Iteratively generates actions based on the current observation and updates the environment.
    4. Handles termination conditions such as maximum turns, timeouts, and completion signals.
    5. Finalizes and returns the generated response along with termination metadata.
    """

    @classmethod
    def init_class(cls, config: DictConfig, **kwargs) -> None:
        """Perform heavy initialization work shared across all instances.

        This method is called only once per class to avoid redundant initialization.

        Args:
            config (DictConfig): Configuration object containing training settings.
            **kwargs: Additional keyword arguments from configuration.
        """
        if cls._class_initialized:
            return
        cls._class_initialized = True

        cls.prompt_length = config.gen_actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.gen_actor_rollout_ref.rollout.response_length

        cls.max_turns = config.gen_actor_rollout_ref.rollout.multi_turn.max_turns
        cls.env_step_timeout = config.gen_actor_rollout_ref.rollout.agent.env.step_timeout

    @rollout_trace_op
    async def run(
        self,
        request: DataProto,
        profiling_collector: TurnProfilingCollector | None = None,
    ) -> tuple[DataProto | None, TerminateReason]:
        """Execute generation for a single request.

        Args:
            request (DataProto): Single input request.
            profiling_collector: Per-trajectory profiling collector, or None if disabled.

        Returns:
            Tuple[DataProto, TerminateReason]: Generated response with metadata and termination reason.
        """
        # Initialize metrics container for multi-turn profiling
        multi_turn_metrics = request.meta_info.setdefault("metrics", {})
        generate_times: list[float] = []
        env_step_times: list[float] = []

        # Trajectory text accumulator and version/id state for output writing.
        traj_text: list[str] = []
        resolved_version: int | None = None
        traj_id = str(request.non_tensor_batch["uid"][0])
        turn_count = 0

        env_class = request.non_tensor_batch.get(
            "env_class", [self.config.gen_actor_rollout_ref.rollout.agent.env.name]
        )[0]
        data_class = request.non_tensor_batch.get(
            "data_class", [self.config.gen_actor_rollout_ref.rollout.agent.data.name]
        )[0]

        self.env = Environment.get_environment(
            env_class,
            self.config,
            self.reward_manager,
            self.max_turns,
        )
        self.agent_data = AgentData.get_agent_data(
            data_class,
            self.config,
            self.reward_manager,
            self.tokenizer,
            self.env,
        )
        self.agent_data.reset()

        # Example of ToolAgent:
        # observation: raw tokens (initial prompt or tool output)
        # update_from_env: create a Step, set Step chat completions, append user tokens to trajectory (masked)
        # finalize_output: turn trajectory into DataProto
        # prepare_generation_request: concatenate prompt_ids and response_ids to raw_prompt_ids
        # update_from_model_token_ids: append assistant tokens to trajectory (unmasked), decode and
        #   parse tool calls, add assistant message to Step chat completions, compute reward if
        #   using step-level reward mode
        # action: tool calls dict
        # env.step: execute tool calls

        observation, info = await self.env.reset(
            task=request,
            seed=request.non_tensor_batch.get("seed", None),
        )

        # Record the initial observation (problem statement) before any generation.
        traj_text.append(f"=== Initial Observation ===\n{_format_observation(observation)}\n\n")

        self.agent_data.init_trajectory(request)

        overlong_terminate = await self.agent_data.update_from_env(observation, 0, False, info)

        if overlong_terminate:
            _set_multi_turn_aggregates(
                multi_turn_metrics,
                generate_times,
                env_step_times,
            )
            finalized = await self.agent_data.finalize_output(request)
            terminate_reason = TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED
            traj_text.append(f"=== End: {terminate_reason.value} ===\n")
            if resolved_version is not None:
                self.traj_writer.write(resolved_version, traj_id, "".join(traj_text))
            return finalized, terminate_reason

        for _ in range(self.max_turns):
            # Currently we still use token-in-token-out generation,
            # but in the future we may switch to chat-completion style generation.

            # TODO: check unnecessary fields in output data_proto and
            # check redundant padding in single-request case
            async with maybe_sticky_session(
                self.rollout_router,
                request.non_tensor_batch["uid"][0],
                self.config.psrl.agentic_rl.sticky_session,
            ):
                with _append_timer(generate_times):
                    gen_request = self.agent_data.prepare_generation_request(request)
                    if profiling_collector is not None:
                        profiling_collector.on_turn_submit()
                    output = await self.rollout_router.generate_async.remote(gen_request)

            if output is None:
                _set_multi_turn_aggregates(
                    multi_turn_metrics,
                    generate_times,
                    env_step_times,
                )
                terminate_reason = TerminateReason.ABORTED
                traj_text.append(f"=== End: {terminate_reason.value} ===\n")
                if resolved_version is not None:
                    self.traj_writer.write(resolved_version, traj_id, "".join(traj_text))
                return None, terminate_reason

            # Capture version tag from the first successful output.
            if resolved_version is None:
                resolved_version = int(output.non_tensor_batch["version_tag"][0])

            # Record profiling data for this turn.
            if profiling_collector is not None:
                profiling_collector.on_turn_complete(output)

            # Decode response text before update_from_model_token_ids, which
            # pops "raw_response_ids" from output.non_tensor_batch via .pop().
            turn_count += 1
            response_ids = output.non_tensor_batch["raw_response_ids"][0]
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            # TODO: we may implement `update_from_model_chat_completion` in the future.
            action, overlong_terminate = await self.agent_data.update_from_model_token_ids(output)

            if overlong_terminate:
                _set_multi_turn_aggregates(
                    multi_turn_metrics,
                    generate_times,
                    env_step_times,
                )
                finalized = await self.agent_data.finalize_output(request)
                terminate_reason = TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED
                traj_text.append(
                    f"=== Turn {turn_count} ===\n"
                    f"--- assistant ---\n{response_text}\n\n"
                )
                traj_text.append(f"=== End: {terminate_reason.value} ===\n")
                self.traj_writer.write(resolved_version, traj_id, "".join(traj_text))
                return finalized, terminate_reason

            try:
                with _append_timer(env_step_times):
                    env_step_output = await asyncio.wait_for(self.env.step(action), timeout=self.env_step_timeout)
                observation = env_step_output["observation"]
                reward = env_step_output["reward"]
                done = env_step_output["done"]
                info = env_step_output["info"]
            except asyncio.TimeoutError:
                _set_multi_turn_aggregates(
                    multi_turn_metrics,
                    generate_times,
                    env_step_times,
                )
                finalized = await self.agent_data.finalize_output(request)
                terminate_reason = TerminateReason.ENV_TIMEOUT
                traj_text.append(
                    f"=== Turn {turn_count} ===\n"
                    f"--- assistant ---\n{response_text}\n\n"
                )
                traj_text.append(f"=== End: {terminate_reason.value} ===\n")
                self.traj_writer.write(resolved_version, traj_id, "".join(traj_text))
                return finalized, terminate_reason

            # Append turn (response + subsequent observation) to trajectory buffer.
            traj_text.append(
                f"=== Turn {turn_count} ===\n"
                f"--- assistant ---\n{response_text}\n\n"
                f"--- observation ---\n{_format_observation(observation)}\n\n"
            )

            overlong_terminate = await self.agent_data.update_from_env(observation, reward, done, info)

            if overlong_terminate:
                _set_multi_turn_aggregates(
                    multi_turn_metrics,
                    generate_times,
                    env_step_times,
                )
                finalized = await self.agent_data.finalize_output(request)
                terminate_reason = TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED
                traj_text.append(f"=== End: {terminate_reason.value} ===\n")
                self.traj_writer.write(resolved_version, traj_id, "".join(traj_text))
                return finalized, terminate_reason

            if done:
                _set_multi_turn_aggregates(
                    multi_turn_metrics,
                    generate_times,
                    env_step_times,
                )
                finalized = await self.agent_data.finalize_output(request)
                terminate_reason = TerminateReason.FINISHED
                traj_text.append(f"=== End: {terminate_reason.value} ===\n")
                self.traj_writer.write(resolved_version, traj_id, "".join(traj_text))
                return finalized, terminate_reason

        _set_multi_turn_aggregates(
            multi_turn_metrics,
            generate_times,
            env_step_times,
        )
        finalized = await self.agent_data.finalize_output(request)
        terminate_reason = TerminateReason.MAX_TURNS_EXCEEDED
        traj_text.append(f"=== End: {terminate_reason.value} ===\n")
        self.traj_writer.write(resolved_version, traj_id, "".join(traj_text))
        return finalized, terminate_reason
