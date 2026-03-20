import asyncio
import logging
import os
import time
from contextlib import contextmanager

from omegaconf import DictConfig
from verl import DataProto

from psrl.environments.base import Environment
from psrl.utils.rollout.rollout_trace import rollout_trace_op
from psrl.workers.agent_loop.agent_data import AgentData
from psrl.workers.agent_loop.loops.base_agent_loop import AgentLoopBase
from psrl.workers.agent_loop.loops.utils import TerminateReason, register
from psrl.workers.agent_loop.sticky_session import sticky_session

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
    async def run(self, request: DataProto) -> tuple[DataProto | None, TerminateReason]:
        """Execute generation for a single request.

        Args:
            request (DataProto): Single input request.

        Returns:
            Tuple[DataProto, TerminateReason]: Generated response with metadata and termination reason.
        """
        # Initialize metrics container for multi-turn profiling
        multi_turn_metrics = request.meta_info.setdefault("metrics", {})
        generate_times: list[float] = []
        env_step_times: list[float] = []

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

        self.agent_data.init_trajectory(request)

        overlong_terminate = await self.agent_data.update_from_env(observation, 0, False, info)

        if overlong_terminate:
            _set_multi_turn_aggregates(
                multi_turn_metrics,
                generate_times,
                env_step_times,
            )
            return await self.agent_data.finalize_output(request), TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED

        for _ in range(self.max_turns):
            # Currently we still use token-in-token-out generation,
            # but in the future we may switch to chat-completion style generation.

            # TODO: check unnecessary fields in output data_proto and
            # check redundant padding in single-request case
            async with sticky_session(self.rollout_router, request):
                with _append_timer(generate_times):
                    output = await self.rollout_router.generate_async.remote(
                        self.agent_data.prepare_generation_request(request)
                    )

            if output is None:
                _set_multi_turn_aggregates(
                    multi_turn_metrics,
                    generate_times,
                    env_step_times,
                )
                return None, TerminateReason.ABORTED

            # TODO: we may implement `update_from_model_chat_completion` in the future.
            action, overlong_terminate = await self.agent_data.update_from_model_token_ids(output)

            if overlong_terminate:
                _set_multi_turn_aggregates(
                    multi_turn_metrics,
                    generate_times,
                    env_step_times,
                )
                return await self.agent_data.finalize_output(request), TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED

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
                return await self.agent_data.finalize_output(request), TerminateReason.ENV_TIMEOUT

            overlong_terminate = await self.agent_data.update_from_env(observation, reward, done, info)

            if overlong_terminate:
                _set_multi_turn_aggregates(
                    multi_turn_metrics,
                    generate_times,
                    env_step_times,
                )
                return await self.agent_data.finalize_output(request), TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED

            if done:
                _set_multi_turn_aggregates(
                    multi_turn_metrics,
                    generate_times,
                    env_step_times,
                )
                return await self.agent_data.finalize_output(request), TerminateReason.FINISHED

        _set_multi_turn_aggregates(
            multi_turn_metrics,
            generate_times,
            env_step_times,
        )
        return await self.agent_data.finalize_output(request), TerminateReason.MAX_TURNS_EXCEEDED
