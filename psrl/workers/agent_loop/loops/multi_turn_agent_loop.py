import asyncio
import logging
import os

from omegaconf import DictConfig
from verl import DataProto

from psrl.environments.base import Environment
from psrl.utils.rollout.rollout_trace import rollout_trace_op
from psrl.workers.agent_loop.agent_data import AgentData
from psrl.workers.agent_loop.loops.base_agent_loop import AgentLoopBase
from psrl.workers.agent_loop.loops.utils import TerminateReason, register

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


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

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the multi-turn agent loop."""
        super().__init__(*args, **kwargs)
        self.env = Environment.get_environment(
            self.config.gen_actor_rollout_ref.rollout.agent.env.name,
            self.config,
            self.reward_manager,
            self.max_turns,
        )
        self.agent_data = AgentData.get_agent_data(
            "tool_agent_data",
            self.config,
            self.reward_manager,
            self.tokenizer,
            self.env.get_tool_schemas(),
        )
        self.agent_data.reset()

    @rollout_trace_op
    async def run(self, request: DataProto) -> tuple[DataProto | None, TerminateReason]:
        """Execute generation for a single request.

        Args:
            request (DataProto): Single input request.

        Returns:
            Tuple[DataProto, TerminateReason]: Generated response with metadata and termination reason.
        """
        observation, info = await self.env.reset(
            task=request,
            seed=request.non_tensor_batch.get("seed", None),
        )

        self.agent_data.init_trajectory(request)

        overlong_terminate = await self.agent_data.update_from_env(observation, 0, False, info)

        if overlong_terminate:
            return await self.agent_data.finalize_output(request), TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED

        for _ in range(self.max_turns):
            # TODO: replace rollout_router with global service
            # Currently we still use token-in-token-out generation,
            # but in the future we may switch to chat-completion style generation.

            # TODO: check unnecessary fields in output data_proto and
            # check redundant padding in single-request case
            output = await self.rollout_router.generate_async.remote(
                self.agent_data.prepare_generation_request(request)
            )

            if output is None:
                return None, TerminateReason.ABORTED

            # TODO: we may implement `update_from_model_chat_completion` in the future.
            action, overlong_terminate = await self.agent_data.update_from_model_token_ids(output)

            if overlong_terminate:
                return await self.agent_data.finalize_output(request), TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED

            try:
                env_step_output = await asyncio.wait_for(self.env.step(action), timeout=self.env_step_timeout)
                observation = env_step_output["observation"]
                reward = env_step_output["reward"]
                done = env_step_output["done"]
                info = env_step_output["info"]
            except asyncio.TimeoutError:
                return await self.agent_data.finalize_output(request), TerminateReason.ENV_TIMEOUT

            overlong_terminate = await self.agent_data.update_from_env(observation, reward, done, info)

            if overlong_terminate:
                return await self.agent_data.finalize_output(request), TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED

            if done:
                return await self.agent_data.finalize_output(request), TerminateReason.FINISHED

        return await self.agent_data.finalize_output(request), TerminateReason.MAX_TURNS_EXCEEDED
