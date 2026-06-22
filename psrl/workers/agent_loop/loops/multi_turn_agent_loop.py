import asyncio
import logging
import os

import ray
from transformers import AutoProcessor, AutoTokenizer
from verl.utils.dataset.rl_dataset import RLHFDataset

from psrl.environments.base import Environment
from psrl.utils.rollout.rollout_trace import rollout_trace_op
from psrl.workers.agent_loop.agent_data import AgentData
from psrl.workers.agent_loop.loops.base_agent_loop import AgentLoopBase
from psrl.workers.agent_loop.loops.utils import DictConfigWrap, TerminateReason, register
from psrl.workers.gen.utils import TokenOutput

psrl_logger = logging.getLogger(__name__)
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

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        rollout_router: ray.actor.ActorHandle | str,
        reward_manager: ray.actor.ActorHandle,
        ps_manager_handle: ray.actor.ActorHandle,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        dataset_cls: type[RLHFDataset],
        data_config: DictConfigWrap,
        **kwargs,
    ):
        super().__init__(
            trainer_config=trainer_config,
            rollout_router=rollout_router,
            reward_manager=reward_manager,
            ps_manager_handle=ps_manager_handle,
            tokenizer=tokenizer,
            processor=processor,
            dataset_cls=dataset_cls,
            data_config=data_config,
            **kwargs,
        )
        self.max_turns = trainer_config.config.gen_actor_rollout_ref.rollout.multi_turn.max_turns
        self.env_step_timeout = trainer_config.config.gen_actor_rollout_ref.rollout.agent.env.step_timeout

    def get_generate_fields(self) -> list[str]:
        fields = super().get_generate_fields()
        fields.extend(["env_class", "data_class", "seed"])
        return fields

    @rollout_trace_op
    async def run(self, request: dict) -> tuple[TokenOutput | list[TokenOutput] | None, TerminateReason]:
        """Execute generation for a single request.

        Args:
            request (dict): Single input request.

        Returns:
            Tuple[TokenOutput | list[TokenOutput] | None, TerminateReason]:
                Generated response with metadata and termination reason.
        """
        env_class = request.get("env_class", self.config.gen_actor_rollout_ref.rollout.agent.env.name)
        data_class = request.get("data_class", self.config.gen_actor_rollout_ref.rollout.agent.data.name)

        self.env = Environment.get_environment(
            env_class,
            self.config,
            self.reward_manager,
            self.max_turns,
            tokenizer=self.tokenizer,
            processor=self.processor,
            dataset_cls=self.dataset_cls,
        )
        self.agent_data = AgentData.get_agent_data(
            data_class,
            self.config,
            self.reward_manager,
            self.env,
        )
        self.agent_data.reset()

        observation, info = await self.env.reset(
            task=request,
            seed=request.get("seed", None),
        )

        self.agent_data.init_trajectory(request)

        overlong_terminate = await self.agent_data.update_from_env(observation, 0, False, info)

        if overlong_terminate:
            return await self.agent_data.finalize_output(), TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED

        for _ in range(self.max_turns):
            # Currently we still use token-in-token-out generation,
            # but in the future we may switch to chat-completion style generation.

            # TODO: check unnecessary fields in output data_proto and
            # check redundant padding in single-request case
            output = await self.generate_sequence(
                self.agent_data.prepare_generation_request(request),
                is_sticky_session=self.config.psrl.routing_strategy.enable_trajectory_sticky,
            )

            if output is None:
                return None, TerminateReason.ABORTED

            # TODO: we may implement `update_from_model_chat_completion` in the future.
            action, overlong_terminate = await self.agent_data.update_from_model_token_ids(output)

            if overlong_terminate:
                return await self.agent_data.finalize_output(), TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED

            try:
                with self.timer.env():
                    env_step_output = await asyncio.wait_for(
                        self.env.step(action),
                        timeout=self.env_step_timeout,
                    )
                observation = env_step_output["observation"]
                reward = env_step_output["reward"]
                done = env_step_output["done"]
                info = env_step_output["info"]
            except asyncio.TimeoutError:
                return await self.agent_data.finalize_output(), TerminateReason.ENV_TIMEOUT

            overlong_terminate = await self.agent_data.update_from_env(observation, reward, done, info)

            if overlong_terminate:
                return await self.agent_data.finalize_output(), TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED

            if done:
                return await self.agent_data.finalize_output(), TerminateReason.FINISHED

        return await self.agent_data.finalize_output(), TerminateReason.MAX_TURNS_EXCEEDED
