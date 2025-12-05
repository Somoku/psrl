import logging
import os

from verl import DataProto
from verl.utils.profiler import simple_timer

from psrl.workers.agent_loop.manager import AgentLoopBase, AgentLoopOutput, register

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.gen_actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.gen_actor_rollout_ref.rollout.response_length

    async def run(self, request: DataProto) -> AgentLoopOutput:
        metrics = {}
        messages = request.non_tensor_batch.get("messages", [[]])[0]
        request_ids = request.non_tensor_batch.get("uid", [None])[0]
        sampling_params = request.meta_info.get("sampling_params", None)

        prompt_ids = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        )

        with simple_timer("generate_sequences", metrics):
            response_ids = await self.rollout_router.generate(
                request_ids=request_ids, prompt_ids=prompt_ids, sampling_params=sampling_params
            )
        response_mask = [1] * len(response_ids)

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            num_turns=1,
            metrics=metrics,
        )
        return output
