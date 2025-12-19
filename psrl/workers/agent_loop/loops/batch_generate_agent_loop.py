import logging
import os

import numpy as np
from verl import DataProto

from psrl.workers.agent_loop.loops.base_agent_loop import AgentLoopBase
from psrl.workers.agent_loop.loops.utils import register

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@register("batch_generate_only_agent")
class BatchGenerateAgentLoop(AgentLoopBase):
    """Agent loop that performs batch generation for multiple requests simultaneously."""

    def __init__(self, *args, **kwargs):
        """Initialize the batch generation agent loop."""
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.gen_actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.gen_actor_rollout_ref.rollout.response_length

    async def run(self, request: DataProto) -> DataProto:
        """Execute batch generation for the given requests.

        Args:
            request (DataProto): Batch of input requests.

        Returns:
            DataProto: Generated responses with metadata.
        """
        output = await self.rollout_router.generate.remote(request)
        assert "eos_token_id" in output.meta_info, "eos_token_id is not in the meta_info"
        if output is not None:
            batch_size = len(output)
            response_mask_list = []
            num_turns_list = []
            for i in range(batch_size):
                response_ids = output.non_tensor_batch["raw_response_ids"][i]
                valid_length = len(response_ids)
                for idx, token_id in enumerate(response_ids):
                    if token_id == output.meta_info["eos_token_id"]:
                        valid_length = idx + 1
                        break
                assert valid_length <= self.response_length, "response_mask is longer than the response_length"
                response_mask = [1] * valid_length
                response_mask_list.append(response_mask)
                num_turns_list.append(0)
            # response mask: bsz * [1, 1, ..., 1] (since no tool call, all the tokens are valid)
            output.non_tensor_batch["response_mask"] = np.fromiter(response_mask_list, dtype=object)
            output.non_tensor_batch["__num_turns__"] = np.array(num_turns_list)

        reward_input = output
        reward_result = await self.reward_manager.compute_score.remote(reward_input)
        if not self.config.reward_model.launch_reward_fn_async:
            output = self._post_process_and_merge_reward(reward_result, output)

        return output
