import logging
import os

import numpy as np
from verl import DataProto

from psrl.workers.agent_loop.loops.base_agent_loop import AgentLoopBase
from psrl.workers.agent_loop.loops.utils import TerminateReason, register

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@register("batch_generate_only_agent")
class BatchGenerateAgentLoop(AgentLoopBase):
    """Agent loop that performs batch generation for multiple requests simultaneously."""

    async def run(self, request: DataProto) -> tuple[DataProto | None, TerminateReason]:
        """Execute batch generation for the given requests.

        Args:
            request (DataProto): Batch of input requests.

        Returns:
            Tuple[DataProto, TerminateReason]:
                Generated responses with metadata and termination reason.
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
                num_turns_list.append(2)
            # response mask: bsz * [1, 1, ..., 1] (since no tool call, all the tokens are valid)
            output.non_tensor_batch["response_mask"] = np.fromiter(response_mask_list, dtype=object)
            output.non_tensor_batch["__num_turns__"] = np.array(num_turns_list)
            if "rollout_log_probs" in output.non_tensor_batch:
                rollout_log_probs_list = output.non_tensor_batch["rollout_log_probs"]
                trimmed_rollout_log_probs_list = []
                for i in range(batch_size):
                    rollout_log_probs = rollout_log_probs_list[i]
                    rollout_log_probs = rollout_log_probs[: self.response_length]
                    trimmed_rollout_log_probs_list.append(rollout_log_probs)
                output.non_tensor_batch["rollout_log_probs"] = np.fromiter(
                    trimmed_rollout_log_probs_list, dtype=object
                )
        else:
            return None, TerminateReason.UNKNOWN

        reward_input = output
        reward_result = await self.reward_manager.compute_score.remote(reward_input)
        if not self.config.reward_model.launch_reward_fn_async:
            output = self._post_process_and_merge_reward(reward_result, output)

        return output, TerminateReason.FINISHED
