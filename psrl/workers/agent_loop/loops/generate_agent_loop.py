import logging
import os

import numpy as np
from verl import DataProto

from psrl.workers.agent_loop.gateway_client import RolloutGatewayClient
from psrl.workers.agent_loop.loops.base_agent_loop import AgentLoopBase
from psrl.workers.agent_loop.loops.utils import TerminateReason, register

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@register("generate_only_agent")
class GenerateAgentLoop(AgentLoopBase):
    """Agent loop that performs single-request generation in streaming mode."""

    async def run(self, request: DataProto) -> tuple[DataProto | None, TerminateReason]:
        """Execute generation for a single request.

        Args:
            request (DataProto): Single input request.

        Returns:
            Tuple[DataProto, TerminateReason]:
                Generated response with metadata and termination reason.
        """
        if self.config.psrl.server_rollout.enable:
            gateway_client = RolloutGatewayClient.from_config(self.config)
            output = await gateway_client.generate_async(request)
        else:
            output = await self.rollout_router.generate_async.remote(request)
        if output is not None:
            response_ids = output.non_tensor_batch["raw_response_ids"][0]
            response_ids = response_ids[: self.response_length]
            response_mask = [1] * len(response_ids)
            output.non_tensor_batch["raw_response_ids"] = np.array([response_ids])
            output.non_tensor_batch["response_mask"] = np.array([response_mask])
            output.non_tensor_batch["__num_turns__"] = np.array([2])
            if "rollout_log_probs" in output.non_tensor_batch:
                rollout_log_probs = output.non_tensor_batch["rollout_log_probs"][0]
                rollout_log_probs = rollout_log_probs[: self.response_length]
                output.non_tensor_batch["rollout_log_probs"] = np.array([rollout_log_probs])
        else:
            # Indicate that the request is aborted
            return None, TerminateReason.UNKNOWN

        reward_input = output
        reward_result = await self.reward_manager.compute_score.remote(reward_input)
        if not self.config.reward_model.launch_reward_fn_async:
            output = self._post_process_and_merge_reward(reward_result, output)

        return output, TerminateReason.FINISHED
