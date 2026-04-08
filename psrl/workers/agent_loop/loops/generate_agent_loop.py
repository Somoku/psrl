import logging
import os

import numpy as np
from verl import DataProto

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
        output = await self.generate_sequence(request)
        if output is not None:
            response_ids = output.token_ids
            response_ids = response_ids[: self.response_length]
            response_mask = [1] * len(response_ids)
            interrupted = output.interrupted
            interrupted_by_scheduler = output.interrupted_by_scheduler
            rollout_instance_id = output.rollout_instance_id
            request.non_tensor_batch["raw_response_ids"] = np.array([response_ids])
            request.non_tensor_batch["response_mask"] = np.array([response_mask])
            request.non_tensor_batch["__num_turns__"] = np.array([2])
            request.non_tensor_batch["interrupted"] = np.array([interrupted])
            request.non_tensor_batch["interrupted_by_scheduler"] = np.array([interrupted_by_scheduler])
            request.non_tensor_batch["multi_modal_data"] = np.array([output.multi_modal_data], dtype=object)
            request.non_tensor_batch["rollout_instance_id"] = np.array([rollout_instance_id])
            if output.log_probs is not None:
                rollout_log_probs = output.log_probs
                rollout_log_probs = rollout_log_probs[: self.response_length]
                request.non_tensor_batch["rollout_log_probs"] = np.array([rollout_log_probs], dtype=object)
            if output.routed_experts is not None:
                # TODO(linsh): support router replay
                routed_experts = output.routed_experts
                request.non_tensor_batch["routed_experts"] = np.array([routed_experts], dtype=object)
        else:
            # Indicate that the request is aborted
            return None, TerminateReason.UNKNOWN

        reward_input = request
        reward_result = await self.reward_manager.compute_score.remote(reward_input)
        if not self.config.reward.launch_reward_fn_async:
            output = self._post_process_and_merge_reward(reward_result, request)
        # psrl_logger.info(f"output of {request.non_tensor_batch['uid'][0]} = {output}")

        return output, TerminateReason.FINISHED
