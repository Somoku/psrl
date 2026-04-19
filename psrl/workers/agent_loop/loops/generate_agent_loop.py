import logging
import os

import numpy as np
from verl import DataProto

from psrl.utils.profiling.collector import TurnProfilingCollector
from psrl.workers.agent_loop.gateway_client import RolloutGatewayClient
from psrl.workers.agent_loop.loops.base_agent_loop import AgentLoopBase
from psrl.workers.agent_loop.loops.utils import TerminateReason, register

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@register("generate_only_agent")
class GenerateAgentLoop(AgentLoopBase):
    """Agent loop that performs single-request generation in streaming mode."""

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
            Tuple[DataProto, TerminateReason]:
                Generated response with metadata and termination reason.
        """
        if profiling_collector is not None:
            profiling_collector.on_turn_submit()
        if self.config.psrl.server_rollout.enable:
            gateway_client = RolloutGatewayClient.from_config(self.config)
            output = await gateway_client.generate_async(request)
        else:
            output = await self.rollout_router.generate_async.remote(request)
        if output is not None:
            if profiling_collector is not None:
                profiling_collector.on_turn_complete(output)
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

            # Write per-trajectory text file when enabled.
            version = int(output.non_tensor_batch["version_tag"][0])
            traj_id = str(request.non_tensor_batch["uid"][0])
            prompt_ids_raw = request.non_tensor_batch.get("raw_prompt_ids", [None])[0]
            prompt_text = (
                self.tokenizer.decode(prompt_ids_raw, skip_special_tokens=True)
                if prompt_ids_raw is not None
                else ""
            )
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            traj_text = (
                f"=== Prompt ===\n{prompt_text}\n\n"
                f"=== Response ===\n{response_text}\n"
            )
            self.traj_writer.write(version, traj_id, traj_text)
        else:
            # Indicate that the request is aborted
            return None, TerminateReason.UNKNOWN

        reward_input = output
        reward_result = await self.reward_manager.compute_score.remote(reward_input)
        if not self.config.reward_model.launch_reward_fn_async:
            output = self._post_process_and_merge_reward(reward_result, output)

        return output, TerminateReason.FINISHED
