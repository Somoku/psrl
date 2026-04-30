import logging
import os

from psrl.workers.agent_loop.loops.base_agent_loop import AgentLoopBase
from psrl.workers.agent_loop.loops.utils import TerminateReason, register
from psrl.workers.gen_dplb.utils import TokenOutput

psrl_logger = logging.getLogger(__name__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@register("generate_only_agent")
class GenerateAgentLoop(AgentLoopBase):
    """Agent loop that performs single-request generation in streaming mode."""

    async def run(self, request: dict) -> tuple[TokenOutput | list[TokenOutput] | None, TerminateReason]:
        """Execute generation for a single request.

        Args:
            request (dict): Single input request.

        Returns:
            Tuple[TokenOutput, TerminateReason]:
                Generated response with metadata and termination reason.
        """
        output = await self.generate_sequence(request)
        if output is None:
            # Indicate that the request is aborted
            return None, TerminateReason.UNKNOWN

        output.response_ids = output.response_ids[: self.response_length]
        output.response_mask = output.response_mask[: self.response_length]
        if output.response_log_probs is not None:
            output.response_log_probs = output.response_log_probs[: self.response_length]
        if output.routed_experts is not None:
            output.routed_experts = output.routed_experts[: len(output.prompt_ids) + self.response_length]
        output.num_turns = 2

        output = await self.compute_reward_score(
            request["uid"],
            output,
            request.get("validate", False),
        )
        if output is None:
            # Request was aborted during reward computation (e.g. staleness check failed)
            return None, TerminateReason.ABORTED

        return output, TerminateReason.FINISHED
