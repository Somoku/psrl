import os
import logging
import numpy as np

import ray

from verl import DataProto
from verl.utils.profiler import simple_timer

from psrl.workers.agent_loop.loops.utils import register
from psrl.workers.agent_loop.loops.base_agent_loop import AgentLoopBase

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

@register("generate_only_agent")
class GenerateAgentLoop(AgentLoopBase):
    """Agent loop that performs single-request generation in streaming mode."""

    def __init__(self, *args, **kwargs):
        """Initialize the generation agent loop."""
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.gen_actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.gen_actor_rollout_ref.rollout.response_length

    async def run(self, request: DataProto) -> DataProto:
        """Execute generation for a single request.
        
        Args:
            request (DataProto): Single input request.
            
        Returns:
            DataProto: Generated response with metadata.
        """
        # TODO(linsh): add profiling
        # with simple_timer("generate_sequences"):
        output = await self.rollout_router.generate_async(request)
        if output is not None:
            response_ids = output.non_tensor_batch["raw_response_ids"][0]
            response_mask = [1] * len(response_ids)
            output.non_tensor_batch["response_mask"] = np.array([response_mask[: self.response_length]])
            output.non_tensor_batch["__num_turns__"] = np.array([0])

        return output
