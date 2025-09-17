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
        # TODO(linsh): add profiling
        # with simple_timer("generate_sequences"):
        output = self.rollout_router.generate(request)
        if output is not None:
            batch_size = len(output)
            response_mask_list = []
            num_turns_list = []
            for i in range(batch_size):
                response_ids = output.non_tensor_batch["raw_response_ids"][i]
                response_mask = [1] * len(response_ids)
                response_mask_list.append(response_mask[: self.response_length])
                num_turns_list.append(0)
            # response mask: bsz * [1, 1, ..., 1] (the num of 1 is the max of response_length and actual length of response_ids)
            output.non_tensor_batch["response_mask"] = np.fromiter(response_mask_list, dtype=object)
            output.non_tensor_batch["__num_turns__"] = np.array(num_turns_list)

        return output
