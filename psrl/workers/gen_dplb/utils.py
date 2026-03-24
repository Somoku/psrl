from typing import Any
from dataclasses import dataclass
from pydantic import BaseModel

# (worker_id, data_parallel_rank)
RolloutInstanceId = tuple[str, int]
INVALID_ROLLOUT_INSTANCE_ID: RolloutInstanceId = ("", -1)

# Default configuration constants
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_CONNECTIONS = 2000

@dataclass
class TokenInput:
    input_ids: list[int]
    """input token ids"""
    request_id: int
    """the unique request id"""
    prompt_id: int
    """the unique prompt id"""
    rollout_instance_id: RolloutInstanceId | None = None
    """the rollout instance id assigned for this generation"""
    version_tag: int = -1
    """the version tag of the model used for this generation"""
    cu_response_len: int | None = None
    """the current response length for this generation"""
    is_validate: bool = False
    """whether this request is for validation purpose"""

@dataclass
class TokenOutput:
    token_ids: list[int]
    """response token ids"""
    log_probs: list[float] | None = None
    """logprobs of response token ids"""
    routed_experts: Any | None = None
    """routed experts of response token ids"""
    stop_reason: str | None = None
    """stop reason: 'completed', 'aborted', or None for unknown"""
    num_preempted: int | None = None
    """number of preempted times for metric calculation"""
    interrupted: bool = False
    """whether the generation is interrupted"""
    interrupted_by_scheduler: bool = False
    """whether the generation is interrupted by scheduler (preempted)"""
    update_status: Any | None = None
    """status of the request"""
    rollout_instance_id: RolloutInstanceId | None = None
    """the rollout instance id used for this generation"""
