from enum import Enum

from omegaconf import DictConfig
from pydantic import BaseModel

AGENT_LOOP_REGISTRY: dict[str, dict] = {}


def register(agent_name: str):
    """Register an agent loop class with the given name.

    Args:
        agent_name (str): Name to register the agent loop under.

    Returns:
        function: Decorator function for registering the agent loop class.
    """
    from psrl.workers.agent_loop.loops.base_agent_loop import AgentLoopBase

    def decorator(subclass: type[AgentLoopBase]) -> type[AgentLoopBase]:
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        AGENT_LOOP_REGISTRY[agent_name] = {"_target_": fqdn}
        return subclass

    return decorator


class DummyConfig:
    """Wrapper class to make hydra.utils.instantiate compatible with configuration objects.

    This class wraps the configuration to provide the expected interface for Hydra instantiation.
    """

    def __init__(self, config: DictConfig) -> None:
        self.config = config


class TerminateReason(Enum):
    FINISHED = "finished"
    MAX_RESPONSE_LENGTH_EXCEEDED = "max_response_length_exceeded"
    MAX_TURNS_EXCEEDED = "max_turns_exceeded"
    ENV_TIMEOUT = "env_timeout"
    TIMEOUT = "timeout"
    ABORTED = "aborted"
    UNKNOWN = "unknown"
    ERROR = "error"

    def needs_manager_retry(self) -> bool:
        """Return True iff this termination reason represents a transient error
        that wasted a buffer slot and should trigger manager-level recovery.

        Transient errors (ERROR / UNKNOWN) mean the agent crashed or produced
        zero turns: the slot contributed nothing and must be replaced.

        Intentional terminations do NOT need recovery:
        - ABORTED: killed by staleness / redundant / proactive-filter; the PS
          manager already accounts for the slot.
        - All others (FINISHED, MAX_TURNS_EXCEEDED, etc.): have a valid
          trajectory and are handled by the normal occupation flow.
        """
        return self in (TerminateReason.ERROR, TerminateReason.UNKNOWN)


class AgentLoopMetrics(BaseModel):
    """Agent loop performance metrics."""

    generate_sequences: float = 0.0
    """Time spent on sequence generation in seconds."""
    tool_calls: float = 0.0
    """Time spent on tool calls in seconds."""


class AgentLoopOutput(BaseModel):
    """Output data structure from agent loop execution."""

    prompt_ids: list[int]
    """Prompt token ids."""
    response_ids: list[int]
    """Response token ids including LLM generated token, tool response token."""
    response_mask: list[int]
    """Response mask, 1 for LLM generated token, 0 for tool response token."""
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""
    metrics: AgentLoopMetrics
    """Auxiliary performance metrics"""
