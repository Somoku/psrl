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
