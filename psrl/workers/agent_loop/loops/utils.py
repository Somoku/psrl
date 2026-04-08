from enum import Enum

from omegaconf import DictConfig

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


class DictConfigWrap:
    """Wrapper for DictConfig to avoid hydra.utils.instantiate recursive resolve."""

    def __init__(self, config: DictConfig):
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
