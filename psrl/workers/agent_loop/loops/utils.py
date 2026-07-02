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
    """Why an agent-loop trajectory stopped."""

    FINISHED = "finished"
    MAX_RESPONSE_LENGTH_EXCEEDED = "max_response_length_exceeded"
    MAX_TURNS_EXCEEDED = "max_turns_exceeded"
    ENV_TIMEOUT = "env_timeout"
    TRAJECTORY_TIMEOUT = "trajectory_timeout"
    ABORTED = "aborted"
    UNKNOWN = "unknown"
    ROLLOUT_ERROR = "rollout_error"

    @property
    def is_successful(self) -> bool:
        """Return whether the trajectory has usable training content."""
        return self in (
            TerminateReason.FINISHED,
            TerminateReason.MAX_TURNS_EXCEEDED,
            TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED,
        )

    @property
    def is_timeout(self) -> bool:
        """Return whether a timeout stopped the trajectory."""
        return self in (
            TerminateReason.ENV_TIMEOUT,
            TerminateReason.TRAJECTORY_TIMEOUT,
        )

    @property
    def is_error(self) -> bool:
        """Return whether the slot was wasted by a transient error."""
        return self in (TerminateReason.ROLLOUT_ERROR, TerminateReason.UNKNOWN)

    @property
    def is_aborted(self) -> bool:
        """Return whether PSRL intentionally aborted the trajectory."""
        return self is TerminateReason.ABORTED

    def needs_worker_retry(self) -> bool:
        """Return whether the worker should retry before manager recovery."""
        return self.is_timeout or self.is_error

    def needs_manager_retry(self) -> bool:
        """Return whether manager must refill a wasted buffer slot."""
        return self.is_error or self is TerminateReason.TRAJECTORY_TIMEOUT
