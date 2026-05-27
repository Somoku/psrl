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
    """Why an agent loop trajectory stopped.

    Values are grouped into four categories, each with a different downstream
    effect (see the helper methods below):

    1. Successful (``is_successful``): trajectory has usable content; emit it.
    2. Timeout (``is_timeout``): transient, may retry; trajectory possibly partial.
    3. Error (``is_error``): slot wasted; worker retries, then manager recovers.
    4. Aborted (``is_aborted``): intentional kill; drop silently, no retry.

    Worker / manager recovery decisions go through ``needs_worker_retry`` and
    ``needs_manager_retry`` so callers do not hardcode the membership tuples.
    """

    # --- Successful terminations: trajectory is usable for training. ---
    FINISHED = "finished"
    """Agent or environment signalled normal completion (e.g. ``done=True``)."""
    MAX_TURNS_EXCEEDED = "max_turns_exceeded"
    """Hit the per-trajectory turns budget before finishing."""
    MAX_RESPONSE_LENGTH_EXCEEDED = "max_response_length_exceeded"
    """Hit the prompt+response token budget; trajectory truncated but valid."""

    # --- Timeouts: transient; trajectory may be partial or empty. ---
    ENV_TIMEOUT = "env_timeout"
    """A single ``env.step()`` exceeded its ``step_timeout``."""
    TRAJECTORY_TIMEOUT = "trajectory_timeout"
    """Outer ``asyncio.wait_for(trajectory_timeout)`` cut off the whole run."""

    # --- Errors: slot wasted; needs both worker retry and manager recovery. ---
    ROLLOUT_ERROR = "rollout_error"
    """The rollout call propagated an exception (router / vLLM / network)."""
    UNKNOWN = "unknown"
    """Couldn't classify (e.g. zero turns produced, ``run()`` returned ``None``
    without setting ``ABORTED``). Treated as an error for safety."""

    # --- Intentional abort: slot already accounted for, no retry. ---
    ABORTED = "aborted"
    """Killed on purpose by PSRL (sibling group failure, staleness filter,
    proactive-filter, ...). The PS manager has already accounted for the
    slot, so we must NOT call ``notify_group_failed`` again."""

    # ----- helpers ---------------------------------------------------------

    @property
    def is_successful(self) -> bool:
        """Trajectory has usable content and should be emitted to the buffer."""
        return self in (
            TerminateReason.FINISHED,
            TerminateReason.MAX_TURNS_EXCEEDED,
            TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED,
        )

    @property
    def is_timeout(self) -> bool:
        """A timeout fired (env step or whole-trajectory wall-clock)."""
        return self in (
            TerminateReason.ENV_TIMEOUT,
            TerminateReason.TRAJECTORY_TIMEOUT,
        )

    @property
    def is_error(self) -> bool:
        """Slot was wasted by a transient failure (rollout error / unknown)."""
        return self in (TerminateReason.ROLLOUT_ERROR, TerminateReason.UNKNOWN)

    @property
    def is_aborted(self) -> bool:
        """Trajectory was killed on purpose by PSRL."""
        return self is TerminateReason.ABORTED

    def needs_worker_retry(self) -> bool:
        """Whether ``worker.py`` should retry this trajectory before giving up.

        Includes both timeouts and errors: each is plausibly transient and
        deserves the configured ``retry_limit`` budget. Successful and aborted
        terminations short-circuit out of the retry loop immediately.
        """
        return self.is_timeout or self.is_error

    def needs_manager_retry(self) -> bool:
        """Whether the manager must dispatch a replacement to refill the slot.

        True iff this termination wasted a buffer slot AND the manager is the
        only component that can refill it (the worker has already exhausted
        its own retry budget by the time this is checked).

        - ``is_error``: yes -- the slot contributed nothing.
        - ``TRAJECTORY_TIMEOUT``: yes -- with typical retry_limit=1 the
          worker exhausts its budget immediately without a real retry.
          The group will remain incomplete forever unless the manager
          dispatches a replacement.
        - ``is_aborted``: no -- the PS manager already accounted for the slot.
        - ``ENV_TIMEOUT``: no -- produces a partial trajectory that is still
          emitted (handled by the normal occupation flow).
        - ``is_successful``: no -- handled by the normal occupation flow.
        """
        return self.is_error or self is TerminateReason.TRAJECTORY_TIMEOUT


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
