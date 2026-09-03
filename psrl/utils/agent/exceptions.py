"""
Map a black-box agent harness's failures onto PSRL terminate reasons.

PSRL drives third-party harnesses (Harbor/terminus-2, mini-SWE-agent, MLGym) whose
failures arrive as opaque exceptions, type names, or status strings.
`TerminateReason` already encodes what the framework does about each outcome, so the
only per-harness variable is *recognising* which outcome a failure represents. That
is what a classifier does.

Getting the mapping wrong is not cosmetic. `TerminateReason.needs_manager_retry()`
makes the manager abort every sibling in a GRPO group and request a replacement
prompt, so filing a deliberate abort under `ROLLOUT_ERROR` makes the system treat
its own cleanup as a fresh fault and tear down healthy trajectories. The reasons
that matter most:

- `ABORTED`: PSRL/PSManager already ended this trajectory on purpose. No data, no
  retry. The manager owns the cleanup.
- `MAX_RESPONSE_LENGTH_EXCEEDED` / `MAX_TURNS_EXCEEDED`: a budget ran out, but the
  turns produced so far are valid on-policy data. Keep them.
- `ROLLOUT_ERROR`: an unexpected fault. The worker retries, then the group is
  refilled.

Subclass `AgentExceptionClassifier` per harness and put the subclass next to that
harness's runner (see `examples/sciaccel_rl/exceptions.py`). The base class handles
only what is harness-independent: PSRL's own sentinel exception types.
"""

from __future__ import annotations

from psrl.utils.common.http_utils import (
    PromptOverflowError,
    RequestAbortedByGatewayError,
)
from psrl.workers.agent_loop.loops.utils import TerminateReason

# vLLM's context-overflow wording. litellm's `ExceptionCheckers` and Harbor's
# `LiteLLMModel._is_context_length_error` both miss it, so anything that proxies a
# vLLM 400 through litellm has to match the text itself.
VLLM_OVERFLOW_MARKERS = (
    "maximum model length",
    "decoder prompt",
)


def is_prompt_overflow(exc: BaseException | str) -> bool:
    """
    Return whether `exc` is a vLLM context-window overflow (HTTP 400).

    Args:
        exc (BaseException | str): Exception or message text to inspect.

    Returns:
        bool: True when the text carries a vLLM overflow marker.
    """
    message = str(exc).lower()
    return any(marker in message for marker in VLLM_OVERFLOW_MARKERS)


class AgentExceptionClassifier:
    """
    Translate one harness's failures into `TerminateReason`.

    Subclasses declare `TYPE_MARKERS` and `MESSAGE_MARKERS` tables rather than
    reimplementing the scan. Override `classify` only when a harness exposes
    something neither table can express.

    Classification order is deliberate, most reliable signal first:

    1. PSRL sentinel exception *types*, which survive intact when the failure comes
       from our own HTTP layer.
    2. The harness's exception *type name*, available as a plain string when the
       harness reports failures as data (Harbor's `ExceptionInfo.exception_type`).
       Preferred over message text because wording drifts between versions.
    3. Message text, the last resort for harnesses that flatten everything into a
       string (litellm re-raising a vLLM 400 is the motivating case).
    """

    # Ordered (type-name substring, reason) pairs. Matched case-insensitively against
    # an exception's class name, so `"AgentTimeout"` matches `AgentTimeoutError`.
    TYPE_MARKERS: tuple[tuple[str, TerminateReason], ...] = ()

    # Ordered (message substring, reason) pairs. First match wins, so put the more
    # specific marker first when two could both match.
    MESSAGE_MARKERS: tuple[tuple[str, TerminateReason], ...] = ()

    def classify(
        self,
        exc: BaseException | str | None = None,
        *,
        exc_type: str | None = None,
    ) -> TerminateReason | None:
        """
        Classify a harness failure.

        Returns `None` rather than defaulting to `ROLLOUT_ERROR` so "unrecognised"
        stays distinguishable from "definitely a fault". A new harness's unknown
        failures then surface at the call site instead of being silently retried.

        Args:
            exc (BaseException | str | None): Exception object, or the message text
                a harness recorded in place of one. `None` means no failure.
            exc_type (str | None): Exception class name when the harness reports it
                separately from the message, as Harbor's `ExceptionInfo` does.

        Returns:
            TerminateReason | None: The matching reason, or `None` if unrecognised.
        """
        if exc is None and not exc_type:
            return None

        if isinstance(exc, BaseException):
            sentinel = self.classify_exception_type(exc)
            if sentinel is not None:
                return sentinel

        type_name = exc_type or (type(exc).__name__ if isinstance(exc, BaseException) else "")
        if type_name:
            by_type = self.classify_type_name(type_name)
            if by_type is not None:
                return by_type

        return self.classify_message(str(exc) if exc is not None else "")

    def classify_exception_type(self, exc: BaseException) -> TerminateReason | None:
        """
        Classify PSRL's own sentinel exception types.

        These are raised by `psrl.utils.common.http_utils` when SMG returns a known
        sentinel code, so they are exact and take precedence over any text match.

        Args:
            exc (BaseException): The exception to inspect.

        Returns:
            TerminateReason | None: Reason for a known sentinel, else `None`.
        """
        if isinstance(exc, RequestAbortedByGatewayError):
            return TerminateReason.ABORTED
        if isinstance(exc, PromptOverflowError):
            return TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED
        return None

    def classify_type_name(self, type_name: str) -> TerminateReason | None:
        """
        Classify by exception class name against `TYPE_MARKERS`.

        Args:
            type_name (str): Exception class name, e.g. `AgentTimeoutError`.

        Returns:
            TerminateReason | None: First matching reason, else `None`.
        """
        lowered = type_name.lower()
        for marker, reason in self.TYPE_MARKERS:
            if marker.lower() in lowered:
                return reason
        return None

    def classify_message(self, message: str) -> TerminateReason | None:
        """
        Classify by message text against `MESSAGE_MARKERS`.

        Args:
            message (str): Exception text.

        Returns:
            TerminateReason | None: First matching reason, else `None`.
        """
        if not message:
            return None
        lowered = message.lower()
        for marker, reason in self.MESSAGE_MARKERS:
            if marker.lower() in lowered:
                return reason
        if is_prompt_overflow(lowered):
            return TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED
        return None


__all__ = [
    "VLLM_OVERFLOW_MARKERS",
    "AgentExceptionClassifier",
    "is_prompt_overflow",
]
