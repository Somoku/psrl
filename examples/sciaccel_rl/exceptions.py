"""
Harbor/terminus-2 failure classification for SciAccel-RL.

Harbor reports a trial failure as data (`TrialResult.exception_info`) rather than by
raising, and the message has already passed through litellm, which discards both the
original exception class and SMG's `x-smg-error-code` header. So the markers below
are the only signal left. Each one is anchored to where it comes from.

Two Harbor failures look alike but are not: `AgentTimeoutError` means the agent used
up its budget, which is a normal end with usable turns, while
`AgentSetupTimeoutError` means the container never came up, so there is nothing to
train on. Collapsing them loses that distinction.
"""

from __future__ import annotations

from psrl.utils.agent.exceptions import AgentExceptionClassifier
from psrl.workers.agent_loop.loops.utils import TerminateReason


class HarborExceptionClassifier(AgentExceptionClassifier):
    """
    Map Harbor/terminus-2 failures onto PSRL terminate reasons.

    Used by `SciAccelAgentLoop` and by `examples/sciaccel_rl/eval` so training and
    evaluation classify identically.
    """

    # Harbor's own exception classes, from `harbor/trial/errors.py`. All four subclass
    # `asyncio.TimeoutError`, so matching on the class name is the only way to tell
    # them apart. Order matters: `AgentSetupTimeout` must precede `AgentTimeout`
    # because the latter is a substring of the former.
    TYPE_MARKERS = (
        # The container or agent never started, so no turn was ever produced.
        ("AgentSetupTimeout", TerminateReason.ROLLOUT_ERROR),
        ("EnvironmentStartTimeout", TerminateReason.ROLLOUT_ERROR),
        ("SandboxBuildFailed", TerminateReason.ROLLOUT_ERROR),
        # The agent ran and used up its externally enforced wall-clock budget. The
        # turns it produced are valid on-policy data, so this is a truncation.
        ("AgentTimeout", TerminateReason.AGENT_TIMEOUT),
        # Grading timed out after the agent finished. The trajectory is still good, it
        # just scores 0 because no verifier reward arrived.
        ("VerifierTimeout", TerminateReason.VERIFIER_ERROR),
    )

    # Message markers, for failures litellm flattened into a bare `BadRequestError`.
    MESSAGE_MARKERS = (
        # SMG's `request_aborted` sentinel body, from
        # third_party/smg/model_gateway/src/routers/grpc/routing_loop/runtime.rs.
        # PSManager aborts a group's siblings when one member fails, and it clears the
        # buffer entry itself, so no further group recovery should be requested here.
        ("Request aborted by PS Manager", TerminateReason.ABORTED),
        # Harbor's own wording for the externally enforced agent budget, which arrives
        # as text when `exception_type` is unavailable.
        ("Agent execution timed out", TerminateReason.AGENT_TIMEOUT),
        # An in-container command exceeded its own timeout during setup. Observed on a
        # session that produced zero turns, so it is infrastructure, not policy.
        ("Command timed out", TerminateReason.ROLLOUT_ERROR),
        # `Job.run` returned no trial at all, so nothing ran.
        ("no_trials", TerminateReason.ROLLOUT_ERROR),
    )
    # A vLLM context overflow relayed by litellm is handled by the base class, which
    # falls back to `is_prompt_overflow` and returns MAX_RESPONSE_LENGTH_EXCEEDED.


__all__ = ["HarborExceptionClassifier"]
