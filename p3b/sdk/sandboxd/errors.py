"""Failures, typed by attribution rather than by where they happened.

The caller's response differs per type, and an infrastructure fault recorded as
a model or harness failure corrupts a reward. So a sandbox that was never
created, one the kernel killed, and one whose command merely ran out of time are
three different exceptions and must stay that way.
"""

from __future__ import annotations


class SandboxError(RuntimeError):
    """Base for every sandbox failure."""


class SandboxCapacityTimeout(SandboxError):
    """Admission never granted a sandbox in time.

    No sandbox existed, so no episode ran: this is capacity planning rather than
    a task, model, or harness failure. Reported as its own type so a caller can
    classify it apart from a cancelled or failed episode.
    """


class SandboxCapabilityError(SandboxError):
    """No configured backend can serve the spec.

    Admission fails fast rather than losing a guarantee quietly: a caller that
    needs a full-state resume must not be served a filesystem resume and told it
    is the same thing.
    """


class SandboxSessionLost(SandboxError):
    """The sandbox stopped for a reason nobody asked for.

    The work in it is unrecoverable, so the rollout must be replaced -- but it
    must not be counted as evidence that the harness is broken.
    """

    def __init__(self, message: str, *, sandbox_id: str = "", exit_reason: str = "") -> None:
        super().__init__(message)
        self.sandbox_id = sandbox_id
        self.exit_reason = exit_reason


class SandboxOomError(SandboxError):
    """The kernel killed the sandbox for memory.

    A resource fault rather than a model or task failure, and distinct from a
    session lost for any other reason: an operator raises the request for this
    one and investigates the node for the other.
    """


class SandboxCommandTimeout(SandboxError, TimeoutError):
    """A command exceeded its deadline.

    `sandbox_preserved` says whether the sandbox survived. A backend that can
    stop the overrunning process keeps the sandbox and its filesystem, so the
    caller continues the episode with a fresh shell. One that cannot must
    destroy it, because a process left running would corrupt the next command's
    output. The two need different responses, so they must not arrive the same.
    """

    def __init__(self, message: str, *, sandbox_preserved: bool = False) -> None:
        super().__init__(message)
        self.sandbox_preserved = sandbox_preserved


class SandboxSetupError(SandboxError):
    """A task's preparation step failed inside its sandbox.

    The environment is not what the task needs, so a captured version of it
    would be wrong for every later reuse. A task preparation fault rather than a
    sandbox or model one.
    """


class SandboxTransportError(SandboxError):
    """The control channel failed rather than the command.

    A lost transport says nothing about the workload, so a caller must not read
    it as a task result.
    """
