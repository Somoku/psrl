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
    # Wall-clock exhaustion is distinct from the configured turn cap in metrics.
    AGENT_TIMEOUT = "agent_timeout"
    # The agent finished normally but grading failed or timed out, so the trajectory is
    # complete and trainable while carrying no verifier reward.
    VERIFIER_ERROR = "verifier_error"
    # An environment step exceeded `agent.env.step_timeout`. Both call sites build a
    # trajectory from the turns completed before the timeout, so this keeps its data.
    ENV_TIMEOUT = "env_timeout"
    TRAJECTORY_TIMEOUT = "trajectory_timeout"
    # A call inside the episode timed out. Kept apart from `TRAJECTORY_TIMEOUT`, which
    # means the configured episode budget expired, because this is an infrastructure fault.
    DOWNSTREAM_TIMEOUT = "downstream_timeout"
    # The episode never ran: node capacity admission never granted its sandbox within the
    # configured deadline. Distinct from every other failure because no task code executed,
    # so it says nothing about the model or the harness.
    SANDBOX_CAPACITY_TIMEOUT = "sandbox_capacity_timeout"
    # The episode ran and its trajectory is trainable, but the grader sandbox was never
    # admitted, so no verifier ever scored it. Kept separate from `VERIFIER_ERROR` so the
    # capacity breaker can count the fault, and from `SANDBOX_CAPACITY_TIMEOUT` because the
    # episode itself completed and its data is kept.
    GRADER_CAPACITY_TIMEOUT = "grader_capacity_timeout"
    # The episode's sandbox stopped on its own while a command was running, so the work in
    # it was lost. A lifecycle or host fault rather than evidence about the model or the
    # harness: the group must be replaced, but it must not be counted against the harness.
    CONTAINER_LOST = "container_lost"
    # The episode was cancelled from outside, typically while the run was shutting down.
    # Kept apart from `ROLLOUT_ERROR` so teardown cannot be mistaken for a rollout fault.
    ROLLOUT_CANCELLED = "rollout_cancelled"
    # The child never began its episode: admission and provisioning consumed the whole setup
    # allowance. Distinct from `SANDBOX_CAPACITY_TIMEOUT`, which is admission alone, and from
    # `TRAJECTORY_TIMEOUT`, which means the episode itself ran out of time.
    ROLLOUT_DEADLINE_EXCEEDED = "rollout_deadline_exceeded"
    ABORTED = "aborted"
    UNKNOWN = "unknown"
    ROLLOUT_ERROR = "rollout_error"

    @property
    def is_successful(self) -> bool:
        """Return whether the trajectory has usable training content.

        Every member here is a *budget* that ran out (context window, turn cap, wall
        clock, env step) or a post-run grading failure. In each case the turns produced
        before the limit are valid on-policy data, so the trajectory is truncated and
        trained rather than discarded.
        """
        return self in (
            TerminateReason.FINISHED,
            TerminateReason.MAX_TURNS_EXCEEDED,
            TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED,
            TerminateReason.AGENT_TIMEOUT,
            TerminateReason.VERIFIER_ERROR,
            TerminateReason.GRADER_CAPACITY_TIMEOUT,
            TerminateReason.ENV_TIMEOUT,
        )

    @property
    def is_ungraded(self) -> bool:
        """Return whether the episode ran but never received a verifier score.

        The trajectory is real on-policy data, so `is_successful` keeps it, but no
        grader ever looked at it. A reward of 0 here is the absence of a measurement,
        not a measured failure, and nothing downstream can tell the two apart once the
        empty reward dict has defaulted to 0.0.

        Training it as a zero is worse than dropping it: siblings of the same task in
        the same GRPO group split 0.0 against 1.0 purely on whether a container timed
        out, which is a gradient of pure infrastructure noise pointing in a direction
        the policy cannot influence.

        `GRADER_CAPACITY_TIMEOUT` is ungraded for the same reason: the grader sandbox was
        never admitted, so its zero would also be the absence of a measurement.
        """
        return self in (TerminateReason.VERIFIER_ERROR, TerminateReason.GRADER_CAPACITY_TIMEOUT)

    @property
    def is_budget_truncated(self) -> bool:
        """Return whether a harness budget cut the episode off mid-work.

        These trajectories are still valid on-policy data, so `is_successful` keeps
        them, but their reward reports the cutoff rather than the quality of the
        model's choices. Grading a run that was never allowed to finish as a failure
        makes the group-relative advantage penalise every token in it, and under
        `token-mean` a long truncated trajectory outweighs many short ones. The
        cheapest way for the policy to shed that penalty is to emit fewer tokens per
        turn, which spends the turn cap faster and truncates more often.

        Measured over 1983 episodes of GRPO-sciaccel-Qwen35-4B-v2_repair-L1: tokens
        per turn fell 1125 to 327, `max_turns_exceeded` rose from 29% to 39%, and the
        score collapsed from 0.573 at step 11 to 0.078 at step 16.

        `AGENT_TIMEOUT` and `ENV_TIMEOUT` are excluded deliberately, because they are
        infrastructure faults rather than budget exhaustion. `VERIFIER_ERROR` is
        excluded too and handled by `is_ungraded`, which masks it for a different
        reason: its tokens are honest, but its reward was never measured.
        """
        return self in (
            TerminateReason.MAX_TURNS_EXCEEDED,
            TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED,
        )

    @property
    def is_timeout(self) -> bool:
        """Return whether the configured episode budget stopped the trajectory.

        Only `TRAJECTORY_TIMEOUT` qualifies, because it is raised by the episode-level
        `wait_for`, which owns no partial output. `DOWNSTREAM_TIMEOUT` is an
        infrastructure fault rather than a budget, so it reports through `is_error`.
        `AGENT_TIMEOUT` and `ENV_TIMEOUT` return a finalized trajectory, and this
        property feeds `needs_worker_retry`, where re-running an episode whose turns
        were already accepted would duplicate them.
        """
        return self is TerminateReason.TRAJECTORY_TIMEOUT

    @property
    def is_error(self) -> bool:
        """Return whether the slot was wasted by a transient error."""
        return self in (
            TerminateReason.ROLLOUT_ERROR,
            TerminateReason.DOWNSTREAM_TIMEOUT,
            TerminateReason.SANDBOX_CAPACITY_TIMEOUT,
            TerminateReason.CONTAINER_LOST,
            TerminateReason.ROLLOUT_CANCELLED,
            TerminateReason.ROLLOUT_DEADLINE_EXCEEDED,
            TerminateReason.UNKNOWN,
        )

    @property
    def is_coordination_fault(self) -> bool:
        """Return whether scheduling or teardown ended the episode, not the rollout.

        `SANDBOX_CAPACITY_TIMEOUT` means the sandbox was never admitted, so no episode ran
        at all; `GRADER_CAPACITY_TIMEOUT` means grading never ran, so the reward was never
        measured; `CONTAINER_LOST` means the sandbox stopped out from under a running
        command; `ROLLOUT_CANCELLED` means something outside the episode stopped it, usually
        shutdown. None is evidence about the model, the harness, or the task, so none may be
        retried in place nor counted as a rollout failure.

        `ROLLOUT_DEADLINE_EXCEEDED` is deliberately not one of them: provisioning beyond the
        whole setup allowance says the environment could not deliver a sandbox on time, which
        is the same class of signal as a broken harness and belongs in the breaker's count.
        """
        return self in (
            TerminateReason.SANDBOX_CAPACITY_TIMEOUT,
            TerminateReason.GRADER_CAPACITY_TIMEOUT,
            TerminateReason.CONTAINER_LOST,
            TerminateReason.ROLLOUT_CANCELLED,
        )

    @property
    def is_aborted(self) -> bool:
        """Return whether PSRL intentionally aborted the trajectory."""
        return self is TerminateReason.ABORTED

    def needs_worker_retry(self) -> bool:
        """Return whether the worker should re-run the episode in place.

        Only reasons that produced no usable data qualify, since `worker.py` nulls the
        output for anything this returns True for. Retrying a data-bearing reason would
        both discard the trajectory and duplicate the work.

        Coordination faults and a rollout that never started are excluded: an in-place
        retry cannot create node capacity and cannot make provisioning faster, so it would
        only spend the retry budget and re-enter the same wait.

        Note this is inert at the default `rollout.agent.retry_limit=1`, which yields a
        single attempt.
        """
        if self.is_coordination_fault or self is TerminateReason.ROLLOUT_DEADLINE_EXCEEDED:
            return False
        return self.is_timeout or self.is_error

    def needs_manager_retry(self) -> bool:
        """Return whether manager must refill a wasted buffer slot.

        A train buffer entry is all-or-nothing: `AgentLoopManager` occupies it only once
        all `alg_rollout_n` trajectories arrive, and `PSManager.abort_requests` clears the
        whole entry when fewer remain. So every reason that reaches the worker without
        usable data must refill the group, or the surviving siblings wait forever.

        `ABORTED` is the sole exception: PSManager raised it after already clearing the
        entry, so requesting another refill would double-count the failure.
        """
        return not self.is_successful and not self.is_aborted

    def counts_toward_refill_breaker(self) -> bool:
        """Return whether this failure is evidence that the run cannot make progress.

        The refill breaker exists to stop a run whose harness or environment fails
        deterministically: it counts groups that lost their slot and asks whether any
        group succeeded in between. Coordination faults are the one exclusion, because
        they record a scheduling or shutdown problem rather than a failing rollout, and
        they arrive in bursts that would otherwise fill the streak on their own. Counting
        them turns a recoverable stall into a fatal abort, and they are reported through
        their own counters instead.
        """
        return self.needs_manager_retry() and not self.is_coordination_fault
