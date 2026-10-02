"""Derive rollout deadlines from one episode budget."""

from __future__ import annotations

from dataclasses import dataclass

# The budget covers agent turns, patch collection, and grading after provisioning.
DEFAULT_EPISODE_TIMEOUT_S = 7200.0
# Setup includes image pulls, container creation, snapshots, and harness preparation.
SETUP_ALLOWANCE_S = 900.0
# Allow the episode deadline to stop the harness before the exec backstop fires.
EXEC_BACKSTOP_S = 60.0


@dataclass(frozen=True)
class AgentLoopTimeouts:
    """Every deadline that bounds a rollout, derived from the episode budget.

    The rollout owns every deadline here. Nothing downstream — not the staleness
    inventory, not the agent loop manager — may decide on its own schedule that a
    dispatched rollout is finished: a coordination layer guessing at a shorter
    deadline than the one the work is actually held to just kills healthy episodes.
    """

    episode_timeout_s: float
    setup_allowance_s: float
    admission_timeout_s: float

    def __post_init__(self) -> None:
        for name in ("episode_timeout_s", "setup_allowance_s"):
            if getattr(self, name) <= 0:
                raise ValueError(f"Agent loop timeout {name} must be greater than zero.")

    @property
    def harness_exec_timeout_s(self) -> float:
        """Wall clock the harness CLI itself is allowed before it is force-stopped."""
        return self.episode_timeout_s + EXEC_BACKSTOP_S

    @property
    def setup_timeout_s(self) -> float:
        """Wall clock a child may spend being admitted and provisioned."""
        return self.setup_allowance_s + max(self.admission_timeout_s, 0.0)

    @property
    def child_deadline_s(self) -> float:
        """Wall clock a child may spend in total, across every phase it owns."""
        return self.setup_timeout_s + self.harness_exec_timeout_s

    def describe(self) -> str:
        """Return a one-line summary of the ladder for the startup log."""
        return (
            f"episode={self.episode_timeout_s:.0f}s, setup<={self.setup_timeout_s:.0f}s "
            f"(admission<={self.admission_timeout_s:.0f}s), child<={self.child_deadline_s:.0f}s, "
            f"harness_exec<={self.harness_exec_timeout_s:.0f}s"
        )


def resolve_agent_loop_timeouts(
    trajectory_timeout_s: float | None,
    admission_timeout_s: float | None,
) -> AgentLoopTimeouts:
    """Derive the ladder from the episode budget and the admission deadline.

    Args:
        trajectory_timeout_s (float | None): The configured episode budget. ``None``
            selects the default rather than disabling enforcement, because a rollout with
            no bound at all cannot be distinguished from a wedged one.
        admission_timeout_s (float | None): The capacity admission deadline, or ``None``
            when node capacity admission is not in use.

    Raises:
        ValueError: When the admission deadline contradicts the episode it precedes.
    """
    episode = DEFAULT_EPISODE_TIMEOUT_S if trajectory_timeout_s is None else float(trajectory_timeout_s)
    if episode <= 0:
        raise ValueError(
            f"rollout.agent.trajectory_timeout must be greater than zero, got {episode!r}. "
            "Set the episode budget explicitly; a rollout with no bound cannot be told apart from a "
            "wedged one."
        )
    admission = 0.0 if admission_timeout_s is None else float(admission_timeout_s)
    ladder = AgentLoopTimeouts(
        episode_timeout_s=episode,
        setup_allowance_s=SETUP_ALLOWANCE_S,
        admission_timeout_s=admission,
    )
    validate_agent_loop_timeouts(ladder)
    return ladder


def validate_agent_loop_timeouts(ladder: AgentLoopTimeouts) -> None:
    """Refuse a ladder whose admission deadline outlasts the episode it precedes.

    The rest of the ladder cannot disagree with itself: every other deadline is derived
    from the episode budget, and the value object refuses a non-positive allowance, so the
    ordering is a property of the construction rather than a setting. The admission
    deadline is the one number that comes from configuration, and the one that can
    silently invert the ladder. A queued rollout allowed to wait past its own episode
    budget is starved by definition: the queue, not the work, has become the run's
    dominant cost, and the run discovers it only as a slow trickle of capacity faults.

    That makes this a capacity planning fault, so it is refused here rather than papered
    over by raising the number past the work it precedes: a node that cannot admit inside
    one episode cannot admit inside two either.

    Raises:
        ValueError: When the admission deadline is at or beyond the episode budget.
    """
    if ladder.admission_timeout_s >= ladder.episode_timeout_s:
        raise ValueError(
            f"Sandbox admission may wait {ladder.admission_timeout_s:g}s while the episode budget is only "
            f"{ladder.episode_timeout_s:g}s, so a queued rollout would spend longer waiting for a sandbox "
            "than working in one. Give the node more admittable capacity (raise the envelope, lower "
            "capacity.utilization, rebalance capacity.classes, or shrink the per-sandbox reservation) "
            "instead of raising this deadline, or lower trajectory_timeout if episodes really are that short."
        )


def resolve_from_config(config) -> AgentLoopTimeouts:
    """Read and derive the ladder from a trainer config tree.

    The only place these configuration paths are read, so a key rename cannot leave the
    loops, the worker, and the manager enforcing different ladders.
    """
    agent_config = config.gen_actor_rollout_ref.rollout.agent
    return resolve_agent_loop_timeouts(
        agent_config.get("trajectory_timeout"),
        agent_config.sandbox.capacity.get("acquire_timeout_s"),
    )
