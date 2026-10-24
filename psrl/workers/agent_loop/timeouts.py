"""The wall-clock budgets for one rollout.

Two numbers, not a ladder.

`trajectory_timeout` is the episode budget: wall clock the agent may spend working,
starting once the sandbox is provisioned. Sandbox admission and provisioning are
deliberately outside it — they are scheduling latency rather than work, so charging
them to the budget would make it fire on a healthy episode whenever the fleet is
busy. Those phases are bounded by the sandbox service's own liveness, not by a
deadline here.

`grading_timeout` is separate because grading runs after the episode and an operator
often wants a different answer for it, including none at all.
"""

from __future__ import annotations

from dataclasses import dataclass

# Episode seconds of agent work used when the configuration leaves the budget unset.
DEFAULT_TRAJECTORY_TIMEOUT_S = 7200.0

# Margin over agent work and grading that a sandbox is allowed to exist. It covers
# the parts of a sandbox's life that are not agent or grading time: provisioning
# (a cold image pull), teardown, and the slack between two clocks. It is the one
# internal constant that replaces the old "2x/3x/4x episode" ratio chain.
LIFETIME_SLACK_S = 3600.0


@dataclass(frozen=True)
class RolloutTimeouts:
    """How long the two halves of a rollout may run.

    `trajectory_timeout_s` bounds the agent's work; the episode clock starts once the
    sandbox is provisioned. `grading_timeout_s` bounds grading, which runs after the
    episode; `None` means grading is not timed.
    """

    trajectory_timeout_s: float
    grading_timeout_s: float | None = None

    def __post_init__(self) -> None:
        if self.trajectory_timeout_s <= 0:
            raise ValueError(
                f"rollout.agent.trajectory_timeout must be greater than zero, got "
                f"{self.trajectory_timeout_s!r}. A rollout with no bound cannot be told apart "
                "from a wedged one."
            )
        if self.grading_timeout_s is not None and self.grading_timeout_s <= 0:
            raise ValueError(
                f"rollout.agent.grading_timeout must be greater than zero or null (unbounded), "
                f"got {self.grading_timeout_s!r}."
            )

    def describe(self) -> str:
        """Return a one-line summary of the two budgets for the startup log."""
        grading = "unbounded" if self.grading_timeout_s is None else f"{self.grading_timeout_s:.0f}s"
        return f"trajectory={self.trajectory_timeout_s:.0f}s, grading={grading}"

    @property
    def sandbox_lifetime_s(self) -> float:
        """Return how long a sandbox for this rollout may exist.

        A node reclaims a sandbox by its lifetime, so it has to cover everything the
        sandbox is for: the agent's work, the grading that follows it, and a margin
        for provisioning and teardown. Stating it here lets the sandbox service
        reclaim a sandbox from a caller that died, without a second deadline for the
        same thing.
        """
        return self.trajectory_timeout_s + (self.grading_timeout_s or 0.0) + LIFETIME_SLACK_S


def resolve_rollout_timeouts(
    trajectory_timeout_s: float | None,
    grading_timeout_s: float | None,
) -> RolloutTimeouts:
    """Resolve the two budgets from configuration.

    Args:
        trajectory_timeout_s (float | None): The episode budget. ``None`` selects the
            default rather than disabling enforcement, because a rollout with no bound
            at all cannot be distinguished from a wedged one.
        grading_timeout_s (float | None): The grading budget. ``None`` means grading is
            not timed.
    """
    trajectory = (
        DEFAULT_TRAJECTORY_TIMEOUT_S if trajectory_timeout_s is None else float(trajectory_timeout_s)
    )
    grading = None if grading_timeout_s is None else float(grading_timeout_s)
    return RolloutTimeouts(trajectory_timeout_s=trajectory, grading_timeout_s=grading)


def resolve_from_config(config) -> RolloutTimeouts:
    """Read the two budgets from a trainer config tree.

    The only place these configuration paths are read, so a key rename cannot leave the
    loop and the trainer reporting different budgets.
    """
    agent_config = config.gen_actor_rollout_ref.rollout.agent
    return resolve_rollout_timeouts(
        agent_config.get("trajectory_timeout"),
        agent_config.get("grading_timeout"),
    )
