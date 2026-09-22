"""The single ladder of deadlines that bounds one rollout episode.

A rollout used to be measured against several independently configured clocks: the
harness enforced its own two-hour cap, the manager declared an entry dead after one hour
of silence, and the framework's episode budget was off entirely. Those numbers could
contradict each other, and they did: an entry whose children were running normally was
abandoned and refilled at the one-hour mark, while the harness was still willing to work
for another hour. Every number below is now derived from the one the user sets, so no two
clocks can disagree, and the derived ones are reported together at startup.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

psrl_logger = logging.getLogger(__file__)

# The episode budget covers everything a child does after provisioning: the agent's own
# turns, patch collection, and grading. The allowances below cover the phases outside that
# window, so a child is never bounded twice by two different numbers.
DEFAULT_EPISODE_TIMEOUT_S = 7200.0
# Admission plus container creation, snapshot, and harness preparation. Generous because a
# cold image pull on a busy node is slow, and because the admission deadline already bounds
# the part of it that can queue.
SETUP_ALLOWANCE_S = 900.0
# The harness process is stopped by the episode budget, which can cancel it and report why.
# Its own exec timeout only has to outlive that, so it acts as a backstop rather than as a
# second, silently different limit.
EXEC_BACKSTOP_S = 60.0
# A heartbeat is a liveness signal, not a deadline: the watchdog compares its age against
# the stall threshold. One per minute is far below any meaningful episode and costs one
# cheap RPC per running episode.
MIN_HEARTBEAT_S = 30.0
MAX_HEARTBEAT_S = 300.0
HEARTBEAT_DIVISOR = 60.0
# How many missed heartbeats mean the episode is gone rather than slow. Independent of the
# episode budget: with a heartbeat signal, silence is the only thing that cannot be
# explained by an episode that is simply long.
STALL_HEARTBEAT_MULTIPLIER = 3.0
MIN_STALL_TIMEOUT_S = 300.0


@dataclass(frozen=True)
class AgentLoopTimeouts:
    """Every deadline that bounds a rollout, derived from the episode budget."""

    episode_timeout_s: float
    setup_allowance_s: float
    admission_timeout_s: float
    heartbeat_interval_s: float
    entry_stall_timeout_s: float

    def __post_init__(self) -> None:
        for name in (
            "episode_timeout_s",
            "setup_allowance_s",
            "heartbeat_interval_s",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"Agent loop timeout {name} must be greater than zero.")
        if self.entry_stall_timeout_s < 0:
            raise ValueError("Agent loop timeout entry_stall_timeout_s cannot be negative.")

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
            f"harness_exec<={self.harness_exec_timeout_s:.0f}s, heartbeat={self.heartbeat_interval_s:.0f}s, "
            f"stall={self.entry_stall_timeout_s:.0f}s"
        )


def derive_heartbeat_interval_s(episode_timeout_s: float) -> float:
    """Return the liveness period for an episode budget, clamped to a sane range."""
    return max(MIN_HEARTBEAT_S, min(MAX_HEARTBEAT_S, episode_timeout_s / HEARTBEAT_DIVISOR))


def resolve_agent_loop_timeouts(
    trajectory_timeout_s: float | None,
    admission_timeout_s: float | None,
    *,
    entry_stall_timeout_s: float | None = None,
) -> AgentLoopTimeouts:
    """Derive the ladder, honouring explicit overrides only where they make sense.

    Args:
        trajectory_timeout_s (float | None): The configured episode budget. ``None``
            selects the default rather than disabling enforcement, because a rollout with
            no bound at all cannot be distinguished from a wedged one.
        admission_timeout_s (float | None): The capacity admission deadline, or ``None``
            when node capacity admission is not in use.
        entry_stall_timeout_s (float | None): Optional override of the derived stall
            threshold. Only raise it; the derived value is what makes a silent entry
            recoverable in minutes. Set it to zero to disable the watchdog entirely, which
            accepts that a wedged worker blocks its buffer until the run is stopped.

    Raises:
        ValueError: When an override contradicts the ladder it is part of.
    """
    episode = DEFAULT_EPISODE_TIMEOUT_S if trajectory_timeout_s is None else float(trajectory_timeout_s)
    if episode <= 0:
        raise ValueError(
            f"rollout.agent.trajectory_timeout must be greater than zero, got {episode!r}. "
            "Set the episode budget explicitly; a rollout with no bound cannot be told apart from a "
            "wedged one."
        )
    admission = 0.0 if admission_timeout_s is None else float(admission_timeout_s)
    heartbeat = derive_heartbeat_interval_s(episode)
    if entry_stall_timeout_s is not None and entry_stall_timeout_s <= 0:
        # Explicitly disabling the watchdog accepts that a wedged worker blocks its buffer
        # until the run is stopped, which is sometimes what an experiment wants.
        stall = 0.0
    else:
        derived_stall = max(MIN_STALL_TIMEOUT_S, STALL_HEARTBEAT_MULTIPLIER * heartbeat)
        stall = derived_stall if entry_stall_timeout_s is None else float(entry_stall_timeout_s)
        if stall < heartbeat:
            raise ValueError(
                f"psrl.agentic_rl.entry_stall_timeout_s={stall:g} is shorter than the "
                f"{heartbeat:g}s liveness heartbeat, so every running entry would look stalled. "
                "Leave it null to derive the threshold."
            )
    ladder = AgentLoopTimeouts(
        episode_timeout_s=episode,
        setup_allowance_s=SETUP_ALLOWANCE_S,
        admission_timeout_s=admission,
        heartbeat_interval_s=heartbeat,
        entry_stall_timeout_s=stall,
    )
    if admission > episode:
        # Legal, but it means a queued rollout spends longer waiting for a sandbox than it
        # spends working in one, which is almost never what the operator intended.
        psrl_logger.warning(
            f"Sandbox admission may wait {admission:g}s while the episode budget is only {episode:g}s, "
            "so a rollout can spend longer queueing than working. Raise trajectory_timeout or lower "
            "capacity.acquire_timeout_s if that is not intended."
        )
    return ladder


def resolve_from_config(config) -> AgentLoopTimeouts:
    """Read and derive the ladder from a trainer config tree.

    The only place the three configuration paths are read, so a key rename cannot leave the
    loops, the worker, and the manager enforcing different ladders.
    """
    agent_config = config.gen_actor_rollout_ref.rollout.agent
    return resolve_agent_loop_timeouts(
        agent_config.get("trajectory_timeout"),
        agent_config.sandbox.capacity.get("acquire_timeout_s"),
        entry_stall_timeout_s=config.psrl.agentic_rl.get("entry_stall_timeout_s"),
    )


def validate_lease_max_age_s(lease_max_age_s: float | None, ladder: AgentLoopTimeouts) -> None:
    """Refuse a capacity lease age cap that would reclaim a healthy sandbox.

    The age cap is the last-resort recovery for a lease whose release was missed, so it has
    to outlive the longest sandbox a healthy run can hold. Nothing else in the design checks
    this: the lease is renewed by its live owner, so the cap firing is the only feedback, and
    by then it has already taken capacity from a running sandbox.

    Args:
        lease_max_age_s (float | None): The configured cap, or ``None`` when disabled.
        ladder (AgentLoopTimeouts): The ladder the cap has to outlive.

    Raises:
        ValueError: When the cap is not longer than the child deadline.
    """
    if lease_max_age_s is not None and lease_max_age_s <= ladder.child_deadline_s:
        raise ValueError(
            f"rollout.agent.sandbox.capacity.lease_max_age_s={lease_max_age_s:g} must exceed the agent "
            f"loop's child deadline of {ladder.child_deadline_s:g}s, or a healthy sandbox could have its "
            "capacity reclaimed while it is still running. Raise it, or lower trajectory_timeout."
        )
