"""One explicit ownership state machine for a sandbox lease.

The lease lifecycle used to be spread across flags and callbacks, which meant the
only place the rules lived was prose. An illegal transition is now a raise, so a
change that would double-release a lease or reclaim one a caller still owns fails
at the point of the mistake rather than as an accounting drift hours later.

The states are deliberately few. Each one is a place where ownership or capacity
is held differently, and nothing else belongs here.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from enum import Enum
from types import MappingProxyType

psrl_logger = logging.getLogger(__file__)


class LeaseState(str, Enum):
    """
    Where a lease is in its lifecycle.
    """

    # The workflow phase is reserved and the request is waiting for admission.
    PREPARE = "prepare"
    # A capacity request is queued with the node.
    QUEUED = "queued"
    # The backend is creating the sandbox, so a runtime object may already exist.
    PROVISIONING = "provisioning"
    # A caller owns the sandbox.
    LEASED = "leased"
    # Destruction is in flight, or a failed one is being retried.
    RECLAIMING = "reclaiming"
    # Terminal. The sandbox is destroyed and its capacity is returned.
    RELEASED = "released"


class IllegalLeaseTransition(RuntimeError):
    """
    Raised when a transition would break an ownership invariant.
    """


# Allowed transitions. A lease never moves backwards, and RELEASED is terminal, so
# a double release and a release that races a retry both fail here.
_TRANSITIONS: Mapping[LeaseState, frozenset[LeaseState]] = MappingProxyType(
    {
        LeaseState.PREPARE: frozenset({LeaseState.QUEUED, LeaseState.RECLAIMING}),
        LeaseState.QUEUED: frozenset({LeaseState.PROVISIONING, LeaseState.RECLAIMING}),
        LeaseState.PROVISIONING: frozenset({LeaseState.LEASED, LeaseState.RECLAIMING}),
        LeaseState.LEASED: frozenset({LeaseState.RECLAIMING}),
        # Reclaiming to itself is a retry of a destruction the backend refused.
        LeaseState.RECLAIMING: frozenset({LeaseState.RECLAIMING, LeaseState.RELEASED}),
        LeaseState.RELEASED: frozenset(),
    }
)

# States in which the lease still holds node capacity or a runtime object.
_HELD_STATES = frozenset({LeaseState.QUEUED, LeaseState.PROVISIONING, LeaseState.LEASED, LeaseState.RECLAIMING})


def allowed_transitions(state: LeaseState) -> frozenset[LeaseState]:
    """
    Return the states a lease may move to from one state.
    """
    return _TRANSITIONS[state]


class LeaseStateMachine:
    """
    Track one lease's state and refuse an impossible move.
    """

    def __init__(
        self,
        on_transition: Callable[[LeaseState, LeaseState], None] | None = None,
    ) -> None:
        self._state = LeaseState.PREPARE
        self._entered_at = time.monotonic()
        self._history: list[tuple[LeaseState, float]] = [(LeaseState.PREPARE, self._entered_at)]
        self._on_transition = on_transition

    @property
    def state(self) -> LeaseState:
        """
        Return the current state.
        """
        return self._state

    @property
    def terminal(self) -> bool:
        """
        Return whether the lease has finished, so nothing more may happen to it.
        """
        return self._state is LeaseState.RELEASED

    @property
    def holds_resources(self) -> bool:
        """
        Return whether the lease still holds a runtime object or capacity.

        This is the predicate that decides whether an unreclaimable lease has to
        keep its reservation charged, so it is stated once here rather than
        re-derived at each call site.
        """
        return self._state in _HELD_STATES

    @property
    def age_s(self) -> float:
        """
        Return how long the lease has been in its current state.
        """
        return time.monotonic() - self._entered_at

    @property
    def history(self) -> tuple[tuple[LeaseState, float], ...]:
        """
        Return the states this lease passed through, oldest first.
        """
        return tuple(self._history)

    def transition(self, target: LeaseState) -> None:
        """
        Move to a new state, or raise when the move is not allowed.

        Args:
            target (LeaseState): The state to enter.

        Raises:
            IllegalLeaseTransition: When the pair is not in the transition table.
        """
        if target is self._state and target is not LeaseState.RECLAIMING:
            raise IllegalLeaseTransition(
                f"Sandbox lease is already {self._state.value!r}, and repeating a state is not a transition."
            )
        if target not in _TRANSITIONS[self._state]:
            raise IllegalLeaseTransition(
                f"Sandbox lease cannot move from {self._state.value!r} to {target.value!r}. "
                f"Allowed: {sorted(state.value for state in _TRANSITIONS[self._state])}."
            )
        previous = self._state
        self._state = target
        self._entered_at = time.monotonic()
        self._history.append((target, self._entered_at))
        if self._on_transition is not None:
            self._on_transition(previous, target)

    def describe(self) -> str:
        """
        Return a compact trail, which is what a stuck lease needs explained.
        """
        return " -> ".join(state.value for state, _ in self._history)
