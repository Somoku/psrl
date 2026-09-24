"""The ownership state machine, which is what makes a lease mistake a crash.

The machine exists so a double release or a reclaim of a lease a caller still
owns fails where the mistake is made, rather than as an accounting drift later.
"""

from __future__ import annotations

import pytest
from psrl.sandbox.ownership import (
    IllegalLeaseTransition,
    LeaseState,
    LeaseStateMachine,
    allowed_transitions,
)

pytestmark = pytest.mark.cpu_test


def test_a_new_machine_starts_before_admission() -> None:
    machine = LeaseStateMachine()

    assert machine.state is LeaseState.PREPARE
    assert not machine.terminal
    assert not machine.holds_resources


def test_the_happy_path_walks_every_state_once() -> None:
    machine = LeaseStateMachine()

    for state in (LeaseState.QUEUED, LeaseState.PROVISIONING, LeaseState.LEASED, LeaseState.RECLAIMING):
        machine.transition(state)
    machine.transition(LeaseState.RELEASED)

    assert machine.state is LeaseState.RELEASED
    assert machine.terminal
    assert machine.describe() == "prepare -> queued -> provisioning -> leased -> reclaiming -> released"


def test_a_release_is_refused_while_a_caller_still_owns_the_sandbox() -> None:
    # `leased` only reaches `reclaiming`. Anything that skipped it would release a
    # sandbox out from under the episode that is using it.
    machine = LeaseStateMachine()
    machine.transition(LeaseState.QUEUED)
    machine.transition(LeaseState.PROVISIONING)
    machine.transition(LeaseState.LEASED)

    with pytest.raises(IllegalLeaseTransition, match="cannot move"):
        machine.transition(LeaseState.RELEASED)

    assert machine.state is LeaseState.LEASED


def test_a_released_lease_is_terminal() -> None:
    machine = LeaseStateMachine()
    for state in (LeaseState.QUEUED, LeaseState.PROVISIONING, LeaseState.LEASED, LeaseState.RECLAIMING):
        machine.transition(state)
    machine.transition(LeaseState.RELEASED)

    with pytest.raises(IllegalLeaseTransition):
        machine.transition(LeaseState.RECLAIMING)


def test_repeating_a_state_is_not_a_transition() -> None:
    machine = LeaseStateMachine()
    machine.transition(LeaseState.QUEUED)

    with pytest.raises(IllegalLeaseTransition, match="already"):
        machine.transition(LeaseState.QUEUED)


def test_a_reclaim_retry_is_the_one_repeatable_state() -> None:
    # A destruction the backend refused is retried, so `reclaiming` has to be able
    # to re-enter itself without the retry path being a special case.
    machine = LeaseStateMachine()
    machine.transition(LeaseState.RECLAIMING)

    machine.transition(LeaseState.RECLAIMING)

    assert machine.state is LeaseState.RECLAIMING


def test_a_lease_that_cannot_be_admitted_goes_straight_to_reclamation() -> None:
    machine = LeaseStateMachine()

    machine.transition(LeaseState.RECLAIMING)

    assert machine.state is LeaseState.RECLAIMING


def test_only_a_finished_or_unadmitted_lease_holds_nothing() -> None:
    # This predicate is what decides whether an unreclaimable lease keeps its
    # reservation charged, so it is stated once rather than at each call site.
    machine = LeaseStateMachine()

    holdings = {LeaseState.PREPARE: False}
    for state in (
        LeaseState.QUEUED,
        LeaseState.PROVISIONING,
        LeaseState.LEASED,
        LeaseState.RECLAIMING,
        LeaseState.RELEASED,
    ):
        machine.transition(state)
        holdings[state] = machine.holds_resources

    assert holdings == {
        LeaseState.PREPARE: False,
        LeaseState.QUEUED: True,
        LeaseState.PROVISIONING: True,
        LeaseState.LEASED: True,
        LeaseState.RECLAIMING: True,
        LeaseState.RELEASED: False,
    }


def test_every_state_has_a_defined_exit() -> None:
    # A state with no allowed exit would be a place a lease can never leave.
    for state in LeaseState:
        if state is LeaseState.RELEASED:
            assert allowed_transitions(state) == frozenset()
            continue
        assert allowed_transitions(state), f"{state.value} has no allowed transition."


def test_the_trail_records_the_time_of_each_state() -> None:
    machine = LeaseStateMachine()
    machine.transition(LeaseState.QUEUED)

    history = machine.history

    assert [state for state, _ in history] == [LeaseState.PREPARE, LeaseState.QUEUED]
    assert history[0][1] <= history[1][1]
    assert machine.age_s >= 0
