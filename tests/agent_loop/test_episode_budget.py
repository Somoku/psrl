"""The episode budget must not charge scheduling latency to the episode.

A sandboxed harness spends real time waiting for node capacity before its first turn.
Charging that wait to the episode budget makes a healthy episode fail whenever the node
is busy, which is why the clock starts only once provisioning is done.
"""

import asyncio

import pytest
from psrl.workers.agent_loop.loops.budget import (
    EpisodeBudget,
    EpisodeBudgetExpired,
    RolloutDeadlineExceeded,
)

pytestmark = pytest.mark.cpu_test


def _sleep_task(delay: float) -> asyncio.Task:
    async def work() -> str:
        await asyncio.sleep(delay)
        return "finished"

    return asyncio.ensure_future(work())


@pytest.mark.asyncio
async def test_a_budget_expiry_is_not_a_timeout_error() -> None:
    """An episode may raise TimeoutError itself, so the two must stay distinguishable."""
    budget = EpisodeBudget(5.0, starts_after_provisioning=True)

    async def episode() -> None:
        budget.arm()
        raise TimeoutError("a downstream call timed out")

    with pytest.raises(TimeoutError, match="a downstream call timed out"):
        await budget.wait_for(asyncio.ensure_future(episode()))


@pytest.mark.asyncio
async def test_disabled_budget_awaits_the_task() -> None:
    budget = EpisodeBudget(None, starts_after_provisioning=True)

    assert await budget.wait_for(_sleep_task(0.01)) == "finished"


@pytest.mark.asyncio
async def test_provisioning_wait_is_excluded_from_the_budget() -> None:
    """Waiting for admission must not consume the episode's own time."""
    budget = EpisodeBudget(0.3, starts_after_provisioning=True)

    async def episode() -> str:
        await asyncio.sleep(0.5)  # admission and environment preparation
        budget.arm()
        return await _sleep_task(0.1)  # the episode itself, well inside 0.3s

    assert await budget.wait_for(asyncio.ensure_future(episode())) == "finished"
    assert budget.wait_s() >= 0.5


@pytest.mark.asyncio
async def test_an_episode_that_outlives_its_budget_is_stopped() -> None:
    budget = EpisodeBudget(0.1, starts_after_provisioning=True)

    async def episode() -> str:
        budget.arm()
        return await _sleep_task(5.0)

    with pytest.raises(EpisodeBudgetExpired, match="Episode budget of 0.1s expired"):
        await budget.wait_for(asyncio.ensure_future(episode()))


@pytest.mark.asyncio
async def test_a_loop_without_provisioning_is_timed_from_the_call() -> None:
    """A loop that never arms keeps its previous behaviour of bounding the whole call."""
    budget = EpisodeBudget(0.1)

    with pytest.raises(EpisodeBudgetExpired):
        await budget.wait_for(_sleep_task(5.0))

    assert budget.armed


@pytest.mark.asyncio
async def test_an_unarmed_budget_does_not_stop_a_finished_episode() -> None:
    """Before the episode starts, its own deadlines bound the task, not the budget."""
    budget = EpisodeBudget(0.05, starts_after_provisioning=True)

    async def provisioning() -> str:
        await asyncio.sleep(0.2)
        return "provisioned"

    assert await budget.wait_for(asyncio.ensure_future(provisioning())) == "provisioned"
    assert not budget.armed


@pytest.mark.asyncio
async def test_arming_is_idempotent_and_records_only_the_first_wait() -> None:
    budget = EpisodeBudget(5.0, starts_after_provisioning=True)
    await asyncio.sleep(0.05)

    budget.arm()
    first_wait = budget.wait_s()
    budget.arm()

    assert budget.wait_s() == first_wait
    assert first_wait >= 0.05


@pytest.mark.asyncio
async def test_a_rollout_that_never_starts_is_stopped_by_the_setup_allowance() -> None:
    """Provisioning must not be able to hold a sandbox forever waiting for an episode."""
    budget = EpisodeBudget(60.0, starts_after_provisioning=True, setup_limit_s=0.05)

    async def wedged_provisioning() -> None:
        await asyncio.sleep(5)

    with pytest.raises(RolloutDeadlineExceeded, match="setup allowance"):
        await budget.wait_for(asyncio.ensure_future(wedged_provisioning()))

    assert not budget.armed, "No episode started, so the episode clock must not have started."


@pytest.mark.asyncio
async def test_the_setup_allowance_does_not_leak_into_the_episode_clock() -> None:
    """A generous setup allowance must not shorten the episode that follows it."""
    budget = EpisodeBudget(0.3, starts_after_provisioning=True, setup_limit_s=5.0)

    async def episode() -> str:
        await asyncio.sleep(0.2)  # provisioning, inside the setup allowance
        budget.arm()
        return await _sleep_task(0.2)  # episode work, inside the episode budget

    assert await budget.wait_for(asyncio.ensure_future(episode())) == "finished"
    assert budget.wait_s() >= 0.2, "Setup time must not be charged to the episode."
