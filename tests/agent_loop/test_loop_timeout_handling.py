"""Timeout classification and progress reporting in the agent loop wrapper."""

import asyncio
from types import SimpleNamespace

import pytest
import torch
from psrl.workers.agent_loop.loops.base_agent_loop import AgentLoopBase
from psrl.workers.agent_loop.loops.budget import EpisodeBudget
from psrl.workers.agent_loop.loops.utils import TerminateReason
from psrl.workers.agent_loop.timeouts import resolve_agent_loop_timeouts
from tensordict import TensorDict

pytestmark = pytest.mark.cpu_test


class _Loop(AgentLoopBase):
    """Concrete subclass so `__new__` can skip `__init__` on the ABC."""

    async def run(self, request):
        raise NotImplementedError


def _loop(trajectory_timeout: float | None = None) -> _Loop:
    loop = _Loop.__new__(_Loop)
    loop.config = SimpleNamespace(
        gen_actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(agent=SimpleNamespace(trajectory_timeout=trajectory_timeout))
        )
    )
    # Mirrors `AgentLoopBase.__init__`, which `__new__` skipped.
    loop.episode_budget = EpisodeBudget(
        trajectory_timeout,
        starts_after_provisioning=loop.episode_starts_after_provisioning,
    )
    # Mirrors `AgentLoopBase.__init__`: the ladder is what bounds the episode.
    loop.timeouts = resolve_agent_loop_timeouts(trajectory_timeout, None)
    loop.last_error = None
    loop.last_error_traceback = ""
    return loop


def _request() -> TensorDict:
    return TensorDict({"uid": torch.tensor([7])}, batch_size=[1])


@pytest.mark.asyncio
async def test_downstream_timeout_keeps_the_traceback() -> None:
    # The whole point: without the traceback a downstream timeout is undiagnosable,
    # which is how it surfaced as "<no underlying exception was captured>".
    loop = _loop()

    async def run(request):
        raise asyncio.TimeoutError("gateway call exceeded 60s")

    loop.run = run
    output, reason = await loop.run_with_termination_handling(_request(), raise_on_error=False)

    assert output is None
    assert reason is TerminateReason.DOWNSTREAM_TIMEOUT
    assert "gateway call exceeded 60s" in loop.last_error_traceback
    assert loop.last_error is not None


@pytest.mark.asyncio
async def test_downstream_timeout_is_an_error_not_a_budget() -> None:
    # A downstream timeout must stay out of `is_timeout`, or it reads as the episode
    # budget and hides the infrastructure fault.
    loop = _loop()

    async def run(request):
        raise asyncio.TimeoutError("downstream")

    loop.run = run
    _, reason = await loop.run_with_termination_handling(_request(), raise_on_error=False)

    assert reason.is_error is True
    assert reason.is_timeout is False
    assert reason.is_successful is False
    assert reason.needs_manager_retry() is True


@pytest.mark.asyncio
async def test_episode_budget_expiry_is_reported_separately() -> None:
    loop = _loop(trajectory_timeout=0.05)

    async def run(request):
        await asyncio.sleep(10)

    loop.run = run
    output, reason = await loop.run_with_termination_handling(_request(), raise_on_error=False)

    assert output is None
    assert reason is TerminateReason.TRAJECTORY_TIMEOUT
    assert reason.is_timeout is True
    assert reason.is_error is False
    assert "Episode budget" in loop.last_error_traceback


@pytest.mark.asyncio
async def test_a_generic_error_is_still_reported_as_a_rollout_error() -> None:
    loop = _loop()

    async def run(request):
        raise ValueError("boom")

    loop.run = run
    _, reason = await loop.run_with_termination_handling(_request(), raise_on_error=False)

    assert reason is TerminateReason.ROLLOUT_ERROR
    assert "boom" in loop.last_error_traceback


@pytest.mark.asyncio
async def test_sandbox_capacity_timeout_is_its_own_reason() -> None:
    """A sandbox that was never admitted must not read as a failing harness."""
    from psrl.sandbox import SandboxCapacityTimeout

    loop = _loop()

    async def run(request):
        raise SandboxCapacityTimeout("lease never admitted")

    loop.run = run
    _, reason = await loop.run_with_termination_handling(_request(), raise_on_error=False)

    assert reason is TerminateReason.SANDBOX_CAPACITY_TIMEOUT
    assert reason.is_coordination_fault is True
    assert reason.needs_manager_retry() is True
    assert reason.needs_worker_retry() is False
    assert reason.counts_toward_refill_breaker() is False
    assert "lease never admitted" in loop.last_error_traceback
