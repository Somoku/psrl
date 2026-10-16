"""Timeout classification and progress reporting in the agent loop wrapper."""

import asyncio
from types import SimpleNamespace

import pytest
import torch
from psrl.workers.agent_loop.loops.base_agent_loop import AgentLoopBase
from psrl.workers.agent_loop.loops.budget import EpisodeBudget
from psrl.workers.agent_loop.loops.utils import TerminateReason
from psrl.workers.agent_loop.timeouts import resolve_rollout_timeouts
from psrl.workers.gen.utils import TokenOutput
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
    # Mirrors `AgentLoopBase.__init__`: the episode budget is what bounds the episode.
    loop.timeouts = resolve_rollout_timeouts(trajectory_timeout, None)
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
async def test_a_rollout_that_never_started_keeps_its_reason() -> None:
    """A setup deadline is the loop's own verdict and must survive the wrapper.

    The loop returns no output for reasons that need no retry and no traceback -- a
    provisioning deadline, a sandbox that was never admitted -- and the wrapper used to
    replace those with a bare "no output" error. The worker can then only infer a reason
    from the exception chain, which erases the difference between a rollout that never
    ran and one whose harness is broken, in the manager's breaker and in the logs.
    """
    loop = _loop()

    async def run(request):
        return None, TerminateReason.ROLLOUT_DEADLINE_EXCEEDED

    loop.run = run
    output, reason = await loop.run_with_termination_handling(_request(), raise_on_error=True)

    assert output is None
    assert reason is TerminateReason.ROLLOUT_DEADLINE_EXCEEDED
    assert reason.needs_manager_retry() is True
    assert reason.needs_worker_retry() is False


@pytest.mark.asyncio
async def test_a_successful_reason_without_output_is_still_refused() -> None:
    # The guard that the branch above narrows: a loop claiming success with no output is
    # a bug, and must not be reported to the manager as a completed rollout.
    loop = _loop()

    async def run(request):
        return None, TerminateReason.FINISHED

    loop.run = run
    with pytest.raises(RuntimeError, match="did not return a valid TokenOutput"):
        await loop.run_with_termination_handling(_request(), raise_on_error=True)


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
async def test_the_base_cancels_an_episode_that_the_loop_does_not_own() -> None:
    # Default (`discard`): the base enforces the budget by cancelling `run`, and the
    # partial work the CLI had produced is lost.
    loop = _loop(trajectory_timeout=0.05)
    completed: list[str] = []

    async def run(request):
        await asyncio.sleep(0.2)
        completed.append("returned")
        return TokenOutput(prompt_ids=[1], response_ids=[2], response_mask=[1]), TerminateReason.AGENT_TIMEOUT

    loop.run = run
    output, reason = await loop.run_with_termination_handling(_request(), raise_on_error=False)

    assert completed == [], "a loop that does not own its budget must be stopped by the base"
    assert output is None
    assert reason is TerminateReason.TRAJECTORY_TIMEOUT


@pytest.mark.asyncio
async def test_a_loop_that_owns_its_budget_is_not_cancelled_by_the_base() -> None:
    """In `keep` mode the loop stops its own agent work and wraps the episode up.

    The base must not cancel the very episode the loop is collecting and grading, so a
    run that outlives the budget still returns its own result -- here `AGENT_TIMEOUT`
    with the partial output -- rather than `TRAJECTORY_TIMEOUT`.
    """
    loop = _loop(trajectory_timeout=0.05)
    loop.episode_budget_enforced_by_loop = True
    # `_attach_loop_timing` reads the wall clock off the loop's timer.
    loop.timer = SimpleNamespace(as_dict=lambda: {})
    loop.ps_manager_handle = None
    completed: list[str] = []

    async def run(request):
        await asyncio.sleep(0.2)  # Outlives the budget the base would otherwise enforce.
        completed.append("returned")
        return TokenOutput(prompt_ids=[1], response_ids=[2], response_mask=[1]), TerminateReason.AGENT_TIMEOUT

    loop.run = run
    output, reason = await loop.run_with_termination_handling(_request(), raise_on_error=False)

    assert completed == ["returned"], "the base cancelled a loop that owns its own budget"
    assert output is not None
    assert reason is TerminateReason.AGENT_TIMEOUT


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
