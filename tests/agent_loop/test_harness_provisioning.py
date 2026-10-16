"""Retry and classification of the harness loop's provisioning phase.

Provisioning is the one phase every episode of a step runs at the same instant, so its
failures are contention on the node rather than a verdict on the task. These tests pin
what the loop does with them: re-attempt the ones another attempt could plausibly fix,
and report the exhaustion as a rollout that never started.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from psrl.sandbox import (
    SandboxCommandTimeout,
    SandboxProvisionError,
    SandboxSource,
    SandboxSpec,
    SandboxTransportError,
)
from psrl.workers.agent_loop.harness import HarnessTaskContext
from psrl.workers.agent_loop.loops import harness_agent_loop
from psrl.workers.agent_loop.loops.harness_agent_loop import HarnessAgentLoop
from psrl.workers.agent_loop.loops.utils import TerminateReason

pytestmark = pytest.mark.cpu_test


class _ConcreteLoop(HarnessAgentLoop):
    """A concrete subclass, so the abstract lifecycle class can be allocated directly."""

    async def prepare_harness_task(self, request):  # pragma: no cover - always stubbed
        raise NotImplementedError


def _loop() -> _ConcreteLoop:
    return _ConcreteLoop.__new__(_ConcreteLoop)


def _task() -> HarnessTaskContext:
    return HarnessTaskContext(
        state=None,
        prompt="fix",
        sandbox_spec=SandboxSpec(SandboxSource.image("rollout")),
    )


@pytest.mark.asyncio
async def test_a_transient_provisioning_fault_is_retried(monkeypatch) -> None:
    loop = _loop()
    attempts: list[int] = []
    provisioned = SimpleNamespace(task="task")

    async def provision_once(request, timing, run_start):
        attempts.append(len(attempts) + 1)
        if len(attempts) < 3:
            raise SandboxProvisionError(SimpleNamespace(ref="docker/abc"), TimeoutError("node did not answer"))
        return provisioned

    monkeypatch.setattr(harness_agent_loop, "_PROVISIONING_RETRY_BACKOFF_S", 0.0)
    loop._provision_once = provision_once

    assert await loop._provision_for_run({}, {}, 0.0) is provisioned
    assert len(attempts) == 3, "A transient fault must be re-attempted, not reported."


@pytest.mark.asyncio
async def test_exhausted_attempts_raise_the_last_fault(monkeypatch) -> None:
    loop = _loop()
    attempts: list[int] = []

    async def provision_once(request, timing, run_start):
        attempts.append(len(attempts) + 1)
        raise SandboxCommandTimeout(f"probe {len(attempts)} overran its budget", sandbox_preserved=False)

    monkeypatch.setattr(harness_agent_loop, "_PROVISIONING_ATTEMPTS", 3)
    monkeypatch.setattr(harness_agent_loop, "_PROVISIONING_RETRY_BACKOFF_S", 0.0)
    loop._provision_once = provision_once

    with pytest.raises(SandboxCommandTimeout, match="probe 3 overran"):
        await loop._provision_for_run({}, {}, 0.0)

    assert len(attempts) == 3, "Every allowed attempt must be spent before giving up."


@pytest.mark.asyncio
async def test_a_deterministic_provisioning_fault_is_not_retried(monkeypatch) -> None:
    # A missing runtime tree or an unusable image fails the same way every time, so
    # retrying only delays the diagnosis and burns the episode's setup allowance.
    loop = _loop()
    attempts: list[int] = []

    async def provision_once(request, timing, run_start):
        attempts.append(len(attempts) + 1)
        raise RuntimeError("harness executable is missing or not executable")

    monkeypatch.setattr(harness_agent_loop, "_PROVISIONING_ATTEMPTS", 3)
    monkeypatch.setattr(harness_agent_loop, "_PROVISIONING_RETRY_BACKOFF_S", 0.0)
    loop._provision_once = provision_once

    with pytest.raises(RuntimeError, match="missing or not executable"):
        await loop._provision_for_run({}, {}, 0.0)

    assert len(attempts) == 1


@pytest.mark.asyncio
async def test_a_failed_attempt_tears_down_what_it_built(monkeypatch) -> None:
    """An attempt that fails must release its own lease, session, and task.

    The retry starts from a clean slate because nothing else knows those resources exist:
    the caller never received them, so it cannot clean them up.
    """
    loop = _loop()
    task = _task()
    session = SimpleNamespace(ref="docker/abc")
    lease = SimpleNamespace(session=session, release=AsyncMock())
    loop.prepare_harness_task = AsyncMock(return_value=task)
    loop.attach_runtime_mount = lambda value: value
    loop.apply_sandbox_lifetime = lambda value: value
    loop.create_session = AsyncMock(return_value="session")
    loop.sandbox_manager = SimpleNamespace(prepare=AsyncMock(), acquire=AsyncMock(return_value=lease))
    loop._try_snapshot_clean_sandbox = AsyncMock(return_value=None)
    loop._cleanup_harness_run = AsyncMock()

    async def prepare_harness_sandbox(task, sandbox, timing):
        raise SandboxProvisionError(sandbox, RuntimeError("sandbox initialization failed"))

    loop.prepare_harness_sandbox = prepare_harness_sandbox

    with pytest.raises(SandboxProvisionError, match="initialization failed"):
        await loop._provision_once({}, {}, 0.0)

    loop._cleanup_harness_run.assert_awaited_once_with(task, "session", lease, None, None)


@pytest.mark.asyncio
async def test_a_failed_attempt_cancels_its_speculative_preparation() -> None:
    loop = _loop()
    task = _task()
    cancelled = asyncio.Event()

    async def prepare(*args, **kwargs):
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    async def create_session(request):
        await asyncio.sleep(0.01)  # let the image pull start before failing the attempt
        raise SandboxTransportError("router refused the session")

    loop.prepare_harness_task = AsyncMock(return_value=task)
    loop.attach_runtime_mount = lambda value: value
    loop.apply_sandbox_lifetime = lambda value: value
    loop.sandbox_manager = SimpleNamespace(prepare=prepare, acquire=AsyncMock())
    loop.create_session = create_session
    loop._cleanup_harness_run = AsyncMock()

    with pytest.raises(SandboxTransportError, match="refused the session"):
        await loop._provision_once({}, {}, 0.0)

    assert cancelled.is_set(), "A failed attempt must not leave its image pull running."
    loop._cleanup_harness_run.assert_awaited_once_with(task, None, None, None, None)


@pytest.mark.asyncio
async def test_a_rollout_that_never_provisioned_reports_a_setup_deadline() -> None:
    """Exhausted provisioning is a rollout that never started, not a downstream timeout.

    No task code ran, so the reason must not read as evidence about the harness, and it
    must stay distinct from `DOWNSTREAM_TIMEOUT`, which means a call inside a running
    episode overran its budget.
    """
    loop = _loop()
    loop.resolve_request_settings = lambda request: None
    loop._cleanup_harness_run = AsyncMock()

    async def provision_for_run(request, timing, run_start):
        raise SandboxTransportError("node unreachable")

    loop._provision_for_run = provision_for_run

    output, reason = await loop.run({"uid": 7})

    assert output is None
    assert reason is TerminateReason.ROLLOUT_DEADLINE_EXCEEDED
    assert reason.needs_manager_retry() is True
    assert reason.needs_worker_retry() is False
    assert isinstance(loop.last_error, SandboxTransportError)
    # The exhausted attempt already released its own resources, so teardown has nothing
    # left to do and must not be given a half-built sandbox to destroy.
    loop._cleanup_harness_run.assert_awaited_once_with(None, None, None, None, None)
