"""A capacity refusal makes a create wait instead of failing the work that wanted it.

A backend that keeps its own ledger refuses a request it cannot hold right now rather than
queueing it, and the refusal clears as other sandboxes are released. These tests pin that the
caller waits for that, that the wait is bounded, and that a fault which can never clear is not
retried.
"""

from __future__ import annotations

import time

import pytest
from psrl.sandbox import (
    SandboxCapabilityError,
    SandboxCapacityTimeout,
    SandboxManager,
    SandboxSource,
    SandboxSpec,
)

from tests.sandbox.test_manager import FakeBackend, FakeSession

pytestmark = pytest.mark.asyncio


class RefusingBackend(FakeBackend):
    """A backend that refuses the first `refusals` creates, then creates one."""

    def __init__(self, refusals: int, error: BaseException | None = None) -> None:
        super().__init__(set())
        self._refusals = refusals
        self._error = error
        self.create_attempts = 0

    async def create(self, spec: SandboxSpec):
        self.create_attempts += 1
        if self.create_attempts <= self._refusals:
            raise self._error if self._error is not None else SandboxCapacityTimeout("node is full")
        return await super().create(spec)


async def test_a_refused_create_waits_and_then_succeeds() -> None:
    backend = RefusingBackend(refusals=2)
    manager = SandboxManager({"fake": backend}, "fake", capacity_wait_s=5.0)

    lease = await manager.acquire(SandboxSpec(SandboxSource.image("image")))

    assert backend.create_attempts == 3
    assert isinstance(lease.session, FakeSession)
    await lease.release()


async def test_an_endless_refusal_gives_up_after_its_budget() -> None:
    backend = RefusingBackend(refusals=10_000)
    manager = SandboxManager({"fake": backend}, "fake", capacity_wait_s=1.0)

    started = time.monotonic()
    with pytest.raises(SandboxCapacityTimeout):
        await manager.acquire(SandboxSpec(SandboxSource.image("image")))
    elapsed = time.monotonic() - started

    assert backend.create_attempts > 1
    assert elapsed == pytest.approx(1.0, abs=0.5)


async def test_a_fault_that_can_never_clear_is_not_retried() -> None:
    # A spec no backend can serve needs a deployment change, so waiting only burns the budget.
    backend = RefusingBackend(refusals=10_000, error=SandboxCapabilityError("no backend serves this"))
    manager = SandboxManager({"fake": backend}, "fake", capacity_wait_s=600.0)

    started = time.monotonic()
    with pytest.raises(SandboxCapabilityError):
        await manager.acquire(SandboxSpec(SandboxSource.image("image")))

    assert backend.create_attempts == 1
    assert time.monotonic() - started < 1.0


async def test_a_callers_own_deadline_shortens_the_wait() -> None:
    backend = RefusingBackend(refusals=10_000)
    manager = SandboxManager({"fake": backend}, "fake", capacity_wait_s=600.0)

    started = time.monotonic()
    with pytest.raises(SandboxCapacityTimeout):
        await manager._provision(
            backend,
            None,
            "grader",
            None,
            lambda grant: backend.create(SandboxSpec(SandboxSource.image("image"))),
            deadline_s=0.5,
        )
    elapsed = time.monotonic() - started

    assert backend.create_attempts > 1
    assert elapsed == pytest.approx(0.5, abs=0.4)
