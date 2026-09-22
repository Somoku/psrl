"""The rollout liveness heartbeat that keeps a slow episode from being declared dead.

A healthy episode reports only when it finishes, so without a heartbeat the manager cannot
tell a long episode from a wedged worker and has to guess with a timeout long enough to
cover a whole episode. The heartbeat replaces that guess with a silence check.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from psrl.workers.agent_loop.worker import PSRL_AgentLoopWorker

pytestmark = pytest.mark.cpu_test

# `@ray.remote` replaces the class with an actor handle factory, so reach inside it to get the
# plain class the actor methods are defined on.
_WORKER_CLASS = PSRL_AgentLoopWorker.__ray_metadata__.modified_class


def _worker(heartbeat_s: float) -> PSRL_AgentLoopWorker:
    worker = _WORKER_CLASS.__new__(_WORKER_CLASS)
    # The heartbeat is the only thing under test, so it needs nothing but its period.
    worker.timeouts = SimpleNamespace(heartbeat_interval_s=heartbeat_s)
    return worker


def _manager(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(touch_inflight_group=SimpleNamespace(remote=AsyncMock(**kwargs)))


@pytest.mark.asyncio
async def test_the_heartbeat_touches_the_entry_while_the_child_runs() -> None:
    worker = _worker(0.02)
    worker.agent_loop_manager = _manager()

    heartbeat = asyncio.ensure_future(worker._report_liveness(7))
    await asyncio.sleep(0.1)
    heartbeat.cancel()
    await asyncio.gather(heartbeat, return_exceptions=True)

    assert worker.agent_loop_manager.touch_inflight_group.remote.await_count >= 2
    assert worker.agent_loop_manager.touch_inflight_group.remote.await_args_list[0].args == (7,)


@pytest.mark.asyncio
async def test_a_failing_heartbeat_does_not_kill_the_episode() -> None:
    """The manager decides what silence means; the heartbeat only reports."""
    worker = _worker(0.02)
    worker.agent_loop_manager = _manager(side_effect=RuntimeError("manager busy"))

    heartbeat = asyncio.ensure_future(worker._report_liveness(7))
    await asyncio.sleep(0.1)
    assert not heartbeat.done(), "A failed heartbeat must not end the heartbeat loop."

    heartbeat.cancel()
    await asyncio.gather(heartbeat, return_exceptions=True)


@pytest.mark.asyncio
async def test_the_heartbeat_tolerates_a_missing_manager_handle() -> None:
    worker = _worker(0.01)
    worker.agent_loop_manager = None

    heartbeat = asyncio.ensure_future(worker._report_liveness(7))
    await asyncio.sleep(0.05)
    assert not heartbeat.done()

    heartbeat.cancel()
    await asyncio.gather(heartbeat, return_exceptions=True)
