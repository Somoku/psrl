"""The wire adapter one sandbox node actor puts in front of its agent.

Driven without Ray on purpose. The adapter's job is to forward a call and to make its
failure survive an actor boundary, and both are properties of the method rather than of
the transport. What needs a real cluster is serialization and node affinity, which
`smoke_ray_plane.py` covers.
"""

from __future__ import annotations

import pickle

import pytest

ray = pytest.importorskip("ray", reason="The Ray plane adapter needs Ray importable.")

from psrl.sandbox.ray_plane import SandboxNodeActor, _wire_safe  # noqa: E402

pytestmark = pytest.mark.cpu_test


class Unpicklable(Exception):
    """A failure that cannot cross a boundary, the way an aiohttp connector error cannot.

    It takes more constructor arguments than its `args` carries, so it pickles cleanly
    and then fails to rebuild, which is exactly the shape that hid the real error.
    """

    def __init__(self, detail: str, path: str) -> None:
        super().__init__(detail)
        self.path = path


class FakeAgent:
    """The parts of the node agent surface the adapter forwards to."""

    def __init__(self) -> None:
        self.node_id = "node-a"
        self.liveness_reporter = None

    def snapshot(self) -> dict[str, float]:
        """Plain, because the metric hook reads counters and touches no daemon."""
        return {"ownership/leases": 1.0}

    async def status(self, backend: str, sandbox_id: str) -> str:
        return "running"

    async def boom(self) -> None:
        raise Unpicklable("daemon is not listening", "/var/run/docker.sock")


def _adapter() -> SandboxNodeActor:
    """Build the adapter around a fake agent, without Ray constructing it."""
    actor = SandboxNodeActor.__new__(SandboxNodeActor)
    actor.node_id = "node-a"
    actor.placement = None
    actor.capacity = None
    actor.manager = None
    actor.agent = FakeAgent()
    return actor


async def test_a_synchronous_agent_call_is_forwarded_rather_than_awaited() -> None:
    # The metric hook is the trainer's per-step read and it returns a plain mapping, so
    # an adapter that always awaits raised on every collection instead of reporting.
    adapter = _adapter()

    assert await adapter.snapshot() == {"ownership/leases": 1.0}


async def test_an_asynchronous_agent_call_is_still_awaited() -> None:
    adapter = _adapter()

    assert await adapter.status("docker", "sandbox-1") == "running"


async def test_a_failure_that_cannot_be_rebuilt_is_replaced_by_one_that_can() -> None:
    # Ray replaces an unpicklable error with one that hides what actually broke, so the
    # node reports a carrier that names the original type and message instead.
    adapter = _adapter()

    with pytest.raises(Exception) as raised:
        await adapter._call("boom", adapter.agent.boom)

    assert "Unpicklable" in str(raised.value)
    assert "daemon is not listening" in str(raised.value)
    assert pickle.loads(pickle.dumps(raised.value))


def test_a_failure_that_survives_the_wire_is_left_alone() -> None:
    # Wrapping a perfectly good error would cost the caller its type, so the test is the
    # wire rather than a list of exception classes.
    original = RuntimeError("image is missing")

    assert _wire_safe("acquire", "node-a", original) is original
