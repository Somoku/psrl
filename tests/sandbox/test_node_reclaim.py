"""One node-level sweep decides every reclamation, by one reason per sandbox."""

import asyncio

import pytest
from psrl.sandbox.core import (
    PauseMode,
    SandboxBusyError,
    SandboxCapabilities,
    SandboxExitReason,
    SandboxFeature,
    SandboxRef,
)
from psrl.sandbox.node_reclaim import NodeReclaimLoop, ReclaimWindows, ResidentSandbox

WINDOWS = ReclaimWindows(pause_window_s=10, reap_window_s=30, lifetime_s=100, sweep_interval_s=1)


class FakeSession:
    """A session whose activity and capabilities a test states directly."""

    def __init__(self, sandbox_id="sb", *, busy=False, last_activity_at=0.0, features=frozenset()):
        self._id = sandbox_id
        self.busy = busy
        self.last_activity_at = last_activity_at
        self._features = features
        self.paused_as: PauseMode | None = None
        self.pause_error: Exception | None = None

    @property
    def ref(self):
        return SandboxRef("fake", self._id)

    @property
    def capabilities(self):
        return SandboxCapabilities(self._features)

    async def pause(self, mode):
        if self.pause_error is not None:
            raise self.pause_error
        self.paused_as = mode


def resident(session, *, created_at=0.0, owner_alive=None, released=None):
    return ResidentSandbox(
        session,
        release=lambda reason: released.append(reason) if released is not None else None,
        created_at=created_at,
        owner_alive=owner_alive,
    )


def loop_over(residents):
    return NodeReclaimLoop(lambda: residents, WINDOWS)


@pytest.mark.asyncio
async def test_a_busy_sandbox_is_never_reclaimed():
    # Age alone would read a long command as idle, because a backend stamps
    # activity when a command returns.
    session = FakeSession(busy=True, last_activity_at=0.0)
    released = []

    outcome = await loop_over([resident(session, released=released)]).sweep(now=50)

    assert released == []
    assert outcome.skipped_busy == 1
    assert session.paused_as is None


@pytest.mark.asyncio
async def test_a_sandbox_that_reports_no_activity_is_never_idle():
    # A backend that reports nothing cannot be told apart from a quiet one. Swept
    # inside the lifetime, so the backstop is not what keeps it alive here.
    session = FakeSession(last_activity_at=None)
    released = []

    outcome = await loop_over([resident(session, released=released)]).sweep(now=50)

    assert released == []
    assert outcome.skipped_busy == 1


@pytest.mark.asyncio
async def test_an_idle_sandbox_is_paused_before_it_is_destroyed():
    session = FakeSession(last_activity_at=0.0)
    released = []

    outcome = await loop_over([resident(session, released=released)]).sweep(now=15)

    assert outcome.paused == ["sb"]
    assert released == []


@pytest.mark.asyncio
async def test_a_pause_prefers_a_resident_freeze_when_the_backend_has_one():
    frozen = FakeSession("frozen", last_activity_at=0.0, features=frozenset({SandboxFeature.FREEZE}))
    hibernating = FakeSession("hib", last_activity_at=0.0)

    await loop_over([resident(frozen), resident(hibernating)]).sweep(now=15)

    assert frozen.paused_as is PauseMode.FREEZE
    assert hibernating.paused_as is PauseMode.HIBERNATE


@pytest.mark.asyncio
async def test_a_sandbox_that_turns_out_to_be_busy_is_decided_again_next_sweep():
    session = FakeSession(last_activity_at=0.0)
    session.pause_error = SandboxBusyError("a command started between the read and the pause")
    reclaim = loop_over([resident(session)])

    outcome = await reclaim.sweep(now=15)

    assert outcome.paused == []
    assert outcome.failures == 0


@pytest.mark.asyncio
async def test_a_backend_without_a_pause_is_not_a_failure():
    session = FakeSession(last_activity_at=0.0)
    session.pause_error = NotImplementedError("no pause on this backend")

    outcome = await loop_over([resident(session)]).sweep(now=15)

    assert outcome.paused == []
    assert outcome.failures == 0


@pytest.mark.asyncio
async def test_an_idle_sandbox_past_the_reap_window_is_destroyed():
    released = []

    outcome = await loop_over([resident(FakeSession(last_activity_at=0.0), released=released)]).sweep(now=40)

    assert released == [SandboxExitReason.REAPED_IDLE]
    assert outcome.reaped_idle == ["sb"]


@pytest.mark.asyncio
async def test_a_never_idle_sandbox_still_ends_at_its_lifetime():
    # The backstop is what bounds a sandbox that keeps running commands.
    session = FakeSession(last_activity_at=199.0)
    released = []

    outcome = await loop_over([resident(session, created_at=0.0, released=released)]).sweep(now=200)

    assert released == [SandboxExitReason.REAPED_LIFETIME]
    assert outcome.reaped_lifetime == ["sb"]


@pytest.mark.asyncio
async def test_a_sandbox_whose_owner_is_gone_is_reclaimed_without_waiting_for_idle():
    # Waiting for the idle window would hold a slot for a caller that no longer exists.
    session = FakeSession(busy=True, last_activity_at=0.0)
    released = []

    outcome = await loop_over([resident(session, created_at=0.0, owner_alive=lambda: False, released=released)]).sweep(
        now=1
    )

    assert released == [SandboxExitReason.RECLAIMED_ORPHAN]
    assert outcome.reclaimed_orphan == ["sb"]


@pytest.mark.asyncio
async def test_each_sandbox_is_reclaimed_for_exactly_one_reason_per_sweep():
    # Idle past its reap window and past its lifetime at once: the sweep must not
    # release it twice.
    released = []
    session = FakeSession(last_activity_at=0.0)

    outcome = await loop_over([resident(session, created_at=0.0, released=released)]).sweep(now=500)

    assert len(released) == 1
    assert outcome.released == 1


@pytest.mark.asyncio
async def test_one_failing_sandbox_does_not_stop_the_sweep():
    class Exploding(FakeSession):
        async def pause(self, mode):
            raise RuntimeError("daemon refused")

    good = FakeSession("good", last_activity_at=0.0)

    outcome = await loop_over([resident(Exploding("bad", last_activity_at=0.0)), resident(good)]).sweep(now=15)

    assert outcome.failures == 1
    assert outcome.paused == ["good"]


@pytest.mark.asyncio
async def test_counters_accumulate_across_sweeps_for_the_metric_hook():
    reclaim = loop_over([resident(FakeSession(last_activity_at=0.0))])

    await reclaim.sweep(now=15)
    await reclaim.sweep(now=15)

    assert reclaim.snapshot()["reclaim/paused"] == 1.0


@pytest.mark.asyncio
async def test_the_sweep_stops_cleanly_and_leaves_its_sandboxes_in_place():
    released = []
    reclaim = loop_over([resident(FakeSession(last_activity_at=0.0), released=released)])

    reclaim.start()
    await asyncio.sleep(0)
    await reclaim.stop()

    assert released == []


@pytest.mark.asyncio
async def test_a_paused_sandbox_is_not_paused_again():
    session = FakeSession(last_activity_at=0.0)
    held = resident(session)
    reclaim = loop_over([held])

    first = await reclaim.sweep(now=15)
    session.paused_as = None
    second = await reclaim.sweep(now=15)

    assert first.paused == ["sb"]
    assert second.paused == []
    assert session.paused_as is None
