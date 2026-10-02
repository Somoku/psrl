"""Node-level reclamation: one loop, one clock, four reasons.

Reclamation belongs to the node, not to the caller that happened to create a
sandbox. A caller that exits must not leave its sandboxes resident until some
other mechanism notices, and a node that holds a container holds its memory
whoever asked for it.

That is why this is one sweep rather than two. The idle windows used to run in
the worker's manager while orphan reclamation ran in a node collector, so the
same question -- which sandboxes on this node should go -- was answered by two
components on two clocks, and a worker exiting silently stopped half the answer.

The four reasons are ordered by how much they preserve, and the order is the
policy: a sandbox is paused before it is destroyed, and the lifetime is a
backstop behind the idle windows rather than a competitor to them.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from psrl.sandbox.async_utils import complete_cleanup
from psrl.sandbox.core import PauseMode, SandboxBusyError, SandboxExitReason, SandboxFeature, SandboxSession

psrl_logger = logging.getLogger(__file__)


@dataclass(frozen=True)
class ReclaimWindows:
    """The spans this sweep enforces, taken from the one timing contract.

    They arrive resolved rather than derived here, because the orderings between
    them are asserted once in `TimingContract.validate()` and must not be
    re-derived anywhere else.
    """

    pause_window_s: float
    reap_window_s: float
    lifetime_s: float
    sweep_interval_s: float


@dataclass
class ReclaimOutcome:
    """What one sweep did, counted by reason.

    Counted rather than logged, because the planes report and the trainer's metric
    hook is the only reader.
    """

    paused: list[str] = field(default_factory=list)
    reaped_idle: list[str] = field(default_factory=list)
    reaped_lifetime: list[str] = field(default_factory=list)
    reclaimed_orphan: list[str] = field(default_factory=list)
    skipped_busy: int = 0
    failures: int = 0

    @property
    def released(self) -> int:
        """Return how many sandboxes this sweep destroyed, for any reason."""
        return len(self.reaped_idle) + len(self.reaped_lifetime) + len(self.reclaimed_orphan)

    def as_dict(self) -> dict[str, float]:
        """Flatten for the metric hook."""
        return {
            "reclaim/paused": float(len(self.paused)),
            "reclaim/reaped_idle": float(len(self.reaped_idle)),
            "reclaim/reaped_lifetime": float(len(self.reaped_lifetime)),
            "reclaim/reclaimed_orphan": float(len(self.reclaimed_orphan)),
            "reclaim/skipped_busy": float(self.skipped_busy),
            "reclaim/failures": float(self.failures),
        }


class ResidentSandbox:
    """One sandbox this node holds, with the facts the sweep decides on."""

    def __init__(
        self,
        session: SandboxSession,
        *,
        release: Callable[[SandboxExitReason], object],
        created_at: float,
        owner_alive: Callable[[], bool] | None = None,
    ) -> None:
        self.session = session
        self._release = release
        self.created_at = created_at
        self._owner_alive = owner_alive
        self.paused = False

    @property
    def sandbox_id(self) -> str:
        """Return the backend's id for this sandbox."""
        return self.session.ref.sandbox_id

    def idle_for(self, now: float) -> float | None:
        """Return how long this sandbox has been idle, or None when it is not.

        Idle is two conditions, not one. A command must not be in flight, and no
        command boundary may be more recent than the window: a backend that stamps
        activity when a command *returns* leaves the stamp stale for the whole of a
        long command, so age alone would read a running test suite as idle.

        A backend that reports no activity at all is never idle, because the policy
        cannot tell a quiet sandbox from an unreported one.
        """
        if self.session.busy:
            return None
        last = self.session.last_activity_at
        if last is None:
            return None
        return max(0.0, now - last)

    def age(self, now: float) -> float:
        """Return how long this sandbox has existed, idle or not."""
        return max(0.0, now - self.created_at)

    def owner_gone(self) -> bool:
        """Return whether nobody is coming back for this sandbox."""
        return self._owner_alive is not None and not self._owner_alive()

    def pause_mode(self) -> PauseMode:
        """Return the strongest pause the backend offers, preferring a resident freeze.

        A freeze keeps the sandbox on this host and is therefore cheaper to undo. A
        hibernation releases compute and is the only option a provider offers.
        """
        if self.session.capabilities.supports(SandboxFeature.FREEZE):
            return PauseMode.FREEZE
        return PauseMode.HIBERNATE

    async def pause(self) -> None:
        """Pause this sandbox, releasing compute while keeping its state."""
        await self.session.pause(self.pause_mode())
        self.paused = True

    async def release(self, reason: SandboxExitReason) -> None:
        """Destroy this sandbox and return what it held.

        The hook may be synchronous, because a node that tracks its own sandboxes
        can forget one without awaiting. A destruction that has started must finish
        even if this sweep is cancelled, or the node keeps memory it has already
        stopped accounting for.
        """
        result = self._release(reason)
        if inspect.isawaitable(result):
            await complete_cleanup(result)


class NodeReclaimLoop:
    """The node's single reclamation sweep.

    It owns no sandboxes. It reads the node's resident set through `residents`,
    so the component that tracks lifecycle stays the one source of truth and this
    loop cannot drift from it.
    """

    def __init__(
        self,
        residents: Callable[[], Sequence[ResidentSandbox]],
        windows: ReclaimWindows,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._residents = residents
        self.windows = windows
        self._clock = clock
        self._task: asyncio.Task[None] | None = None
        self._closed = False
        self._totals: dict[str, float] = {}

    async def sweep(self, *, now: float | None = None) -> ReclaimOutcome:
        """Decide once for every sandbox this node holds.

        One `switch` rather than several passes, so a sandbox is considered for
        exactly one reason per sweep and the reasons cannot race each other.
        """
        current = self._clock() if now is None else now
        outcome = ReclaimOutcome()
        for resident in list(self._residents()):
            try:
                await self._decide(resident, current, outcome)
            except asyncio.CancelledError:
                raise
            except Exception:
                outcome.failures += 1
                psrl_logger.warning(
                    "Sandbox reclamation failed for %s. Deciding again next sweep.",
                    resident.sandbox_id,
                    exc_info=True,
                )
        self._accumulate(outcome)
        return outcome

    async def _decide(self, resident: ResidentSandbox, now: float, outcome: ReclaimOutcome) -> None:
        """Apply the first reason that holds, strongest preservation first."""
        # An owner that is gone is not coming back, so its sandbox is reclaimed
        # whatever its idle state: waiting for the idle window would hold a node
        # slot for a caller that no longer exists.
        if resident.owner_gone():
            await resident.release(SandboxExitReason.RECLAIMED_ORPHAN)
            outcome.reclaimed_orphan.append(resident.sandbox_id)
            return
        # The lifetime is the backstop behind the idle windows: a sandbox that keeps
        # running commands is never idle and still has to end.
        if resident.age(now) >= self.windows.lifetime_s:
            await resident.release(SandboxExitReason.REAPED_LIFETIME)
            outcome.reaped_lifetime.append(resident.sandbox_id)
            return
        idle = resident.idle_for(now)
        if idle is None:
            outcome.skipped_busy += 1
            return
        if idle >= self.windows.reap_window_s:
            await resident.release(SandboxExitReason.REAPED_IDLE)
            outcome.reaped_idle.append(resident.sandbox_id)
            return
        if idle >= self.windows.pause_window_s and not resident.paused:
            try:
                await resident.pause()
            except (NotImplementedError, SandboxBusyError):
                # Not idle in fact, or no pause to offer. Either way the next sweep
                # decides again, and neither is a failure to report.
                return
            outcome.paused.append(resident.sandbox_id)

    def start(self) -> None:
        """Start the sweep, if there is a loop to run it on.

        The cadence comes from the timing contract rather than from a knob, because
        a sweep slower than the window it enforces reports an elapsed window late,
        and one much faster wakes the node for nothing.
        """
        if self._task is not None and not self._task.done():
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No loop means no sandbox this node could be running either.
            return
        self._closed = False
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while not self._closed:
            await asyncio.sleep(self.windows.sweep_interval_s)
            try:
                await self.sweep()
            except asyncio.CancelledError:
                raise
            except Exception:
                psrl_logger.warning("Sandbox reclamation sweep failed. Retrying next interval.", exc_info=True)

    async def stop(self) -> None:
        """Stop sweeping, leaving the sandboxes it was watching in place."""
        self._closed = True
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    def snapshot(self) -> dict[str, float]:
        """Return cumulative reclamation counters for the metric hook."""
        return dict(self._totals)

    def _accumulate(self, outcome: ReclaimOutcome) -> None:
        for key, value in outcome.as_dict().items():
            self._totals[key] = self._totals.get(key, 0.0) + value
