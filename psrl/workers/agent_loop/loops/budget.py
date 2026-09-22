"""Wall-clock budget for one rollout episode.

An episode is more than the work it performs. A sandboxed harness first has to be
admitted by node capacity and prepared, and that wait is scheduling latency rather
than model or harness time. Charging it to the episode budget makes the budget fire
on a healthy episode whenever the node is busy, so the clock here starts when the
loop says the episode itself has begun.
"""

from __future__ import annotations

import asyncio
import time
from typing import TypeVar

_T = TypeVar("_T")


class EpisodeBudgetExpired(Exception):
    """Raised by `EpisodeBudget.wait_for` when the episode budget runs out.

    Deliberately not a `TimeoutError`: an episode is free to raise one of those from
    inside its own work, and the two must stay distinguishable at the call site.
    """


class RolloutDeadlineExceeded(Exception):
    """Raised when a rollout exceeds its deadline before the episode has even started.

    The episode clock is not running yet, so this is not a budget expiry: the child spent
    its whole allowance being admitted and provisioned and never began. Reported separately
    because the fix is a capacity or setup setting rather than a longer episode.
    """


class EpisodeBudget:
    """A wall-clock budget that can start counting after resource provisioning.

    A loop that provisions a sandbox constructs this with
    ``starts_after_provisioning=True`` and calls `arm` once its sandbox and
    environment are ready. Until then `wait_for` bounds the task by `setup_limit_s`
    instead, because the wait is bounded by the admission deadline and the backend
    timeouts rather than by the episode. Any other loop arms on its first `wait_for`,
    which keeps its previous behaviour of timing the whole call.

    ``total_s`` of ``None`` disables enforcement entirely.

    ``setup_limit_s`` bounds the pre-episode phase. Without it, a loop that never arms has
    nothing bounding it, and a wedged provisioning step would hold the sandbox until the
    manager's stall watchdog noticed the silence.
    """

    def __init__(
        self,
        total_s: float | None,
        *,
        starts_after_provisioning: bool = False,
        setup_limit_s: float | None = None,
    ) -> None:
        self.total_s = total_s
        self.setup_limit_s = setup_limit_s
        self._starts_after_provisioning = starts_after_provisioning
        self._created_at = time.monotonic()
        self._armed_at: float | None = None
        self._wait_s = 0.0
        self._armed = asyncio.Event()

    @property
    def armed(self) -> bool:
        """Return whether the episode clock has started."""
        return self._armed_at is not None

    def arm(self) -> None:
        """Start the episode clock, recording how long provisioning took."""
        if self._armed_at is not None:
            return
        now = time.monotonic()
        self._armed_at = now
        self._wait_s = now - self._created_at
        self._armed.set()

    def wait_s(self) -> float:
        """Return the wall-clock time spent before the episode started."""
        if self._armed_at is not None:
            return self._wait_s
        return time.monotonic() - self._created_at

    def remaining_s(self) -> float | None:
        """Return the budget left, or ``None`` while the episode has not started."""
        if self.total_s is None or self._armed_at is None:
            return None
        return max(0.0, self.total_s - (time.monotonic() - self._armed_at))

    def setup_remaining_s(self) -> float | None:
        """Return the provisioning allowance left, or ``None`` when unbounded."""
        if self.setup_limit_s is None or self._armed_at is not None:
            return None
        return max(0.0, self.setup_limit_s - (time.monotonic() - self._created_at))

    async def wait_for(self, task: asyncio.Task[_T]) -> _T:
        """Await `task` under the budget.

        Returns:
            The task's result.

        Raises:
            EpisodeBudgetExpired: When the budget expires after the episode has started,
                or when an unprovisioned loop exceeds it.
            RolloutDeadlineExceeded: When the task is still provisioning after
                `setup_limit_s`.
        """
        if not self._starts_after_provisioning:
            self.arm()
        if self.total_s is None and self.setup_limit_s is None:
            return await task
        while True:
            remaining = self.remaining_s()
            if remaining is None:
                # Provisioning is still in progress. Wait for either the episode or the
                # arming that starts the episode clock, bounded by the setup allowance.
                setup_remaining = self.setup_remaining_s()
                if setup_remaining is not None and setup_remaining <= 0:
                    raise RolloutDeadlineExceeded(
                        f"Rollout spent its whole {self.setup_limit_s:g}s setup allowance without starting "
                        f"an episode. The sandbox was never made ready, so no agent work happened."
                    )
                armed = asyncio.ensure_future(self._armed.wait())
                try:
                    done, _ = await asyncio.wait(
                        {task, armed},
                        timeout=setup_remaining,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    armed.cancel()
                if task in done:
                    return task.result()
                if not done:
                    raise RolloutDeadlineExceeded(
                        f"Rollout spent its whole {self.setup_limit_s:g}s setup allowance without starting "
                        f"an episode. The sandbox was never made ready, so no agent work happened."
                    )
                continue
            done, _ = await asyncio.wait({task}, timeout=remaining)
            if task in done:
                return task.result()
            raise EpisodeBudgetExpired(
                f"Episode budget of {self.total_s}s expired "
                f"(waited {self.wait_s():.0f}s before the episode started)."
            )
