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


class EpisodeBudget:
    """A wall-clock budget that can start counting after resource provisioning.

    A loop that provisions a sandbox constructs this with
    ``starts_after_provisioning=True`` and calls `arm` once its sandbox and
    environment are ready. Until then `wait_for` does not bound the task at all:
    admission and provisioning are bounded by the sandbox service's own liveness, not
    by a deadline here, so a slow pull cannot be mistaken for a wedged episode. Any
    other loop arms on its first `wait_for`, which keeps its previous behaviour of
    timing the whole call.

    ``total_s`` of ``None`` disables enforcement entirely.
    """

    def __init__(
        self,
        total_s: float | None,
        *,
        starts_after_provisioning: bool = False,
    ) -> None:
        self.total_s = total_s
        self._starts_after_provisioning = starts_after_provisioning
        self._created_at = time.monotonic()
        self._armed_at: float | None = None
        self._wait_s = 0.0
        self._armed = asyncio.Event()
        # Set by the episode once its own work is done, so the phases that follow it --
        # grading in particular -- run under their own budget instead of this one.
        self._released = asyncio.Event()

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

    def release(self) -> None:
        """Stop enforcing the budget now that the episode's own work is done.

        The episode ends when its work ends, not when the whole ``run`` returns:
        patch collection is episode work, but grading that follows it is a separate
        phase with its own budget. Releasing here is what keeps a slow grader from
        being cut by, or charged to, the episode clock.
        """
        self._released.set()

    @property
    def released(self) -> bool:
        """Return whether the episode has declared its own work finished."""
        return self._released.is_set()

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

    async def wait_for(self, task: asyncio.Task[_T]) -> _T:
        """Await `task` under the budget.

        Returns:
            The task's result.

        Raises:
            EpisodeBudgetExpired: When the episode budget expires after the episode has
                started and before it releases the budget.
        """
        if not self._starts_after_provisioning:
            self.arm()
        if self.total_s is None:
            return await task
        while True:
            if self._released.is_set():
                # The episode's timed work is done. Its tail -- grading -- runs under
                # its own budget, so this one no longer bounds the task.
                return await task
            remaining = self.remaining_s()
            released = asyncio.ensure_future(self._released.wait())
            try:
                if remaining is None:
                    # The episode has not started, so provisioning is still in flight. That
                    # phase is not charged to the episode and carries no deadline here, so
                    # wait for the arming that starts the clock, for a release, or for the
                    # task itself.
                    armed = asyncio.ensure_future(self._armed.wait())
                    try:
                        done, _ = await asyncio.wait(
                            {task, armed, released}, return_when=asyncio.FIRST_COMPLETED
                        )
                    finally:
                        armed.cancel()
                else:
                    done, _ = await asyncio.wait(
                        {task, released}, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
                    )
            finally:
                released.cancel()
            if task in done:
                return task.result()
            if self._released.is_set():
                return await task
            if remaining is None:
                continue
            raise EpisodeBudgetExpired(
                f"Episode budget of {self.total_s}s expired "
                f"(waited {self.wait_s():.0f}s before the episode started)."
            )
