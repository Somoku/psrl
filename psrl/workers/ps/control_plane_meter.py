"""Counting rollout -> PS control-plane round trips (implementation item C2).

C2's criterion is that a steady-state weight sync costs **zero** rollout -> PS RPCs:
the plan is compiled at handshake and cached locally, so only a holder-set change should
require the index. "Zero" is not a claim that can be made by reading the code -- the
calls are spread across the pull path, the shared-pull lock, and the version read -- so
it needs to be counted.

The meter is deliberately on the *caller* side. Counting arrivals at the PS actor would
miss nothing, but it also could not distinguish a steady-state sync from a handshake or
a retry, and those are exactly what the criterion excludes. A rollout-side meter can be
armed around one sync and report what that sync cost.

The design's three measures (3.2.1) in decreasing order of value -- eliminate, reduce
frequency, speed up -- have different signatures here, which the breakdown preserves:
eliminating a call removes its name from the tally entirely, whereas making it faster
leaves the count unchanged. Reporting only a total would hide that difference.
"""

from __future__ import annotations

import threading
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class ControlPlaneMeter:
    """Counts control-plane calls by name, with an arm/disarm window.

    Thread-safe because the rollout side issues these from asyncio callbacks and from
    the engine's collective-RPC threads, and an undercount would make the criterion
    look satisfied when it is not -- the one direction of error that matters here.
    """

    counts: Counter = field(default_factory=Counter)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _armed: bool = False

    def record(self, name: str) -> None:
        with self._lock:
            if self._armed:
                self.counts[name] += 1

    @property
    def total(self) -> int:
        with self._lock:
            return sum(self.counts.values())

    def breakdown(self) -> dict[str, int]:
        with self._lock:
            return dict(self.counts)

    def reset(self) -> None:
        with self._lock:
            self.counts.clear()

    @contextmanager
    def measuring(self):
        """Count calls made inside the block, starting from zero.

        Nested use is not supported and not needed: a sync is the unit being measured,
        and nesting would silently attribute an inner window's calls to the outer one.
        """
        with self._lock:
            if self._armed:
                raise RuntimeError("ControlPlaneMeter.measuring() does not nest")
            self.counts.clear()
            self._armed = True
        try:
            yield self
        finally:
            with self._lock:
                self._armed = False

    def assert_steady_state_is_free(self, allowed: int = 0) -> None:
        """Raise unless the measured window stayed within ``allowed`` round trips.

        Names the offending calls rather than reporting a count, because the useful
        question after a regression is *which* call came back, not how many did.
        """
        total = self.total
        if total > allowed:
            detail = ", ".join(f"{name}={count}" for name, count in sorted(self.counts.items()))
            raise AssertionError(
                f"steady-state sync made {total} control-plane round trip(s), "
                f"budget is {allowed}: {detail}"
            )


#: Process-wide meter. The rollout worker is one process per engine, so a module-level
#: instance is per-engine, which is the granularity the criterion is stated at.
GLOBAL_CONTROL_PLANE_METER = ControlPlaneMeter()
