"""Reclamation sweep cost at node scale. Run: python tests/sandbox/bench_reclaim.py"""

import asyncio
import sys
import time

sys.path.insert(0, ".")
from psrl.sandbox.core import SandboxCapabilities, SandboxRef
from psrl.sandbox.node_reclaim import NodeReclaimLoop, ReclaimWindows, ResidentSandbox


class S:
    def __init__(self, i, busy, last):
        self._i = i
        self.busy = busy
        self.last_activity_at = last

    @property
    def ref(self):
        return SandboxRef("docker", f"sb-{self._i}")

    @property
    def capabilities(self):
        return SandboxCapabilities(frozenset())

    async def pause(self, mode):
        pass


async def main():
    W = ReclaimWindows(pause_window_s=10, reap_window_s=30, lifetime_s=100000, sweep_interval_s=1)
    for n in (800, 3200, 12800):
        # Production densities: DSec reports up to 3,200 containers per node.
        res = [ResidentSandbox(S(i, i % 4 == 0, 0.0), release=lambda r: None, created_at=0.0) for i in range(n)]
        loop = NodeReclaimLoop(lambda held=res: held, W)
        t = time.perf_counter()
        out = await loop.sweep(now=15)  # pause window reached, nothing reaped
        dt = time.perf_counter() - t
        print(
            f"  {n:6d} sandboxes -> sweep {dt * 1000:7.2f} ms  "
            f"({dt / n * 1e6:5.1f} us each)  paused={len(out.paused)} busy={out.skipped_busy}"
        )


asyncio.run(main())
