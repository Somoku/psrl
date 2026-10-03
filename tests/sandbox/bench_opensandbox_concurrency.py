"""Concurrency stress test for the OpenSandbox backend via the p3b service.

The two scheduling modes put a different amount of software in the create path,
and that is what this measures.

direct mode
    p3b drives the container runtime itself and stages OpenSandbox's agent into
    each sandbox. No OpenSandbox control plane is deployed. Creates are bounded
    only by p3b's own semaphore and the daemon, so concurrency is real.

provider mode
    p3b sends every request to an OpenSandbox gateway and its scheduler places
    the sandbox. p3b does cross-backend quota only.

What the two modes share is the data plane: the agent is the same Go binary, it
listens on the same port, and the SDK reaches it directly either way. So exec
latency should be flat across modes, while create latency should not -- that
asymmetry is the result worth reading.

Both shapes are driven through the same sandboxd SDK, so the numbers include
everything a caller actually pays.

Usage, against whichever mode the service is configured for:

    SANDBOXD_ENDPOINT=unix:///run/sandboxd.sock \\
        python bench_opensandbox_concurrency.py

Label the run so the output says which shape produced it:

    SANDBOX_MODE=direct CONCURRENCIES=8,16,32,64,128 \\
        python bench_opensandbox_concurrency.py
"""
from __future__ import annotations

import asyncio
import os
import statistics
import sys
import time
from typing import NamedTuple

from sandboxd import SandboxClient, SandboxSpec, Source, Resources
from sandboxd.execd import execd_agent_factory

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

ENDPOINT       = os.environ.get("SANDBOXD_ENDPOINT", "unix:///run/sandboxd.sock")
# A label for the output only. The service's own configuration decides which mode
# is actually in use; naming it here is how a saved run says what it measured.
SANDBOX_MODE   = os.environ.get("SANDBOX_MODE", "unspecified")
SANDBOX_IMAGE  = os.environ.get("SANDBOX_IMAGE", "alpine:latest")
SANDBOX_MEMORY = int(os.environ.get("SANDBOX_MEMORY_MB", "256"))
RESOURCE_CLASS = os.environ.get("RESOURCE_CLASS", "rollout")
CONCURRENCIES  = [int(x) for x in os.environ.get("CONCURRENCIES", "8,16,32,64").split(",")]


class Result(NamedTuple):
    idx: int
    create_s: float
    exec_s: float
    release_s: float
    output: str


async def one_sandbox(client: SandboxClient, idx: int) -> Result:
    """Create, exec one command, release. Measures each phase separately."""
    spec = SandboxSpec(
        source=Source.image(SANDBOX_IMAGE),
        resources=Resources(memory_mb=SANDBOX_MEMORY, cpu_count=1),
        resource_class=RESOURCE_CLASS,
        # Pin the backend so p3b routes this to opensandbox regardless of the
        # default_backend setting in the service config.
        backend="opensandbox",
    )

    t0 = time.monotonic()
    sb = await client.create(spec)
    create_s = time.monotonic() - t0

    t1 = time.monotonic()
    result = await sb.exec(f"echo opensandbox-{idx}")
    exec_s = time.monotonic() - t1

    t2 = time.monotonic()
    await sb.release()
    release_s = time.monotonic() - t2

    return Result(idx, create_s, exec_s, release_s, result.stdout.strip())


async def burst(n: int) -> dict:
    """Launch N sandboxes concurrently and collect per-phase latencies."""
    # The agent factory is what lets the SDK reach execd inside each sandbox.
    # Without it the SDK has no protocol for the agent and the node refuses to
    # proxy, because an opensandbox sandbox runs its own.
    agents = execd_agent_factory()
    async with SandboxClient(ENDPOINT, agent_factory=agents) as client:
        t0 = time.monotonic()
        tasks = [asyncio.create_task(one_sandbox(client, i)) for i in range(n)]
        raw = await asyncio.gather(*tasks, return_exceptions=True)
        total_s = time.monotonic() - t0
    # The agent pool is separate from the client's, so it is closed explicitly.
    await agents.close()

    errors = [r for r in raw if isinstance(r, Exception)]
    ok: list[Result] = [r for r in raw if isinstance(r, Result)]

    def _pct(times: list[float], p: float) -> float | None:
        if not times:
            return None
        return round(sorted(times)[int(len(times) * p)] * 1000, 1)

    creates  = [r.create_s  for r in ok]
    execs    = [r.exec_s    for r in ok]
    releases = [r.release_s for r in ok]

    return {
        "n":           n,
        "succeeded":   len(ok),
        "failed":      len(errors),
        "errors":      [str(e) for e in errors[:3]],
        "total_wall_s": round(total_s, 3),
        "throughput":  round(len(ok) / total_s, 2) if total_s > 0 else 0,
        # create latency is the primary signal — it covers admission + Docker
        # container start + execd sidecar readiness.
        "create_p50":  _pct(creates, 0.50),
        "create_p95":  _pct(creates, 0.95),
        "create_max":  round(max(creates) * 1000, 1) if creates else None,
        # exec latency isolates the data plane: SDK → execd inside the sandbox.
        "exec_p50":    _pct(execs, 0.50),
        "exec_p95":    _pct(execs, 0.95),
        # release latency is usually cheap but spikes under cgroup pressure.
        "release_p50": _pct(releases, 0.50),
        "release_p95": _pct(releases, 0.95),
    }


def _hdr(label: str, value: str) -> None:
    print(f"  {label:<28} {value}")


async def main() -> None:
    print()
    print("=== OpenSandbox Concurrency Benchmark ===")
    _hdr("Endpoint:",          ENDPOINT)
    _hdr("Scheduling mode:",   SANDBOX_MODE)
    _hdr("Image:",             SANDBOX_IMAGE)
    _hdr("Memory per sandbox:", f"{SANDBOX_MEMORY} MB")
    _hdr("Resource class:",    RESOURCE_CLASS)
    print()

    # Warm-up: one sandbox to absorb first-connect latency and image pull.
    print("Warming up (1 sandbox) …", flush=True)
    warm = await burst(1)
    if warm["failed"]:
        print(f"  Warm-up failed: {warm['errors']}")
        print()
        print("Cannot continue: check that the p3b service is running with an")
        print("opensandbox backend and that the image is reachable.")
        sys.exit(1)
    print(f"  create={warm['create_p50']} ms  exec={warm['exec_p50']} ms  "
          f"release={warm['release_p50']} ms")
    print()

    all_results = []
    for n in CONCURRENCIES:
        print(f"--- {n} concurrent sandboxes @ {SANDBOX_MEMORY} MB ---", flush=True)
        r = await burst(n)
        all_results.append(r)
        print(f"  succeeded:           {r['succeeded']}/{r['n']}")
        print(f"  total wall:          {r['total_wall_s']} s")
        print(f"  throughput:          {r['throughput']} sandboxes/s")
        print(f"  create  p50/p95/max: {r['create_p50']} / {r['create_p95']} / {r['create_max']} ms")
        print(f"  exec    p50/p95:     {r['exec_p50']} / {r['exec_p95']} ms")
        print(f"  release p50/p95:     {r['release_p50']} / {r['release_p95']} ms")
        if r["failed"]:
            print(f"  ERRORS ({r['failed']}): {r['errors']}")
        print()

    # Summary table
    print("=== Summary ===")
    print(f"{'N':>6}  {'ok':>5}  {'wall_s':>7}  {'tput/s':>7}  "
          f"{'cr_p50':>8}  {'cr_p95':>8}  {'ex_p50':>8}")
    for r in all_results:
        print(f"{r['n']:>6}  {r['succeeded']:>5}  {r['total_wall_s']:>7.2f}  "
              f"{r['throughput']:>7.2f}  "
              f"{str(r['create_p50']):>8}  {str(r['create_p95']):>8}  "
              f"{str(r['exec_p50']):>8}")

    print()
    _print_analysis(all_results)


def _print_analysis(results: list[dict]) -> None:
    """Print a plain-language read of the numbers."""
    ok_runs = [r for r in results if r["succeeded"] == r["n"]]
    if not ok_runs:
        print("All runs had failures — check capacity and image availability.")
        return

    best_tput = max(ok_runs, key=lambda r: r["throughput"])
    print(f"Peak throughput: {best_tput['throughput']} sandboxes/s "
          f"at n={best_tput['n']}")

    # Detect create latency degradation under concurrency
    creates_by_n = [(r["n"], r["create_p50"]) for r in ok_runs if r["create_p50"]]
    if len(creates_by_n) >= 2:
        n_low, c_low   = creates_by_n[0]
        n_high, c_high = creates_by_n[-1]
        ratio = c_high / c_low if c_low else 0
        if ratio > 3:
            print(f"  Create latency scaled poorly: {c_low} ms at n={n_low} → "
                  f"{c_high} ms at n={n_high} ({ratio:.1f}×).")
            print("  This is typical under cgroup v1 serialization or a")
            print("  single docker daemon; consider max_create_concurrency in")
            print("  the opensandbox-server config or network_mode: host.")
        else:
            print(f"  Create latency scaled well: {c_low} ms at n={n_low} → "
                  f"{c_high} ms at n={n_high} ({ratio:.1f}×).")

    # Compare exec latency: this should be near-flat across concurrencies
    execs_by_n = [(r["n"], r["exec_p50"]) for r in ok_runs if r["exec_p50"]]
    if len(execs_by_n) >= 2:
        _, e_low  = execs_by_n[0]
        _, e_high = execs_by_n[-1]
        eratio = e_high / e_low if e_low else 0
        if eratio > 2:
            print(f"  Exec latency degraded {eratio:.1f}× under concurrency —")
            print("  execd or the sandbox network path may be a bottleneck.")

    print()
    print("Interpretation guide:")
    print("  create_p50  = admission + container start + execd sidecar ready")
    print("  exec_p50    = SDK → execd inside sandbox (data plane only)")
    print("  release_p50 = DELETE /sandboxes/{id} round trip")
    print()
    print("To compare with the docker backend, run bench_p3b_concurrency.py")
    print("with the same SANDBOXD_ENDPOINT and CONCURRENCIES.")


if __name__ == "__main__":
    asyncio.run(main())
