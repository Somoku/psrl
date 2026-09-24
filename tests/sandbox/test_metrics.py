"""Operation latencies and the image plane's metrics.

Two rules carry the suite: a percentile comes from the samples the plane actually
recorded, and a warm-start or locality ratio says whether the optimization is being
used rather than whether it was configured.
"""

from __future__ import annotations

import pytest
from psrl.sandbox.backends.docker import DockerBackend
from psrl.sandbox.core import ExecMode
from psrl.sandbox.metrics import SandboxMetrics

from tests.sandbox.test_docker_backend import FakeDockerEngine

pytestmark = pytest.mark.cpu_test


def test_a_percentile_comes_from_the_recorded_samples() -> None:
    metrics = SandboxMetrics()
    for duration in range(1, 101):
        metrics.record("pull_image", float(duration))

    pull = metrics.snapshot().operations["pull_image"]

    # Nearest rank: the 50th of 100 samples is the 50th, and the 95th is the 95th.
    assert pull.p50_seconds == 50.0
    assert pull.p95_seconds == 95.0
    assert pull.count == 100
    assert pull.mean_seconds == 50.5


def test_an_operation_with_no_latency_reports_a_zero_percentile() -> None:
    # A counted-only operation, such as an egress rule install, has no samples to rank.
    metrics = SandboxMetrics()
    metrics.count("egress_allowlist")

    assert metrics.snapshot().operations["egress_allowlist"].p95_seconds == 0.0


def test_the_sample_ring_is_bounded_so_a_long_run_cannot_grow_it() -> None:
    metrics = SandboxMetrics()
    for duration in range(5000):
        metrics.record("exec", float(duration))

    pull = metrics.snapshot().operations["exec"]

    assert pull.count == 5000
    # The quantiles describe the recent window, which is the one an operator acts on.
    assert pull.p50_seconds > 4000


def test_a_latency_outlier_moves_the_ceiling_but_not_the_median() -> None:
    metrics = SandboxMetrics()
    for _ in range(99):
        metrics.record("create", 1.0)
    metrics.record("create", 600.0)

    create = metrics.snapshot().operations["create"]

    assert create.max_seconds == 600.0
    assert create.p50_seconds == 1.0
    assert create.p95_seconds == 1.0


def _backend(engine, **overrides) -> DockerBackend:
    return DockerBackend(default_exec_mode=ExecMode.ONE_SHOT, engine=engine, **overrides)


async def test_the_image_plane_reports_pull_percentiles_and_a_warm_hit_ratio() -> None:
    engine = FakeDockerEngine()
    backend = _backend(engine, warm_pool={"enabled": True, "batch_size": 2, "depth_fraction": 0.5})
    backend.metrics.record("pull_image", 10.0)
    backend.metrics.record("pull_image", 30.0)
    backend.metrics.record("create", 5.0)
    backend.warm_pool._claimed = 1

    image = backend.image_snapshot()

    assert image["image/pull_s_p50"] == 10.0
    assert image["image/pull_s_p95"] == 30.0
    # One claim against one cold create: half of the starts were warm.
    assert image["image/warm_pool_hit_ratio"] == 0.5


async def test_a_deployment_that_never_creates_reports_a_zero_hit_ratio() -> None:
    # A ratio over no attempts is zero rather than a division, and zero is the honest
    # reading: nothing has been warmed and nothing has been created either.
    backend = _backend(FakeDockerEngine())

    image = backend.image_snapshot()

    assert image["image/warm_pool_hit_ratio"] == 0.0
    assert image["image/prefetch_coverage"] == 0.0


async def test_prefetch_coverage_accumulates_over_the_run() -> None:
    # Coverage is only meaningful against the whole working set a run asked for, so it
    # accumulates rather than describing one call.
    backend = _backend(FakeDockerEngine())

    backend.note_prefetch(requested=4, warmed=3)
    backend.note_prefetch(requested=4, warmed=4)

    assert backend.image_snapshot()["image/prefetch_coverage"] == 7 / 8


async def test_the_manager_reports_the_image_plane() -> None:
    from psrl.sandbox import SandboxManager

    backend = _backend(FakeDockerEngine())
    manager = SandboxManager({"docker": backend}, "docker")
    backend.metrics.record("pull_image", 12.0)
    backend.note_prefetch(requested=1, warmed=1)

    snapshot = manager.snapshot()

    assert snapshot["image/pull_s_p50"] == 12.0
    assert snapshot["image/prefetch_coverage"] == 1.0
    await manager.shutdown()


async def test_the_image_plane_reports_the_whole_family() -> None:
    # A guard against a family that is documented and half wired: the trainer hook
    # reads these four names, so they all have to exist even before any pull happens.
    backend = _backend(FakeDockerEngine())

    assert set(backend.image_snapshot()) == {
        "image/pull_s_p50",
        "image/pull_s_p95",
        "image/warm_pool_hit_ratio",
        "image/prefetch_coverage",
    }
    await backend.shutdown()
