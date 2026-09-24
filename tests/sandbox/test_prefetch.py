"""The per-run prefetch plan and the step that executes it.

Two rules carry the suite: the plan is the run's working set rather than the corpus,
and a reference that could not be warmed is reported rather than raised, because a
task whose image missed the prefetch still runs.
"""

from __future__ import annotations

import pytest
from psrl.sandbox import PrefetchPlan, PrefetchReport, SandboxManager, SandboxSource, SandboxSpec
from psrl.sandbox.backends.docker import DockerBackend
from psrl.sandbox.core import ExecMode

from tests.sandbox.test_docker_backend import FakeDockerEngine
from tests.sandbox.test_manager import FakeBackend

pytestmark = pytest.mark.cpu_test


def _spec(reference: str = "python:3.11") -> SandboxSpec:
    return SandboxSpec(SandboxSource.image(reference))


def _backend(engine) -> DockerBackend:
    return DockerBackend(default_exec_mode=ExecMode.ONE_SHOT, engine=engine)


def test_the_plan_is_the_working_set_deduplicated_by_reference() -> None:
    # A task set rarely visits every image, and warming one image twice would pull it
    # twice, so the plan is the deduplicated working set of the run.
    plan = PrefetchPlan.from_specs([_spec("python:3.11"), _spec("python:3.11"), _spec("task-a:latest")])

    assert plan.references == ("python:3.11", "task-a:latest")
    assert plan.size == 2


def test_a_source_that_is_not_an_image_has_nothing_to_prefetch() -> None:
    # A template or a snapshot is materialized by its own backend, so a plan that named
    # one would warm the wrong thing.
    plan = PrefetchPlan.from_specs([SandboxSpec(SandboxSource.template("task-template"))])

    assert plan.references == ()


def test_an_empty_plan_reports_no_coverage_rather_than_dividing_by_zero() -> None:
    assert PrefetchReport().coverage == 0.0


async def test_prefetch_warms_every_reference_in_the_plan() -> None:
    engine = FakeDockerEngine()
    backend = _backend(engine)

    warmed = await backend.prefetch_images(["python:3.11", "task-a:latest"])

    assert warmed == 2
    assert sorted(engine.pulled) == ["python:3.11", "task-a:latest"]
    assert backend.image_snapshot()["image/prefetch_coverage"] == 1.0


async def test_a_reference_that_cannot_be_pulled_is_counted_rather_than_raised() -> None:
    # A prefetch is an optimization, so one unreachable image must not fail the step
    # that was meant to make the rollout faster.
    engine = FakeDockerEngine()
    backend = _backend(engine)

    async def refuse(reference: str, auth=None) -> None:
        raise OSError("registry unreachable")

    engine.pull_image = refuse  # type: ignore[assignment]

    warmed = await backend.prefetch_images(["python:3.11", "task-a:latest"])

    assert warmed == 0
    assert backend.image_snapshot()["image/prefetch_coverage"] == 0.0


async def test_prefetch_refuses_a_concurrency_it_cannot_honor() -> None:
    with pytest.raises(ValueError, match="concurrency"):
        await _backend(FakeDockerEngine()).prefetch_images(["a:1"], concurrency=0)


async def test_the_manager_reports_what_the_prefetch_step_achieved() -> None:
    engine = FakeDockerEngine()
    backend = _backend(engine)
    manager = SandboxManager({"docker": backend}, "docker")

    report = await manager.prefetch(PrefetchPlan.from_specs([_spec("a:1"), _spec("b:1")]))

    assert report.requested == 2
    assert report.warmed == 2
    assert report.coverage == 1.0
    assert manager.snapshot()["image/prefetch_coverage"] == 1.0
    await manager.shutdown()


async def test_a_backend_that_materializes_lazily_reports_zero_coverage() -> None:
    # A provider loads layers on demand, so there is nothing to warm and the honest
    # report is that none of the working set was pre-materialized.
    manager = SandboxManager({"fake": FakeBackend(set())}, "fake")

    report = await manager.prefetch(PrefetchPlan(("a:1",)))

    assert report == PrefetchReport(requested=1, warmed=0)
    await manager.shutdown()


async def test_an_empty_plan_asks_no_backend_for_anything() -> None:
    manager = SandboxManager({"fake": FakeBackend(set())}, "fake")

    assert await manager.prefetch(PrefetchPlan()) == PrefetchReport()
    await manager.shutdown()
