"""OOM detection, container release, and OOM priority for Docker sandboxes.

A container the kernel OOM-kills leaves its exec stream open forever, so the stop has to
be detected rather than waited out, and it must be reported as a resource fault.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest
from psrl.sandbox import ResourceSpec, SandboxOomError, SandboxSource, SandboxSpec
from psrl.sandbox.backends.docker import DockerBackend, DockerSecurityConfig
from psrl.sandbox.backends.docker_engine import DockerEngineError

from tests.sandbox.test_docker_backend import FakeDockerEngine

pytestmark = pytest.mark.cpu_test

_RUNNING = {"State": {"Running": True, "OOMKilled": False}}
_OOM_KILLED = {"State": {"Running": False, "OOMKilled": True, "ExitCode": 137}}
_STOPPED = {"State": {"Running": False, "OOMKilled": False, "ExitCode": 0}}


def _spec(memory_mb: int | None = 8192) -> SandboxSpec:
    resources = ResourceSpec(memory_mb=memory_mb, cpu_count=2.0) if memory_mb else ResourceSpec()
    return SandboxSpec(SandboxSource.image("image"), resources=resources)


async def _session(engine: FakeDockerEngine, backend: DockerBackend, memory_mb: int | None = 8192):
    return await backend.create(_spec(memory_mb))


class TestOomScoreAdj:
    """A sandbox must not outrank the training processes it shares a node with."""

    async def test_the_default_leaves_host_oom_priorities_alone(self):
        engine = FakeDockerEngine()
        await _session(engine, DockerBackend(engine=engine))

        assert "OomScoreAdj" not in engine.config["HostConfig"]

    async def test_deprioritizing_the_sandbox_is_opt_in(self):
        engine = FakeDockerEngine()
        backend = DockerBackend(engine=engine, security=DockerSecurityConfig(oom_score_adj=-500))
        await _session(engine, backend)

        assert engine.config["HostConfig"]["OomScoreAdj"] == -500

    async def test_a_policy_override_wins(self):
        engine = FakeDockerEngine()
        backend = DockerBackend(
            engine=engine,
            policy_profiles={"mini_swe": {"oom_score_adj": 250}},
        )
        await backend.create(SandboxSpec(SandboxSource.image("image"), policy_profile="mini_swe"))

        assert engine.config["HostConfig"]["OomScoreAdj"] == 250

    def test_an_out_of_range_value_is_rejected(self):
        with pytest.raises(ValueError, match="oom_score_adj"):
            DockerSecurityConfig(oom_score_adj=-2000)


class TestOomDetection:
    """A sandbox that dies mid-command must be reported and released."""

    async def test_an_oom_killed_container_raises_a_sandbox_oom_error(self, monkeypatch):
        engine = FakeDockerEngine()
        session = await _session(engine, DockerBackend(engine=engine))
        engine.exec = AsyncMock(side_effect=DockerEngineError(500, "exec stream died"))
        monkeypatch.setattr(engine, "inspect_container", AsyncMock(return_value=_OOM_KILLED))

        with pytest.raises(SandboxOomError, match="OOM-killed"):
            await session.exec("work")

        assert engine.removes >= 1, "A dead sandbox must be released, not left behind."

    async def test_the_error_names_the_memory_limit(self, monkeypatch):
        engine = FakeDockerEngine()
        session = await _session(engine, DockerBackend(engine=engine), memory_mb=16384)
        engine.exec = AsyncMock(side_effect=DockerEngineError(500, "exec stream died"))
        monkeypatch.setattr(engine, "inspect_container", AsyncMock(return_value=_OOM_KILLED))

        with pytest.raises(SandboxOomError, match="memory_limit_mb=16384"):
            await session.exec("work")

    async def test_an_ordinary_stop_is_reported_as_a_stop(self, monkeypatch):
        engine = FakeDockerEngine()
        session = await _session(engine, DockerBackend(engine=engine))
        engine.exec = AsyncMock(side_effect=DockerEngineError(500, "exec stream died"))
        monkeypatch.setattr(engine, "inspect_container", AsyncMock(return_value=_STOPPED))

        with pytest.raises(RuntimeError, match="stopped \\(exited\\)"):
            await session.exec("work")

    async def test_a_healthy_container_still_raises_the_original_failure(self, monkeypatch):
        # The container is alive, so the stream failure is the real story and must
        # not be rewritten as a container fault.
        engine = FakeDockerEngine()
        session = await _session(engine, DockerBackend(engine=engine))
        engine.exec = AsyncMock(side_effect=DockerEngineError(500, "boom"))
        monkeypatch.setattr(engine, "inspect_container", AsyncMock(return_value=_RUNNING))

        with pytest.raises(DockerEngineError, match="boom"):
            await session.exec("work")


class TestContainerWatcher:
    """A command must not outlive its container."""

    async def test_a_command_is_aborted_when_its_container_dies(self, monkeypatch):
        engine = FakeDockerEngine()
        backend = DockerBackend(engine=engine, container_watch_interval_s=0.01)
        session = await _session(engine, backend)

        died = {"value": False}

        async def hangs(container_id, command, *, cwd=None, env=None, timeout_s=None):
            died["value"] = True
            await asyncio.sleep(30)

        async def inspect(container_id):
            return _OOM_KILLED if died["value"] else _RUNNING

        engine.exec = hangs
        monkeypatch.setattr(engine, "inspect_container", inspect)

        with pytest.raises(SandboxOomError, match="OOM-killed"):
            await asyncio.wait_for(session.exec("work", timeout_s=30), timeout=5)

        assert engine.removes >= 1

    async def test_a_command_that_ends_normally_is_untouched(self, monkeypatch):
        engine = FakeDockerEngine()
        backend = DockerBackend(engine=engine, container_watch_interval_s=0.01)
        session = await _session(engine, backend)
        engine.exec = AsyncMock(return_value=(0, b"ok", b"", False))
        monkeypatch.setattr(engine, "inspect_container", AsyncMock(return_value=_RUNNING))

        result = await session.exec("work", timeout_s=5)

        assert result.stdout == "ok"
        assert engine.removes == 0
