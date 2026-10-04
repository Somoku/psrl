"""The shipped SDK against a running service and a real Docker daemon.

Every layer is the real one: the Python SDK, the Go control plane and node, and
the node's own daemon. A test with a fake service would only prove the SDK
agrees with my model of it, and the wire format is exactly what a model gets
wrong.

Skipped unless `SANDBOXD_BINARY` names a built service and a Docker socket is
present, so the suite still runs where neither is.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import pytest
from sandboxd import (
    Feature,
    Resources,
    SandboxCapabilityError,
    SandboxClient,
    SandboxError,
    SandboxSpec,
    SandboxStatus,
    SnapshotKind,
    Source,
)

BINARY = os.environ.get("SANDBOXD_BINARY", "")
DOCKER_SOCKET = os.environ.get("DOCKER_SOCKET", "/var/run/docker.sock")
IMAGE = os.environ.get("SANDBOX_TEST_IMAGE", "alpine:latest")

pytestmark = pytest.mark.skipif(
    not BINARY or not Path(BINARY).exists() or not Path(DOCKER_SOCKET).exists(),
    reason="needs SANDBOXD_BINARY and a docker socket",
)


def service_config(directory: Path, fleet_memory_mb: int) -> Path:
    """Write a deployment that states intent once and derives the rest."""
    config = {
        "listen": f"unix://{directory}/sandboxd.sock",
        "node_id": "node-test",
        "owner_id": f"pysandbox-test-{int(time.time() * 1000)}",
        # Short, so a reclamation test does not wait a real episode.
        "timing": {"episode_deadline_s": 2, "node_ttl_s": 8, "rpc_timeout_s": 5},
        "fleet": {"memory_mb": fleet_memory_mb, "cpu_millis": 100000, "disk_mb": 100000},
        "node": {"memory_mb": fleet_memory_mb, "cpu_millis": 100000, "disk_mb": 100000},
        "classes": {"rollout": {"guaranteed_share": 0.7}, "grader": {"guaranteed_share": 0.2}},
        "backends": [{"type": "docker", "mode": "direct", "socket": DOCKER_SOCKET, "api_version": "v1.40"}],
        "default_backend": "docker",
    }
    path = directory / "sandboxd.json"
    path.write_text(json.dumps(config))
    return path


class Service:
    """A sandboxd process, for the life of one test."""

    def __init__(self, fleet_memory_mb: int = 4096) -> None:
        self._dir = tempfile.TemporaryDirectory(prefix="sandboxd-test-")
        self.root = Path(self._dir.name)
        self.socket = self.root / "sandboxd.sock"
        config = service_config(self.root, fleet_memory_mb)
        self.process = subprocess.Popen(
            [BINARY, "-config", str(config)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._await_socket()

    def _await_socket(self, timeout_s: float = 30.0) -> None:
        """Wait until the service is accepting, or report why it is not.

        Probing the socket rather than sleeping: a fixed sleep is either slow or
        flaky, and a service that died needs its own output in the failure.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"sandboxd exited {self.process.returncode}: {self.process.stdout.read()}")
            if self.socket.exists():
                probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    probe.connect(str(self.socket))
                    return
                except OSError:
                    pass
                finally:
                    probe.close()
            time.sleep(0.05)
        raise RuntimeError("sandboxd did not start listening")

    def close(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)
        self._dir.cleanup()


@pytest.fixture
def service():
    running = Service()
    yield running
    running.close()


@pytest.fixture
def small_service():
    running = Service(fleet_memory_mb=128)
    yield running
    running.close()


def client_for(running: Service) -> SandboxClient:
    return SandboxClient(f"unix://{running.socket}")


def spec(memory_mb: int = 64, **overrides) -> SandboxSpec:
    return SandboxSpec(
        source=Source.image(IMAGE),
        resources=Resources(memory_mb=memory_mb, cpu_count=0.25),
        resource_class="rollout",
        **overrides,
    )


@pytest.mark.asyncio
async def test_a_sandbox_round_trips_through_the_sdk(service):
    async with client_for(service) as client:
        async with await client.create(spec()) as sandbox:
            assert sandbox.node_id == "node-test"
            assert await sandbox.status() is SandboxStatus.RUNNING

            result = await sandbox.exec("echo sdk-reached-it")

            assert result.exit_code == 0
            assert "sdk-reached-it" in result.stdout


@pytest.mark.asyncio
async def test_a_failing_command_is_a_result_rather_than_an_error(service):
    # Reading a task failure as a transport fault would make a broken node and a
    # failing test indistinguishable.
    async with client_for(service) as client:
        async with await client.create(spec()) as sandbox:
            result = await sandbox.exec("exit 7")

            assert result.exit_code == 7


@pytest.mark.asyncio
async def test_a_command_carries_its_directory_and_environment(service):
    async with client_for(service) as client:
        async with await client.create(spec()) as sandbox:
            result = await sandbox.exec("pwd && echo $MARK", cwd="/tmp", env={"MARK": "carried"})

            assert "/tmp" in result.stdout
            assert "carried" in result.stdout


@pytest.mark.asyncio
async def test_files_round_trip_through_the_sandbox(service):
    async with client_for(service) as client:
        async with await client.create(spec()) as sandbox:
            payload = b"bytes with \x00 a null and \xff a high byte"

            await sandbox.write_bytes("/tmp/probe.bin", payload)

            assert await sandbox.read_bytes("/tmp/probe.bin") == payload


@pytest.mark.asyncio
async def test_the_context_manager_releases_the_sandbox(service):
    async with client_for(service) as client:
        async with await client.create(spec()) as sandbox:
            handle = sandbox.handle

        quota = await client.quota()
        assert quota.classes["rollout"].granted_memory_mb == 0, "the release must return the charge"
        assert handle.sandbox_id


@pytest.mark.asyncio
async def test_a_released_sandbox_reports_terminated_rather_than_failing(service):
    async with client_for(service) as client:
        sandbox = await client.create(spec())
        await sandbox.release()

        assert await sandbox.status() is SandboxStatus.TERMINATED


@pytest.mark.asyncio
async def test_releasing_twice_is_safe(service):
    # A retried release is normal: the first may have timed out after the
    # service already destroyed the sandbox.
    async with client_for(service) as client:
        sandbox = await client.create(spec())

        await sandbox.release()
        await sandbox.release()


@pytest.mark.asyncio
async def test_a_spec_no_backend_can_serve_is_refused_by_name(service):
    # Refused rather than degraded, and the refusal says what was missing.
    async with client_for(service) as client:
        unserveable = spec()
        unserveable = SandboxSpec(
            source=unserveable.source,
            resources=unserveable.resources,
            resource_class="rollout",
            required_features=frozenset({Feature.NATIVE_FORK}),
        )

        with pytest.raises(SandboxCapabilityError, match="native_fork"):
            await client.create(unserveable)


@pytest.mark.asyncio
async def test_a_full_fleet_refuses_rather_than_oversubscribing(small_service):
    async with client_for(small_service) as client:
        held = [await client.create(spec()) for _ in range(2)]
        try:
            with pytest.raises(SandboxError):
                await client.create(spec())
        finally:
            for sandbox in held:
                await sandbox.release()


@pytest.mark.asyncio
async def test_a_snapshot_captures_the_filesystem(service):
    async with client_for(service) as client:
        async with await client.create(spec()) as sandbox:
            await sandbox.exec("echo kept > /tmp/mark")

            snapshot = await sandbox.snapshot(SnapshotKind.FILESYSTEM)

            assert snapshot.snapshot_id
            await client.delete_snapshot(snapshot)


@pytest.mark.asyncio
async def test_pause_and_resume_keep_the_sandbox_usable(service):
    async with client_for(service) as client:
        async with await client.create(spec()) as sandbox:
            from sandboxd import PauseMode

            await sandbox.pause(PauseMode.FREEZE)
            assert await sandbox.status() is SandboxStatus.PAUSED

            await sandbox.resume()

            assert (await sandbox.exec("true")).exit_code == 0


@pytest.mark.asyncio
async def test_a_group_is_created_member_by_member(service):
    async with client_for(service) as client:
        group = await client.create_group([spec(), spec(), spec()])
        try:
            assert len(group) == 3
            assert len({member.sandbox_id for member in group}) == 3, "members must be distinct sandboxes"
        finally:
            await asyncio.gather(*(member.release() for member in group))


@pytest.mark.asyncio
async def test_a_failed_group_leaves_nothing_charged(small_service):
    # 128MB takes two members, so the third fails and the first two come back.
    async with client_for(small_service) as client:
        with pytest.raises(SandboxError):
            await client.create_group([spec(), spec(), spec()])

        quota = await client.quota()
        assert quota.classes["rollout"].granted_memory_mb == 0


@pytest.mark.asyncio
async def test_the_fleet_and_quota_reports_are_readable(service):
    # The planes report and never log, and the trainer's metric hook is the only
    # reader, so these answer even on an idle fleet.
    async with client_for(service) as client:
        fleet = await client.fleet()
        quota = await client.quota()

        assert len(fleet.nodes) == 1
        assert fleet.nodes[0].node_id == "node-test"
        assert "rollout" in quota.classes
        assert "sandbox/fleet/nodes" in fleet.as_metrics()
        assert "sandbox/quota/rollout/granted_memory_mb" in quota.as_metrics()


@pytest.mark.asyncio
async def test_concurrent_creates_stay_inside_the_fleet(small_service):
    # A 128MB fleet of 64MB sandboxes takes exactly two, whatever the interleaving.
    async with client_for(small_service) as client:
        results = await asyncio.gather(*(client.create(spec()) for _ in range(10)), return_exceptions=True)
        created = [r for r in results if not isinstance(r, BaseException)]
        try:
            assert len(created) == 2, f"admitted {len(created)} onto a two-sandbox fleet"
        finally:
            await asyncio.gather(*(sandbox.release() for sandbox in created))


@pytest.mark.asyncio
async def test_many_commands_run_on_one_sandbox(service):
    # The data-plane path is the hot one: an episode issues dozens of commands,
    # so it has to hold up under repetition rather than only once.
    async with client_for(service) as client:
        async with await client.create(spec()) as sandbox:
            for i in range(40):
                result = await sandbox.exec(f"echo turn-{i}")
                assert result.exit_code == 0
                assert f"turn-{i}" in result.stdout
