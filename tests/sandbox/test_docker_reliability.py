"""Regression coverage for Docker preparation and failure cleanup."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import pytest
from psrl.sandbox import (
    SandboxManager,
    SandboxOomError,
    SandboxSource,
    SandboxSpec,
    SandboxStatus,
    SnapshotKind,
    SnapshotRef,
)
from psrl.sandbox.backends.docker import DockerBackend, cli
from psrl.sandbox.backends.docker.engine import DockerEngineClient, DockerEngineError
from psrl.sandbox.core import ExecMode, SandboxCommandTimeout, SandboxProvisionError

from tests.sandbox.test_docker_backend import FakeDockerEngine
from tests.sandbox.test_docker_exec import FakeShell

pytestmark = pytest.mark.cpu_test


class StreamResponse:
    """
    Model a Docker response stream without requiring a daemon or socket.
    """

    status = 200
    headers = {}

    def __init__(self, chunks):
        self.chunks = chunks
        self.content = self
        self.closed = False
        self.chunks_read = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    def __aiter__(self):
        return self.iter_chunked(65536)

    async def iter_chunked(self, size):
        for chunk in self.chunks:
            self.chunks_read += 1
            yield chunk


@pytest.mark.parametrize("field", ["error", "errorDetail"])
async def test_pull_rejects_errors_in_successful_http_stream(monkeypatch, field) -> None:
    error = b'{"error":"denied"}\n' if field == "error" else b'{"errorDetail":{"message":"denied"}}\n'
    response = StreamResponse([b'{"status":"Pulling"}\n', error])
    engine = DockerEngineClient()
    monkeypatch.setattr(engine, "_get_session", AsyncMock(return_value=SimpleNamespace(post=lambda *a, **k: response)))

    with pytest.raises(DockerEngineError, match="denied"):
        await engine.pull_image("private/image")

    assert response.closed, "Failed image pulls must release the connection."


async def test_exec_output_budget_truncates_and_drains(monkeypatch) -> None:
    from tests.sandbox.test_docker_engine import _frame

    response = StreamResponse([_frame(1, b"1234"), _frame(2, b"5678"), _frame(1, b"90")])
    engine = DockerEngineClient(max_exec_output_bytes=5)
    monkeypatch.setattr(
        engine, "_get_stream_session", AsyncMock(return_value=SimpleNamespace(post=lambda *a, **k: response))
    )
    stdout, stderr, truncated = await engine._read_exec_output("id", None)
    assert (stdout, stderr, truncated) == (b"1234", b"5", True)
    assert response.chunks_read == 3
    assert response.closed


async def test_prepare_shares_pull_and_survives_one_cancelled_waiter(monkeypatch) -> None:
    engine = FakeDockerEngine()
    backend = DockerBackend(default_exec_mode=ExecMode.ONE_SHOT, engine=engine)
    manager = SandboxManager({"docker": backend}, "docker")
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def pull(reference, auth):
        engine.pulled.append(reference)
        entered.set()
        await finish.wait()

    monkeypatch.setattr(engine, "pull_image", pull)
    spec = SandboxSpec(SandboxSource.image("image"))
    first = asyncio.create_task(manager.prepare(spec))
    second = asyncio.create_task(manager.prepare(spec))
    await entered.wait()
    first.cancel()
    await asyncio.gather(first, return_exceptions=True)
    finish.set()
    await second
    await manager.prepare(spec)

    assert engine.pulled == ["image"], "Concurrent waiters must share one image download."
    assert engine.config is None, "Preparation must not allocate a container."
    assert not manager._leases, "Preparation must not acquire container capacity."
    await manager.shutdown()


async def test_prepare_limits_distinct_image_downloads_and_drains_on_shutdown(monkeypatch) -> None:
    engine = FakeDockerEngine()
    backend = DockerBackend(default_exec_mode=ExecMode.ONE_SHOT, engine=engine, image_pull_concurrency=2)
    entered = asyncio.Event()
    active = 0
    peak = 0

    async def pull(reference, auth):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if active == 2:
            entered.set()
        try:
            await asyncio.Future()
        finally:
            active -= 1

    monkeypatch.setattr(engine, "pull_image", pull)
    tasks = [asyncio.create_task(backend.prepare(SandboxSpec(SandboxSource.image(str(i))))) for i in range(10)]
    await entered.wait()
    await backend.shutdown()
    await asyncio.gather(*tasks, return_exceptions=True)

    assert peak == 2, "Image pulls must respect the configured concurrency bound."
    assert active == 0 and not backend._image_tasks, "Shutdown must drain pending preparation."
    assert engine.closed, "Shutdown must close the transport."


@pytest.mark.parametrize(
    "error",
    [aiohttp.ClientPayloadError("broken"), DockerEngineError(500, "daemon error")],
)
async def test_exec_stream_failure_terminates_the_container(monkeypatch, error) -> None:
    engine = FakeDockerEngine()
    backend = DockerBackend(default_exec_mode=ExecMode.ONE_SHOT, engine=engine)
    session = await backend.create(SandboxSpec(SandboxSource.image("image")))
    monkeypatch.setattr(engine, "exec", AsyncMock(side_effect=error))

    with pytest.raises(type(error)):
        await session.exec("background-work")

    assert engine.removes == 1, "Disconnected exec processes must not outlive their sandbox."


async def test_oversized_exec_output_keeps_the_sandbox(monkeypatch) -> None:
    """A bounded answer must not cost the episode its container or its workspace."""
    engine = FakeDockerEngine()
    session = await DockerBackend(default_exec_mode=ExecMode.ONE_SHOT, engine=engine).create(
        SandboxSpec(SandboxSource.image("image"))
    )
    monkeypatch.setattr(engine, "exec", AsyncMock(return_value=(0, b"x" * 32, b"", True)))

    result = await session.exec("cat large-file")

    assert result.truncated is True
    assert "output truncated" in result.stdout, "A truncated answer must say so."
    assert engine.removes == 0, "A large command result must not terminate a healthy sandbox."


async def test_exec_deadline_includes_engine_setup(monkeypatch) -> None:
    engine = FakeDockerEngine()
    session = await DockerBackend(default_exec_mode=ExecMode.ONE_SHOT, engine=engine).create(
        SandboxSpec(SandboxSource.image("image"))
    )

    async def blocked(*args, **kwargs):
        await asyncio.Future()

    monkeypatch.setattr(engine, "exec", blocked)
    with pytest.raises(TimeoutError) as raised:
        await session.exec("blocked", timeout_s=0.01)

    assert engine.removes == 1, "An expired command must release its container."
    # A one-shot exec has no process to signal, so a live orphan would write into the
    # next command's output. Destroying the container is the only safe answer there.
    assert raised.value.sandbox_preserved is False


async def test_a_persistent_shell_deadline_keeps_the_sandbox(monkeypatch) -> None:
    # One slow turn must not cost an episode its filesystem, its working directory, and
    # its activated environment.
    engine = FakeDockerEngine()
    shell = FakeShell(replies=1)
    signalled: list[int] = []

    async def kill_group(container_id: str, pid: int) -> bool:
        signalled.append(pid)
        return True

    backend = DockerBackend(
        default_exec_mode=ExecMode.PERSISTENT,
        engine=engine,
        shell_factory=lambda container_id: shell,
    )
    monkeypatch.setattr(backend, "_signal_shell_group", kill_group)
    session = await backend.create(SandboxSpec(SandboxSource.image("image")))

    with pytest.raises(SandboxCommandTimeout) as raised:
        await session.exec("slow-turn", timeout_s=0.05)

    assert raised.value.sandbox_preserved is True
    assert signalled == [4242]
    assert engine.removes == 0, "A signalled command must not take its container with it."
    assert await session.status() is not SandboxStatus.TERMINATED


async def test_a_deadline_the_shell_cannot_be_signalled_for_releases_the_container(monkeypatch) -> None:
    # No reported process id means there is no way to stop the command, and the safe
    # answer is the container's removal rather than a live orphan in it.
    engine = FakeDockerEngine()
    shell = FakeShell(first_body="ready", replies=1)

    backend = DockerBackend(
        default_exec_mode=ExecMode.PERSISTENT,
        engine=engine,
        shell_factory=lambda container_id: shell,
    )
    session = await backend.create(SandboxSpec(SandboxSource.image("image")))

    with pytest.raises(SandboxCommandTimeout) as raised:
        await session.exec("slow-turn", timeout_s=0.05)

    assert raised.value.sandbox_preserved is False
    assert engine.removes == 1


async def test_snapshot_references_are_immutable(monkeypatch) -> None:
    engine = FakeDockerEngine()
    monkeypatch.setattr(engine, "commit_container", AsyncMock(return_value="sha256:image"), raising=False)
    session = await DockerBackend(default_exec_mode=ExecMode.ONE_SHOT, engine=engine).create(
        SandboxSpec(SandboxSource.image("image"))
    )

    first = await session.snapshot(SnapshotKind.FILESYSTEM)
    second = await session.snapshot(SnapshotKind.FILESYSTEM)

    assert first.snapshot_id != second.snapshot_id, "Later snapshots must not retarget earlier references."


@pytest.mark.parametrize("inspection", [{"Running": True, "ExitCode": 0}, {"Running": False, "ExitCode": None}, {}])
async def test_exec_requires_a_final_exit_status(monkeypatch, inspection) -> None:
    engine = DockerEngineClient()
    monkeypatch.setattr(
        engine,
        "_request",
        AsyncMock(
            side_effect=[
                ({}, b'{"Id":"exec"}', 201),
                ({}, json.dumps(inspection), 200),
            ]
        ),
    )

    monkeypatch.setattr(engine, "_read_exec_output", AsyncMock(return_value=(b"", b"", False)))
    with pytest.raises(DockerEngineError, match="final exit status"):
        await engine.exec("container", ["work"])


def test_gc_listing_is_scoped_to_its_lease_namespace(monkeypatch) -> None:
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(cli.subprocess, "run", run)
    cli.DockerContainerRuntime(("docker",)).list_owned("namespace")

    assert "label=psrl.lease_store=namespace" in calls[0], "GC must not inspect other lease namespaces."


async def test_snapshot_deletion_accepts_the_same_reference_as_restore(monkeypatch) -> None:
    engine = FakeDockerEngine()
    remove = AsyncMock()
    monkeypatch.setattr(engine, "remove_image", remove, raising=False)
    backend = DockerBackend(default_exec_mode=ExecMode.ONE_SHOT, engine=engine)

    await backend.delete_snapshot(SnapshotRef("docker", "image-id", SnapshotKind.FILESYSTEM))

    remove.assert_awaited_once_with("image-id")


async def test_failed_lifecycle_cleanup_still_closes_engine(monkeypatch) -> None:
    engine = FakeDockerEngine()
    backend = DockerBackend(default_exec_mode=ExecMode.ONE_SHOT, engine=engine)

    def fail():
        raise OSError("lease store unavailable")

    monkeypatch.setattr(backend.lifecycle, "close", fail)
    with pytest.raises(OSError):
        await backend.shutdown()

    assert engine.closed, "A cleanup failure must not leak the connection pool."


async def test_start_failure_is_not_replaced_by_cleanup_failure(monkeypatch) -> None:
    engine = FakeDockerEngine()
    backend = DockerBackend(default_exec_mode=ExecMode.ONE_SHOT, engine=engine)
    monkeypatch.setattr(engine, "start_container", AsyncMock(side_effect=ValueError("start failed")))
    monkeypatch.setattr(engine, "remove_container", AsyncMock(side_effect=OSError("cleanup failed")))

    with pytest.raises(SandboxProvisionError, match="start failed"):
        await backend.create(SandboxSpec(SandboxSource.image("image")))


async def test_a_readiness_failure_carries_the_session_when_cleanup_also_fails(monkeypatch) -> None:
    # A container that starts but never becomes usable must not be handed out, and one that also cannot
    # be destroyed must stay flagged for cleanup. A bare re-raise would free the slot while it lives.
    engine = FakeDockerEngine()
    backend = DockerBackend(default_exec_mode=ExecMode.ONE_SHOT, engine=engine)
    monkeypatch.setattr(engine, "exec", AsyncMock(return_value=(1, b"", b"not ready", False)))
    monkeypatch.setattr(engine, "remove_container", AsyncMock(side_effect=OSError("cleanup failed")))

    with pytest.raises(SandboxProvisionError, match="did not become ready") as raised:
        await backend.create(SandboxSpec(SandboxSource.image("image")))

    # The manager adopts this session and retries destruction instead of freeing the
    # slot it still occupies.
    assert raised.value.session is not None
    assert raised.value.session.sandbox_id in backend._sessions


async def test_closed_engine_cannot_reopen_its_connection_pool() -> None:
    engine = DockerEngineClient()
    await engine.close()

    with pytest.raises(RuntimeError, match="client is closed"):
        await engine.info()


async def test_cancelling_exec_joins_transport_task_before_return(monkeypatch):
    engine = FakeDockerEngine()
    backend = DockerBackend(default_exec_mode=ExecMode.ONE_SHOT, engine=engine)
    session = await backend.create(SandboxSpec(SandboxSource.image("image")))
    entered, stopped = asyncio.Event(), asyncio.Event()

    async def execute(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Future()
        finally:
            stopped.set()

    monkeypatch.setattr(engine, "exec", execute)
    task = asyncio.create_task(session.exec("work"))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set()
    assert engine.removes == 1
    await backend.shutdown()


async def test_lost_create_response_retains_capacity_until_deletion(monkeypatch):
    from psrl.sandbox import ResourceSpec

    from tests.sandbox.test_manager import FakeCapacityCoordinator

    engine = FakeDockerEngine()
    backend = DockerBackend(default_exec_mode=ExecMode.ONE_SHOT, engine=engine)
    coordinator = FakeCapacityCoordinator()
    manager = SandboxManager(
        {"docker": backend},
        "docker",
        capacity_coordinator=coordinator,
        capacity_owner_id="owner",
        capacity_heartbeat_interval_s=30,
    )
    monkeypatch.setattr(engine, "create_container", AsyncMock(side_effect=aiohttp.ClientPayloadError("Lost reply.")))
    monkeypatch.setattr(engine, "remove_container", AsyncMock(side_effect=OSError("Daemon unavailable.")))
    with pytest.raises(SandboxProvisionError):
        await manager.acquire(SandboxSpec(SandboxSource.image("image"), resources=ResourceSpec(1, 1)))
    assert not coordinator.released
    assert len(manager._pending_releases) == 1
    remove = AsyncMock()
    monkeypatch.setattr(engine, "remove_container", remove)
    await asyncio.wait_for(manager._reaper_task, timeout=2)
    assert len(coordinator.released) == 1
    assert remove.call_args.args[0].startswith("psrl-sandbox-")
    await manager.shutdown()


async def test_lifecycle_pool_is_independent_of_command_stream_pool():
    engine = DockerEngineClient(connection_limit=1)
    control = await engine._get_session()
    streams = await engine._get_stream_session()
    assert control.connector is not streams.connector
    assert streams.timeout.total is None
    await engine.close()
    assert control.closed and streams.closed


async def test_failed_remove_does_not_claim_termination(monkeypatch):
    engine = FakeDockerEngine()
    backend = DockerBackend(default_exec_mode=ExecMode.ONE_SHOT, engine=engine)
    session = await backend.create(SandboxSpec(SandboxSource.image("image")))
    remove = engine.remove_container
    monkeypatch.setattr(engine, "remove_container", AsyncMock(side_effect=OSError("Daemon unavailable.")))
    with pytest.raises(OSError):
        await session.terminate()
    assert not session._terminated
    assert backend.metrics_snapshot().active_sessions == 1
    monkeypatch.setattr(engine, "remove_container", remove)
    await session.terminate()
    assert backend.metrics_snapshot().active_sessions == 0
    await backend.shutdown()


async def test_cancelled_create_reclaims_container_after_daemon_reply(monkeypatch):
    engine = FakeDockerEngine()
    backend = DockerBackend(default_exec_mode=ExecMode.ONE_SHOT, engine=engine)
    entered, finish = asyncio.Event(), asyncio.Event()
    create = engine.create_container

    async def delayed_create(name, config):
        entered.set()
        await finish.wait()
        return await create(name, config)

    monkeypatch.setattr(engine, "create_container", delayed_create)
    task = asyncio.create_task(backend.create(SandboxSpec(SandboxSource.image("image"))))
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert engine.removes == 1
    assert not backend._sessions
    await backend.shutdown()


async def test_timeout_while_waiting_for_session_does_not_cancel_active_command(monkeypatch):
    engine = FakeDockerEngine()
    backend = DockerBackend(default_exec_mode=ExecMode.ONE_SHOT, engine=engine)
    session = await backend.create(SandboxSpec(SandboxSource.image("image")))
    entered, finish = asyncio.Event(), asyncio.Event()

    async def execute(*args, **kwargs):
        entered.set()
        await finish.wait()
        return 0, b"done", b"", False

    monkeypatch.setattr(engine, "exec", execute)
    active = asyncio.create_task(session.exec("active"))
    await entered.wait()
    with pytest.raises(TimeoutError):
        await session.exec("queued", timeout_s=0.01)
    assert engine.removes == 0
    finish.set()
    assert (await active).stdout == "done"
    await backend.shutdown()


async def test_a_removal_error_for_a_container_that_is_already_gone_is_success(monkeypatch):
    # The API can fail after the container is gone. Retrying forever would keep a capacity
    # reservation charged for memory nobody is using.
    engine = FakeDockerEngine()
    backend = DockerBackend(default_exec_mode=ExecMode.ONE_SHOT, engine=engine)
    session = await backend.create(SandboxSpec(SandboxSource.image("image")))

    async def remove_then_report_gone(container_id):
        engine.state = "removed"
        raise DockerEngineError(500, "driver failed")

    monkeypatch.setattr(engine, "remove_container", remove_then_report_gone)
    cli_calls: list = []
    monkeypatch.setattr(cli, "force_remove_container_ids", lambda *a, **k: cli_calls.append(a) or [])

    await session.terminate()

    assert await session.status() is SandboxStatus.TERMINATED
    assert not cli_calls, "An already-gone container must not reach the CLI."
    assert not backend._sessions
    await backend.shutdown()


async def test_the_cli_is_the_last_hop_when_the_engine_api_refuses_to_delete(monkeypatch):
    engine = FakeDockerEngine()
    backend = DockerBackend(default_exec_mode=ExecMode.ONE_SHOT, engine=engine)
    session = await backend.create(SandboxSpec(SandboxSource.image("image")))
    requested: list[list[str]] = []

    async def always_fail(container_id):
        raise DockerEngineError(500, "unlinkat: device or resource busy")

    def cli_remove(ids, **kwargs):
        requested.append(list(ids))
        engine.state = "removed"
        return list(ids)

    monkeypatch.setattr(engine, "remove_container", always_fail)
    monkeypatch.setattr(
        "psrl.sandbox.backends.docker.session.force_remove_container_ids",
        cli_remove,
    )

    await session.terminate()

    assert requested == [["container-id"]]
    assert await session.status() is SandboxStatus.TERMINATED
    await backend.shutdown()


async def test_a_container_neither_api_nor_cli_can_delete_keeps_its_reservation(monkeypatch):
    engine = FakeDockerEngine()
    backend = DockerBackend(default_exec_mode=ExecMode.ONE_SHOT, engine=engine)
    manager = SandboxManager({"docker": backend}, "docker")
    lease = await manager.acquire(SandboxSpec(SandboxSource.image("image")))

    async def always_fail(container_id):
        raise DockerEngineError(500, "unlinkat: device or resource busy")

    monkeypatch.setattr(engine, "remove_container", always_fail)
    monkeypatch.setattr(
        "psrl.sandbox.backends.docker.session.force_remove_container_ids",
        lambda ids, **kwargs: [],
    )

    await lease.release()

    assert not lease.released
    assert lease in manager._pending_releases
    assert manager.ownership_snapshot()["pending_releases"] == 1
    manager._reaper_task.cancel()
    await asyncio.gather(manager._reaper_task, return_exceptions=True)


def _die(container_id: str, action: str = "die", nano: int = 1) -> dict:
    return {"id": container_id, "Action": action, "timeNano": nano}


async def test_a_stop_is_observed_from_the_event_stream_without_polling(monkeypatch):
    engine = FakeDockerEngine()
    engine.events_queue = asyncio.Queue()
    backend = DockerBackend(default_exec_mode=ExecMode.ONE_SHOT, engine=engine, container_watch_interval_s=3600)
    session = await backend.create(SandboxSpec(SandboxSource.image("image")))
    entered = asyncio.Event()

    async def hang(*args, **kwargs):
        entered.set()
        await asyncio.Future()

    monkeypatch.setattr(engine, "exec", hang)
    command = asyncio.create_task(session.exec("work"))
    await entered.wait()
    # `oom` precedes the `die` that follows it, and outranks it.
    await engine.events_queue.put(_die("container-id", "oom"))
    await engine.events_queue.put(_die("container-id", "die", nano=2))

    with pytest.raises(SandboxOomError):
        await asyncio.wait_for(command, timeout=2)
    await backend.shutdown()


async def test_a_plain_stop_is_reported_as_an_exit_not_an_oom(monkeypatch):
    engine = FakeDockerEngine()
    engine.events_queue = asyncio.Queue()
    backend = DockerBackend(default_exec_mode=ExecMode.ONE_SHOT, engine=engine, container_watch_interval_s=3600)
    session = await backend.create(SandboxSpec(SandboxSource.image("image")))
    entered = asyncio.Event()

    async def hang(*args, **kwargs):
        entered.set()
        await asyncio.Future()

    monkeypatch.setattr(engine, "exec", hang)
    command = asyncio.create_task(session.exec("work"))
    await entered.wait()
    await engine.events_queue.put(_die("container-id"))

    with pytest.raises(RuntimeError, match="stopped"):
        await asyncio.wait_for(command, timeout=2)
    assert not isinstance(command.exception(), SandboxOomError)
    await backend.shutdown()


async def test_a_daemon_without_event_access_falls_back_to_polling(monkeypatch):
    # FakeDockerEngine refuses events, which is how a restricted daemon behaves.
    engine = FakeDockerEngine()
    backend = DockerBackend(default_exec_mode=ExecMode.ONE_SHOT, engine=engine, container_watch_interval_s=0.01)
    session = await backend.create(SandboxSpec(SandboxSource.image("image")))
    entered = asyncio.Event()

    async def hang(*args, **kwargs):
        entered.set()
        await asyncio.Future()

    monkeypatch.setattr(engine, "exec", hang)
    command = asyncio.create_task(session.exec("work"))
    await entered.wait()
    engine.state = "exited"

    with pytest.raises(RuntimeError, match="stopped"):
        await asyncio.wait_for(command, timeout=3)
    assert backend.container_events.supported is False
    await backend.shutdown()


async def test_a_stop_during_a_stream_gap_is_recovered_by_reinspection():
    engine = FakeDockerEngine()
    engine.events_queue = asyncio.Queue()
    backend = DockerBackend(default_exec_mode=ExecMode.ONE_SHOT, engine=engine, container_watch_interval_s=3600)
    watcher = backend.container_events
    waiting = asyncio.create_task(watcher.wait_for_stop("container-id"))
    await asyncio.sleep(0)
    await watcher._ready.wait()

    # Drop the stream the way a daemon restart does, losing its event history.
    async def refuse(*args, **kwargs):
        raise DockerEngineError(500, "daemon restarting")
        yield {}  # pragma: no cover

    engine.events = refuse
    engine.state = "exited"
    watcher._task.cancel()
    await asyncio.gather(watcher._task, return_exceptions=True)
    await watcher._resync()

    assert await asyncio.wait_for(waiting, timeout=1) == "exited"
    await backend.shutdown()


async def test_the_event_stream_replays_from_the_last_event_after_a_drop():
    engine = FakeDockerEngine()
    engine.events_queue = asyncio.Queue()
    backend = DockerBackend(default_exec_mode=ExecMode.ONE_SHOT, engine=engine, container_watch_interval_s=3600)
    watcher = backend.container_events
    waiting = asyncio.create_task(watcher.wait_for_stop("other"))
    await asyncio.sleep(0)
    await watcher._ready.wait()
    await engine.events_queue.put(_die("unrelated", "die", nano=5_000_000_000))
    await asyncio.sleep(0.05)

    assert watcher._last_event_at == 5.0
    waiting.cancel()
    await asyncio.gather(waiting, return_exceptions=True)
    await backend.shutdown()
