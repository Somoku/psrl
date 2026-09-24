from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from psrl.sandbox import (
    EgressAction,
    EgressPolicy,
    EgressRule,
    MountSpec,
    PauseMode,
    ResourceSpec,
    SandboxBusyError,
    SandboxFeature,
    SandboxSource,
    SandboxSpec,
    SnapshotKind,
    SnapshotRef,
    VolumeSpec,
)
from psrl.sandbox.backends.e2b import (
    AgentEnvBackend,
    AgentEnvClientFactory,
    AgentEnvStateDriver,
    CubeSandboxClientFactory,
    CubeSandboxStateDriver,
    E2BBackend,
    E2BNativeStateDriver,
    template_name_for_image,
)


class FakeCommands:
    def __init__(self) -> None:
        # Set to hold a command open, so a test can observe a sandbox with work in
        # flight rather than a sandbox that merely finished one.
        self.entered: asyncio.Event | None = None
        self.release: asyncio.Event | None = None

    async def run(self, command: str, **kwargs):
        if command == "slow":
            assert self.entered is not None and self.release is not None
            self.entered.set()
            await self.release.wait()
            return SimpleNamespace(exit_code=0, stdout="", stderr="")
        if command == "fail":
            error = RuntimeError("failed")
            error.exit_code = 3
            error.stdout = "partial"
            error.stderr = "failure"
            raise error
        return SimpleNamespace(exit_code=7, stdout=command, stderr=kwargs.get("cwd", ""))


class FakeFiles:
    def __init__(self) -> None:
        self.data = {}

    async def read(self, path: str, format: str):
        return bytearray(self.data[path])

    async def write(self, path: str, data: bytes):
        self.data[path] = data


class FakeClient:
    """An SDK sandbox, with the SDK's own documented method signatures."""

    def __init__(self, sandbox_id: str) -> None:
        self.sandbox_id = sandbox_id
        self.commands = FakeCommands()
        self.files = FakeFiles()
        self.running = True
        self.paused = False
        # Recorded so a test can assert the calls the provider's own documentation
        # describes, rather than a request shape PSRL invented.
        self.snapshot_names: list[str | None] = []
        self.forks: list[tuple[int, int | None]] = []
        self.timeouts: list[int] = []

    async def is_running(self):
        return self.running

    async def kill(self):
        self.running = False

    async def pause(self, **opts):
        self.paused = True

    async def set_timeout(self, timeout: int):
        self.timeouts.append(timeout)

    async def create_snapshot(self, name: str | None = None, **opts):
        self.snapshot_names.append(name)
        return SimpleNamespace(snapshot_id="snapshot-1", names=[f"team/{name}"] if name else [])

    async def fork(self, count: int = 1, timeout: int | None = None, **opts):
        # The SDK returns one entry per requested fork, each a sandbox or an exception.
        self.forks.append((count, timeout))
        return [FakeClient(f"fork-{index + 1}") for index in range(count)]


class FakeFactory:
    def __init__(self) -> None:
        self.client = FakeClient("sandbox-1")
        self.deleted_snapshots: list[str] = []
        # Recording the connect calls is how a test sees that a resume renewed the life.
        self.connects: list[tuple[str, int | None]] = []

    async def create(self, spec: SandboxSpec):
        return self.client

    async def connect(self, sandbox_id: str, *, timeout: int | None = None):
        self.connects.append((sandbox_id, timeout))
        self.client.paused = False
        return self.client

    async def close(self):
        return None

    async def delete_snapshot(self, snapshot_id: str):
        self.deleted_snapshots.append(snapshot_id)


class FakeControl:
    def __init__(self) -> None:
        self.api_key = "test-key"
        self.api_url = "http://provider.test"
        self.requests: list[tuple[str, str, dict]] = []
        self.deleted: list[str] = []
        # What a filtered list returns, so a test can describe the orphans a lost
        # create may have left behind.
        self.listed: list[dict] = []
        self.list_fails = False

    async def request(self, method: str, path: str, *, expected: tuple[int, ...], payload=None):
        self.requests.append((method, path, dict(payload or {})))
        if path.startswith("/sandboxes?"):
            if self.list_fails:
                raise RuntimeError("provider listing is unavailable")
            return list(self.listed)
        if path.endswith("/fork"):
            count = int((payload or {}).get("count") or 1)
            return [{"sandbox": {"sandboxID": f"fork-{index + 1}"}} for index in range(count)]
        if path.endswith("/snapshots"):
            return {"snapshotID": "snapshot-1", "names": ["baseline"]}
        return {"sandboxID": "created-1"}

    async def delete_idempotent(self, path: str):
        self.deleted.append(path)

    async def close(self):
        return None


class FakeSDKClass:
    """The SDK's class-level entry points, recording what PSRL asked the provider for."""

    def __init__(self) -> None:
        self.created: list[dict] = []
        self.connected: list[tuple[str, dict]] = []
        self.deleted: list[str] = []

    async def create(self, **kwargs):
        self.created.append(kwargs)
        return FakeClient(str(kwargs.get("template") or "sandbox-1"))

    async def connect(self, sandbox_id: str, **kwargs):
        self.connected.append((sandbox_id, kwargs))
        return FakeClient(sandbox_id)

    async def delete_snapshot(self, snapshot_id: str, **kwargs) -> None:
        self.deleted.append(snapshot_id)


class FakeAgentEnvFactory(AgentEnvClientFactory):
    """The real AgentEnv create mapping, with only the SDK boundary replaced.

    Driving the real factory is what makes these tests about the mapping rather than about
    a request shape a test double agreed to.
    """

    def __init__(self, control: FakeControl | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.control = control
        self.sdk = FakeSDKClass()

    def _sandbox_class(self):  # type: ignore[override]
        return self.sdk

    @property
    def deleted_snapshots(self) -> list[str]:
        return self.sdk.deleted


class FakeCubeFactory(CubeSandboxClientFactory):
    async def connect(self, sandbox_id: str):
        return FakeClient(sandbox_id)


@pytest.mark.asyncio
async def test_e2b_compatible_data_plane_and_hibernate() -> None:
    factory = FakeFactory()
    backend = E2BBackend(client_factory=factory)
    session = await backend.create(SandboxSpec(SandboxSource.template("base")))

    result = await session.exec("echo ok", cwd="/workspace")
    await session.write_bytes("/tmp/value", b"value")
    await session.pause(PauseMode.HIBERNATE)

    assert result.exit_code == 7
    assert result.stderr == "/workspace"
    assert await session.read_bytes("/tmp/value") == b"value"
    assert factory.client.paused
    assert backend.capabilities.supports(SandboxFeature.HIBERNATE)

    failed = await session.exec("fail")
    assert failed.exit_code == 3
    assert failed.stdout == "partial"


@pytest.mark.asyncio
async def test_e2b_native_state_capabilities() -> None:
    factory = FakeFactory()
    backend = E2BBackend(client_factory=factory)
    session = await backend.create(SandboxSpec(SandboxSource.template("base")))

    snapshot = await session.snapshot(SnapshotKind.FULL_STATE)
    child = (await session.fork(1))[0]
    restored = await backend.restore(snapshot)
    await backend.delete_snapshot(snapshot)

    assert snapshot.snapshot_id == "snapshot-1"
    assert child.ref.sandbox_id == "fork-1"
    assert restored.ref.sandbox_id == "sandbox-1"
    assert backend.capabilities.supports(SandboxFeature.FULL_STATE_SNAPSHOT)
    assert backend.capabilities.supports(SandboxFeature.NATIVE_FORK)
    assert factory.deleted_snapshots == ["snapshot-1"]


@pytest.mark.asyncio
async def test_generic_e2b_rejects_image_and_runtime_resource_overrides() -> None:
    # A custom factory owns its own mapping contract, so validate the official
    # factory directly rather than pretending all injected clients are E2B.
    from psrl.sandbox.backends.e2b import E2BSDKClientFactory

    sdk_factory = E2BSDKClientFactory()
    with pytest.raises(RuntimeError, match="template source"):
        await sdk_factory.create(SandboxSpec(SandboxSource.image("image")))
    with pytest.raises(RuntimeError, match="fixed by the template"):
        await sdk_factory.create(
            SandboxSpec(
                SandboxSource.template("template"),
                resources=ResourceSpec(memory_mb=1024),
            )
        )


def test_agentenv_does_not_overclaim_optional_e2b_state_apis() -> None:
    backend = AgentEnvBackend(client_factory=FakeFactory())

    assert backend.capabilities.supports(SandboxFeature.HIBERNATE)
    assert not backend.capabilities.supports(SandboxFeature.NATIVE_FORK)


@pytest.mark.asyncio
async def test_agentenv_create_names_the_template_an_image_resolves_to() -> None:
    # An image is not a create-time source for this provider: it becomes a template, and
    # the sandbox is created from that template's name.
    factory = FakeAgentEnvFactory()

    client = await factory.create(
        SandboxSpec(
            SandboxSource.image("registry/image:tag"),
            env={"A": "B"},
            metadata={"task": "1"},
            idle_timeout_s=12.1,
            idempotency_key="rollout-1",
        )
    )

    template = template_name_for_image("registry/image:tag")
    assert client.sandbox_id == template
    asked = factory.sdk.created[0]
    assert asked["template"] == template
    # The provider's timeout is the sandbox's own life.
    assert asked["timeout"] == 13
    assert asked["envs"] == {"A": "B"}
    assert asked["metadata"]["task"] == "1"
    assert asked["metadata"]["psrl.idempotency_key"] == "rollout-1"


@pytest.mark.asyncio
async def test_agentenv_refuses_what_the_template_owns() -> None:
    # The template's build fixes the shape, so a spec that overrides resources, names a
    # host mount, or asks for provider storage is refused here rather than at the server.
    factory = FakeAgentEnvFactory()

    with pytest.raises(RuntimeError, match="resources are fixed"):
        await factory.create(
            SandboxSpec(SandboxSource.image("image"), resources=ResourceSpec(cpu_count=4))
        )
    with pytest.raises(RuntimeError, match="no host bind mount"):
        await factory.create(
            SandboxSpec(SandboxSource.image("image"), mounts=(MountSpec(source="/host", target="/work"),))
        )
    with pytest.raises(RuntimeError, match="volumes are not wired"):
        await factory.create(
            SandboxSpec(SandboxSource.image("image"), volumes=(VolumeSpec(name="cache", target="/cache"),))
        )


@pytest.mark.asyncio
async def test_agentenv_create_from_a_template_names_it_unchanged() -> None:
    factory = FakeAgentEnvFactory()

    await factory.create(SandboxSpec(SandboxSource.template("task-template")))

    assert factory.sdk.created[0]["template"] == "task-template"


@pytest.mark.asyncio
async def test_cube_factory_maps_template_resources_and_rejects_images() -> None:
    control = FakeControl()
    factory = FakeCubeFactory(control=control)
    spec = SandboxSpec(
        SandboxSource.template("template-1"),
        resources=ResourceSpec(cpu_count=2, memory_mb=4096, disk_mb=8192),
    )

    await factory.create(spec)

    _, path, payload = control.requests[0]
    assert path == "/sandboxes"
    assert payload["templateID"] == "template-1"
    assert payload["cpuCount"] == 2
    assert payload["memoryMB"] == 4096
    assert payload["diskSizeMB"] == 8192
    with pytest.raises(RuntimeError, match="built template"):
        await factory.create(SandboxSpec(SandboxSource.image("image")))


@pytest.mark.asyncio
async def test_provider_specific_state_drivers_advertise_only_real_semantics() -> None:
    factory = FakeFactory()
    agentenv = E2BBackend(
        name="agentenv",
        client_factory=factory,
        state_driver=AgentEnvStateDriver(factory),
    )
    cube = E2BBackend(
        name="cubesandbox",
        client_factory=factory,
        state_driver=CubeSandboxStateDriver(factory),
    )
    session = await agentenv.create(SandboxSpec(SandboxSource.template("base")))

    snapshot = await session.snapshot(SnapshotKind.FULL_STATE)
    child = (await session.fork(1))[0]
    await agentenv.delete_snapshot(snapshot)

    assert child.ref.sandbox_id == "fork-1"
    assert agentenv.capabilities.supports(SandboxFeature.NATIVE_FORK)
    assert cube.capabilities.supports(SandboxFeature.FULL_STATE_SNAPSHOT)
    assert not cube.capabilities.supports(SandboxFeature.NATIVE_FORK)
    # The snapshot object is the authoritative target: a template is a derivative of
    # a snapshot, so deleting the derivative would leave the source behind.
    assert factory.deleted_snapshots == ["snapshot-1"]


@pytest.mark.asyncio
async def test_a_resume_carries_the_life_the_caller_asked_for() -> None:
    # A provider resume starts the expiry clock over, so sending nothing would let the
    # sandbox expire against a deadline the caller never asked for.
    factory = FakeFactory()
    backend = E2BBackend(
        name="agentenv",
        client_factory=factory,
        state_driver=AgentEnvStateDriver(factory),
    )
    session = await backend.create(
        SandboxSpec(SandboxSource.template("base"), lifetime_timeout_s=7200)
    )

    await session.resume()

    # Connecting is what resumes a paused sandbox, and the life travels with it because
    # the provider restarts the expiry clock here.
    assert factory.connects == [("sandbox-1", 7200)]


@pytest.mark.asyncio
async def test_a_fork_carries_the_life_the_members_asked_for() -> None:
    # A fork starts each child's provider clock over, so a group's children get the
    # life their members asked for rather than the provider's default.
    factory = FakeFactory()
    backend = E2BBackend(
        name="agentenv",
        client_factory=factory,
        state_driver=AgentEnvStateDriver(factory),
    )
    session = await backend.create(
        SandboxSpec(SandboxSource.template("base"), lifetime_timeout_s=7200)
    )

    await session.fork(2)

    # The SDK takes the whole count in one call and reports per entry, so there is no
    # batch size for PSRL to negotiate.
    assert factory.client.forks == [(2, 7200)]


def _agentenv(control: FakeControl, factory) -> E2BBackend:
    return E2BBackend(name="agentenv", client_factory=factory, state_driver=AgentEnvStateDriver(factory))


@pytest.mark.asyncio
async def test_a_restore_addresses_the_snapshot_by_the_name_it_was_captured_under() -> None:
    # The provider accepts a name anywhere it accepts an id, and a name survives a
    # re-import, so a captured snapshot is restored by its name rather than by its id.
    factory = FakeAgentEnvFactory()
    backend = _agentenv(FakeControl(), factory)
    session = await backend.create(SandboxSpec(SandboxSource.template("base")))
    snapshot = await session.snapshot(SnapshotKind.FULL_STATE)

    await backend.restore(snapshot, SandboxSpec(SandboxSource.template("unused")))

    assert factory.sdk.created[-1]["template"] == snapshot.metadata["psrl.snapshot.name"]


@pytest.mark.asyncio
async def test_a_restore_falls_back_to_the_snapshot_id_when_no_name_is_recorded() -> None:
    factory = FakeAgentEnvFactory()
    backend = _agentenv(FakeControl(), factory)

    await backend.restore(
        SnapshotRef("agentenv", "snapshot-1", SnapshotKind.FULL_STATE),
        SandboxSpec(SandboxSource.template("unused")),
    )

    assert factory.sdk.created[-1]["template"] == "snapshot-1"


@pytest.mark.asyncio
async def test_a_fork_refuses_while_a_command_is_in_flight() -> None:
    # A fork briefly pauses the source while it captures state, so it must not overlap
    # a running command. The exec lock is the in-flight signal, taken without waiting.
    control = FakeControl()
    factory = FakeFactory()
    backend = _agentenv(control, factory)
    session = await backend.create(SandboxSpec(SandboxSource.template("base")))
    busy = await _hold_open(session)

    with pytest.raises(SandboxBusyError, match="cannot be forked"):
        await session.fork(2)

    await busy.release()


@pytest.mark.asyncio
async def test_a_pause_refuses_while_a_command_is_in_flight() -> None:
    control = FakeControl()
    factory = FakeFactory()
    backend = _agentenv(control, factory)
    session = await backend.create(SandboxSpec(SandboxSource.template("base")))
    busy = await _hold_open(session)

    with pytest.raises(SandboxBusyError, match="not idle"):
        await session.pause(PauseMode.HIBERNATE)

    await busy.release()


class _Busy:
    """One command held open, so a test can observe a sandbox that is working."""

    def __init__(self, client, running: asyncio.Task) -> None:
        self.client = client
        self.running = running

    async def release(self) -> None:
        assert self.client.commands.release is not None
        self.client.commands.release.set()
        await self.running


async def _hold_open(session) -> _Busy:
    session.client.commands.entered = asyncio.Event()
    session.client.commands.release = asyncio.Event()
    running = asyncio.create_task(session.exec("slow"))
    await session.client.commands.entered.wait()
    return _Busy(session.client, running)


def _losing_backend(control: FakeControl) -> AgentEnvBackend:
    """An AgentEnv backend whose create loses its reply, as a provider timeout would."""
    factory = FakeAgentEnvFactory(control=control)

    async def lose_reply(spec):
        raise RuntimeError("lost reply")

    factory.create = lose_reply  # type: ignore[method-assign]
    return AgentEnvBackend(client_factory=factory)


async def test_a_lost_create_is_reclaimed_by_listing_on_the_idempotency_key() -> None:
    # A create whose reply was lost can still have produced a sandbox. The metadata carries the
    # idempotency key, so a filtered list destroys it before the failure surfaces and a retry cannot build a second.
    control = FakeControl()
    control.listed = [{"sandboxID": "orphan-1"}, {"sandboxID": "orphan-2"}]
    backend = _losing_backend(control)

    with pytest.raises(RuntimeError, match="lost reply"):
        await backend.create(SandboxSpec(SandboxSource.template("base"), idempotency_key="task-1:rollout"))

    assert ("GET", "/sandboxes?metadata.psrl.idempotency_key=task-1:rollout", {}) in control.requests
    assert control.deleted == ["/sandboxes/orphan-1", "/sandboxes/orphan-2"]
    assert backend.metrics_snapshot().operations["lost_create_reclaimed"].count == 1


async def test_a_lost_create_without_an_idempotency_key_has_nothing_to_filter_on() -> None:
    # No key means no filter, so the orphan cannot be found and is left to the provider's own
    # idle timeout. Pinned here because it is the boundary of what this backend promises.
    control = FakeControl()
    control.listed = [{"sandboxID": "orphan-1"}]
    backend = _losing_backend(control)

    with pytest.raises(RuntimeError, match="lost reply"):
        await backend.create(SandboxSpec(SandboxSource.template("base")))

    assert not [path for _, path, _ in control.requests if path.startswith("/sandboxes?")]
    assert control.deleted == []


async def test_a_reclaim_whose_listing_fails_is_counted_rather_than_raised() -> None:
    # The create failure is what the caller has to see. A reclaim that could not
    # complete must not replace it with a different error.
    control = FakeControl()
    control.list_fails = True
    backend = _losing_backend(control)

    with pytest.raises(RuntimeError, match="lost reply"):
        await backend.create(SandboxSpec(SandboxSource.template("base"), idempotency_key="task-1:rollout"))

    assert backend.metrics_snapshot().operations["lost_create_reclaim_failed"].count == 1


async def test_a_create_that_succeeds_reclaims_nothing() -> None:
    # The listing is a recovery path, not something every create pays for.
    control = FakeControl()
    control.listed = [{"sandboxID": "orphan-1"}]
    backend = _agentenv(control, FakeAgentEnvFactory(control=control))

    session = await backend.create(SandboxSpec(SandboxSource.template("base"), idempotency_key="task-1:rollout"))

    assert session.ref.sandbox_id == "base"
    assert not [path for _, path, _ in control.requests if path.startswith("/sandboxes?")]
    assert control.deleted == []


async def test_a_reclaimed_listing_skips_an_entry_the_provider_did_not_name() -> None:
    # A malformed entry is skipped rather than failing the whole reclaim, so one odd
    # response cannot leave every other orphan in place.
    control = FakeControl()
    control.listed = [{}, {"sandbox_id": "orphan-1"}]
    backend = _losing_backend(control)

    with pytest.raises(RuntimeError, match="lost reply"):
        await backend.create(SandboxSpec(SandboxSource.template("base"), idempotency_key="task-1:rollout"))

    assert control.deleted == ["/sandboxes/orphan-1"]


async def test_a_restore_clears_what_the_provider_forbids_on_a_snapshot_create() -> None:
    # The provider rejects a restore that carries environment variables, volumes, mounts, or a
    # resource request, since those are fixed by the snapshot. Sending them would return a 400.
    factory = FakeAgentEnvFactory()
    backend = _agentenv(FakeControl(), factory)
    spec = SandboxSpec(
        SandboxSource.template("ignored"),
        env={"TOKEN": "x"},
        volumes=(VolumeSpec(name="cache", target="/cache"),),
        mounts=(MountSpec(source="/host", target="/work"),),
        resources=ResourceSpec(cpu_count=4, memory_mb=8192),
    )

    await backend.restore(
        SnapshotRef("agentenv", "snapshot-1", SnapshotKind.FULL_STATE),
        spec,
    )

    asked = factory.sdk.created[-1]
    assert asked["template"] == "snapshot-1"
    # The snapshot carries all three, and the provider rejects a restore that passes them.
    assert "envs" not in asked
    assert "volume_mounts" not in asked


async def test_a_restore_keeps_the_overrides_the_provider_allows() -> None:
    # A lifetime, user metadata, and the network policy are the allowed overrides.
    factory = FakeAgentEnvFactory()
    backend = _agentenv(FakeControl(), factory)
    spec = SandboxSpec(
        SandboxSource.template("ignored"),
        metadata={"task": "1"},
        lifetime_timeout_s=900,
        egress=EgressPolicy(rules=(EgressRule(EgressAction.ALLOW, "example.com"),)),
    )

    await backend.restore(
        SnapshotRef("agentenv", "snapshot-1", SnapshotKind.FULL_STATE),
        spec,
    )

    asked = factory.sdk.created[-1]
    assert asked["metadata"]["task"] == "1"
    assert asked["timeout"] == 900


async def test_a_partial_fork_destroys_the_children_that_did_start() -> None:
    # The SDK reports one entry per fork and they succeed independently, so a partial result
    # must not leak a refused group's children that did boot, since nothing else would destroy them.
    factory = FakeFactory()
    backend = E2BBackend(client_factory=factory, state_driver=E2BNativeStateDriver(factory))
    session = await backend.create(SandboxSpec(SandboxSource.template("base")))
    started: list[FakeClient] = []

    async def partial(count: int = 1, timeout: int | None = None, **opts):
        started.append(FakeClient("fork-1"))
        started.append(FakeClient("fork-2"))
        return [started[0], RuntimeError("rate limited"), started[1]]

    session.client.fork = partial  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="rate limited"):
        await session.fork(3)

    assert [child.running for child in started] == [False, False]


async def test_a_fork_that_returns_fewer_entries_than_requested_is_refused() -> None:
    factory = FakeFactory()
    backend = E2BBackend(client_factory=factory, state_driver=E2BNativeStateDriver(factory))
    session = await backend.create(SandboxSpec(SandboxSource.template("base")))

    async def short(count: int = 1, timeout: int | None = None, **opts):
        return [FakeClient("fork-1")]

    session.client.fork = short  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="1 entry"):
        await session.fork(2)


async def test_a_snapshot_is_named_so_a_restore_can_address_it_by_name() -> None:
    # A qualified name survives a re-import where a bare id does not, so the name the
    # provider assigns travels in the snapshot's metadata.
    factory = FakeFactory()
    backend = E2BBackend(client_factory=factory, state_driver=E2BNativeStateDriver(factory))
    session = await backend.create(SandboxSpec(SandboxSource.template("base")))

    snapshot = await session.snapshot(SnapshotKind.FULL_STATE)

    assert factory.client.snapshot_names == [snapshot.metadata["psrl.snapshot.name"]]
    assert snapshot.metadata["names"] == [f"team/{snapshot.metadata['psrl.snapshot.name']}"]


async def test_a_command_result_error_is_surfaced_rather_than_dropped() -> None:
    # The SDK's result carries a message of its own when the command could not run at
    # all, so a caller that only saw an empty stderr would read it as a quiet success.
    factory = FakeFactory()
    backend = E2BBackend(client_factory=factory)
    session = await backend.create(SandboxSpec(SandboxSource.template("base")))

    async def failing(cmd: str, **kwargs):
        return SimpleNamespace(exit_code=1, stdout="", stderr="", error="no such binary: pytest")

    session.client.commands.run = failing  # type: ignore[method-assign]

    result = await session.exec("pytest")

    assert result.exit_code == 1
    assert "no such binary: pytest" in result.stderr


async def test_a_lifetime_the_provider_cannot_grant_is_refused() -> None:
    # The provider documents a ceiling on a sandbox's life rather than clamping it, so an
    # over-long request fails here where the spec becomes a provider request, not at the server.
    control = FakeControl()
    backend = _agentenv(control, FakeAgentEnvFactory(control=control))

    with pytest.raises(ValueError, match="86400"):
        await backend.create(SandboxSpec(SandboxSource.template("base"), lifetime_timeout_s=200_000))


def test_the_backend_wires_the_sdk_factory_and_its_own_template_surface() -> None:
    # The default construction path: the sandbox surface comes from the SDK factory, the
    # template surface from its own client, and neither touches the network or the optional package until a call.
    backend = AgentEnvBackend(api_url="http://provider.test", api_key="key")

    assert isinstance(backend.client_factory, AgentEnvClientFactory)
    assert isinstance(backend.state_driver, AgentEnvStateDriver)
    assert backend.templates is not None
    assert backend.capabilities.supports(SandboxFeature.NATIVE_FORK)


def test_the_backend_does_not_declare_a_capability_it_cannot_honour() -> None:
    # Volumes have no published SDK shape, so the capability is absent and admission
    # refuses a spec that needs provider storage rather than sending a guessed request.
    backend = AgentEnvBackend(api_url="http://provider.test", api_key="key")

    assert not backend.capabilities.supports(SandboxFeature.VOLUME)
    assert not backend.capabilities.supports(SandboxFeature.HOST_MOUNT)
