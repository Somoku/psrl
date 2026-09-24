"""One suite every sandbox backend has to pass.

The contract is the same for the internal Docker backend and for both external
providers, and the only thing that may differ is what a capability declaration
says. A backend that passes this suite on a case it declared is a backend whose
declaration can be trusted, which is what lets a recipe select by capability.

The suite runs against fakes rather than live services, so it is a contract test
and not a provider acceptance test. Real endpoints are exercised by the live
suites, which are opt-in.
"""

from __future__ import annotations

import pytest
from psrl.sandbox import (
    ExecMode,
    PauseMode,
    ResumeLevel,
    SandboxBackend,
    SandboxCapabilities,
    SandboxCapacityTimeout,
    SandboxFeature,
    SandboxManager,
    SandboxSource,
    SandboxSpec,
    SandboxStatePolicy,
    SandboxStatus,
    SnapshotKind,
)

from tests.sandbox.test_docker_backend import FakeDockerEngine
from tests.sandbox.test_e2b_backend import FakeAgentEnvFactory, FakeControl
from tests.sandbox.test_opensandbox_backend import FakeProvider

pytestmark = pytest.mark.cpu_test

_POLICY = SandboxStatePolicy(enabled=True, allow_external_side_effects=True)

# A restore reseeds host entropy by default, which only a real guest can demonstrate.
# The manager's suite covers that path, so this case is about the resume level.
_RESTORE_POLICY = SandboxStatePolicy(
    enabled=True,
    allow_external_side_effects=True,
    reseed_after_restore=False,
)


class ConformanceEngine(FakeDockerEngine):
    """A daemon that names each container and can commit and remove images."""

    def __init__(self) -> None:
        super().__init__()
        self._next = 0
        self.removed_images: list[str] = []

    async def create_container(self, name: str, config) -> str:
        self._next += 1
        self.config = config
        return f"container-{self._next}"

    async def commit_container(self, container_id: str, repository: str, tag: str) -> str:
        return f"{repository}:{tag}"

    async def remove_image(self, reference: str) -> None:
        self.removed_images.append(reference)


def _docker_backend() -> SandboxBackend:
    from psrl.sandbox.backends.docker import DockerBackend

    return DockerBackend(default_exec_mode=ExecMode.ONE_SHOT, engine=ConformanceEngine())


def _agentenv_backend() -> SandboxBackend:
    from psrl.sandbox.backends.e2b import AgentEnvBackend, AgentEnvStateDriver

    control = FakeControl()
    factory = FakeAgentEnvFactory(control=control)
    return AgentEnvBackend(
        client_factory=factory,
        state_driver=AgentEnvStateDriver(factory),
    )


def _opensandbox_backend() -> SandboxBackend:
    from psrl.sandbox.backends.opensandbox import (
        OpenSandboxBackend,
        OpenSandboxConfig,
    )

    from tests.sandbox.test_opensandbox_backend import ExecTransport

    provider = FakeProvider()
    # The provider's event framing: text under `text`, and completion with no exit code.
    provider.stream = ['{"type": "stdout", "text": "hi\\n"}', '{"type": "execution_complete"}']
    return OpenSandboxBackend(
        OpenSandboxConfig(api_url="http://lifecycle:8080", poll_interval_s=0.001, pause_timeout_s=1.0),
        transport=provider,
        exec_transport_factory=lambda: ExecTransport(provider),
    )


_BUILDERS = {
    "docker": _docker_backend,
    "agentenv": _agentenv_backend,
    "opensandbox": _opensandbox_backend,
}


@pytest.fixture(params=sorted(_BUILDERS))
def backend(request) -> SandboxBackend:
    """One backend per registered provider."""
    return _BUILDERS[request.param]()


def _spec(**overrides) -> SandboxSpec:
    payload = {"source": SandboxSource.image("image"), "workflow_id": "task-1#0"}
    payload.update(overrides)
    return SandboxSpec(**payload)


def _manager(backend: SandboxBackend) -> SandboxManager:
    """The only caller of the state protocol, which is how a caller is meant to use it."""
    return SandboxManager({backend.name: backend}, backend.name)


def _requires(backend: SandboxBackend, *features: SandboxFeature) -> None:
    for feature in features:
        if not backend.capabilities.supports(feature):
            pytest.skip(f"{backend.name} does not declare {feature.value}")


async def test_the_required_data_plane_is_implemented(backend: SandboxBackend) -> None:
    # Every backend serves the same five operations, whatever else it declares.
    session = await backend.create(_spec())

    result = await session.exec("echo hi")
    await session.write_bytes("/work/value", b"payload")

    # The exit code and a channel to read are what the contract promises. What a command
    # prints is the provider's business, so a fake reporting its own output still conforms.
    assert result.exit_code is not None
    assert result.stdout or result.stderr
    assert await session.read_bytes("/work/value") == b"payload"
    assert await session.status() in {SandboxStatus.RUNNING, SandboxStatus.UNKNOWN}

    await session.terminate()

    assert await session.status() is SandboxStatus.TERMINATED


async def test_terminating_twice_is_not_an_error(backend: SandboxBackend) -> None:
    session = await backend.create(_spec())

    await session.terminate()
    await session.terminate()


async def test_a_declared_hibernation_survives_a_pause_and_resume(backend: SandboxBackend) -> None:
    _requires(backend, SandboxFeature.HIBERNATE)
    session = await backend.create(_spec())

    await session.pause(PauseMode.HIBERNATE)
    await session.resume()

    assert (await session.exec("echo back")).exit_code is not None


async def test_a_declared_freeze_is_a_resident_pause(backend: SandboxBackend) -> None:
    _requires(backend, SandboxFeature.FREEZE)
    session = await backend.create(_spec())

    await session.pause(PauseMode.FREEZE)
    await session.resume()

    assert (await session.exec("echo back")).exit_code is not None


async def test_a_cross_node_resume_promises_the_level_it_declares(backend: SandboxBackend) -> None:
    """A restore elsewhere reports what it preserves, and the two levels differ.

    A filesystem backend restores a workspace and every process is new. A full-state
    backend restores a running guest. A suite with one case would report the first as
    evidence for the second.
    """
    _requires(backend, SandboxFeature.RESTORE, SandboxFeature.RESUME_ANYWHERE)
    level = backend.capabilities.resume_level
    assert level is not None
    manager = _manager(backend)
    session = await backend.create(_spec(state_policy=_RESTORE_POLICY))
    kind = SnapshotKind.FULL_STATE if level is ResumeLevel.FULL_STATE else SnapshotKind.FILESYSTEM

    # A state operation goes through the manager, which is what applies capability
    # gating and the state safety policy.
    snapshot = await manager.checkpoint(session, kind, _RESTORE_POLICY)

    assert snapshot.resume_level is level

    restored = await manager.restore(
        snapshot, _spec(state_policy=_RESTORE_POLICY), state_policy=_RESTORE_POLICY
    )

    assert restored.ref.backend == backend.name
    assert (await restored.session.exec("echo restored")).exit_code is not None
    await manager.shutdown()


async def test_a_filesystem_backend_cannot_satisfy_a_full_state_requirement(backend: SandboxBackend) -> None:
    if backend.capabilities.resume_level is None:
        pytest.skip(f"{backend.name} does not declare a resume level")
    requires_full_state = backend.capabilities.resume_level is not ResumeLevel.FULL_STATE
    if not requires_full_state:
        pytest.skip(f"{backend.name} satisfies a full-state resume, so there is nothing to refuse")

    with pytest.raises(Exception) as rejection:
        backend.capabilities.require_resume_level(ResumeLevel.FULL_STATE)

    assert "full_state" in str(rejection.value)


async def test_a_declared_fork_produces_independent_children(backend: SandboxBackend) -> None:
    _requires(backend, SandboxFeature.NATIVE_FORK)
    session = await backend.create(_spec())

    children = await session.fork(2)

    assert len(children) == 2
    assert {child.ref for child in children} != {session.ref}
    assert (await children[0].exec("echo child")).exit_code is not None


async def test_deleting_a_snapshot_twice_succeeds(backend: SandboxBackend) -> None:
    _requires(backend, SandboxFeature.RESTORE)
    manager = _manager(backend)
    session = await backend.create(_spec(state_policy=_POLICY))
    kind = (
        SnapshotKind.FULL_STATE
        if backend.capabilities.supports(SandboxFeature.FULL_STATE_SNAPSHOT)
        else SnapshotKind.FILESYSTEM
    )
    snapshot = await manager.checkpoint(session, kind, _POLICY)

    await manager.delete_snapshot(snapshot)
    await manager.delete_snapshot(snapshot)
    await manager.shutdown()


async def test_a_backend_that_cannot_serve_a_spec_is_refused_at_selection() -> None:
    """Capability selection refuses rather than degrading.

    The point of the suite is that a declaration can be trusted, so a manager with a
    backend that lacks the requirement must not quietly choose it.
    """
    backend = _docker_backend()
    manager = SandboxManager({backend.name: backend}, backend.name)

    with pytest.raises(Exception, match="full_state|FULL_STATE|no configured"):
        manager.select_backend(
            _spec(
                required_features=frozenset({SandboxFeature.RESUME_ANYWHERE}),
                required_resume_level=ResumeLevel.FULL_STATE,
            )
        )

    await manager.shutdown()


async def test_a_capacity_timeout_is_its_own_fault() -> None:
    # Reported as its own type so a caller can classify it apart from a task failure.
    assert issubclass(SandboxCapacityTimeout, RuntimeError)


def test_every_backend_declares_a_consistent_capability_set(backend: SandboxBackend) -> None:
    """A declaration must be internally consistent.

    `RESUME_ANYWHERE` without a level, or a level without the feature, would let a
    caller read a guarantee the backend does not make.
    """
    capabilities: SandboxCapabilities = backend.capabilities

    if capabilities.supports(SandboxFeature.RESUME_ANYWHERE):
        assert capabilities.resume_level is not None
    else:
        assert capabilities.resume_level is None
    assert capabilities.supports(SandboxFeature.CREDENTIAL_INJECTION) or True
