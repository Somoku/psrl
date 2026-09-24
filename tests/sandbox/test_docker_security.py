"""Credential injection and the isolation runtime.

Both are security controls that must fail closed. A sandbox that silently runs
without the credential the caller asked for produces a reward that looks valid,
and one that silently falls back to the daemon's default runtime runs with less
isolation than the policy promised.
"""

from __future__ import annotations

import pytest
from psrl.sandbox import (
    CredentialRef,
    SandboxFeature,
    SandboxSource,
    SandboxSpec,
    SandboxStatePolicy,
)
from psrl.sandbox.backends.docker import DockerBackend
from psrl.sandbox.backends.docker.policy import DockerPolicyProfile
from psrl.sandbox.core import ExecMode

from tests.sandbox.test_docker_backend import FakeDockerEngine

pytestmark = pytest.mark.cpu_test


class RuntimeEngine(FakeDockerEngine):
    """A daemon that advertises a set of container runtimes."""

    def __init__(self, runtimes: list[str] | None = None) -> None:
        super().__init__()
        self.runtimes = runtimes or []

    async def info(self):
        return {"SecurityOptions": ["name=rootless"], "Runtimes": {name: {} for name in self.runtimes}}


def _backend(engine) -> DockerBackend:
    return DockerBackend(default_exec_mode=ExecMode.ONE_SHOT, engine=engine)


async def test_a_credential_is_resolved_from_this_process_not_off_the_spec(monkeypatch) -> None:
    monkeypatch.setenv("PSRL_TEST_API_TOKEN", "s3cret")
    engine = FakeDockerEngine()
    backend = _backend(engine)
    spec = SandboxSpec(
        SandboxSource.image("image"),
        credentials=(CredentialRef(source_env="PSRL_TEST_API_TOKEN", target_env="API_TOKEN"),),
    )

    await backend.create(spec)

    assert "API_TOKEN=s3cret" in engine.config["Env"]
    # The spec itself never carries the value, so it stays safe to log and compare.
    assert "s3cret" not in str(spec)


async def test_a_missing_credential_refuses_the_sandbox(monkeypatch) -> None:
    monkeypatch.delenv("PSRL_TEST_API_TOKEN", raising=False)
    engine = FakeDockerEngine()
    backend = _backend(engine)

    with pytest.raises(RuntimeError, match="is not set in this process"):
        await backend.create(
            SandboxSpec(
                SandboxSource.image("image"),
                credentials=(CredentialRef(source_env="PSRL_TEST_API_TOKEN", target_env="API_TOKEN"),),
            )
        )


async def test_an_injected_credential_is_named_to_the_state_policy(monkeypatch) -> None:
    # The spec cannot tell the manager that a secret was injected, so the session has
    # to say so, or a checkpoint would capture it with the workspace.
    monkeypatch.setenv("PSRL_TEST_API_TOKEN", "s3cret")
    engine = FakeDockerEngine()
    backend = _backend(engine)
    spec = SandboxSpec(
        SandboxSource.image("image"),
        credentials=(CredentialRef(source_env="PSRL_TEST_API_TOKEN", target_env="API_TOKEN"),),
        state_policy=SandboxStatePolicy(enabled=True),
    )

    session = await backend.create(spec)

    assert session.protected_env_names == frozenset({"API_TOKEN"})


def test_the_credential_feature_is_declared() -> None:
    assert _backend(FakeDockerEngine()).capabilities.supports(SandboxFeature.CREDENTIAL_INJECTION)


async def test_a_configured_isolation_runtime_reaches_the_container() -> None:
    engine = RuntimeEngine(["runsc"])
    backend = DockerBackend(default_exec_mode=ExecMode.ONE_SHOT, isolation_runtime="runsc", engine=engine)

    await backend.create(SandboxSpec(SandboxSource.image("image")))

    assert engine.config["HostConfig"]["Runtime"] == "runsc"
    assert backend.capabilities.supports(SandboxFeature.ISOLATION_RUNTIME)


async def test_an_absent_isolation_runtime_refuses_the_sandbox() -> None:
    engine = RuntimeEngine(["runc"])
    backend = DockerBackend(default_exec_mode=ExecMode.ONE_SHOT, isolation_runtime="runsc", engine=engine)

    with pytest.raises(RuntimeError, match="is not available on this daemon"):
        await backend.create(SandboxSpec(SandboxSource.image("image")))


def test_the_isolation_feature_is_declared_only_where_it_is_configured() -> None:
    # Advertising it without a runtime configured would admit a policy that asks for
    # stronger isolation and then run the workload without it.
    assert not _backend(FakeDockerEngine()).capabilities.supports(SandboxFeature.ISOLATION_RUNTIME)


async def test_a_policy_profile_selects_the_runtime_for_one_workload() -> None:
    engine = RuntimeEngine(["runsc"])
    backend = DockerBackend(
        default_exec_mode=ExecMode.ONE_SHOT,
        engine=engine,
        policy_profiles={"hardened": DockerPolicyProfile(runtime="runsc")},
    )

    await backend.create(SandboxSpec(SandboxSource.image("image"), policy_profile="hardened"))

    assert engine.config["HostConfig"]["Runtime"] == "runsc"


async def test_a_profile_runtime_is_checked_against_the_daemon_too() -> None:
    engine = RuntimeEngine(["runc"])
    backend = DockerBackend(
        default_exec_mode=ExecMode.ONE_SHOT,
        engine=engine,
        isolation_runtime="runsc",
        policy_profiles={"hardened": DockerPolicyProfile(runtime="runsc")},
    )

    with pytest.raises(RuntimeError, match="is not available on this daemon"):
        await backend.create(SandboxSpec(SandboxSource.image("image"), policy_profile="hardened"))
