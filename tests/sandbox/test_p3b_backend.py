"""The p3b backend's conversion and refusal contract.

What is worth asserting here is the boundary, not the service: that a PSRL spec
becomes the service's spec without losing a field, that a spec carrying
something the service has no field for is refused rather than silently stripped,
and that the backend does not charge a worker's node envelope for a sandbox the
service placed.

The service itself is covered by the Go suite and by
`tests/sandbox/test_pysandbox_live.py`, which needs a running binary.
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest
from psrl.sandbox.backends import p3b as p3b_backend
from psrl.sandbox.backends.p3b import P3bBackend, _to_service_spec
from psrl.sandbox.core import (
    CredentialRef,
    EgressPolicy,
    ExecMode,
    MountSpec,
    ResourceSpec,
    ResumeLevel,
    SandboxFeature,
    SandboxSource,
    SandboxSpec,
)

_OWNER = "worker-1"
_CLASS = "rollout"


def make_backend() -> P3bBackend:
    return P3bBackend(endpoint="unix:///run/sandboxd-test.sock")


def spec(**overrides) -> SandboxSpec:
    base = {
        "source": SandboxSource.image("python:3.11-slim"),
        "resources": ResourceSpec(cpu_count=2, memory_mb=2048),
        "resource_class": "rollout",
    }
    return SandboxSpec(**(base | overrides))


def convert(s: SandboxSpec, owner: str = "", resource_class: str = _CLASS):
    """Call the module-level conversion function directly."""
    return _to_service_spec(s, owner, resource_class)


class TestItDoesNotChargeTheNodeTwice:
    def test_it_declines_the_node_envelope(self):
        # The service's ledger already bounds the fleet. Charging this worker's
        # node as well would refuse a sandbox twice for the same memory.
        assert not make_backend().uses_node_capacity


class TestTheSpecSurvivesTheConversion:
    def test_the_source_and_resources_carry_over(self):
        converted = convert(spec())

        assert converted.source.reference == "python:3.11-slim"
        assert converted.source.kind == "image"
        assert converted.resources.cpu_count == 2
        assert converted.resources.memory_mb == 2048

    def test_the_resource_class_carries_over(self):
        converted = convert(spec(resource_class="grader"), resource_class="rollout")

        assert converted.resource_class == "grader"

    def test_a_spec_with_no_class_takes_the_backend_default(self):
        converted = convert(spec(resource_class="default"), resource_class="rollout")

        assert converted.resource_class == "default"

    def test_env_and_metadata_carry_over(self):
        converted = convert(spec(env={"TASK": "swe"}, metadata={"run": "7"}))

        assert converted.env["TASK"] == "swe"
        assert converted.metadata["run"] == "7"

    def test_the_idempotency_key_carries_over(self):
        converted = convert(spec(idempotency_key="episode-12"))

        assert converted.idempotency_key == "episode-12"

    def test_the_workflow_id_carries_over(self):
        converted = convert(spec(workflow_id="group-3"))

        assert converted.workflow_id == "group-3"

    def test_the_owner_the_service_reclaims_by_travels_with_the_spec(self):
        """The owner id is the name the node reclaims this sandbox by.

        The service reads it from the spec's metadata, and this worker's renewal
        heartbeat names the same string. A rename on one side alone leaves every
        sandbox owned by an owner nothing renews, and a node then reclaims them
        while the worker is still using them.
        """
        converted = convert(spec(), owner="worker-2")

        assert converted.metadata["owner_id"] == "worker-2"

    def test_the_starting_directory_carries_over(self):
        # The service applies this to the container, so a command that names no
        # directory still starts there.
        converted = convert(spec(workdir="/workspace/repo"))

        assert converted.workdir == "/workspace/repo"

    def test_the_exec_mode_carries_over_unchanged(self):
        """The exec mode cannot be chosen later, only at creation.

        The service fixes a sandbox's strategy when it is created and the exec
        interface carries no mode, so a dropped mode is not a deferred decision --
        every harness that declared a persistent shell would be served one-shot,
        and would lose `cd`, `source` and its exported variables between turns.
        """
        for mode in ExecMode:
            assert convert(spec(exec_mode=mode)).exec_mode.value == mode.value

    def test_the_required_features_carry_over(self):
        # A feature left behind here is not merely unrequested, it is dropped, and
        # the sandbox lands somewhere that cannot honour it.
        converted = convert(
            spec(
                required_features=frozenset({SandboxFeature.FREEZE, SandboxFeature.RESUME_ANYWHERE}),
                required_resume_level=ResumeLevel.FULL_STATE,
            )
        )

        assert {feature.value for feature in converted.required_features} == {
            "freeze",
            "resume_anywhere",
        }
        assert converted.required_resume_level.value == "full_state"

    def test_the_backend_pin_carries_over(self):
        # The service validates the requirements against the pinned backend, so
        # forwarding the pin cannot weaken a check.
        converted = convert(spec(backend="docker"))

        assert converted.backend == "docker"

    def test_a_spec_that_states_nothing_stays_unset(self):
        # Forwarding must not invent a value: an unset mode, level or pin has to
        # arrive unset, or the service takes a decision the caller left to it.
        converted = convert(spec())

        assert converted.exec_mode is None
        assert converted.required_resume_level is None
        assert converted.backend is None
        assert converted.workdir is None
        assert not converted.required_features


class TestEverySpecFieldIsAccountedFor:
    """A field that reaches neither set is dropped without anyone deciding.

    That is how workdir, the exec mode, the required features, the resume level
    and the backend pin were all lost -- the service grew a field for each and the
    conversion was never revisited. Failing on a new PSRL field forces the decision
    to be made here, rather than discovered as a workload that quietly ran differently.
    """

    def test_the_three_sets_cover_the_whole_spec(self):
        declared = p3b_backend._FORWARDED_FIELDS | p3b_backend._OUT_OF_SCOPE_FIELDS
        actual = {field.name for field in dataclasses.fields(SandboxSpec)}

        assert declared == actual, (
            "a PSRL SandboxSpec field is not accounted for by the p3b conversion: "
            f"undecided {sorted(actual - declared)}, stale {sorted(declared - actual)}"
        )

    def test_the_forwarded_and_refused_sets_do_not_overlap(self):
        # A field in both sets is a contradiction that the exhaustiveness check
        # above cannot see on its own.
        assert not (p3b_backend._FORWARDED_FIELDS & p3b_backend._REFUSED_FIELDS)
        assert not (p3b_backend._FORWARDED_FIELDS & frozenset(p3b_backend._NOT_FORWARDED_FIELDS))


class TestWhatTheServiceCannotCarryIsRefused:
    """A dropped field runs and produces a wrong result; a refusal costs a fix."""

    def test_a_host_mount_is_refused(self):
        with pytest.raises(ValueError, match="bind mount"):
            convert(spec(mounts=(MountSpec(source="/data", target="/data"),)))

    def test_an_egress_policy_is_refused(self):
        with pytest.raises(ValueError, match="egress policy"):
            convert(spec(egress=EgressPolicy()))

    def test_injected_credentials_are_refused(self):
        with pytest.raises(ValueError, match="credentials"):
            convert(spec(credentials=(CredentialRef(source_env="HF_TOKEN", target_env="HF_TOKEN"),)))

    def test_pinned_gpus_are_refused(self):
        with pytest.raises(ValueError, match="GPU"):
            convert(spec(assigned_gpus=(0, 1)))


class TestCapabilitiesAreTheFleetsUnion:
    def test_it_declares_what_the_fleet_can_serve(self):
        declared = make_backend().capabilities

        assert declared.supports(SandboxFeature.FULL_STATE_SNAPSHOT)
        assert declared.supports(SandboxFeature.RESUME_ANYWHERE)
        assert declared.resume_level is ResumeLevel.FULL_STATE

    def test_a_host_mount_is_not_claimed(self):
        assert not make_backend().capabilities.supports(SandboxFeature.HOST_MOUNT)


class TestAReattachIsRefusedRatherThanFaked:
    async def test_connect_is_refused(self):
        with pytest.raises(NotImplementedError, match="agent endpoint"):
            await make_backend().connect("sb-1")


class _RecordingClient:
    """A stand-in for the SDK client that records renewals and closes."""

    def __init__(self) -> None:
        self.renewals: list[str] = []
        self.closed = False

    async def renew_owner(self, owner: str) -> None:
        self.renewals.append(owner)

    async def close(self) -> None:
        self.closed = True


class TestItRenewsOwnershipWhileItHoldsSandboxes:
    """The node reclaims an owner's sandboxes once it stops hearing from it.

    A sandbox's own lifetime is a backstop measured in hours, so until it expires
    a killed worker's containers stay up holding capacity. This renewal is the
    only thing that bounds how long that lasts, which is why it starts with the
    first sandbox and stops the moment the worker leaves.
    """

    async def test_it_renews_the_owner_on_a_timer(self, monkeypatch):
        monkeypatch.setattr(p3b_backend, "OWNER_HEARTBEAT_INTERVAL_S", 0.01)
        backend = make_backend()
        client = _RecordingClient()
        backend._client = client
        backend.adopt_owner_id(_OWNER)

        backend._ensure_owner_heartbeat()
        await asyncio.sleep(0.05)
        await backend.shutdown()

        assert client.renewals, "a worker holding sandboxes must renew its ownership"
        assert set(client.renewals) == {_OWNER}

    async def test_starting_it_more_than_once_keeps_one_loop(self, monkeypatch):
        monkeypatch.setattr(p3b_backend, "OWNER_HEARTBEAT_INTERVAL_S", 0.01)
        backend = make_backend()
        backend._client = _RecordingClient()
        backend.adopt_owner_id(_OWNER)

        backend._ensure_owner_heartbeat()
        first = backend._heartbeat_task
        backend._ensure_owner_heartbeat()

        assert backend._heartbeat_task is first, "every sandbox shares one renewal loop"
        await backend.shutdown()

    async def test_it_stops_renewing_when_it_shuts_down(self, monkeypatch):
        monkeypatch.setattr(p3b_backend, "OWNER_HEARTBEAT_INTERVAL_S", 0.01)
        backend = make_backend()
        client = _RecordingClient()
        backend._client = client
        backend.adopt_owner_id(_OWNER)

        backend._ensure_owner_heartbeat()
        await asyncio.sleep(0.05)
        await backend.shutdown()
        seen = len(client.renewals)
        await asyncio.sleep(0.05)

        assert len(client.renewals) == seen, "a closed backend must not keep renewing"
        assert backend._heartbeat_task is None
        assert client.closed

    async def test_a_worker_with_no_owner_renews_nothing(self, monkeypatch):
        # Without an owner id there is no name for the renewal, and the fleet
        # would have nothing to key liveness on.
        monkeypatch.setattr(p3b_backend, "OWNER_HEARTBEAT_INTERVAL_S", 0.01)
        backend = make_backend()
        client = _RecordingClient()
        backend._client = client

        backend._ensure_owner_heartbeat()
        await asyncio.sleep(0.03)
        await backend.shutdown()

        assert not client.renewals
