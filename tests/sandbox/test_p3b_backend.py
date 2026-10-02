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

import pytest

from psrl.sandbox.backends.p3b import P3bBackend
from psrl.sandbox.core import (
    CredentialRef,
    EgressPolicy,
    MountSpec,
    ResourceSpec,
    ResumeLevel,
    SandboxFeature,
    SandboxSource,
    SandboxSpec,
)


def backend() -> P3bBackend:
    return P3bBackend(endpoint="unix:///run/sandboxd-test.sock")


def spec(**overrides) -> SandboxSpec:
    base = {
        "source": SandboxSource.image("python:3.11-slim"),
        "resources": ResourceSpec(cpu_count=2, memory_mb=2048),
        "resource_class": "rollout",
    }
    return SandboxSpec(**(base | overrides))


class TestItDoesNotChargeTheNodeTwice:
    def test_it_declines_the_node_envelope(self):
        # The service's ledger already bounds the fleet. Charging this worker's
        # node as well would refuse a sandbox twice for the same memory.
        assert not backend().uses_node_capacity


class TestTheSpecSurvivesTheConversion:
    def test_the_source_and_resources_carry_over(self):
        converted = backend()._to_service_spec(spec())

        assert converted.source.reference == "python:3.11-slim"
        assert converted.source.kind == "image"
        assert converted.resources.cpu_count == 2
        assert converted.resources.memory_mb == 2048

    def test_the_resource_class_carries_over(self):
        # The class decides which guarantee the sandbox draws on, so losing it
        # would charge the wrong share.
        converted = backend()._to_service_spec(spec(resource_class="grader"))

        assert converted.resource_class == "grader"

    def test_a_spec_with_no_class_takes_the_backend_default(self):
        converted = backend()._to_service_spec(spec(resource_class="default"))

        assert converted.resource_class == "default"

    def test_env_and_metadata_carry_over(self):
        converted = backend()._to_service_spec(spec(env={"TASK": "swe"}, metadata={"run": "7"}))

        assert converted.env["TASK"] == "swe"
        assert converted.metadata["run"] == "7"

    def test_the_idempotency_key_carries_over(self):
        # A retried create must not produce a second sandbox.
        converted = backend()._to_service_spec(spec(idempotency_key="episode-12"))

        assert converted.idempotency_key == "episode-12"

    def test_the_workflow_id_carries_over(self):
        converted = backend()._to_service_spec(spec(workflow_id="group-3"))

        assert converted.workflow_id == "group-3"

    def test_the_owner_is_carried_as_metadata_for_correlation(self):
        instance = backend()
        instance.adopt_owner_id("worker-2")

        converted = instance._to_service_spec(spec())

        assert converted.metadata["psrl_owner_id"] == "worker-2"


class TestWhatTheServiceCannotCarryIsRefused:
    """A dropped field runs and produces a wrong result; a refusal costs a fix."""

    def test_a_host_mount_is_refused(self):
        # The service may place on another machine, where this worker's path
        # does not exist.
        with pytest.raises(ValueError, match="host bind mount"):
            backend()._to_service_spec(spec(mounts=(MountSpec(source="/data", target="/data"),)))

    def test_an_egress_policy_is_refused(self):
        with pytest.raises(ValueError, match="egress policy"):
            backend()._to_service_spec(spec(egress=EgressPolicy()))

    def test_injected_credentials_are_refused(self):
        with pytest.raises(ValueError, match="credentials"):
            backend()._to_service_spec(
                spec(credentials=(CredentialRef(source_env="HF_TOKEN", target_env="HF_TOKEN"),))
            )

    def test_pinned_gpus_are_refused(self):
        # The service assigns its own devices on the node it chose.
        with pytest.raises(ValueError, match="pin host GPU"):
            backend()._to_service_spec(spec(assigned_gpus=(0, 1)))


class TestCapabilitiesAreTheFleetsUnion:
    def test_it_declares_what_the_fleet_can_serve(self):
        # A union rather than an intersection: the service routes a spec to a
        # backend that can meet it, so a full-state resume is servable by the
        # fleet even though Docker alone could not.
        declared = backend().capabilities

        assert declared.supports(SandboxFeature.FULL_STATE_SNAPSHOT)
        assert declared.supports(SandboxFeature.RESUME_ANYWHERE)
        assert declared.resume_level is ResumeLevel.FULL_STATE

    def test_a_host_mount_is_not_claimed(self):
        # Claimed capability and refused spec would contradict each other.
        assert not backend().capabilities.supports(SandboxFeature.HOST_MOUNT)


class TestAReattachIsRefusedRatherThanFaked:
    async def test_connect_is_refused(self):
        # The agent endpoint only travels with a create reply, so a session for
        # a sandbox this worker did not create cannot be built.
        with pytest.raises(NotImplementedError, match="agent endpoint"):
            await backend().connect("sb-1")
