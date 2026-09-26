from pathlib import Path

import pytest
from omegaconf import OmegaConf
from psrl.sandbox.backends import DockerBackend
from psrl.sandbox.capacity import SandboxCapacityConfig
from psrl.sandbox.config import SandboxManagerConfig, build_sandbox_manager, placement_manager_config
from psrl.sandbox.core import SandboxBackend, SandboxCapabilities


class InjectedBackend(SandboxBackend):
    """A backend a deployment can build in code, which is what a remote backend is."""

    def __init__(self, name: str = "fake") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> SandboxCapabilities:
        return SandboxCapabilities()

    async def create(self, spec):  # pragma: no cover, since the manager under test never calls it
        raise NotImplementedError

    async def connect(self, sandbox_id):  # pragma: no cover, for the same reason
        raise NotImplementedError


def test_hydra_backend_registry_is_hot_pluggable() -> None:
    config = OmegaConf.create(
        {
            "default_backend": "local_docker",
            "backends": {
                "local_docker": {
                    "_target_": "psrl.sandbox.backends.DockerBackend",
                    "name": "local_docker",
                    "docker_host": "unix:///var/run/docker.sock",
                    "lifecycle": {"gc_enabled": False},
                    "disk_admission": {"path": "/dockerdata", "min_free_mb": 1024},
                }
            },
        }
    )

    manager = build_sandbox_manager(config)

    assert manager is not None
    assert isinstance(manager.backend(), DockerBackend)
    assert manager.default_backend == "local_docker"
    assert manager.backend().lifecycle.config.docker_command == (
        "docker",
        "--host",
        "unix:///var/run/docker.sock",
    )
    assert manager.backend().disk_admission.min_free_mb == 1024


def test_null_backend_mapping_reports_missing_default_backend() -> None:
    config = OmegaConf.create({"default_backend": "docker", "backends": None})

    with pytest.raises(ValueError, match="not configured"):
        build_sandbox_manager(config)


def test_capacity_config_is_owned_by_worker_manager() -> None:
    config = OmegaConf.create(
        {
            "default_backend": "docker",
            "capacity": {
                "memory_mb": 64000,
                "cpu_cores": 32,
                "utilization": 0.5,
                "lease_ttl_s": 180,
                "heartbeat_interval_s": 30,
            },
            "backends": {
                "docker": {
                    "_target_": "psrl.sandbox.backends.DockerBackend",
                    "lifecycle": {"gc_enabled": False},
                }
            },
        }
    )

    manager = build_sandbox_manager(config, capacity_coordinator=object(), owner_id="worker-1")

    assert manager is not None
    assert manager._capacity_coordinator is not None
    assert manager._capacity_owner_id == "worker-1"
    assert manager._capacity_heartbeat_interval_s == 30


def test_capacity_defaults_are_single_envelope_knobs() -> None:
    assert SandboxCapacityConfig() == SandboxCapacityConfig(
        memory_mb=None,
        cpu_cores=None,
        utilization=0.5,
        lease_ttl_s=180,
        heartbeat_interval_s=30,
    )
    # An admission deadline is on by default: without it an exhausted node makes the trainer
    # wait forever instead of reporting a capacity fault.
    assert SandboxCapacityConfig().acquire_timeout_s is not None


def test_shipped_rollout_yaml_declares_the_capacity_classes() -> None:
    """Guard the hand-forked rollout config against drifting from the schema.

    A class name or deadline that the yaml spells differently would only surface at
    runtime, where it reads as an undeclared class or a stalled entry.
    """
    import yaml

    cfg = yaml.safe_load(Path("psrl/trainer/config/rollout/psrl_rollout.yaml").read_text())
    capacity = SandboxCapacityConfig(**cfg["agent"]["sandbox"]["capacity"])

    assert set(capacity.classes) == {"rollout", "grader"}
    # Assert the properties the coordinator relies on, not the exact shares: the split is
    # tuned from measured phase concurrency and is expected to move, while a grader left
    # without a real reserve is starved behind a rollout flood and a total at or above one
    # makes the guarantees unsatisfiable at the same time.
    assert sum(share.guaranteed_share for share in capacity.classes.values()) < 1
    assert capacity.classes["grader"].guaranteed_share > 0
    assert capacity.classes["rollout"].guaranteed_share > 0
    assert capacity.acquire_timeout_s is not None


def test_capacity_classes_and_deadline_reach_the_manager_config() -> None:
    config = OmegaConf.create(
        {
            "default_backend": "docker",
            "capacity": {
                "memory_mb": 64000,
                "cpu_cores": 32,
                "classes": {"rollout": {"guaranteed_share": 0.6}},
                "acquire_timeout_s": 900,
            },
            "backends": {
                "docker": {
                    "_target_": "psrl.sandbox.backends.DockerBackend",
                    "lifecycle": {"gc_enabled": False},
                }
            },
        }
    )

    manager = build_sandbox_manager(config, capacity_coordinator=object(), owner_id="worker-1")

    assert manager._capacity_coordinator is not None
    assert config.capacity.classes.rollout.guaranteed_share == 0.6
    assert config.capacity.acquire_timeout_s == 900


def test_a_backend_built_in_code_can_be_injected() -> None:
    # A remote backend needs its transport, which Hydra cannot synthesize from YAML, so the
    # manager has to accept one that a deployment built.
    remote = InjectedBackend()
    config = SandboxManagerConfig(default_backend="fake", backends={})

    manager = build_sandbox_manager(config, extra_backends=[remote])

    assert manager._backends == {"fake": remote}
    assert manager.default_backend == "fake"


def test_an_injected_backend_still_has_to_be_the_default_when_it_is_named() -> None:
    remote = InjectedBackend()
    config = SandboxManagerConfig(default_backend="docker", backends={})

    with pytest.raises(ValueError, match="not configured"):
        build_sandbox_manager(config, extra_backends=[remote])


def test_a_backend_cannot_be_both_configured_and_injected() -> None:
    # Both are keyed by the backend's own name, so one would silently shadow the other.
    config = SandboxManagerConfig(
        default_backend="fake",
        backends={"fake": {"_target_": "tests.sandbox.test_config.InjectedBackend"}},
    )

    with pytest.raises(ValueError, match="both configured and built in code"):
        build_sandbox_manager(config, extra_backends=[InjectedBackend()])


def test_a_placing_worker_holds_no_local_backend_and_keeps_its_timing() -> None:
    # A placed sandbox consumes another node's envelope, so a worker that also held the local
    # backend would charge itself for containers it never runs.
    config = OmegaConf.create(
        {
            "default_backend": "docker",
            "backends": {"docker": {"_target_": "psrl.sandbox.backends.DockerBackend"}},
            "capacity": {"utilization": 0.25, "acquire_timeout_s": 60},
            "timing": {"episode_deadline_s": 120},
        }
    )

    placed = placement_manager_config(config, backend_name="docker")

    assert placed.backends == {}
    assert placed.default_backend == "docker"
    assert placed.capacity.utilization == 0.25
    assert placed.capacity.acquire_timeout_s == 60
    assert placed.timing.episode_deadline_s == 120


def test_a_placing_worker_still_validates_the_timing_ladder() -> None:
    # The orderings are asserted where the manager is built, and a placed manager is built
    # the same way, so an inverted configuration fails here too.
    config = OmegaConf.create(
        {
            "default_backend": "docker",
            "backends": {},
            "timing": {"pause_window_s": 600.0, "reap_window_s": 60.0},
        }
    )

    placed = placement_manager_config(config, backend_name="docker")

    with pytest.raises(ValueError, match="shorter than"):
        build_sandbox_manager(placed, extra_backends=[InjectedBackend("docker")])
