from pathlib import Path

import pytest
from omegaconf import OmegaConf
from psrl.sandbox.backends import DockerBackend
from psrl.sandbox.capacity import SandboxCapacityConfig
from psrl.sandbox.config import build_sandbox_manager


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
    assert capacity.classes["rollout"].guaranteed_share == 0.6
    assert capacity.classes["grader"].guaranteed_share == 0.25
    assert sum(share.guaranteed_share for share in capacity.classes.values()) < 1
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
