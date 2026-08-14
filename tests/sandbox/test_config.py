import pytest
from omegaconf import OmegaConf
from psrl.sandbox.backends import DockerBackend
from psrl.sandbox.config import build_sandbox_manager


def test_disabled_manager_does_not_instantiate_backends() -> None:
    config = OmegaConf.create({"default_backend": None, "backends": {}})

    assert build_sandbox_manager(config) is None


def test_hydra_backend_registry_is_hot_pluggable() -> None:
    config = OmegaConf.create(
        {
            "default_backend": "local_docker",
            "backends": {
                "local_docker": {
                    "_target_": "psrl.sandbox.backends.DockerBackend",
                    "name": "local_docker",
                    "docker_host": "unix:///var/run/docker.sock",
                }
            },
        }
    )

    manager = build_sandbox_manager(config)

    assert manager is not None
    assert isinstance(manager.backend(), DockerBackend)
    assert manager.default_backend == "local_docker"


def test_null_backend_mapping_reports_missing_default_backend() -> None:
    config = OmegaConf.create({"default_backend": "docker", "backends": None})

    with pytest.raises(ValueError, match="not configured"):
        build_sandbox_manager(config)
