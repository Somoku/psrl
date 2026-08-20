from dataclasses import dataclass, field
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from psrl.sandbox.core import SandboxBackend
from psrl.sandbox.manager import SandboxManager


@dataclass
class SandboxManagerConfig:
    """Worker-local sandbox backend registry."""

    default_backend: str | None = None
    backends: dict[str, Any] = field(default_factory=dict)


def build_sandbox_manager(config: DictConfig | SandboxManagerConfig | None) -> SandboxManager | None:
    """Instantiate configured backends, or return ``None`` when disabled."""
    if config is None:
        return None
    default_backend = config.get("default_backend") if isinstance(config, DictConfig) else config.default_backend
    if default_backend is None:
        return None
    backend_configs = config.get("backends", {}) if isinstance(config, DictConfig) else config.backends
    if isinstance(backend_configs, DictConfig):
        raw_configs = OmegaConf.to_container(backend_configs, resolve=True)
    else:
        raw_configs = backend_configs or {}
    backends: dict[str, SandboxBackend] = {}
    for name, backend_config in raw_configs.items():
        if isinstance(backend_config, SandboxBackend):
            raise TypeError(
                f"Sandbox backend {name!r} is already instantiated; "
                "sandbox configuration must remain declarative until worker initialization."
            )
        backend = hydra.utils.instantiate(backend_config)
        if not isinstance(backend, SandboxBackend):
            raise TypeError(f"Configured sandbox backend {name!r} is not a SandboxBackend.")
        backends[name] = backend
    return SandboxManager(backends, default_backend)
