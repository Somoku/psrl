from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from psrl.sandbox.capacity import SandboxCapacityConfig
from psrl.sandbox.core import SandboxBackend
from psrl.sandbox.manager import SandboxManager


@dataclass
class SandboxManagerConfig:
    """Worker-local sandbox backend registry."""

    default_backend: str = "docker"
    backends: dict[str, Any] = field(default_factory=dict)
    capacity: SandboxCapacityConfig = field(default_factory=SandboxCapacityConfig)


def _config_section(config: DictConfig | SandboxManagerConfig, name: str):
    """Read one optional section, tolerating a config that omits it.

    A minimal config that only names a backend is a valid deployment, so a missing
    section means defaults rather than an error about a key nobody mentioned.
    """
    if isinstance(config, DictConfig):
        return config.get(name)
    return getattr(config, name, None)


def resolve_capacity(config: DictConfig | SandboxManagerConfig) -> SandboxCapacityConfig:
    """Normalize the capacity envelope from a Hydra config or a typed value.

    Public because a node agent builds its own envelope from the same sandbox config
    a worker does. Deriving it there rather than passing it separately is what stops
    a node from being created with no admission at all.
    """
    capacity = _config_section(config, "capacity")
    if isinstance(capacity, DictConfig):
        capacity = OmegaConf.to_container(capacity, resolve=True)
    return capacity if isinstance(capacity, SandboxCapacityConfig) else SandboxCapacityConfig(**dict(capacity or {}))


def _lifecycle_timings(backend: SandboxBackend) -> tuple[float | None, float | None]:
    """Return a Docker backend's crash-recovery lease and sweep interval, if it has them."""
    lifecycle = getattr(backend, "lifecycle", None)
    settings = getattr(lifecycle, "config", None)
    if settings is None:
        return None, None
    return (
        getattr(settings, "lease_ttl_s", None),
        getattr(settings, "gc_interval_s", None),
    )


def assert_lifecycle_fits_capacity(
    capacity: SandboxCapacityConfig,
    lifecycle_lease_ttl_s: float | None,
    lifecycle_gc_interval_s: float | None,
) -> None:
    """Refuse a backend whose crash recovery outlives the reservation protecting the node.

    The lease the owner refreshes and the crash-recovery sweep are two clocks that must
    be ordered, and a backend's own lease is built by Hydra and is not known until the
    backend exists, so this is the one place that can assert it.

    Raises:
        ValueError: When recovery cannot complete inside the capacity lease.
    """
    if lifecycle_lease_ttl_s is None or lifecycle_gc_interval_s is None:
        return
    if lifecycle_lease_ttl_s + lifecycle_gc_interval_s >= capacity.lease_ttl_s:
        raise ValueError(
            f"Sandbox crash recovery (lease {lifecycle_lease_ttl_s:g}s + sweep {lifecycle_gc_interval_s:g}s) "
            f"must complete inside the capacity lease TTL ({capacity.lease_ttl_s:g}s), or a dead worker's "
            "containers outlive the reservation that protected the node."
        )


def placement_manager_config(
    config: DictConfig | SandboxManagerConfig,
    *,
    backend_name: str,
) -> SandboxManagerConfig:
    """Return the manager config for a worker whose sandboxes all live on other nodes.

    It holds no local backend, because the node agents own the daemons and the envelopes and
    this worker would otherwise account for containers it never runs. The capacity block
    still carries over, because it is what bounds the leases this worker takes out.
    """
    return SandboxManagerConfig(
        default_backend=backend_name,
        backends={},
        capacity=resolve_capacity(config),
    )


def build_sandbox_manager(
    config: DictConfig | SandboxManagerConfig,
    capacity_coordinator=None,
    owner_id: str | None = None,
    extra_backends: Sequence[SandboxBackend] = (),
) -> SandboxManager:
    """Instantiate the configured backends and bind the worker's node capacity.

    `extra_backends` carries backends a deployment built in code rather than declared,
    which is the case for any backend whose constructor takes a live object: a remote
    backend needs its transport, and Hydra cannot synthesize one from YAML.
    """
    backend_configs = _config_section(config, "backends") or {}
    if isinstance(backend_configs, DictConfig):
        backend_configs = OmegaConf.to_container(backend_configs, resolve=True)
    backends: dict[str, SandboxBackend] = {}
    for name, backend_config in backend_configs.items():
        if isinstance(backend_config, SandboxBackend):
            raise TypeError(
                f"Sandbox backend {name!r} is already instantiated; "
                "sandbox configuration must remain declarative until worker initialization."
            )
        backend = hydra.utils.instantiate(backend_config)
        if not isinstance(backend, SandboxBackend):
            raise TypeError(f"Configured sandbox backend {name!r} is not a SandboxBackend.")
        backends[name] = backend
    for backend in extra_backends:
        name = backend.name
        if name in backends:
            raise ValueError(
                f"Sandbox backend {name!r} is both configured and built in code. Rename one of them, "
                "because a backend is keyed by its own name."
            )
        backends[name] = backend
    if config.default_backend not in backends:
        raise ValueError(f"Default sandbox backend {config.default_backend!r} is not configured.")
    capacity = resolve_capacity(config)
    if capacity_coordinator is not None and not owner_id:
        raise ValueError("Sandbox capacity coordinator requires a non-empty owner_id.")
    for backend in backends.values():
        # A backend declared in YAML is built by Hydra, which knows nothing about who will
        # own it, and a backend that reads its owner from the environment only finds one
        # inside an agent loop worker. This is the single place that knows both the owner
        # and the backend set, so it is where the two are joined: without it a placed
        # sandbox carries no owner label, and every mechanism keyed on that label — the
        # startup sweep, the node reclaimer, teardown at exit — silently does nothing.
        # A backend that already knows its owner keeps it.
        adopt_owner = getattr(backend, "adopt_owner_id", None)
        if owner_id and callable(adopt_owner):
            adopt_owner(owner_id)
        assert_lifecycle_fits_capacity(capacity, *_lifecycle_timings(backend))
    return SandboxManager(
        backends,
        config.default_backend,
        capacity_coordinator=capacity_coordinator,
        capacity_owner_id=owner_id,
        capacity_heartbeat_interval_s=capacity.heartbeat_interval_s if capacity_coordinator is not None else None,
    )
