from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from psrl.sandbox.capacity import SandboxCapacityConfig
from psrl.sandbox.core import SandboxBackend
from psrl.sandbox.manager import SandboxManager


@dataclass(frozen=True)
class SandboxTimingConfig:
    """Idle windows, derived from the one number an operator can estimate.

    An operator knows roughly how long one episode takes. The pause window, the
    reap window, and the absolute lifetime all follow from it, in that order, so
    the ordering cannot be configured wrongly. A deployment that needs a
    different shape can still set the windows directly, and then the orderings
    are asserted rather than derived.
    """

    # How long one episode may take, including its grading phase.
    episode_deadline_s: float | None = None
    # How many episodes of silence before an idle sandbox releases its compute.
    pause_after_episodes: float = 2.0
    # How many pause windows before an idle sandbox is destroyed instead.
    reap_after_pause_windows: float = 3.0
    # How many reap windows before the absolute lifetime backstop fires.
    lifetime_after_reap_windows: float | None = 4.0
    # Explicit overrides. None derives from the deadline above.
    pause_window_s: float | None = None
    reap_window_s: float | None = None
    lifetime_s: float | None = None

    def __post_init__(self) -> None:
        for name in ("pause_after_episodes", "reap_after_pause_windows", "lifetime_after_reap_windows"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"Sandbox timing {name} must be at least one window.")
        if self.episode_deadline_s is not None and self.episode_deadline_s <= 0:
            raise ValueError("Sandbox timing episode_deadline_s must be greater than zero when set.")

    def resolve_pause_window_s(self) -> float | None:
        """Return the idle window after which a sandbox releases its compute."""
        if self.pause_window_s is not None:
            return self.pause_window_s
        if self.episode_deadline_s is None:
            return None
        return self.episode_deadline_s * self.pause_after_episodes

    def resolve_reap_window_s(self) -> float | None:
        """Return the idle window after which a sandbox is destroyed."""
        if self.reap_window_s is not None:
            return self.reap_window_s
        pause = self.resolve_pause_window_s()
        if pause is None:
            return None
        return pause * self.reap_after_pause_windows

    def resolve_lifetime_s(self) -> float | None:
        """Return the absolute lifetime backstop, whether or not the sandbox is idle."""
        if self.lifetime_s is not None:
            return self.lifetime_s
        reap = self.resolve_reap_window_s()
        if reap is None or self.lifetime_after_reap_windows is None:
            return None
        return reap * self.lifetime_after_reap_windows


@dataclass
class SandboxManagerConfig:
    """Worker-local sandbox backend registry."""

    default_backend: str = "docker"
    backends: dict[str, Any] = field(default_factory=dict)
    capacity: SandboxCapacityConfig = field(default_factory=SandboxCapacityConfig)
    timing: SandboxTimingConfig = field(default_factory=SandboxTimingConfig)


def assert_timing_orderings(
    capacity: SandboxCapacityConfig,
    timing: SandboxTimingConfig,
    lifecycle_lease_ttl_s: float | None = None,
    lifecycle_gc_interval_s: float | None = None,
) -> None:
    """Refuse a configuration whose timeouts are in an order the module cannot honor.

    Each of these is a rule the code relies on. A violation produces a slow leak or
    a sandbox reclaimed while in use, and neither failure points at the
    configuration, so this is asserted where the configuration is built rather
    than documented and hoped for.

    Raises:
        ValueError: When an ordering is inverted.
    """
    pause = timing.resolve_pause_window_s()
    reap = timing.resolve_reap_window_s()
    lifetime = timing.resolve_lifetime_s()
    if pause is not None and reap is not None and pause >= reap:
        raise ValueError(
            f"Sandbox idle pause window ({pause:g}s) must be shorter than the reap window ({reap:g}s), or a "
            "sandbox is destroyed before it is ever paused."
        )
    if reap is not None and lifetime is not None and reap >= lifetime:
        raise ValueError(
            f"Sandbox idle reap window ({reap:g}s) must be shorter than the absolute lifetime ({lifetime:g}s), "
            "or the lifetime backstop fires first and the reaper never runs."
        )
    if capacity.lease_ttl_s <= 0:
        raise ValueError("Sandbox capacity lease_ttl_s must be greater than zero.")
    if lifecycle_lease_ttl_s is not None and lifecycle_gc_interval_s is not None:
        # Crash recovery has to be strictly faster than the capacity lease, or a dead worker's
        # containers outlive the reservation and new work is admitted against spoken-for memory.
        if lifecycle_lease_ttl_s + lifecycle_gc_interval_s >= capacity.lease_ttl_s:
            raise ValueError(
                f"Sandbox crash recovery (lease {lifecycle_lease_ttl_s:g}s + sweep {lifecycle_gc_interval_s:g}s) "
                f"must complete inside the capacity lease TTL ({capacity.lease_ttl_s:g}s), or a dead worker's "
                "containers outlive the reservation that protected the node."
            )


def _config_section(config: DictConfig | SandboxManagerConfig, name: str):
    """Read one optional section, tolerating a config that omits it.

    A minimal config that only names a backend is a valid deployment, so a missing
    section means defaults rather than an error about a key nobody mentioned.
    """
    if isinstance(config, DictConfig):
        return config.get(name)
    return getattr(config, name, None)


def _resolve_capacity(config: DictConfig | SandboxManagerConfig) -> SandboxCapacityConfig:
    """Normalize the capacity envelope from a Hydra config or a typed value."""
    capacity = _config_section(config, "capacity")
    if isinstance(capacity, DictConfig):
        capacity = OmegaConf.to_container(capacity, resolve=True)
    return capacity if isinstance(capacity, SandboxCapacityConfig) else SandboxCapacityConfig(**dict(capacity or {}))


def _resolve_timing(config: DictConfig | SandboxManagerConfig) -> SandboxTimingConfig:
    """Normalize the derived idle windows from a Hydra config or a typed value."""
    timing = _config_section(config, "timing") or SandboxTimingConfig()
    if isinstance(timing, DictConfig):
        timing = OmegaConf.to_container(timing, resolve=True)
    return timing if isinstance(timing, SandboxTimingConfig) else SandboxTimingConfig(**dict(timing))


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


def placement_manager_config(
    config: DictConfig | SandboxManagerConfig,
    *,
    backend_name: str,
) -> SandboxManagerConfig:
    """Return the manager config for a worker whose sandboxes all live on other nodes.

    It holds no local backend, because the node agents own the daemons and the envelopes and
    this worker would otherwise account for containers it never runs. The capacity and timing
    blocks still carry over: capacity bounds a node, and the idle policy is about this
    worker's own leases either way.
    """
    return SandboxManagerConfig(
        default_backend=backend_name,
        backends={},
        capacity=_resolve_capacity(config),
        timing=_resolve_timing(config),
    )


def build_sandbox_manager(
    config: DictConfig | SandboxManagerConfig,
    capacity_coordinator=None,
    owner_id: str | None = None,
    extra_backends: Sequence[SandboxBackend] = (),
) -> SandboxManager:
    """Instantiate the configured backends and bind the worker's node capacity.

    Every timeout ordering the module relies on is asserted here rather than
    documented and hoped for. A violation produces a slow leak or a sandbox
    reclaimed while it is in use, and neither failure points at the configuration.

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
    capacity = _resolve_capacity(config)
    timing = _resolve_timing(config)
    if capacity_coordinator is not None and not owner_id:
        raise ValueError("Sandbox capacity coordinator requires a non-empty owner_id.")
    for backend in backends.values():
        lifecycle_lease_ttl_s, lifecycle_gc_interval_s = _lifecycle_timings(backend)
        assert_timing_orderings(
            capacity,
            timing,
            lifecycle_lease_ttl_s=lifecycle_lease_ttl_s,
            lifecycle_gc_interval_s=lifecycle_gc_interval_s,
        )
    manager = SandboxManager(
        backends,
        config.default_backend,
        capacity_coordinator=capacity_coordinator,
        capacity_owner_id=owner_id,
        capacity_heartbeat_interval_s=capacity.heartbeat_interval_s if capacity_coordinator is not None else None,
    )
    manager.configure_idle_policy(timing.resolve_pause_window_s(), timing.resolve_reap_window_s())
    manager.configure_lifetime(timing.resolve_lifetime_s())
    return manager
