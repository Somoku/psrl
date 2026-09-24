"""Typed Docker security, workload policy, and disk admission configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from psrl.sandbox.core import rewrite_loopback_proxy

# Proxy variables a workload may inherit from the worker, whose loopback host means the
# worker's own namespace rather than the sandbox's.
PROXY_URL_ENV_KEYS = (
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
)


def append_no_proxy_alias(value: str, host_alias: str) -> str:
    """
    Add the gateway alias to a `no_proxy` list without duplicating an entry.
    """
    entries = [entry.strip() for entry in value.split(",") if entry.strip()]
    return ",".join(dict.fromkeys([*entries, host_alias]))


@dataclass(frozen=True)
class DockerSecurityConfig:
    """
    Security controls applied to every Docker sandbox.
    """

    require_rootless: bool = False
    pids_limit: int = 4096
    cap_drop: tuple[str, ...] = ("ALL",)
    cap_add: tuple[str, ...] = ()
    no_new_privileges: bool = True
    read_only_rootfs: bool = False
    seccomp_profile: str | None = None
    user: str | None = None
    tmpfs: Mapping[str, str] = field(default_factory=dict)
    # Inherited by the whole container, so it only shifts the host ranking. None keeps the
    # kernel free to kill a sandbox before the trainer sharing its node.
    oom_score_adj: int | None = None

    def __post_init__(self) -> None:
        if self.pids_limit <= 0:
            raise ValueError("Docker pids_limit must be greater than zero.")
        if self.oom_score_adj is not None and not -1000 <= self.oom_score_adj <= 1000:
            raise ValueError("Docker oom_score_adj must be within [-1000, 1000] or None.")


@dataclass(frozen=True)
class DockerPolicyProfile:
    """
    Typed per-workload Docker policy overrides.
    """

    network_mode: str | None = None
    extra_hosts: tuple[str, ...] = ()
    host_gateway_alias: str | None = None
    rewrite_loopback_proxies: bool = False
    pids_limit: int | None = None
    cap_drop: tuple[str, ...] | None = None
    cap_add: tuple[str, ...] | None = None
    no_new_privileges: bool | None = None
    read_only_rootfs: bool | None = None
    seccomp_profile: str | None = None
    user: str | None = None
    tmpfs: Mapping[str, str] = field(default_factory=dict)
    oom_score_adj: int | None = None
    runtime: str | None = None

    def __post_init__(self) -> None:
        if self.pids_limit is not None and self.pids_limit <= 0:
            raise ValueError("Docker policy pids_limit must be greater than zero.")
        if self.oom_score_adj is not None and not -1000 <= self.oom_score_adj <= 1000:
            raise ValueError("Docker policy oom_score_adj must be within [-1000, 1000] or None.")
        if self.rewrite_loopback_proxies and not self.host_gateway_alias:
            raise ValueError("Docker proxy rewriting requires host_gateway_alias.")

    @classmethod
    def from_value(cls, value: DockerPolicyProfile | Mapping[str, Any]) -> DockerPolicyProfile:
        """
        Normalize Hydra mappings into an immutable policy.
        """
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("Docker policy profiles must use typed mappings, not raw CLI arguments.")
        normalized = dict(value)
        for key in ("extra_hosts", "cap_drop", "cap_add"):
            if normalized.get(key) is not None:
                normalized[key] = tuple(normalized[key])
        return cls(**normalized)

    def resolve_callback_url(self, url: str) -> str:
        """
        Translate a worker URL into one the sandbox can reach.
        """
        if not self.host_gateway_alias:
            return url
        return rewrite_loopback_proxy(url, self.host_gateway_alias)


@dataclass(frozen=True)
class DockerDiskAdmissionConfig:
    """
    Host disk headroom required before a sandbox can be created.
    """

    path: str | None = None
    min_free_mb: int = 0
    wait_timeout_s: float = 300.0
    poll_interval_s: float = 5.0

    def __post_init__(self) -> None:
        if self.min_free_mb < 0 or self.wait_timeout_s < 0 or self.poll_interval_s <= 0:
            raise ValueError(
                "Docker disk min_free_mb and wait_timeout_s must not be negative, "
                "and poll_interval_s must be positive."
            )
        if self.min_free_mb > 0 and not self.path:
            raise ValueError("Docker disk path is required when min_free_mb is positive.")

    @classmethod
    def from_value(
        cls,
        value: DockerDiskAdmissionConfig | Mapping[str, Any] | None,
    ) -> DockerDiskAdmissionConfig:
        """
        Normalize a Hydra mapping into immutable disk admission policy.
        """
        if isinstance(value, cls):
            return value
        return cls(**dict(value or {}))
