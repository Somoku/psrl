"""A pool of prepared containers, off the rollout critical path.

A cold create pays a pull, a container start, and the task's own setup. A pool
entry has already paid all three, so a claim adopts it instead of repeating the
work.

Two rules keep a pool from becoming a leak. An entry is keyed by the spec shape it
was prepared for, because a container cannot be repurposed for a different image,
and it has its own TTL and budget, because a pool that only ever grows is a node
that only ever fills up.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from psrl.sandbox.core import SandboxSpec

psrl_logger = logging.getLogger(__file__)

# The label that marks an entry as pooled rather than owned by an episode. Its owner
# heartbeat stays live, so it tells an operator why a container has no episode behind it.
WARM_POOL_LABEL = "psrl.warm_pool"


def pool_key(spec: SandboxSpec) -> str:
    """Return the identity a pool entry is prepared for.

    Everything that would force a different container is in the key: the image or
    template, the resources, the policy profile, the environment, and the mounts.
    Two specs that share a key can share a container, and two that do not cannot.
    """
    identity = {
        "source": [spec.source.kind.value, spec.source.reference],
        "cpu_count": spec.resources.cpu_count,
        "memory_mb": spec.resources.memory_mb,
        "disk_mb": spec.resources.disk_mb,
        "gpu_count": spec.resources.gpu_count,
        "workdir": spec.workdir,
        "policy_profile": spec.policy_profile,
        # Metadata becomes container labels, so two specs that differ only there would
        # otherwise adopt a container labelled for the other one.
        "metadata": sorted(spec.metadata.items()),
        "idempotency_key": spec.idempotency_key,
        "env": sorted(spec.env.items()),
        "mounts": sorted((mount.source, mount.target, mount.read_only) for mount in spec.mounts),
        "volumes": sorted((volume.name, volume.target, volume.read_only) for volume in spec.volumes),
        "exec_mode": spec.exec_mode.value if spec.exec_mode else None,
        "egress": (
            None
            if spec.egress is None
            else [
                spec.egress.default_action.value,
                [(rule.action.value, rule.target, list(rule.ports)) for rule in spec.egress.rules],
            ]
        ),
        "credential_bindings": [
            (binding.name, binding.credential, binding.hosts) for binding in spec.credential_bindings
        ],
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]


@dataclass(frozen=True)
class WarmPoolConfig:
    """How many prepared containers a node keeps, and for how long.

    Depth is a fraction of the rollout batch rather than an absolute count, because
    an operator can estimate how much of one step should skip its cold start and
    cannot estimate a number of containers.
    """

    enabled: bool = False
    # How many sandboxes one step runs. The pool depth follows from it, so an
    # operator states a number they know instead of one they cannot estimate.
    batch_size: int = 0
    # Share of one step's concurrent sandboxes that may sit in the pool.
    depth_fraction: float = 0.1
    # Upper bound, so a large batch does not reserve a whole node for prefetching.
    max_entries: int = 4
    ttl_s: float = 900.0

    def __post_init__(self) -> None:
        if not 0 <= self.depth_fraction <= 1:
            raise ValueError("Docker warm pool depth_fraction must be within [0, 1].")
        if self.max_entries < 0:
            raise ValueError("Docker warm pool max_entries must not be negative.")
        if self.ttl_s <= 0:
            raise ValueError("Docker warm pool ttl_s must be greater than zero.")

    def depth_for(self, concurrent_sandboxes: int) -> int:
        """
        Return how many entries a batch of this size may pool.
        """
        if not self.enabled or concurrent_sandboxes <= 0:
            return 0
        wanted = int(concurrent_sandboxes * self.depth_fraction)
        return max(0, min(self.max_entries, wanted))

    def depth(self) -> int:
        """
        Return the depth this deployment's batch size asks for.
        """
        return self.depth_for(self.batch_size)

    @classmethod
    def from_value(cls, value: WarmPoolConfig | Mapping[str, Any] | None) -> WarmPoolConfig:
        """
        Normalize a Hydra mapping into immutable pool configuration.
        """
        if isinstance(value, cls):
            return value
        return cls(**dict(value or {}))


@dataclass
class _PoolEntry:
    container_id: str
    key: str
    prepared_at: float


@dataclass
class WarmPoolSnapshot:
    """
    Pool state for the trainer's per-step hook.
    """

    entries: int = 0
    claimed: int = 0
    expired: int = 0
    refusals: int = 0
    keys: int = 0

    def as_dict(self) -> dict[str, float]:
        return {
            "warm_pool/entries": float(self.entries),
            "warm_pool/claimed": float(self.claimed),
            "warm_pool/expired": float(self.expired),
            "warm_pool/refusals": float(self.refusals),
            "warm_pool/keys": float(self.keys),
        }


class DockerWarmPool:
    """Prepared containers on one node, keyed by the spec shape they fit.

    The pool never builds or destroys anything itself. It asks its owner, which is
    the backend, because that is the only thing that knows how to make a container
    and how to be sure one is gone.
    """

    def __init__(
        self,
        config: WarmPoolConfig,
        *,
        prepare: Callable[[str, SandboxSpec], Awaitable[str]],
        destroy: Callable[[str], Awaitable[None]],
    ) -> None:
        self.config = config
        self._prepare = prepare
        self._destroy = destroy
        self._entries: dict[str, deque[_PoolEntry]] = {}
        self._claimed = 0
        self._expired = 0
        self._refusals = 0

    def keys(self) -> tuple[str, ...]:
        """
        Return the spec shapes this pool currently holds a container for.
        """
        return tuple(key for key, entries in self._entries.items() if entries)

    def size(self) -> int:
        """
        Return the number of unclaimed entries.
        """
        return sum(len(entries) for entries in self._entries.values())

    def depth_for(self, concurrent_sandboxes: int = 0) -> int:
        """
        Return the depth one batch is allowed, defaulting to the configured batch size.
        """
        return self.config.depth_for(concurrent_sandboxes or self.config.batch_size)

    def poolable(self, spec: SandboxSpec) -> bool:
        """Return whether a spec may be prepared ahead of an episode.

        Three shapes are refused. A credential, or a destination policy that brokers one,
        would hold that enforcement open before any episode exists. An idempotency key
        names one episode's container, and a pooled entry belongs to no episode yet. The
        pool targets the common case: an image, a resource request, and a task's setup,
        repeated across the steps of one run.
        """
        if not self.config.enabled:
            return False
        if spec.idempotency_key is not None:
            return False
        return spec.egress is None and not spec.credentials and not spec.credential_bindings

    async def fill(self, spec: SandboxSpec, *, want: int) -> int:
        """Prepare containers until the pool holds `want` for this spec shape.

        A prepare failure is counted rather than raised. Prefetching is an
        optimization, and a spec that cannot be prepared still runs on the ordinary
        create path.
        """
        if not self.config.enabled or want <= 0:
            return 0
        key = pool_key(spec)
        entries = self._entries.setdefault(key, deque())
        made = 0
        while len(entries) < want:
            try:
                container_id = await self._prepare(key, spec)
            except Exception:
                self._refusals += 1
                psrl_logger.warning(
                    f"Docker warm pool could not prepare an entry for key {key!r}. The spec still runs on "
                    "the ordinary create path.",
                    exc_info=True,
                )
                break
            entries.append(_PoolEntry(container_id=str(container_id), key=key, prepared_at=time.time()))
            made += 1
        return made

    async def claim(self, spec: SandboxSpec, *, now: float | None = None) -> str | None:
        """Adopt a prepared container for a spec, or report that none fits.

        A claim is only valid for an equivalent spec shape, and an expired entry is
        never handed out: its image may have been pruned, and adopting it would turn
        a cache miss into a confusing failure later.
        """
        if not self.config.enabled:
            return None
        current = time.time() if now is None else now
        await self.expire(now=current)
        entries = self._entries.get(pool_key(spec))
        if not entries:
            return None
        entry = entries.popleft()
        if not entries:
            self._entries.pop(entry.key, None)
        self._claimed += 1
        return entry.container_id

    async def expire(self, *, now: float | None = None) -> list[str]:
        """
        Destroy the entries past their TTL and return what was removed.
        """
        current = time.time() if now is None else now
        expired: list[str] = []
        for key, entries in list(self._entries.items()):
            while entries and current - entries[0].prepared_at > self.config.ttl_s:
                expired.append(entries.popleft().container_id)
            if not entries:
                self._entries.pop(key, None)
        for container_id in expired:
            try:
                await self._destroy(container_id)
            except Exception:
                psrl_logger.warning(f"Docker warm pool could not destroy {container_id!r}.", exc_info=True)
        self._expired += len(expired)
        return expired

    async def drain(self) -> None:
        """
        Destroy every entry, for example when the backend shuts down.
        """
        await self.expire(now=time.time() + self.config.ttl_s * 2)

    def snapshot(self) -> WarmPoolSnapshot:
        """
        Return pool state for the metrics sink.
        """
        return WarmPoolSnapshot(
            entries=self.size(),
            claimed=self._claimed,
            expired=self._expired,
            refusals=self._refusals,
            keys=len(self.keys()),
        )
