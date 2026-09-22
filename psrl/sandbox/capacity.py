"""Node-wide weighted admission for local sandbox runtimes."""

from __future__ import annotations

import asyncio
import functools
import logging
import math
import os
import re
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

from psrl.sandbox.core import ResourceSpec, SandboxCapacityTimeout

psrl_logger = logging.getLogger(__file__)

_BYTES_PER_MIB = 1024 * 1024
_UNLIMITED_BYTES = 1 << 60
_MEMORY_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([kmgt]?)b?")


def parse_memory_mb(value: str | int | None) -> int | None:
    """Parse a Docker memory value into whole MiB."""
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return max(1, math.ceil(value / _BYTES_PER_MIB))
    match = _MEMORY_RE.fullmatch(str(value).strip().lower())
    if match is None:
        raise ValueError(f"Invalid sandbox memory limit: {value!r}.")
    amount = float(match.group(1))
    multiplier = {"": 1 / _BYTES_PER_MIB, "k": 1 / 1024, "m": 1, "g": 1024, "t": 1024 * 1024}
    return max(1, math.ceil(amount * multiplier[match.group(2)]))


def _read_int(path: str) -> int | None:
    try:
        with open(path) as handle:
            raw = handle.read().strip()
    except OSError:
        return None
    if not raw or raw == "max":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _cgroup_paths(controller: str, filename: str) -> list[str]:
    paths: list[str] = []
    try:
        with open("/proc/self/cgroup") as handle:
            lines = list(handle)
    except OSError:
        lines = []
    for line in lines:
        parts = line.strip().split(":", 2)
        if len(parts) != 3:
            continue
        group = parts[2].strip("/")
        if parts[0] == "0" and parts[1] == "":
            paths.append(f"/sys/fs/cgroup/{group}/{filename}")
        elif controller in parts[1].split(","):
            paths.append(f"/sys/fs/cgroup/{controller}/{group}/{filename}")
    return paths


@functools.lru_cache(maxsize=1)
def detect_node_memory_mb() -> int | None:
    """Return the local cgroup or machine memory limit in MiB."""
    paths = _cgroup_paths("memory", "memory.max") + _cgroup_paths("memory", "memory.limit_in_bytes")
    paths += ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]
    for path in paths:
        value = _read_int(path)
        if value is not None and 0 < value < _UNLIMITED_BYTES:
            return value // _BYTES_PER_MIB
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


@functools.lru_cache(maxsize=1)
def detect_node_cpu_cores() -> float | None:
    """Return the local cgroup or CPU-affinity capacity in cores."""
    candidates: list[float] = []
    try:
        candidates.append(float(len(os.sched_getaffinity(0))))
    except AttributeError:
        if os.cpu_count():
            candidates.append(float(os.cpu_count()))

    for path in _cgroup_paths("cpu", "cpu.max") + ["/sys/fs/cgroup/cpu.max"]:
        try:
            with open(path) as handle:
                quota, period = handle.read().split()[:2]
            if quota != "max":
                candidates.append(int(quota) / int(period))
        except (OSError, ValueError, IndexError):
            continue

    quota_paths = _cgroup_paths("cpu", "cpu.cfs_quota_us") + ["/sys/fs/cgroup/cpu/cpu.cfs_quota_us"]
    period_paths = _cgroup_paths("cpu", "cpu.cfs_period_us") + ["/sys/fs/cgroup/cpu/cpu.cfs_period_us"]
    for quota_path, period_path in zip(quota_paths, period_paths, strict=False):
        quota = _read_int(quota_path)
        period = _read_int(period_path)
        if quota is not None and period is not None and quota > 0 and period > 0:
            candidates.append(quota / period)
    return min(candidates) if candidates else None


@dataclass(frozen=True)
class ResourceClassShare:
    """One sandbox resource class's slice of the node envelope.

    ``guaranteed_share`` is reclaimed for this class whenever one of its requests
    is queued, so a class can never be starved by a different class that arrives
    first or asks for less. ``max_share`` optionally caps everything the class may
    hold at once, which bounds how much of the elastic pool it can borrow. Both
    are fractions of the resolved envelope.
    """

    guaranteed_share: float
    max_share: float | None = None

    def __post_init__(self) -> None:
        if not 0 < self.guaranteed_share <= 1:
            raise ValueError("Resource class guaranteed_share must be in (0, 1].")
        if self.max_share is not None and not self.guaranteed_share <= self.max_share <= 1:
            raise ValueError("Resource class max_share must be in [guaranteed_share, 1].")


def resolve_class_shares(
    value: Mapping[str, ResourceClassShare | Mapping[str, object]] | None,
) -> dict[str, ResourceClassShare]:
    """Normalize a class mapping from typed values or a resolved Hydra config."""
    if not value:
        return {}
    shares: dict[str, ResourceClassShare] = {}
    for name, share in value.items():
        key = str(name).strip()
        if not key:
            raise ValueError("Sandbox capacity class name cannot be empty.")
        shares[key] = share if isinstance(share, ResourceClassShare) else ResourceClassShare(**dict(share))
    return shares


@dataclass(frozen=True)
class SandboxCapacityConfig:
    """One node resource envelope shared by all local sandboxes.

    Unset memory or CPU selects node-local auto-detection. ``utilization`` is the
    only safety knob and applies to both resources.

    ``classes`` gives each sandbox resource class a guaranteed share of the
    envelope. The guarantees are deliberately allowed to sum to less than one:
    what is left over is a single elastic pool that a class may borrow from only
    while no other class has a request waiting. Keeping the sum below one is what
    keeps every guarantee simultaneously satisfiable without preempting a running
    sandbox, and it is why an admission preference for smaller requests is not
    needed. ``acquire_timeout_s`` bounds how long a request may stay queued and
    defaults to a finite value, so an under-provisioned node reports a capacity
    fault instead of hanging.

    ``classes`` is a builtin generic dict rather than a ``Mapping`` because
    OmegaConf rejects ``Mapping`` annotations when it builds a structured schema
    for this config, and a config that cannot be composed is a startup failure
    rather than a runtime one.
    """

    memory_mb: int | None = None
    cpu_cores: float | None = None
    utilization: float = 0.5
    lease_ttl_s: float = 180
    heartbeat_interval_s: float = 30
    classes: dict[str, ResourceClassShare] = field(default_factory=dict)
    acquire_timeout_s: float | None = 1800.0
    # Safety net for a lease whose release was missed. `lease_ttl_s` reclaims a lease of a
    # worker that stopped renewing, which cannot happen while the worker is alive and leaking:
    # the lease is renewed forever by an owner that will never release it again, so nothing
    # else in the design can free it. This age cap is the only recovery path, and it must
    # exceed the longest legitimate sandbox lifetime or it would reclaim a working sandbox.
    lease_max_age_s: float | None = 6 * 3600.0

    def __post_init__(self) -> None:
        if self.memory_mb is not None and self.memory_mb <= 0:
            raise ValueError("Sandbox capacity memory_mb must be greater than zero when configured.")
        if self.cpu_cores is not None and self.cpu_cores <= 0:
            raise ValueError("Sandbox capacity cpu_cores must be greater than zero when configured.")
        if not 0 < self.utilization <= 1:
            raise ValueError("Sandbox capacity utilization must be in (0, 1].")
        if self.lease_ttl_s <= 0 or not 0 < self.heartbeat_interval_s < self.lease_ttl_s:
            raise ValueError("Sandbox capacity heartbeat must be positive and shorter than lease_ttl_s.")
        object.__setattr__(self, "classes", resolve_class_shares(self.classes))
        guaranteed_total = sum(share.guaranteed_share for share in self.classes.values())
        if guaranteed_total > 1 + 1e-9:
            raise ValueError(
                f"Sandbox capacity class guarantees sum to {guaranteed_total:g}, which exceeds the node "
                "envelope. Keep the sum below one so every guarantee stays satisfiable without preempting "
                "a running sandbox."
            )
        if self.acquire_timeout_s is not None and self.acquire_timeout_s <= 0:
            raise ValueError("Sandbox capacity acquire_timeout_s must be greater than zero when configured.")
        if self.lease_max_age_s is not None and self.lease_max_age_s <= self.lease_ttl_s:
            raise ValueError(
                f"Sandbox capacity lease_max_age_s={self.lease_max_age_s} must exceed lease_ttl_s="
                f"{self.lease_ttl_s}, or a lease of a live worker could be reclaimed before its heartbeat "
                "mechanism is given a chance to recover it."
            )


@dataclass(frozen=True)
class ResourceQuantity:
    """Integer resource quantity used for exact accounting."""

    memory_mb: int
    cpu_millis: int

    @classmethod
    def from_spec(cls, resources: ResourceSpec) -> ResourceQuantity:
        """Create a quantity from a complete sandbox resource request."""
        if resources.memory_mb is None or resources.cpu_count is None:
            raise ValueError("Node-capacity admission requires sandbox memory_mb and cpu_count.")
        return cls(resources.memory_mb, math.ceil(resources.cpu_count * 1000))

    def fits(self, available_capacity: ResourceQuantity) -> bool:
        return self.memory_mb <= available_capacity.memory_mb and self.cpu_millis <= available_capacity.cpu_millis

    def __add__(self, other: ResourceQuantity) -> ResourceQuantity:
        return ResourceQuantity(self.memory_mb + other.memory_mb, self.cpu_millis + other.cpu_millis)

    def __sub__(self, other: ResourceQuantity) -> ResourceQuantity:
        return ResourceQuantity(self.memory_mb - other.memory_mb, self.cpu_millis - other.cpu_millis)


@dataclass(frozen=True)
class ResolvedSandboxCapacity:
    """Resolved node envelope and its provenance."""

    resources: ResourceQuantity
    memory_source: str
    cpu_source: str


def resolve_sandbox_capacity(
    config: SandboxCapacityConfig,
    *,
    detected_memory_mb: int | None = None,
    detected_cpu_cores: float | None = None,
) -> ResolvedSandboxCapacity:
    """Resolve and scale the node envelope once on the target node."""
    memory_mb = config.memory_mb or detected_memory_mb or detect_node_memory_mb()
    cpu_cores = config.cpu_cores or detected_cpu_cores or detect_node_cpu_cores()
    if not memory_mb or not cpu_cores:
        raise RuntimeError(
            "Sandbox node capacity could not be detected; configure memory_mb and cpu_cores explicitly."
        )
    return ResolvedSandboxCapacity(
        resources=ResourceQuantity(
            memory_mb=max(1, math.floor(memory_mb * config.utilization)),
            cpu_millis=max(1, math.floor(cpu_cores * 1000 * config.utilization)),
        ),
        memory_source="configured" if config.memory_mb is not None else "detected",
        cpu_source="configured" if config.cpu_cores is not None else "detected",
    )


def _scaled(base: ResourceQuantity, share: float) -> ResourceQuantity:
    """Return one share of an envelope, floored to whole units."""
    return ResourceQuantity(
        memory_mb=max(1, math.floor(base.memory_mb * share)),
        cpu_millis=max(1, math.floor(base.cpu_millis * share)),
    )


@dataclass
class _Allocation:
    owner_id: str
    resource_class: str
    resources: ResourceQuantity
    allocated_at: float


@dataclass
class _Waiter:
    lease_id: str
    owner_id: str
    resource_class: str
    resources: ResourceQuantity
    future: asyncio.Future[None]
    queued_at: float
    # Set by the sweeper when the owner lease expired, so `acquire` can report a
    # capacity fault instead of the bare cancellation that `future.cancel()` raises.
    expired: bool = False


class SandboxCapacityCoordinator:
    """Single-node, class-aware admission coordinator.

    Every request names the resource class of the sandbox it will create. Each
    class owns a FIFO queue and a guaranteed share of the envelope, so the order
    classes happen to arrive in can never starve one of them: a request inside
    its own guarantee is admitted as soon as the envelope has room, whatever the
    other classes are doing. Guarantees leave an elastic pool when they sum to
    less than one, and a class may borrow from that pool only while no other
    class has a request waiting. A class that was never declared in
    ``SandboxCapacityConfig.classes`` has no guarantee, so it can only use the
    elastic pool; the coordinator warns once so the omission is visible instead
    of silently starving that class.
    """

    def __init__(self, config: SandboxCapacityConfig | dict) -> None:
        if isinstance(config, dict):
            config = SandboxCapacityConfig(**config)
        self._config = config
        self._capacity = resolve_sandbox_capacity(config)
        self.available_capacity = self._capacity.resources
        self._guaranteed = {
            name: _scaled(self._capacity.resources, share.guaranteed_share)
            for name, share in config.classes.items()
        }
        self._ceilings = {
            name: _scaled(self._capacity.resources, share.max_share)
            for name, share in config.classes.items()
            if share.max_share is not None
        }
        self._allocations: dict[str, _Allocation] = {}
        self._used_by_class: dict[str, ResourceQuantity] = {}
        self._owners: dict[str, float] = {}
        self._waiters: dict[str, _Waiter] = {}
        self._class_queues: dict[str, deque[str]] = {}
        self._waiting_by_class: dict[str, int] = {}
        self._undeclared_classes: set[str] = set()
        self._lock = asyncio.Lock()
        self._sweeper: asyncio.Task[None] | None = None
        self._granted = 0
        self._released = 0
        self._expired = 0
        self._wait_timeouts = 0
        self._expired_waiters = 0
        self._stale_leases_reclaimed = 0
        self._total_wait_s = 0.0
        self._granted_by_class: dict[str, int] = {}
        self._wait_s_by_class: dict[str, float] = {}
        guaranteed_summary = ", ".join(
            f"{name}: {share.memory_mb}MiB" for name, share in sorted(self._guaranteed.items())
        )
        psrl_logger.info(
            f"Sandbox node envelope: memory_mb={self.available_capacity.memory_mb}, "
            f"cpu_cores={self.available_capacity.cpu_millis / 1000:g}, "
            f"sources={self._capacity.memory_source!r}/{self._capacity.cpu_source!r}, "
            f"class_guarantees={{{guaranteed_summary}}}, "
            f"acquire_timeout_s={self._config.acquire_timeout_s!r}."
        )

    def _ensure_sweeper(self) -> None:
        if self._sweeper is None:
            self._sweeper = asyncio.create_task(self._sweep_expired())

    async def acquire(
        self,
        lease_id: str,
        owner_id: str,
        memory_mb: int,
        cpu_count: float,
        resource_class: str = "default",
        timeout_s: float | None = None,
    ) -> None:
        """Wait until this request's class can fit its complete resource vector.

        Args:
            lease_id (str): Unique capacity lease id for the new sandbox.
            owner_id (str): Worker that owns the lease and refreshes it.
            memory_mb (int): Requested container memory in MiB.
            cpu_count (float): Requested container CPUs.
            resource_class (str): Role of the sandbox, used to pick its guarantee.
            timeout_s (float | None): Queue deadline. ``None`` uses the configured
                ``acquire_timeout_s``.

        Raises:
            ValueError: When the request exceeds the envelope or the lease id exists.
            SandboxCapacityTimeout: When the deadline expires, or when the owner
                lease expired while the request was still queued.
            asyncio.CancelledError: When the caller cancelled the wait.
        """
        self._ensure_sweeper()
        resource_class = resource_class.strip()
        if not resource_class:
            raise ValueError("Sandbox capacity resource_class cannot be empty.")
        resources = ResourceQuantity(memory_mb, math.ceil(cpu_count * 1000))
        if not resources.fits(self._capacity.resources):
            raise ValueError(
                f"Sandbox request memory_mb={memory_mb}, cpu_count={cpu_count:g} exceeds node envelope "
                f"memory_mb={self._capacity.resources.memory_mb}, "
                f"cpu_cores={self._capacity.resources.cpu_millis / 1000:g}."
            )
        deadline = self._config.acquire_timeout_s if timeout_s is None else timeout_s
        loop = asyncio.get_running_loop()
        waiter = _Waiter(
            lease_id,
            owner_id,
            resource_class,
            resources,
            loop.create_future(),
            time.monotonic(),
        )
        async with self._lock:
            if lease_id in self._allocations or lease_id in self._waiters:
                raise ValueError(f"Sandbox capacity lease {lease_id!r} already exists.")
            self._warn_undeclared_class(resource_class)
            self._owners[owner_id] = time.monotonic() + self._config.lease_ttl_s
            self._enqueue(waiter)
            self._drain_waiters()
        try:
            if deadline is None:
                await waiter.future
            else:
                await asyncio.wait_for(waiter.future, timeout=deadline)
        except asyncio.TimeoutError:
            waited = time.monotonic() - waiter.queued_at
            async with self._lock:
                self._wait_timeouts += 1
                self._discard_waiter(lease_id)
                self._drain_waiters()
            raise SandboxCapacityTimeout(
                f"Sandbox lease {lease_id!r} of resource_class={resource_class!r} was not admitted within "
                f"{deadline:g}s (waited {waited:.0f}s). Envelope memory_mb={self._capacity.resources.memory_mb}, "
                f"cpu_cores={self._capacity.resources.cpu_millis / 1000:g}. This is a capacity planning fault, "
                "not a task failure: raise the envelope, lower utilization, or rebalance capacity.classes."
            ) from None
        except asyncio.CancelledError:
            async with self._lock:
                expired = waiter.expired
                if expired:
                    self._expired_waiters += 1
                self._discard_waiter(lease_id)
                self._drain_waiters()
            if expired:
                raise SandboxCapacityTimeout(
                    f"Sandbox lease {lease_id!r} of resource_class={resource_class!r} was still queued when its "
                    f"owner lease {owner_id!r} expired, so the request lost its place. The worker stopped "
                    "refreshing capacity heartbeats."
                ) from None
            raise

    async def release(self, lease_id: str) -> None:
        """Release one admitted resource vector."""
        async with self._lock:
            if self._release_locked(lease_id):
                self._released += 1
                self._drain_waiters()

    async def cancel(self, lease_id: str) -> None:
        """Remove a canceled request whether it is queued or already admitted."""
        async with self._lock:
            waiter = self._waiters.pop(lease_id, None)
            if waiter is not None:
                self._dequeue(waiter)
                waiter.future.cancel()
            if self._release_locked(lease_id):
                self._released += 1
            self._drain_waiters()

    async def renew_owner(self, owner_id: str) -> None:
        """Renew every active lease belonging to one worker."""
        async with self._lock:
            if owner_id in self._owners:
                self._owners[owner_id] = time.monotonic() + self._config.lease_ttl_s

    async def release_owner(self, owner_id: str) -> None:
        """Release all capacity held by a worker that is shutting down."""
        async with self._lock:
            self._owners.pop(owner_id, None)
            lease_ids = [lease_id for lease_id, item in self._allocations.items() if item.owner_id == owner_id]
            for lease_id in lease_ids:
                self._release_locked(lease_id)
            owner_waiters = [waiter for waiter in self._waiters.values() if waiter.owner_id == owner_id]
            for waiter in owner_waiters:
                self._waiters.pop(waiter.lease_id)
                self._dequeue(waiter)
                waiter.future.cancel()
            self._released += len(lease_ids)
            self._drain_waiters()

    async def snapshot(self) -> dict:
        """Return a compact accounting and queue snapshot."""
        async with self._lock:
            return {
                "capacity": asdict(self._capacity.resources),
                "available_capacity": asdict(self.available_capacity),
                "allocations": len(self._allocations),
                "owners": len(self._owners),
                "waiters": len(self._waiters),
                "granted": self._granted,
                "released": self._released,
                "expired": self._expired,
                "mean_wait_s": self._total_wait_s / self._granted if self._granted else 0.0,
                "memory_source": self._capacity.memory_source,
                "cpu_source": self._capacity.cpu_source,
                "capacity_wait_timeouts": self._wait_timeouts,
                "expired_waiters": self._expired_waiters,
                "stale_leases_reclaimed": self._stale_leases_reclaimed,
                "oldest_lease_age_s": self._oldest_lease_age_s(),
                "per_class": self._class_snapshot(),
            }

    def _oldest_lease_age_s(self) -> float:
        """Return the age of the longest-held lease, which exposes a leak as it grows."""
        if not self._allocations:
            return 0.0
        now = time.monotonic()
        return max(now - item.allocated_at for item in self._allocations.values())

    def _class_snapshot(self) -> dict[str, dict]:
        """Return guarantee, usage, and queue depth for every class seen so far."""
        names = set(self._guaranteed) | set(self._used_by_class) | set(self._waiting_by_class)
        names |= self._undeclared_classes
        return {
            name: {
                "guaranteed": asdict(self._guaranteed.get(name, ResourceQuantity(0, 0))),
                "ceiling": asdict(self._ceilings[name]) if name in self._ceilings else None,
                "used": asdict(self._used_by_class.get(name, ResourceQuantity(0, 0))),
                "waiters": self._waiting_by_class.get(name, 0),
                "granted": self._granted_by_class.get(name, 0),
                "mean_wait_s": (
                    self._wait_s_by_class.get(name, 0.0) / self._granted_by_class[name]
                    if self._granted_by_class.get(name)
                    else 0.0
                ),
            }
            for name in sorted(names)
        }

    async def shutdown(self) -> None:
        """Stop background expiry work."""
        if self._sweeper is not None:
            self._sweeper.cancel()
            await asyncio.gather(self._sweeper, return_exceptions=True)

    def _drain_waiters(self) -> None:
        """Admit queued requests, class by class, each in FIFO order.

        A request is admitted when it fits the free envelope and either stays inside
        its class guarantee or, when it would borrow, no other class has a request
        waiting. The class order is fixed only for determinism: it cannot decide who
        makes progress, because the borrow rule already stops a class from taking the
        elastic pool while another class waits. One pass is enough, since a grant only
        ever consumes capacity.
        """
        for resource_class in sorted(self._waiting_by_class):
            while self._waiting_by_class.get(resource_class, 0) > 0:
                waiter = self._head_waiter(resource_class)
                if waiter is None:
                    break
                if not waiter.resources.fits(self.available_capacity):
                    break
                if not self._fits_guarantee(resource_class, waiter.resources):
                    if self._other_classes_waiting(resource_class):
                        break
                    if not self._fits_ceiling(resource_class, waiter.resources):
                        break
                self._grant(waiter)

    def _grant(self, waiter: _Waiter) -> None:
        """Admit one waiter and charge its class."""
        now = time.monotonic()
        self._waiters.pop(waiter.lease_id)
        self._dequeue(waiter)
        self.available_capacity -= waiter.resources
        self._allocations[waiter.lease_id] = _Allocation(
            owner_id=waiter.owner_id,
            resource_class=waiter.resource_class,
            resources=waiter.resources,
            allocated_at=now,
        )
        self._add_usage(waiter.resource_class, waiter.resources)
        self._granted += 1
        self._granted_by_class[waiter.resource_class] = self._granted_by_class.get(waiter.resource_class, 0) + 1
        wait_s = now - waiter.queued_at
        self._total_wait_s += wait_s
        self._wait_s_by_class[waiter.resource_class] = self._wait_s_by_class.get(waiter.resource_class, 0.0) + wait_s
        waiter.future.set_result(None)

    def _head_waiter(self, resource_class: str) -> _Waiter | None:
        """Return the oldest queued request of one class, pruning stale entries."""
        queue = self._class_queues.get(resource_class)
        if queue is None:
            return None
        while queue and queue[0] not in self._waiters:
            queue.popleft()
        return self._waiters.get(queue[0]) if queue else None

    def _enqueue(self, waiter: _Waiter) -> None:
        self._waiters[waiter.lease_id] = waiter
        self._class_queues.setdefault(waiter.resource_class, deque()).append(waiter.lease_id)
        self._waiting_by_class[waiter.resource_class] = self._waiting_by_class.get(waiter.resource_class, 0) + 1

    def _dequeue(self, waiter: _Waiter) -> None:
        self._waiting_by_class[waiter.resource_class] = max(
            0, self._waiting_by_class.get(waiter.resource_class, 0) - 1
        )

    def _discard_waiter(self, lease_id: str) -> None:
        """Drop a queued request, releasing capacity if it was granted meanwhile."""
        waiter = self._waiters.pop(lease_id, None)
        if waiter is not None:
            self._dequeue(waiter)
        if self._release_locked(lease_id):
            self._released += 1

    def _other_classes_waiting(self, resource_class: str) -> bool:
        """Whether any other class has a request queued."""
        return any(
            count > 0 and name != resource_class for name, count in self._waiting_by_class.items()
        )

    def _fits_guarantee(self, resource_class: str, resources: ResourceQuantity) -> bool:
        """Whether this request stays inside its class's guaranteed share."""
        guaranteed = self._guaranteed.get(resource_class)
        if guaranteed is None:
            return False
        return (self._usage(resource_class) + resources).fits(guaranteed)

    def _fits_ceiling(self, resource_class: str, resources: ResourceQuantity) -> bool:
        """Whether this request stays under its class's optional total ceiling."""
        ceiling = self._ceilings.get(resource_class)
        if ceiling is None:
            return True
        return (self._usage(resource_class) + resources).fits(ceiling)

    def _usage(self, resource_class: str) -> ResourceQuantity:
        return self._used_by_class.get(resource_class, ResourceQuantity(0, 0))

    def _add_usage(self, resource_class: str, resources: ResourceQuantity) -> None:
        self._used_by_class[resource_class] = self._usage(resource_class) + resources

    def _sub_usage(self, resource_class: str, resources: ResourceQuantity) -> None:
        self._used_by_class[resource_class] = self._usage(resource_class) - resources

    def _warn_undeclared_class(self, resource_class: str) -> None:
        """Report a class that cannot have a guarantee because it was never declared.

        Only a mixed deployment is at risk: when nothing is declared, a single shared
        class is the intended setup rather than an omission.
        """
        if not self._guaranteed or resource_class in self._guaranteed:
            return
        if resource_class in self._undeclared_classes:
            return
        self._undeclared_classes.add(resource_class)
        psrl_logger.warning(
            f"Sandbox resource class {resource_class!r} has no guaranteed share, so it can only use capacity "
            f"that no declared class is waiting for. Declare it under the node capacity 'classes' setting "
            f"(declared: {sorted(self._guaranteed)!r})."
        )

    def _release_locked(self, lease_id: str) -> bool:
        allocation = self._allocations.pop(lease_id, None)
        if allocation is None:
            return False
        self.available_capacity += allocation.resources
        self._sub_usage(allocation.resource_class, allocation.resources)
        return True

    async def _sweep_expired(self) -> None:
        interval = min(self._config.heartbeat_interval_s, self._config.lease_ttl_s / 2)
        while True:
            await asyncio.sleep(interval)
            now = time.monotonic()
            async with self._lock:
                expired_owners = [owner_id for owner_id, expires_at in self._owners.items() if expires_at <= now]
                for owner_id in expired_owners:
                    self._owners.pop(owner_id)
                expired_leases = [
                    lease_id for lease_id, item in self._allocations.items() if item.owner_id in expired_owners
                ]
                for lease_id in expired_leases:
                    self._release_locked(lease_id)
                expired_waiters = [waiter for waiter in self._waiters.values() if waiter.owner_id in expired_owners]
                for waiter in expired_waiters:
                    self._waiters.pop(waiter.lease_id)
                    self._dequeue(waiter)
                    waiter.expired = True
                    waiter.future.cancel()
                self._expired += len(expired_leases)
                aged_leases = self._reclaim_aged_leases(now)
                if expired_leases or expired_waiters or aged_leases:
                    self._drain_waiters()

    def _reclaim_aged_leases(self, now: float) -> list[str]:
        """Reclaim leases that outlived every legitimate sandbox lifetime.

        A lease a caller forgot to release is otherwise unreclaimable: its owner is alive and
        renewing, so `lease_ttl_s` never expires it. Reclaiming it here converts a silent
        capacity leak, which fills the node and blocks every later sandbox, into a warning
        that names the owner and the size it was holding.
        """
        max_age = self._config.lease_max_age_s
        if max_age is None:
            return []
        aged = [
            (lease_id, item)
            for lease_id, item in self._allocations.items()
            if now - item.allocated_at > max_age
        ]
        for lease_id, item in aged:
            self._release_locked(lease_id)
            self._stale_leases_reclaimed += 1
            psrl_logger.warning(
                f"Reclaimed node capacity lease {lease_id!r} of resource_class={item.resource_class!r} after "
                f"{now - item.allocated_at:.0f}s, which is longer than any sandbox may live "
                f"(lease_max_age_s={max_age:g}). Its owner {item.owner_id!r} is still renewing, so the lease "
                f"was never released: {item.resources} had been charged for a sandbox that is gone. Check for "
                "a sandbox cleanup path that can be interrupted before it returns its capacity."
            )
        return [lease_id for lease_id, _ in aged]

