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
    """
    Parse a Docker memory value into whole MiB.
    """
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
    """
    Return the local cgroup or machine memory limit in MiB.
    """
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
    """
    Return the local cgroup or CPU-affinity capacity in cores.
    """
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


def read_cgroup_memory_mb(name: str | None = None) -> float | None:
    """Return the memory a cgroup is actually holding, in MiB.

    Used to report observed usage beside the charged envelope. A request is a
    reservation. This is the footprint, and comparing them is what shows whether an
    overcommit setting is honest.

    Args:
        name (str | None): A cgroup slice below the root, such as a sandbox parent.
            None reads this process's own cgroup.

    Returns:
        float | None: Current usage in MiB, or None when the file is unavailable.
    """
    candidates: list[str] = []
    suffix = f"/{name.strip('/')}" if name else ""
    # cgroup v2 is one hierarchy with a single numeric file.
    candidates.append(f"/sys/fs/cgroup{suffix}/memory.current")
    # cgroup v1 puts memory under its own controller.
    candidates.append(f"/sys/fs/cgroup/memory{suffix}/memory.usage_in_bytes")
    for path in candidates:
        value = _read_int(path)
        if value is not None and value >= 0:
            return value / _BYTES_PER_MIB
    return None


@dataclass(frozen=True)
class ResourceClassShare:
    """
    Fractions of the node envelope used as admission priority and a hard ceiling.
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
    """
    Normalize a class mapping from typed values or a resolved Hydra config.
    """
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
    """
    One node resource envelope shared by local sandboxes.

    Class guarantees prioritize requests within their share. Borrowers use FIFO
    ordering with bounded bypass. Capacity remains charged until explicit release
    or owner expiry, regardless of allocation age.
    """

    memory_mb: int | None = None
    cpu_cores: float | None = None
    # Devices are declared rather than detected: a device count cannot be estimated
    # from a host, and an operator either owns them or does not.
    gpu_count: int | None = None
    disk_mb: int | None = None
    utilization: float = 0.5
    # Memory is the one dimension a sandbox can be admitted against above its
    # reservation, because a waiting agent's pages can be reclaimed, unlike CPU or a device.
    memory_overcommit_ratio: float = 1.0
    lease_ttl_s: float = 180
    heartbeat_interval_s: float = 30
    classes: dict[str, ResourceClassShare] = field(default_factory=dict)
    acquire_timeout_s: float | None = 1800.0
    # Member queue deadline as a fraction of the configured default. A capacity
    # shortage costs an episode retry, and the fraction follows the episode length.
    acquire_deadline_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.memory_mb is not None and self.memory_mb <= 0:
            raise ValueError("Sandbox capacity memory_mb must be greater than zero when configured.")
        if self.cpu_cores is not None and self.cpu_cores <= 0:
            raise ValueError("Sandbox capacity cpu_cores must be greater than zero when configured.")
        if self.gpu_count is not None and self.gpu_count < 0:
            raise ValueError("Sandbox capacity gpu_count must not be negative.")
        if self.disk_mb is not None and self.disk_mb <= 0:
            raise ValueError("Sandbox capacity disk_mb must be greater than zero when configured.")
        if not 0 < self.utilization <= 1:
            raise ValueError("Sandbox capacity utilization must be in (0, 1].")
        if self.memory_overcommit_ratio < 1:
            raise ValueError(
                "Sandbox capacity memory_overcommit_ratio must be at least 1. A value below one would "
                "under-admit the node rather than overcommit it."
            )
        if not 0 < self.acquire_deadline_fraction <= 1:
            raise ValueError("Sandbox capacity acquire_deadline_fraction must be in (0, 1].")
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


@dataclass(frozen=True)
class ResourceQuantity:
    """
    Integer resource quantity used for exact accounting.

    Devices are counted rather than measured, because a node hands out specific
    device indices and two sandboxes must never be granted the same one.
    """

    memory_mb: int
    cpu_millis: int
    gpu_count: int = 0
    disk_mb: int = 0

    @classmethod
    def from_spec(cls, resources: ResourceSpec) -> ResourceQuantity:
        """
        Create a quantity from a complete sandbox resource request.

        An envelope needs a memory and a CPU number to be sound, so a request that
        omits either is rejected rather than admitted against a guess. Device and
        disk requests are optional and default to none.
        """
        if resources.memory_mb is None or resources.cpu_count is None:
            raise ValueError("Node-capacity admission requires sandbox memory_mb and cpu_count.")
        return cls(
            resources.memory_mb,
            math.ceil(resources.cpu_count * 1000),
            resources.gpu_count or 0,
            resources.disk_mb or 0,
        )

    def fits(self, available_capacity: ResourceQuantity) -> bool:
        return (
            self.memory_mb <= available_capacity.memory_mb
            and self.cpu_millis <= available_capacity.cpu_millis
            and self.gpu_count <= available_capacity.gpu_count
            and self.disk_mb <= available_capacity.disk_mb
        )

    def __add__(self, other: ResourceQuantity) -> ResourceQuantity:
        return ResourceQuantity(
            self.memory_mb + other.memory_mb,
            self.cpu_millis + other.cpu_millis,
            self.gpu_count + other.gpu_count,
            self.disk_mb + other.disk_mb,
        )

    def __sub__(self, other: ResourceQuantity) -> ResourceQuantity:
        return ResourceQuantity(
            self.memory_mb - other.memory_mb,
            self.cpu_millis - other.cpu_millis,
            self.gpu_count - other.gpu_count,
            self.disk_mb - other.disk_mb,
        )


@dataclass(frozen=True)
class CapacityGrant:
    """
    What admission granted for one sandbox.

    The concrete device indices travel with the grant because only the node knows
    which devices are free, and a sandbox must be given exactly the devices its
    reservation was charged for.
    """

    lease_id: str
    resources: ResourceQuantity
    gpu_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class ResolvedSandboxCapacity:
    """
    Resolved node envelope and its provenance.

    `physical` is what the node actually has. `resources` is what admission may
    charge, which is the physical envelope for every dimension except memory when
    overcommit is configured. Reporting utilization against the physical envelope
    is what shows whether an overcommit setting is honest.
    """

    resources: ResourceQuantity
    physical: ResourceQuantity
    memory_source: str
    cpu_source: str


def resolve_sandbox_capacity(
    config: SandboxCapacityConfig,
    *,
    detected_memory_mb: int | None = None,
    detected_cpu_cores: float | None = None,
) -> ResolvedSandboxCapacity:
    """
    Resolve and scale the node envelope once on the target node.
    """
    memory_mb = config.memory_mb or detected_memory_mb or detect_node_memory_mb()
    cpu_cores = config.cpu_cores or detected_cpu_cores or detect_node_cpu_cores()
    if not memory_mb or not cpu_cores:
        raise RuntimeError(
            "Sandbox node capacity could not be detected; configure memory_mb and cpu_cores explicitly."
        )
    physical = ResourceQuantity(
        memory_mb=max(1, math.floor(memory_mb * config.utilization)),
        cpu_millis=max(1, math.floor(cpu_cores * 1000 * config.utilization)),
        gpu_count=config.gpu_count or 0,
        disk_mb=config.disk_mb or 0,
    )
    admissible = ResourceQuantity(
        memory_mb=max(1, math.floor(physical.memory_mb * config.memory_overcommit_ratio)),
        cpu_millis=physical.cpu_millis,
        gpu_count=physical.gpu_count,
        disk_mb=physical.disk_mb,
    )
    return ResolvedSandboxCapacity(
        resources=admissible,
        physical=physical,
        memory_source="configured" if config.memory_mb is not None else "detected",
        cpu_source="configured" if config.cpu_cores is not None else "detected",
    )


def _scaled(base: ResourceQuantity, share: float) -> ResourceQuantity:
    """Return one share of an envelope, floored to whole units.

    A device is indivisible, so a class holding any share of a node that has devices
    is guaranteed at least one. Flooring instead would hand a small class zero, and a
    guarantee of zero devices is one no device request can ever fit inside, so that
    class could only ever borrow and its share would silently not apply to devices.
    """
    return ResourceQuantity(
        memory_mb=max(1, math.floor(base.memory_mb * share)),
        cpu_millis=max(1, math.floor(base.cpu_millis * share)),
        gpu_count=base.gpu_count if share >= 1 else min(base.gpu_count, max(1, math.floor(base.gpu_count * share))),
        disk_mb=max(1, math.floor(base.disk_mb * share)) if base.disk_mb else 0,
    )


@dataclass
class _Allocation:
    owner_id: str
    resource_class: str
    resources: ResourceQuantity
    allocated_at: float
    gpu_indices: tuple[int, ...] = ()


@dataclass
class _Waiter:
    lease_id: str
    owner_id: str
    resource_class: str
    resources: ResourceQuantity
    future: asyncio.Future[CapacityGrant]
    queued_at: float
    # Set by the sweeper when the owner lease expired, so `acquire` can report a
    # capacity fault instead of the bare cancellation that `future.cancel()` raises.
    expired: bool = False
    bypasses: int = 0


class SandboxCapacityCoordinator:
    """
    Admit complete CPU and memory vectors through per-class FIFO queues.

    Guaranteed requests have priority, with a bounded bypass count to let large
    borrowers eventually drain the envelope. Active allocations are never preempted.
    """

    def __init__(self, config: SandboxCapacityConfig | dict) -> None:
        if isinstance(config, dict):
            config = SandboxCapacityConfig(**config)
        self._config = config
        self._capacity = resolve_sandbox_capacity(config)
        self.available_capacity = self._capacity.resources
        # Devices are handed out by index, so the pool is the source of truth for which
        # ones are free. Two sandboxes with one index would fight over a single device.
        self._free_gpus: list[int] = list(range(self._capacity.resources.gpu_count))
        self._guaranteed = {
            name: _scaled(self._capacity.resources, share.guaranteed_share) for name, share in config.classes.items()
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
        self._closed = False
        self._granted = 0
        self._released = 0
        self._expired = 0
        self._wait_timeouts = 0
        self._expired_waiters = 0
        # Borrower starvation is only ever reported. A deadline, not a reservation, is what
        # turns genuine starvation into a capacity fault the operator can act on.
        self._max_bypasses = 0
        self._total_wait_s = 0.0
        self._granted_by_class: dict[str, int] = {}
        self._wait_s_by_class: dict[str, float] = {}
        guaranteed_summary = ", ".join(
            f"{name}: {share.memory_mb}MiB" for name, share in sorted(self._guaranteed.items())
        )
        psrl_logger.info(
            f"Sandbox node envelope: memory_mb={self.available_capacity.memory_mb}, "
            f"cpu_cores={self.available_capacity.cpu_millis / 1000:g}, "
            f"gpus={self.available_capacity.gpu_count}, disk_mb={self.available_capacity.disk_mb}, "
            f"sources={self._capacity.memory_source!r}/{self._capacity.cpu_source!r}, "
            f"class_guarantees={{{guaranteed_summary}}}, "
            f"acquire_timeout_s={self.member_queue_deadline_s!r}."
        )

    @property
    def member_queue_deadline_s(self) -> float | None:
        """
        Return how long one request waits before it is a capacity fault.

        Derived from the configured deadline rather than set separately, so an
        operator states how long a phase may take and the queue inherits a
        fraction of it. A member that cannot be admitted costs a retry instead of
        holding the whole phase.
        """
        if self._config.acquire_timeout_s is None:
            return None
        return self._config.acquire_timeout_s * self._config.acquire_deadline_fraction

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
        gpu_count: int = 0,
        disk_mb: int = 0,
    ) -> CapacityGrant:
        """Wait until this request's class can fit its complete resource vector.

        Args:
            lease_id (str): Unique capacity lease id for the new sandbox.
            owner_id (str): Worker that owns the lease and refreshes it.
            memory_mb (int): Requested container memory in MiB.
            cpu_count (float): Requested container CPUs.
            resource_class (str): Role of the sandbox, used to pick its guarantee.
            timeout_s (float | None): Queue deadline. ``None`` derives one from the
                configured phase deadline.
            gpu_count (int): Requested accelerator devices.
            disk_mb (int): Requested sandbox disk in MiB.

        Returns:
            CapacityGrant: The admitted vector and the device indices it was charged for.

        Raises:
            ValueError: When the request exceeds the envelope or the lease id exists.
            SandboxCapacityTimeout: When the deadline expires, or when the owner
                lease expired while the request was still queued.
            asyncio.CancelledError: When the caller cancelled the wait.
        """
        if self._closed:
            raise RuntimeError("Sandbox capacity coordinator is closed.")
        self._ensure_sweeper()
        resource_class = resource_class.strip()
        if not resource_class:
            raise ValueError("Sandbox capacity resource_class cannot be empty.")
        # Built directly rather than through a spec: this is the caller's already
        # resolved request, and a zero device or disk count means "none" here.
        resources = ResourceQuantity(
            memory_mb=memory_mb,
            cpu_millis=math.ceil(cpu_count * 1000),
            gpu_count=gpu_count,
            disk_mb=disk_mb,
        )
        if not resources.fits(self._capacity.resources):
            raise ValueError(
                f"Sandbox request memory_mb={memory_mb}, cpu_count={cpu_count:g} exceeds node envelope "
                f"memory_mb={self._capacity.resources.memory_mb}, "
                f"cpu_cores={self._capacity.resources.cpu_millis / 1000:g}."
            )
        ceiling = self._ceilings.get(resource_class)
        if ceiling is not None and not resources.fits(ceiling):
            raise ValueError(f"Sandbox request exceeds the ceiling for class {resource_class!r}.")
        deadline = self.member_queue_deadline_s if timeout_s is None else timeout_s
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
                return await waiter.future
            return await asyncio.wait_for(waiter.future, timeout=deadline)
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
        """
        Release one admitted resource vector.
        """
        async with self._lock:
            if self._release_locked(lease_id):
                self._released += 1
                self._drain_waiters()

    async def cancel(self, lease_id: str) -> None:
        """
        Remove a canceled request whether it is queued or already admitted.
        """
        async with self._lock:
            waiter = self._waiters.pop(lease_id, None)
            if waiter is not None:
                self._dequeue(waiter)
                waiter.future.cancel()
            if self._release_locked(lease_id):
                self._released += 1
            self._drain_waiters()

    async def renew_owner(self, owner_id: str) -> None:
        """
        Renew every active lease belonging to one worker.
        """
        async with self._lock:
            if owner_id in self._owners:
                self._owners[owner_id] = time.monotonic() + self._config.lease_ttl_s

    async def release_owner(self, owner_id: str) -> None:
        """
        Release all capacity held by a worker that is shutting down.
        """
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
        """
        Return a compact accounting and queue snapshot.
        """
        async with self._lock:
            return {
                "capacity": asdict(self._capacity.resources),
                "physical_capacity": asdict(self._capacity.physical),
                "available_capacity": asdict(self.available_capacity),
                "free_gpus": len(self._free_gpus),
                "memory_overcommit_ratio": self._config.memory_overcommit_ratio,
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
                "max_borrow_bypasses": self._max_bypasses,
                "oldest_lease_age_s": self._oldest_lease_age_s(),
                "per_class": self._class_snapshot(),
            }

    def _oldest_lease_age_s(self) -> float:
        """
        Return the age of the longest-held lease, which exposes a leak as it grows.
        """
        if not self._allocations:
            return 0.0
        now = time.monotonic()
        return max(now - item.allocated_at for item in self._allocations.values())

    def _class_snapshot(self) -> dict[str, dict]:
        """
        Return guarantee, usage, and queue depth for every class seen so far.
        """
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
        """
        Close admission and wake every queued caller before stopping expiry.
        """
        async with self._lock:
            self._closed = True
            for waiter in self._waiters.values():
                if not waiter.future.done():
                    waiter.future.set_exception(RuntimeError("Sandbox capacity coordinator is closed."))
            self._waiters.clear()
            self._class_queues.clear()
            self._waiting_by_class.clear()
        if self._sweeper is not None:
            self._sweeper.cancel()
            await asyncio.gather(self._sweeper, return_exceptions=True)

    def _drain_waiters(self) -> None:
        """Admit guarantees first, then let borrowers share the slack in arrival order.

        A request inside its class guarantee is the contract, so it is admitted whenever the
        envelope has room and is never blocked by a borrower. When a guaranteed head does not
        fit, the remaining slack is reserved for it rather than lent out. Borrowers otherwise
        use best-effort FIFO: an oldest borrower that cannot fit does not block a younger one
        that can, because head-of-line blocking would idle the node for the length of the
        longest running episode.
        """
        while True:
            heads = [self._head_waiter(name) for name in self._class_queues]
            eligible = [
                head for head in heads if head is not None and self._fits_ceiling(head.resource_class, head.resources)
            ]
            if not eligible:
                return
            guaranteed = [head for head in eligible if self._fits_guarantee(head.resource_class, head.resources)]
            fitting = [head for head in (guaranteed or eligible) if head.resources.fits(self.available_capacity)]
            if not fitting:
                # Hold the slack for a queued guarantee, or wait for a borrower to fit.
                return
            waiter = min(fitting, key=lambda item: item.queued_at)
            self._count_bypasses(eligible, waiter)
            self._grant(waiter)

    def _count_bypasses(self, eligible: list[_Waiter], waiter: _Waiter) -> None:
        """Record every request that a younger one was admitted ahead of.

        Skipping is deliberate, so this is reported rather than acted on. A climbing count
        means the class shares no longer match the workload, which `acquire_timeout_s`
        eventually turns into a capacity fault.
        """
        for head in eligible:
            if head.queued_at < waiter.queued_at:
                head.bypasses += 1
                self._max_bypasses = max(self._max_bypasses, head.bypasses)

    def _grant(self, waiter: _Waiter) -> None:
        """
        Admit one waiter, charge its class, and hand out its devices.
        """
        now = time.monotonic()
        self._waiters.pop(waiter.lease_id)
        self._dequeue(waiter)
        gpu_indices = self._take_gpus(waiter.resources.gpu_count)
        self.available_capacity -= waiter.resources
        self._allocations[waiter.lease_id] = _Allocation(
            owner_id=waiter.owner_id,
            resource_class=waiter.resource_class,
            resources=waiter.resources,
            allocated_at=now,
            gpu_indices=gpu_indices,
        )
        self._add_usage(waiter.resource_class, waiter.resources)
        self._granted += 1
        self._granted_by_class[waiter.resource_class] = self._granted_by_class.get(waiter.resource_class, 0) + 1
        wait_s = now - waiter.queued_at
        self._total_wait_s += wait_s
        self._wait_s_by_class[waiter.resource_class] = self._wait_s_by_class.get(waiter.resource_class, 0.0) + wait_s
        if not waiter.future.done():
            waiter.future.set_result(CapacityGrant(waiter.lease_id, waiter.resources, gpu_indices))

    def _take_gpus(self, count: int) -> tuple[int, ...]:
        """
        Reserve the lowest free device indices for one sandbox.
        """
        if count <= 0:
            return ()
        if count > len(self._free_gpus):
            # The admission check already proved the request fits, so reaching here
            # means the pool and the envelope disagree. Fail rather than grant twice.
            raise RuntimeError(
                f"Sandbox capacity has {len(self._free_gpus)} free device(s) but the envelope admitted "
                f"a request for {count}."
            )
        taken = tuple(self._free_gpus[:count])
        del self._free_gpus[:count]
        return taken

    def _return_gpus(self, indices: tuple[int, ...]) -> None:
        """
        Return device indices to the pool.
        """
        for index in indices:
            if index not in self._free_gpus:
                self._free_gpus.append(index)
        self._free_gpus.sort()

    def _head_waiter(self, resource_class: str) -> _Waiter | None:
        """
        Return the oldest queued request of one class, pruning stale entries.
        """
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
        """
        Drop a queued request, releasing capacity if it was granted meanwhile.
        """
        waiter = self._waiters.pop(lease_id, None)
        if waiter is not None:
            self._dequeue(waiter)
        if self._release_locked(lease_id):
            self._released += 1

    def _fits_guarantee(self, resource_class: str, resources: ResourceQuantity) -> bool:
        """
        Whether this request stays inside its class's guaranteed share.
        """
        guaranteed = self._guaranteed.get(resource_class)
        if guaranteed is None:
            return False
        return (self._usage(resource_class) + resources).fits(guaranteed)

    def _fits_ceiling(self, resource_class: str, resources: ResourceQuantity) -> bool:
        """
        Whether this request stays under its class's optional total ceiling.
        """
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
        self._return_gpus(allocation.gpu_indices)
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
                if expired_leases or expired_waiters:
                    self._drain_waiters()
