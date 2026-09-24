"""Docker Engine API sandbox backend."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import aiohttp

from psrl.sandbox.async_utils import complete_cleanup
from psrl.sandbox.backends.docker.cli import signal_container_process_group
from psrl.sandbox.backends.docker.devices import gpu_visibility_env, resolve_device_mounts
from psrl.sandbox.backends.docker.egress import (
    ISOLATED_NETWORK_MODE,
    DockerEgressEnforcer,
    EgressEnforcementError,
    EgressPlan,
    EgressRules,
    container_ipv4,
    plan_egress,
    resolve_iptables_command,
)
from psrl.sandbox.backends.docker.engine import (
    DockerEngine,
    DockerEngineClient,
    DockerEngineError,
)
from psrl.sandbox.backends.docker.events import ContainerEventWatcher
from psrl.sandbox.backends.docker.exec import (
    DockerShellProcess,
    ExecBudget,
    ExecStrategy,
    OneShotExec,
    PersistentShellExec,
    ShellProcess,
)
from psrl.sandbox.backends.docker.lifecycle import DockerLifecycle, DockerLifecycleConfig
from psrl.sandbox.backends.docker.policy import (
    PROXY_URL_ENV_KEYS,
    DockerDiskAdmissionConfig,
    DockerPolicyProfile,
    DockerSecurityConfig,
    append_no_proxy_alias,
)
from psrl.sandbox.backends.docker.pool import (
    WARM_POOL_LABEL,
    DockerWarmPool,
    WarmPoolConfig,
    pool_key,
)
from psrl.sandbox.backends.docker.session import DockerSession
from psrl.sandbox.capacity import read_cgroup_memory_mb
from psrl.sandbox.core import (
    ExecMode,
    ResumeLevel,
    SandboxBackend,
    SandboxCapabilities,
    SandboxFeature,
    SandboxProvisionError,
    SandboxSession,
    SandboxSource,
    SandboxSourceKind,
    SandboxSpec,
    SandboxStatus,
    SnapshotKind,
    SnapshotRef,
    resolve_credentials,
    rewrite_loopback_proxy,
)
from psrl.sandbox.metrics import SandboxMetrics, SandboxMetricsSnapshot
from psrl.sandbox.reclaimer import lease_store_id
from psrl.sandbox.snapshot_store import (
    LocalSnapshotBudget,
    SnapshotStore,
    SnapshotStoreConfig,
    build_snapshot_store,
)

psrl_logger = logging.getLogger(__file__)

# FILESYSTEM_SNAPSHOT + RESTORE: docker commit turns a container's writable layer into a reusable
# image, and restore creates one from it: a cheap, self-contained grader snapshot with no cold start.
#
# RESUME_ANYWHERE is filesystem-level, never full state. Docker commits the writable layer
# but not memory, so the level stops a conformance run proving a harness survives a resume.
_DOCKER_FEATURES = frozenset(
    {
        SandboxFeature.FREEZE,
        SandboxFeature.HOST_MOUNT,
        SandboxFeature.FILESYSTEM_SNAPSHOT,
        SandboxFeature.RESTORE,
        SandboxFeature.RESUME_ANYWHERE,
    }
)
# Grace for a creation already past admission to settle before shutdown cancels it, so a
# container that was just created still gets an owner that can destroy it.
_CREATE_DRAIN_TIMEOUT_S = 30.0

# The repository namespace this backend commits its filesystem snapshots into. It owns the
# name, so identifying local snapshot images by it is exact rather than a guess.
_SNAPSHOT_LOCAL_PREFIX = "psrl/snapshot/"

# How many snapshot references the local cache bookkeeping remembers.
_SNAPSHOT_USE_LIMIT = 4096


class DockerBackend(SandboxBackend):
    """
    Local Docker backend using one persistent Engine API connection pool.
    """

    def __init__(
        self,
        name: str = "docker",
        docker_host: str | None = None,
        policy_profiles: Mapping[str, DockerPolicyProfile | Mapping[str, Any]] | None = None,
        security: DockerSecurityConfig | Mapping[str, Any] | None = None,
        keepalive_command: Sequence[str] = ("tail", "-f", "/dev/null"),
        command_interpreter: Sequence[str] = ("bash", "-lc"),
        request_timeout_s: float = 180.0,
        container_watch_interval_s: float = 15.0,
        connection_limit: int = 128,
        image_pull_concurrency: int = 2,
        max_exec_output_bytes: int = 16 * 1024 * 1024,
        max_observation_chars: int = 0,
        silence_timeout_s: float | None = None,
        startup_timeout_s: float = 300.0,
        egress_firewall_command: Sequence[str] | str | None = None,
        auto_pull: bool = True,
        registry_auth: Mapping[str, str] | None = None,
        snapshot_store: SnapshotStoreConfig | Mapping[str, Any] | None = None,
        snapshot_local_cache_fraction: float = 0.3,
        warm_pool: WarmPoolConfig | Mapping[str, Any] | None = None,
        owner_id_env: str = "PSRL_ACTOR_ID",
        lifecycle: DockerLifecycleConfig | Mapping[str, Any] | None = None,
        disk_admission: DockerDiskAdmissionConfig | Mapping[str, Any] | None = None,
        cgroup_parent: str | None = None,
        isolation_runtime: str | None = None,
        default_exec_mode: ExecMode = ExecMode.PERSISTENT,
        shell_factory: Callable[[str], ShellProcess] | None = None,
        engine: DockerEngine | None = None,
    ) -> None:
        self._name = name
        self.docker_host = docker_host
        self.policy_profiles = {
            key: DockerPolicyProfile.from_value(value) for key, value in (policy_profiles or {}).items()
        }
        self.security = (
            security if isinstance(security, DockerSecurityConfig) else DockerSecurityConfig(**dict(security or {}))
        )
        self.keepalive_command = tuple(keepalive_command)
        self.command_interpreter = tuple(command_interpreter)
        # Retained so a session can describe its own truncation without reaching into the
        # engine client, which test doubles and other engines do not have to expose.
        self.max_exec_output_bytes = max_exec_output_bytes
        self.max_observation_chars = max_observation_chars
        if silence_timeout_s is not None and silence_timeout_s <= 0:
            raise ValueError("Docker silence_timeout_s must be greater than zero when set.")
        self.silence_timeout_s = silence_timeout_s
        if startup_timeout_s <= 0:
            raise ValueError("Docker startup_timeout_s must be greater than zero.")
        self.startup_timeout_s = startup_timeout_s
        self.egress = DockerEgressEnforcer(resolve_iptables_command(egress_firewall_command))
        self._egress_rules: dict[str, EgressRules] = {}
        self._pool_fills: dict[str, asyncio.Task[None]] = {}
        self.default_exec_mode = default_exec_mode
        self._shell_factory = shell_factory or self._spawn_shell
        if container_watch_interval_s <= 0:
            raise ValueError("Docker container_watch_interval_s must be greater than zero.")
        self.container_watch_interval_s = container_watch_interval_s
        if not self.keepalive_command or not self.command_interpreter:
            raise ValueError("Docker keepalive_command and command_interpreter cannot be empty.")
        self.owner_id = os.getenv(owner_id_env, "")
        self.lifecycle = DockerLifecycle(
            self.owner_id,
            DockerLifecycleConfig.from_value(lifecycle, docker_host=docker_host),
        )
        self.disk_admission = DockerDiskAdmissionConfig.from_value(disk_admission)
        self.cgroup_parent = cgroup_parent
        self.isolation_runtime = isolation_runtime
        self._verified_runtimes: set[str] = set()
        self._runtime_lock = asyncio.Lock()
        self.auto_pull = auto_pull
        self.registry_auth = dict(registry_auth or {})
        self.engine = engine or DockerEngineClient(
            docker_host,
            request_timeout_s=request_timeout_s,
            connection_limit=connection_limit,
            max_exec_output_bytes=max_exec_output_bytes,
        )
        if not 0 < snapshot_local_cache_fraction <= 1:
            raise ValueError("Docker snapshot_local_cache_fraction must be in (0, 1].")
        self.snapshot_local_cache_fraction = snapshot_local_cache_fraction
        self._snapshot_last_used: dict[str, float] = {}
        # Last observed size of the local snapshot cache. Reported from the metric path
        # without I/O, because a metric that reads the daemon can stall a training step.
        self._snapshot_cache_mb = 0.0
        # One run's prefetch plan, accumulated so coverage can be reported against the
        # whole working set rather than against a single call.
        self._prefetch_requested = 0
        self._prefetch_warmed = 0
        self.warm_pool = DockerWarmPool(
            WarmPoolConfig.from_value(warm_pool),
            prepare=self._prepare_pool_entry,
            destroy=self._destroy_pool_entry,
        )
        self.snapshot_store: SnapshotStore | None = build_snapshot_store(
            snapshot_store,
            self.engine,
            registry_auth=self.registry_auth,
        )
        self.metrics = SandboxMetrics()
        self.container_events = ContainerEventWatcher(self.engine, self.engine.inspect_container)
        self._rootless_checked = False
        self._rootless_lock = asyncio.Lock()
        if image_pull_concurrency < 1:
            raise ValueError("Docker image_pull_concurrency must be positive.")
        self._image_pull_slots = asyncio.Semaphore(image_pull_concurrency)
        self._image_tasks: dict[str, asyncio.Task[None]] = {}
        self._closed = False
        self._close_event: asyncio.Event | None = None
        self._sessions: dict[str, DockerSession] = {}
        self._creates: set[asyncio.Task] = set()

    async def prepare(self, spec: SandboxSpec) -> None:
        """Warm an image, and a pooled container when the deployment keeps a pool.

        Preparation allocates no capacity lease and owns no session, so it stays off
        the caller's admission path. A pool entry only removes the cold start, and a
        spec that cannot be pooled still runs on the ordinary create path.
        """
        if self._closed:
            raise RuntimeError("Docker backend is closed.")
        if spec.source.kind != SandboxSourceKind.IMAGE:
            raise RuntimeError("DockerBackend requires an image source.")
        reference = spec.source.reference
        task = self._image_tasks.get(reference)
        if task is None:
            task = asyncio.create_task(self._prepare_image(reference))
            self._image_tasks[reference] = task
            task.add_done_callback(lambda done: self._finish_image_prepare(reference, done))
        await asyncio.shield(task)
        self.refill_warm_pool(spec)

    def refill_warm_pool(self, spec: SandboxSpec) -> None:
        """Start filling the pool for a spec shape without waiting for it.

        Preparation is off the critical path by contract, and a container start is not
        something an acquire should wait for. One fill per spec shape runs at a time, so
        a burst of acquires cannot launch a burst of container starts.
        """
        if self._closed or not self.warm_pool.poolable(spec):
            return
        key = pool_key(spec)
        existing = self._pool_fills.get(key)
        if existing is not None and not existing.done():
            return
        task = asyncio.create_task(self._fill_warm_pool(spec, key))
        self._pool_fills[key] = task
        task.add_done_callback(lambda done: self._pool_fills.pop(key, None))

    async def _fill_warm_pool(self, spec: SandboxSpec, key: str) -> None:
        """
        Fill one spec shape's pool, reporting a failure instead of raising it.
        """
        try:
            await self.warm_pool.fill(spec, want=self.warm_pool.depth_for())
        except asyncio.CancelledError:
            raise
        except Exception:
            psrl_logger.warning(f"Docker warm pool fill for key {key!r} failed.", exc_info=True)

    async def _prepare_pool_entry(self, key: str, spec: SandboxSpec) -> str:
        """Start one container ahead of an episode and label it as pooled.

        The entry is deliberately not tracked as a session: nobody owns it until a
        claim happens, and a session that outlived its claim would outlive the
        episode that adopted it.
        """
        policy = self._resolve_policy(spec)
        egress = plan_egress(spec.egress, default_network_mode=policy.network_mode)
        config = self._build_container_config(spec, policy, egress)
        config["Labels"][WARM_POOL_LABEL] = key
        name = f"psrl-warm-{key}-{uuid.uuid4().hex[:8]}"
        container_id = await self._create_or_recover(name, config, spec)
        self.metrics.count("warm_pool_prepare")
        return container_id

    async def _destroy_pool_entry(self, container_id: str) -> None:
        """
        Remove an unclaimed pool entry.
        """
        await complete_cleanup(self.release_egress(container_id))
        await self.engine.remove_container(container_id)

    def note_snapshot_use(self, reference: str) -> None:
        """Record that a local snapshot image was just used.

        This is the signal the cache budget ranks by. A local snapshot that a restore
        keeps reading is the one worth keeping, and its creation time says nothing
        about that.
        """
        self._snapshot_last_used.pop(reference, None)
        self._snapshot_last_used[reference] = time.monotonic()
        while len(self._snapshot_last_used) > _SNAPSHOT_USE_LIMIT:
            self._snapshot_last_used.pop(next(iter(self._snapshot_last_used)), None)

    async def snapshot_cache_usage(self) -> dict[str, float]:
        """Return how much disk the node-local snapshot cache is holding.

        Base images are outside this figure on purpose: they are shared and worth
        caching, and the budget exists so a snapshot cannot evict them.
        """
        images = await self._local_snapshot_images()
        return {
            "snapshot_cache/entries": float(len(images)),
            "snapshot_cache/size_mb": float(sum(images.values())),
        }

    async def evict_local_snapshots(self, total_mb: int) -> list[str]:
        """Drop the least recently used local snapshots until the cache fits its budget.

        Args:
            total_mb (int): The node's sandbox disk allowance. The cache may hold a
                configured fraction of it, and the rest stays available to base images
                and to the sandboxes themselves.

        Returns:
            list[str]: The references removed.
        """
        images = await self._local_snapshot_images()
        if not images:
            return []
        budget = LocalSnapshotBudget(
            total_mb=total_mb,
            fraction=self.snapshot_local_cache_fraction,
            usage=dict(self._snapshot_last_used),
        )
        evictions = budget.select_evictions(images)
        removed: list[str] = []
        for reference in evictions:
            try:
                await self.engine.remove_image(reference)
            except Exception:
                psrl_logger.warning(f"Could not evict local snapshot {reference!r}.", exc_info=True)
                continue
            self._snapshot_last_used.pop(reference, None)
            removed.append(reference)
        self._snapshot_cache_mb = sum(size for name, size in images.items() if name not in set(removed))
        if removed:
            self.metrics.count("snapshot_cache_evicted")
            psrl_logger.info(f"Evicted {len(removed)} local snapshot image(s) to stay inside the cache budget.")
        return removed

    def _snapshot_prefixes(self) -> tuple[str, ...]:
        """Return the local references that count as this node's snapshot cache.

        A node-local commit lives under one prefix, and a snapshot pulled back from
        the shared store lives under the store's own registry namespace. Counting only
        the first would let every restored snapshot grow the cache without bound,
        which is exactly the disk-fill the budget exists to prevent.
        """
        prefixes = [_SNAPSHOT_LOCAL_PREFIX]
        store = self.snapshot_store
        if store is not None and store.config.registry:
            namespace = store.config.snapshot_namespace.strip("/")
            prefixes.append(f"{store.config.registry.rstrip('/')}/{namespace}/")
        return tuple(prefixes)

    async def _local_snapshot_images(self) -> dict[str, float]:
        """Return the local snapshot images and their sizes in MiB.

        A tag is preferred as the removable reference, and a digest is used when a
        pull left no tag, which is what a digest-pinned restore produces. The cache
        figure is refreshed here because this is the only place that can list images,
        and the metrics plane reads it synchronously.
        """
        try:
            images = await self.engine.list_images()
        except (DockerEngineError, aiohttp.ClientError, OSError, TimeoutError, asyncio.TimeoutError):
            psrl_logger.warning("Could not read the local image list for the snapshot cache.", exc_info=True)
            return {}
        prefixes = self._snapshot_prefixes()
        found: dict[str, float] = {}
        for image in images:
            references = [str(reference) for reference in image.get("RepoTags") or () if reference]
            if not references:
                references = [str(digest) for digest in image.get("RepoDigests") or () if digest]
            matches = [reference for reference in references if reference.startswith(prefixes)]
            if not matches:
                continue
            # One entry per image, so a snapshot with both a tag and a digest is not
            # counted twice against the budget.
            found[matches[0]] = float(image.get("Size", 0) or 0) / (1024 * 1024)
        self._snapshot_cache_mb = sum(found.values())
        return found

    def warm_pool_snapshot(self) -> dict[str, float]:
        """
        Return the pool's counters for the metrics sink.
        """
        return self.warm_pool.snapshot().as_dict()

    def usage_snapshot(self) -> dict[str, float]:
        """Return what the sandboxes actually use, not what admission charged.

        The envelope is a reservation and this is the observed footprint. Reporting
        both is what lets an operator see whether an overcommit setting is honest
        rather than merely accepted.
        """
        metrics = self.metrics.snapshot()
        usage = {
            "usage/sessions": float(metrics.active_sessions),
            "usage/peak_sessions": float(metrics.peak_active_sessions),
            "usage/sandbox_memory_mb": metrics.current_memory_bytes / (1024 * 1024),
            "usage/sandbox_peak_memory_mb": metrics.peak_memory_bytes / (1024 * 1024),
        }
        # Only a configured sandbox slice measures the sandboxes. With no parent the read
        # would report this worker's own cgroup, so it is left out rather than reported wrong.
        if self.cgroup_parent is not None:
            observed = read_cgroup_memory_mb(self.cgroup_parent)
            if observed is not None:
                usage["usage/cgroup_memory_mb"] = observed
        usage["snapshot_cache/size_mb"] = self._snapshot_cache_mb
        return usage

    async def prefetch_images(self, references: Sequence[str], *, concurrency: int = 2) -> int:
        """Warm a run's working set so the first create does not pay the pull.

        Bounded by the deployment's own pull concurrency as well as by `concurrency`, because
        a prefetch that saturates the registry makes the rollout it was meant to speed up
        slower. A reference that cannot be warmed is counted rather than raised: the task
        that needs it still runs, it just pays the pull.
        """
        if concurrency < 1:
            raise ValueError("Docker image prefetch concurrency must be positive.")
        slots = asyncio.Semaphore(concurrency)

        async def warm(reference: str) -> bool:
            async with slots:
                try:
                    await self._prepare_image(reference)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    psrl_logger.warning(
                        f"Docker prefetch of {reference!r} failed. Its task will pay the pull.",
                        exc_info=True,
                    )
                    return False
            return True

        results = await asyncio.gather(*(warm(reference) for reference in references))
        warmed = sum(1 for warmed_ok in results if warmed_ok)
        self.note_prefetch(requested=len(references), warmed=warmed)
        return warmed

    def image_snapshot(self) -> dict[str, float]:
        """Return the image plane: pull cost and warm-start effectiveness.

        Pull cost is what decides whether the prefetch step earns its disk, and the
        warm hit ratio says whether a prepared container is actually adopted. Both are
        read by the trainer's per-step hook and never logged here.
        """
        operations = self.metrics.snapshot().operations
        pull = operations.get("pull_image")
        creates = operations.get("create")
        claims = self.warm_pool.snapshot().claimed
        attempts = claims + (creates.count if creates else 0)
        return {
            "image/pull_s_p50": pull.p50_seconds if pull else 0.0,
            "image/pull_s_p95": pull.p95_seconds if pull else 0.0,
            "image/warm_pool_hit_ratio": (claims / attempts) if attempts else 0.0,
            "image/prefetch_coverage": (
                self._prefetch_warmed / self._prefetch_requested if self._prefetch_requested else 0.0
            ),
        }

    def note_prefetch(self, *, requested: int, warmed: int) -> None:
        """Record how much of one run's working set the prefetch step materialized.

        Cumulative over the run, because coverage is only meaningful against the whole
        working set a run asked for, not against one call.
        """
        self._prefetch_requested += max(0, requested)
        self._prefetch_warmed += max(0, warmed)

    def _finish_image_prepare(self, reference: str, task: asyncio.Task[None]) -> None:
        """
        Forget completed downloads and observe failures after waiter cancellation.
        """
        self._image_tasks.pop(reference, None)
        if not task.cancelled():
            task.exception()

    async def _prepare_image(self, reference: str) -> None:
        """
        Recheck the daemon cache after admission before downloading an image.
        """
        async with self._image_pull_slots:
            if await self.engine.image_exists(reference):
                return
            if not self.auto_pull:
                raise RuntimeError(f"Docker image {reference!r} is unavailable and auto_pull is disabled.")
            await self._await_disk_headroom()
            with self.metrics.measure("pull_image"):
                await self.engine.pull_image(reference, self.registry_auth or None)

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> SandboxCapabilities:
        """
        Return the declared feature set, including egress only where it can be enforced.

        A destination allowlist needs a host firewall, so a node without one must not
        advertise the feature: a spec that requires containment would otherwise be
        admitted into a sandbox that is open.
        """
        features = _DOCKER_FEATURES | {SandboxFeature.CREDENTIAL_INJECTION}
        if self.egress.available():
            features = features | {SandboxFeature.EGRESS_POLICY}
        if self.isolation_runtime is not None:
            features = features | {SandboxFeature.ISOLATION_RUNTIME}
        if self.warm_pool.config.enabled:
            # A recipe can ask for a warm start on this backend or on a provider, so
            # the same intent has to be expressible on both.
            features = features | {SandboxFeature.WARM_POOL}
        return SandboxCapabilities(features, resume_level=ResumeLevel.FILESYSTEM)

    @property
    def uses_node_capacity(self) -> bool:
        """
        Return that Docker sessions share the worker node's resources.
        """
        return True

    @property
    def is_open(self) -> bool:
        """
        Return whether this backend still accepts work and owns its sessions.
        """
        return not self._closed

    def forget_session(self, container_id: str) -> None:
        """
        Drop a destroyed session from the backend's ownership registry.
        """
        self._sessions.pop(container_id, None)
        self.container_events.forget(container_id)

    async def wait_for_stop(self, container_id: str) -> str | None:
        """
        Wait for a container stop, or report that the daemon does not serve events.
        """
        return await self.container_events.wait_for_stop(container_id)

    def metrics_snapshot(self) -> SandboxMetricsSnapshot:
        """
        Return Docker lifecycle, latency, and memory metrics.
        """
        return self.metrics.snapshot()

    async def _await_disk_headroom(self, required_mb: int = 0) -> None:
        """Block until the Docker data path has room for this request, then give up.

        The threshold covers the configured headroom plus what this sandbox will
        write, so a disk request is enforced rather than only accounted. The
        envelope caps how many sandboxes hold a disk reservation at once, and this
        is what stops the last one from being admitted against a full volume.
        """
        policy = self.disk_admission
        if not policy.path or (policy.min_free_mb <= 0 and required_mb <= 0):
            return
        needed_mb = policy.min_free_mb + max(0, required_mb)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + policy.wait_timeout_s
        while True:
            try:
                usage = await asyncio.to_thread(shutil.disk_usage, policy.path)
                free_mb = usage.free // (1024 * 1024)
            except OSError as exc:
                raise RuntimeError(f"Could not inspect Docker data path {policy.path!r}.") from exc
            if free_mb >= needed_mb:
                return
            if loop.time() >= deadline:
                with self.metrics.measure("disk_admission_failure"):
                    raise RuntimeError(
                        f"Docker data path {policy.path!r} has {free_mb} MiB free, below the "
                        f"{needed_mb} MiB admission threshold, after waiting "
                        f"{policy.wait_timeout_s:.0f}s."
                    )
            with self.metrics.measure("disk_admission_wait"):
                # Wake on shutdown as well as on the poll interval, or a closing worker waits
                # out the full admission timeout for a sandbox it is no longer going to use.
                await self._wait_or_close(min(policy.poll_interval_s, max(0.0, deadline - loop.time())))
            if self._closed:
                raise RuntimeError("Docker backend is closed.")

    async def _wait_or_close(self, timeout_s: float) -> None:
        """
        Sleep for an interval unless the backend closes first.
        """
        if self._close_event is None:
            self._close_event = asyncio.Event()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._close_event.wait(), timeout=timeout_s)

    async def _check_rootless(self) -> None:
        if not self.security.require_rootless or self._rootless_checked:
            return
        async with self._rootless_lock:
            if self._rootless_checked:
                return
            info = await self.engine.info()
            security_options = [str(option).lower() for option in info.get("SecurityOptions", [])]
            if not any("rootless" in option for option in security_options):
                raise RuntimeError(
                    "Docker backend requires a rootless daemon, but Docker did not report rootless mode."
                )
            self._rootless_checked = True

    def _container_name(self, spec: SandboxSpec) -> str:
        if spec.idempotency_key:
            # Scoped to this worker: two workers sampling the same task share an idempotency
            # key, and sharing one container would put two trajectories in one sandbox.
            digest = hashlib.sha256(f"{self.owner_id}\0{self.name}\0{spec.idempotency_key}".encode()).hexdigest()[:20]
            return f"psrl-sandbox-{digest}"
        return f"psrl-sandbox-{uuid.uuid4().hex[:20]}"

    def _resolve_policy(self, spec: SandboxSpec) -> DockerPolicyProfile:
        if spec.policy_profile is None:
            return DockerPolicyProfile()
        try:
            return self.policy_profiles[spec.policy_profile]
        except KeyError as exc:
            raise RuntimeError(f"Unknown Docker policy profile {spec.policy_profile!r}.") from exc

    def _build_container_config(
        self,
        spec: SandboxSpec,
        policy: DockerPolicyProfile,
        egress: EgressPlan | None = None,
    ) -> dict[str, Any]:
        labels = {**dict(spec.metadata), "psrl.sandbox": "true"}
        labels["psrl.lease_store"] = lease_store_id(self.lifecycle.config.heartbeat_dir)
        if self.owner_id:
            labels["psrl.actor_id"] = self.owner_id
        if spec.idempotency_key:
            labels["psrl.idempotency_key"] = spec.idempotency_key

        mounts = []
        for mount in spec.mounts:
            if not Path(mount.source).is_absolute() or not Path(mount.target).is_absolute():
                raise ValueError("Docker bind-mount source and target must be absolute paths.")
            mounts.append(
                {
                    "Type": "bind",
                    "Source": mount.source,
                    "Target": mount.target,
                    "ReadOnly": mount.read_only,
                }
            )
        # The devices a spec was admitted for travel with the container, so a
        # sandbox can only open what its reservation paid for.
        mounts, device_targets = resolve_device_mounts(spec.assigned_gpus, existing=mounts)
        if device_targets:
            labels["psrl.gpu_indices"] = ",".join(str(index) for index in spec.assigned_gpus)

        cap_drop = self.security.cap_drop if policy.cap_drop is None else policy.cap_drop
        cap_add = self.security.cap_add if policy.cap_add is None else policy.cap_add
        no_new_privileges = (
            self.security.no_new_privileges if policy.no_new_privileges is None else policy.no_new_privileges
        )
        read_only = self.security.read_only_rootfs if policy.read_only_rootfs is None else policy.read_only_rootfs
        pids_limit = self.security.pids_limit if policy.pids_limit is None else policy.pids_limit
        seccomp_profile = policy.seccomp_profile or self.security.seccomp_profile
        if seccomp_profile == "unconfined":
            raise ValueError("Docker seccomp cannot be disabled for PSRL sandboxes.")
        security_options = ["no-new-privileges"] if no_new_privileges else []
        if seccomp_profile:
            security_options.append(f"seccomp={seccomp_profile}")
        tmpfs = {**dict(self.security.tmpfs), **dict(policy.tmpfs)}

        host_config: dict[str, Any] = {
            "AutoRemove": False,
            "Init": True,
            "CapDrop": list(cap_drop),
            "CapAdd": list(cap_add),
            "ReadonlyRootfs": read_only,
            "PidsLimit": pids_limit,
            "SecurityOpt": security_options,
            "Mounts": mounts,
        }
        network_mode = policy.network_mode
        if egress is not None:
            network_mode = egress.network_mode
        if network_mode:
            host_config["NetworkMode"] = network_mode
        extra_hosts = list(policy.extra_hosts)
        if policy.host_gateway_alias and not any(
            entry.partition(":")[0] == policy.host_gateway_alias for entry in extra_hosts
        ):
            extra_hosts.append(f"{policy.host_gateway_alias}:host-gateway")
        if extra_hosts:
            host_config["ExtraHosts"] = extra_hosts
        if tmpfs:
            host_config["Tmpfs"] = tmpfs
        if self.cgroup_parent:
            host_config["CgroupParent"] = self.cgroup_parent
        runtime = self._effective_runtime(policy)
        if runtime:
            host_config["Runtime"] = runtime
        if spec.resources.cpu_count is not None:
            host_config["NanoCpus"] = int(spec.resources.cpu_count * 1_000_000_000)
        if spec.resources.memory_mb is not None:
            host_config["Memory"] = spec.resources.memory_mb * 1024 * 1024
            host_config["MemorySwap"] = host_config["Memory"]
        oom_score_adj = self.security.oom_score_adj if policy.oom_score_adj is None else policy.oom_score_adj
        if oom_score_adj is not None:
            host_config["OomScoreAdj"] = oom_score_adj

        environment = dict(spec.env)
        environment.update(self._resolve_credentials(spec))
        environment.update(gpu_visibility_env(spec.assigned_gpus))
        if policy.host_gateway_alias:
            if policy.rewrite_loopback_proxies:
                for key in PROXY_URL_ENV_KEYS:
                    if key in environment:
                        environment[key] = rewrite_loopback_proxy(environment[key], policy.host_gateway_alias)
            no_proxy = append_no_proxy_alias(
                ",".join(filter(None, (environment.get("no_proxy"), environment.get("NO_PROXY")))),
                policy.host_gateway_alias,
            )
            environment["no_proxy"] = no_proxy
            environment["NO_PROXY"] = no_proxy

        config: dict[str, Any] = {
            "Image": spec.source.reference,
            "Cmd": list(self.keepalive_command),
            "Entrypoint": [],
            "Env": [f"{key}={value}" for key, value in sorted(environment.items())],
            "Labels": labels,
            "HostConfig": host_config,
        }
        if spec.workdir:
            config["WorkingDir"] = spec.workdir
        user = policy.user or self.security.user
        if user:
            config["User"] = user
        identity = {
            "container": config,
            "idle_timeout_s": spec.idle_timeout_s,
        }
        labels["psrl.spec_hash"] = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return config

    def _resolve_credentials(self, spec: SandboxSpec) -> dict[str, str]:
        """
        Resolve injected secrets from this process, never from the spec.
        """
        return resolve_credentials(spec)

    def _effective_runtime(self, policy: DockerPolicyProfile) -> str | None:
        """
        Return the isolation runtime this sandbox will use.
        """
        return policy.runtime or self.isolation_runtime

    async def _check_runtime(self, runtime: str | None) -> None:
        """Refuse a sandbox whose isolation runtime this daemon lacks.

        Falling back to the default runtime would run the workload with less
        isolation than the policy promised, and nothing downstream could tell. A
        profile can name a stronger runtime than the deployment default, so the
        check follows the effective runtime rather than the configured one.
        """
        if not runtime or runtime in self._verified_runtimes:
            return
        async with self._runtime_lock:
            if runtime in self._verified_runtimes:
                return
            info = await self.engine.info()
            available = {str(name) for name in (info.get("Runtimes") or {})}
            if runtime not in available:
                raise RuntimeError(
                    f"Docker isolation runtime {runtime!r} is not available on this daemon "
                    f"(available: {sorted(available) or 'none'}). Install it, or change the policy profile."
                )
            self._verified_runtimes.add(runtime)

    async def create(self, spec: SandboxSpec) -> SandboxSession:
        if self._closed:
            raise RuntimeError("Docker backend is closed.")
        task = asyncio.create_task(self._create_session(spec))
        self._creates.add(task)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await complete_cleanup(self._discard_create(task))
            raise
        finally:
            self._creates.discard(task)

    async def _discard_create(self, task: asyncio.Task) -> None:
        session = await task
        try:
            await session.terminate()
        except Exception as exc:
            raise SandboxProvisionError(session, exc) from exc

    def _track_session(
        self,
        container_id: str,
        spec: SandboxSpec | None = None,
        policy: DockerPolicyProfile | None = None,
    ) -> DockerSession:
        existing = self._sessions.get(container_id)
        if existing is not None:
            return existing
        session = DockerSession(
            self,
            container_id,
            spec=spec,
            policy=policy or DockerPolicyProfile(),
            lifetime_timeout_s=spec.lifetime_timeout_s if spec else None,
            exec_strategy=self._exec_strategy(container_id, spec),
        )
        self._sessions[container_id] = session
        self.metrics.session_started()
        return session

    def _spawn_shell(self, container_id: str) -> ShellProcess:
        """
        Start a login shell inside a container over the Docker CLI.
        """
        shell = self.command_interpreter[0] if self.command_interpreter else "bash"
        argv = [*self.lifecycle.config.docker_command, "exec", "-i", container_id, shell, "-l"]
        try:
            return DockerShellProcess(argv)
        except FileNotFoundError as exc:
            # A persistent shell needs a client on this node. Failing here names the
            # reason, rather than surfacing a bare missing executable later.
            raise RuntimeError(
                f"Docker persistent shell needs {argv[0]!r} on this node. Install the Docker client, or "
                "declare ExecMode.ONE_SHOT for this workload."
            ) from exc

    def _exec_strategy(self, container_id: str, spec: SandboxSpec | None) -> ExecStrategy:
        """
        Build the exec strategy a spec asked for.

        A harness that keeps shell state across turns needs the persistent shell.
        A grader that runs one command is cheaper without one, and pays no shell
        process for the life of the sandbox.
        """
        mode = (spec.exec_mode if spec is not None else None) or self.default_exec_mode
        if mode is ExecMode.ONE_SHOT:
            return OneShotExec(
                self.engine,
                container_id,
                self.command_interpreter,
                max_observation_chars=self.max_observation_chars,
            )
        return PersistentShellExec(
            lambda: self._shell_factory(container_id),
            max_observation_chars=self.max_observation_chars,
            kill_group=lambda pid: self._signal_shell_group(container_id, pid),
        )

    async def _signal_shell_group(self, container_id: str, pid: int) -> bool:
        """Stop a command the persistent shell started, keeping the container alive.

        The shell is an exec'd session leader, so its process group holds the command and
        its children. Signalling the group is what lets a deadline end a command without
        discarding the episode's filesystem.

        Args:
            container_id (str): The container the shell runs in.
            pid (int): The shell's own process id inside the container.

        Returns:
            bool: Whether the signal was delivered.
        """
        return await asyncio.to_thread(
            signal_container_process_group,
            container_id,
            pid,
            docker_command=self.lifecycle.config.docker_command,
        )

    async def _create_session(self, spec: SandboxSpec) -> SandboxSession:
        if self._closed:
            raise RuntimeError("Docker backend is closed.")
        if spec.source.kind != SandboxSourceKind.IMAGE:
            raise RuntimeError("DockerBackend requires an image source.")
        await self._await_disk_headroom(spec.resources.disk_mb or 0)
        await self._check_rootless()
        policy = self._resolve_policy(spec)
        await self._check_runtime(self._effective_runtime(policy))
        await asyncio.to_thread(self.lifecycle.start)
        egress = plan_egress(spec.egress, default_network_mode=policy.network_mode)
        name = self._container_name(spec)
        config = self._build_container_config(spec, policy, egress)
        adopted = await self.warm_pool.claim(spec)
        if adopted is not None:
            # A pooled container already paid the pull, start, and image materialization,
            # so a claim only has to remember it and confirm it still answers.
            self.metrics.count("warm_pool_claim")
            session = self._track_session(adopted, spec, policy)
            try:
                await self._enforce_egress(adopted, egress)
                await self._await_ready(session, spec)
            except BaseException as error:
                # An unusable claimed entry must not be handed out, and one that cannot be
                # destroyed must still report cleanup: a bare raise would leak the slot.
                try:
                    await session.terminate()
                except Exception as cleanup_error:
                    raise SandboxProvisionError(session, error) from cleanup_error
                raise
            return session
        with self.metrics.measure("create"):
            container_id = await self._create_or_recover(name, config, spec)
        session = self._track_session(container_id, spec, policy)
        try:
            await self._enforce_egress(container_id, egress)
            await self._await_ready(session, spec)
        except BaseException as error:
            # An unusable or uncontained sandbox must not be handed out, since the caller
            # cannot tell it from a working one. The cleanup failure rides in the error.
            try:
                await session.terminate()
            except Exception as cleanup_error:
                raise SandboxProvisionError(session, error) from cleanup_error
            raise
        return session

    async def _await_ready(self, session: DockerSession, spec: SandboxSpec) -> None:
        """
        Confirm the sandbox answers before it is handed to a caller.

        A container that is running is not the same as a container that can run a
        command, and treating started as ready moves every startup failure into the
        first turn of an episode.
        """
        strategy = session.exec_strategy
        if isinstance(strategy, PersistentShellExec):
            await strategy.start(timeout_s=self.startup_timeout_s)
            return
        probe = await strategy.run("echo ready", budget=ExecBudget(timeout_s=self.startup_timeout_s))
        if probe.exit_code != 0 or "ready" not in (probe.stdout + probe.stderr):
            raise RuntimeError(
                f"Docker sandbox {session.sandbox_id!r} did not become ready within "
                f"{self.startup_timeout_s:g}s: {(probe.stdout + probe.stderr).strip()[:200]!r}."
            )

    async def _enforce_egress(self, container_id: str, plan: EgressPlan) -> None:
        """
        Apply a sandbox's egress policy, or refuse the sandbox.
        """
        if not plan.needs_firewall and plan.network_mode != ISOLATED_NETWORK_MODE:
            return
        if not plan.needs_firewall:
            self.metrics.count("egress_isolated")
            return
        inspection = await self.engine.inspect_container(container_id)
        address = container_ipv4(inspection or {})
        if not address:
            raise EgressEnforcementError(
                f"Docker sandbox {container_id!r} has no bridge address, so an egress allowlist cannot be "
                "applied to it."
            )
        self._egress_rules[container_id] = await self.egress.install(address, plan)
        self.metrics.count("egress_allowlist")

    async def image_index(self) -> dict[str, set[str]]:
        """Return the image digests and references this node's daemon holds.

        Placement uses this to send a task to a node that already has its image. A
        digest is what makes a locality claim exact, and a reference is what a caller
        usually knows, so both are reported and the digest is trusted more.

        An empty index is reported when the daemon cannot be read, which yields a cold
        node rather than a wrong one.
        """
        index: dict[str, set[str]] = {"digests": set(), "references": set()}
        try:
            images = await self.engine.list_images()
        except (DockerEngineError, aiohttp.ClientError, OSError, TimeoutError, asyncio.TimeoutError):
            psrl_logger.warning("Could not read the Docker image index for placement.", exc_info=True)
            return index
        for image in images:
            for digest in image.get("RepoDigests") or ():
                name, _, value = str(digest).partition("@")
                if name and value:
                    index["digests"].add(f"{name}@{value}")
                    index["references"].add(name)
            for reference in image.get("RepoTags") or ():
                if reference and reference != "<none>:<none>":
                    index["references"].add(str(reference))
        return index

    async def publish_snapshot(self, local_image: str, *, run_id: str, workflow_id: str) -> str:
        """Publish a committed image and return the reference that restores it anywhere.

        A store that cannot publish fails here rather than returning the node-local
        tag: a reference that only this node can use looks like success and fails
        later, on a node that never had the image.
        """
        if self.snapshot_store is None:
            raise RuntimeError(
                "This Docker backend has no snapshot store, so a snapshot cannot be made restorable "
                "elsewhere. Configure sandbox.snapshot_store, or drop the RESUME_ANYWHERE requirement."
            )
        return await self.snapshot_store.publish(local_image, run_id=run_id, workflow_id=workflow_id)

    def collect_snapshots(self, *, now: float | None = None) -> list[str]:
        """
        Expire the store records past their retention.
        """
        if self.snapshot_store is None:
            return []
        return self.snapshot_store.collect(now=now)

    async def release_egress(self, container_id: str) -> None:
        """
        Remove one sandbox's firewall rules, if it had any.
        """
        rules = self._egress_rules.pop(container_id, None)
        if rules is None:
            return
        try:
            await self.egress.remove(rules)
        except Exception:
            psrl_logger.warning(f"Could not remove egress rules for Docker sandbox {container_id!r}.", exc_info=True)

    async def _create_or_recover(
        self,
        name: str,
        config: Mapping[str, Any],
        spec: SandboxSpec,
    ) -> str:
        """
        Create once, pull a missing image once, or join an exact retry.
        """
        try:
            return await self._create_and_start(name, config)
        except DockerEngineError as exc:
            create_error = exc
        if create_error.status == 404 and self.auto_pull:
            await self.prepare(spec)
            try:
                return await self._create_and_start(name, config)
            except DockerEngineError as exc:
                create_error = exc
        if create_error.status == 409 and spec.idempotency_key:
            return await self._recover_idempotent_conflict(name, config, spec, create_error)
        raise create_error

    async def _create_and_start(self, name: str, config: Mapping[str, Any]) -> str:
        """
        Create and start a container without leaking a failed start.
        """
        try:
            container_id = await self.engine.create_container(name, config)
        except (aiohttp.ClientError, TimeoutError, asyncio.TimeoutError) as exc:
            # A lost response can hide a successful create. Keep the unique name owned.
            raise SandboxProvisionError(self._track_session(name), exc) from exc
        try:
            await self.engine.start_container(container_id)
        except BaseException as error:
            session = self._track_session(container_id)
            try:
                await session.terminate()
            except Exception as cleanup_error:
                raise SandboxProvisionError(session, error) from cleanup_error
            raise
        return container_id

    async def _recover_idempotent_conflict(
        self,
        name: str,
        config: Mapping[str, Any],
        spec: SandboxSpec,
        conflict: DockerEngineError,
    ) -> str:
        """Adopt this worker's own earlier attempt at the same sandbox.

        The container name is scoped to this worker, so a conflict is always a retry of a
        request this process already made, never another worker's. It may still be mid-start,
        which is why a created container is waited on rather than replaced.
        """
        deadline = asyncio.get_running_loop().time() + 5.0
        while True:
            existing = await self.engine.inspect_container(name)
            if existing is None:
                return await self._create_and_start(name, config)
            labels = (existing.get("Config") or {}).get("Labels") or {}
            if labels.get("psrl.idempotency_key") != spec.idempotency_key:
                raise RuntimeError(f"Docker container name {name!r} has an unrelated owner.") from conflict
            expected_hash = config["Labels"]["psrl.spec_hash"]
            if labels.get("psrl.spec_hash") != expected_hash:
                raise RuntimeError(
                    f"Docker idempotency key {spec.idempotency_key!r} was reused with a different spec."
                ) from conflict
            container_id = str(existing["Id"])
            status = str((existing.get("State") or {}).get("Status", "unknown"))
            if status == "running":
                return container_id
            if status == "paused":
                await self.engine.unpause_container(container_id)
                return container_id
            if status in {"created", "restarting"}:
                if asyncio.get_running_loop().time() >= deadline:
                    raise RuntimeError(f"Docker sandbox {container_id!r} is still starting.")
                await asyncio.sleep(0.05)
                continue
            await self.engine.remove_container(container_id)
            return await self._create_and_start(name, config)

    async def connect(self, sandbox_id: str) -> SandboxSession:
        if self._closed:
            raise RuntimeError("Docker backend is closed.")
        session = self._track_session(sandbox_id)
        status = await session.status()
        if status == SandboxStatus.PAUSED:
            await session.resume()
        elif status != SandboxStatus.RUNNING:
            raise RuntimeError(f"Docker sandbox {sandbox_id!r} is not running (status={status.value}).")
        return session

    async def restore(self, snapshot: SnapshotRef, spec: SandboxSpec | None = None) -> SandboxSession:
        """
        Create a session from a filesystem snapshot (a committed image).
        """
        if snapshot.kind != SnapshotKind.FILESYSTEM:
            raise NotImplementedError(f"DockerBackend only restores FILESYSTEM snapshots, got {snapshot.kind.value}.")
        image = snapshot.metadata.get("psrl.docker.image") or snapshot.snapshot_id
        if not image:
            raise ValueError(f"Snapshot {snapshot.snapshot_id!r} has no docker image reference.")
        if spec is None:
            raise ValueError("DockerBackend.restore requires a SandboxSpec.")
        restored_spec = replace(spec, source=SandboxSource.image(str(image)))
        # Recorded as a use, which is what keeps a snapshot a restore keeps reading out
        # of the eviction order.
        self.note_snapshot_use(str(image))
        return await self.create(restored_spec)

    async def delete_snapshot(self, snapshot: SnapshotRef) -> None:
        """
        Remove the committed image backing a filesystem snapshot.

        A published snapshot is deleted from the store as well as locally, because
        leaving the record behind would make every later collection retry a manifest
        nobody references.
        """
        if snapshot.kind != SnapshotKind.FILESYSTEM:
            return
        if snapshot.metadata.get("psrl.snapshot.published") and self.snapshot_store is not None:
            await self.snapshot_store.delete(snapshot.snapshot_id)
        image = snapshot.metadata.get("psrl.docker.image") or snapshot.snapshot_id
        if image:
            await self.engine.remove_image(str(image))

    async def shutdown(self) -> None:
        """Stop crash recovery and close the persistent Engine connection pool.

        Image downloads and disk admission are released before in-flight creations are
        joined, because `create` shields its own work: a create parked on either of those
        would otherwise hold shutdown for the whole pull or the whole admission timeout.
        """
        self._closed = True
        if self._close_event is not None:
            self._close_event.set()
        # A fill in flight holds a container start, and shutdown is not the place to
        # wait for one.
        fills = list(self._pool_fills.values())
        for task in fills:
            task.cancel()
        if fills:
            await asyncio.gather(*fills, return_exceptions=True)
        # Pool entries belong to nobody, so nothing else would destroy them.
        await self.warm_pool.drain()
        await self.container_events.close()
        image_tasks = list(self._image_tasks.values())
        for task in image_tasks:
            task.cancel()
        await asyncio.gather(*image_tasks, return_exceptions=True)
        creates = list(self._creates)
        if creates:
            _, pending = await asyncio.wait(creates, timeout=_CREATE_DRAIN_TIMEOUT_S)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        session_results = await asyncio.gather(
            *(session.terminate() for session in list(self._sessions.values())),
            return_exceptions=True,
        )
        try:
            await asyncio.to_thread(self.lifecycle.close)
        finally:
            await self.engine.close()
        errors = [result for result in session_results if isinstance(result, BaseException)]
        if errors:
            raise RuntimeError("Docker shutdown could not confirm container cleanup.") from errors[0]
