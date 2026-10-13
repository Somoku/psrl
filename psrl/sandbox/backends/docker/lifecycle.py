"""Docker sandbox ownership leases and crash recovery."""

from __future__ import annotations

import atexit
import logging
import os
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from psrl.sandbox.backends.docker.cli import (
    DockerContainerRuntime,
    force_remove_containers_by_label,
)
from psrl.sandbox.reclaimer import (
    NodeReclaimer,
    remove_owner_heartbeat,
    spawn_node_reclaimer,
    write_owner_heartbeat,
)

psrl_logger = logging.getLogger(__file__)

_DEFAULT_HEARTBEAT_DIR = os.path.join(tempfile.gettempdir(), "psrl-sandbox-heartbeats")

# How many heartbeats an owner may miss before the node treats it as gone and
# reclaims everything it holds. The budget exists because a heartbeat can be late
# for reasons that are not death: a starved writer thread, a filesystem hiccup, a
# host under memory pressure. Too few and a live worker loses live sandboxes — the
# failure is not "a container lingers", it is "an episode dies mid-command" — so the
# floor is generous and a config may only raise the count.
_MIN_LEASE_TTL_BEATS = 8


@dataclass(frozen=True)
class DockerLifecycleConfig:
    """Configuration for graceful cleanup and node-level crash recovery."""

    heartbeat_dir: str = _DEFAULT_HEARTBEAT_DIR
    # The tolerance is beats, not seconds: lease_ttl_s must cover
    # `_MIN_LEASE_TTL_BEATS` of them, so a worker survives a run of late beats
    # instead of being reclaimed for one. Raising `lease_ttl_s` is bounded from
    # above by the node's capacity lease — a sandbox lease that outlives the
    # capacity reservation frees memory while the container still holds it — so
    # the tolerance is widened by beating more often, not by waiting longer.
    lease_ttl_s: float = 120.0
    heartbeat_interval_s: float = 15.0
    gc_interval_s: float = 30.0
    gc_idle_exit_cycles: int = 10
    gc_enabled: bool = True
    # A stopped container is reaped despite a live owner once it stays stopped this long.
    # Must outlast the session's own stop diagnosis, which takes seconds.
    stopped_grace_s: float = 300.0
    # A containerd shim whose container record is gone is reclaimed once it is at least this
    # old. The guard is what keeps a container mid-create or mid-delete out of scope.
    orphan_task_min_age_s: float = 300.0
    docker_command: tuple[str, ...] = ("docker",)

    def __post_init__(self) -> None:
        if not self.heartbeat_dir:
            raise ValueError("Docker heartbeat_dir cannot be empty.")
        if self.heartbeat_interval_s <= 0 or self.lease_ttl_s <= 0:
            raise ValueError("Docker heartbeat_interval_s and lease_ttl_s must be greater than zero.")
        if self.heartbeat_interval_s >= self.lease_ttl_s:
            raise ValueError("Docker heartbeat_interval_s must be smaller than lease_ttl_s.")
        if self.lease_ttl_s < _MIN_LEASE_TTL_BEATS * self.heartbeat_interval_s:
            raise ValueError(
                f"Docker lease_ttl_s ({self.lease_ttl_s:g}s) must cover at least {_MIN_LEASE_TTL_BEATS} heartbeats "
                f"of {self.heartbeat_interval_s:g}s ({_MIN_LEASE_TTL_BEATS * self.heartbeat_interval_s:g}s). A "
                "shorter budget reclaims a worker's sandboxes because a heartbeat was late, not because the "
                "worker was gone."
            )
        if self.gc_interval_s < 1 or self.gc_idle_exit_cycles < 1:
            raise ValueError("Docker gc_interval_s and gc_idle_exit_cycles must be at least one.")
        if self.stopped_grace_s <= self.gc_interval_s:
            raise ValueError(
                "Docker stopped_grace_s must exceed gc_interval_s, or a single sweep would reap a stopped "
                "container before its owner can read the exit state."
            )
        if not self.docker_command:
            raise ValueError("Docker docker_command cannot be empty.")

    @classmethod
    def from_value(
        cls,
        value: DockerLifecycleConfig | Mapping[str, Any] | None,
        *,
        docker_host: str | None = None,
    ) -> DockerLifecycleConfig:
        """Normalize Hydra mappings and attach an explicit Docker endpoint."""
        if isinstance(value, cls):
            if docker_host and value.docker_command == ("docker",):
                return replace(value, docker_command=("docker", "--host", docker_host))
            return value
        normalized = dict(value or {})
        command: Sequence[str] = normalized.get("docker_command", ("docker",))
        if isinstance(command, str):
            command = (command,)
        if docker_host and "docker_command" not in normalized:
            command = ("docker", "--host", docker_host)
        normalized["docker_command"] = tuple(command)
        return cls(**normalized)


class DockerLifecycle:
    """Own one worker lease and the shared node garbage collector.

    A worker writes one heartbeat regardless of how many containers it owns.
    This avoids one event-loop task and one file per sandbox while retaining
    crash recovery after SIGKILL or OOM. The detached collector is shared by all
    workers that use the same heartbeat directory.
    """

    def __init__(self, owner_id: str, config: DockerLifecycleConfig) -> None:
        self.owner_id = owner_id
        self.config = config
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._gc_thread: threading.Thread | None = None
        self._gc_process = None
        self._next_gc_probe = 0.0
        self._startup_swept = False
        self._atexit_registered = False
        self._closed = False

    def start(self) -> None:
        """Start the owner lease and ensure a live node collector exists."""
        if not self.owner_id:
            return
        with self._lock:
            if self._closed:
                raise RuntimeError("Docker lifecycle is closed.")
            if not self._atexit_registered:
                atexit.register(self.close)
                self._atexit_registered = True
            # The lease is written whether or not this process also supervises a
            # collector: it is what tells every other worker's collector that this
            # worker is alive, and a run that only writes it while collecting would
            # look dead to the first collector that shares the directory.
            if self._heartbeat_thread is None or not self._heartbeat_thread.is_alive():
                try:
                    write_owner_heartbeat(self.config.heartbeat_dir, self.owner_id)
                except Exception:
                    # A lease that could not be written yet is not a reason to refuse
                    # the lifecycle: the loop retries on the next beat, and refusing
                    # here would leave the worker with no heartbeat at all.
                    psrl_logger.warning(
                        f"Could not write the initial Docker owner lease for {self.owner_id!r}; "
                        "the heartbeat will retry.",
                        exc_info=True,
                    )
                self._stop_event.clear()
                self._heartbeat_thread = threading.Thread(
                    target=self._heartbeat_loop,
                    name="psrl-docker-lease",
                    daemon=True,
                )
                self._heartbeat_thread.start()
            if not self.config.gc_enabled:
                return
            if self._gc_thread is None or not self._gc_thread.is_alive():
                self._gc_thread = threading.Thread(
                    target=self._gc_loop,
                    name="psrl-docker-gc",
                    daemon=True,
                )
                self._gc_thread.start()
            self._reclaim_previous_run()
            self._ensure_gc()

    def _reclaim_previous_run(self) -> None:
        """Reap a previous run's abandoned containers before this one fills the node.

        The collector would find them on its own, but not before this worker has already
        been admitted against an envelope that does not know their memory is still spoken
        for. Sweeping once here closes that over-commit window.
        """
        if self._startup_swept:
            return
        self._startup_swept = True
        try:
            outcome = self._reclaimer().sweep()
        except Exception:
            psrl_logger.warning("Could not sweep abandoned Docker sandboxes at startup.", exc_info=True)
            return
        if outcome.removed:
            psrl_logger.info(f"Reclaimed {len(outcome.removed)} abandoned Docker sandbox(es) left by an earlier run.")
        # A container this node could not destroy still holds its memory, and the envelope
        # this run is about to be admitted against does not account for it. Saying so is
        # the difference between starting short and starting short without knowing it,
        # which is the state that turned a leak into a silent hang.
        if outcome.unremovable:
            psrl_logger.error(
                f"{len(outcome.unremovable)} Docker sandbox(es) left by an earlier run could not be removed, "
                f"so their memory is still spoken for while this run is admitted against a full envelope: "
                f"{list(outcome.unremovable)}."
            )
        if outcome.remaining is None:
            psrl_logger.warning(
                "Could not count the Docker sandboxes this node still holds, so the startup sweep cannot "
                "confirm the node is clean."
            )
        elif outcome.remaining:
            psrl_logger.warning(
                f"This node still holds {outcome.remaining} Docker sandbox(es) after the startup sweep."
            )

    def _reclaimer(self) -> NodeReclaimer:
        """
        Build a reclaimer over this node's Docker runtime.
        """
        return NodeReclaimer(
            self.config.heartbeat_dir,
            self.config.lease_ttl_s,
            DockerContainerRuntime(self.config.docker_command),
            stopped_grace_s=self.config.stopped_grace_s,
            orphan_task_min_age_s=self.config.orphan_task_min_age_s,
        )

    def _ensure_gc(self) -> None:
        """
        Probe collector ownership at most once per sweep interval under the lock.
        """
        now = time.monotonic()
        if now < self._next_gc_probe:
            return
        if self._gc_process is None or self._gc_process.poll() is not None:
            gc_process = spawn_node_reclaimer(
                self.config.heartbeat_dir,
                self.config.lease_ttl_s,
                self.config.gc_interval_s,
                idle_exit_cycles=self.config.gc_idle_exit_cycles,
                stopped_grace_s=self.config.stopped_grace_s,
                orphan_task_min_age_s=self.config.orphan_task_min_age_s,
                docker_command=self.config.docker_command,
            )
            if gc_process is None:
                raise RuntimeError("Could not start Docker sandbox crash recovery.")
            self._gc_process = gc_process
        self._next_gc_probe = now + self.config.gc_interval_s

    def _heartbeat_loop(self) -> None:
        """Refresh the worker lease until this lifecycle closes.

        Liveness is the one thing this loop must not lose: the node's collector
        destroys every container whose owner lease has gone stale, so a heartbeat that
        stops while its worker is alive reclaims live sandboxes. Every failure is
        therefore absorbed and retried on the next beat, and the only way out is
        `close`. It is a thread of its own rather than a task on the worker's event
        loop, so work that blocks the loop cannot take the lease down with it.
        """
        while not self._stop_event.wait(self.config.heartbeat_interval_s):
            try:
                write_owner_heartbeat(self.config.heartbeat_dir, self.owner_id)
            except Exception:
                psrl_logger.warning(
                    f"Could not refresh Docker owner lease {self.owner_id!r}; retrying on the next heartbeat.",
                    exc_info=True,
                )
        psrl_logger.info(f"Docker owner lease heartbeat stopped for {self.owner_id!r}.")

    def _gc_loop(self) -> None:
        """Keep a node collector alive, separately from the lease heartbeat.

        Supervision runs in its own thread because spawning a collector can fail in
        ways the lease must not inherit: the two share nothing but the config, and a
        failed spawn is not a reason to stop saying this worker is alive.
        """
        while not self._stop_event.wait(self.config.gc_interval_s):
            try:
                with self._lock:
                    if self._closed:
                        return
                    self._ensure_gc()
            except Exception:
                psrl_logger.warning("Could not keep the Docker sandbox collector alive.", exc_info=True)
        psrl_logger.info("Docker sandbox collector supervision stopped.")

    def close(self) -> None:
        """Remove owned containers and stop the worker heartbeat exactly once."""
        if not self.owner_id:
            return
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stop_event.set()
            heartbeat_thread = self._heartbeat_thread
            self._heartbeat_thread = None
            gc_thread = self._gc_thread
            self._gc_thread = None
        # Join the writer before removing the lease, or a heartbeat already in flight
        # recreates the file and leaves a live-looking lease for a dead owner.
        for thread in (heartbeat_thread, gc_thread):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=1.0)
        force_remove_containers_by_label(
            "psrl.actor_id",
            self.owner_id,
            docker_command=self.config.docker_command,
        )
        try:
            remove_owner_heartbeat(self.config.heartbeat_dir, self.owner_id)
        except OSError as exc:
            psrl_logger.warning(f"Could not remove Docker owner lease {self.owner_id!r}: {exc}.")
        if self._atexit_registered:
            atexit.unregister(self.close)
            self._atexit_registered = False
