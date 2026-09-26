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


@dataclass(frozen=True)
class DockerLifecycleConfig:
    """Configuration for graceful cleanup and node-level crash recovery."""

    heartbeat_dir: str = _DEFAULT_HEARTBEAT_DIR
    lease_ttl_s: float = 120.0
    heartbeat_interval_s: float = 30.0
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
            if not self.config.gc_enabled:
                return
            if self._heartbeat_thread is None or not self._heartbeat_thread.is_alive():
                write_owner_heartbeat(self.config.heartbeat_dir, self.owner_id)
                self._stop_event.clear()
                self._heartbeat_thread = threading.Thread(
                    target=self._heartbeat_loop,
                    name="psrl-docker-lease",
                    daemon=True,
                )
                self._heartbeat_thread.start()
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
        """Refresh the worker lease independently of the asyncio event loop."""
        while not self._stop_event.wait(self.config.heartbeat_interval_s):
            try:
                write_owner_heartbeat(self.config.heartbeat_dir, self.owner_id)
                with self._lock:
                    if self._closed:
                        return
                    self._ensure_gc()
            except (OSError, RuntimeError) as exc:
                psrl_logger.warning(f"Could not refresh Docker owner lease {self.owner_id!r}: {exc}.")

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
        # Join the writer before removing the lease, or a heartbeat already in flight
        # recreates the file and leaves a live-looking lease for a dead owner.
        if heartbeat_thread is not None and heartbeat_thread is not threading.current_thread():
            heartbeat_thread.join(timeout=1.0)
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
