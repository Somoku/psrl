"""Node-level sandbox reclamation and owner lease heartbeats.

The reclaimer is a peer of the sandbox stack rather than a layer inside it. It
owns one question: which containers on this node has nobody come back for. It
reaches the runtime through `SandboxContainerRuntime`, so a sweep is testable
without a container daemon, and it has a real process entry point so a node
collector is a program rather than an inline source string.

A worker writes one heartbeat regardless of how many sandboxes it owns, and the
container carries the owner id in a label. A sandbox whose owner heartbeat has
aged out belongs to nobody, and that is the only lease this module interprets.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import logging
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import BinaryIO, Protocol

psrl_logger = logging.getLogger(__file__)

# Container states that can never serve another command again.
STOPPED_CONTAINER_STATES = frozenset({"exited", "dead"})

# One sweep sleeps at most this long, so the loop stays responsive to the idle
# exit and the interval remains a real upper bound between sweeps.
_MAX_SWEEP_INTERVAL_S = 60.0


def lease_store_id(heartbeat_dir: str) -> str:
    """
    Identify the lease namespace shared by workers and their collector.

    The digest is what makes two deployments on one node independent: a worker
    only ever reclaims containers that advertise its own store.
    """
    return hashlib.sha256(os.path.abspath(heartbeat_dir).encode()).hexdigest()[:16]


def owner_heartbeat_path(heartbeat_dir: str, owner_id: str) -> str:
    """
    Return the heartbeat file path for one sandbox owner.
    """
    owner_key = hashlib.sha256(owner_id.encode()).hexdigest()
    return os.path.join(heartbeat_dir, f"owner-{owner_key}.hb")


def write_owner_heartbeat(heartbeat_dir: str, owner_id: str) -> None:
    """
    Create or refresh one sandbox owner's heartbeat file.
    """
    os.makedirs(heartbeat_dir, exist_ok=True)
    path = owner_heartbeat_path(heartbeat_dir, owner_id)
    with open(path, "a"):
        pass
    os.utime(path, None)


def remove_owner_heartbeat(heartbeat_dir: str, owner_id: str) -> None:
    """
    Delete an owner's heartbeat file when it exists.
    """
    try:
        os.unlink(owner_heartbeat_path(heartbeat_dir, owner_id))
    except FileNotFoundError:
        pass


def owner_heartbeat_age_s(heartbeat_dir: str, owner_id: str, *, now: float | None = None) -> float | None:
    """
    Return an owner heartbeat's age, or `None` when it is missing.
    """
    try:
        modified_at = os.stat(owner_heartbeat_path(heartbeat_dir, owner_id)).st_mtime
    except FileNotFoundError:
        return None
    current_time = time.time() if now is None else now
    return max(0.0, current_time - modified_at)


@dataclass(frozen=True)
class OwnedContainer:
    """
    One sandbox container as the runtime sees it.
    """

    container_id: str
    owner_id: str
    state: str


class SandboxContainerRuntime(Protocol):
    """
    The container operations a reclaimer needs, and nothing more.
    """

    def list_owned(self, lease_store: str) -> Sequence[OwnedContainer] | None:
        """
        List containers in one lease store, or return `None` when the runtime is unreachable.
        """

    def remove_containers(self, container_ids: Sequence[str]) -> Sequence[str]:
        """
        Force-remove containers and return the ids the runtime still lists.
        """


@dataclass(frozen=True)
class ReclaimOutcome:
    """
    What one sweep removed and what the node still holds.
    """

    removed: tuple[str, ...] = ()
    # Containers the reclaimer asked the runtime to destroy and could not. A node that
    # still holds one of these still holds its memory, so it must not take new work.
    unremovable: tuple[str, ...] = ()
    # None means the runtime could not be queried. Collectors treat that as an unhealthy
    # sweep, not an idle node, so a transient daemon failure cannot silently exit recovery.
    remaining: int | None = None


@dataclass
class _StoppedTracker:
    """Remember when a container was first seen stopped.

    The owning session inspects a stopped container to classify an OOM kill
    before it deletes it, so reaping one immediately would race that diagnosis
    away. Two observations separated by the grace period prove nobody is coming
    back for it.
    """

    seen_at: dict[str, float] = field(default_factory=dict)

    def select(self, containers: Sequence[OwnedContainer], grace_s: float, now: float) -> set[str]:
        stopped = {item.container_id for item in containers if item.state in STOPPED_CONTAINER_STATES}
        for container_id in stopped:
            self.seen_at.setdefault(container_id, now)
        for container_id in set(self.seen_at) - stopped:
            self.seen_at.pop(container_id, None)
        return {container_id for container_id in stopped if now - self.seen_at[container_id] > grace_s}

    def forget(self, container_ids: Sequence[str]) -> None:
        for container_id in container_ids:
            self.seen_at.pop(container_id, None)


class NodeReclaimer:
    """
    Reclaim sandboxes whose owner lease expired, and stopped ones nobody diagnoses.

    The reclaimer holds no durable state. A restart rebuilds its view from the
    runtime and the heartbeat directory, which is why a collector can exit on
    node idleness and a new one can start on the next activity.
    """

    def __init__(
        self,
        heartbeat_dir: str,
        lease_ttl_s: float,
        runtime: SandboxContainerRuntime,
        *,
        stopped_grace_s: float = 300.0,
    ) -> None:
        if not heartbeat_dir:
            raise ValueError("Node reclaimer heartbeat_dir cannot be empty.")
        if lease_ttl_s <= 0:
            raise ValueError("Node reclaimer lease_ttl_s must be greater than zero.")
        if stopped_grace_s <= 0:
            raise ValueError("Node reclaimer stopped_grace_s must be greater than zero.")
        self.heartbeat_dir = heartbeat_dir
        self.lease_ttl_s = lease_ttl_s
        self.runtime = runtime
        self.stopped_grace_s = stopped_grace_s
        self.lease_store = lease_store_id(heartbeat_dir)
        self._stopped = _StoppedTracker()

    def sweep(self, *, now: float | None = None) -> ReclaimOutcome:
        """
        Reclaim every container this node no longer has a live owner for.
        """
        current_time = time.time() if now is None else now
        containers = self.runtime.list_owned(self.lease_store)
        if containers is None:
            return ReclaimOutcome(removed=(), remaining=None)
        stale_owners = self._stale_owners(containers, current_time)
        abandoned = self._stopped.select(containers, self.stopped_grace_s, current_time) if containers else set()
        doomed = [
            item.container_id for item in containers if item.owner_id in stale_owners or item.container_id in abandoned
        ]
        removed, unremovable = self._remove(doomed)
        if removed:
            self._stopped.forget(removed)
        # Re-list after a removal so a container the runtime refused to delete is
        # still counted, and the collector never mistakes it for a clean node.
        remaining = self._count_remaining(containers, removed)
        self._prune_owner_heartbeats(remaining, current_time)
        return ReclaimOutcome(removed=removed, unremovable=unremovable, remaining=remaining)

    def _stale_owners(self, containers: Sequence[OwnedContainer], now: float) -> set[str]:
        """Read each lease once, because a filesystem error is not evidence of death."""
        stale: set[str] = set()
        for owner_id in {item.owner_id for item in containers}:
            try:
                age = owner_heartbeat_age_s(self.heartbeat_dir, owner_id, now=now)
            except OSError as exc:
                psrl_logger.warning(f"Could not inspect sandbox owner lease {owner_id!r}: {exc}.")
                continue
            if age is None or age > self.lease_ttl_s:
                stale.add(owner_id)
        return stale

    def _remove(self, container_ids: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """
        Destroy containers and report both what went and what would not.
        """
        if not container_ids:
            return (), ()
        try:
            still_present = set(self.runtime.remove_containers(list(container_ids)))
        except Exception:
            # An unreachable runtime is not evidence that anything was removed.
            psrl_logger.warning("Sandbox reclaimer could not remove stale containers.", exc_info=True)
            return (), tuple(container_ids)
        removed = tuple(item for item in container_ids if item not in still_present)
        unremovable = tuple(item for item in container_ids if item in still_present)
        if unremovable:
            psrl_logger.warning(
                f"Sandbox reclaimer could not remove {len(unremovable)} container(s): {list(unremovable)}."
            )
        return removed, unremovable

    def _count_remaining(self, containers: Sequence[OwnedContainer], removed: Sequence[str]) -> int:
        removed_ids = set(removed)
        survivors = [item for item in containers if item.container_id not in removed_ids]
        if not removed_ids:
            return len(survivors)
        try:
            refreshed = self.runtime.list_owned(self.lease_store)
        except Exception:
            refreshed = None
        if refreshed is None:
            # The removal was not confirmed, so report what was listed before it.
            return len(containers) - len(removed_ids)
        return len(refreshed)

    def _prune_owner_heartbeats(self, remaining: int, now: float) -> None:
        """Remove expired heartbeat files that no surviving container references.

        A collector has no list of live owners other than the containers it can
        see, and an expired file that a returning worker re-creates is harmless.
        A file left behind forever would keep that worker's lease looking live
        after it exits.
        """
        if remaining is None:
            return
        try:
            entries = list(os.scandir(self.heartbeat_dir))
        except OSError:
            return
        live: set[str] = set()
        try:
            containers = self.runtime.list_owned(self.lease_store)
        except Exception:
            containers = None
        if containers:
            live = {item.owner_id for item in containers}
        live_paths = {owner_heartbeat_path(self.heartbeat_dir, owner_id) for owner_id in live}
        for entry in entries:
            if not entry.name.startswith("owner-") or not entry.name.endswith(".hb"):
                continue
            if entry.path in live_paths:
                continue
            try:
                if now - entry.stat().st_mtime > self.lease_ttl_s:
                    os.unlink(entry.path)
            except OSError:
                continue


def _gc_lock_path(heartbeat_dir: str) -> str:
    """
    Return the advisory lock path that admits one collector per lease store.
    """
    digest = lease_store_id(heartbeat_dir)
    return os.path.join(os.path.dirname(os.path.abspath(heartbeat_dir)), f"psrl-sandbox-gc-{digest}.lock")


def _acquire_gc_lock(lock_path: str) -> BinaryIO | None:
    """
    Acquire a process-scoped advisory lock without stale lock cleanup.
    """
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    handle = open(lock_path, "a+b")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()).encode())
        handle.flush()
    except BlockingIOError:
        handle.close()
        return None
    except BaseException:
        handle.close()
        raise
    return handle


def _release_gc_lock(handle: BinaryIO) -> None:
    """
    Release and close a collector advisory lock.
    """
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def run_reclaimer_loop(
    reclaimer: NodeReclaimer,
    *,
    interval_s: float,
    idle_exit_cycles: int = 10,
) -> int:
    """
    Sweep periodically and exit after sustained node idleness.

    Only one collector per lease store runs, so a worker that restarts its node
    collector cannot have two processes racing the same containers.
    """
    if interval_s < 1 or idle_exit_cycles < 1:
        raise ValueError("Node reclaimer requires interval_s >= 1 and idle_exit_cycles >= 1.")
    if reclaimer.stopped_grace_s <= interval_s:
        raise ValueError(
            "Node reclaimer requires stopped_grace_s > interval_s, or one sweep cannot grant a grace "
            "period across two observations."
        )
    lock_handle = _acquire_gc_lock(_gc_lock_path(reclaimer.heartbeat_dir))
    if lock_handle is None:
        return 0
    try:
        os.makedirs(reclaimer.heartbeat_dir, exist_ok=True)
        idle_cycles = 0
        while True:
            try:
                outcome = reclaimer.sweep()
            except Exception:
                psrl_logger.warning("Sandbox reclaimer sweep failed. Retrying next interval.", exc_info=True)
                outcome = ReclaimOutcome(removed=(), remaining=None)
            if outcome.remaining is None or outcome.remaining > 0:
                idle_cycles = 0
            else:
                idle_cycles += 1
                if idle_cycles >= idle_exit_cycles:
                    psrl_logger.info("Sandbox reclaimer exiting because no sandbox containers remain.")
                    return 0
            time.sleep(min(interval_s, _MAX_SWEEP_INTERVAL_S))
    finally:
        _release_gc_lock(lock_handle)


def _build_runtime(kind: str, docker_command: Sequence[str]) -> SandboxContainerRuntime:
    """Resolve a runtime name to an implementation.

    The import is local because the Docker runtime lives with the other Docker
    code, which in turn imports this module for its lease store id.
    """
    if kind == "docker":
        from psrl.sandbox.backends.docker.cli import DockerContainerRuntime

        return DockerContainerRuntime(docker_command)
    raise ValueError(f"Unknown sandbox container runtime {kind!r}.")


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run a node collector as a program.
    """
    parser = argparse.ArgumentParser(description="Reclaim PSRL sandbox containers whose owner is gone.")
    parser.add_argument("--heartbeat-dir", required=True)
    parser.add_argument("--lease-ttl-s", type=float, required=True)
    parser.add_argument("--interval-s", type=float, required=True)
    parser.add_argument("--idle-exit-cycles", type=int, default=10)
    parser.add_argument("--stopped-grace-s", type=float, default=300.0)
    parser.add_argument("--runtime", default="docker")
    parser.add_argument("--docker-command", nargs="+", default=["docker"])
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    reclaimer = NodeReclaimer(
        arguments.heartbeat_dir,
        arguments.lease_ttl_s,
        _build_runtime(arguments.runtime, arguments.docker_command),
        stopped_grace_s=arguments.stopped_grace_s,
    )
    return run_reclaimer_loop(
        reclaimer,
        interval_s=arguments.interval_s,
        idle_exit_cycles=arguments.idle_exit_cycles,
    )


def spawn_node_reclaimer(
    heartbeat_dir: str,
    lease_ttl_s: float,
    interval_s: float,
    *,
    idle_exit_cycles: int = 10,
    stopped_grace_s: float = 300.0,
    docker_command: Sequence[str] = ("docker",),
) -> subprocess.Popen | None:
    """
    Start a detached node collector that shares one advisory lock per lease store.
    """
    if interval_s < 1 or lease_ttl_s <= 0 or not docker_command:
        raise ValueError("Node reclaimer requires interval_s >= 1, lease_ttl_s > 0, and a Docker command.")
    package_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        return subprocess.Popen(
            [
                sys.executable,
                "-m",
                "psrl.sandbox.reclaimer",
                "--heartbeat-dir",
                heartbeat_dir,
                "--lease-ttl-s",
                str(lease_ttl_s),
                "--interval-s",
                str(interval_s),
                "--idle-exit-cycles",
                str(idle_exit_cycles),
                "--stopped-grace-s",
                str(stopped_grace_s),
                "--docker-command",
                *docker_command,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
            cwd=package_root,
        )
    except OSError as exc:
        psrl_logger.warning(f"Could not spawn the node sandbox reclaimer: {exc}.")
        return None


if __name__ == "__main__":
    raise SystemExit(main())
