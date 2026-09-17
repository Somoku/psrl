"""Docker CLI helpers used only by out-of-process crash recovery."""

from __future__ import annotations

import fcntl
import hashlib
import logging
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from typing import BinaryIO

psrl_logger = logging.getLogger(__file__)


def _command(docker_command: Sequence[str], *args: str) -> list[str]:
    """Build one Docker CLI invocation from a configured command prefix."""
    if not docker_command:
        raise ValueError("Docker command cannot be empty.")
    return [*docker_command, *args]


def force_remove_containers_by_label(
    label_key: str,
    label_value: str,
    *,
    docker_command: Sequence[str] = ("docker",),
) -> list[str]:
    """Force-remove all Docker containers matching one label."""
    label = f"{label_key}={label_value}"
    try:
        result = subprocess.run(
            _command(docker_command, "ps", "-aq", "--filter", f"label={label}"),
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            psrl_logger.warning(
                f"Could not list Docker containers with label {label!r}: "
                f"{result.stderr.decode(errors='replace').strip()}."
            )
            return []
        container_ids = result.stdout.decode().split()
        if not container_ids:
            return []
        remove_result = subprocess.run(
            _command(docker_command, "rm", "-f", "-v", *container_ids),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=120,
        )
        if remove_result.returncode != 0:
            psrl_logger.warning(
                f"Docker could not remove every container with label {label!r}: "
                f"{remove_result.stderr.decode(errors='replace').strip()}."
            )
        return container_ids
    except (OSError, subprocess.TimeoutExpired) as exc:
        psrl_logger.warning(f"Failed to remove Docker containers with label {label!r}: {exc}.")
        return []


def lease_store_id(heartbeat_dir: str) -> str:
    """
    Identify the lease namespace shared by workers and their collector.
    """
    return hashlib.sha256(os.path.abspath(heartbeat_dir).encode()).hexdigest()[:16]


def owner_heartbeat_path(heartbeat_dir: str, owner_id: str) -> str:
    """Return the heartbeat file path for one sandbox owner."""
    owner_key = hashlib.sha256(owner_id.encode()).hexdigest()
    return os.path.join(heartbeat_dir, f"owner-{owner_key}.hb")


def write_owner_heartbeat(heartbeat_dir: str, owner_id: str) -> None:
    """Create or refresh one sandbox owner's heartbeat file."""
    os.makedirs(heartbeat_dir, exist_ok=True)
    path = owner_heartbeat_path(heartbeat_dir, owner_id)
    with open(path, "a"):
        pass
    os.utime(path, None)


def remove_owner_heartbeat(heartbeat_dir: str, owner_id: str) -> None:
    """Delete an owner's heartbeat file when it exists."""
    try:
        os.unlink(owner_heartbeat_path(heartbeat_dir, owner_id))
    except FileNotFoundError:
        pass


def owner_heartbeat_age_s(heartbeat_dir: str, owner_id: str, *, now: float | None = None) -> float | None:
    """Return an owner heartbeat's age, or `None` when it is missing."""
    try:
        modified_at = os.stat(owner_heartbeat_path(heartbeat_dir, owner_id)).st_mtime
    except FileNotFoundError:
        return None
    current_time = time.time() if now is None else now
    return max(0.0, current_time - modified_at)


def _list_sandbox_containers(
    docker_command: Sequence[str],
    *,
    lease_store: str,
) -> list[tuple[str, str]] | None:
    """Return `(container_id, owner_id)` for every owned PSRL sandbox."""
    try:
        result = subprocess.run(
            _command(
                docker_command,
                "ps",
                "-a",
                "--filter",
                "label=psrl.sandbox=true",
                "--filter",
                f"label=psrl.lease_store={lease_store}",
                "--format",
                '{{.ID}}\t{{.Label "psrl.actor_id"}}',
            ),
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        psrl_logger.warning(f"Could not list PSRL sandbox containers: {exc}.")
        return None
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        psrl_logger.warning(f"Could not list PSRL sandbox containers: {stderr}.")
        return None
    containers = []
    for line in result.stdout.decode(errors="replace").splitlines():
        container_id, _, owner_id = line.partition("\t")
        if container_id.strip() and owner_id.strip():
            containers.append((container_id.strip(), owner_id.strip()))
    return containers


def _prune_owner_heartbeats(heartbeat_dir: str, live_owners: set[str], ttl_s: float, now: float) -> None:
    """Remove expired heartbeat files that no listed container still references."""
    live_paths = {owner_heartbeat_path(heartbeat_dir, owner_id) for owner_id in live_owners}
    try:
        entries = os.scandir(heartbeat_dir)
    except OSError:
        return
    with entries:
        for entry in entries:
            if not entry.name.startswith("owner-") or not entry.name.endswith(".hb") or entry.path in live_paths:
                continue
            try:
                if now - entry.stat().st_mtime > ttl_s:
                    os.unlink(entry.path)
            except OSError:
                continue


def sweep_stale_sandboxes(
    heartbeat_dir: str,
    ttl_s: float,
    *,
    docker_command: Sequence[str] = ("docker",),
    now: float | None = None,
) -> tuple[list[str], int | None]:
    """Reap containers whose owner lease expired and return the remaining count.

    A `None` remaining count means Docker could not be queried. Collectors treat
    that as an unhealthy sweep, never as an idle node, so transient daemon
    failures cannot make crash recovery silently exit.
    """
    current_time = time.time() if now is None else now
    store_id = lease_store_id(heartbeat_dir)
    containers = _list_sandbox_containers(docker_command, lease_store=store_id)
    if containers is None:
        return [], None
    # Read each lease once. A filesystem error is not evidence of owner death.
    stale_owners = set()
    for owner_id in {owner_id for _, owner_id in containers}:
        try:
            age = owner_heartbeat_age_s(heartbeat_dir, owner_id, now=current_time)
        except OSError as exc:
            psrl_logger.warning(f"Could not inspect Docker owner lease {owner_id!r}: {exc}.")
            continue
        if age is None or age > ttl_s:
            stale_owners.add(owner_id)
    stale = [container_id for container_id, owner_id in containers if owner_id in stale_owners]
    removed: list[str] = []
    if stale:
        try:
            result = subprocess.run(
                _command(docker_command, "rm", "-f", "-v", *stale),
                capture_output=True,
                timeout=120,
            )
            if result.returncode == 0:
                removed = stale
            else:
                psrl_logger.warning(
                    f"Docker could not reap every stale sandbox: {result.stderr.decode(errors='replace').strip()}."
                )
                refreshed = _list_sandbox_containers(docker_command, lease_store=store_id)
                if refreshed is None:
                    return [], None
                remaining_ids = {container_id for container_id, _ in refreshed}
                removed = [container_id for container_id in stale if container_id not in remaining_ids]
                containers = refreshed
        except (OSError, subprocess.TimeoutExpired) as exc:
            psrl_logger.warning(f"Failed to reap stale sandboxes: {exc}.")

    removed_ids = set(removed)
    remaining_containers = [item for item in containers if item[0] not in removed_ids]
    live_owners = {owner_id for _, owner_id in remaining_containers}
    _prune_owner_heartbeats(heartbeat_dir, live_owners, ttl_s, current_time)
    return removed, len(remaining_containers)


def _gc_lock_path(heartbeat_dir: str) -> str:
    """Return the advisory lock path that admits one collector per lease store."""
    digest = lease_store_id(heartbeat_dir)
    return os.path.join(os.path.dirname(os.path.abspath(heartbeat_dir)), f"psrl-sandbox-gc-{digest}.lock")


def _acquire_gc_lock(lock_path: str) -> BinaryIO | None:
    """Acquire a process-scoped advisory lock without stale lock cleanup."""
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
    """Release and close a collector advisory lock."""
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def run_gc_loop(
    heartbeat_dir: str,
    ttl_s: float,
    interval_s: float,
    *,
    idle_exit_cycles: int = 10,
    docker_command: Sequence[str] = ("docker",),
) -> int:
    """Sweep stale sandboxes periodically and exit after sustained node idleness."""
    if interval_s < 1 or ttl_s <= 0 or idle_exit_cycles < 1:
        raise ValueError("Node GC requires interval_s >= 1, ttl_s > 0, and idle_exit_cycles >= 1.")
    lock_handle = _acquire_gc_lock(_gc_lock_path(heartbeat_dir))
    if lock_handle is None:
        return 0
    try:
        os.makedirs(heartbeat_dir, exist_ok=True)
        idle_cycles = 0
        while True:
            try:
                _, remaining = sweep_stale_sandboxes(
                    heartbeat_dir,
                    ttl_s,
                    docker_command=docker_command,
                )
            except Exception:
                psrl_logger.warning("Node sandbox GC sweep failed; retrying next interval.", exc_info=True)
                remaining = None
            if remaining is None or remaining > 0:
                idle_cycles = 0
            else:
                idle_cycles += 1
                if idle_cycles >= idle_exit_cycles:
                    psrl_logger.info("Node sandbox GC exiting because no sandbox containers remain.")
                    return 0
            time.sleep(interval_s)
    finally:
        _release_gc_lock(lock_handle)


_GC_CHILD_CODE = (
    "import sys; sys.path.insert(0, sys.argv[1]); "
    "from psrl.sandbox.utils.docker_utils import run_gc_loop; "
    "raise SystemExit(run_gc_loop(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), "
    "idle_exit_cycles=int(sys.argv[5]), docker_command=tuple(sys.argv[6:])))"
)


def spawn_node_gc(
    heartbeat_dir: str,
    ttl_s: float,
    interval_s: float,
    *,
    idle_exit_cycles: int = 10,
    docker_command: Sequence[str] = ("docker",),
) -> subprocess.Popen | None:
    """Start a detached collector that shares one advisory lock per lease store."""
    if interval_s < 1 or ttl_s <= 0 or not docker_command:
        raise ValueError("Node GC requires interval_s >= 1, ttl_s > 0, and a Docker command.")
    package_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    try:
        return subprocess.Popen(
            [
                sys.executable,
                "-c",
                _GC_CHILD_CODE,
                package_root,
                heartbeat_dir,
                str(ttl_s),
                str(interval_s),
                str(idle_exit_cycles),
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
        psrl_logger.warning(f"Could not spawn node sandbox GC: {exc}.")
        return None
