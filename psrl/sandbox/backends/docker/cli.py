"""Docker CLI helpers and the Docker container runtime the reclaimer drives."""

from __future__ import annotations

import concurrent.futures
import logging
import re
import subprocess
import threading
import time
from collections.abc import Sequence

from psrl.sandbox.reclaimer import OwnedContainer

psrl_logger = logging.getLogger(__file__)

# Serialize and throttle dangling-image pruning across concurrent episodes.
_PRUNE_LOCK = threading.Lock()
_LAST_PRUNE_MONOTONIC = 0.0
# Images removed per `docker rmi` call, to bound the argument list.
_PRUNE_BATCH_SIZE = 200

# Cleanup runs here rather than on asyncio's default executor, whose small shared
# pool stalls episode I/O when many `docker rm` calls block on a loaded daemon.
CLEANUP_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="psrl-docker-cleanup",
)


def _command(docker_command: Sequence[str], *args: str) -> list[str]:
    """
    Build one Docker CLI invocation from a configured command prefix.
    """
    if not docker_command:
        raise ValueError("Docker command cannot be empty.")
    return [*docker_command, *args]


def signal_container_process_group(
    container_id: str,
    pid: int,
    *,
    docker_command: Sequence[str] = ("docker",),
    signal_name: str = "TERM",
    timeout_secs: float = 30.0,
) -> bool:
    """Signal one process group inside a container, through the shell's builtin.

    An exec'd process is a session leader, so a command it started runs in its own
    process group and signalling the negative id reaches the command and its children.

    The signal goes through `/bin/sh -c` rather than a `kill` binary, because `kill` is a
    shell builtin and the binary is not present in every image. Returns False when the
    signal was not delivered, which is what tells a caller there is no safe way to keep
    the container.

    Args:
        container_id (str): The container to signal in.
        pid (int): The process id the signal targets, used as a group id.
        docker_command (Sequence[str]): Command prefix for the Docker CLI.
        signal_name (str): Signal name without the `SIG` prefix.
        timeout_secs (float): Deadline for the CLI call.

    Returns:
        bool: Whether the signal was delivered.
    """
    argv = _command(
        docker_command,
        "exec",
        container_id,
        "/bin/sh",
        "-c",
        f"kill -{signal_name} -- -{int(pid)}",
    )
    try:
        result = subprocess.run(argv, capture_output=True, timeout=timeout_secs)
    except (OSError, subprocess.TimeoutExpired) as exc:
        psrl_logger.warning(f"Could not signal process group {pid} in container {container_id}: {exc}.")
        return False
    if result.returncode != 0:
        psrl_logger.warning(
            f"Signalling process group {pid} in container {container_id} failed: "
            f"{result.stderr.decode(errors='replace').strip()}."
        )
        return False
    return True


def force_remove_containers_by_label(
    label_key: str,
    label_value: str,
    *,
    docker_command: Sequence[str] = ("docker",),
) -> list[str]:
    """
    Force-remove all Docker containers matching one label.
    """
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


def force_remove_container_ids(
    container_ids: Sequence[str],
    *,
    docker_command: Sequence[str] = ("docker",),
) -> list[str]:
    """Force-remove containers by id and return the ids Docker no longer lists.

    The CLI is a last resort for a container the Engine API refuses to delete. It is a
    separate code path, so it can succeed where the API keeps failing.
    """
    ids = [container_id for container_id in container_ids if container_id]
    if not ids:
        return []
    try:
        subprocess.run(
            _command(docker_command, "rm", "-f", "-v", *ids),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        psrl_logger.warning(f"Failed to force-remove Docker containers {ids!r}: {exc}.")
        return []
    return [container_id for container_id in ids if not _container_exists(container_id, docker_command)]


def _container_exists(container_id: str, docker_command: Sequence[str]) -> bool:
    """
    Report whether Docker still lists one container, defaulting to present.
    """
    try:
        result = subprocess.run(
            _command(docker_command, "inspect", "--format", "{{.Id}}", container_id),
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return True
    return result.returncode == 0


class DockerContainerRuntime:
    """The Docker CLI view of this node's sandbox containers.

    Implements the reclaimer's container runtime interface. It is deliberately
    CLI based rather than Engine API based: reclamation has to keep working when
    the API path is what is failing, and a collector that shares the worker's
    client would share its failure mode.
    """

    def __init__(self, docker_command: Sequence[str] = ("docker",)) -> None:
        if not docker_command:
            raise ValueError("Docker container runtime requires a Docker command.")
        self.docker_command = tuple(docker_command)

    def list_owned(self, lease_store: str) -> Sequence[OwnedContainer] | None:
        """
        Return every PSRL sandbox container in one lease store.
        """
        try:
            result = subprocess.run(
                _command(
                    self.docker_command,
                    "ps",
                    "-a",
                    "--filter",
                    "label=psrl.sandbox=true",
                    "--filter",
                    f"label=psrl.lease_store={lease_store}",
                    "--format",
                    '{{.ID}}\t{{.Label "psrl.actor_id"}}\t{{.State}}',
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
        containers: list[OwnedContainer] = []
        for line in result.stdout.decode(errors="replace").splitlines():
            container_id, _, remainder = line.partition("\t")
            owner_id, _, state = remainder.partition("\t")
            if container_id.strip() and owner_id.strip():
                containers.append(
                    OwnedContainer(container_id.strip(), owner_id.strip(), state.strip().lower())
                )
        return containers

    def remove_containers(self, container_ids: Sequence[str]) -> Sequence[str]:
        """
        Force-remove containers and return the ids Docker still lists.
        """
        ids = [container_id for container_id in container_ids if container_id]
        if not ids:
            return []
        try:
            subprocess.run(
                _command(self.docker_command, "rm", "-f", "-v", *ids),
                capture_output=True,
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            psrl_logger.warning(f"Docker could not reap every stale sandbox: {exc}.")
            return ids
        return [container_id for container_id in ids if _container_exists(container_id, self.docker_command)]


def sanitize_compose_project_name(name: str) -> str:
    """
    Render a name the way Docker Compose derives a project name.

    Mirrors Harbor's private sanitizer, so callers can find an episode's
    containers by the `com.docker.compose.project` label.

    Args:
        name (str): Raw name, normally a Harbor session id.

    Returns:
        str: The sanitized project name.
    """
    name = name.lower()
    if not re.match(r"^[a-z0-9]", name):
        name = "0" + name
    return re.sub(r"[^a-z0-9_-]", "-", name)


def force_remove_compose_project(session_id: str) -> list[str]:
    """
    Force-remove every container Compose created for one episode.

    Cancelling the coroutine that awaits a job does not stop its containers, so
    an abandoned verifier keeps holding CPU and memory until this runs.

    Args:
        session_id (str): Harbor session id used as the Compose project name.

    Returns:
        list[str]: Container IDs that were force-removed.
    """
    if not session_id:
        return []
    return force_remove_containers_by_label(
        "com.docker.compose.project",
        sanitize_compose_project_name(session_id),
    )


def force_remove_compose_images(session_id: str) -> int:
    """
    Remove the tagged images Compose built for one episode.

    Compose names images `<project>-<service>` from a fresh session id, so an
    abandoned episode leaves tagged images that a dangling sweep never sees.

    Args:
        session_id (str): Harbor session id used as the Compose project name.

    Returns:
        int: Number of images removed.
    """
    if not session_id:
        return 0

    project = sanitize_compose_project_name(session_id)
    try:
        # The project prefix selects exactly this episode's images.
        listed = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}", "--filter", f"reference={project}-*"],
            capture_output=True,
            timeout=120,
        )
        names = [name for name in listed.stdout.decode(errors="replace").split() if name]
        if not names:
            return 0
        subprocess.run(
            ["docker", "rmi", "-f", *names],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=300,
        )
        psrl_logger.info(f"Removed {len(names)} image(s) for episode {project}.")
        return len(names)
    except subprocess.TimeoutExpired:
        psrl_logger.warning(f"Timeout removing images for episode {project}.")
        return 0
    except Exception as exc:
        psrl_logger.warning(f"Failed to remove images for episode {project}: {exc}.")
        return 0


def prune_dangling_images(min_interval_secs: float = 900.0, timeout_secs: float = 600.0) -> bool:
    """
    Remove dangling Docker images, at most once per `min_interval_secs`.

    Batched `docker rmi -f` by explicit ID replaces `docker image prune -f`,
    which can hang on a degraded daemon. Only untagged images are touched, so
    task images stay warm for the next episode.

    Args:
        min_interval_secs (float): Minimum wall-clock gap between prunes.
        timeout_secs (float): Upper bound on the whole removal loop.

    Returns:
        bool: Whether a prune actually ran on this call.
    """
    global _LAST_PRUNE_MONOTONIC

    if not _PRUNE_LOCK.acquire(blocking=False):
        return False
    try:
        now = time.monotonic()
        if _LAST_PRUNE_MONOTONIC and now - _LAST_PRUNE_MONOTONIC < min_interval_secs:
            return False
        _LAST_PRUNE_MONOTONIC = now

        deadline = now + timeout_secs
        removed = 0
        # A dangling ID held by a live container keeps reappearing, so track attempts.
        attempted: set[str] = set()
        while time.monotonic() < deadline:
            listed = subprocess.run(
                ["docker", "images", "-f", "dangling=true", "-q"],
                capture_output=True,
                timeout=120,
            )
            ids = [item for item in listed.stdout.decode(errors="replace").split() if item and item not in attempted]
            if not ids:
                break
            batch = ids[:_PRUNE_BATCH_SIZE]
            attempted.update(batch)
            subprocess.run(
                ["docker", "rmi", "-f", *batch],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=300,
            )
            removed += len(batch)
        psrl_logger.info(f"Removed {removed} untagged Docker image(s).")
        return True
    except subprocess.TimeoutExpired:
        psrl_logger.warning(f"Timeout removing untagged images after {timeout_secs}s.")
        return False
    except Exception as exc:
        psrl_logger.warning(f"Failed to prune dangling images: {exc}.")
        return False
    finally:
        _PRUNE_LOCK.release()
