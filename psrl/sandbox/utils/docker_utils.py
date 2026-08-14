import logging
import os
import shlex
import subprocess

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

_REAPER_POLL_INTERVAL_SECS = 5


def force_remove_containers_by_label(
    label_key: str,
    label_value: str,
    binary: str = "docker",
) -> list[str]:
    """
    Force-remove Docker containers matching one label.

    The implementation batches IDs into one `docker rm -f` invocation to keep
    graceful shutdown overhead independent of the number of containers.

    Args:
        label_key (str): Docker label key to filter by.
        label_value (str): Docker label value to filter by.
        binary (str): Docker CLI executable.

    Returns:
        list[str]: Container IDs submitted to `docker rm -f`.
    """
    label = f"{label_key}={label_value}"
    try:
        result = subprocess.run(
            [binary, "ps", "-aq", "--filter", f"label={label}"],
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
            [binary, "rm", "-f", *container_ids],
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


def spawn_actor_reaper(
    actor_id: str,
    log_dir: str | None = None,
    poll_interval: int = _REAPER_POLL_INTERVAL_SECS,
    binary: str = "docker",
) -> subprocess.Popen:
    """
    Spawn a detached process that reclaims worker-owned containers after death.

    The reaper is a shell process so it becomes ready without importing Python
    or PSRL. `start_new_session=True` isolates it from the Ray actor's process
    group. Each container is removed independently and retried so one stuck
    container cannot hide failures for the remaining containers.

    Args:
        actor_id (str): Worker identity stored in the `psrl.actor_id` label.
        log_dir (str | None): Optional directory for reaper diagnostics.
        poll_interval (int): Parent liveness polling interval in seconds.
        binary (str): Docker CLI executable.

    Returns:
        subprocess.Popen: Detached reaper process.
    """
    if poll_interval < 1:
        raise ValueError("Docker reaper poll_interval must be at least one second.")
    parent_pid = os.getpid()
    docker = shlex.quote(binary)
    label = shlex.quote(f"psrl.actor_id={actor_id}")
    script = f"""
set -u
while kill -0 {parent_pid} 2>/dev/null; do
    sleep {poll_interval}
done
attempt=0
while [ "$attempt" -lt 3 ]; do
    attempt=$((attempt + 1))
    ids=$({docker} ps -aq --filter label={label} 2>/dev/null)
    [ -z "$ids" ] && exit 0
    for cid in $ids; do
        {docker} rm -f "$cid" >/dev/null 2>&1 || true
    done
    sleep 2
done
"""
    log_handle = None
    log_output = subprocess.DEVNULL
    if log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
            log_handle = open(os.path.join(log_dir, f"reaper_{actor_id}.log"), "ab")
            log_output = log_handle
        except OSError as exc:
            psrl_logger.warning(f"Could not open Docker reaper log under {log_dir!r}: {exc}.")

    try:
        return subprocess.Popen(
            ["bash", "-c", script],
            stdin=subprocess.DEVNULL,
            stdout=log_output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        if log_handle is not None:
            log_handle.close()
