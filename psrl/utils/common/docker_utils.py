"""
Manage labeled Docker containers and actor reaper sidecars.

The reaper uses a shell process so cleanup remains available after its actor exits.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


# Upper bound on reaper polling latency after parent exit.
_REAPER_POLL_INTERVAL_SECS = 5


async def cleanup_containers_by_label(
    label_key: str,
    label_value: str,
    stop_timeout: int = 10,
) -> list[str]:
    """
    Stop Docker containers matching a specific label.

    Uses ``docker ps -q --filter label={key}={value}`` to find containers,
    then ``docker stop -t {stop_timeout}`` to stop them. Idempotent: no-op
    if no containers match.

    Args:
        label_key: Docker label key to filter by.
        label_value: Docker label value to filter by.
        stop_timeout: Seconds to wait for graceful stop before Docker force-kills.

    Returns:
        List of container IDs that were stopped.
    """
    try:
        find_proc = await asyncio.create_subprocess_exec(
            "docker",
            "ps",
            "-q",
            "--filter",
            f"label={label_key}={label_value}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await find_proc.communicate()
        container_ids = [cid for cid in stdout.decode().strip().split() if cid]

        if not container_ids:
            psrl_logger.debug(f"No containers found with label {label_key}={label_value!r}.")
            return []

        psrl_logger.info(f"Stopping count={len(container_ids)} container(s) with label {label_key}={label_value!r}.")
        stop_proc = await asyncio.create_subprocess_exec(
            "docker",
            "stop",
            "-t",
            str(stop_timeout),
            *container_ids,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(stop_proc.communicate(), timeout=30.0)
        psrl_logger.info(f"Stopped count={len(container_ids)} container(s) with label {label_key}={label_value!r}.")
        return container_ids

    except asyncio.TimeoutError:
        psrl_logger.warning(f"Timeout stopping containers with label {label_key}={label_value!r}.")
        return []
    except Exception as e:
        psrl_logger.warning(f"Failed to cleanup containers with label {label_key}={label_value!r}: {e}.")
        return []


def force_remove_containers_by_label(
    label_key: str,
    label_value: str,
) -> list[str]:
    """
    Force-remove Docker containers matching a specific label.

    Args:
        label_key: Docker label key to filter by.
        label_value: Docker label value to filter by.

    Returns:
        List of container IDs that were force-removed.
    """
    try:
        find_out = subprocess.run(
            ["docker", "ps", "-aq", "--filter", f"label={label_key}={label_value}"],
            capture_output=True,
            timeout=30,
        )
        container_ids = [cid for cid in find_out.stdout.decode().strip().split() if cid]

        if not container_ids:
            psrl_logger.debug(f"No containers found with label {label_key}={label_value!r}.")
            return []

        psrl_logger.info(
            f"Force-removing count={len(container_ids)} container(s) with label {label_key}={label_value!r}."
        )
        subprocess.run(
            ["docker", "rm", "-f", *container_ids],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
        )
        psrl_logger.info(
            f"Force-removed count={len(container_ids)} container(s) with label {label_key}={label_value!r}."
        )
        return container_ids

    except subprocess.TimeoutExpired:
        psrl_logger.warning(f"Timeout force-removing containers with label {label_key}={label_value!r}.")
        return []
    except Exception as e:
        psrl_logger.warning(f"Failed to force-remove containers with label {label_key}={label_value!r}: {e}.")
        return []


def spawn_actor_reaper(
    actor_id: str,
    log_dir: str | None = None,
    poll_interval: int = _REAPER_POLL_INTERVAL_SECS,
) -> subprocess.Popen:
    """
    Spawn the per-actor bash reaper sidecar.

    The sidecar polls the parent PID and removes matching containers after parent exit.

    Args:
        actor_id: Stable identifier for the spawning actor. Must match the
            ``psrl.actor_id`` label stamped on every container the actor spawns.
        log_dir: If given, append the reaper's stdout/stderr to
            ``<log_dir>/reaper_<actor_id>.log`` for post-mortem debugging.
            If None, output is discarded.
        poll_interval: Seconds between ``kill -0`` liveness checks. The reaper
            reaps within roughly this many seconds of parent death.

    Returns:
        The :class:`subprocess.Popen` handle. Callers should retain it (e.g.
        on ``self``) so it is not garbage-collected, and should call
        ``terminate()`` on graceful shutdown via ``atexit`` to skip a
        redundant post-mortem sweep.
    """
    parent_pid = os.getpid()
    label = f"psrl.actor_id={actor_id}"
    # NOTE(reaper): Python interpolates actor values at spawn time. Shell variables
    # such as `$$`, `$(date ...)`, and `$ids` remain for runtime expansion.
    script = f"""
set -u
echo "[reaper start] pid=$$ parent_pid={parent_pid} actor_id={actor_id} ts=$(date -Is)"
while kill -0 {parent_pid} 2>/dev/null; do
    sleep {poll_interval}
done
echo "[reaper] parent {parent_pid} gone at $(date -Is); reaping label={label}"
attempt=0
max_attempts=3
while [ $attempt -lt $max_attempts ]; do
    attempt=$((attempt + 1))
    ids=$(docker ps -aq --filter "label={label}" 2>/dev/null)
    if [ -z "$ids" ]; then
        if [ $attempt -eq 1 ]; then
            echo "[reaper] no containers to reap"
        else
            echo "[reaper] all containers reaped after $((attempt - 1)) pass(es)"
        fi
        break
    fi
    n=$(echo "$ids" | wc -l)
    echo "[reaper] pass $attempt/$max_attempts: force-removing $n container(s)"
    fail=0
    for cid in $ids; do
        out=$(docker rm -f "$cid" 2>&1)
        rc=$?
        if [ $rc -ne 0 ]; then
            fail=$((fail + 1))
            echo "[reaper] rm -f $cid FAILED rc=$rc: $out"
        fi
    done
    if [ $fail -eq 0 ]; then
        echo "[reaper] pass $attempt/$max_attempts: all $n removed"
        break
    fi
    echo "[reaper] pass $attempt/$max_attempts: $fail failure(s); will retry"
    sleep 2
done
remaining=$(docker ps -aq --filter "label={label}" 2>/dev/null)
if [ -n "$remaining" ]; then
    rn=$(echo "$remaining" | wc -l)
    echo "[reaper] STILL ALIVE after $max_attempts pass(es): $rn container(s):"
    echo "$remaining" | sed 's/^/[reaper]   /'
else
    echo "[reaper] done at $(date -Is)"
fi
"""
    if log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
            log_fd: int | object = open(os.path.join(log_dir, f"reaper_{actor_id}.log"), "ab")
        except OSError as e:
            psrl_logger.warning(
                f"Could not open reaper log file under {log_dir!r}: {e}. Reaper output will be discarded."
            )
            log_fd = subprocess.DEVNULL
    else:
        log_fd = subprocess.DEVNULL

    return subprocess.Popen(
        ["nohup", "setsid", "bash", "-c", script],
        stdout=log_fd,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
