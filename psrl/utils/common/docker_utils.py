"""
Docker container management utilities.

Provides reusable functions for Docker container lifecycle management,
primarily cleanup of containers identified by labels.

Two cleanup paths are exposed here:

- ``cleanup_containers_by_label`` (async, ``docker stop``): the fast path used
  by per-episode ``finally`` blocks while the trainer is alive.
- ``force_remove_containers_by_label`` (sync, ``docker rm -f``): the last-resort
  sweep used by the per-actor reaper sidecar after its parent process has died,
  and as the synchronous belt-and-suspenders cleanup invoked from the actor's
  ``atexit`` handler.

The reaper sidecar itself is launched by ``spawn_actor_reaper`` as a small
``bash`` process (NOT a Python child) so that it has no import phase during
which a stray SIGTERM could kill it before its signal handlers and polling
loop are ready. On this host the conda Python lives on a slow network
filesystem and ``python -m psrl.utils.common.docker_utils`` takes 10-60 s to
finish importing ``psrl/__init__.py``; bash starts in milliseconds and has
no import phase at all.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


# Default poll interval for the bash reaper sidecar (seconds).
# A worst-case container leak after parent death is bounded by this value
# plus the time taken by ``docker rm -f`` (typically <30 s for hundreds of
# containers).
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
            "docker", "ps", "-q", "--filter", f"label={label_key}={label_value}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await find_proc.communicate()
        container_ids = [cid for cid in stdout.decode().strip().split() if cid]

        if not container_ids:
            psrl_logger.debug(
                f"No containers found with label {label_key}={label_value!r}."
            )
            return []

        psrl_logger.info(
            f"Stopping {len(container_ids)} container(s) with label "
            f"{label_key}={label_value!r}."
        )
        stop_proc = await asyncio.create_subprocess_exec(
            "docker", "stop", "-t", str(stop_timeout), *container_ids,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(stop_proc.communicate(), timeout=30.0)
        psrl_logger.info(
            f"Stopped {len(container_ids)} container(s) with label "
            f"{label_key}={label_value!r}."
        )
        return container_ids

    except asyncio.TimeoutError:
        psrl_logger.warning(
            f"Timeout stopping containers with label {label_key}={label_value!r}."
        )
        return []
    except Exception as e:
        psrl_logger.warning(
            f"Failed to cleanup containers with label {label_key}={label_value!r}: {e}."
        )
        return []


def force_remove_containers_by_label(
    label_key: str,
    label_value: str,
) -> list[str]:
    """
    Force-remove Docker containers matching a specific label.

    Synchronous sibling of :func:`cleanup_containers_by_label`. Uses
    ``docker rm -f`` rather than ``docker stop`` so containers stuck on a
    ``sleep`` or hung ``exec`` are reclaimed immediately, not after the
    per-image ``stop`` grace period. Idempotent.

    Called from two places:

    - The actor's ``atexit`` hook (synchronous, in-process belt cleanup).
    - The bash reaper sidecar runs the equivalent ``docker rm -f`` directly
      via shell after detecting parent death; this Python helper is not used
      there because the sidecar is a pure shell process.

    Args:
        label_key: Docker label key to filter by.
        label_value: Docker label value to filter by.

    Returns:
        List of container IDs that were force-removed.
    """
    try:
        find_out = subprocess.run(
            ["docker", "ps", "-aq", "--filter", f"label={label_key}={label_value}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        container_ids = [
            cid for cid in find_out.stdout.decode().strip().split() if cid
        ]

        if not container_ids:
            psrl_logger.debug(
                f"No containers found with label {label_key}={label_value!r}."
            )
            return []

        psrl_logger.info(
            f"Force-removing {len(container_ids)} container(s) with label "
            f"{label_key}={label_value!r}."
        )
        subprocess.run(
            ["docker", "rm", "-f", *container_ids],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
        )
        psrl_logger.info(
            f"Force-removed {len(container_ids)} container(s) with label "
            f"{label_key}={label_value!r}."
        )
        return container_ids

    except subprocess.TimeoutExpired:
        psrl_logger.warning(
            f"Timeout force-removing containers with label "
            f"{label_key}={label_value!r}."
        )
        return []
    except Exception as e:
        psrl_logger.warning(
            f"Failed to force-remove containers with label "
            f"{label_key}={label_value!r}: {e}."
        )
        return []


def spawn_actor_reaper(
    actor_id: str,
    log_dir: str | None = None,
    poll_interval: int = _REAPER_POLL_INTERVAL_SECS,
) -> subprocess.Popen:
    """
    Spawn the per-actor bash reaper sidecar.

    Bash starts in milliseconds even from a slow network filesystem, so there
    is no startup window during which a SIGTERM from the dying actor (e.g. via
    the actor's own ``atexit.terminate()`` call, or Ray's process-group
    teardown) could kill the reaper before it has a chance to install signal
    handlers. The sidecar polls the spawning PID via ``kill -0`` every
    ``poll_interval`` seconds; on parent death it ``docker rm -f``s every
    container carrying ``psrl.actor_id=<actor_id>``. ``nohup setsid`` plus
    ``start_new_session=True`` give three layers of protection so SIGHUP /
    SIGTERM directed at the parent's process group will not propagate.

    Args:
        actor_id: Stable identifier for the spawning actor; must match the
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
    # NOTE(reaper): all variables that come from Python are interpolated into
    # the script body via f-string at spawn time; the ``$$``, ``$(date ...)``
    # and ``$ids`` references are evaluated by bash at runtime.
    #
    # The sweep is wrapped in a small retry loop (3 passes, 2s apart) and
    # invokes ``docker rm -f`` per container ID rather than via a single
    # ``xargs`` so a single-id failure does not mask the rest, and so we can
    # log the actual stderr from docker. Without this, a single stuck
    # container (e.g. one with active ``docker exec`` sessions on an
    # overloaded host) could survive a one-shot sweep and we would never know
    # which one or why.
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
            log_fd: int | object = open(
                os.path.join(log_dir, f"reaper_{actor_id}.log"), "ab"
            )
        except OSError as e:
            psrl_logger.warning(
                f"Could not open reaper log file under {log_dir!r}: {e}; "
                f"discarding reaper output."
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
