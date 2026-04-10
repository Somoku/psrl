"""
Docker container management utilities.

Provides reusable functions for Docker container lifecycle management,
primarily cleanup of containers identified by labels.
"""

from __future__ import annotations

import asyncio
import logging
import os

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


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
