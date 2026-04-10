"""
General-purpose async subprocess execution.

Provides a reusable utility for running external commands with:
- Configurable timeout with SIGTERM → SIGKILL escalation (15s grace period)
- stdout/stderr capture to log files
- Clean return value (returncode, stdout, stderr)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

_SIGTERM_GRACE_PERIOD = 15.0


async def run_subprocess(
    cmd: list[str],
    *,
    timeout: int = 1800,
    stdout_path: str | None = None,
    stderr_path: str | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> tuple[int, str, str]:
    """
    Run a command as an async subprocess with timeout and log capture.

    Uses SIGTERM → SIGKILL escalation: on timeout, sends SIGTERM first and waits
    up to 15 seconds for graceful exit before sending SIGKILL.

    Args:
        cmd: Command and arguments to execute.
        timeout: Overall timeout in seconds. Defaults to 1800.
        stdout_path: If provided, write stdout to this file path.
        stderr_path: If provided, write stderr to this file path.
        cwd: Working directory for the subprocess.
        env: Environment variables. If None, inherits current environment.

    Returns:
        Tuple of (returncode, stdout_text, stderr_text).
        returncode is -1 if the process was killed due to timeout.
    """
    effective_env = env if env is not None else os.environ.copy()

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=effective_env,
        cwd=cwd,
    )

    start_time = time.time()

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        psrl_logger.warning(
            f"Subprocess timed out after {elapsed:.1f}s (limit={timeout}s), "
            f"sending SIGTERM to pid={process.pid}."
        )
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=_SIGTERM_GRACE_PERIOD)
            psrl_logger.info(f"Subprocess pid={process.pid} exited gracefully after SIGTERM.")
        except asyncio.TimeoutError:
            psrl_logger.warning(
                f"Subprocess pid={process.pid} did not exit after SIGTERM, sending SIGKILL."
            )
            process.kill()
            await process.wait()
        return -1, "", ""

    stdout_text = stdout_bytes.decode(errors="replace")
    stderr_text = stderr_bytes.decode(errors="replace")

    # Persist logs if paths provided.
    if stdout_path:
        _write_log(stdout_path, stdout_text)
    if stderr_path:
        _write_log(stderr_path, stderr_text)

    return process.returncode, stdout_text, stderr_text


def _write_log(path: str, content: str) -> None:
    """
    Write content to a log file, creating parent directories if needed.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        psrl_logger.warning(f"Failed to write log to {path!r}: {e}.")
