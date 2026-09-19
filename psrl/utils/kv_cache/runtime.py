"""Lifecycle management for the LMCache MP server and coordinator processes.

LMCache runs out of process under the multiprocess (MP) connector, so PSRL
owns process startup, port allocation, log capture, and readiness polling.
This module is intentionally transport-only: it never inspects or mutates
cache state. Cache operations live in :mod:`psrl.utils.kv_cache.manager`.
"""

import asyncio
import logging
import os
import subprocess
import sys

import aiohttp

from psrl.utils.kv_cache.config import LMCacheConfig

psrl_logger = logging.getLogger(__file__)

# Readiness polling cadence and per-probe timeout.
_POLL_INTERVAL_S = 0.5
_PROBE_TIMEOUT_S = 2.0

# Grace period for a terminated process before it is killed.
_TERMINATE_GRACE_S = 10.0


class LMCacheMPRuntimeError(RuntimeError):
    """Raised when an MP process fails to start or become ready."""


async def _probe(url: str) -> bool:
    """
    Return whether an HTTP GET to `url` succeeds.

    Args:
        url (str): Absolute URL to probe.

    Returns:
        bool: True if the endpoint returned a 2xx response.
    """
    timeout = aiohttp.ClientTimeout(total=_PROBE_TIMEOUT_S)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                return 200 <= resp.status < 300
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
        return False


async def _await_ready(url: str, timeout_s: float, what: str) -> None:
    """
    Wait until `url` responds or the timeout elapses.

    Args:
        url (str): Readiness endpoint.
        timeout_s (float): Maximum seconds to wait.
        what (str): Human-readable process name for error messages.

    Raises:
        LMCacheMPRuntimeError: If the endpoint never became ready.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        if await _probe(url):
            return
        await asyncio.sleep(_POLL_INTERVAL_S)
    raise LMCacheMPRuntimeError(
        f"{what} did not become ready at {url} within {timeout_s:.0f}s. Check the process log for startup errors."
    )


def _spawn(argv: list[str], log_path: str, what: str) -> subprocess.Popen:
    """
    Launch a detached LMCache process with its output redirected to a file.

    Args:
        argv (list[str]): Full command line.
        log_path (str): File to append stdout/stderr to.
        what (str): Human-readable process name for logging.

    Returns:
        subprocess.Popen: The running process handle.

    Raises:
        LMCacheMPRuntimeError: If the process could not be spawned.
    """
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    psrl_logger.info(f"[LMCache] Starting {what}: {' '.join(argv)} (log: {log_path})")
    try:
        with open(log_path, "ab") as log_file:
            return subprocess.Popen(
                argv,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except OSError as e:
        raise LMCacheMPRuntimeError(f"Failed to start {what}: {e}") from e


def _terminate(process: subprocess.Popen, what: str) -> None:
    """
    Terminate a child process, escalating to SIGKILL after a grace period.

    Args:
        process (subprocess.Popen): The process to stop.
        what (str): Human-readable process name for logging.
    """
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=_TERMINATE_GRACE_S)
    except subprocess.TimeoutExpired:
        psrl_logger.warning(f"[LMCache] {what} ignored SIGTERM. Killing it.")
        process.kill()


def _server_cli(*args: str) -> list[str]:
    """
    Build a `lmcache` CLI command line runnable from any environment.

    Running the module with the current interpreter avoids depending on the
    console script being on PATH inside Ray workers.

    Args:
        *args: Subcommand and arguments, e.g. `("server", "--port", "5555")`.

    Returns:
        list[str]: The full argv.
    """
    return [sys.executable, "-m", "lmcache.cli.main", *args]


class LMCacheMPRuntime:
    """
    Own one node's LMCache MP server process.

    The server owns all L1 state for the local KV ranks, so one instance is
    started per node and every local vLLM worker connects to it.
    """

    def __init__(self, config: LMCacheConfig) -> None:
        """
        Initialize the runtime.

        Args:
            config (LMCacheConfig): Resolved LMCache configuration with the
                runtime ports already allocated.
        """
        self.config = config
        self._process: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        """HTTP base URL of this node's MP server."""
        return self.config.http_base_url

    async def start(self, log_path: str) -> None:
        """
        Start the MP server and wait for it to become ready.

        Args:
            log_path (str): File to receive the server's output.

        Raises:
            LMCacheMPRuntimeError: If the server fails to start or report
                healthy within `coordinator_health_timeout_s`.
        """
        if self._process is not None:
            return
        argv = _server_cli("server", *self.config.to_server_argv())
        self._process = _spawn(argv, log_path, "LMCache MP server")
        try:
            await _await_ready(
                f"{self.base_url}/healthcheck",
                float(self.config.coordinator_health_timeout_s),
                "LMCache MP server",
            )
        except LMCacheMPRuntimeError:
            self.stop()
            raise
        psrl_logger.info(f"[LMCache] MP server ready at {self.base_url}.")

    def stop(self) -> None:
        """Terminate the MP server process if it is running."""
        process = self._process
        self._process = None
        if process is not None:
            _terminate(process, "MP server")


async def start_coordinator(config: LMCacheConfig, log_path: str) -> subprocess.Popen:
    """
    Start the cluster-wide MP coordinator and wait for it to become ready.

    The coordinator provides instance registration and peer discovery for
    P2P. It is shared by every rollout instance, so callers must start it
    once per training run before launching any MP server.

    Args:
        config (LMCacheConfig): Resolved LMCache configuration.
        log_path (str): File to receive the coordinator's output.

    Returns:
        subprocess.Popen: The running coordinator process handle.

    Raises:
        LMCacheMPRuntimeError: If the coordinator fails to start or report
            ready within `coordinator_health_timeout_s`.
    """
    argv = _server_cli("coordinator", *config.to_coordinator_argv())
    process = _spawn(argv, log_path, "LMCache coordinator")
    readiness_url = f"http://{config.coordinator_host}:{config.coordinator_port}/instances"
    try:
        await _await_ready(
            readiness_url,
            float(config.coordinator_health_timeout_s),
            "LMCache coordinator",
        )
    except LMCacheMPRuntimeError:
        _terminate(process, "coordinator")
        raise
    psrl_logger.info(f"[LMCache] Coordinator ready at {config.coordinator_host}:{config.coordinator_port}.")
    return process
