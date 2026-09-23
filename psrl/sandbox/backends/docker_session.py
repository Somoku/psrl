"""Docker session execution, diagnostics, and confirmed destruction."""

from __future__ import annotations

import asyncio
import functools
import logging
import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING

import aiohttp

from psrl.sandbox.async_utils import complete_cleanup
from psrl.sandbox.backends.docker_policy import DockerPolicyProfile
from psrl.sandbox.core import (
    ExecResult,
    PauseMode,
    ResourceUsage,
    SandboxCapabilities,
    SandboxOomError,
    SandboxRef,
    SandboxSession,
    SandboxSpec,
    SandboxStatus,
    SandboxTransportError,
    SnapshotKind,
    SnapshotRef,
)
from psrl.sandbox.utils.docker_utils import CLEANUP_EXECUTOR, force_remove_container_ids

if TYPE_CHECKING:
    from psrl.sandbox.backends.docker import DockerBackend

psrl_logger = logging.getLogger(__file__)

# Backoff for a lifetime cleanup the daemon keeps refusing. Bounded, because a wedged daemon
# would otherwise log once per second per session for the rest of the run.
_LIFETIME_RETRY_ATTEMPTS = 6
_LIFETIME_RETRY_BASE_S = 1.0
_LIFETIME_RETRY_MAX_S = 30.0


def _describe_command(command: str, limit: int = 120) -> str:
    """
    Return a one-line preview of a command for a diagnostic message.
    """
    flat = " ".join(command.split())
    return repr(flat if len(flat) <= limit else f"{flat[:limit]}...")


class DockerSession(SandboxSession):
    """
    One Docker container session.
    """

    def __init__(
        self,
        backend: DockerBackend,
        sandbox_id: str,
        *,
        spec: SandboxSpec | None = None,
        policy: DockerPolicyProfile | None = None,
        lifetime_timeout_s: float | None = None,
    ) -> None:
        self.backend = backend
        self.sandbox_id = sandbox_id
        self._spec = spec
        self._policy = policy or DockerPolicyProfile()
        self._command_count = 0
        self._terminate_lock = asyncio.Lock()
        self._terminated = False
        self._exec_lock = asyncio.Lock()
        self._timeout_task = (
            asyncio.create_task(self._terminate_at_deadline(lifetime_timeout_s))
            if lifetime_timeout_s is not None
            else None
        )

    async def _terminate_at_deadline(self, timeout_s: float, attempt: int = 1) -> None:
        """
        Enforce Docker lifetime locally because Engine has no native TTL.
        """
        try:
            await asyncio.sleep(timeout_s)
            self._timeout_task = None
            with self.backend.metrics.measure("lifetime_timeout"):
                await self.terminate()
        except asyncio.CancelledError:
            return
        except Exception:
            if attempt >= _LIFETIME_RETRY_ATTEMPTS or not self.backend.is_open:
                psrl_logger.error(
                    f"Docker sandbox {self.sandbox_id} outlived its lifetime but could not be removed after "
                    f"{attempt} attempts. The node collector owns it now.",
                    exc_info=True,
                )
                return
            delay = min(_LIFETIME_RETRY_BASE_S * 2 ** (attempt - 1), _LIFETIME_RETRY_MAX_S)
            psrl_logger.warning(
                f"Docker lifetime cleanup failed for {self.sandbox_id}. Retrying in {delay:g}s.", exc_info=True
            )
            self._timeout_task = asyncio.create_task(self._terminate_at_deadline(delay, attempt + 1))

    @property
    def ref(self) -> SandboxRef:
        return SandboxRef(self.backend.name, self.sandbox_id)

    @property
    def capabilities(self) -> SandboxCapabilities:
        return self.backend.capabilities

    async def snapshot(self, kind: SnapshotKind) -> SnapshotRef:
        """
        Capture a filesystem snapshot by committing the container's writable layer.
        """
        if kind != SnapshotKind.FILESYSTEM:
            raise NotImplementedError(f"DockerSession only supports FILESYSTEM snapshots, got {kind.value}.")
        repo = f"psrl/snapshot/{self.sandbox_id}"
        tag = uuid.uuid4().hex
        image_id = await self.backend.engine.commit_container(self.sandbox_id, repo, tag)
        image_tag = f"{repo}:{tag}"
        return SnapshotRef(
            backend=self.backend.name,
            snapshot_id=image_tag,
            kind=SnapshotKind.FILESYSTEM,
            metadata={"psrl.docker.image": image_tag, "psrl.docker.image_id": image_id},
        )

    @property
    def spec(self) -> SandboxSpec | None:
        return self._spec

    @property
    def command_count(self) -> int:
        return self._command_count

    def resolve_callback_url(self, url: str) -> str:
        """
        Rewrite worker-loopback URLs through the configured host gateway.
        """
        return self._policy.resolve_callback_url(url)

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> ExecResult:
        if timeout_s is not None and timeout_s <= 0:
            raise ValueError("Docker command timeout must be positive.")
        try:
            return await asyncio.wait_for(
                self._exec(command, cwd=cwd, env=env, timeout_s=timeout_s),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"Docker command timed out (requested timeout={timeout_s!r}).") from exc

    async def _exec(
        self,
        command: str,
        *,
        cwd: str | None,
        env: Mapping[str, str] | None,
        timeout_s: float | None,
    ) -> ExecResult:
        async with self._exec_lock:
            if self._terminated:
                raise RuntimeError("Docker sandbox is terminated.")
            self._command_count += 1
            exec_task = asyncio.create_task(self._run_command(command, cwd=cwd, env=env, timeout_s=timeout_s))
            watcher = asyncio.create_task(self._watch_container())
            try:
                exit_code, stdout, stderr, truncated = await self._collect_command(exec_task, watcher, timeout_s)
            finally:
                exec_task.cancel()
                watcher.cancel()
                await complete_cleanup(asyncio.gather(exec_task, watcher, return_exceptions=True))
        text_stdout = stdout.decode(errors="replace")
        text_stderr = stderr.decode(errors="replace")
        if truncated:
            psrl_logger.warning(
                f"Docker sandbox {self.sandbox_id} produced more output than the "
                f"{self.backend.max_exec_output_bytes}-byte diagnostic budget while running "
                f"{_describe_command(command)}. The command completed and its output is truncated."
            )
            marker = (
                f"\n[psrl: output truncated at {self.backend.max_exec_output_bytes} bytes. Write the "
                "result to a file and read that file when the full output matters]\n"
            )
            if text_stdout:
                text_stdout += marker
            else:
                text_stderr += marker
        return ExecResult(
            exit_code=exit_code,
            stdout=text_stdout,
            stderr=text_stderr,
            truncated=truncated,
        )

    async def _collect_command(
        self,
        exec_task: asyncio.Task,
        watcher: asyncio.Task,
        timeout_s: float | None,
    ) -> tuple[int, bytes, bytes, bool]:
        """Await a command, or destroy the sandbox and report why it could not finish.

        Closing an exec connection does not stop the process, so every outcome other than a
        completed command destroys the session. A container that stopped underneath the
        command outranks the symptom, because it explains it.
        """
        try:
            done, _ = await asyncio.wait(
                {exec_task, watcher},
                timeout=timeout_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
        except asyncio.CancelledError:
            await complete_cleanup(self._abort_command(exec_task, watcher))
            raise
        if watcher in done:
            reason = watcher.result()
            await complete_cleanup(self._abort_command(exec_task, watcher))
            raise self._stopped_error(reason)
        if exec_task not in done:
            reason = await complete_cleanup(self._abort_command(exec_task, watcher, diagnose=True))
            if reason is not None:
                raise self._stopped_error(reason)
            raise TimeoutError(f"Docker command timed out (requested timeout={timeout_s!r}).")
        try:
            with self.backend.metrics.measure("exec"):
                return exec_task.result()
        except asyncio.CancelledError:
            # The command stream was aborted, which does not stop the process it started.
            await complete_cleanup(self._abort_command(exec_task, watcher))
            raise
        except (TimeoutError, asyncio.TimeoutError, aiohttp.ClientError, SandboxTransportError) as exc:
            reason = await complete_cleanup(self._abort_command(exec_task, watcher, diagnose=True))
            if reason is not None:
                raise self._stopped_error(reason) from exc
            raise

    async def _run_command(
        self,
        command: str,
        *,
        cwd: str | None,
        env: Mapping[str, str] | None,
        timeout_s: float | None,
    ) -> tuple[int, bytes, bytes, bool]:
        """
        Run one command inside the container and return its raw result.
        """
        return await self.backend.engine.exec(
            self.sandbox_id,
            [*self.backend.command_interpreter, command],
            cwd=cwd,
            env=env,
            timeout_s=timeout_s,
        )

    async def _watch_container(self) -> str:
        """Report why the container stopped, as soon as the daemon says it did.

        A container that dies mid-command leaves the exec stream open forever, so the command
        has to be aborted on the stop rather than on a timeout.
        """
        reason = await self.backend.wait_for_stop(self.sandbox_id)
        if reason is not None:
            return reason
        return await self._poll_for_stop()

    async def _poll_for_stop(self) -> str:
        """Detect a stop by polling, for a daemon that will not serve its event stream.

        DEPRECATED(claude): Delete this and `container_watch_interval_s` with it once every
        deployment's daemon grants event access. Polling costs one inspect per interval per
        running command and cannot detect a stop faster than that interval.
        """
        while True:
            await asyncio.sleep(self.backend.container_watch_interval_s)
            reason = await self._container_stop_reason()
            if reason is not None:
                return reason

    async def _abort_command(
        self,
        exec_task: asyncio.Task,
        watcher: asyncio.Task,
        *,
        diagnose: bool = False,
    ) -> str | None:
        exec_task.cancel()
        watcher.cancel()
        await asyncio.gather(exec_task, watcher, return_exceptions=True)
        try:
            return await self._container_stop_reason() if diagnose else None
        finally:
            try:
                await self.terminate()
            except Exception:
                psrl_logger.warning("Docker command cleanup remains owned by the sandbox lease.", exc_info=True)

    async def _container_stop_reason(self) -> str | None:
        """
        Return why the container is no longer running, or None while it is.
        """
        try:
            inspection = await self.backend.engine.inspect_container(self.sandbox_id)
        except (aiohttp.ClientError, SandboxTransportError, TimeoutError, asyncio.TimeoutError):
            return None
        if inspection is None:
            return "removed"
        state = inspection.get("State") or {}
        # Only an explicit `Running: False` proves the container stopped. An inspect
        # payload that omits it must not turn a stream failure into a stop.
        if state.get("Running") is not False:
            return None
        return "oom_killed" if state.get("OOMKilled") else "exited"

    def _stopped_error(self, reason: str) -> RuntimeError:
        if reason == "oom_killed":
            memory_mb = self._spec.resources.memory_mb if self._spec is not None else None
            return SandboxOomError(
                f"Docker sandbox {self.sandbox_id!r} was OOM-killed during a command. memory_limit_mb={memory_mb}."
            )
        return RuntimeError(f"Docker sandbox {self.sandbox_id!r} stopped ({reason}) while a command was running.")

    async def read_bytes(self, path: str) -> bytes:
        with self.backend.metrics.measure("read_bytes"):
            return await self.backend.engine.read_file(self.sandbox_id, path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        with self.backend.metrics.measure("write_bytes"):
            await self.backend.engine.write_file(self.sandbox_id, path, data)

    async def status(self) -> SandboxStatus:
        if self._terminated:
            return SandboxStatus.TERMINATED
        with self.backend.metrics.measure("status"):
            inspection = await self.backend.engine.inspect_container(self.sandbox_id)
        if inspection is None:
            return SandboxStatus.TERMINATED
        state = str((inspection.get("State") or {}).get("Status", "unknown"))
        return {
            "running": SandboxStatus.RUNNING,
            "paused": SandboxStatus.PAUSED,
            "exited": SandboxStatus.EXITED,
            "dead": SandboxStatus.EXITED,
        }.get(state, SandboxStatus.UNKNOWN)

    async def stats(self) -> ResourceUsage:
        with self.backend.metrics.measure("stats"):
            stats = await self.backend.engine.stats(self.sandbox_id)
        memory = stats.get("memory_stats") or {}
        current = int(memory.get("usage", 0) or 0)
        peak = int(memory.get("max_usage", 0) or (memory.get("stats") or {}).get("peak", 0) or current)
        cpu_total = int(((stats.get("cpu_stats") or {}).get("cpu_usage") or {}).get("total_usage", 0) or 0)
        self.backend.metrics.observe_memory(current, peak)
        return ResourceUsage(memory_bytes=current, peak_memory_bytes=peak, cpu_total_ns=cpu_total)

    async def terminate(self) -> None:
        await complete_cleanup(self._terminate())

    async def _terminate(self) -> None:
        async with self._terminate_lock:
            if self._terminated:
                return
            with self.backend.metrics.measure("terminate"):
                await self._destroy_container()
            self._terminated = True
            if self._timeout_task is not None:
                self._timeout_task.cancel()
            self.backend.forget_session(self.sandbox_id)
            self.backend.metrics.session_stopped()

    async def _destroy_container(self) -> None:
        """Delete the container, or prove it is already gone, or raise.

        The Engine API reports a container that no longer exists as a 404, which the engine
        treats as success. Any other failure is ambiguous, so confirm with an inspect before
        keeping a capacity reservation charged for a container that may not exist.
        """
        try:
            await self.backend.engine.remove_container(self.sandbox_id)
            return
        except Exception as api_error:
            if await self._container_is_gone():
                return
            # The CLI is a separate code path and can succeed where the API keeps failing.
            removed = await asyncio.get_running_loop().run_in_executor(
                CLEANUP_EXECUTOR,
                functools.partial(
                    force_remove_container_ids,
                    [self.sandbox_id],
                    docker_command=self.backend.lifecycle.config.docker_command,
                ),
            )
            if removed or await self._container_is_gone():
                psrl_logger.warning(
                    f"Docker sandbox {self.sandbox_id} needed a CLI force-remove after the Engine API "
                    f"failed: {api_error!r}."
                )
                return
            raise

    async def _container_is_gone(self) -> bool:
        """Report whether Docker no longer knows this container, defaulting to present."""
        try:
            return await self.backend.engine.inspect_container(self.sandbox_id) is None
        except Exception:
            return False

    async def pause(self, mode: PauseMode) -> None:
        if mode != PauseMode.FREEZE:
            raise RuntimeError("DockerBackend supports freeze, not hibernation.")
        with self.backend.metrics.measure("pause"):
            await self.backend.engine.pause_container(self.sandbox_id)

    async def resume(self) -> None:
        with self.backend.metrics.measure("resume"):
            await self.backend.engine.unpause_container(self.sandbox_id)
