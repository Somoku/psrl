"""Docker session execution, diagnostics, and confirmed destruction."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import aiohttp

from psrl.sandbox.async_utils import acquire_nowait, complete_cleanup
from psrl.sandbox.backends.docker.exec import ExecBudget, ExecStrategy, OneShotExec
from psrl.sandbox.backends.docker.policy import DockerPolicyProfile
from psrl.sandbox.core import (
    ExecResult,
    PauseMode,
    ResourceUsage,
    SandboxBusyError,
    SandboxCapabilities,
    SandboxCommandTimeout,
    SandboxDiagnostics,
    SandboxExitReason,
    SandboxFeature,
    SandboxOomError,
    SandboxRef,
    SandboxSession,
    SandboxSpec,
    SandboxStatus,
    SandboxTransportError,
    SnapshotKind,
    SnapshotRef,
)
from psrl.sandbox.snapshot_store import SnapshotPublishError

if TYPE_CHECKING:
    from psrl.sandbox.backends.docker import DockerBackend

psrl_logger = logging.getLogger(__file__)

# Backoff for a lifetime cleanup the daemon keeps refusing. Bounded, because a wedged daemon
# would otherwise log once per second per session for the rest of the run.
_LIFETIME_RETRY_ATTEMPTS = 6
_LIFETIME_RETRY_BASE_S = 1.0
_LIFETIME_RETRY_MAX_S = 30.0

# Diagnosed stop reasons, so a post mortem reads the same vocabulary whichever
# code path noticed the stop.
_STOP_REASONS = {
    "oom_killed": SandboxExitReason.OOM,
    "removed": SandboxExitReason.UNKNOWN,
    "exited": SandboxExitReason.FAILED,
}

# Bytes of container log retained for a diagnosis.
_DIAGNOSTIC_LOG_BYTES = 16 * 1024


def _remaining_deadline(started_at: float, timeout_s: float | None) -> float | None:
    """
    Return the part of one deadline the command itself still has.
    """
    if timeout_s is None:
        return None
    return max(1e-3, timeout_s - (time.monotonic() - started_at))


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
        exec_strategy: ExecStrategy | None = None,
    ) -> None:
        self.backend = backend
        self.sandbox_id = sandbox_id
        self._spec = spec
        self._policy = policy or DockerPolicyProfile()
        self._command_count = 0
        self._terminate_lock = asyncio.Lock()
        self._terminated = False
        self._exec_lock = asyncio.Lock()
        self._exec_strategy = exec_strategy or OneShotExec(
            backend.engine,
            sandbox_id,
            backend.command_interpreter,
            max_observation_chars=backend.max_observation_chars,
        )
        self._exit_reason = SandboxExitReason.UNKNOWN
        self._busy = False
        self._created_at = time.monotonic()
        self._timeout_task = (
            asyncio.create_task(self._terminate_at_deadline(lifetime_timeout_s))
            if lifetime_timeout_s is not None
            else None
        )

    @property
    def exec_strategy(self) -> ExecStrategy:
        """
        Return the strategy that runs this session's commands.
        """
        return self._exec_strategy

    @property
    def busy(self) -> bool:
        """
        Return whether a command is executing right now.
        """
        return self._busy

    @property
    def last_activity_at(self) -> float | None:
        """
        Return the time of the last command boundary, or the creation time.

        A session that has not run a command yet reports its creation, so an idle
        policy can still reclaim one that was created and then abandoned.
        """
        return self._exec_strategy.last_activity_at or self._created_at

    @property
    def exit_reason(self) -> SandboxExitReason:
        """
        Return why this session stopped existing.
        """
        return self._exit_reason

    @property
    def callback_host_alias(self) -> str | None:
        """
        Return the host gateway alias this sandbox reaches the worker by.
        """
        return self._policy.host_gateway_alias

    @property
    def protected_env_names(self) -> frozenset[str]:
        """
        Return the injected credential names, which a snapshot must not capture.
        """
        if self._spec is None:
            return frozenset()
        return frozenset(credential.target_env for credential in self._spec.credentials)

    async def _terminate_at_deadline(self, timeout_s: float, attempt: int = 1) -> None:
        """
        Enforce Docker lifetime locally because Engine has no native TTL.
        """
        try:
            await asyncio.sleep(timeout_s)
            self._timeout_task = None
            self._exit_reason = SandboxExitReason.REAPED_LIFETIME
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
        """Capture a filesystem snapshot by committing the container's writable layer.

        A node-local commit is the cheap case and stays the default. When the caller
        asked for a resume on another node, the commit is published to the store and
        the returned reference is the registry digest, so the restore path on any
        node is the ordinary create that already pulls a missing image.
        """
        if kind != SnapshotKind.FILESYSTEM:
            raise NotImplementedError(f"DockerSession only supports FILESYSTEM snapshots, got {kind.value}.")
        repo = f"psrl/snapshot/{self.sandbox_id}"
        tag = uuid.uuid4().hex
        image_id = await self.backend.engine.commit_container(self.sandbox_id, repo, tag)
        image_tag = f"{repo}:{tag}"
        # A fresh commit is the most recent use of the local snapshot cache.
        self.backend.note_snapshot_use(image_tag)
        metadata: dict[str, Any] = {"psrl.docker.image": image_tag, "psrl.docker.image_id": image_id}
        reference = image_tag
        if self._publishes_snapshot():
            reference = await self.backend.publish_snapshot(
                image_tag,
                run_id=self._run_id(),
                workflow_id=self._workflow_scope(),
            )
            metadata["psrl.docker.image"] = reference
            metadata["psrl.snapshot.published"] = True
            self.backend.metrics.count("snapshot_published")
        return SnapshotRef(
            backend=self.backend.name,
            snapshot_id=reference,
            kind=SnapshotKind.FILESYSTEM,
            metadata=metadata,
            resume_level=self.capabilities.resume_level,
        )

    def _publishes_snapshot(self) -> bool:
        """Return whether this checkpoint has to be restorable from another node.

        Publishing is on demand, so the common node-local checkpoint pays no push
        cost. A caller that requires a resume elsewhere and cannot have one is
        refused here, because returning a node-local reference would look like
        success and only fail on the node that never had the image.
        """
        requires = self._spec is not None and SandboxFeature.RESUME_ANYWHERE in self._spec.required_features
        store = self.backend.snapshot_store
        if store is None:
            if requires:
                raise SnapshotPublishError(
                    "This Docker backend has no snapshot store, so a checkpoint cannot be made restorable "
                    "elsewhere. Configure sandbox.snapshot_store, or drop the RESUME_ANYWHERE requirement."
                )
            return False
        return store.config.publish == "always" or requires

    def _run_id(self) -> str:
        """
        Return the run this sandbox belongs to, which namespaces its snapshots.
        """
        if self._spec is None:
            return "unscoped"
        return str(self._spec.metadata.get("psrl.run_id") or self._spec.metadata.get("run_id") or "run")

    def _workflow_scope(self) -> str:
        """
        Return the workflow this snapshot belongs to, or a name for a scoped one.
        """
        if self._spec is None or not self._spec.workflow_id:
            return "unscoped"
        return str(self._spec.workflow_id)

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
        silence_timeout_s: float | None = None,
    ) -> ExecResult:
        """Run one command through the session's exec strategy.

        Args:
            command (str): Command line interpreted by the strategy's shell.
            cwd (str | None): Working directory for this command only.
            env (Mapping[str, str] | None): Variables added for this command only.
            timeout_s (float | None): Total deadline for the command.
            silence_timeout_s (float | None): Longest stretch with no output at all.
                Reported separately from the total deadline, because "stuck" and
                "slow" need different answers.

        Raises:
            SandboxCommandTimeout: When the deadline expires. Whether the sandbox
                survived is on the exception, because a one-shot exec cannot be
                signalled and has to take its container with it while a persistent
                shell can be stopped and replaced.
            SandboxOomError: When the container was OOM-killed while the command ran.
        """
        if timeout_s is not None and timeout_s <= 0:
            raise ValueError("Docker command timeout must be positive.")
        if silence_timeout_s is None:
            silence_timeout_s = self.backend.silence_timeout_s
        budget = ExecBudget(timeout_s=timeout_s, silence_timeout_s=silence_timeout_s)
        # A bounded call must not be left queued, and keeping the wait separate is what
        # makes a caller's cancellation the only thing that can cancel a command.
        started_at = time.monotonic()
        try:
            await asyncio.wait_for(self._exec_lock.acquire(), timeout=timeout_s)
        except asyncio.TimeoutError as exc:
            raise SandboxCommandTimeout(
                f"Docker sandbox {self.sandbox_id} still had a command running after {timeout_s!r}s, so this "
                "command was never started.",
                sandbox_preserved=True,
            ) from exc
        try:
            return await self._exec_bounded(
                command,
                cwd=cwd,
                env=env,
                budget=replace(budget, timeout_s=_remaining_deadline(started_at, timeout_s)),
            )
        finally:
            self._exec_lock.release()

    async def _exec_bounded(
        self,
        command: str,
        *,
        cwd: str | None,
        env: Mapping[str, str] | None,
        budget: ExecBudget,
    ) -> ExecResult:
        """
        Run one command under the exec lock its caller already holds.
        """
        if self._terminated:
            raise RuntimeError("Docker sandbox is terminated.")
        self._command_count += 1
        self._busy = True
        try:
            return await self._exec_locked(command, cwd=cwd, env=env, budget=budget)
        finally:
            self._busy = False

    async def _exec_locked(
        self,
        command: str,
        *,
        cwd: str | None,
        env: Mapping[str, str] | None,
        budget: ExecBudget,
    ) -> ExecResult:
        exec_task = asyncio.create_task(self._exec_strategy.run(command, cwd=cwd, env=env, budget=budget))
        watcher = asyncio.create_task(self._watch_container())
        try:
            result = await self._collect_command(exec_task, watcher, budget.timeout_s)
        finally:
            exec_task.cancel()
            watcher.cancel()
            await complete_cleanup(asyncio.gather(exec_task, watcher, return_exceptions=True))
        return self._annotate_truncation(result, command)

    def _annotate_truncation(self, result: ExecResult, command: str) -> ExecResult:
        """
        Tell the caller what was cut, so a bounded answer is not read as a full one.
        """
        if not result.truncated:
            return result
        psrl_logger.warning(
            f"Docker sandbox {self.sandbox_id} produced more output than the "
            f"{self.backend.max_exec_output_bytes}-byte diagnostic budget while running "
            f"{_describe_command(command)}. The command completed and its output is truncated."
        )
        marker = (
            f"\n[psrl: output truncated at {self.backend.max_exec_output_bytes} bytes. Write the "
            "result to a file and read that file when the full output matters]\n"
        )
        if result.stdout:
            return replace(result, stdout=result.stdout + marker)
        return replace(result, stderr=result.stderr + marker)

    async def _collect_command(
        self,
        exec_task: asyncio.Task,
        watcher: asyncio.Task,
        timeout_s: float | None,
    ) -> ExecResult:
        """Await a command, and settle what to do when it does not finish.

        A container that stopped underneath the command outranks every symptom,
        because it explains them. A command that merely overran its deadline is
        different: the strategy is asked to stop it first, and only a strategy that
        cannot is allowed to take the sandbox down with it.
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
            return await complete_cleanup(self._command_overran(exec_task, watcher, timeout_s))
        try:
            with self.backend.metrics.measure("exec"):
                return exec_task.result()
        except asyncio.CancelledError:
            # The command stream was aborted, which does not stop the process it started.
            await complete_cleanup(self._abort_command(exec_task, watcher))
            raise
        except (TimeoutError, asyncio.TimeoutError) as exc:
            return await complete_cleanup(self._command_overran(exec_task, watcher, timeout_s, cause=exc))
        except (aiohttp.ClientError, SandboxTransportError) as exc:
            reason = await complete_cleanup(self._abort_command(exec_task, watcher, diagnose=True))
            if reason is not None:
                raise self._stopped_error(reason) from exc
            raise

    async def _command_overran(
        self,
        exec_task: asyncio.Task,
        watcher: asyncio.Task,
        timeout_s: float | None,
        *,
        cause: BaseException | None = None,
    ) -> None:
        """Stop an overrunning command, keeping the sandbox when that is possible.

        Losing an episode's filesystem and shell state to one slow turn is worse than
        the orphan process this guards against, so the strategy is asked to signal the
        command first. A strategy that cannot has no safe answer other than destroying
        the container, because a process left running would write into the next
        command's output.

        Raises:
            SandboxCommandTimeout: Always, carrying whether the sandbox survived.
        """
        preserved = await complete_cleanup(self._exec_strategy.abort_running_command())
        if not preserved:
            reason = await complete_cleanup(self._abort_command(exec_task, watcher, diagnose=True))
            if reason is not None:
                raise self._stopped_error(reason)
        else:
            exec_task.cancel()
            watcher.cancel()
            await complete_cleanup(asyncio.gather(exec_task, watcher, return_exceptions=True))
        message = (
            f"Docker sandbox {self.sandbox_id} stopped a command that overran its {timeout_s!r}s deadline."
            if preserved
            else f"Docker sandbox {self.sandbox_id} was destroyed by a command that overran its "
            f"{timeout_s!r}s deadline, because this exec mode cannot stop a running command."
        )
        raise SandboxCommandTimeout(message, sandbox_preserved=preserved) from cause

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
        self._exit_reason = _STOP_REASONS.get(reason, SandboxExitReason.UNKNOWN)
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

    async def diagnostics(self) -> SandboxDiagnostics:
        """Return read-only evidence about this sandbox, including a log tail.

        Once a sandbox can run on another node, the operator no longer knows which
        container to inspect, and it may already be gone. This is what the node
        that hosted it can still answer.
        """
        inspection: dict = {}
        try:
            inspection = dict(await self.backend.engine.inspect_container(self.sandbox_id) or {})
        except (aiohttp.ClientError, SandboxTransportError, TimeoutError, asyncio.TimeoutError):
            inspection = {}
        return SandboxDiagnostics(
            ref=self.ref,
            status=SandboxStatus.TERMINATED if self._terminated else await self.status(),
            exit_reason=self._exit_reason,
            log_tail=self._log_tail(inspection),
            inspect=inspection,
            usage=await self.stats() if not self._terminated else ResourceUsage(),
        )

    def _log_tail(self, inspection: Mapping) -> str:
        """
        Return the bounded tail of what the container logged.
        """
        state = inspection.get("State") or {}
        error = state.get("Error")
        if error:
            return str(error)[-_DIAGNOSTIC_LOG_BYTES:]
        return ""

    async def terminate(self) -> None:
        await complete_cleanup(self._terminate())

    async def _terminate(self) -> None:
        async with self._terminate_lock:
            if self._terminated:
                return
            await complete_cleanup(self._exec_strategy.close())
            await complete_cleanup(self.backend.release_egress(self.sandbox_id))
            with self.backend.metrics.measure("terminate"):
                await self._destroy_container()
            if self._exit_reason is SandboxExitReason.UNKNOWN:
                self._exit_reason = SandboxExitReason.RELEASED
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
            removed = await self.backend.force_remove(self.sandbox_id)
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
        # The exec lock is the in-flight signal, taken without waiting. A command that started
        # between the idle check and here means not idle, and freezing stops it halfway.
        if not await acquire_nowait(self._exec_lock):
            raise SandboxBusyError(f"Docker sandbox {self.sandbox_id!r} has a command in flight, so it is not idle.")
        try:
            with self.backend.metrics.measure("pause"):
                await self.backend.engine.pause_container(self.sandbox_id)
        finally:
            self._exec_lock.release()

    async def resume(self) -> None:
        with self.backend.metrics.measure("resume"):
            await self.backend.engine.unpause_container(self.sandbox_id)
