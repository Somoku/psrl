"""Command execution strategies for a Docker sandbox session.

The session contract is one `exec`, and there are two strategies behind it.

- **One shot** starts a process per command. It is the cheaper choice for a
  grader step that runs exactly one command and never needs the previous one's
  environment.
- **Persistent shell** holds one login shell open for the life of the sandbox, so
  a working directory, an activated environment, and an exported variable survive
  from one turn to the next. That is what a multi-turn harness expects, and
  without it every command restarts from the image's defaults.

Both are reached through the same `exec` signature, so a caller cannot tell which
one it is talking to. The strategy is chosen by `ExecMode` on the spec, because
the choice is a property of the workload rather than of the backend.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import select
import subprocess
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from psrl.sandbox.backends.docker.text import truncate_observation
from psrl.sandbox.core import ExecResult

psrl_logger = logging.getLogger(__file__)

# The completion sentinel. A random token per shell keeps a command that happens
# to print something similar from being read as the end of its own output.
_MARKER_PREFIX = "///PSRL-DONE:"
_MARKER_SUFFIX = ":PSRL-DONE///"
# The status is captured loosely on purpose. A shell can leave the variable unexpanded, and
# a digits-only pattern would never see that sentinel, leaving the loop waiting for a line.
#
# The trailing newline is matched because the sentinel's own `printf` writes it. Leaving
# it out appends a newline the workload never emitted to every observation.
_SENTINEL_RE = re.compile(rf"{re.escape(_MARKER_PREFIX)}(.*?)\?(\w+){re.escape(_MARKER_SUFFIX)}\r?\n?")

# How long one read waits before the loop rechecks the silence and total clocks.
_READ_SLICE_S = 0.2

# Bytes retained from one read, so a command that floods its output cannot make
# the worker allocate without bound before the character budget is applied.
_READ_CHUNK_BYTES = 256 * 1024

# The probe that proves the shell answers and reports the id a deadline has to signal.
_READINESS_MARKER = "ready"
_READINESS_PROBE = f"echo {_READINESS_MARKER} $$"
_SHELL_PID_RE = re.compile(rf"{_READINESS_MARKER}\s+(\d+)")


@dataclass(frozen=True)
class ExecBudget:
    """
    The two clocks that bound one command.

    `timeout_s` is the total deadline and `silence_timeout_s` is the longest
    stretch with no output at all. A command that prints nothing for the silence
    window is stuck, which is a different failure from one that is merely slow,
    and reporting it as the same timeout hides the cause.
    """

    timeout_s: float | None = None
    silence_timeout_s: float | None = None

    def __post_init__(self) -> None:
        for name, value in (("timeout_s", self.timeout_s), ("silence_timeout_s", self.silence_timeout_s)):
            if value is not None and value <= 0:
                raise ValueError(f"Docker exec {name} must be greater than zero when set.")


class ShellProcess(Protocol):
    """
    The transport a persistent shell needs, and nothing more.

    It is a seam so the sentinel protocol is testable without a container: a test
    drives the same write and read calls that a detached `docker exec` would.
    """

    def write(self, text: str) -> None:
        """
        Send command text to the shell's standard input.
        """

    def poll(self) -> int | None:
        """
        Return the shell's exit status once it has exited.
        """

    def close(self) -> None:
        """
        Terminate the shell and release its pipe.
        """


class DockerShellProcess:
    """
    One detached login shell inside a container, driven over the Docker CLI.

    The CLI is used rather than the Engine API because a persistent shell needs a
    bidirectional stream, and `docker exec -i` is the supported way to obtain one.
    It is also a separate code path from the Engine client, so a shell can be
    established even when the API transport is the thing that is unhealthy.
    """

    def __init__(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if not argv:
            raise ValueError("Docker shell requires a command.")
        self._process = subprocess.Popen(
            list(argv),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            env=None if env is None else {**os.environ, **dict(env)},
        )
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("Docker shell did not expose its pipes.")

    @property
    def pid(self) -> int:
        """
        Return the shell process id, which the diagnostics surface reports.
        """
        return self._process.pid

    def write(self, text: str) -> None:
        self._process.stdin.write(text.encode())
        self._process.stdin.flush()

    def poll(self) -> int | None:
        return self._process.poll()

    def read_available(self, timeout_s: float) -> bytes:
        """
        Read whatever the shell has produced within a short window.

        Blocking read used by the async loop through an executor, because the
        selector is the only portable way to wait on a pipe that may legitimately
        stay silent for a long time.
        """
        stdout = self._process.stdout
        assert stdout is not None, "Shell process stdout is not initialized."
        ready, _, _ = select.select([stdout], [], [], timeout_s)
        if not ready:
            return b""
        return os.read(stdout.fileno(), _READ_CHUNK_BYTES)

    def close(self) -> None:
        if self._process.poll() is not None:
            return
        for stream in (self._process.stdin, self._process.stdout):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)


class ExecStrategy(Protocol):
    """
    One way to run a command in a sandbox.
    """

    async def run(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        budget: ExecBudget | None = None,
    ) -> ExecResult:
        """
        Run one command and return its result.
        """

    async def abort_running_command(self) -> bool:
        """Stop a command that has not finished, without destroying the sandbox.

        Closing an exec stream does not stop the process it started, so a strategy
        that abandons a command must leave the container with a live process in it.
        Returning False says the session has no way to stop it and must destroy the
        sandbox instead, which is the safe answer when the command cannot be signalled.

        Returns:
            bool: Whether the command was stopped and the sandbox stays usable.
        """

    async def close(self) -> None:
        """
        Release whatever the strategy holds.
        """

    @property
    def last_activity_at(self) -> float | None:
        """
        Return the monotonic time of the last command boundary, start or return.
        """


class CommandTransport(Protocol):
    """
    The Engine operation a one shot strategy needs.
    """

    async def exec(
        self,
        container_id: str,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> tuple[int, bytes, bytes, bool]: ...


class OneShotExec:
    """
    Run each command as a new process inside the container.

    Stateless by construction: the working directory and environment come from the
    session's own defaults plus whatever the caller passes for this command.
    """

    def __init__(
        self,
        transport: CommandTransport,
        container_id: str,
        interpreter: Sequence[str],
        *,
        max_observation_chars: int = 0,
        resolve_command_prefix: Callable[[], Awaitable[Sequence[str]]] | None = None,
    ) -> None:
        if not interpreter:
            raise ValueError("Docker command interpreter cannot be empty.")
        self.transport = transport
        self.container_id = container_id
        self.interpreter = tuple(interpreter)
        self.max_observation_chars = max_observation_chars
        # Resolved once, lazily, before the first command. The prefix runs each command in
        # its own session so a signal the agent or one of its children sends to their
        # process group cannot reach the container's keepalive init and stop the container
        # underneath a running command. It has to be probed rather than assumed, because it
        # needs a `setsid` that supports `-w` and not every image ships one; resolving it
        # here rather than at construction keeps every creation path covered by one probe.
        self._resolve_command_prefix = resolve_command_prefix
        self._command_prefix: tuple[str, ...] | None = None
        self._last_activity_at: float | None = None

    async def _prefix(self) -> tuple[str, ...]:
        """
        Return the process-isolation prefix, probing the image at most once.
        """
        if self._command_prefix is None:
            if self._resolve_command_prefix is None:
                self._command_prefix = ()
            else:
                self._command_prefix = tuple(await self._resolve_command_prefix())
        return self._command_prefix

    @property
    def last_activity_at(self) -> float | None:
        return self._last_activity_at

    async def run(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        budget: ExecBudget | None = None,
    ) -> ExecResult:
        self._last_activity_at = time.monotonic()
        timeout_s = None if budget is None else budget.timeout_s
        prefix = await self._prefix()
        try:
            exit_code, stdout, stderr, truncated = await self.transport.exec(
                self.container_id,
                [*prefix, *self.interpreter, command],
                cwd=cwd,
                env=env,
                timeout_s=timeout_s,
            )
        finally:
            self._last_activity_at = time.monotonic()
        return ExecResult(
            exit_code=exit_code,
            stdout=truncate_observation(stdout.decode(errors="replace"), self.max_observation_chars),
            stderr=truncate_observation(stderr.decode(errors="replace"), self.max_observation_chars),
            truncated=truncated,
        )

    async def abort_running_command(self) -> bool:
        """
        Report that this strategy cannot stop a command it has already started.
        """
        return False

    async def close(self) -> None:
        return None


class PersistentShellExec:
    """Hold one login shell open and drive it with a completion sentinel.

    A command is written to the shell's standard input followed by a sentinel
    echo that carries the exit status, and the read loop stops at the sentinel.
    That is what preserves shell state while still reporting a per command exit
    code, which a bare attached shell cannot do.

    A shell that goes silent past its silence window is assumed desynchronized:
    the process is destroyed and the next command starts a fresh shell. Losing
    shell state is worse than the alternative, which is attributing one command's
    output to the next one.

    A command that overruns its deadline is signalled instead of forcing the whole
    sandbox to be destroyed. The shell reports its own process id when it starts, and
    the sandbox runs it as a session leader, so signalling that group stops the
    command and its children while the filesystem survives.
    """

    def __init__(
        self,
        spawn: Any,
        *,
        max_observation_chars: int = 0,
        kill_group: Any = None,
    ) -> None:
        self._spawn = spawn
        self.max_observation_chars = max_observation_chars
        self._kill_group = kill_group
        self._process: ShellProcess | None = None
        self._start_lock = asyncio.Lock()
        self._last_activity_at: float | None = None
        self._token = os.urandom(6).hex()
        self._remote_pid: int | None = None

    @property
    def remote_pid(self) -> int | None:
        """
        Return the shell's own process id inside the sandbox, once it is known.
        """
        return self._remote_pid

    @property
    def last_activity_at(self) -> float | None:
        return self._last_activity_at

    @property
    def started(self) -> bool:
        """
        Return whether a live shell is currently held.
        """
        return self._process is not None and self._process.poll() is None

    async def start(self, *, timeout_s: float = 300.0) -> None:
        """Start the shell and confirm it answers, so the first command is not the probe.

        The probe also reports the shell's own process id, which is what a deadline
        needs to signal a command later. A probe that only echoed readiness would leave
        an overrunning command with nothing able to stop it.
        """
        async with self._start_lock:
            if self.started:
                return
            self._process = self._spawn()
            self._remote_pid = None
            result = await self._run_inner(
                _READINESS_PROBE,
                cwd=None,
                env=None,
                budget=ExecBudget(timeout_s=timeout_s),
            )
            output = result.stdout + result.stderr
            if result.exit_code != 0 or _READINESS_MARKER not in output:
                await self.close()
                raise RuntimeError(
                    f"Docker sandbox shell did not become ready within {timeout_s:g}s: {output.strip()[:200]!r}."
                )
            self._remote_pid = _parse_shell_pid(output)

    async def abort_running_command(self) -> bool:
        """Signal the running command's process group and drop the shell.

        Returns False when the shell never reported an id, because signalling the wrong
        process group would either miss the command or reach something else in the
        container. The caller then has to destroy the sandbox, which is the answer that
        cannot be wrong.
        """
        pid = self._remote_pid
        if pid is None or self._kill_group is None:
            return False
        try:
            signalled = await self._kill_group(pid)
        except Exception:
            psrl_logger.warning(f"Could not signal the Docker sandbox shell group {pid}.", exc_info=True)
            return False
        if not signalled:
            return False
        # Whatever part of the command's output the shell still holds belongs to a
        # command that did not finish, so the next one must not read it as its own.
        await self.close()
        return True

    def _wrap(self, command: str, *, cwd: str | None, env: Mapping[str, str] | None) -> str:
        """
        Build the text one command needs, including its sentinel.

        Per-command overrides are applied as a prefix rather than by restarting the
        shell, which is the whole point of keeping one open.
        """
        prefix: list[str] = []
        for key, value in (env or {}).items():
            prefix.append(f"export {key}={_quote(value)};")
        if cwd:
            prefix.append(f"cd {_quote(cwd)} || exit 200;")
        body = command if command.endswith("\n") else command + "\n"
        sentinel = (
            f"__psrl_rc=$?; sleep 0.01; printf '{_MARKER_PREFIX}%s?{self._token}{_MARKER_SUFFIX}\\n' \"$__psrl_rc\"\n"
        )
        return "".join(prefix) + body + sentinel

    async def run(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        budget: ExecBudget | None = None,
    ) -> ExecResult:
        # Starting implicitly would consume the shell's probe reply as this command's output,
        # so the session starts it first. An exited shell still buffers, so check attached, not live.
        if self._process is None:
            raise RuntimeError("Docker sandbox shell is not started.")
        self._last_activity_at = time.monotonic()
        try:
            return await self._run_inner(command, cwd=cwd, env=env, budget=budget)
        finally:
            self._last_activity_at = time.monotonic()

    async def _run_inner(
        self,
        command: str,
        *,
        cwd: str | None,
        env: Mapping[str, str] | None,
        budget: ExecBudget | None,
    ) -> ExecResult:
        process = self._process
        if process is None:
            raise RuntimeError("Docker sandbox shell is not started.")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, process.write, self._wrap(command, cwd=cwd, env=env))
        timeout_s = None if budget is None else budget.timeout_s
        silence_s = None if budget is None else budget.silence_timeout_s
        started_at = loop.time()
        last_output_at = started_at
        buffer = ""
        truncated = False
        while True:
            now = loop.time()
            if timeout_s is not None and now - started_at >= timeout_s:
                return await self._fail(process, f"exceeded its {timeout_s:g}s deadline", timeout=True)
            if silence_s is not None and now - last_output_at >= silence_s:
                return await self._fail(process, f"produced no output for {silence_s:g}s", silence=True)
            windows = (
                _remaining(now, started_at, timeout_s),
                _remaining(now, last_output_at, silence_s),
            )
            slice_s = max(0.01, min(_READ_SLICE_S, *windows))
            try:
                chunk = await asyncio.wait_for(
                    loop.run_in_executor(None, process.read_available, slice_s),
                    timeout=slice_s + 1.0,
                )
            except asyncio.TimeoutError:
                chunk = b""
            if chunk:
                buffer += chunk.decode(errors="replace")
                last_output_at = loop.time()
                parsed = self._parse(buffer)
                if parsed is not None:
                    body, exit_code = parsed
                    if exit_code is None:
                        return await self._fail(process, "reported an unparseable exit status")
                    return ExecResult(
                        exit_code=exit_code,
                        stdout=truncate_observation(body, self.max_observation_chars),
                        stderr="",
                        truncated=truncated,
                    )
                if len(buffer) > _READ_CHUNK_BYTES * 8:
                    buffer = buffer[-_READ_CHUNK_BYTES * 4 :]
                    truncated = True
                continue
            if process.poll() is not None:
                # The shell itself exited, so no sentinel is coming and whatever state
                # it held is gone. Report the buffered output rather than a bare hang.
                parsed = self._parse(buffer)
                body, exit_code = parsed if parsed is not None else (buffer, process.poll())
                self._process = None
                return ExecResult(
                    exit_code=int(exit_code if exit_code is not None else 1),
                    stdout=truncate_observation(body, self.max_observation_chars),
                    stderr="Docker sandbox shell exited before the command completed.",
                    truncated=truncated,
                )

    def _parse(self, buffer: str) -> tuple[str, int | None] | None:
        """
        Split an output buffer at its sentinel, tolerating an unusable status.

        A sentinel that arrived is the end of this command's output whatever its
        payload says, so a status that is not an integer is reported rather than
        waited on.
        """
        match = _SENTINEL_RE.search(buffer)
        if match is None:
            return None
        body = buffer[: match.start()] + buffer[match.end() :]
        try:
            return body, int(match.group(1))
        except ValueError:
            psrl_logger.warning(f"Sandbox shell returned an unparseable exit status {match.group(1)!r}.")
            return body, None

    async def _fail(
        self,
        process: ShellProcess,
        reason: str,
        *,
        timeout: bool = False,
        silence: bool = False,
    ) -> ExecResult:
        """Destroy a desynchronized shell and report a self-contained failure.

        A command that never completed leaves the shell holding part of its
        output, so the next command would otherwise read that output as its own.
        The sandbox stays usable and the next command starts a fresh shell.
        """
        label = "silence timeout" if silence else "timeout" if timeout else "shell failure"
        psrl_logger.warning(f"Docker sandbox shell was replaced after a {label}: {reason}.")
        await self.close()
        message = f"Docker sandbox command {reason}. The sandbox shell was restarted and lost its state."
        if timeout:
            raise TimeoutError(message)
        return ExecResult(exit_code=1, stdout="", stderr=message)

    async def close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        await asyncio.get_running_loop().run_in_executor(None, process.close)


def _parse_shell_pid(output: str) -> int | None:
    """Return the shell's own process id from its readiness probe output.

    A shell that answered but reported no id is usable, it just cannot be signalled,
    so this reports None rather than failing the shell.
    """
    match = _SHELL_PID_RE.search(output)
    if match is None:
        psrl_logger.warning("Docker sandbox shell reported no process id, so a command deadline cannot signal it.")
        return None
    return int(match.group(1))


def _remaining(now: float, started_at: float, window_s: float | None) -> float:
    """
    Return how much of one window is left, or the read slice when unbounded.
    """
    if window_s is None:
        return _READ_SLICE_S
    return max(0.0, window_s - (now - started_at))


def _quote(value: str) -> str:
    """
    Quote one value for a POSIX shell.
    """
    return "'" + value.replace("'", "'\\''") + "'"
