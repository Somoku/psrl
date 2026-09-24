"""The two exec strategies and the observation budget they share.

The persistent shell is validated through a fake process, because the sentinel
protocol is the part that must be right: a command whose output is attributed to
the next one is worse than a command that fails.
"""

from __future__ import annotations

import asyncio

import pytest
from psrl.sandbox.backends.docker.exec import (
    _MARKER_PREFIX,
    _MARKER_SUFFIX,
    ExecBudget,
    OneShotExec,
    PersistentShellExec,
)
from psrl.sandbox.backends.docker.text import truncate_observation

pytestmark = pytest.mark.cpu_test


def _token_for(written: str) -> str:
    """Recover the sentinel token a wrapped command asks the shell to print."""
    start = written.index(_MARKER_PREFIX)
    end = written.index(_MARKER_SUFFIX, start)
    return written[start + len(_MARKER_PREFIX) : end].split("?", 1)[1]


class FakeShell:
    """A shell process that replies the way a login shell with the sentinel would."""

    def __init__(
        self,
        body: str = "ready",
        exit_code: int = 0,
        replies: int | None = None,
        first_body: str = "ready 4242",
        first_exit_code: int = 0,
    ) -> None:
        self.body = body
        # The probe runs before any command, so the first reply is the shell
        # answering readiness and every later one is a command's output.
        self.first_body = first_body
        self.first_exit_code = first_exit_code
        self.exit_code = exit_code
        self.written: list[str] = []
        self.closed = False
        # None answers every command. A count answers that many and then goes silent, which
        # is how a probe that succeeded is separated from a command that never returns.
        self._replies = replies
        self._pending = b""
        self._shell_exit_code: int | None = None

    def write(self, text: str) -> None:
        self.written.append(text)
        if self._replies is not None:
            if self._replies <= 0:
                return
            self._replies -= 1
        first = not self.written[:-1]
        status = self.first_exit_code if first else self.exit_code
        body = self.first_body if first else self.body
        sentinel = f"{_MARKER_PREFIX}{status}?{_token_for(text)}{_MARKER_SUFFIX}"
        self._pending += f"{body}\n{sentinel}\n".encode()

    def poll(self) -> int | None:
        return self._shell_exit_code

    def read_available(self, timeout_s: float) -> bytes:
        chunk, self._pending = self._pending, b""
        return chunk

    def close(self) -> None:
        self.closed = True
        self._shell_exit_code = 0


class UnparseableShell(FakeShell):
    """A shell whose later replies carry no exit status."""

    def write(self, text: str) -> None:
        self.written.append(text)
        if self._replies is not None:
            if self._replies <= 0:
                self._pending += f"noise{_MARKER_PREFIX}not-a-number?x{_MARKER_SUFFIX}\n".encode()
                return
            self._replies -= 1
        sentinel = f"{_MARKER_PREFIX}{self.exit_code}?{_token_for(text)}{_MARKER_SUFFIX}"
        self._pending += f"{self.body}\n{sentinel}\n".encode()


class FakeEngine:
    """A one-shot transport whose command results the test decides."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.result = (0, b"ok", b"", False)

    async def exec(self, container_id, command, **kwargs):
        self.calls.append(list(command))
        return self.result


async def test_one_shot_runs_the_configured_interpreter_for_each_command() -> None:
    engine = FakeEngine()
    engine.result = (0, b"hello", b"", False)
    strategy = OneShotExec(engine, "container", ("bash", "-lc"))

    result = await strategy.run("echo hello")

    assert result.stdout == "hello"
    assert engine.calls == [["bash", "-lc", "echo hello"]]


async def test_one_shot_records_activity_at_both_boundaries() -> None:
    # The in-flight half of the idle test needs a stamp from before the command, so
    # a long command is never mistaken for an idle sandbox.
    engine = FakeEngine()
    strategy = OneShotExec(engine, "container", ("bash", "-lc"))

    assert strategy.last_activity_at is None
    await strategy.run("echo hello")

    assert strategy.last_activity_at is not None


async def test_one_shot_applies_the_observation_budget_to_both_streams() -> None:
    engine = FakeEngine()
    engine.result = (1, b"x" * 400, b"y" * 400, False)
    strategy = OneShotExec(engine, "container", ("bash", "-lc"), max_observation_chars=100)

    result = await strategy.run("noisy")

    assert len(result.stdout) <= 200
    assert len(result.stderr) <= 200
    assert "characters omitted" in result.stdout


async def test_the_persistent_shell_returns_the_body_and_the_exit_status() -> None:
    shell = FakeShell(body="first line")
    strategy = PersistentShellExec(lambda: shell)
    await strategy.start(timeout_s=1)

    result = await strategy.run("echo first line")

    assert result.exit_code == 0
    assert result.stdout.strip() == "first line"
    assert shell.written


async def test_the_persistent_shell_reports_a_non_zero_exit_status() -> None:
    shell = FakeShell(body="boom", exit_code=2)
    strategy = PersistentShellExec(lambda: shell)
    await strategy.start(timeout_s=1)

    result = await strategy.run("exit 2")

    assert result.exit_code == 2


async def test_the_persistent_shell_carries_state_by_prefixing_instead_of_restarting() -> None:
    shell = FakeShell()
    strategy = PersistentShellExec(lambda: shell)
    await strategy.start(timeout_s=1)

    await strategy.run("cd /work", cwd="/work", env={"MODE": "test"})

    written = shell.written[-1]
    assert written.startswith("export MODE='test';cd '/work' || exit 200;")
    assert "cd /work" in written


async def test_the_persistent_shell_quotes_a_value_that_contains_a_quote() -> None:
    shell = FakeShell()
    strategy = PersistentShellExec(lambda: shell)
    await strategy.start(timeout_s=1)

    await strategy.run("echo hi", env={"PATCH": "it's here"})

    assert "export PATCH='it'\\''s here';" in shell.written[-1]


async def test_the_persistent_shell_starts_once_and_probes_readiness() -> None:
    shell = FakeShell()
    spawned: list[FakeShell] = []

    def spawn() -> FakeShell:
        spawned.append(shell)
        return shell

    strategy = PersistentShellExec(spawn)
    await strategy.start(timeout_s=1)
    await strategy.run("echo again")

    assert len(spawned) == 1


async def test_a_shell_that_never_answers_readiness_is_refused() -> None:
    shell = FakeShell(body="not up yet", first_body="not up yet")
    strategy = PersistentShellExec(lambda: shell)

    with pytest.raises(RuntimeError, match="did not become ready"):
        await strategy.start(timeout_s=1)

    assert shell.closed


async def test_a_shell_that_never_answers_a_command_is_replaced() -> None:
    # A desynchronized shell would otherwise hand this command's output to the next
    # one, so it is destroyed and the failure stays local to this command.
    shell = FakeShell(replies=1)
    strategy = PersistentShellExec(lambda: shell)
    await strategy.start(timeout_s=1)

    result = await strategy.run("sleep 100", budget=ExecBudget(timeout_s=5, silence_timeout_s=0.05))

    assert result.exit_code == 1
    assert "no output" in result.stderr
    assert shell.closed
    assert not strategy.started


async def test_a_command_past_its_deadline_raises_a_timeout() -> None:
    shell = FakeShell(replies=1)
    strategy = PersistentShellExec(lambda: shell)
    await strategy.start(timeout_s=1)

    with pytest.raises(TimeoutError):
        await strategy.run("sleep 100", budget=ExecBudget(timeout_s=0.05))

    assert shell.closed


async def test_a_shell_that_exits_reports_the_output_it_produced() -> None:
    shell = FakeShell(replies=1)
    strategy = PersistentShellExec(lambda: shell)
    await strategy.start(timeout_s=1)
    shell._pending = b"partial output\n"
    shell._shell_exit_code = 3

    result = await strategy.run("crash")

    assert result.exit_code == 3
    assert "partial output" in result.stdout
    assert "exited before the command completed" in result.stderr


async def test_an_unparseable_sentinel_replaces_the_shell() -> None:
    shell = UnparseableShell(replies=1)
    strategy = PersistentShellExec(lambda: shell)
    await strategy.start(timeout_s=1)

    result = await strategy.run("broken")

    assert result.exit_code == 1
    assert shell.closed


async def test_a_command_that_floods_its_output_stays_bounded() -> None:
    shell = FakeShell(body="x" * 100)
    strategy = PersistentShellExec(lambda: shell, max_observation_chars=50)
    await strategy.start(timeout_s=1)

    result = await strategy.run("noisy")

    assert len(result.stdout) <= 100
    assert "characters omitted" in result.stdout


async def test_closing_an_unstarted_shell_is_not_an_error() -> None:
    strategy = PersistentShellExec(lambda: FakeShell())

    await strategy.close()

    assert not strategy.started


async def test_a_blocking_read_still_yields_to_the_event_loop() -> None:
    # The read loop runs a blocking selector in an executor, so an unrelated task
    # must still get scheduled while a command waits.
    shell = FakeShell(replies=1)
    strategy = PersistentShellExec(lambda: shell)
    await strategy.start(timeout_s=1)
    ticks: list[int] = []

    async def ticker() -> None:
        for _ in range(3):
            ticks.append(1)
            await asyncio.sleep(0.01)

    ticker_task = asyncio.create_task(ticker())
    with pytest.raises(TimeoutError):
        await strategy.run("sleep 100", budget=ExecBudget(timeout_s=0.05))
    await ticker_task

    assert len(ticks) == 3


def test_the_observation_budget_keeps_both_ends() -> None:
    text = "A" * 500 + "MIDDLE" + "B" * 500

    truncated = truncate_observation(text, max_chars=100)

    assert truncated.startswith("A")
    assert truncated.endswith("B")
    assert "characters omitted" in truncated
    assert len(truncated) <= 200


def test_an_observation_inside_its_budget_is_untouched() -> None:
    text = "short output"

    assert truncate_observation(text, max_chars=100) == text
    assert truncate_observation(text, max_chars=0) == text


def test_an_exec_budget_refuses_a_non_positive_window() -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        ExecBudget(timeout_s=0)
    with pytest.raises(ValueError, match="silence_timeout_s"):
        ExecBudget(silence_timeout_s=-1)


async def test_the_readiness_probe_reports_the_shells_own_process_id() -> None:
    # A deadline needs that id to signal an overrunning command, so a shell that answers
    # without one is usable but cannot be stopped without discarding the container.
    shell = FakeShell()
    strategy = PersistentShellExec(lambda: shell)
    await strategy.start(timeout_s=1)

    assert strategy.remote_pid == 4242
    assert "$$" in shell.written[0], "The probe has to ask the shell for its own id."


async def test_a_shell_that_reports_no_process_id_is_still_usable() -> None:
    shell = FakeShell(first_body="ready")
    strategy = PersistentShellExec(lambda: shell)
    await strategy.start(timeout_s=1)

    assert strategy.remote_pid is None
    assert (await strategy.run("echo hi")).exit_code == 0


async def test_aborting_a_running_command_signals_the_group_and_drops_the_shell() -> None:
    shell = FakeShell()
    signalled: list[int] = []

    async def kill_group(pid: int) -> bool:
        signalled.append(pid)
        return True

    strategy = PersistentShellExec(lambda: shell, kill_group=kill_group)
    await strategy.start(timeout_s=1)

    assert await strategy.abort_running_command() is True
    assert signalled == [4242]
    assert shell.closed, "A shell holding part of a command's output cannot serve the next command."


async def test_aborting_without_a_process_id_reports_that_it_cannot() -> None:
    # Signalling an unknown group would either miss the command or reach something else
    # in the container, so the caller is told to take the safe answer instead.
    shell = FakeShell(first_body="ready")

    async def kill_group(pid: int) -> bool:
        raise AssertionError("A shell with no id must not be signalled.")

    strategy = PersistentShellExec(lambda: shell, kill_group=kill_group)
    await strategy.start(timeout_s=1)

    assert await strategy.abort_running_command() is False
    assert not shell.closed


async def test_aborting_without_a_signaller_reports_that_it_cannot() -> None:
    shell = FakeShell()
    strategy = PersistentShellExec(lambda: shell)
    await strategy.start(timeout_s=1)

    assert await strategy.abort_running_command() is False


async def test_a_signal_the_host_refuses_reports_that_it_cannot() -> None:
    shell = FakeShell()

    async def refuse(pid: int) -> bool:
        return False

    strategy = PersistentShellExec(lambda: shell, kill_group=refuse)
    await strategy.start(timeout_s=1)

    assert await strategy.abort_running_command() is False
    assert not shell.closed


async def test_one_shot_cannot_stop_a_command_it_has_started() -> None:
    # It has no process to signal, so its session has to destroy the container rather
    # than leave a process writing into the next command's output.
    strategy = OneShotExec(FakeEngine(), "container", ("bash", "-lc"))

    assert await strategy.abort_running_command() is False
