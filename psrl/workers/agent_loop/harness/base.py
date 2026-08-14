"""Task-scoped lifecycle contract for sandboxed coding harnesses."""

import asyncio
import contextlib
import shlex
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from omegaconf import DictConfig, OmegaConf

from psrl.sandbox import ExecResult, SandboxSession


@dataclass(frozen=True)
class HarnessInstallConfig:
    """Optional in-sandbox installation policy for one harness executable."""

    check_command: str | None = None
    command: str | None = None
    timeout_s: float = 300.0

    def __post_init__(self) -> None:
        if self.timeout_s <= 0:
            raise ValueError("Harness install timeout_s must be greater than zero.")


@dataclass(frozen=True)
class HarnessConfig:
    """Backend-neutral configuration shared by all coding harnesses."""

    kind: str
    executable: str
    model: str | None = None
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    home_dir: str = "/root"
    time_budget_s: float = 7200.0
    output_tail_chars: int = 16_384
    callback_base_url: str | None = None
    install: HarnessInstallConfig = field(default_factory=HarnessInstallConfig)

    def __post_init__(self) -> None:
        if not self.kind or not self.executable:
            raise ValueError("Harness kind and executable cannot be empty.")
        if self.time_budget_s <= 0 or self.output_tail_chars <= 0:
            raise ValueError("Harness time_budget_s and output_tail_chars must be greater than zero.")

    @classmethod
    def from_value(cls, value: "HarnessConfig" | DictConfig | Mapping[str, Any]) -> "HarnessConfig":
        """Normalize a Hydra mapping into an immutable harness configuration."""
        if isinstance(value, cls):
            return value
        if isinstance(value, DictConfig):
            raw = OmegaConf.to_container(value, resolve=True)
        else:
            raw = dict(value)
        if not isinstance(raw, Mapping):
            raise TypeError("Harness configuration must be a mapping.")
        normalized = dict(raw)
        normalized["args"] = tuple(str(item) for item in normalized.get("args", ()))
        normalized["env"] = {str(key): str(item) for key, item in dict(normalized.get("env", {})).items()}
        install = normalized.get("install", {})
        normalized["install"] = (
            install if isinstance(install, HarnessInstallConfig) else HarnessInstallConfig(**dict(install or {}))
        )
        return cls(**normalized)


@dataclass(frozen=True)
class HarnessRuntime:
    """Per-task values bound after the TITO session and sandbox exist."""

    session_id: str
    session_root_url: str
    workdir: str
    model: str


@dataclass(frozen=True)
class HarnessResult:
    """Small process result retained outside the disposable sandbox."""

    exit_code: int
    stderr_tail: str = ""


class Harness(ABC):
    """One task-owned coding harness, including preparation, run, and abort.

    The object deliberately combines adapter and active-run state. It is never
    shared between tasks, so the sandbox lease remains the single lifecycle
    owner while an optional abort cancels the currently active CLI request.
    """

    def __init__(self, config: HarnessConfig, sandbox: SandboxSession) -> None:
        self.config = config
        self.sandbox = sandbox
        self._active_exec: asyncio.Task[ExecResult] | None = None
        self._log_dir = str(PurePosixPath(config.home_dir) / ".psrl-harness")

    async def prepare(self, runtime: HarnessRuntime) -> None:
        """Verify or install the CLI, then write harness-specific configuration."""
        check_command = self.config.install.check_command or f"command -v {shlex.quote(self.config.executable)}"
        check = await self.sandbox.exec(check_command, timeout_s=30)
        if check.exit_code != 0:
            install_command = self.config.install.command
            if not install_command:
                raise RuntimeError(
                    f"Harness executable {self.config.executable!r} is unavailable and no install command "
                    "is configured."
                )
            installed = await self.sandbox.exec(install_command, timeout_s=self.config.install.timeout_s)
            if installed.exit_code != 0:
                raise RuntimeError(
                    f"Harness installation failed with exit code {installed.exit_code}: {installed.stderr.strip()}"
                )
            check = await self.sandbox.exec(check_command, timeout_s=30)
            if check.exit_code != 0:
                raise RuntimeError(f"Harness executable {self.config.executable!r} is unavailable after installation.")
        mkdir = await self.sandbox.exec(f"mkdir -p {shlex.quote(self._log_dir)}", timeout_s=30)
        if mkdir.exit_code != 0:
            raise RuntimeError(f"Could not create harness state directory: {mkdir.stderr.strip()}")
        await self._prepare(runtime)

    @abstractmethod
    async def _prepare(self, runtime: HarnessRuntime) -> None:
        """Write adapter-specific files after the executable check succeeds."""

    @abstractmethod
    def build_command(self, prompt: str, runtime: HarnessRuntime) -> Sequence[str]:
        """Build the CLI argv without embedding environment secrets."""

    @abstractmethod
    def build_env(self, runtime: HarnessRuntime) -> Mapping[str, str]:
        """Build environment variables for one task-scoped inference endpoint."""

    async def run(self, prompt: str, runtime: HarnessRuntime) -> HarnessResult:
        """Run the harness CLI and retain only bounded diagnostic tails."""
        stderr_path = str(PurePosixPath(self._log_dir) / "stderr.log")
        argv = self.build_command(prompt, runtime)
        command = f"{shlex.join([str(item) for item in argv])} > /dev/null 2> {shlex.quote(stderr_path)}"
        task = asyncio.create_task(
            self.sandbox.exec(
                command,
                cwd=runtime.workdir,
                env=self.build_env(runtime),
                timeout_s=self.config.time_budget_s,
            )
        )
        self._active_exec = task
        try:
            result = await task
        finally:
            if self._active_exec is task:
                self._active_exec = None
        if result.exit_code == 0:
            return HarnessResult(exit_code=0)
        return HarnessResult(
            exit_code=result.exit_code,
            stderr_tail=await self._read_tail(stderr_path),
        )

    async def abort(self) -> None:
        """Cancel the active exec; sandbox lease release supplies the hard stop."""
        task = self._active_exec
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    async def _read_tail(self, path: str) -> str:
        result = await self.sandbox.exec(
            f"tail -c {self.config.output_tail_chars} {shlex.quote(path)}",
            timeout_s=30,
        )
        return result.stdout if result.exit_code == 0 else ""
