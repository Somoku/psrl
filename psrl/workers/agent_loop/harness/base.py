"""Task-scoped lifecycle contract for sandboxed coding harnesses."""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from omegaconf import DictConfig, OmegaConf

from psrl.sandbox import ExecResult, SandboxSession


@dataclass(frozen=True)
class HarnessInstallConfig:
    """Optional in-sandbox installation policy for one harness executable."""

    strategy: str = "command"
    check_command: str | None = None
    command: str | None = None
    timeout_s: float = 300.0
    node_tarball_env: str | None = None
    cli_tarball_env: str | None = None
    node_tarball_path: str = "/tmp/node22.tarball"
    cli_tarball_path: str = "/tmp/harness-cli.tgz"
    node_install_dir: str = "/opt/node22"
    npm_prefix: str = "/usr/local"
    retries: int = 3
    retry_backoff_s: float = 2.0

    def __post_init__(self) -> None:
        if self.timeout_s <= 0:
            raise ValueError("Harness install timeout_s must be greater than zero.")
        if self.strategy not in ("command", "npm_tarball"):
            raise ValueError(f"Unsupported harness install strategy {self.strategy!r}.")
        if self.retries <= 0:
            raise ValueError("Harness install retries must be greater than zero.")
        if self.retry_backoff_s < 0:
            raise ValueError("Harness install retry_backoff_s cannot be negative.")


@dataclass(frozen=True)
class HarnessCompactionConfig:
    """Context-compaction policy shared by external coding harnesses.

    ``trigger_tokens`` defaults to the rollout prompt plus response budget
    (minus ``safety_tokens``). ``context_window_tokens`` is the actual CLI
    context capacity and may be larger when a harness adds system/tool text.
    Set ``safety_tokens=0`` for the exact rollout-budget threshold.
    """

    enabled: bool = True
    context_window_tokens: int | None = None
    trigger_tokens: int | None = None
    safety_tokens: int = 512

    def resolve(self, default_context_window_tokens: int) -> tuple[int, int] | None:
        """Resolve CLI capacity and the rollout-budget compaction trigger."""
        if not self.enabled:
            return None
        rollout_budget = int(default_context_window_tokens)
        if rollout_budget <= 0:
            raise ValueError("Default harness rollout budget must be greater than zero.")
        if self.safety_tokens < 0:
            raise ValueError("Harness compaction safety_tokens cannot be negative.")

        window = int(self.context_window_tokens or rollout_budget)
        if window <= 0:
            raise ValueError("Harness compaction context_window_tokens must be greater than zero.")
        trigger = int(
            self.trigger_tokens if self.trigger_tokens is not None else (rollout_budget - self.safety_tokens)
        )
        if trigger <= 0:
            raise ValueError(
                "Harness compaction trigger_tokens must be greater than zero; "
                "reduce safety_tokens or increase the context window."
            )
        if trigger > window:
            raise ValueError(
                f"Harness compaction trigger_tokens ({trigger}) cannot exceed context_window_tokens ({window})."
            )
        return window, trigger


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
    permission_mode: str | None = None
    allowed_tools: tuple[str, ...] = ()
    system_prompt: str | None = None
    tools: str | None = None
    compaction: HarnessCompactionConfig = field(default_factory=HarnessCompactionConfig)
    install: HarnessInstallConfig = field(default_factory=HarnessInstallConfig)

    def __post_init__(self) -> None:
        if not self.kind or not self.executable:
            raise ValueError("Harness kind and executable cannot be empty.")
        if self.time_budget_s <= 0 or self.output_tail_chars <= 0:
            raise ValueError("Harness time_budget_s and output_tail_chars must be greater than zero.")

    @classmethod
    def from_value(cls, value: HarnessConfig | DictConfig | Mapping[str, Any]) -> HarnessConfig:
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
        normalized["allowed_tools"] = tuple(str(item) for item in normalized.get("allowed_tools", ()))
        normalized["env"] = {str(key): str(item) for key, item in dict(normalized.get("env", {})).items()}
        compaction = normalized.get("compaction", {})
        normalized["compaction"] = (
            compaction
            if isinstance(compaction, HarnessCompactionConfig)
            else HarnessCompactionConfig(**dict(compaction or {}))
        )
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
    context_window_tokens: int | None = None
    compaction_token_limit: int | None = None
    max_turns: int | None = None
    max_output_tokens: int | None = None


@dataclass(frozen=True)
class HarnessResult:
    """Small process result retained outside the disposable sandbox."""

    exit_code: int
    stdout_tail: str = ""
    stderr_tail: str = ""
    command_name: str = ""
    workdir: str = ""
    stdout_path: str = ""
    stderr_path: str = ""

    def diagnostic_text(self) -> str:
        """Return bounded, actionable process diagnostics for rollout failures."""
        return "\n".join(
            (
                f"exit_code={self.exit_code}",
                f"command={self.command_name or '<unknown>'}",
                f"workdir={self.workdir or '<unknown>'}",
                f"stdout_path={self.stdout_path or '<unknown>'}",
                f"stderr_path={self.stderr_path or '<unknown>'}",
                "stdout_tail:",
                self.stdout_tail or "<empty>",
                "stderr_tail:",
                self.stderr_tail or "<empty>",
            )
        )


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
            installed = await self._install_cli(check_command)
            if installed.exit_code != 0:
                raise RuntimeError(
                    f"Harness installation failed with exit code {installed.exit_code}: "
                    f"stdout={installed.stdout[-self.config.output_tail_chars :]!r}, "
                    f"stderr={installed.stderr[-self.config.output_tail_chars :]!r}."
                )
            check = await self.sandbox.exec(check_command, timeout_s=30)
            if check.exit_code != 0:
                raise RuntimeError(
                    f"Harness executable {self.config.executable!r} is unavailable after installation. "
                    f"check_stdout={check.stdout[-self.config.output_tail_chars :]!r}, "
                    f"check_stderr={check.stderr[-self.config.output_tail_chars :]!r}, "
                    f"install_stdout={installed.stdout[-self.config.output_tail_chars :]!r}, "
                    f"install_stderr={installed.stderr[-self.config.output_tail_chars :]!r}."
                )
        mkdir = await self.sandbox.exec(f"mkdir -p {shlex.quote(self._log_dir)}", timeout_s=30)
        if mkdir.exit_code != 0:
            raise RuntimeError(f"Could not create harness state directory: {mkdir.stderr.strip()}")
        await self._prepare(runtime)

    async def _install_cli(self, check_command: str) -> ExecResult:
        """Install the configured CLI, optionally using npm tarballs."""
        install = self.config.install
        if install.strategy == "command":
            if not install.command:
                raise RuntimeError(
                    f"Harness executable {self.config.executable!r} is unavailable and no install command "
                    "is configured."
                )
            return await self.sandbox.exec(install.command, timeout_s=install.timeout_s)

        if not install.node_tarball_env or not install.cli_tarball_env:
            raise RuntimeError("npm_tarball harness installation requires node_tarball_env and cli_tarball_env.")
        if not os.environ.get(install.node_tarball_env):
            raise RuntimeError(
                f"Host environment variable {install.node_tarball_env!r} is not set for npm installation."
            )
        if not os.environ.get(install.cli_tarball_env):
            raise RuntimeError(
                f"Host environment variable {install.cli_tarball_env!r} is not set for npm installation."
            )

        node_dir = shlex.quote(install.node_install_dir)
        node_tarball = shlex.quote(install.node_tarball_path)
        cli_tarball = shlex.quote(install.cli_tarball_path)
        npm_prefix = shlex.quote(install.npm_prefix)
        command = (
            "set -euo pipefail; "
            f"if [ ! -x {node_dir}/bin/node ]; then "
            f"mkdir -p {node_dir}; "
            f"if tar -tf {node_tarball} >/dev/null 2>&1; then "
            f"tar -xf {node_tarball} -C {node_dir} --strip-components=1; "
            f"elif command -v xz >/dev/null 2>&1; then "
            f"xz -dc {node_tarball} | tar -xf - -C {node_dir} --strip-components=1; "
            "else echo 'Node tarball is compressed but xz is unavailable.' >&2; exit 127; fi; "
            "fi; "
            f"ln -sf {node_dir}/bin/node /usr/local/bin/node; "
            f"ln -sf {node_dir}/bin/npm /usr/local/bin/npm; "
            f"ln -sf {node_dir}/bin/npx /usr/local/bin/npx; "
            "hash -r; "
            f"npm install -g --prefix={npm_prefix} --no-audit --no-fund {cli_tarball}; "
            f"{check_command}"
        )
        last_result = ExecResult(1, "", "")
        for attempt in range(install.retries):
            last_result = await self.sandbox.exec(command, timeout_s=install.timeout_s)
            if last_result.exit_code == 0:
                return last_result
            if attempt + 1 < install.retries and install.retry_backoff_s:
                await asyncio.sleep(install.retry_backoff_s * (attempt + 1))
        return last_result

    @staticmethod
    def inherited_proxy_env() -> dict[str, str]:
        """Copy proxy variables from the Ray worker environment without values in code."""
        keys = (
            "http_proxy",
            "https_proxy",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "all_proxy",
            "ALL_PROXY",
            "no_proxy",
            "NO_PROXY",
        )
        return {key: os.environ[key] for key in keys if os.environ.get(key)}

    @staticmethod
    def callback_no_proxy(runtime: HarnessRuntime, env: Mapping[str, str]) -> dict[str, str]:
        """Keep the in-cluster callback endpoint outside the public proxy."""
        result = dict(env)
        host = urlsplit(runtime.session_root_url).hostname
        if not host:
            return result
        for key in ("no_proxy", "NO_PROXY"):
            entries = [item.strip() for item in result.get(key, "").split(",") if item.strip()]
            if host not in entries:
                entries.append(host)
            result[key] = ",".join(entries)
        return result

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
        """Run the harness CLI and retain bounded stdout/stderr diagnostic tails."""
        stderr_path = str(PurePosixPath(self._log_dir) / "stderr.log")
        stdout_path = str(PurePosixPath(self._log_dir) / "stdout.log")
        argv = self.build_command(prompt, runtime)
        command = (
            f"{shlex.join([str(item) for item in argv])} > {shlex.quote(stdout_path)} 2> {shlex.quote(stderr_path)}"
        )
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
            return HarnessResult(
                exit_code=0,
                command_name=str(argv[0]) if argv else self.config.executable,
                workdir=runtime.workdir,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
        return HarnessResult(
            exit_code=result.exit_code,
            stdout_tail=await self._read_tail(stdout_path),
            stderr_tail=await self._read_tail(stderr_path),
            command_name=str(argv[0]) if argv else self.config.executable,
            workdir=runtime.workdir,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    async def collect_diagnostics(self, result: HarnessResult) -> HarnessResult:
        """Read process output on demand for failures discovered after exit 0."""
        if result.stdout_tail or result.stderr_tail or not result.stdout_path or not result.stderr_path:
            return result
        return replace(
            result,
            stdout_tail=await self._read_tail(result.stdout_path),
            stderr_tail=await self._read_tail(result.stderr_path),
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
