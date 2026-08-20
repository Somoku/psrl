"""Unit tests for task-scoped coding harness adapters."""

from __future__ import annotations

import ast
import asyncio
from collections.abc import Mapping
from pathlib import Path

import pytest
from examples.mini_swe.config import build_runtime_config
from examples.mini_swe.harness_task import build_harness_prompt, collect_git_patch
from omegaconf import OmegaConf
from psrl.sandbox import (
    ExecResult,
    SandboxCapabilities,
    SandboxRef,
    SandboxSession,
    SandboxSource,
    SandboxSpec,
    SandboxStatePolicy,
    SandboxStatus,
)
from psrl.sandbox.backends.docker import DockerBackend, DockerPolicyProfile, DockerSession
from psrl.workers.agent_loop.harness import (
    HarnessCompactionConfig,
    HarnessConfig,
    HarnessRuntime,
    HarnessTaskContext,
    clean_snapshot_compatible,
    create_harness,
)
from psrl.workers.agent_loop.harness.claude_code import ClaudeCodeHarness
from psrl.workers.agent_loop.harness.codex import CodexHarness


class FakeSandbox(SandboxSession):
    """Minimal async data plane with deterministic command behavior."""

    def __init__(
        self,
        *,
        executable_available: bool = True,
        block_cli: bool = False,
        cli_exit_code: int = 0,
    ) -> None:
        self.executable_available = executable_available
        self.block_cli = block_cli
        self.cli_exit_code = cli_exit_code
        self.commands: list[tuple[str, str | None, Mapping[str, str] | None]] = []
        self.writes: dict[str, bytes] = {}
        self.cli_started = asyncio.Event()
        self.cli_cancelled = False

    @property
    def ref(self) -> SandboxRef:
        return SandboxRef("fake", "sandbox-1")

    @property
    def capabilities(self) -> SandboxCapabilities:
        return SandboxCapabilities()

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> ExecResult:
        self.commands.append((command, cwd, env))
        if command.startswith("command -v"):
            return ExecResult(0 if self.executable_available else 1, "/usr/bin/tool\n", "")
        if command.startswith("npm install"):
            self.executable_available = True
            return ExecResult(0, "installed", "")
        if command.startswith("tail -c"):
            return ExecResult(0, "bounded tail", "")
        if command.startswith(("claude ", "codex ")) and self.block_cli:
            self.cli_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                self.cli_cancelled = True
                raise
        if command.startswith(("claude ", "codex ")):
            return ExecResult(self.cli_exit_code, "", "")
        return ExecResult(0, "", "")

    async def read_bytes(self, path: str) -> bytes:
        return self.writes[path]

    async def write_bytes(self, path: str, data: bytes) -> None:
        self.writes[path] = data

    async def status(self) -> SandboxStatus:
        return SandboxStatus.RUNNING

    async def terminate(self) -> None:
        return None


def _runtime(
    context_window_tokens: int | None = None,
    compaction_token_limit: int | None = None,
    max_turns: int | None = None,
    max_output_tokens: int | None = None,
) -> HarnessRuntime:
    return HarnessRuntime(
        session_id="session-1",
        session_root_url="http://router/sessions/session-1",
        workdir="/testbed",
        model="Qwen/Qwen3",
        context_window_tokens=context_window_tokens,
        compaction_token_limit=compaction_token_limit,
        max_turns=max_turns,
        max_output_tokens=max_output_tokens,
    )


def test_harness_compaction_budget_uses_a_safe_trigger_before_the_window() -> None:
    assert HarnessCompactionConfig().resolve(10_240) == (10_240, 9_728)
    assert HarnessCompactionConfig(safety_tokens=0).resolve(10_240) == (10_240, 10_240)
    assert HarnessCompactionConfig(context_window_tokens=32_000, safety_tokens=0).resolve(10_240) == (
        32_000,
        10_240,
    )
    with pytest.raises(ValueError, match="cannot exceed"):
        HarnessCompactionConfig(trigger_tokens=10_241).resolve(10_240)


def test_harness_prompt_preserves_native_miniswe_task_boundary() -> None:
    prompt = build_harness_prompt("Fix the parser.")

    assert prompt == (
        "<pr_description>\nFix the parser.\n</pr_description>\n\n"
        "Implement the required changes in the current repository and verify the fix."
    )


@pytest.mark.asyncio
async def test_claude_code_uses_session_root_and_bounded_process_output() -> None:
    sandbox = FakeSandbox(cli_exit_code=1)
    config = HarnessConfig.from_value(
        {
            "kind": "claude_code",
            "executable": "claude",
            "args": ["--max-budget-usd", "0"],
            "permission_mode": "default",
            "allowed_tools": ["Bash", "Read", "Edit"],
            "system_prompt": "minimal system",
            "tools": "Bash,Read,Edit",
            "env": {"ANTHROPIC_BASE_URL": "http://must-not-escape", "CUSTOM": "value"},
        }
    )
    harness = create_harness(config, sandbox)

    await harness.prepare(_runtime())
    result = await harness.run("fix the bug", _runtime())

    assert isinstance(harness, ClaudeCodeHarness)
    assert result.exit_code == 1
    assert result.stderr_tail == "bounded tail"
    assert b"hasCompletedOnboarding" in sandbox.writes["/root/.claude.json"]
    cli_command, cwd, env = next(item for item in sandbox.commands if item[0].startswith("claude "))
    assert "--output-format stream-json" in cli_command
    assert "--permission-mode default" in cli_command
    assert "--allowedTools Bash Read Edit" in cli_command
    assert "--system-prompt 'minimal system'" in cli_command
    assert "--tools Bash,Read,Edit" in cli_command
    assert "--max-turns" not in cli_command
    assert "> /root/.psrl-harness/stdout.log" in cli_command
    assert cwd == "/testbed"
    assert env is not None
    assert env["ANTHROPIC_BASE_URL"] == "http://router/sessions/session-1"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "session-1"
    assert env["CUSTOM"] == "value"


@pytest.mark.asyncio
async def test_successful_harness_skips_diagnostic_tail_commands() -> None:
    sandbox = FakeSandbox()
    harness = create_harness(HarnessConfig(kind="codex", executable="codex"), sandbox)

    await harness.prepare(_runtime())
    result = await harness.run("fix", _runtime())

    assert result.exit_code == 0
    assert result.stdout_tail == ""
    assert result.stderr_tail == ""
    assert not any(command.startswith("tail -c") for command, _, _ in sandbox.commands)


@pytest.mark.asyncio
async def test_claude_code_receives_token_based_compaction_settings() -> None:
    sandbox = FakeSandbox()
    harness = create_harness(HarnessConfig(kind="claude_code", executable="claude"), sandbox)

    env = harness.build_env(_runtime(10_240, 9_728))

    assert env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "10240"
    assert env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] == "95"
    assert "CLAUDE_CODE_MAX_OUTPUT_TOKENS" not in env


def test_claude_code_receives_framework_turn_and_output_limits() -> None:
    sandbox = FakeSandbox()
    harness = create_harness(HarnessConfig(kind="claude_code", executable="claude"), sandbox)
    runtime = _runtime(max_turns=12, max_output_tokens=2048)

    command = harness.build_command("fix", runtime)
    env = harness.build_env(runtime)

    assert "--max-turns 12" in " ".join(command)
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "2048"


@pytest.mark.asyncio
async def test_failed_trajectory_can_collect_output_after_successful_cli_exit() -> None:
    sandbox = FakeSandbox()
    harness = create_harness(HarnessConfig(kind="codex", executable="codex"), sandbox)
    await harness.prepare(_runtime())

    result = await harness.run("fix", _runtime())
    diagnostic_result = await harness.collect_diagnostics(result)

    assert diagnostic_result.stdout_tail == "bounded tail"
    assert diagnostic_result.stderr_tail == "bounded tail"
    assert sum(command.startswith("tail -c") for command, _, _ in sandbox.commands) == 2


@pytest.mark.asyncio
async def test_patch_collection_includes_staged_and_untracked_changes() -> None:
    sandbox = FakeSandbox()

    await collect_git_patch(sandbox, "/testbed")

    assert ("git add -N . && git diff --binary HEAD -- .", "/testbed", None) in sandbox.commands


@pytest.mark.asyncio
async def test_codex_writes_responses_provider_config_and_uses_session_as_key() -> None:
    sandbox = FakeSandbox()
    config = HarnessConfig(kind="codex", executable="codex")
    harness = create_harness(config, sandbox)

    await harness.prepare(_runtime())

    assert isinstance(harness, CodexHarness)
    codex_config = sandbox.writes["/root/.codex/config.toml"].decode()
    assert 'base_url = "http://router/sessions/session-1/v1"' in codex_config
    assert 'wire_api = "responses"' in codex_config
    assert "requires_openai_auth = false" in codex_config
    assert "supports_websockets = false" in codex_config
    assert harness.build_env(_runtime())["OPENAI_API_KEY"] == "session-1"
    assert harness.build_command("fix", _runtime())[:3] == (
        "codex",
        "exec",
        "--skip-git-repo-check",
    )


@pytest.mark.asyncio
async def test_codex_writes_compaction_threshold_with_context_headroom() -> None:
    sandbox = FakeSandbox()
    harness = create_harness(HarnessConfig(kind="codex", executable="codex"), sandbox)

    await harness.prepare(_runtime(10_240, 9_728))

    codex_config = sandbox.writes["/root/.codex/config.toml"].decode()
    assert "model_context_window = 10809" in codex_config
    assert "model_auto_compact_token_limit = 9728" in codex_config


@pytest.mark.asyncio
async def test_optional_install_runs_once_before_adapter_preparation() -> None:
    sandbox = FakeSandbox(executable_available=False)
    config = HarnessConfig.from_value(
        {
            "kind": "codex",
            "executable": "codex",
            "install": {"command": "npm install -g @openai/codex"},
        }
    )

    await create_harness(config, sandbox).prepare(_runtime())

    commands = [item[0] for item in sandbox.commands]
    assert commands.count("command -v codex") == 2
    assert "npm install -g @openai/codex" in commands


@pytest.mark.asyncio
async def test_abort_cancels_active_cli_exec() -> None:
    sandbox = FakeSandbox(block_cli=True)
    harness = create_harness(HarnessConfig(kind="codex", executable="codex"), sandbox)
    await harness.prepare(_runtime())
    run_task = asyncio.create_task(harness.run("fix", _runtime()))
    await sandbox.cli_started.wait()

    await harness.abort()

    with pytest.raises(asyncio.CancelledError):
        await run_task
    assert sandbox.cli_cancelled


def test_registry_returns_new_task_owned_harnesses() -> None:
    config = HarnessConfig(kind="codex", executable="codex")
    sandbox = FakeSandbox()

    assert create_harness(config, sandbox) is not create_harness(config, sandbox)


def test_example_config_selects_both_harnesses() -> None:
    path = Path(__file__).parents[2] / "examples/mini_swe/config/swebench_harness_config.yaml"
    configs = OmegaConf.load(path)

    assert [item.name for item in configs] == ["mini_swe_claude_code", "mini_swe_codex"]
    assert [HarnessConfig.from_value(item.harness).kind for item in configs] == ["claude_code", "codex"]
    assert all(item.sandbox_config.environment.cwd == "/testbed" for item in configs)

    runtime = build_runtime_config(
        {"sandbox_config": OmegaConf.to_container(configs[0].sandbox_config, resolve=True)},
        require_agent_templates=False,
    )
    assert runtime.agent.system_template == ""
    assert runtime.sandbox_config.environment.cwd == "/testbed"


def test_task_context_only_enables_compatible_clean_snapshot() -> None:
    state_policy = SandboxStatePolicy(enabled=True)
    rollout = SandboxSpec(source=SandboxSource.image("image"), state_policy=state_policy)
    compatible = HarnessTaskContext(
        state={"task": "opaque"},
        prompt="solve",
        sandbox_spec=rollout,
        clean_sandbox_spec=SandboxSpec(source=SandboxSource.image("image"), state_policy=state_policy),
    )
    incompatible = HarnessTaskContext(
        state=None,
        prompt="solve",
        sandbox_spec=rollout,
        clean_sandbox_spec=SandboxSpec(source=SandboxSource.image("other-image"), state_policy=state_policy),
    )
    disabled = HarnessTaskContext(
        state=None,
        prompt="solve",
        sandbox_spec=SandboxSpec(source=SandboxSource.image("image")),
        clean_sandbox_spec=SandboxSpec(source=SandboxSource.image("image")),
    )

    assert clean_snapshot_compatible(compatible)
    assert not clean_snapshot_compatible(incompatible)
    assert not clean_snapshot_compatible(disabled)


def test_generic_harness_loop_has_no_task_specific_imports() -> None:
    path = Path(__file__).parents[2] / "psrl/workers/agent_loop/loops/harness_agent_loop.py"
    module = ast.parse(path.read_text())
    imports = {
        node.module for node in ast.walk(module) if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not any(name.startswith("examples.") for name in imports)


def test_docker_session_translates_loopback_callback_through_policy() -> None:
    backend = DockerBackend(
        policy_profiles={
            "mini_swe": DockerPolicyProfile(host_gateway_alias="host.docker.internal"),
        },
        engine=object(),
    )
    session = DockerSession(
        backend,
        "container-1",
        spec=SandboxSpec(
            source=SandboxSource.image("image"),
            policy_profile="mini_swe",
        ),
    )

    assert session.resolve_callback_url("http://127.0.0.1:8080/sessions/s1") == (
        "http://host.docker.internal:8080/sessions/s1"
    )
    assert session.resolve_callback_url("https://router.internal/sessions/s1") == (
        "https://router.internal/sessions/s1"
    )
