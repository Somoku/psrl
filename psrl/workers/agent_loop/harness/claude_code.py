"""Claude Code harness adapter."""

import json
import shlex
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath

from psrl.workers.agent_loop.harness.base import Harness, HarnessRuntime


class ClaudeCodeHarness(Harness):
    """Run Claude Code against a TITO session-scoped Messages endpoint."""

    async def _prepare(self, runtime: HarnessRuntime) -> None:
        claude_dir = PurePosixPath(self.config.home_dir) / ".claude"
        result = await self.sandbox.exec(f"mkdir -p {shlex.quote(str(claude_dir))}", timeout_s=30)
        if result.exit_code != 0:
            raise RuntimeError(f"Could not create Claude Code config directory: {result.stderr.strip()}")
        await self.sandbox.write_bytes(
            str(PurePosixPath(self.config.home_dir) / ".claude.json"),
            json.dumps({"hasCompletedOnboarding": True}).encode(),
        )
        await self.sandbox.write_bytes(
            str(claude_dir / "settings.json"),
            json.dumps(
                {
                    "permissions": {"defaultMode": "bypassPermissions"},
                    "env": {"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"},
                }
            ).encode(),
        )

    def build_command(self, prompt: str, runtime: HarnessRuntime) -> Sequence[str]:
        return (
            self.config.executable,
            "-p",
            prompt,
            "--permission-mode",
            "bypassPermissions",
            "--output-format",
            "stream-json",
            "--include-partial-messages",
            "--include-hook-events",
            "--verbose",
            *self.config.args,
        )

    def build_env(self, runtime: HarnessRuntime) -> Mapping[str, str]:
        return {
            **self.config.env,
            "HOME": self.config.home_dir,
            "ANTHROPIC_BASE_URL": runtime.session_root_url,
            "ANTHROPIC_AUTH_TOKEN": runtime.session_id,
            "ANTHROPIC_MODEL": self.config.model or runtime.model,
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_TELEMETRY": "1",
        }
