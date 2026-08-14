"""Codex CLI harness adapter."""

import json
import shlex
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath

from psrl.workers.agent_loop.harness.base import Harness, HarnessRuntime


class CodexHarness(Harness):
    """Run Codex against a TITO session-scoped Responses endpoint."""

    async def _prepare(self, runtime: HarnessRuntime) -> None:
        codex_dir = PurePosixPath(self.config.home_dir) / ".codex"
        result = await self.sandbox.exec(f"mkdir -p {shlex.quote(str(codex_dir))}", timeout_s=30)
        if result.exit_code != 0:
            raise RuntimeError(f"Could not create Codex config directory: {result.stderr.strip()}")
        model = self.config.model or runtime.model
        config = "\n".join(
            (
                f"model = {json.dumps(model)}",
                'model_provider = "psrl"',
                'approval_policy = "never"',
                'sandbox_mode = "danger-full-access"',
                "",
                "[model_providers.psrl]",
                'name = "PSRL TITO"',
                f"base_url = {json.dumps(runtime.session_root_url.rstrip('/') + '/v1')}",
                'env_key = "OPENAI_API_KEY"',
                'wire_api = "responses"',
                "requires_openai_auth = false",
                "supports_websockets = false",
                "",
            )
        )
        await self.sandbox.write_bytes(str(codex_dir / "config.toml"), config.encode())

    def build_command(self, prompt: str, runtime: HarnessRuntime) -> Sequence[str]:
        return (
            self.config.executable,
            "exec",
            "--skip-git-repo-check",
            *self.config.args,
            prompt,
        )

    def build_env(self, runtime: HarnessRuntime) -> Mapping[str, str]:
        return {
            **self.config.env,
            "HOME": self.config.home_dir,
            "OPENAI_API_KEY": runtime.session_id,
            "CODEX_HOME": str(PurePosixPath(self.config.home_dir) / ".codex"),
        }
