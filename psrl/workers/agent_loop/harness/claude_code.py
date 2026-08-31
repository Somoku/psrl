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
        permission_mode = self.config.permission_mode or "default"
        result = await self.sandbox.exec(f"mkdir -p {shlex.quote(str(claude_dir))}", timeout_s=30)
        if result.exit_code != 0:
            raise RuntimeError(f"Could not create Claude Code config directory: {result.stderr.strip()}")
        await self.sandbox.write_bytes(
            str(PurePosixPath(self.config.home_dir) / ".claude.json"),
            json.dumps({"hasCompletedOnboarding": True}).encode(),
        )
        deny = [] if self.config.subagents_enabled else ["Agent"]
        settings_env = {"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
        if self.config.disable_prompt_caching:
            settings_env["DISABLE_PROMPT_CACHING"] = "1"
        if not self.config.thinking_enabled:
            settings_env.update(
                {
                    "CLAUDE_CODE_DISABLE_THINKING": "1",
                    "MAX_THINKING_TOKENS": "0",
                }
            )
        elif self.config.thinking_budget_tokens is not None:
            settings_env["MAX_THINKING_TOKENS"] = str(self.config.thinking_budget_tokens)
        if not self.config.interleaved_thinking:
            settings_env["DISABLE_INTERLEAVED_THINKING"] = "1"
        await self.sandbox.write_bytes(
            str(claude_dir / "settings.json"),
            json.dumps(
                {
                    "autoCompactEnabled": self.config.compaction.enabled,
                    "permissions": {
                        "defaultMode": permission_mode,
                        "deny": deny,
                    },
                    "env": settings_env,
                }
            ).encode(),
        )

    def build_command(self, prompt: str, runtime: HarnessRuntime) -> Sequence[str]:
        command = [
            self.config.executable,
            "-p",
            prompt,
            "--permission-mode",
            self.config.permission_mode or "default",
            "--output-format",
            "stream-json",
            "--include-partial-messages",
            "--include-hook-events",
            "--verbose",
            "--model",
            self.config.model or runtime.model,
        ]
        if self.config.allowed_tools:
            command.extend(("--allowedTools", *self.config.allowed_tools))
        if self.config.system_prompt is not None and self.config.system_prompt_mode != "none":
            prompt_flag = "--append-system-prompt" if self.config.system_prompt_mode == "append" else "--system-prompt"
            command.extend((prompt_flag, self.config.system_prompt))
        if self.config.tools is not None:
            command.extend(("--tools", self.config.tools))
        if self.config.setting_sources:
            command.extend(("--setting-sources", self.config.setting_sources))
        if runtime.max_turns is not None:
            # Keep the framework's episode bound enforced by Claude Code too;
            # otherwise the CLI can continue making tool-use turns after PSRL's
            # post-hoc max-turn classification has already become irrelevant.
            command.extend(("--max-turns", str(runtime.max_turns)))
        if self.config.reasoning_effort is not None:
            command.extend(("--effort", self.config.reasoning_effort))
        command.extend(self.config.args)
        return tuple(command)

    def build_env(self, runtime: HarnessRuntime) -> Mapping[str, str]:
        env = {
            **self.inherited_proxy_env(),
            **self.config.env,
            "HOME": self.config.home_dir,
            "IS_SANDBOX": "1",
            "ANTHROPIC_BASE_URL": runtime.session_root_url,
            "ANTHROPIC_AUTH_TOKEN": runtime.session_id,
            "ANTHROPIC_MODEL": self.config.model or runtime.model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": self.config.model or runtime.model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": self.config.model or runtime.model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": self.config.model or runtime.model,
            "CLAUDE_CODE_SUBAGENT_MODEL": self.config.model or runtime.model,
            "DISABLE_TELEMETRY": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
        }
        if self.config.max_output_tokens is not None:
            env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(self.config.max_output_tokens)
        if self.config.supported_capabilities:
            model = self.config.model or runtime.model
            env.update(
                {
                    "ANTHROPIC_CUSTOM_MODEL_OPTION": model,
                    "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": model,
                    "ANTHROPIC_CUSTOM_MODEL_OPTION_SUPPORTED_CAPABILITIES": ",".join(
                        self.config.supported_capabilities
                    ),
                }
            )
        if self.config.disable_prompt_caching:
            env["DISABLE_PROMPT_CACHING"] = "1"
        if self.config.disable_experimental_betas:
            env["CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"] = "1"
        if not self.config.thinking_enabled:
            env["CLAUDE_CODE_DISABLE_THINKING"] = "1"
            env["MAX_THINKING_TOKENS"] = "0"
        elif self.config.thinking_budget_tokens is not None:
            env["MAX_THINKING_TOKENS"] = str(self.config.thinking_budget_tokens)
        if not self.config.interleaved_thinking:
            env["DISABLE_INTERLEAVED_THINKING"] = "1"
        if not self.config.subagents_enabled:
            env["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] = "1"
        if runtime.context_window_tokens and runtime.compaction_token_limit:
            # Claude Code accepts the compaction window in tokens and the
            # trigger as a percentage of that window.  Keep explicit adapter
            # env overrides authoritative for experiments and compatibility.
            trigger_percent = max(
                1,
                min(100, (runtime.compaction_token_limit * 100) // runtime.context_window_tokens),
            )
            env.setdefault("CLAUDE_CODE_AUTO_COMPACT_WINDOW", str(runtime.context_window_tokens))
            env.setdefault("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE", str(trigger_percent))
        return self.callback_no_proxy(runtime, env)
