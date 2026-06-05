"""Agent data adapter for mini-SWE-agent trajectories."""

from __future__ import annotations

import logging
import os

import ray
from omegaconf import DictConfig

from psrl.environments.base import Environment
from psrl.workers.agent_loop.agent_data.base import AgentData
from psrl.workers.agent_loop.agent_data.conversation_agent_data import (
    ConversationAgentData,
    normalize_openai_messages,
)

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@AgentData.register("mini_swe_agent_data")
class MiniSWEAgentData(ConversationAgentData):
    """Attach SWE patch and grader metadata to a conversation trajectory."""

    def __init__(
        self,
        config: DictConfig,
        reward_manager: ray.actor.ActorHandle,
        env: Environment,
        **kwargs,
    ):
        super().__init__(config, reward_manager, env, **kwargs)
        self.patch: str | None = None
        self.problem_statement = ""
        self.grader_result: dict = {}

    def reset(self) -> None:
        """Reset trajectory and SWE-specific state."""
        super().reset()
        self.patch = None
        self.problem_statement = ""
        self.grader_result = {}

    def init_trajectory(self, request: dict) -> None:
        """Initialize trajectory metadata and capture the problem statement."""
        super().init_trajectory(request)
        self.problem_statement = self.session_data.extra_info.get("problem_statement", "")

    def set_patch(self, patch: str | None) -> None:
        """Store the patch submitted by mini-SWE-agent."""
        self.patch = patch

    def set_grader_result(self, result: dict) -> None:
        """Store the optional post-rollout grading result."""
        self.grader_result = result

    async def encode_messages(
        self,
        messages: list[dict],
        *,
        add_generation_prompt: bool = True,
    ) -> list[int]:
        """Tokenize normalized OpenAI-format messages."""
        return await self._apply_chat_template_ids(
            normalize_openai_messages(messages),
            add_generation_prompt=add_generation_prompt,
        )

    async def finalize_output(self):
        """Finalize output after attaching SWE metadata to the agent namespace."""
        num_turns = self.session_data.trajectories[-1].assistant_turns
        self.session_data.agent_reward_info.update(
            {
                "patch": self.patch,
                "num_turns": num_turns,
                "actual_num_turns": num_turns,
                "alignment_failed": False,
                "alignment_failure_reason": "",
                "grader_result": self.grader_result,
                "acc": float(bool(self.grader_result.get("resolved", False))),
            }
        )
        return await super().finalize_output()
