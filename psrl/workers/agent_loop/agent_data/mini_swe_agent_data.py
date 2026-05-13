"""
MiniSWEAgentData — AgentData adapter for mini-SWE-agent subprocess-proxy pattern.

Extends `ConversationAgentData` with SWE-specific fields and reward metadata:
- `set_patch` / `set_grader_result` for post-rollout grading results.
- Extended `finalize_output` that builds `agent_reward_info` (patch, acc, etc.)
  before delegating to the base class.
- `encode_messages` helper for the generation loop's token-budget check.

Chat-style interaction semantics (turn encoding, trajectory building) are fully
inherited from `ConversationAgentData`. No action parsing is performed
(`decode_action_from_token_ids` raises `NotImplementedError`; the base class
treats this as `action=None`).
"""

from __future__ import annotations

import logging
import os

import numpy as np
import ray
from omegaconf import DictConfig
from transformers import AutoTokenizer
from verl import DataProto

from psrl.environments.base import ConversationType, Environment
from psrl.workers.agent_loop.agent_data.base import AgentData, Trajectory
from psrl.workers.agent_loop.agent_data.conversation_agent_data import (
    ConversationAgentData,
    normalize_openai_messages,  # re-exported for backward compatibility
)

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@AgentData.register("mini_swe_agent_data")
class MiniSWEAgentData(ConversationAgentData):
    """
    AgentData adapter for mini-SWE-agent subprocess-proxy pattern.

    The agent runs in a worker thread and communicates with the async
    generation loop via queues. The generation loop calls:

    - `update_from_env(full_messages, ...)` on turn 0 to set the prompt.
    - `update_from_env([new_user_message], ...)` on turns 1+ to append
      the user delta.
    - `update_from_model_token_ids(output)` after each vLLM call to append
      the assistant response.

    After all turns, the loop calls `set_patch`, optionally `set_grader_result`,
    and then `finalize_output`.
    """

    def __init__(
        self,
        config: DictConfig,
        reward_manager: ray.actor.ActorHandle,
        tokenizer: AutoTokenizer,
        env: Environment,
        **kwargs,
    ):
        """
        Initialize MiniSWEAgentData.

        Args:
            config (DictConfig): PSRL trainer configuration.
            reward_manager: Ray actor handle for computing rewards.
            tokenizer (AutoTokenizer): Tokenizer for converting between text and tokens.
            env (Environment): Associated environment instance.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(config, reward_manager, tokenizer, env)
        self.patch: str | None = None
        self.problem_statement: str = ""
        self._grader_result: dict = {}

    def reset(self) -> None:
        """Reset agent data for a new episode."""
        super().reset()
        self.patch = None
        self.problem_statement = ""
        self._grader_result = {}

    def init_trajectory(self, request: DataProto) -> None:
        """
        Initialize a new trajectory from the input request.

        Extracts uid, parent_id, and problem_statement from the request.

        Args:
            request (DataProto): Input request containing uid, parent_id,
                and extra_info (which may carry problem_statement).
        """
        super().init_trajectory(request)
        extra_info_raw = request.non_tensor_batch.get("extra_info", [{}])[0]
        if isinstance(extra_info_raw, dict):
            self.problem_statement = extra_info_raw.get("problem_statement", "")

    def set_patch(self, patch: str | None) -> None:
        """
        Store the extracted patch for reward computation.

        Args:
            patch (str | None): Generated patch string, or None.
        """
        self.patch = patch

    def set_grader_result(self, result: dict) -> None:
        """
        Store the post-rollout grading result from `swebench_grader.grade_fresh_container`.

        The result dict is forwarded into `agent_reward_info` during
        `finalize_output` so that `compute_score` can read it from
        `extra_info.grader_result`.

        Args:
            result (dict): Grading result from `grade_fresh_container`.
        """
        self._grader_result = result

    def encode_messages(
        self,
        messages: list[dict],
        *,
        add_generation_prompt: bool = True,
    ) -> list[int]:
        """
        Tokenize OpenAI-format messages into token IDs.

        This helper is used by `MiniSWEAgentLoop._generation_loop` for the
        token-budget check only. For trajectory building, use `update_from_env`
        and `update_from_model_token_ids`.

        Args:
            messages (list[dict]): OpenAI-format message dicts.
            add_generation_prompt (bool): Whether to add the generation prompt suffix.

        Returns:
            list[int]: Token IDs.
        """
        normalized = normalize_openai_messages(messages)
        return self._apply_chat_template_ids(
            normalized, add_generation_prompt=add_generation_prompt,
        )

    async def finalize_output(self, request: DataProto) -> DataProto:
        """
        Finalize trajectory and prepare `DataProto` output for training.

        Attaches SWE-specific reward metadata (`agent_reward_info`) containing
        patch, grader result, and acc to the request, then delegates to
        `super().finalize_output()` for truncation, non-tensor-batch building,
        routing metadata, and reward computation.

        Args:
            request (DataProto): Original request DataProto containing metadata.

        Returns:
            DataProto: Finalized DataProto with reward.
        """
        num_turns = self.trajectory.assistant_turns
        swe_reward_info = {
            "patch": self.patch,
            "num_turns": num_turns,
            "actual_num_turns": num_turns,
            "alignment_failed": False,
            "alignment_failure_reason": "",
            # Grader result (populated by agent loop after fresh-container eval,
            # empty dict for toy / simple-test data sources).
            "grader_result": self._grader_result,
            # Emit acc (resolve_rate, 0/1) alongside the shaped score metric
            # so wandb shows both train/score and train/acc.
            "acc": float(bool(self._grader_result.get("resolved", False))),
        }
        request.non_tensor_batch["agent_reward_info"] = np.array([swe_reward_info])
        psrl_logger.info(
            f"[finalize_output] uid={request.non_tensor_batch.get('uid', ['?'])[0]}, "
            f"agent_reward_info={swe_reward_info}, "
            f"non_tensor_batch_keys={list(request.non_tensor_batch.keys())}."
        )
        return await super().finalize_output(request)
