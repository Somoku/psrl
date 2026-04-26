"""
mini-SWE-agent AgentData for PSRL.

Handles tokenization of OpenAI-format messages, per-turn recording into
PSRL's native `Step`/`Trajectory` structures, strict token-level
trajectory reconstruction, and `DataProto` finalization.

This class does NOT use the traditional env-step-driven methods
(`encode_observation`, `decode_action_from_token_ids`, `update_from_env`,
`update_from_model_token_ids`). Instead it exposes `encode_messages()`,
`record_turn()`, `reconstruct_and_validate()`, and `set_patch()` for
the subprocess-proxy interaction pattern.
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

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

# Maximum trailing chat-template tokens tolerated after assistant content.
_MAX_TRAILING_TEMPLATE_TOKENS = 3


def normalize_openai_messages(openai_messages: list[dict]) -> list[dict]:
    """
    Normalize OpenAI-format messages for `tokenizer.apply_chat_template`.

    Handles `content` as a list of text blocks, None, or non-string.
    """
    messages: list[dict] = []
    for msg in openai_messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    text_parts.append(
                        part.get("text", str(part))
                        if part.get("type") == "text"
                        else str(part.get("text", part))
                    )
                else:
                    text_parts.append(str(part))
            content = "\n".join(text_parts)
        elif content is None:
            content = ""
        else:
            content = str(content)
        messages.append({"role": role, "content": content})
    return messages


@AgentData.register("mini_swe_agent_data")
class MiniSWEAgentData(AgentData[dict, None]):
    """
    AgentData adapter for mini-SWE-agent subprocess-proxy pattern.

    Uses PSRL's native `Step` and `Trajectory` classes. Per-turn metadata
    (raw messages, prompt_ids, response_ids, logprobs) is stored in
    `Step.info["turn_record"]` for trajectory reconstruction.
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
            config: PSRL trainer configuration.
            reward_manager: Ray actor handle for computing rewards.
            tokenizer: Tokenizer for converting between text and tokens.
            env: The associated environment instance.
        """
        super().__init__(config, reward_manager, tokenizer, env)
        self.patch: str | None = None
        self.problem_statement: str = ""
        self._alignment_failure_reason: str = ""
        self.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})

    def reset(self) -> None:
        """
        Reset agent data for a new episode.
        """
        self.trajectory = Trajectory()
        self.patch = None
        self.problem_statement = ""
        self._alignment_failure_reason = ""
        self._grader_result: dict = {}

    def init_trajectory(self, request: DataProto) -> None:
        """
        Initialize a new trajectory from the input request.

        Args:
            request: DataProto containing uid and parent_id.
        """
        assert len(request) == 1, "Only single request initialization is supported."

        request_id = request.non_tensor_batch.get("uid", [0])[0]
        parent_id = request.non_tensor_batch.get("parent_id", [None])[0]

        self.trajectory = Trajectory(
            request_id=request_id,
            parent_id=parent_id,
        )

        extra_info_raw = request.non_tensor_batch.get("extra_info", [{}])[0]
        if isinstance(extra_info_raw, dict):
            self.problem_statement = extra_info_raw.get("problem_statement", "")

    def encode_messages(
        self,
        messages: list[dict],
        *,
        add_generation_prompt: bool = True,
    ) -> list[int]:
        """
        Tokenize OpenAI-format messages into token IDs.

        Args:
            messages: List of OpenAI-format message dicts.
            add_generation_prompt: Whether to add the generation prompt suffix.

        Returns:
            List of token IDs.
        """
        normalized = normalize_openai_messages(messages)
        return self.tokenizer.apply_chat_template(
            normalized,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            **self.apply_chat_template_kwargs,
        )

    def update_from_external_turn(
        self,
        turn_index: int,
        messages: list[dict],
        prompt_ids: list[int],
        response_ids: list[int],
        response_text: str,
        response_logprobs: list[float],
        output: DataProto | None = None,
    ) -> None:
        """
        Record one interaction turn from an external agent into the trajectory.

        Creates a new `Step` and stores per-turn metadata in `Step.info["turn_record"]`
        for later trajectory reconstruction. Also updates routing metadata
        (`rollout_instance_id` / `version_tag`) from the generation output.

        Args:
            turn_index (int): Zero-based turn index.
            messages (list[dict]): Original OpenAI-format messages for this turn.
            prompt_ids (list[int]): Tokenized prompt IDs.
            response_ids (list[int]): Generated response token IDs.
            response_text (str): Decoded response text.
            response_logprobs (list[float]): Per-token log probabilities.
            output (DataProto | None): Generation output for updating routing state.
        """
        normalized = normalize_openai_messages(messages)

        step = self.start_step(
            observation={"turn_index": turn_index},
            reward=None,
            done=False,
            info={},
        )

        step.chat_completions = normalized
        step.model_response = response_text
        step.info["turn_record"] = {
            "turn_index": turn_index,
            "messages": [{"role": m["role"], "content": m["content"]} for m in normalized],
            "prompt_ids": list(prompt_ids),
            "response_ids": list(response_ids),
            "response_text": response_text,
            "response_logprobs": list(response_logprobs),
        }

        if output is not None:
            self.update_trajectory_state_from_output(output)

    def set_patch(self, patch: str | None) -> None:
        """
        Store the extracted patch for reward computation.

        Args:
            patch: Generated patch string, or None.
        """
        self.patch = patch

    def set_grader_result(self, result: dict) -> None:
        """
        Store the post-rollout grading result from `swebench_grader.grade_fresh_container`.

        The result dict is forwarded into `agent_reward_info` during
        `finalize_output` so that `compute_score` can read it from
        `extra_info.grader_result`.  The `resolved` and related fields are
        also stored at the top level for easy wandb metric emission.

        Args:
            result (dict): Grading result from `grade_fresh_container`.
        """
        self._grader_result: dict = result

    # TODO: a workaround solution for token misalignment.
    # Will replace it with 
    async def reconstruct_and_validate(self) -> bool:
        """
        Strict replay validation of the recorded trajectory.

        Re-renders each turn's messages to token IDs and validates:
        1. Prompt IDs match the recorded prompt IDs.
        2. Each turn's prompt is a prefix extension of the previous turn.
        3. Assistant response span matches the generated tokens (with up to
           3 trailing template tokens tolerated).

        On success, fills `Trajectory.prompt_ids`, `response_ids`,
        `response_mask`, and `response_logprobs`.

        On failure, clears these fields and returns False.

        Returns:
            True if reconstruction succeeded, False otherwise.
        """
        steps = self.trajectory.steps
        if not steps:
            return True

        # Accumulators.
        initial_prompt_ids: list[int] = []
        response_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        expected_prefix_ids: list[int] | None = None

        for step in steps:
            record = step.info.get("turn_record")
            if record is None:
                self._alignment_failure_reason = "Step missing turn_record."
                psrl_logger.warning(self._alignment_failure_reason)
                self._clear_trajectory_alignment()
                return False

            messages = record["messages"]
            recorded_prompt_ids = record["prompt_ids"]
            recorded_response_ids = record["response_ids"]
            recorded_logprobs = record["response_logprobs"]
            turn_index = record["turn_index"]

            # Re-render prompt.
            rendered_prompt_ids = self.encode_messages(messages, add_generation_prompt=True)

            # Validate prompt matches recorded.
            if rendered_prompt_ids != recorded_prompt_ids:
                self._alignment_failure_reason = (
                    f"Turn {turn_index}: prompt mismatch "
                    f"(rendered={len(rendered_prompt_ids)}, recorded={len(recorded_prompt_ids)})."
                )
                psrl_logger.warning(self._alignment_failure_reason)
                self._clear_trajectory_alignment()
                return False

            # Validate logprob lengths.
            if len(recorded_response_ids) != len(recorded_logprobs):
                self._alignment_failure_reason = (
                    f"Turn {turn_index}: logprob length mismatch "
                    f"(ids={len(recorded_response_ids)}, logprobs={len(recorded_logprobs)})."
                )
                psrl_logger.warning(self._alignment_failure_reason)
                self._clear_trajectory_alignment()
                return False

            if expected_prefix_ids is None:
                # First turn: set initial prompt.
                initial_prompt_ids = list(rendered_prompt_ids)
                expected_prefix_ids = list(rendered_prompt_ids)
            else:
                # Subsequent turns: validate prefix and append delta.
                if rendered_prompt_ids[:len(expected_prefix_ids)] != expected_prefix_ids:
                    self._alignment_failure_reason = (
                        f"Turn {turn_index}: prompt prefix mismatch "
                        f"(expected_prefix_len={len(expected_prefix_ids)}, "
                        f"prompt_len={len(rendered_prompt_ids)})."
                    )
                    psrl_logger.warning(self._alignment_failure_reason)
                    self._clear_trajectory_alignment()
                    return False

                delta_ids = rendered_prompt_ids[len(expected_prefix_ids):]
                response_ids.extend(delta_ids)
                response_mask.extend([0] * len(delta_ids))
                response_logprobs.extend([0.0] * len(delta_ids))

            # Replay assistant content to validate span.
            response_text = record["response_text"]
            assistant_messages = [
                *messages,
                {"role": "assistant", "content": response_text},
            ]
            after_assistant_ids = self.encode_messages(
                assistant_messages, add_generation_prompt=False,
            )

            if after_assistant_ids[:len(rendered_prompt_ids)] != rendered_prompt_ids:
                self._alignment_failure_reason = (
                    f"Turn {turn_index}: assistant replay prefix mismatch."
                )
                psrl_logger.warning(self._alignment_failure_reason)
                self._clear_trajectory_alignment()
                return False

            assistant_span_ids = after_assistant_ids[len(rendered_prompt_ids):]

            # Validate assistant span with trailing template token tolerance.
            trailing_template_ids: list[int] = []
            if assistant_span_ids != recorded_response_ids:
                gen = recorded_response_ids
                if (
                    len(assistant_span_ids) > len(gen)
                    and (len(assistant_span_ids) - len(gen)) <= _MAX_TRAILING_TEMPLATE_TOKENS
                    and assistant_span_ids[:len(gen)] == gen
                ):
                    trailing_template_ids = assistant_span_ids[len(gen):]
                else:
                    self._alignment_failure_reason = (
                        f"Turn {turn_index}: assistant span mismatch "
                        f"(replayed={len(assistant_span_ids)}, "
                        f"generated={len(recorded_response_ids)})."
                    )
                    psrl_logger.warning(self._alignment_failure_reason)
                    self._clear_trajectory_alignment()
                    return False

            # Append model response tokens (mask=1).
            response_ids.extend(recorded_response_ids)
            response_mask.extend([1] * len(recorded_response_ids))
            response_logprobs.extend(recorded_logprobs)

            # Append trailing template tokens (mask=0).
            if trailing_template_ids:
                response_ids.extend(trailing_template_ids)
                response_mask.extend([0] * len(trailing_template_ids))
                response_logprobs.extend([0.0] * len(trailing_template_ids))

            expected_prefix_ids = after_assistant_ids

        # Fill trajectory.
        self.trajectory.prompt_ids = initial_prompt_ids
        self.trajectory.response_ids = response_ids
        self.trajectory.response_mask = response_mask
        self.trajectory.response_logprobs = response_logprobs

        return True

    def _clear_trajectory_alignment(self) -> None:
        """
        Clear trajectory alignment data on reconstruction failure.
        """
        self.trajectory.response_ids = []
        self.trajectory.response_mask = []
        self.trajectory.response_logprobs = []

    async def finalize_output(self, request: DataProto) -> DataProto:
        """
        Finalize trajectory and prepare `DataProto` output for training.

        Pre-processes the trajectory (reconstruct, handle alignment failure,
        store mini-swe-specific reward fields), then delegates to
        `super().finalize_output()` for truncation, non_tensor_batch building,
        routing metadata, and reward computation.

        Args:
            request (DataProto): Original request DataProto containing metadata.

        Returns:
            Finalized DataProto with reward.
        """
        alignment_ok = await self.reconstruct_and_validate()
        num_turns = len(self.trajectory.steps)

        if not alignment_ok or not self.trajectory.response_ids:
            pad_token_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
            if not self.trajectory.prompt_ids:
                self.trajectory.prompt_ids = [pad_token_id]
            self.trajectory.response_ids = [pad_token_id]
            self.trajectory.response_mask = [1]
            self.trajectory.response_logprobs = [0.0]
            effective_num_turns = 0
        else:
            effective_num_turns = num_turns

        # Set turn counts so base's `__num_turns__` formula gives the correct value.
        # Base formula: `assistant_turns + user_turns + 1`.
        self.trajectory.assistant_turns = effective_num_turns
        self.trajectory.user_turns = effective_num_turns

        # Store mini-swe-specific fields under `agent_reward_info` (not directly
        # in `extra_info`).  RewardManager merges this into `extra_info` after
        # union, so the reward function sees all fields transparently.
        swe_reward_info = {
            "patch": self.patch,
            "num_turns": effective_num_turns,
            "actual_num_turns": num_turns,
            "alignment_failed": not alignment_ok,
            "alignment_failure_reason": self._alignment_failure_reason,
            # Grader result (populated by agent loop after fresh-container eval,
            # empty dict for toy / simple-test data sources).
            "grader_result": self._grader_result,
            # Emit acc (resolve_rate, 0/1) alongside the shaped score metric
            # so wandb shows both train/score and train/acc (OpenClaw-RL L886).
            "acc": float(bool(self._grader_result.get("resolved", False))),
        }
        request.non_tensor_batch["agent_reward_info"] = np.array([swe_reward_info])

        psrl_logger.info(
            f"[finalize_output] uid={request.non_tensor_batch.get('uid', ['?'])[0]}, "
            f"agent_reward_info={swe_reward_info}, "
            f"non_tensor_batch_keys={list(request.non_tensor_batch.keys())}"
        )

        # Delegate to base: truncation, non_tensor_batch, routing metadata, reward.
        return await super().finalize_output(request)

    # --- White-box protocol stubs (not used in black-box pattern) ---

    def format_chat_completions(self, observation: dict, *, is_init: bool) -> ConversationType:
        """
        Not used. This is a `@whitebox_protocol` method.
        """
        raise NotImplementedError("MiniSWEAgentData implements the black-box protocol.")

    def encode_observation(self, observation: dict, *, is_init: bool) -> tuple[list[int], bool]:
        """
        Not used. This is a `@whitebox_protocol` method.
        """
        raise NotImplementedError("MiniSWEAgentData implements the black-box protocol.")

    def decode_action_from_token_ids(self, token_ids: list[int]) -> None:
        """
        Not used. This is a `@whitebox_protocol` method.
        """
        raise NotImplementedError("MiniSWEAgentData implements the black-box protocol.")
