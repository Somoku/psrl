"""
mini-SWE-Agent AgentData for PSRL.

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

    def record_turn(
        self,
        turn_index: int,
        messages: list[dict],
        prompt_ids: list[int],
        response_ids: list[int],
        response_text: str,
        response_logprobs: list[float],
    ) -> None:
        """
        Record one interaction turn into the trajectory.

        Creates a new `Step` and stores per-turn metadata in `Step.info["turn_record"]`
        for later trajectory reconstruction.

        Args:
            turn_index: Zero-based turn index.
            messages: Original OpenAI-format messages for this turn.
            prompt_ids: Tokenized prompt IDs.
            response_ids: Generated response token IDs.
            response_text: Decoded response text.
            response_logprobs: Per-token log probabilities.
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

    def set_patch(self, patch: str | None) -> None:
        """
        Store the extracted patch for reward computation.

        Args:
            patch: Generated patch string, or None.
        """
        self.patch = patch

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

        Calls `reconstruct_and_validate()`. If reconstruction fails, sets reward=0.
        Truncates to configured response_length. Computes reward via reward_manager.

        Args:
            request: Original request DataProto containing metadata.

        Returns:
            Finalized DataProto with reward.
        """
        alignment_ok = await self.reconstruct_and_validate()

        num_turns = len(self.trajectory.steps)
        actual_num_turns = num_turns

        if not alignment_ok or not self.trajectory.response_ids:
            # Alignment failed or no response: produce minimal valid output.
            pad_token_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
            prompt_ids = self.trajectory.prompt_ids if self.trajectory.prompt_ids else [pad_token_id]
            response_ids = [pad_token_id]
            response_mask = [1]
            response_logprobs = [0.0]
            effective_num_turns = 0
        else:
            max_response_length = self.config.gen_actor_rollout_ref.rollout.response_length
            prompt_ids = self.trajectory.prompt_ids
            response_ids = self.trajectory.response_ids[:max_response_length]
            response_mask = self.trajectory.response_mask[:max_response_length]
            response_logprobs = self.trajectory.response_logprobs[:max_response_length]
            effective_num_turns = num_turns

        # Guarantee at least one mask=1 token.
        if not any(m == 1 for m in response_mask):
            pad_token_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
            response_ids.append(pad_token_id)
            response_mask.append(1)
            response_logprobs.append(0.0)

        # Pad logprobs to match response length.
        if len(response_logprobs) < len(response_ids):
            response_logprobs.extend([0.0] * (len(response_ids) - len(response_logprobs)))

        # Build output DataProto.
        non_tensor_batch = request.non_tensor_batch.copy()
        non_tensor_batch["raw_prompt_ids"] = np.array([prompt_ids])
        non_tensor_batch["raw_response_ids"] = np.array([response_ids])
        non_tensor_batch["response_mask"] = np.array([response_mask])
        non_tensor_batch["rollout_log_probs"] = np.array([response_logprobs])
        non_tensor_batch["__num_turns__"] = np.array([effective_num_turns * 2 + 1], dtype=np.int32)

        # Store extra fields for reward computation.
        extra_info_raw = non_tensor_batch.get("extra_info", [{}])[0]
        if isinstance(extra_info_raw, dict):
            extra_info_raw["patch"] = self.patch
            extra_info_raw["num_turns"] = effective_num_turns
            extra_info_raw["actual_num_turns"] = actual_num_turns
            extra_info_raw["alignment_failed"] = not alignment_ok
            extra_info_raw["alignment_failure_reason"] = self._alignment_failure_reason

        data = DataProto(non_tensor_batch=non_tensor_batch, meta_info=request.meta_info)

        # Compute reward via reward_manager.
        reward_result = await self.reward_manager.compute_score.remote(data)
        if not self.config.reward_model.launch_reward_fn_async:
            data = self._post_process_and_merge_reward(reward_result, data)

        return data

    # --- Abstract methods not used in subprocess-proxy pattern ---

    def format_chat_completions(self, observation: dict, *, is_init: bool) -> ConversationType:
        """
        Not used in the subprocess-proxy pattern.
        """
        raise NotImplementedError(
            "MiniSWEAgentData uses encode_messages() and record_turn() instead "
            "of the traditional env-step-driven methods."
        )

    def encode_observation(self, observation: dict, *, is_init: bool) -> tuple[list[int], bool]:
        """
        Not used in the subprocess-proxy pattern.
        """
        raise NotImplementedError(
            "MiniSWEAgentData uses encode_messages() and record_turn() instead "
            "of the traditional env-step-driven methods."
        )

    def decode_action_from_token_ids(self, token_ids: list[int]) -> None:
        """
        Not used in the subprocess-proxy pattern.
        """
        raise NotImplementedError(
            "MiniSWEAgentData uses encode_messages() and record_turn() instead "
            "of the traditional env-step-driven methods."
        )
