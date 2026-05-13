"""
ConversationAgentData — base class for agents with chat-message observations.

Provides default implementations of `update_from_env`, `update_from_model_token_ids`,
and `encode_observation` that work for any agent whose environment sends and receives
OpenAI-format chat messages. Concrete subclasses only need to override
`decode_action_from_token_ids` (and optionally `encode_observation` to inject
extra template arguments such as `tools=tool_schemas`).

Extension recipe for a new agent type::

    @AgentData.register("search_agent_data")
    class SearchAgentData(ConversationAgentData):
        _SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)

        def decode_action_from_token_ids(self, token_ids):
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            m = self._SEARCH_RE.search(text)
            return {"type": "search", "query": m.group(1).strip()} if m else None
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


# Fixed base used for `_compute_user_delta`: a minimal three-message
# conversation that no chat template position-strips. The dummy assistant
# carries no `<think>` tags, so Qwen3's `last_query_index` branch has
# nothing to strip; the explicit system message blocks Qwen2.5 from
# auto-injecting its default system prompt.
_FIXED_BASE_FOR_USER_DELTA: list[dict] = [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "I am a user."},
    {"role": "assistant", "content": "I am an assistant."},
]


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


class ConversationAgentData(AgentData[ConversationType, object]):
    """
    Agent data base class for chat-message-based interactions.

    Provides complete default implementations of `update_from_env` and
    `update_from_model_token_ids`. Concrete subclasses only need to:

    - Override `decode_action_from_token_ids` to parse agent-specific action
      formats (tool calls, search queries, etc.).  Chat-only agents (e.g.
      `MiniSWEAgentData`) can leave the default `raise NotImplementedError`,
      which causes `update_from_model_token_ids` to return `action=None`.
    - Optionally override `encode_observation` to inject extra template kwargs
      (e.g. `tools=tool_schemas` for the initial turn in `ToolAgentData`).

    The user-delta tokenization for subsequent turns uses a fixed-base anchor
    (`_compute_user_delta`) that is immune to position-dependent chat-template
    behavior such as Qwen3's `<think>` stripping from non-final assistant
    messages.
    """

    def __init__(
        self,
        config: DictConfig,
        reward_manager: ray.actor.ActorHandle,
        tokenizer: AutoTokenizer,
        env: Environment,
        **kwargs,
    ):
        super().__init__(config, reward_manager, tokenizer, env)
        self.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        self._init_fixed_base_prefix()

    # --- Chat-template helpers ---

    def _apply_chat_template_ids(
        self,
        messages: list[dict],
        *,
        add_generation_prompt: bool,
        **extra_template_kwargs,
    ) -> list[int]:
        """
        Apply the active chat template and return token IDs as `list[int]`.

        Normalises both `list[int]` and `BatchEncoding` return types so callers
        always receive a plain list. Accepts per-call extra keyword arguments
        (e.g. `tools=tool_schemas`) merged on top of `apply_chat_template_kwargs`.

        Args:
            messages (list[dict]): OpenAI-format message dicts.
            add_generation_prompt (bool): Whether to append the generation prompt.
            **extra_template_kwargs: Extra kwargs forwarded to `apply_chat_template`
                (e.g. `tools=...` for the initial `ToolAgentData` turn).

        Returns:
            list[int]: Token IDs.
        """
        # Safety guard: truncate excessively long message content to prevent
        # tokenizer segfaults (Rust regex/normalizer crashes on very large strings).
        # 512k chars ≈ 128k tokens — well above any reasonable context window.
        _MAX_CONTENT_CHARS = 512_000
        safe_messages = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > _MAX_CONTENT_CHARS:
                content = content[:_MAX_CONTENT_CHARS] + "\n... [truncated by safety guard]"
                safe_messages.append({**msg, "content": content})
            else:
                safe_messages.append(msg)

        raw = self.tokenizer.apply_chat_template(
            safe_messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            **self.apply_chat_template_kwargs,
            **extra_template_kwargs,
        )
        return raw if isinstance(raw, list) else list(raw["input_ids"])

    def _init_fixed_base_prefix(self) -> None:
        """
        Precompute `_fixed_base_prefix_ids` and `_turn_end_id` for `_compute_user_delta`.

        Trims a single trailing `\\n` token from the fixed-base encoding so
        that user-delta tokens start with `\\n<|im_start|>...`, splicing cleanly
        after a `response_ids_K` that ends with `<|im_end|>`.

        Also records `_turn_end_id` — the last token of the prefix (before the
        stripped `\\n`), which is the per-turn stop token for the active chat
        template (e.g., `<|im_end|>` for Qwen-style models). This is used in
        `update_from_model_token_ids` to detect responses truncated by max_tokens.
        """
        fixed_base_ids = self._apply_chat_template_ids(
            _FIXED_BASE_FOR_USER_DELTA, add_generation_prompt=False,
        )
        nl_tokens = self.tokenizer.tokenize("\n")
        nl_id: int | None = (
            self.tokenizer.convert_tokens_to_ids(nl_tokens[0])
            if len(nl_tokens) == 1
            else None
        )
        self._newline_id: int | None = nl_id
        if nl_id is not None and len(fixed_base_ids) > 0 and fixed_base_ids[-1] == nl_id:
            self._fixed_base_prefix_ids: list[int] = fixed_base_ids[:-1]
        else:
            self._fixed_base_prefix_ids = list(fixed_base_ids)
            psrl_logger.warning(
                f"Fixed-base prefix did not end with a single newline token "
                f"(newline_id={nl_id!r}, "
                f"last_id={fixed_base_ids[-1] if fixed_base_ids else None!r}). "
                f"User-delta tokens may not splice cleanly after response tokens."
            )
        # The per-turn stop token: last token of the prefix (e.g., <|im_end|> = 151645 for
        # Qwen). Responses truncated by max_tokens will not end with this token; the check
        # in `update_from_model_token_ids` uses this to insert the missing separator.
        self._turn_end_id: int | None = (
            self._fixed_base_prefix_ids[-1] if self._fixed_base_prefix_ids else None
        )

    def _compute_user_delta(self, user_message: dict) -> list[int]:
        """
        Compute the token delta for a single new user/tool message via fixed-base anchoring.

        The returned tokens correspond to
        ``\\n<|im_start|>user\\n{content}<|im_end|>\\n<|im_start|>assistant\\n``
        (for standard Qwen templates), suitable for splicing right after a
        recorded `response_ids_K` that ends with `<|im_end|>`. Anchoring on
        `_FIXED_BASE_FOR_USER_DELTA` avoids position-dependent template behavior
        (e.g. Qwen3's `<think>` stripping from non-final assistant messages).

        Args:
            user_message (dict): One OpenAI-format message dict.

        Returns:
            list[int]: Delta token IDs (always non-empty in practice).
        """
        normalized = normalize_openai_messages([user_message])[0]
        full_ids = self._apply_chat_template_ids(
            [*_FIXED_BASE_FOR_USER_DELTA, normalized], add_generation_prompt=True,
        )
        return full_ids[len(self._fixed_base_prefix_ids):]

    # --- Observation encoding ---

    def format_chat_completions(
        self,
        observation: ConversationType,
        *,
        is_init: bool,
    ) -> ConversationType:
        """Return the observation as-is; it is already in `ConversationType` format."""
        return list(observation)

    def encode_observation(
        self,
        observation: ConversationType,
        *,
        is_init: bool,
    ) -> tuple[list[int], bool]:
        """
        Encode an environment observation into token IDs.

        On the initial turn (`is_init=True`), encodes the full observation
        with the generation prompt appended and returns `is_prompt=True`.
        On subsequent turns, encodes only the last message via
        `_compute_user_delta` and returns `is_prompt=False`.

        Subclasses may override the `is_init=True` branch to inject extra
        template kwargs (e.g. `tools=tool_schemas` in `ToolAgentData`).

        Args:
            observation (ConversationType): List of OpenAI-format message dicts.
            is_init (bool): Whether this is the first turn of the trajectory.

        Returns:
            tuple[list[int], bool]: (token_ids, is_prompt).
        """
        if not observation:
            return [], is_init
        if is_init:
            return (
                self._apply_chat_template_ids(observation, add_generation_prompt=True),
                True,
            )
        return self._compute_user_delta(observation[-1]), False

    # --- Default protocol implementations ---

    async def update_from_env(
        self,
        observation: ConversationType,
        reward: float | None,
        done: bool,
        info: dict,
        **kwargs,
    ) -> bool:
        """
        Process one environment observation and update the trajectory.

        On the initial turn this encodes the full prompt and stores it in
        `trajectory.prompt_ids` (via `append_prompt_ids`). On subsequent turns
        it encodes only the new user/tool message via fixed-base delta and
        appends to `trajectory.response_ids` with mask=0 (via `append_user_tokens`).

        Args:
            observation (ConversationType): New messages from the environment.
            reward (float | None): Reward for this step (if applicable).
            done (bool): Whether the episode has terminated.
            info (dict): Extra metadata from the environment.

        Returns:
            bool: True if the response-length budget is exceeded.
        """
        is_init = len(self.trajectory.steps) == 0
        if len(self.trajectory.steps) < self._max_turns():
            step = self.start_step(
                observation=observation, reward=reward, done=done, info=info,
            )
            step.chat_completions = self.format_chat_completions(
                observation, is_init=is_init,
            )
            token_ids, is_prompt = self.encode_observation(observation, is_init=is_init)
            if token_ids:
                if is_prompt:
                    self.append_prompt_ids(token_ids)
                else:
                    self.append_user_tokens(token_ids)
        return self.trajectory.response_length >= (
            self.config.gen_actor_rollout_ref.rollout.response_length
        )

    async def update_from_model_token_ids(
        self,
        output: DataProto,
        **kwargs,
    ) -> tuple[object, bool]:
        """
        Process model-generated token IDs and update the trajectory.

        Appends the generated tokens with mask=1 (`append_assistant_tokens`),
        decodes them into text, and attempts to parse an action via
        `decode_action_from_token_ids` (`NotImplementedError` is treated as
        no action, suitable for chat-only agents). Computes step-level reward
        when `traj_reward_mode == "step"`.

        Args:
            output (DataProto): Model output containing raw response IDs and
                optional log-probabilities.

        Returns:
            tuple[object, bool]: (action, overlong) where `action` is the parsed
                action (None for chat-only agents) and `overlong` signals whether
                the response-length budget is now exceeded.
        """
        if len(self.trajectory.steps) == 0:
            self.start_step(observation=[], reward=None, done=False, info={})

        self.update_trajectory_state_from_output(output)

        response_ids = output.non_tensor_batch.pop("raw_response_ids", [None])[0]
        rollout_logprobs = output.non_tensor_batch.pop("rollout_log_probs", [None])[0]
        self.append_assistant_tokens(response_ids, logprobs=rollout_logprobs)

        # When vLLM hits max_tokens, generation stops without emitting the turn-end token
        # (e.g., <|im_end|> for Qwen). Without it, `_compute_user_delta`'s leading \n
        # splices directly after a content token, producing a malformed context for both
        # generation and training. Insert the missing token as a non-trained template token.
        if (
            response_ids
            and self._turn_end_id is not None
            and response_ids[-1] != self._turn_end_id
        ):
            self.append_user_tokens([self._turn_end_id])

        assistant_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        self.add_step_chat_message({"role": "assistant", "content": assistant_text})
        self.set_step_model_response(assistant_text)

        try:
            action = self.decode_action_from_token_ids(response_ids)
        except NotImplementedError:
            action = None
        except Exception as exc:
            psrl_logger.error(f"Failed to decode action from token ids: {exc!r}.")
            action = []
        if action is not None:
            self.set_step_action(action)

        if self.config.gen_actor_rollout_ref.rollout.agent.traj_reward_mode == "step":
            output.non_tensor_batch["__num_turns__"] = np.array(
                [self.trajectory.assistant_turns + self.trajectory.user_turns + 1],
                dtype=np.int32,
            )
            self.get_current_step().model_reward = await self.reward_manager.compute_score.remote(output)

        overlong = self.trajectory.response_length >= (
            self.config.gen_actor_rollout_ref.rollout.response_length
        )
        return action, overlong

    # --- Abstract hook subclasses must implement ---

    def decode_action_from_token_ids(self, token_ids: list[int]) -> object:
        """
        Parse an action from the model's generated token IDs.

        Chat-only agents (e.g. `MiniSWEAgentData`) should leave this raising
        `NotImplementedError` — `update_from_model_token_ids` will treat the
        action as None. Tool-calling or search agents should return a structured
        action (list of tool-call dicts for `ToolAgentData`, etc.).

        Args:
            token_ids (list[int]): Generated response token IDs.

        Returns:
            object: Parsed action; type depends on the subclass.
        """
        raise NotImplementedError

    # --- Lifecycle ---

    def _max_turns(self) -> int:
        """
        Return the per-trajectory turn cap from config, or a large fallback.

        Reads `gen_actor_rollout_ref.rollout.multi_turn.max_turns` when present.
        For agents whose loops control turn count externally (e.g. `MiniSWEAgentData`
        driven by `mini_swe_agent_loop`), the fallback of 10 000 is effectively
        unlimited and the outer loop is the actual gate.
        """
        multi_turn = getattr(
            self.config.gen_actor_rollout_ref.rollout, "multi_turn", None,
        )
        if multi_turn is not None and hasattr(multi_turn, "max_turns"):
            return multi_turn.max_turns
        return 10_000

    def reset(self) -> None:
        """Reset the trajectory to initial state."""
        self.trajectory = Trajectory()

    def init_trajectory(self, request: DataProto) -> None:
        """
        Initialize a new trajectory from the input request.

        Args:
            request (DataProto): Input request containing uid and parent_id.
        """
        assert len(request) == 1, "Only single request initialization is supported."
        request_id = request.non_tensor_batch.get("uid", [0])[0]
        parent_id_arr = request.non_tensor_batch.get("parent_id", None)
        parent_id = parent_id_arr[0] if parent_id_arr is not None else None
        self.trajectory = Trajectory(request_id=request_id, parent_id=parent_id)
