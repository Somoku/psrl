"""
ToolAgentData — agent data for tool-calling interactions.

Extends `ConversationAgentData` with two overrides:
- `encode_observation` injects `tools=tool_schemas` on the initial turn.
- `decode_action_from_token_ids` parses tool calls from generated token IDs.

All other behaviour (trajectory building, masking, `update_from_env`,
`update_from_model_token_ids`, `init_trajectory`, `reset`) is inherited
from `ConversationAgentData` unchanged.
"""

import json
import logging
import os

import ray
from omegaconf import DictConfig
from transformers import AutoTokenizer

from psrl.environments.base import ConversationType, Environment
from psrl.environments.tool_env import ToolAction
from psrl.tools.tool_parser.base import ToolParser
from psrl.workers.agent_loop.agent_data.base import AgentData
from psrl.workers.agent_loop.agent_data.conversation_agent_data import ConversationAgentData

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@AgentData.register("tool_agent_data")
class ToolAgentData(ConversationAgentData):
    """
    Agent data for tool-calling interactions.

    Overrides `encode_observation` to inject `tools=tool_schemas` on the first
    turn so the model receives the tool-schema section in its system prompt.
    Overrides `decode_action_from_token_ids` to parse tool calls via the
    configured `ToolParser`.

    Both `update_from_env` and `update_from_model_token_ids` are inherited
    from `ConversationAgentData` and require no modification.
    """

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer, **kwargs):
        """
        Initialize class-level shared resources (once per class).

        Args:
            config (DictConfig): Trainer configuration.
            tokenizer (AutoTokenizer): Tokenizer shared across instances.
            **kwargs: Additional keyword arguments.
        """
        if cls._class_initialized:
            return
        cls._class_initialized = True
        cls.tokenizer = tokenizer
        cls.tool_parser = ToolParser.get_tool_parser(
            config.gen_actor_rollout_ref.rollout.multi_turn.format, tokenizer,
        )

    def __init__(
        self,
        config: DictConfig,
        reward_manager: ray.actor.ActorHandle,
        tokenizer: AutoTokenizer,
        env: Environment,
        **kwargs,
    ):
        """
        Initialize ToolAgentData.

        Args:
            config (DictConfig): Trainer configuration.
            reward_manager: Ray actor handle for computing rewards.
            tokenizer (AutoTokenizer): Tokenizer for text-to-token conversion.
            env (Environment): Environment instance; must expose `get_tool_schemas`.
            **kwargs: Additional keyword arguments.
        """
        assert hasattr(env, "get_tool_schemas"), (
            "Environment must implement get_tool_schemas method."
        )
        self.init_class(config=config, tokenizer=tokenizer, **kwargs)
        super().__init__(config, reward_manager, tokenizer, env)
        self.tool_schemas = self.env.get_tool_schemas()

    def encode_observation(
        self,
        observation: ConversationType,
        *,
        is_init: bool,
    ) -> tuple[list[int], bool]:
        """
        Encode the initial observation with tool schemas injected.

        On `is_init=True`, calls `_apply_chat_template_ids` with
        `tools=self.tool_schemas` so the tokenizer inserts the tool-schema
        block into the system prompt. On subsequent turns, delegates to the
        inherited fixed-base delta (no tool schemas needed).

        Args:
            observation (ConversationType): List of OpenAI-format message dicts.
            is_init (bool): Whether this is the first turn.

        Returns:
            tuple[list[int], bool]: (token_ids, is_prompt).
        """
        if is_init:
            if not observation:
                return [], True
            return (
                self._apply_chat_template_ids(
                    observation,
                    add_generation_prompt=True,
                    tools=self.tool_schemas,
                ),
                True,
            )
        return super().encode_observation(observation, is_init=is_init)

    def decode_action_from_token_ids(self, token_ids: list[int]) -> ToolAction:
        """
        Parse tool calls from generated token IDs.

        Args:
            token_ids (list[int]): Generated response token IDs.

        Returns:
            ToolAction: List of OpenAI-function-call style dicts, possibly empty.
        """
        _, tool_calls = self.tool_parser.extract_tool_calls(token_ids)
        result = [
            {"type": "function", "function": tc.to_dict()}
            for tc in tool_calls
        ]
        for item in result:
            args = item.get("function", {}).get("arguments")
            if isinstance(args, dict):
                item["function"]["arguments"] = json.dumps(args)
        return result
