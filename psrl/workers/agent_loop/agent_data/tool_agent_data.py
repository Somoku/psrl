import json
import logging
import os
from typing import Any

import numpy as np
import ray
from omegaconf import DictConfig
from transformers import AutoTokenizer
from verl import DataProto

from psrl.environments.base import ConversationType
from psrl.environments.tool_env import ToolAction
from psrl.tools.tool_parser.base import ToolParser
from psrl.workers.agent_loop.agent_data.base import AgentData, Step, Trajectory

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@AgentData.register("tool_agent_data")
class ToolAgentData(AgentData[ConversationType, ToolAction]):
    """
    Agent data implementation for tool-based interactions.

    This class manages trajectories where the agent can invoke tools during
    multi-turn conversations. It handles parsing tool calls from model outputs,
    updating conversation history with tool responses, and computing rewards.

    The class supports both synchronous and asynchronous tool execution and
    integrates with chat templates for proper formatting.
    """

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer, **kwargs):
        """Initialize class-level shared resources for all instances.

        This method sets up the tokenizer, maximum turns, and tool parser that
        will be shared across all ToolAgentData instances.

        Args:
            config: Configuration object containing training settings
            tokenizer: Tokenizer for converting between text and tokens
            **kwargs: Additional keyword arguments from configuration
        """
        if cls._class_initialized:
            return
        cls._class_initialized = True

        # Initialize class-level attributes from config
        cls.tokenizer = tokenizer
        cls.max_turns = config.gen_actor_rollout_ref.rollout.multi_turn.max_turns

        # Initialize tool parser for extracting tool calls from model output
        cls.tool_parser = ToolParser.get_tool_parser(config.gen_actor_rollout_ref.rollout.multi_turn.format, tokenizer)

    def __init__(
        self,
        config: DictConfig,
        reward_manager: ray.actor.ActorHandle,
        tokenizer: AutoTokenizer,
        tool_schemas: list[dict[str, Any]],
        **kwargs,
    ):
        """Initialize ToolAgentData instance.

        Args:
            config: Configuration object containing training settings
            reward_manager: Ray actor handle for computing rewards
            tokenizer: Tokenizer for converting between text and tokens
            tool_schemas: List of tool schema dictionaries for chat template
            **kwargs: Additional keyword arguments from configuration
        """
        self.init_class(config=config, tokenizer=tokenizer, **kwargs)
        super().__init__(config, reward_manager, tokenizer)
        self.tool_schemas = tool_schemas
        self.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})

        # Pre-compute system prompt for efficient token generation
        self.system_prompt = tokenizer.apply_chat_template(
            [{}],
            add_generation_prompt=False,
            tokenize=True,
            **self.apply_chat_template_kwargs,
        )

    def reset(self) -> None:
        """Reset the agent data to initial state.

        Clears the current trajectory and prepares for a new episode.
        """
        self.trajectory = Trajectory()

    def init_trajectory(self, request: DataProto) -> None:
        """Initialize a new trajectory from the input request.

        Extracts request_id and parent_id from the request and creates a new
        Trajectory instance.

        Args:
            request: DataProto containing the initial task and metadata.

        Raises:
            AssertionError: If the request contains more than one item.
        """
        assert len(request) == 1, "We only support single request initialization."

        request_id = request.non_tensor_batch.get("uid", 0)[0]
        parent_id = request.non_tensor_batch.get("parent_id", None)
        if parent_id is not None:
            parent_id = parent_id[0]

        self.trajectory = Trajectory(
            request_id=request_id,
            parent_id=parent_id,
        )

    async def update_from_env(
        self,
        observation: ConversationType,
        reward: float | None,
        done: bool,
        info: dict,
        **kwargs,
    ) -> bool:
        """Update agent data from environment feedback.

        Processes the observation returned by the environment (typically a tool
        response message) and updates the trajectory. Converts the observation
        to token IDs and appends them to the response sequence.

        Args:
            observation: List of conversation messages from the environment
            reward: Tool execution reward (if applicable)
            done: Whether the episode has terminated
            info: Additional information from the environment
            **kwargs: Additional keyword arguments

        Returns:
            bool: True if response length limit is reached, False otherwise
        """
        is_init = len(self.trajectory.steps) == 0
        if len(self.trajectory.steps) < self.max_turns:
            # Create a new step with the environment observation
            step = Step(
                chat_completions=[observation],
                observation=observation,
            )
            self.trajectory.steps.append(step)
            self.trajectory.steps[-1].tool_reward = reward
            self.trajectory.steps[-1].done = done
            self.trajectory.steps[-1].info = info

            if observation:
                # Apply chat template to convert observation to token IDs
                if is_init:
                    # Initial prompt includes tool schemas
                    response_ids = self.tokenizer.apply_chat_template(
                        observation,
                        tools=self.tool_schemas,
                        add_generation_prompt=True,
                        tokenize=True,
                        **self.apply_chat_template_kwargs,
                    )
                else:
                    # Subsequent turns don't need tool schemas again
                    response_ids = self.tokenizer.apply_chat_template(
                        observation,
                        add_generation_prompt=True,
                        tokenize=True,
                    )
                    # Remove system prompt prefix from subsequent turns
                    response_ids = response_ids[len(self.system_prompt) :]

                if is_init:
                    # First observation becomes the prompt
                    self.trajectory.prompt_ids = response_ids
                else:
                    # Update trajectory state with new tokens
                    self.trajectory.user_turns += 1
                    self.trajectory.response_ids += response_ids
                    self.trajectory.response_mask += [0] * len(response_ids)  # User tokens masked
                    self.trajectory.response_length += len(response_ids)
                    if self.trajectory.response_logprobs:
                        # Mask logprobs for user tokens (not generated by model)
                        self.trajectory.response_logprobs += [0.0] * len(response_ids)

        # Check if response length limit is reached
        if self.trajectory.response_length >= self.config.gen_actor_rollout_ref.rollout.response_length:
            return True
        else:
            return False

    async def update_from_model_token_ids(self, output: DataProto, **kwargs) -> tuple[ToolAction, bool]:
        """Update agent data from model token ids output.

        Extracts generated token IDs from model output, parses tool calls if present,
        and updates the trajectory with the assistant's response.

        Args:
            output: DataProto containing model-generated token IDs and logprobs
            **kwargs: Additional keyword arguments

        Returns:
            tuple: (tool_calls, done) where tool_calls is a list of tool call
                dictionaries and done indicates if response limit is reached
        """
        self.update_trajectory_state_from_output(output)

        response_ids = output.non_tensor_batch.pop("raw_response_ids", [None])[0]
        rollout_logprobs = output.non_tensor_batch.pop("rollout_log_probs", [None])[0]

        # Update trajectory state with model-generated tokens
        self.trajectory.assistant_turns += 1
        self.trajectory.response_ids += response_ids
        self.trajectory.response_mask += [1] * len(response_ids)  # Use assistant tokens for training
        self.trajectory.response_length += len(response_ids)
        if rollout_logprobs:
            self.trajectory.response_logprobs += rollout_logprobs

        # Parse tool calls from the model response
        try:
            _, tool_calls = self.tool_parser.extract_tool_calls(response_ids)
            tool_calls_dict = [
                {
                    "type": "function",
                    "function": tool_call.to_dict(),
                }
                for tool_call in tool_calls
            ]
        except Exception as e:
            psrl_logger.error(f"Failed to parse tool calls: {e}")
            tool_calls_dict = []

        # Decode response text and create assistant message
        assistant_content = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        assistant_message = {"role": "assistant", "content": assistant_content}

        # Ensure tool call arguments are JSON strings (required by chat template)
        if len(tool_calls_dict) > 0:
            for i, call in enumerate(tool_calls_dict):
                if isinstance(call.get("function", {}).get("arguments"), dict):
                    tool_calls_dict[i]["function"]["arguments"] = json.dumps(call["function"]["arguments"])

        # Update current step with assistant response
        self.trajectory.steps[-1].chat_completions.append(assistant_message)
        self.trajectory.steps[-1].model_response = assistant_content
        self.trajectory.steps[-1].action = tool_calls_dict

        # Compute step reward if using step-level reward mode
        if self.config.gen_actor_rollout_ref.rollout.agent.traj_reward_mode == "step":
            output.non_tensor_batch["__num_turns__"] = np.array(
                [self.trajectory.assistant_turns + self.trajectory.user_turns + 1], dtype=np.int32
            )
            self.trajectory.steps[-1].model_reward = await self.reward_manager.compute_score.remote(output)

        # Check if response length limit is reached
        if self.trajectory.response_length >= self.config.gen_actor_rollout_ref.rollout.response_length:
            return [], True
        else:
            return tool_calls_dict, False
