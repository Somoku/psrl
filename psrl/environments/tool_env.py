import asyncio
import json
import logging
import os
from typing import Any

import ray
from omegaconf import DictConfig
from verl import DataProto

from psrl.environments.base import ConversationType, Environment, EnvStepOutput
from psrl.tools.base import ToolGroup, initialize_tools_from_config
from psrl.utils.logger import FileOnlyHandler

ToolAction = list[dict] | dict

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@Environment.register("tool_env")
class ToolEnvironment(Environment[ConversationType, ToolAction]):
    """
    Environment for tool-based agent interactions.

    This environment manages interactions where an agent can invoke tools based
    on model-generated tool calls. It handles tool execution, response formatting,
    and turn management for multi-turn tool-using conversations.

    Type Parameters:
        ConversationType: List of conversation messages (observations)
        ToolAction: List of tool call dictionaries or single dict (actions)
    """

    @classmethod
    def init_class(cls, config: DictConfig, **kwargs) -> None:
        """Perform heavy initialization work shared across all instances.

        This method is called only once per class to avoid redundant initialization.

        Args:
            config: Configuration object containing training settings
            **kwargs: Additional keyword arguments from configuration
        """
        if cls._class_initialized:
            return
        cls._class_initialized = True

        cls.max_tool_response_length = config.gen_actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.gen_actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        cls.max_parallel_calls = config.gen_actor_rollout_ref.rollout.multi_turn.max_parallel_calls

        # Initialize tools from configuration file
        tool_config_path = config.gen_actor_rollout_ref.rollout.multi_turn.tool_config_path
        tools = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = ToolGroup(tools=tools)

        # Build a class-level logger for tool calls.
        log_prefix = "ToolEnvironment"
        psrl_logger.propagate = False
        psrl_logger.addHandler(FileOnlyHandler(config.psrl.logging_path, log_prefix))
        psrl_logger.info("ToolEnvironment logger initialized for tool calls.")

    def __init__(
        self,
        config: DictConfig,
        reward_manager: ray.actor.ActorHandle,
        max_turns: int,
    ):
        """
        Initialize the ToolEnvironment.

        Args:
            config: Configuration object containing training settings
            reward_manager: Ray actor handle for computing rewards
            max_turns: Maximum number of turns allowed in an episode
            **kwargs: Additional keyword arguments
        """
        super().__init__(config, reward_manager)

        self.max_turns = max_turns
        self.num_turn = 0

    async def reset(self, task: DataProto, **kwargs) -> tuple[ConversationType, dict]:
        """Reset the environment to an initial state.

        Extracts the initial task prompt from the input DataProto and prepares
        the environment for a new episode.

        Args:
            task: DataProto containing the initial task/prompt
            **kwargs: Additional keyword arguments

        Returns:
            tuple: (initial_observation, info_dict) where initial_observation is
                   the task prompt and info_dict contains additional metadata

        Raises:
            AssertionError: If task doesn't contain exactly one item or lacks
                          'raw_prompt' in non_tensor_batch
        """
        info = {}
        self.num_turn = 0

        self.task = task
        assert len(task) == 1, "We only support single initial prompt in ToolEnvironment."
        assert "raw_prompt" in task.non_tensor_batch, (
            "For ReTool recipe, task must contain 'raw_prompt' in non_tensor_batch"
        )
        initial_observation = task.non_tensor_batch["raw_prompt"].tolist()[0]

        return initial_observation, info

    async def step(self, action: ToolAction) -> EnvStepOutput:
        """Execute a step in the environment by calling tools.

        Takes a tool action (one or more tool calls) from the agent, executes
        the requested tools asynchronously, and returns the results as observations.

        Args:
            action: Tool action - either a single tool call dict or list of tool calls

        Returns:
            EnvStepOutput: Dictionary containing:
                - observation: List of tool response messages
                - reward: Cumulative reward from all tool executions
                - done: Whether the episode should terminate (True if no tools called)
                - info: Additional metadata (empty dict)

        Note:
            - Empty action list terminates the episode (done=True)
            - Exceeding max_turns returns empty observation (done=False)
            - Tool calls are executed concurrently using asyncio.gather
        """
        if isinstance(action, dict):
            action = [action]

        reward = 0.0

        # Empty action indicates agent wants to stop
        if len(action) == 0:
            return EnvStepOutput(
                observation=[],
                reward=reward,
                done=True,
                info={},
            )

        self.num_turn += 1

        # Check if maximum turns exceeded
        if self.num_turn > self.max_turns:
            return EnvStepOutput(
                observation=[],
                reward=reward,
                done=False,
                info={},
            )

        # Execute tool calls with an explicit concurrency cap to control latency spikes.
        sem = asyncio.Semaphore(self.max_parallel_calls)

        async def _guarded_call(tc: dict):
            async with sem:
                return await self._call_tool(tc)

        tool_outputs = await asyncio.gather(*[_guarded_call(tc) for tc in action])

        # Collect observations and rewards from tool executions
        next_observation = []
        for tool_message, tool_reward in tool_outputs:
            next_observation.append(tool_message)
            if tool_reward:
                reward += tool_reward

        return EnvStepOutput(
            observation=next_observation,
            reward=reward,
            done=False,
            info={},
        )

    async def _call_tool(self, tool_call: dict) -> dict[str, str]:
        """Call a single tool and format the response.

        Executes the specified tool with the provided arguments, handles errors,
        and truncates responses if needed based on configuration.

        Args:
            tool_call: Dictionary containing tool call information with structure:
                      {"function": {"name": str, "arguments": str (JSON)}}

        Returns:
            tuple: (tool_message, tool_reward) where:
                - tool_message is a dict with role "tool" and content string
                - tool_reward is an optional float reward from the tool

        Note:
            - Catches and logs all exceptions during tool execution
            - Truncates long responses based on max_tool_response_length
            - Truncation side controlled by tool_response_truncate_side config
        """
        try:
            # Extract tool name and arguments from tool call
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])

            # Retrieve and execute the tool
            tool = self.tools.get_tool(tool_name)
            assert tool is not None, f"Tool {tool_name} not found in ToolGroup."
            tool_output = await tool(**tool_args)
            tool_response_text = tool_output.output["content"]
            tool_reward = tool_output.output.get("score", None)
            tool_metadata = tool_output.output.get("metadata", None)
        except Exception as e:
            psrl_logger.warning(f"Error when executing tool: {e}")
            return (
                {"role": "tool", "content": f"Error when executing tool: {e}"},
                0.0,
            )

        # Truncate tool response if it exceeds maximum length
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                # Center truncation
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        tool_message = {"role": "tool", "content": tool_response_text, "metadata": tool_metadata}

        # Log each tool invocation with its input arguments and (possibly truncated) output.
        psrl_logger.info(
            f"Tool call - name: {tool_name}, args: {tool_args}, "
            f"response: {tool_response_text}, reward: {tool_reward}, metadata: {tool_metadata}"
        )
        return tool_message, tool_reward

    async def close(self) -> None:
        """Cleans up resources used by the environment.

        Currently a no-op as tools don't require explicit cleanup.
        """
        return

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get the JSON schemas for all available tools.

        Returns:
            list: List of tool schema dictionaries in OpenAI function calling format
        """
        return self.tools.json

    @property
    def state(self) -> dict:
        """Get the current state of the environment.

        Returns:
            dict: Dictionary containing:
                - num_turn: Current turn number
                - max_turns: Maximum allowed turns
                - task: Current task DataProto
        """
        return {
            "num_turn": self.num_turn,
            "max_turns": self.max_turns,
            "task": self.task,
        }
