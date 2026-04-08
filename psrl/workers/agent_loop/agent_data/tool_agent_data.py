import json
import logging
import os

import ray
from omegaconf import DictConfig
from transformers import AutoTokenizer
from verl import DataProto

from psrl.environments.base import ConversationType, Environment
from psrl.environments.tool_env import ToolAction
from psrl.tools.tool_parser.base import ToolParser
from psrl.workers.agent_loop.agent_data.base import AgentData, Trajectory
from psrl.workers.gen_dplb.utils import TokenOutput

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
        env: Environment,
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
        assert hasattr(env, "get_tool_schemas"), "Environment must implement get_tool_schemas method."

        self.init_class(config=config, tokenizer=tokenizer, **kwargs)
        super().__init__(config, reward_manager, tokenizer, env)
        self.tool_schemas = self.env.get_tool_schemas()
        self.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})

        # Pre-compute system prompt token ids for incremental turn stripping.
        # This mirrors AgentLoopBase._get_system_prompt_ids() but lives here so
        # AgentData subclasses don't need a reference to the loop.
        self._system_prompt_ids: list[int] = tokenizer.apply_chat_template(
            [{}],
            add_generation_prompt=False,
            tokenize=True,
            **self.apply_chat_template_kwargs,
        )

    async def encode_observation(self, observation: ConversationType, *, is_init: bool) -> tuple[list[int], bool]:
        """Encode tool-env conversation messages into token ids (async).

        For the first observation, we include tool schemas and treat the result as prompt.
        For subsequent observations, we drop the system prompt prefix and treat the result
        as user-side tokens (masked out for training).

        The tokenizer call is offloaded to the default executor so the async event
        loop is never blocked.
        """
        import asyncio

        if not observation:
            return [], is_init

        loop = asyncio.get_running_loop()

        if is_init:
            kwargs = dict(self.apply_chat_template_kwargs)

            def _tokenize_init():
                return self.tokenizer.apply_chat_template(
                    observation,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=True,
                    **kwargs,
                )

            token_ids: list[int] = await loop.run_in_executor(None, _tokenize_init)
            return token_ids, True

        # Incremental turn: re-encode without tools, then strip system-prompt prefix.
        def _tokenize_incremental():
            return self.tokenizer.apply_chat_template(
                observation,
                add_generation_prompt=True,
                tokenize=True,
            )

        token_ids = await loop.run_in_executor(None, _tokenize_incremental)
        token_ids = token_ids[len(self._system_prompt_ids) :]
        return token_ids, False

    def format_chat_completions(self, observation: ConversationType, *, is_init: bool) -> ConversationType:
        """ToolEnvironment already uses ConversationType as observation, so return as-is."""
        return observation

    def decode_action_from_token_ids(self, token_ids: list[int]) -> ToolAction:
        """Decode model generated token ids into ToolAction.

        Returns a list of OpenAI-function-call style dicts:
        [{"type": "function", "function": {"name": ..., "arguments": "..."}}]
        """
        _, tool_calls = self.tool_parser.extract_tool_calls(token_ids)
        tool_calls_dict = [
            {
                "type": "function",
                "function": tool_call.to_dict(),
            }
            for tool_call in tool_calls
        ]
        # Ensure tool call arguments are JSON strings (required by chat template)
        for i, call in enumerate(tool_calls_dict):
            if isinstance(call.get("function", {}).get("arguments"), dict):
                tool_calls_dict[i]["function"]["arguments"] = json.dumps(call["function"]["arguments"])
        return tool_calls_dict

    def reset(self) -> None:
        """Reset the agent data to initial state.

        Clears the current trajectory and prepares for a new episode.
        """
        self.trajectory = Trajectory()

    def init_trajectory(self, request: DataProto) -> None:
        """Initialize a new trajectory from the input request.

        Extracts request_id, parent_id, and per-sample tools_kwargs from the
        request and creates a new Trajectory instance.

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

        # Store per-sample tools_kwargs in trajectory.extra_fields so that
        # ToolEnvironment can access them during tool execution.
        tools_kwargs_arr = request.non_tensor_batch.get("tools_kwargs", None)
        if tools_kwargs_arr is not None:
            self.trajectory.extra_fields["tools_kwargs"] = tools_kwargs_arr[0] or {}
        else:
            self.trajectory.extra_fields["tools_kwargs"] = {}

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
            # Create a new step explicitly (avoids implicit steps[-1] contract).
            step = self.start_step(observation=observation, reward=reward, done=done, info=info)

            # Fill normalized ConversationType for logging/debugging.
            step.chat_completions = self.format_chat_completions(observation, is_init=is_init)

            token_ids, is_prompt = await self.encode_observation(observation, is_init=is_init)
            if token_ids:
                if is_prompt:
                    self.append_prompt_ids(token_ids)
                else:
                    self.append_user_tokens(token_ids)

        # Accumulate non-None tool rewards from env steps for downstream use.
        if not is_init and reward is not None:
            self.trajectory.tool_rewards.append(float(reward))

        # Check if response length limit is reached
        return self.trajectory.response_length >= self.config.gen_actor_rollout_ref.rollout.response_length

    async def update_from_model_token_ids(self, output: TokenOutput, **kwargs) -> tuple[ToolAction, bool]:
        """Update agent data from model token ids output.

        Extracts generated token IDs from model output, parses tool calls if present,
        and updates the trajectory with the assistant's response.

        Args:
            output: TokenOutput containing model-generated token IDs and logprobs
            **kwargs: Additional keyword arguments

        Returns:
            tuple: (tool_calls, done) where tool_calls is a list of tool call
                dictionaries and done indicates if response limit is reached
        """
        # Ensure we have a step to attach model output to.
        if len(self.trajectory.steps) == 0:
            # This should not happen in MultiTurnAgentLoop, but keep a safe fallback
            # for custom loops or unexpected call orders.
            self.start_step(observation=[], reward=None, done=False, info={})

        self.update_trajectory_state_from_output(output)

        response_ids = output.token_ids
        rollout_logprobs = output.log_probs
        routed_experts = output.routed_experts.tolist()

        # Update trajectory state with model-generated tokens
        self.append_assistant_tokens(
            response_ids,
            logprobs=rollout_logprobs,
            routed_experts=routed_experts,
        )

        # Decode response text and create assistant message
        assistant_content = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        assistant_message = {"role": "assistant", "content": assistant_content}

        # Parse tool calls from the model response
        try:
            tool_calls_dict = self.decode_action_from_token_ids(response_ids)
        except Exception as e:
            psrl_logger.error("Failed to parse tool calls: %s", e)
            tool_calls_dict = []

        # Update current step with assistant response
        self.add_step_chat_message(assistant_message)
        self.set_step_model_response(assistant_content)
        self.set_step_action(tool_calls_dict)

        # Compute step reward if using step-level reward mode
        if self.config.gen_actor_rollout_ref.rollout.agent.traj_reward_mode == "step":
            raise NotImplementedError
            # output.non_tensor_batch["__num_turns__"] = np.array(
            #     [self.trajectory.assistant_turns + self.trajectory.user_turns + 1], dtype=np.int32
            # )
            # self.get_current_step().model_reward = await self.reward_manager.compute_score.remote(output)

        # Check if response length limit is reached
        if self.trajectory.response_length >= self.config.gen_actor_rollout_ref.rollout.response_length:
            return [], True
        return tool_calls_dict, False

    def prepare_chat_completion_request(self) -> tuple[list[dict], list[dict] | None]:
        """Build messages and tools from current trajectory state."""
        messages = []
        for step in self.trajectory.steps:
            messages.extend(step.chat_completions)
        tools = self.env.get_tool_schemas()
        return messages, tools

    def _parse_tool_calls(self, tool_calls: list[dict]):
        """Extract tool call actions from OpenAI format tool_calls."""
        # Return the tool_calls list directly — the environment handles the format
        return tool_calls

    async def update_from_model_chat_completion(self, output: dict, **kwargs) -> tuple:
        """Parse chat completion response and update trajectory.

        Args:
            output: Raw chat completion response dict from SMG.

        Returns:
            Tuple of (action, overlong_terminate).
        """
        choice = output["choices"][0]
        assistant_msg = choice["message"]

        # Update step
        current_step = self.trajectory.steps[-1]
        current_step.chat_completions.append(assistant_msg)
        current_step.model_response = assistant_msg.get("content", "") or ""

        # Parse tool calls into action
        tool_calls = assistant_msg.get("tool_calls")
        if tool_calls:
            action = self._parse_tool_calls(tool_calls)
        else:
            action = assistant_msg.get("content", "")

        # Check length limit
        usage = output.get("usage", {})
        total_tokens = usage.get("total_tokens", 0)
        overlong = total_tokens > (
            self.config.gen_actor_rollout_ref.rollout.prompt_length
            + self.config.gen_actor_rollout_ref.rollout.response_length
        )

        current_step.action = action
        return action, overlong
