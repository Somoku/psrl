import copy
import json
import logging
import os
import uuid

import ray
import torch
import numpy as np
from omegaconf import DictConfig
from PIL import Image
from verl.utils import tensordict_utils as tu
from verl.utils.chat_template import apply_chat_template, initialize_system_prompt

from psrl.environments.base import ConversationType, Environment
from psrl.environments.tool_env import ToolAction
from psrl.tools.tool_parser.base import ToolParser
from psrl.workers.agent_loop.agent_data.base import AgentData, Trajectory, SessionData
from psrl.workers.gen_dplb.utils import TokenOutput

psrl_logger = logging.getLogger(__name__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


def sort_tool_schema_keys(value):
    """Return a copy of a tool schema value with all dict keys sorted recursively."""
    if isinstance(value, dict):
        return {key: sort_tool_schema_keys(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [sort_tool_schema_keys(item) for item in value]
    return copy.deepcopy(value)

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

    def __init__(
        self,
        config: DictConfig,
        reward_manager: ray.actor.ActorHandle,
        env: Environment,
        **kwargs,
    ):
        """Initialize ToolAgentData instance.

        Args:
            config: Configuration object containing training settings
            reward_manager: Ray actor handle for computing rewards
            env: Environment instance with ``get_tool_schemas()`` method
            tokenizer: Tokenizer for converting between text and tokens
            processor: Optional multimodal processor (e.g. Qwen2VLProcessor).
            **kwargs: Additional keyword arguments from configuration
        """
        assert hasattr(env, "get_tool_schemas"), "Environment must implement get_tool_schemas method."

        super().__init__(config, reward_manager, env)
        self.tool_schemas = sort_tool_schema_keys(self.env.get_tool_schemas())
        self.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        self.max_turns = self.config.gen_actor_rollout_ref.rollout.multi_turn.max_turns
        self.tool_parser = ToolParser.get_tool_parser(self.config.gen_actor_rollout_ref.rollout.multi_turn.format, self.tokenizer)

        # Pre-compute system prompt token ids for incremental turn stripping.
        # This mirrors AgentLoopBase._get_system_prompt_ids() but lives here so
        # AgentData subclasses don't need a reference to the loop.
        self._system_prompt_ids: list[int] = initialize_system_prompt(
            self.tokenizer,
            **self.apply_chat_template_kwargs,
        )

    async def encode_observation(
        self,
        observation: ConversationType,
        images: list[Image.Image] | None = None,
        videos: list[tuple[torch.Tensor, dict]] | None = None,
        is_init: bool = False,
    ) -> tuple[list[int], bool]:
        """Encode tool-env conversation messages into token ids (async).

        For the first observation we include tool schemas and treat the result as
        prompt ids.  For subsequent observations we re-encode incrementally,
        stripping the system-prompt prefix, and treat the result as user-side
        tokens (masked to 0 during training).

        When a multimodal *processor* is configured (VLM path), the initial
        observation is processed via ``processor(text=..., images=..., ...)`` so
        that vision tokens are embedded correctly.  Subsequent incremental turns
        are still text-only (tool responses are text) and use the tokenizer path.

        All blocking CPU work is offloaded to the default thread-pool executor.

        Returns:
            (token_ids, is_prompt) where is_prompt is True only for the initial
            observation (which becomes part of the padded prompt tensor).
        """
        from verl.utils.tokenizer import normalize_token_ids

        if not observation:
            return [], is_init

        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: apply_chat_template(
                    self.processor,
                    observation,
                    tools=self.tool_schemas if is_init else None,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                )
            )
            
            # split the videos and according metadatas
            if videos is not None:
                videos, video_metadatas = zip(*videos, strict=False)
                videos, video_metadatas = list(videos), list(video_metadatas)
            else:
                video_metadatas = None

            model_inputs = self.processor(
                text=[raw_prompt],
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                return_tensors="pt",
                do_sample_frames=False,
            )
            prompt_ids = normalize_token_ids(model_inputs.pop("input_ids"))
        else:
            tokenized_prompt = await self.loop.run_in_executor(
                None,
                lambda: apply_chat_template(
                    self.tokenizer,
                    observation,
                    tools=self.tool_schemas if is_init else None,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
            prompt_ids = normalize_token_ids(tokenized_prompt)

        if not is_init:
            prompt_ids = prompt_ids[len(self._system_prompt_ids):]

        return prompt_ids, is_init

    def format_chat_completions(self, observation: ConversationType, *, is_init: bool) -> ConversationType:
        """ToolEnvironment already uses ConversationType as observation, so return as-is."""
        return observation

    def decode_action_from_token_ids(self, token_ids: list[int]) -> ToolAction:
        """Decode model generated token ids into ToolAction.

        Returns a list of OpenAI-function-call style dicts:
        [{"type": "function", "function": {"name": ..., "arguments": "..."}}]
        """
        _, tool_calls = self.tool_parser.extract_tool_calls_from_token_ids(token_ids)
        tool_calls_dict = [
            {
                "id": f"call_{uuid.uuid4().hex[:12]}",
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

    def decode_action_from_response_str(self, response_str: str) -> ToolAction:
        """Decode model generated response string into ToolAction.

        Returns a list of OpenAI-function-call style dicts:
        [{"type": "function", "function": {"name": ..., "arguments": "..."}}]
        """
        _, tool_calls = self.tool_parser.extract_tool_calls_from_str(response_str)
        tool_calls_dict = [
            {
                "id": f"call_{uuid.uuid4().hex[:12]}",
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
        self.session_data = SessionData()

    def init_trajectory(self, request: dict) -> None:
        """Initialize a new trajectory from the input request.

        Extracts request_id, parent_id, and per-sample tools_kwargs from the
        request and creates a new Trajectory instance.

        Args:
            request: dict containing the initial task and metadata.

        Raises:
            AssertionError: If the request contains more than one item.
        """
        request_id = request.get("uid", 0)
        parent_id = request.get("parent_id", None)
        validate = request.get("validate", False)
        data_source = request.get("data_source", "unknown")
        reward_model = request.get("reward_model", {})
        extra_info = request.get("extra_info", {})
        reward_model_dicts = request.get("reward_model_dicts", [])

        self.session_data = SessionData(
            request_id=request_id,
            parent_id=parent_id,
            validate=validate,
            data_source=data_source,
            reward_model=reward_model,
            extra_info=extra_info,
            reward_model_dicts=reward_model_dicts,
        )
        self.session_data.trajectories.append(Trajectory())

    async def update_from_env(
        self,
        observation: ConversationType,
        reward: float | list[float] | None,
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
        is_init = len(self.session_data.trajectories[-1].steps) == 0

        multi_modal_data = info.get("multi_modal_data", {})
        images = multi_modal_data.get("images", None)
        videos = multi_modal_data.get("videos", None)

        if images:
            if self.session_data.trajectories[-1].image_data is None:
                self.session_data.trajectories[-1].image_data = []
            elif not isinstance(self.session_data.trajectories[-1].image_data, list):
                self.session_data.trajectories[-1].image_data = [self.session_data.trajectories[-1].image_data]
            for img in images:
                self.session_data.trajectories[-1].image_data.append(img)
        if videos:
            raise NotImplementedError("Video data handling is not yet implemented in ToolAgentData.")

        if len(self.session_data.trajectories[-1].steps) < self.max_turns:
            # Create a new step explicitly (avoids implicit steps[-1] contract).
            if isinstance(reward, float):
                reward = [reward]
            step = self.start_step(observation=observation, reward=reward, done=done, info=info)

            # Fill normalized ConversationType for logging/debugging.
            step.chat_completions = self.format_chat_completions(observation, is_init=is_init)

            # NOTE(linsh): currently we do not support gpt-oss style tool calls.
            token_ids, is_prompt = await self.encode_observation(
                observation,
                images=images,
                videos=videos,
                is_init=is_init,
            )
            if token_ids:
                if is_prompt:
                    self.append_prompt_ids(token_ids)
                else:
                    self.append_user_tokens(token_ids)

        # Accumulate non-None tool rewards from env steps for downstream use.
        if reward is not None:
            # TODO(linsh): find a better approach to handle multiple tool rewards
            # currently we just sum them up for trajectory-level reward computation,
            # but we may want to keep them separate for more detailed credit assignment.
            tool_reward = np.sum(reward) if isinstance(reward, list) else reward
            self.session_data.trajectories[-1].tool_rewards.append(tool_reward)

        # Check if response length limit is reached
        return self.session_data.trajectories[-1].response_length >= self.config.gen_actor_rollout_ref.rollout.response_length

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
        if len(self.session_data.trajectories[-1].steps) == 0:
            # This should not happen in MultiTurnAgentLoop, but keep a safe fallback
            # for custom loops or unexpected call orders.
            self.start_step(observation=[], reward=None, done=False, info={})

        self.update_trajectory_state_from_output(output)

        response_ids = output.response_ids
        rollout_logprobs = output.response_log_probs
        if output.routed_experts is not None:
            routed_experts = output.routed_experts.tolist()
        else:
            routed_experts = None

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
            # TODO(linsh): check whether use global num_turns for step reward
            # NOTE(linsh): step reward is not verified and maybe buggy
            output.num_turns = self.session_data.assistant_turns + self.session_data.user_turns + 1
            tensor_dict = {
                "prompts": torch.tensor(output.prompt_ids, dtype=torch.int64).unsqueeze(0),
                "responses": torch.tensor(output.response_ids, dtype=torch.int64).unsqueeze(0),
                "multi_modal_data": np.array([output.multi_modal_data], dtype=object),
                "num_turns": np.array([output.num_turns]),
                "tool_extra_fields": np.array([output.extra_fields], dtype=object),
                "uid": np.array([self.session_data.request_id]),
                "n_trajectory": np.array([len(self.session_data.trajectories)]),
                "data_source": np.array([self.session_data.data_source]),
            }
            if self.session_data.parent_id is not None:
                tensor_dict["parent_id"] = np.array([self.session_data.parent_id])
            data = tu.get_tensordict(
                tensor_dict=tensor_dict,
                non_tensor_dict={"validate": self.session_data.validate},
            )
            reward_meta_infos = [{
                "reward_model": self.session_data.reward_model,
                "extra_info": self.session_data.extra_info,
                "reward_model_dicts": self.session_data.reward_model_dicts,
            }]

            self.get_current_step().model_reward = await self.reward_manager.compute_score.remote(
                data,
                reward_meta_infos=reward_meta_infos,
            )["reward_score"]

        # Check if response length limit is reached
        if self.session_data.trajectories[-1].response_length >= self.config.gen_actor_rollout_ref.rollout.response_length:
            return [], True
        return tool_calls_dict, False

    def prepare_chat_completion_request(self) -> tuple[list[dict], list[dict] | None]:
        """Build messages and tools from current trajectory state."""
        messages = []
        for step in self.session_data.trajectories[-1].steps:
            messages.extend(step.chat_completions)
        tools = self.tool_schemas
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
        self.session_data.assistant_turns += 1
        self.session_data.trajectories[-1].assistant_turns += 1
        model_response = assistant_msg.get("content", "") or ""

        try:
            tool_calls_dict = self.decode_action_from_response_str(model_response)
        except Exception as e:
            psrl_logger.error("Failed to parse tool calls: %s", e)
            tool_calls_dict = []

        # Update current step with assistant response
        self.add_step_chat_message(assistant_msg)
        self.set_step_model_response(model_response)
        self.set_step_action(tool_calls_dict)

        # Check length limit
        usage = output.get("usage", {})
        total_tokens = usage.get("total_tokens", 0)
        overlong = total_tokens > (
            self.config.gen_actor_rollout_ref.rollout.prompt_length
            + self.config.gen_actor_rollout_ref.rollout.response_length
        )

        return tool_calls_dict, overlong
