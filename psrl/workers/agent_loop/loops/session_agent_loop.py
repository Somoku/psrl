"""Common agent-loop support for SMG SessionRouter and TITO."""

from __future__ import annotations

import logging
import os

import ray
from transformers import AutoProcessor, AutoTokenizer
from verl.utils.dataset.rl_dataset import RLHFDataset

from psrl.utils.common.http_utils import delete, get, post
from psrl.utils.tito.training_data import build_training_arrays
from psrl.workers.agent_loop.agent_data import AgentData
from psrl.workers.agent_loop.loops.base_agent_loop import AgentLoopBase
from psrl.workers.agent_loop.loops.utils import DictConfigWrap

psrl_logger = logging.getLogger(__name__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class SessionAgentLoop(AgentLoopBase):
    """Provide the shared SessionRouter and TITO lifecycle for agent loops."""

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        rollout_router: ray.actor.ActorHandle | str,
        reward_manager: ray.actor.ActorHandle,
        ps_manager_handle: ray.actor.ActorHandle,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        dataset_cls: type[RLHFDataset],
        data_config: DictConfigWrap,
        **kwargs,
    ):
        super().__init__(
            trainer_config=trainer_config,
            rollout_router=rollout_router,
            reward_manager=reward_manager,
            ps_manager_handle=ps_manager_handle,
            tokenizer=tokenizer,
            processor=processor,
            dataset_cls=dataset_cls,
            data_config=data_config,
            **kwargs,
        )
        self.session_router_url = str(kwargs["session_router_url"]).rstrip("/")
        self.max_turns = int(self.config.gen_actor_rollout_ref.rollout.multi_turn.max_turns)

    def get_generate_fields(self) -> list[str]:
        fields = super().get_generate_fields()
        fields.extend(["env_class", "data_class", "seed"])
        return fields

    def build_session_headers(self, request: dict) -> dict[str, str]:
        """Build immutable routing metadata for one session."""
        request_id = request["uid"]
        headers = {
            "x-request-id": str(request_id),
            "x-prompt-id": str(request.get("parent_id", request_id)),
            "x-version-tag": str(request.get("version_tag", 0)),
            "x-is-validate": str(request.get("validate", False)).lower(),
            "x-is-sticky": str(bool(self.config.psrl.routing_strategy.enable_trajectory_sticky)).lower(),
            "x-smg-tito-trajectory-id": str(request.get("trajectory_id", 0)),
        }
        rollout_instance_id = request.get("rollout_instance_id")
        if rollout_instance_id is not None:
            headers["x-base-worker-id"] = str(rollout_instance_id[0])
            headers["x-target-dp-rank"] = str(rollout_instance_id[1])
        return headers

    async def create_session(self, request: dict) -> str:
        """Create one TITO session and bind its routing metadata."""
        response = await post(
            f"{self.session_router_url}/sessions",
            payload={},
            headers=self.build_session_headers(request),
        )
        session_id = str(response["session_id"])
        psrl_logger.debug("Created TITO session %r.", session_id)
        return session_id

    async def delete_session(self, session_id: str) -> None:
        """Delete one TITO session after its requests have drained."""
        try:
            response = await delete(f"{self.session_router_url}/sessions/{session_id}")
            psrl_logger.debug("Deleted TITO session %r with status %r.", session_id, response.status)
        except Exception:
            psrl_logger.warning("Failed to delete TITO session %r.", session_id, exc_info=True)

    def session_api_url(self, session_id: str) -> str:
        """Return the OpenAI-compatible API base for one session."""
        return f"{self.session_router_url}/sessions/{session_id}/v1"

    def get_session_sampling_params(self, request: dict) -> dict:
        """Build per-turn sampling parameters and the TITO training contract."""
        config = self.rollout_config
        if request.get("validate", False):
            config = self.config.train_actor_rollout_ref.rollout.val_kwargs
        params = {
            "temperature": float(config.temperature),
            "top_p": float(config.top_p),
            "top_k": int(config.top_k),
            "repetition_penalty": float(self.rollout_config.get("repetition_penalty", 1.0)),
            "ignore_eos": self.rollout_config.get("ignore_eos", False),
            "max_tokens": int(self.rollout_config.response_length),
            "logprobs": True,
            "top_logprobs": 1,
            "stream": False,
        }
        if request.get("seed") is not None:
            params["seed"] = int(request["seed"])
        return params

    async def chat_completion(
        self,
        session_id: str,
        messages: list[dict],
        sampling_params: dict,
        *,
        tools: list[dict] | None = None,
        chat_template_kwargs: dict | None = None,
    ) -> dict:
        """Send one chat-completion request through a session."""
        payload = {
            "model": self.model_config.path,
            "messages": messages,
            **sampling_params,
        }
        if tools:
            payload["tools"] = tools
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs
        with self.timer.generation():
            return await post(f"{self.session_api_url(session_id)}/chat/completions", payload=payload)

    async def get_training_arrays(
        self,
        session_id: str,
        trajectory_id: int,
    ) -> dict:
        """Fetch, validate, and convert one TITO trajectory into training arrays."""
        session_data = await get(f"{self.session_router_url}/sessions/{session_id}")
        accumulated_token_ids, records = self._select_trajectory(
            session_data,
            session_id,
            trajectory_id,
        )
        self._validate_records(records)
        arrays = build_training_arrays(
            accumulated_token_ids,
            records,
            max_trim_tokens=session_data.get("max_trim_tokens", 0),
        )
        header_info = session_data.get("header_info", {})
        if header_info.get("base_worker_id") is not None and header_info.get("target_dp_rank") is not None:
            arrays["rollout_instance_id"] = (
                header_info["base_worker_id"],
                int(header_info["target_dp_rank"]),
            )
        return arrays

    @staticmethod
    def attach_training_arrays(
        agent_data: AgentData,
        arrays: dict,
        *,
        update_turn_counts: bool = False,
    ) -> None:
        """Attach canonical TITO arrays and optional TITO-derived turn counts."""
        trajectory = agent_data.session_data.trajectories[-1]
        trajectory.prompt_ids = arrays["prompt_ids"]
        trajectory.response_ids = arrays["response_ids"]
        trajectory.response_mask = arrays["response_mask"]
        trajectory.response_logprobs = arrays["logprobs"]
        trajectory.response_length = len(trajectory.response_ids)
        if update_turn_counts:
            trajectory.assistant_turns = arrays["num_turns"]
            trajectory.user_turns = max(0, arrays["num_turns"] - 1)
        if arrays["routed_experts"] is not None:
            trajectory.routed_experts = arrays["routed_experts"]

        agent_data.session_data.response_length = trajectory.response_length
        if update_turn_counts:
            agent_data.session_data.assistant_turns = trajectory.assistant_turns
            agent_data.session_data.user_turns = trajectory.user_turns
        agent_data.session_data.curr_rollout_instance_id = arrays.get("rollout_instance_id")

    @staticmethod
    def _select_trajectory(
        session_data: dict,
        session_id: str,
        trajectory_id: int,
    ) -> tuple[list[int], list[dict]]:
        trajectories = session_data.get("trajectories")
        if trajectories is None:
            return (
                session_data.get("accumulated_token_ids", []),
                session_data.get("records", []),
            )
        if not trajectories:
            raise RuntimeError(f"TITO session {session_id!r} returned no trajectories.")

        trajectory_id_text = str(trajectory_id)
        trajectory = None
        for item in trajectories:
            if str(item.get("trajectory_id", item.get("id", 0))) == trajectory_id_text:
                trajectory = item
                break
        if trajectory is None:
            raise RuntimeError(f"TITO session {session_id!r} has no trajectory_id={trajectory_id!r}.")
        return trajectory.get("accumulated_token_ids", []), trajectory.get("records", [])

    @staticmethod
    def _validate_records(records: list[dict]) -> None:
        for turn, record in enumerate(records):
            for mismatch in record.get("mismatch_report", []):
                if mismatch.get("mismatch_type") != "assistant_text":
                    raise RuntimeError(f"TITO token ID mismatch at turn {turn}: {mismatch!r}.")
