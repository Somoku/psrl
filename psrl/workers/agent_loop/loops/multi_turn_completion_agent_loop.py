import asyncio
import logging
import os

import ray
from transformers import AutoProcessor, AutoTokenizer
from verl import DataProto
from verl.utils.dataset.rl_dataset import RLHFDataset

from psrl.environments.base import Environment
from psrl.utils.common.http_utils import _ensure_http_client, get, post
from psrl.utils.rollout.rollout_trace import rollout_trace_op
from psrl.utils.tito.training_data import build_training_arrays
from psrl.workers.agent_loop.agent_data import AgentData
from psrl.workers.agent_loop.loops.base_agent_loop import AgentLoopBase
from psrl.workers.agent_loop.loops.utils import DictConfigWrap, TerminateReason, register

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@register("multi_turn_completion_agent")
class MultiTurnCompletionAgentLoop(AgentLoopBase):
    """Agent loop that performs multi-turn generation via chat-completion API.

    This loop interacts with a SessionRouter (backed by SMG TITO sessions):
    1. Creates a TITO session.
    2. Each turn: builds messages via ``agent_data.prepare_chat_completion_request()``,
       sends a ``POST /sessions/{sid}/v1/chat/completions`` request, and parses the
       response via ``agent_data.update_from_model_chat_completion()``.
    3. Handles termination (max turns, env timeout, max response length).
    4. After the loop: retrieves accumulated session data via ``GET /sessions/{sid}``
       and builds training arrays with ``build_training_arrays``.
    5. Deletes the session in a ``finally`` block.
    """

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
        self.max_turns = trainer_config.gen_actor_rollout_ref.rollout.multi_turn.max_turns
        self.env_step_timeout = trainer_config.gen_actor_rollout_ref.rollout.agent.env.step_timeout

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    async def _create_session(self) -> str:
        """Create a TITO session and return its ID."""
        base = self.gateway_addr.rstrip("/")
        resp = await post(f"{base}/sessions", payload={})
        session_id = resp["session_id"]
        psrl_logger.info("Created TITO session %s", session_id)
        return session_id

    async def _delete_session(self, session_id: str) -> None:
        """Delete a TITO session (best-effort)."""

        base = self.gateway_addr.rstrip("/")
        url = f"{base}/sessions/{session_id}"
        try:
            client = await _ensure_http_client()
            async with client.delete(url) as resp:
                psrl_logger.info("Deleted TITO session %s (status=%s)", session_id, resp.status)
        except Exception:
            psrl_logger.warning("Failed to delete TITO session %s", session_id, exc_info=True)

    async def _get_session_data(self, session_id: str) -> dict:
        """Retrieve accumulated session data."""
        base = self.gateway_addr.rstrip("/")
        return await get(f"{base}/sessions/{session_id}")

    def _get_chat_sampling_params(self, is_validate: bool = False) -> dict:
        """Build sampling params for the TITO chat-completion path.

        Mirrors ``AgentLoopBase._get_sampling_params`` but without input-length
        awareness (the session server manages accumulated context).  Per-turn
        generation is bounded by ``rollout_config.response_length``.
        """
        top_k = int(self.rollout_config.top_k)
        if top_k < 0:
            top_k = 0

        params = dict(
            temperature=float(self.rollout_config.temperature),
            top_p=float(self.rollout_config.top_p),
            top_k=top_k,
            repetition_penalty=float(self.rollout_config.get("repetition_penalty", 1.0)),
            max_tokens=int(self.rollout_config.response_length),
        )

        if is_validate:
            val_config = self.config.train_actor_rollout_ref.rollout.val_kwargs
            val_top_k = int(val_config.top_k)
            if val_top_k < 0:
                val_top_k = 0
            params["top_k"] = val_top_k
            params["top_p"] = float(val_config.top_p)
            params["temperature"] = float(val_config.temperature)

        return params

    async def _chat_completion(
        self,
        session_id: str,
        messages: list[dict],
        sampling_params: dict,
        tools: list[dict] | None = None,
    ) -> dict:
        """Send a chat-completion request scoped to *session_id*."""
        base = self.gateway_addr.rstrip("/")
        url = f"{base}/sessions/{session_id}/v1/chat/completions"

        payload: dict = {"model": self.model_config.path, "messages": messages}
        if tools:
            payload["tools"] = tools
        payload.update(sampling_params)

        return await post(url, payload=payload)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    @rollout_trace_op
    async def run(self, request: DataProto) -> tuple[DataProto | None, TerminateReason]:
        """Execute generation for a single request using chat completions.

        Args:
            request: Single input request.

        Returns:
            Tuple of generated DataProto and termination reason.
        """
        env_class = request.non_tensor_batch.get(
            "env_class", self.config.gen_actor_rollout_ref.rollout.agent.env.name
        )[0]
        data_class = request.non_tensor_batch.get(
            "data_class", self.config.gen_actor_rollout_ref.rollout.agent.data.name
        )[0]

        self.env = Environment.get_environment(
            env_class,
            self.config,
            self.reward_manager,
            self.max_turns,
            tokenizer=self.tokenizer,
            processor=self.processor,
            dataset_cls=self.dataset_cls,
        )
        self.agent_data = AgentData.get_agent_data(
            data_class,
            self.config,
            self.reward_manager,
            self.env,
        )
        self.agent_data.reset()

        is_validate = request.meta_info.get("validate", False)
        sampling_params = self._get_chat_sampling_params(is_validate)

        session_id: str | None = None
        try:
            # --- Create TITO session ---
            session_id = await self._create_session()

            # --- Environment reset ---
            observation, info = await self.env.reset(
                task=request,
                seed=request.non_tensor_batch.get("seed", None),
            )

            self.agent_data.init_trajectory(request)

            overlong_terminate = await self.agent_data.update_from_env(observation, 0, False, info)
            if overlong_terminate:
                return (
                    await self.agent_data.finalize_output(request),
                    TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED,
                )

            # --- Turn loop ---
            for _ in range(self.max_turns):
                # 1. Build messages/tools from agent data
                messages, tools = self.agent_data.prepare_chat_completion_request()

                # 2. Call chat completion API
                response = await self._chat_completion(session_id, messages, sampling_params, tools)

                # 3. Parse response and extract action
                action, overlong_terminate = await self.agent_data.update_from_model_chat_completion(response)

                if overlong_terminate:
                    return (
                        await self.agent_data.finalize_output(request),
                        TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED,
                    )

                # 4. Step the environment
                try:
                    env_step_output = await asyncio.wait_for(
                        self.env.step(action),
                        timeout=self.env_step_timeout,
                    )
                    observation = env_step_output["observation"]
                    reward = env_step_output["reward"]
                    done = env_step_output["done"]
                    info = env_step_output["info"]
                except asyncio.TimeoutError:
                    return (
                        await self.agent_data.finalize_output(request),
                        TerminateReason.ENV_TIMEOUT,
                    )

                overlong_terminate = await self.agent_data.update_from_env(observation, reward, done, info)

                if overlong_terminate:
                    return (
                        await self.agent_data.finalize_output(request),
                        TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED,
                    )

                if done:
                    break

            # --- Retrieve session data and build training arrays ---
            session_data = await self._get_session_data(session_id)
            max_trim_tokens = session_data.get("max_trim_tokens", 0)

            if "trajectories" in session_data:
                # Multi-trajectory (retry/concurrent branches) — not expected in normal PSRL flow.
                # Log a warning and use the first trajectory.
                psrl_logger.warning(
                    "TITO session %s has %d trajectories (retry/concurrent branches detected), "
                    "using first trajectory for training",
                    session_id,
                    len(session_data["trajectories"]),
                )
                traj = session_data["trajectories"][0]
                accumulated_token_ids = traj.get("accumulated_token_ids", [])
                records = traj.get("records", [])
            else:
                accumulated_token_ids = session_data.get("accumulated_token_ids", [])
                records = session_data.get("records", [])

            # Check mismatch reports: any non-assistant_text mismatch is a TITO algorithm
            # error (e.g. special token count/type mismatch, non-assistant content drift).
            # These must abort training data construction to avoid silently corrupted samples.
            # ASSISTANT_TEXT mismatches are expected cross-turn token boundary differences
            # and are not fatal.
            for i, record in enumerate(records):
                mismatch_report = record.get("mismatch_report", [])
                for entry in mismatch_report:
                    mtype = entry.get("mismatch_type", "")
                    if mtype != "assistant_text":
                        msg = (
                            f"TITO token ID mismatch at turn {i}: "
                            f"type={mtype}, pos={entry.get('position')}, "
                            f"detail={entry.get('detail')}"
                        )
                        psrl_logger.error(msg)
                        raise RuntimeError(msg)

            training_arrays = build_training_arrays(
                accumulated_token_ids, records, max_trim_tokens=max_trim_tokens
            )

            # Attach training arrays to trajectory for finalize_output
            trajectory = self.agent_data.trajectory
            trajectory.prompt_ids = training_arrays["prompt_ids"]
            trajectory.response_ids = training_arrays["response_ids"]
            trajectory.response_mask = training_arrays["response_mask"]
            trajectory.response_logprobs = training_arrays["logprobs"]

            terminate_reason = TerminateReason.FINISHED if done else TerminateReason.MAX_TURNS_EXCEEDED
            return await self.agent_data.finalize_output(request), terminate_reason

        finally:
            if session_id is not None:
                await self._delete_session(session_id)
