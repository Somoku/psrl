import asyncio
import base64
import logging
import os
from abc import ABC, abstractmethod
from contextlib import nullcontext

import numpy as np
import ray
import torch
from PIL import Image
from tensordict import NonTensorData, NonTensorStack, TensorDict
from transformers import AutoProcessor, AutoTokenizer
from verl.utils import tensordict_utils as tu
from verl.utils.chat_template import apply_chat_template, initialize_system_prompt
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.tokenizer import normalize_token_ids

from psrl.utils.common.http_utils import (
    RequestAbortedByGatewayError,
    is_distributed_post_enabled,
    request_json_maybe_distributed,
)
from psrl.utils.common.http_io_thread import get_http_io_thread
from psrl.utils.dataset.utils import _pre_process_inputs
from psrl.utils.rollout.vision_utils import extract_image_ref, serialize_image_inputs
from psrl.workers.agent_loop.loops.utils import DictConfigWrap, TerminateReason
from psrl.workers.agent_loop.sticky_session import sticky_session
from psrl.workers.gen_dplb.utils import TokenInput, TokenOutput
from psrl.workers.ps.request_status_tracker import PSRL_RequestStatus

psrl_logger = logging.getLogger(__name__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class AgentLoopBase(ABC):
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
        """Initialize agent loop instance.
        Base class for agent loops that process requests and interact with LLM servers.

        Args:
            trainer_config (DictConfigWrap): Wrapper containing trainer configuration.
            rollout_router (ray.actor.ActorHandle): Router for distributing requests to LLM servers.
            ps_manager_handle (ray.actor.ActorHandle): Handle to parameter server manager.
            tokenizer (AutoTokenizer): Tokenizer for processing text messages.
            **kwargs: Additional keyword arguments.
        """
        self.config = trainer_config.config
        self.model_config = self.config.gen_actor_rollout_ref.model
        self.rollout_config = self.config.gen_actor_rollout_ref.rollout
        self.rollout_router = rollout_router
        self.use_rust_gateway = isinstance(rollout_router, str)
        if self.use_rust_gateway:
            self.gateway_addr = rollout_router
        else:
            self.gateway_addr = None

        self.reward_manager = reward_manager
        self.ps_manager_handle = ps_manager_handle
        self.tokenizer = tokenizer
        self.processor = processor
        self.dataset_cls = dataset_cls
        self.data_config = data_config.config
        self.apply_chat_template_kwargs = self.data_config.get("apply_chat_template_kwargs", {})
        self.system_prompt = initialize_system_prompt(self.tokenizer, **self.apply_chat_template_kwargs)
        self.loop = asyncio.get_running_loop()
        self.response_length = self.rollout_config.response_length
        self.prompt_length = self.rollout_config.prompt_length
        self.output_in_tq = False

    async def process_vision_info(
        self,
        messages: list[dict],
    ) -> tuple[list | None, list | None]:
        """Extract images and videos from messages.

        Delegates to ``dataset_cls.process_vision_info`` (the same path used by
        ``AgentLoopBase.process_vision_info``), mirroring verl's design where a
        single canonical extraction function is shared across the whole stack.

        When no processor is configured (text-only model) both return values are None.

        Args:
            messages: Chat messages that may contain image/video content parts.

        Returns:
            (images, videos):
                images - list of PIL.Image.Image, or None if none found.
                videos - list of (video_tensor, metadata) tuples, or None.
        """
        if self.processor is None:
            return None, None

        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is None:
            psrl_logger.warning(
                "AgentData.process_vision_info: processor %s has no image_processor attribute; "
                "skipping vision extraction.",
                type(self.processor).__name__,
            )
            return None, None

        if self.dataset_cls is None:
            raise RuntimeError(
                "AgentData.process_vision_info: dataset_cls is required when processor is set. "
                "Pass dataset_cls= when constructing AgentData."
            )

        images, videos = await self.dataset_cls.process_vision_info(
            messages,
            image_patch_size=image_processor.patch_size,
            config=self.config.data,
        )
        return (images if images else None), (videos if videos else None)

    async def apply_chat_template(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        images: list[Image.Image] | None = None,
        videos: list[tuple[torch.Tensor, dict]] | None = None,
        remove_system_prompt: bool = False,
    ):
        """Apply chat template to messages with optional tools, images, and videos.

        Args:
            messages (list[dict]): Input messages.
            tools (list[dict], optional): Tools schemas. Defaults to None.
            images (list[Image.Image], optional): Input images. Defaults to None.
            videos (list[tuple[torch.Tensor, dict]], optional): Input videos. Defaults to None.
            remove_system_prompt (bool, optional): Whether to remove system prompt. Defaults to False.

        Returns:
            list[int]: Prompt token ids.
        """
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: apply_chat_template(
                    self.processor,
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
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
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
            prompt_ids = normalize_token_ids(tokenized_prompt)

        if remove_system_prompt:
            prompt_ids = prompt_ids[len(self.system_prompt) :]

        return prompt_ids

    async def compute_reward_score(
        self,
        outputs: TokenOutput | list[TokenOutput],
        **kwargs,
    ) -> TokenOutput | list[TokenOutput] | None:
        """Compute reward score for the generated response and merge it into the output.

        This function sends the generated response to the reward manager and waits for the computed reward score. If the reward computation is successful, it merges the reward score into the output data structure. If the request is aborted during reward computation (e.g., due to staleness check failure), it returns None to indicate that the request should be aborted.

        Args:
            data (TokenOutput): The output data structure containing the generated response and associated metadata.
        Returns:
            TokenOutput | None: The updated output data structure with the computed reward score, or None if the request was aborted during reward computation.
        """
        if not isinstance(outputs, list):
            outputs = [outputs]
        # NOTE(linsh): Only compute reward for the last trajectory.
        final_output = outputs[-1]

        # Build TensorDict: tensors in batch, metadata in non_tensor_batch/meta_info
        tensor_dict = {
            "prompts": torch.tensor(final_output.prompt_ids, dtype=torch.int64).unsqueeze(0),
            "responses": torch.tensor(final_output.response_ids, dtype=torch.int64).unsqueeze(0),
            "multi_modal_data": np.array([final_output.multi_modal_data], dtype=object),
            "num_turns": np.array([final_output.num_turns]),
            "tool_extra_fields": np.array([final_output.extra_fields], dtype=object),
            "uid": np.array([kwargs.get("uid")]),
            "n_trajectory": np.array([len(outputs)]),
            "data_source": np.array([kwargs.get("data_source", "unknown")]),
            "reward_model": np.array([kwargs.get("reward_model", {})], dtype=object),
            "extra_info": np.array([kwargs.get("extra_info", {})], dtype=object),
            "reward_model_dicts": np.array([kwargs.get("reward_model_dicts", [])], dtype=object),
        }
        if kwargs.get("parent_id") is not None:
            tensor_dict["parent_id"] = np.array([kwargs.get("parent_id")])

        reward_requests = tu.get_tensordict(
            tensor_dict=tensor_dict,
            non_tensor_dict={
                "validate": kwargs.get("validate", False),
            },
        )

        reward_result = await self.reward_manager.compute_score.remote(reward_requests)
        if not self.config.reward.launch_reward_fn_async:
            if not reward_result:
                return None
            # Broadcast to all trajectories
            for output in outputs:
                output.reward_score = reward_result["reward_score"]
                output.extra_fields["reward_extra_info"] = reward_result["reward_extra_info"]
                output.extra_fields["reward_metrics"] = reward_result.get("reward_metrics", {})

        if len(outputs) == 1:
            return outputs[0]
        return outputs

    async def generate_sequence(self, request: dict, is_sticky_session: bool = False) -> "TokenOutput":
        request_input: TokenInput = await self.pre_process_inputs(request)
        sampling_params = self._get_sampling_params(request_input)
        if request_input.stop_token_ids:
            sampling_params["stop_token_ids"] = list(
                set((sampling_params.get("stop_token_ids") or []) + request_input.stop_token_ids)
            )
        if self.config.psrl.rollout_gateway.enable:
            if not self.gateway_addr:
                raise RuntimeError("Rollout gateway is enabled but gateway address is empty.")

            # Route multimodal requests to /v1/chat/completions (which has a full
            # Rust-side vision preprocessing pipeline) and text-only requests to
            # /generate (lower latency, returns output_ids directly).
            mm_data = request_input.multi_modal_data
            has_images = mm_data is not None and bool(mm_data.get("images"))
            has_videos = mm_data is not None and bool(mm_data.get("videos"))

            if has_videos:
                raise NotImplementedError(
                    "Video input is not yet supported via the SMG gateway. "
                    "Implement a /v1/chat/completions video path when ready."
                )

            if has_images:
                return await self._generate_via_chat_completions(
                    request_input, sampling_params, is_sticky_session, mm_data
                )

            # ── Text-only: fast /generate path ──────────────────────────────
            return await self._generate_via_generate_endpoint(
                request_input, sampling_params, is_sticky_session
            )
        else:
            mm_data = request_input.multi_modal_data
            if mm_data is not None and (mm_data.get("images") or mm_data.get("videos")):
                psrl_logger.warning(
                    "request_id=%s: Multi-modal data (images/videos) is present but the "
                    "Ray-actor rollout path does not support forwarding image data. "
                    "Enable the Rust gateway (psrl.rollout_gateway.enable=true) for "
                    "multimodal inference.",
                    request_input.request_id,
                )
            async with sticky_session(self.rollout_router, request) if is_sticky_session else nullcontext():
                # TODO(linsh): move sampling params as param of `route_generate`
                output = await self.rollout_router.route_generate.remote(
                    request_input.input_ids,
                    request_input.request_id,
                    request_input.prompt_id,
                    request_input.version_tag,
                    request_input.rollout_instance_id,
                    request_input.cu_response_len,
                    request_input.is_validate,
                    request_input.stop_token_ids,
                )
        return output

    async def _generate_via_generate_endpoint(
        self,
        request_input: "TokenInput",
        sampling_params: dict,
        is_sticky_session: bool,
    ) -> "TokenOutput":
        """Call SMG /generate (text-only, returns output_ids directly)."""
        request_url = f"{self.gateway_addr.rstrip('/')}/generate"

        payload_sampling_params = dict(sampling_params)
        if hasattr(payload_sampling_params.get("output_kind"), "value"):
            payload_sampling_params["output_kind"] = payload_sampling_params["output_kind"].value

        req_headers = {
            "x-request-id": str(request_input.request_id),
            "x-prompt-id": str(request_input.prompt_id),
            "x-version-tag": str(request_input.version_tag),
            "x-is-validate": str(request_input.is_validate).lower(),
        }
        if request_input.rollout_instance_id is not None:
            replica_id, dp_rank = request_input.rollout_instance_id
            req_headers["x-base-worker-id"] = str(replica_id)
            req_headers["x-target-dp-rank"] = str(dp_rank)

        if is_sticky_session:
            req_headers["x-is-sticky"] = "true"

        payload = {
            "model": self.model_config.path,
            "request_id": str(request_input.request_id),
            "input_ids": request_input.input_ids,
            "sampling_params": payload_sampling_params,
            "stream": False,
            "return_logprob": sampling_params.get("logprobs") is not None,
            # TODO: implement it
            # "return_routed_experts": self.config.gen_actor_rollout_ref.rollout.enable_rollout_routing_replay,
        }

        # Multimodal payload
        if request_input.multi_modal_data is not None:
            images = request_input.multi_modal_data.get("images")
            videos = request_input.multi_modal_data.get("videos")
            if images:
                payload["image_data"] = await serialize_image_inputs(images)
            if videos:
                payload["video_data"] = videos
            if images or videos:
                modalities = []
                if images:
                    modalities.append("multi-images" if len(images) > 1 else "image")
                if videos:
                    modalities.append("video")
                payload["modalities"] = modalities

        # Call SMG /generate directly via aiohttp so we can read both the
        # response body (a JSON array) and the worker-instance headers in one pass.
        gen_responses, base_worker_id, target_dp_rank = await self._post_generate(
            request_url, payload, req_headers
        )

        if not gen_responses:
            psrl_logger.error(
                "Gateway /generate returned empty response for request_id=%s",
                request_input.request_id,
            )
            return None

        first = gen_responses[0]

        # rollout instance id
        replica_id = base_worker_id
        rollout_instance_id = (replica_id, int(target_dp_rank) if target_dp_rank is not None else 0)

        # token ids
        token_ids = first["output_ids"]

        # logprobs: SMG returns output_token_logprobs as List[List[Optional[float]]].
        # Each outer entry is one output token position; each inner list is top-k logprobs
        # for that position. We take the first (top-1) entry at each position.
        log_probs = None
        if sampling_params.get("logprobs") is not None:
            raw_logprobs = first.get("meta_info", {}).get("output_token_logprobs")
            if raw_logprobs is not None:
                log_probs = [next((lp for lp in per_pos if lp is not None), 0.0) for per_pos in raw_logprobs]

        # finish_reason: SMG returns {"type": "stop"} or {"type": "length", "length": N}
        finish_reason_raw = first.get("meta_info", {}).get("finish_reason", {})
        if isinstance(finish_reason_raw, dict):
            finish_reason = finish_reason_raw.get("type", "stop")
        else:
            finish_reason = str(finish_reason_raw)

        # Determine interrupted based on finish_reason
        interrupted = finish_reason == "abort"

        # routing replay is not supported via SMG /generate (no routed_experts field)
        routed_experts = None

        return TokenOutput(
            prompt_ids=request_input.input_ids,
            response_ids=token_ids,
            response_mask=[1] * len(token_ids),
            response_log_probs=log_probs,
            routed_experts=routed_experts,
            multi_modal_data=request_input.multi_modal_data,
            stop_reason=finish_reason,
            interrupted=interrupted,
            update_status=PSRL_RequestStatus.ROLLOUT_COMPLETED,
            rollout_instance_id=rollout_instance_id,
        )

    async def _generate_via_chat_completions(
        self,
        request_input: "TokenInput",
        sampling_params: dict,
        is_sticky_session: bool,
        mm_data: dict,
    ) -> "TokenOutput":
        """Call SMG /v1/chat/completions for multimodal requests.

        SMG's chat route runs a full Rust-side vision preprocessing pipeline
        (image fetch -> pixel preprocessing -> mm_inputs proto). When original
        chat messages are still available, PSRL forwards them after normalizing
        image parts to OpenAI ``image_url`` content. When only token IDs and
        Python image objects remain, PSRL falls back to a synthetic user message
        with base64 data URLs.

        Token IDs are recovered by tokenizing the response text, because the
        OpenAI chat-completion response does not carry raw output_ids.  Logprobs
        are extracted from ``choices[0].logprobs.content`` when available.
        """
        if request_input.raw_prompt is not None:
            messages = await self._normalize_messages(request_input.raw_prompt)
        else:
            image_data_urls = await serialize_image_inputs(mm_data.get("images", []))
            prompt_text = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(request_input.input_ids, skip_special_tokens=True),
            )

            content: list[dict] = [
                {"type": "image_url", "image_url": {"url": url}} for url in image_data_urls
            ]
            content.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": content}]

        need_logprobs = sampling_params.get("logprobs") is not None
        max_tokens = sampling_params.get("max_new_tokens", sampling_params.get("max_tokens", 1024))

        chat_payload = {
            "model": self.model_config.path,
            "messages": messages,
            "temperature": sampling_params.get("temperature", 1.0),
            "top_p": sampling_params.get("top_p", 1.0),
            "top_k": sampling_params.get("top_k", -1),
            "repetition_penalty": sampling_params.get("repetition_penalty", 1.0),
            "max_tokens": max_tokens,
            "stream": False,
            "logprobs": need_logprobs,
            "top_logprobs": 1 if need_logprobs else None,
            "detokenize": True,
        }
        # Remove None values
        chat_payload = {k: v for k, v in chat_payload.items() if v is not None}

        req_headers = {
            "x-request-id": str(request_input.request_id),
            "x-prompt-id": str(request_input.prompt_id),
            "x-version-tag": str(request_input.version_tag),
            "x-is-validate": str(request_input.is_validate).lower(),
        }
        if request_input.rollout_instance_id is not None:
            replica_id, dp_rank = request_input.rollout_instance_id
            req_headers["x-base-worker-id"] = str(replica_id)
            req_headers["x-target-dp-rank"] = str(dp_rank)
        if is_sticky_session:
            req_headers["x-manual-target-worker"] = "true"

        chat_url = f"{self.gateway_addr.rstrip('/')}/v1/chat/completions"

        chat_resp, base_worker_id, target_dp_rank = await self._post_chat(
            chat_url, chat_payload, req_headers
        )
        if chat_resp is None:
            psrl_logger.error(
                "Gateway /v1/chat/completions returned empty response for request_id=%s",
                request_input.request_id,
            )
            return None

        rollout_instance_id = (
            base_worker_id,
            int(target_dp_rank) if target_dp_rank is not None else 0,
        )

        choice = chat_resp["choices"][0]
        response_text: str = choice["message"].get("content") or ""
        finish_reason: str = choice.get("finish_reason") or "stop"
        interrupted = finish_reason == "abort"

        # Tokenize response text to recover token IDs.
        token_ids: list[int] = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.encode(response_text, add_special_tokens=False),
        )

        # Extract per-token logprobs from ChatLogProbs content list.
        log_probs: list[float] | None = None
        if need_logprobs:
            logprobs_field = choice.get("logprobs")
            if logprobs_field and isinstance(logprobs_field, dict):
                content_lps = logprobs_field.get("content")
                if content_lps:
                    log_probs = [entry["logprob"] for entry in content_lps]

        return TokenOutput(
            prompt_ids=request_input.input_ids,
            response_ids=token_ids,
            response_mask=[1] * len(token_ids),
            response_log_probs=log_probs,
            routed_experts=None,
            multi_modal_data=mm_data,
            stop_reason=finish_reason,
            interrupted=interrupted,
            update_status=PSRL_RequestStatus.ROLLOUT_COMPLETED,
            rollout_instance_id=rollout_instance_id,
        )

    async def pre_process_inputs(self, request: dict) -> TokenInput:
        version_tag = request["version_tag"]
        is_validate = request.get("validate", False)
        prompt_id = request.get("parent_id", request["uid"])
        rollout_instance_id = request.get("rollout_instance_id", None)

        multi_modal_data = None
        messages = None
        if "raw_prompt_ids" not in request:
            if request.get("input_ids", None) is not None:
                input_ids = request["input_ids"]
                raw_prompt_ids = _pre_process_inputs(self.tokenizer.pad_token_id, input_ids)
            elif "raw_prompt" in request:
                messages = list(request["raw_prompt"])

                # 1. extract images and videos from messages
                images, videos = await self.process_vision_info(messages)
                multi_modal_data = None
                if images is not None or videos is not None:
                    multi_modal_data = {"images": images, "videos": videos}

                # 2. apply chat template and tokenize
                raw_prompt_ids = await self.apply_chat_template(
                    messages,
                    images=images,
                    videos=videos,
                )
                request["raw_prompt_ids"] = np.array(raw_prompt_ids)
            else:
                raise ValueError(
                    "Request must contain 'raw_prompt_ids', 'raw_prompt', or 'input_ids' "
                    "to build generation input. Got keys: "
                    f"{request.keys()}"
                )
        else:
            raw_prompt_ids = request["raw_prompt_ids"]
            if isinstance(raw_prompt_ids, np.ndarray):
                raw_prompt_ids = raw_prompt_ids.tolist()
            if "raw_prompt" in request:
                messages = list(request["raw_prompt"])

        # Recover multi-modal data stored by prepare_generation_request().
        if multi_modal_data is None:
            multi_modal_data = request.get("multi_modal_data", None)

        raw_response_ids = request.get("raw_response_ids", [])
        if isinstance(raw_response_ids, np.ndarray):
            raw_response_ids = raw_response_ids.tolist()

        raw_prompt_ids.extend(raw_response_ids)

        return TokenInput(
            input_ids=raw_prompt_ids,
            request_id=request["uid"],
            prompt_id=prompt_id,
            rollout_instance_id=rollout_instance_id,
            version_tag=version_tag,
            cu_response_len=len(raw_response_ids),
            multi_modal_data=multi_modal_data,
            raw_prompt=messages,
            is_validate=is_validate,
            stop_token_ids=request.get("stop_token_ids", None),
        )

    async def _normalize_messages(self, messages: list[dict]) -> list[dict]:
        # Extract image refs
        image_refs = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                image_ref = extract_image_ref(part)
                if image_ref is None:
                    continue
                image_refs.append(image_ref)

        encoded_refs = await serialize_image_inputs(image_refs)
        encoded_iter = iter(encoded_refs)
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict) or extract_image_ref(part) is None:
                    continue
                image_url = part.get("image_url")
                if isinstance(image_url, dict) and isinstance(image_url.get("detail"), str):
                    detail = image_url["detail"]
                else:
                    detail = part.get("detail", None)

                part.clear()
                image_url = {"url": next(encoded_iter)}
                if detail is not None:
                    image_url["detail"] = detail
                part.update({"type": "image_url", "image_url": image_url})
        return messages

    def _get_sampling_params(self, request: TokenInput):
        is_validate = request.is_validate
        input_length = len(request.input_ids)

        # When max_model_len is not configured (None), fall back to prompt_length + response_length
        max_model_len = self.rollout_config.max_model_len
        if max_model_len is None:
            max_model_len = self.rollout_config.prompt_length + self.rollout_config.response_length
        max_possible_tokens = max_model_len - input_length
        if max_possible_tokens < 0:
            raise ValueError(
                f"Input length {input_length} exceeds the maximum model length {max_model_len}"
            )

        max_tokens = self.rollout_config.response_length + self.rollout_config.prompt_length - input_length
        max_tokens = max(0, min(max_tokens, max_possible_tokens))
        assert max_tokens <= max_possible_tokens, (
            f"max_tokens {max_tokens} exceeds available context space {max_possible_tokens}"
        )

        # NOTE: SMG Rust encodes top_k as uint32 (cannot represent negatives), so
        # -1 is silently clamped to 0 via `val.max(0) as u32`. Both 0 and -1 mean
        # "consider all tokens" in vLLM, but we normalize here to 0 to make the
        # conversion explicit and keep the value consistent with what vLLM receives.
        top_k = int(self.rollout_config.top_k)
        if top_k < 0:
            top_k = 0

        sampling_params = dict(
            n=1,
            logprobs=0,  # return sampled-token logprob for importance-sampling weight computation
            temperature=float(self.rollout_config.temperature),
            top_p=float(self.rollout_config.top_p),
            top_k=top_k,
            repetition_penalty=float(self.rollout_config.get("repetition_penalty", 1.0)),
            detokenize=False,
            max_new_tokens=max_tokens,
        )

        # override sampling params for validation
        if is_validate:
            val_config = self.config.train_actor_rollout_ref.rollout.val_kwargs
            val_top_k = int(val_config.top_k)
            if val_top_k < 0:
                val_top_k = 0
            sampling_params["top_k"] = val_top_k
            sampling_params["top_p"] = float(val_config.top_p)
            sampling_params["temperature"] = float(val_config.temperature)

        return sampling_params

    @staticmethod
    def _decode_routed_experts_payload(routed_experts_payload: dict | None) -> np.ndarray | None:
        """Decode compact routed experts payload to a numpy array.

        Expected payload schema:
            {
                "shape": [d0, d1, ...],
                "dtype": "int32",
                "data_base64": "..."
            }
        """
        if routed_experts_payload is None:
            return None

        shape = routed_experts_payload["shape"]
        dtype = np.dtype(routed_experts_payload["dtype"])
        data_base64 = routed_experts_payload["data_base64"]

        raw_bytes = base64.b64decode(data_base64)
        routed_experts = np.frombuffer(raw_bytes, dtype=dtype)

        expected_size = int(np.prod(shape, dtype=np.int64))
        if routed_experts.size != expected_size:
            raise ValueError(
                f"Decoded routed_experts size mismatch: got={routed_experts.size}, expected={expected_size}, shape={shape}"  # noqa: E501
            )

        return routed_experts.reshape(shape).copy()

    async def _post_generate(
        self,
        url: str,
        payload: dict,
        headers: dict[str, str],
        max_retries: int = 1,
    ) -> tuple[list[dict], str | None, str | None]:
        """POST to SMG /generate and return (responses, base_worker_id, target_dp_rank).

        SMG's /generate returns a JSON array of GenerateResponse objects alongside the
        worker-instance headers.  This helper reads both in a single aiohttp call so
        that the caller never has to deal with the mismatch between http_utils._post()
        (which assumes a JSON dict body) and the array response shape.

        HTTP I/O is handled by a dedicated background thread so that socket callbacks
        do not contend with the Ray actor's event loop.

        Returns:
            - responses: list of GenerateResponse dicts (may be empty on error)
            - base_worker_id: value of x-base-worker-id response header, or None
            - target_dp_rank: value of x-target-dp-rank response header, or None
        """
        if is_distributed_post_enabled():
            response = await request_json_maybe_distributed(
                "POST",
                url,
                payload=payload,
                headers=headers,
                max_retries=max_retries + 1,
            )
        else:
            io_thread = get_http_io_thread()
            response = await io_thread.request_json(
                "POST",
                url,
                payload=payload,
                headers=headers,
                max_retries=max_retries + 1,
            )
        base_worker_id = response.headers.get("x-base-worker-id", None)
        target_dp_rank = response.headers.get("x-target-dp-rank", None)

        responses = response.data
        # SMG /generate returns either a single dict or a list; normalise to list.
        if isinstance(responses, dict):
            responses = [responses]
        elif not isinstance(responses, list):
            psrl_logger.error(
                "_post_generate: unexpected response type %s from %s.",
                type(responses),
                url,
            )
            responses = []

        return responses, base_worker_id, target_dp_rank

    async def _post_chat(
        self,
        url: str,
        payload: dict,
        headers: dict[str, str],
        max_retries: int = 1,
    ) -> tuple[dict | None, str | None, str | None]:
        """POST to SMG /v1/chat/completions and return (response_dict, base_worker_id, target_dp_rank).

        HTTP I/O is handled by a dedicated background thread.

        Returns:
            - response: the parsed JSON dict (OpenAI ChatCompletionResponse), or None on error
            - base_worker_id: value of x-base-worker-id response header, or None
            - target_dp_rank: value of x-target-dp-rank response header, or None
        """
        if is_distributed_post_enabled():
            response = await request_json_maybe_distributed(
                "POST",
                url,
                payload=payload,
                headers=headers,
                max_retries=max_retries + 1,
            )
        else:
            io_thread = get_http_io_thread()
            response = await io_thread.request_json(
                "POST",
                url,
                payload=payload,
                headers=headers,
                max_retries=max_retries + 1,
            )
        base_worker_id = response.headers.get("x-base-worker-id", None)
        target_dp_rank = response.headers.get("x-target-dp-rank", None)

        if not isinstance(response.data, dict):
            psrl_logger.error(
                "_post_chat: unexpected response type %s from %s.",
                type(response.data),
                url,
            )
            return None, base_worker_id, target_dp_rank

        return response.data, base_worker_id, target_dp_rank

    @abstractmethod
    async def run(self, request: dict) -> tuple[TokenOutput | list[TokenOutput] | None, TerminateReason]:
        """Execute the agent loop for the given request.

        Args:
            request (dict): Input request to process.

        Returns:
            Tuple[TokenOutput | list[TokenOutput] | None, TerminateReason]:
                A tuple containing the output data (if any) and the termination reason.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_generate_fields(self) -> list[str]:
        """Determine which fields to select from the key-value store for generation.

        Returns:
            list[str] | None: List of field names to select from the key-value store,
                              or None to fetch all fields.
        """
        fields = [
            # dataset metadata
            "data_source",
            "reward_model",
            "extra_info",
            "reward_model_dicts",
            # request metadata
            "uid",
            "parent_id",
            "validate",
            "rollout_instance_id",
            # prompt
            "raw_prompt_ids",
            "raw_prompt",
            "input_ids",
            "raw_response_ids",
            # multi-modal data
            "multi_modal_data",
        ]

        return fields

    async def run_with_termination_handling(
        self, request: TensorDict, raise_on_error: bool = True
    ) -> tuple[TokenOutput | list[TokenOutput] | None, TerminateReason]:
        """Run the agent loop with termination event handling.

        This method wraps the run method to catch termination events and handle them appropriately.
        It enables timeouts and error handling based on the provided configuration.

        Args:
            request (KVBatchMeta): Input request to process.
            raise_on_error (bool): Whether to raise exceptions on errors.

        Returns:
            Tuple[TokenOutput | list[TokenOutput] | None, TerminateReason]:
                A tuple containing the output data (if any) and the termination reason.
        """
        try:
            prompt = {}
            for k, v in request.items():
                if isinstance(v, torch.Tensor):
                    prompt[k] = v[0]
                elif isinstance(v, NonTensorStack):
                    prompt[k] = v[0].data
                elif isinstance(v, NonTensorData):
                    prompt[k] = v.data
                else:
                    psrl_logger.exception(f"Unsupported type {type(v)} for key {k}")

            coro = self.run(prompt)
            output, terminate_reason = await coro
            if output is not None:
                return output, terminate_reason
            elif output is None and terminate_reason in (
                TerminateReason.ABORTED,
                TerminateReason.UNKNOWN,
                TerminateReason.TIMEOUT,
                TerminateReason.ENV_TIMEOUT,
                TerminateReason.ERROR,
                TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED,
                TerminateReason.MAX_TURNS_EXCEEDED,
            ):
                # Legitimate termination with no output (e.g. request aborted)
                return None, terminate_reason
            elif not raise_on_error:
                return None, TerminateReason.UNKNOWN
            else:
                raise RuntimeError("Agent loop run did not return a valid TokenOutput output.")
        except RequestAbortedByGatewayError as e:
            # PS Manager has already taken ownership of cleaning the request's data
            # flow (TQ entry cleared, staleness inventory updated).
            psrl_logger.info(
                "Request %s aborted by PS Manager (gateway returned `request_aborted`); "
                "ending data flow without retry.",
                e.request_id or "N/A",
            )
            return None, TerminateReason.ABORTED
        except asyncio.TimeoutError:
            psrl_logger.error(
                "Timeout in agent_loop.run for request %s (this can come from downstream calls, not only trajectory_timeout)",  # noqa: E501
                request.non_tensor_batch.get("uid", "N/A"),
                exc_info=True,
            )
            return None, TerminateReason.TIMEOUT
        except Exception as e:
            if not raise_on_error:
                psrl_logger.error(
                    f"Exception in agent_loop.run for request {request.non_tensor_batch.get('uid', 'N/A')}",
                    exc_info=True,
                )
                return None, TerminateReason.ERROR
            raise e
