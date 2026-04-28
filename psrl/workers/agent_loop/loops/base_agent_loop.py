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
import transfer_queue as tq
from transfer_queue import KVBatchMeta
from tensordict import  NonTensorData, NonTensorStack
from transformers import AutoProcessor, AutoTokenizer
from verl.utils import tensordict_utils as tu
from verl.utils.chat_template import apply_chat_template, initialize_system_prompt
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.tokenizer import normalize_token_ids

from psrl.utils.dataset.utils import _pre_process_inputs
from psrl.workers.agent_loop.loops.utils import DictConfigWrap, TerminateReason
from psrl.workers.agent_loop.sticky_session import sticky_session

# from psrl.utils.common.http_utils import post
from psrl.workers.gen_dplb.utils import TokenInput, TokenOutput
from psrl.workers.ps.request_status_tracker import PSRL_RequestStatus

psrl_logger = logging.getLogger(__name__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class AgentLoopBase(ABC):
    # Debug counter - print for first N requests to check input/output correctness
    _debug_request_count: int = 0
    _debug_max_requests: int = 3

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
        tools: list[dict] = None,
        images: list[Image.Image] = None,
        videos: list[tuple[torch.Tensor, dict]] = None,
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

    async def compute_reward_score(self, data: TokenOutput) -> TokenOutput | None:
        """Compute reward score for the generated response and merge it into the output.

        This function sends the generated response to the reward manager and waits for the computed reward score. If the reward computation is successful, it merges the reward score into the output data structure. If the request is aborted during reward computation (e.g., due to staleness check failure), it returns None to indicate that the request should be aborted.

        Args:
            data (TokenOutput): The output data structure containing the generated response and associated metadata.
        Returns:
            TokenOutput | None: The updated output data structure with the computed reward score, or None if the request was aborted during reward computation.
        """
        reward_requests = tu.get_tensordict({
            "prompts": torch.tensor(data.prompt_ids, dtype=torch.int64).unsqueeze(0),
            "responses": torch.tensor(data.response_ids, dtype=torch.int64).unsqueeze(0),
            "multi_modal_data": np.array([data.multi_modal_data], dtype=object),
            "num_turns": np.array([data.num_turns]),
            "tool_extra_fields": np.array([data.extra_fields], dtype=object),
        })
        reward_result = await self.reward_manager.compute_score.remote(reward_requests)

        if not self.config.reward.launch_reward_fn_async:
            data.reward_score = reward_result["reward_score"]
            data.extra_fields["reward_extra_info"] = reward_result["reward_extra_info"]
            data.extra_fields["reward_metrics"] = reward_result.get("reward_metrics", {})

        return data

    async def generate_sequence(self, request: dict, is_sticky_session: bool = False) -> "TokenOutput":
        # psrl_logger.info(f"Inside {request.non_tensor_batch['uid'][0]=}")
        request_input: TokenInput = await self.pre_process_inputs(request)
        # psrl_logger.info(f"After process: {request_input.request_id=}")
        sampling_params = self._get_sampling_params(request_input)
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

            # TODO(linsh): currently not supported yet in smg caller side.
            if has_images:
                return await self._generate_via_chat_completions(
                    request_input, sampling_params, is_sticky_session, mm_data
                )

            # ── Text-only: fast /generate path ──────────────────────────────
            # psrl_logger.info(f"generate_sequence: {request_input=}")
            psrl_logger.info(f"{request_input.request_id=} Sending generation request to gateway")
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
            req_headers["x-manual-target-worker"] = "true"

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

        psrl_logger.info(
            "[PSRL-DEBUG] request_id=%s: SMG /generate payload sampling_params=%s, "
            "input_ids len=%d, return_logprob=%s",
            request_input.request_id,
            payload_sampling_params,
            len(request_input.input_ids),
            payload["return_logprob"],
        )
        AgentLoopBase._debug_request_count += 1
        if AgentLoopBase._debug_request_count <= AgentLoopBase._debug_max_requests:
            psrl_logger.info(
                f"[PSRL-DEBUG] AgentLoop.generate_sequence req#{AgentLoopBase._debug_request_count} "
                f"(id={request_input.request_id}): "
                f"sampling_params sent to SMG = {payload_sampling_params}, "
                f"input_ids len={len(request_input.input_ids)}, "
                f"return_logprob={payload['return_logprob']}"
            )

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

        psrl_logger.info(f"{gen_responses=}")

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

        psrl_logger.info(
            "[PSRL-DEBUG] request_id=%s: SMG /generate response: "
            "num_output_tokens=%d, has_logprobs=%s, "
            "first_logprob=%s, finish_reason=%s",
            request_input.request_id,
            len(token_ids),
            log_probs is not None,
            log_probs[0] if log_probs else None,
            first.get("meta_info", {}).get("finish_reason"),
        )
        if AgentLoopBase._debug_request_count <= AgentLoopBase._debug_max_requests:
            psrl_logger.info(
                f"[PSRL-DEBUG] AgentLoop.generate_sequence response "
                f"(id={request_input.request_id}): "
                f"num_output_tokens={len(token_ids)}, "
                f"num_logprobs={len(log_probs) if log_probs is not None else 'None'}, "
                f"first_logprob={log_probs[0] if log_probs else None}, "
                f"finish_reason={first.get('meta_info', {}).get('finish_reason')}"
            )

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

        # psrl_logger.info(f"Before return TokenOutput...")

        psrl_logger.info(f"{request_input.request_id=} generated output with {len(token_ids)} tokens")  # noqa: E501
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
        (image fetch → pixel preprocessing → mm_inputs proto) that the /generate
        route does not yet implement.  Images are embedded in the messages as
        ``image_url`` content parts using base64 data-URLs.

        Token IDs are recovered by tokenizing the response text, because the
        OpenAI chat-completion response does not carry raw output_ids.  Logprobs
        are extracted from ``choices[0].logprobs.content`` when available.
        """
        from psrl.utils.rollout.vision_utils import pil_images_to_base64

        images: list = mm_data["images"]
        image_data_urls: list[str] = await pil_images_to_base64(images)

        # Rebuild the messages list from token IDs by decoding the full sequence.
        # The prompt was already tokenized with vision tokens embedded, so we
        # cannot trivially re-derive the original messages.  Instead we send the
        # pre-tokenized prompt as a single user message with the images prepended
        # as image_url content parts — SMG's chat pipeline will handle vision
        # preprocessing independently on its side.
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
        psrl_logger.info(
            "[PSRL-DEBUG] request_id=%s: SMG /v1/chat/completions (multimodal, %d images)",
            request_input.request_id,
            len(images),
        )

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

        psrl_logger.info(
            "[PSRL-DEBUG] request_id=%s: chat/completions response: "
            "num_output_tokens=%d, has_logprobs=%s, finish_reason=%s",
            request_input.request_id,
            len(token_ids),
            log_probs is not None,
            finish_reason,
        )

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
                request["raw_prompt_ids"] = np.array([raw_prompt_ids])
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

        # Recover multi-modal data stored by prepare_generation_request().
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
            is_validate=is_validate,
        )

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

        Returns:
            - responses: list of GenerateResponse dicts (may be empty on error)
            - base_worker_id: value of x-base-worker-id response header, or None
            - target_dp_rank: value of x-target-dp-rank response header, or None
        """
        import aiohttp

        from psrl.utils.common.http_utils import _ensure_http_client

        client = await _ensure_http_client()
        retry_count = 0
        while retry_count <= max_retries:
            try:
                async with client.post(url, json=payload, headers=headers) as response:
                    base_worker_id = response.headers.get("x-base-worker-id", None)
                    target_dp_rank = response.headers.get("x-target-dp-rank", None)

                    if response.status >= 400:
                        response_text = await response.text()
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=response_text,
                            headers=response.headers,
                        )

                    responses = await response.json(content_type=None)
                    # SMG /generate returns either a single dict or a list; normalise to list.
                    if isinstance(responses, dict):
                        responses = [responses]
                    elif not isinstance(responses, list):
                        psrl_logger.error(
                            "_post_generate: unexpected response type %s from %s",
                            type(responses),
                            url,
                        )
                        responses = []

                    return responses, base_worker_id, target_dp_rank
            except Exception as e:
                retry_count += 1
                psrl_logger.info(
                    "Error: %s, retrying... (attempt %s/%s, url=%s)",
                    e,
                    retry_count,
                    max_retries + 1,
                    url,
                )
                if retry_count > max_retries:
                    raise
                await asyncio.sleep(1)

        return [], None, None

    async def _post_chat(
        self,
        url: str,
        payload: dict,
        headers: dict[str, str],
        max_retries: int = 1,
    ) -> tuple[dict | None, str | None, str | None]:
        """POST to SMG /v1/chat/completions and return (response_dict, base_worker_id, target_dp_rank).

        Returns:
            - response: the parsed JSON dict (OpenAI ChatCompletionResponse), or None on error
            - base_worker_id: value of x-base-worker-id response header, or None
            - target_dp_rank: value of x-target-dp-rank response header, or None
        """
        import aiohttp

        from psrl.utils.common.http_utils import _ensure_http_client

        client = await _ensure_http_client()
        retry_count = 0
        while retry_count <= max_retries:
            try:
                async with client.post(url, json=payload, headers=headers) as response:
                    base_worker_id = response.headers.get("x-base-worker-id", None)
                    target_dp_rank = response.headers.get("x-target-dp-rank", None)

                    if response.status >= 400:
                        response_text = await response.text()
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=response_text,
                            headers=response.headers,
                        )

                    resp_json = await response.json(content_type=None)
                    return resp_json, base_worker_id, target_dp_rank
            except Exception as e:
                retry_count += 1
                psrl_logger.info(
                    "Error: %s, retrying... (attempt %s/%s, url=%s)",
                    e,
                    retry_count,
                    max_retries + 1,
                    url,
                )
                if retry_count > max_retries:
                    raise
                await asyncio.sleep(1)

        return None, None, None

    @abstractmethod
    async def run(self, request: dict) -> tuple[TokenOutput | None, TerminateReason]:
        """Execute the agent loop for the given request.

        Args:
            request (dict): Input request to process.

        Returns:
            Tuple[TokenOutput | None, TerminateReason]:
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
            # metadata
            "uid",
            "parent_id",
            "version_tag",
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
        self, request: KVBatchMeta, raise_on_error: bool = True
    ) -> tuple[TokenOutput | None, TerminateReason]:
        """Run the agent loop with termination event handling.

        This method wraps the run method to catch termination events and handle them appropriately.
        It enables timeouts and error handling based on the provided configuration.

        Args:
            request (KVBatchMeta): Input request to process.
            raise_on_error (bool): Whether to raise exceptions on errors.

        Returns:
            Tuple[TokenOutput | None, TerminateReason]:
                A tuple containing the output data (if any) and the termination reason.
        """
        try:
            fields = self.get_generate_fields()
            if fields:
                data = await tq.async_kv_batch_get(keys=request.keys, partition_id=request.partition_id, select_fields=fields)
            else:
                data = await tq.async_kv_batch_get_by_meta(request)
            
            prompt = {}
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    prompt[k] = v[0]
                elif isinstance(v, NonTensorStack):
                    prompt[k] = v[0].data
                elif isinstance(v, NonTensorData):
                    prompt[k] = v.data
                else:
                    psrl_logger.exception(f"Unsupported type {type(v)} for key {k}")

            coro = self.run(prompt)
            # output, terminate_reason = await asyncio.wait_for(
            #     coro, timeout=self.config.gen_actor_rollout_ref.rollout.agent.trajectory_timeout
            # )
            output, terminate_reason = await coro
            if output is not None and isinstance(output, TokenOutput):
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
