import os
import asyncio
import base64
import logging
from contextlib import nullcontext
from abc import ABC, abstractmethod

import numpy as np
import ray
from omegaconf import DictConfig
from transformers import AutoTokenizer
from verl import DataProto
from vllm.sampling_params import RequestOutputKind

from psrl.utils.dataset.utils import _pre_process_inputs
# from psrl.utils.common.http_utils import post
from psrl.workers.gen_dplb.utils import TokenInput, TokenOutput
from psrl.workers.agent_loop.loops.utils import DummyConfig, TerminateReason
from psrl.workers.ps.request_status_tracker import PSRL_RequestStatus
from psrl.workers.agent_loop.sticky_session import sticky_session
from psrl.workers.config.model import HFModelConfig


psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

class AgentLoopBase(ABC):
    _class_initialized = False
    # Debug counter - print for first N requests to check input/output correctness
    _debug_request_count: int = 0
    _debug_max_requests: int = 3

    def __init__(
        self,
        trainer_config: DummyConfig,
        rollout_router: ray.actor.ActorHandle | str,
        reward_manager: ray.actor.ActorHandle,
        ps_manager_handle: ray.actor.ActorHandle,
        tokenizer: AutoTokenizer,
        **kwargs,
    ):
        """Initialize agent loop instance.
        Base class for agent loops that process requests and interact with LLM servers.

        Args:
            trainer_config (DummyConfig): Wrapper containing trainer configuration.
            rollout_router (ray.actor.ActorHandle): Router for distributing requests to LLM servers.
            ps_manager_handle (ray.actor.ActorHandle): Handle to parameter server manager.
            tokenizer (AutoTokenizer): Tokenizer for processing text messages.
            **kwargs: Additional keyword arguments.
        """
        self.init_class(trainer_config.config, **kwargs)
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
        self.loop = asyncio.get_running_loop()

    @classmethod
    def init_class(cls, config: DictConfig, **kwargs):
        """Perform heavy initialization work shared across all instances.

        This method is called only once per class to avoid redundant initialization.

        Args:
            config (DictConfig): Configuration object containing training settings.
            **kwargs: Additional keyword arguments from configuration.
        """
        if cls._class_initialized:
            return
        cls._class_initialized = True

        cls.prompt_length = config.gen_actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.gen_actor_rollout_ref.rollout.response_length

    async def generate_sequence(self, request: DataProto, is_sticky_session: bool = False) -> DataProto:
        # psrl_logger.info(f"Inside {request.non_tensor_batch['uid'][0]=}")
        request_input = self.pre_process_inputs(request)
        # psrl_logger.info(f"After process: {request_input.request_id=}")
        sampling_params = self._get_sampling_params(request_input)
        if self.config.psrl.rollout_gateway.enable:
            if not self.gateway_addr:
                raise RuntimeError("Rollout gateway is enabled but gateway address is empty.")

            # psrl_logger.info(f"generate_sequence: {request_input=}")
            psrl_logger.info(f"{request_input.request_id=} Sending generation request to gateway")

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
                print(
                    f"[PSRL-DEBUG] AgentLoop.generate_sequence req#{AgentLoopBase._debug_request_count} "
                    f"(id={request_input.request_id}): "
                    f"sampling_params sent to SMG = {payload_sampling_params}, "
                    f"input_ids len={len(request_input.input_ids)}, "
                    f"return_logprob={payload['return_logprob']}",
                    flush=True,
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

            # psrl_logger.info(f"{gen_responses=}")

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
                    log_probs = [
                        next((lp for lp in per_pos if lp is not None), 0.0)
                        for per_pos in raw_logprobs
                    ]

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
                print(
                    f"[PSRL-DEBUG] AgentLoop.generate_sequence response "
                    f"(id={request_input.request_id}): "
                    f"num_output_tokens={len(token_ids)}, "
                    f"num_logprobs={len(log_probs) if log_probs is not None else 'None'}, "
                    f"first_logprob={log_probs[0] if log_probs else None}, "
                    f"finish_reason={first.get('meta_info', {}).get('finish_reason')}",
                    flush=True,
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

            output = TokenOutput(
                token_ids=token_ids,
                log_probs=log_probs,
                routed_experts=routed_experts,
                stop_reason=finish_reason,
                interrupted=interrupted,
                update_status=PSRL_RequestStatus.ROLLOUT_COMPLETED,
                rollout_instance_id=rollout_instance_id,
            )
            psrl_logger.info(f"{request_input.request_id=} generated output with {len(token_ids)} tokens")  # noqa: E501
        else:
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

    def pre_process_inputs(self, request: DataProto) -> TokenInput:
        non_tensor_batch = request.non_tensor_batch
        version_tag = non_tensor_batch["version_tag"][0]
        is_validate = request.meta_info.get("validate", False)
        
        if "parent_id" in non_tensor_batch:
            req_prompt_id = non_tensor_batch["parent_id"][0]
        else:
            req_prompt_id = non_tensor_batch["uid"][0]
        # psrl_logger.info(f"Inner {req_prompt_id=}, {non_tensor_batch['uid'][0]=}")
        
        if "rollout_instance_id" in non_tensor_batch:
            rollout_instance_id = non_tensor_batch["rollout_instance_id"][0]
        else:
            rollout_instance_id = None
        
        if "raw_prompt_ids" not in non_tensor_batch:
            input_ids = request.batch["input_ids"][0]
            raw_prompt_ids = _pre_process_inputs(self.tokenizer.pad_token_id, input_ids)
        else:
            raw_prompt_ids = non_tensor_batch["raw_prompt_ids"][0]
            if isinstance(raw_prompt_ids, np.ndarray):
                raw_prompt_ids = raw_prompt_ids.tolist()

        if "raw_response_ids" in non_tensor_batch:
            raw_response_ids = non_tensor_batch["raw_response_ids"][0]
        else:
            raw_response_ids = []
        
        if isinstance(raw_response_ids, np.ndarray):
            raw_response_ids = raw_response_ids.tolist()
        raw_prompt_ids.extend(raw_response_ids)

        return TokenInput(
            input_ids=raw_prompt_ids,
            request_id=int(non_tensor_batch["uid"][0]),
            prompt_id=req_prompt_id,
            rollout_instance_id=rollout_instance_id,
            version_tag=version_tag,
            cu_response_len=len(raw_response_ids),
            is_validate=is_validate,
        )
        

    def _get_sampling_params(self, request: TokenInput):
        is_validate = request.is_validate
        input_length = len(request.input_ids)

        max_possible_tokens = self.rollout_config.max_model_len - input_length
        if max_possible_tokens < 0:
            raise ValueError(
                f"Input length {input_length} exceeds the maximum model length {self.rollout_config.max_model_len}"
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
            # output_kind=RequestOutputKind.CUMULATIVE,
            detokenize=False,
            # max_tokens=1,
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

    def _post_process_and_merge_reward(self, reward_result: dict[int, dict], outputs: DataProto) -> DataProto:
        """Merge the computed reward results back into the output DataProto.

        This method updates the output data with the reward scores and any additional
        information returned by the reward model.

        Args:
            reward_result (Dict[int, dict]): Computed reward results indexed by data item.
            outputs (DataProto): Original output data to be updated.

        Returns:
            DataProto: Updated output data with reward information.
        """
        if outputs.meta_info.get("validate", False):
            return outputs  # Skip merging for validation data

        filtered_request_ids = list(reward_result.keys())
        filtered_request_idxs = [
            idx for idx, uid in enumerate(outputs.non_tensor_batch["uid"].tolist()) if uid in filtered_request_ids
        ]
        outputs = outputs.select_idxs(filtered_request_idxs)
        request_ids = outputs.non_tensor_batch["uid"].tolist()

        rewards = []
        reward_extra_infos = []
        for request_id in request_ids:
            assert request_id in reward_result, f"Missing reward result for request ID: {request_id}"
            result = reward_result[request_id]
            rewards.append(result["reward_score"])
            extra_info = result.get("reward_extra_info", {})
            reward_extra_infos.append(extra_info)
        outputs.non_tensor_batch["reward_scores"] = np.array(rewards)
        outputs.non_tensor_batch["reward_extra_infos"] = np.array(reward_extra_infos, dtype=object)
        return outputs

    @abstractmethod
    async def run(self, request: DataProto) -> DataProto:
        """Execute the agent loop for the given request.

        Args:
            request (DataProto): Input request to process.

        Returns:
            DataProto: Processed response data.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    async def run_with_termination_handling(
        self, request: DataProto, raise_on_error: bool = True
    ) -> tuple[DataProto | None, TerminateReason]:
        """Run the agent loop with termination event handling.

        This method wraps the run method to catch termination events and handle them appropriately.
        It enables timeouts and error handling based on the provided configuration.

        Args:
            request (DataProto): Input request to process.
            raise_on_error (bool): Whether to raise exceptions on errors.

        Returns:
            DataProto: Processed response data.
        """
        try:
            coro = self.run(request)
            # output, terminate_reason = await asyncio.wait_for(
            #     coro, timeout=self.config.gen_actor_rollout_ref.rollout.agent.trajectory_timeout
            # )
            output, terminate_reason = await coro
            if output is not None and isinstance(output, DataProto):
                return output, terminate_reason
            elif not raise_on_error:
                return None, TerminateReason.UNKNOWN
            else:
                raise RuntimeError("Agent loop run did not return a valid DataProto output.")
        except asyncio.TimeoutError:
            psrl_logger.error(
                "Timeout in agent_loop.run for request %s (this can come from downstream calls, not only trajectory_timeout)",
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

