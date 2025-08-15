import logging
import os
import uuid
import asyncio
import numpy as np
from deprecated import deprecated
from contextlib import contextmanager
from copy import deepcopy
from collections.abc import Sequence
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from typing import Any, Dict, Optional, Union, List, Tuple, cast

import torch
import torch.distributed

from vllm import LLM, SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.distributed import parallel_state as vllm_ps
from vllm.inputs import PromptType, TokensPrompt
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind

from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids

def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)

class PSRL_vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
       
        # Monkey patch adapted from NeMo-RL for vLLM to ensure RAY_ADDRESS is set in Ray actors.
        # (https://github.com/NVIDIA-NeMo/RL/blob/124ca30417dafb5b03ba5c1948f8252ddbce0d06/nemo_rl/models/generation/vllm.py#L203)
        try:
            import vllm.utils
            from vllm.utils import cuda_is_initialized, is_in_ray_actor

            def _patched_maybe_force_spawn():
                """Patched version of vllm.utils._maybe_force_spawn.

                This patch changes an `elif is_in_ray_actor()` to an `if` statement.
                This ensures that `os.environ["RAY_ADDRESS"]` is set when running
                within a Ray actor, even if CUDA has already been initialized.
                This is crucial for vLLM workers to connect back to the Ray cluster.
                """
                import os

                if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == "spawn":
                    return

                reason = None
                if cuda_is_initialized():
                    reason = "CUDA is initialized"

                if is_in_ray_actor():
                    # even if we choose to spawn, we need to pass the ray address
                    # to the subprocess so that it knows how to connect to the ray cluster.
                    # env vars are inherited by subprocesses, even if we use spawn.
                    import ray

                    os.environ["RAY_ADDRESS"] = ray.get_runtime_context().gcs_address
                    if reason is None:
                        reason = "In a Ray actor and can only be spawned"

                if reason is not None:
                    psrl_logger.warning(
                        "We must use the `spawn` multiprocessing start method. "
                        "Overriding VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. "
                        "See https://docs.vllm.ai/en/latest/getting_started/"
                        "troubleshooting.html#python-multiprocessing "
                        "for more information. Reason: %s",
                        reason,
                    )
                    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

            vllm.utils._maybe_force_spawn = _patched_maybe_force_spawn
            psrl_logger.info("Successfully patched vllm.utils._maybe_force_spawn.")
        
        except (ImportError, AttributeError):
            # vllm not installed or has a different structure, skipping patch.
            pass

        super().__init__()
        self.config = config

        tensor_parallel_size = config.get("tensor_model_parallel_size", 1)
        pipeline_parallel_size = config.get("pipeline_model_parallel_size", 1)
        model_parallel_size = tensor_parallel_size * pipeline_parallel_size
        assert tensor_parallel_size <= torch.distributed.get_world_size(), "tensor parallel size should be less than or equal to the world size"
        assert pipeline_parallel_size == 1 or config.mode == "psrl_async", "pipeline parallel is only supported in psrl_async mode"
        
        # For async engine and model parallel, we only run the inference engine on the first rank.
        # The inner parallel workers are handled by vLLM + Ray.
        if config.mode == "psrl_async" and model_parallel_size > 1:
            import os
            if os.environ.get("LOCAL_RANK") != '0':
                self.inference_engine = None
                return

        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)
        
        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)
        
        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        # LoRA configuration
        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = (
            {}
            if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs
            else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        )
        
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        if config.mode == "psrl_async" and model_parallel_size > 1:
            # Configure vLLM for tensor/pipeline parallelism within Ray
            # Reset CUDA_VISIBLE_DEVICES to allow vLLM to manage GPU assignment
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

            distributed_executor_backend = "ray"
        elif config.mode == "sync":
            distributed_executor_backend = "external_launcher"
        else:
            distributed_executor_backend = None # auto detect

        llm_kwargs = dict(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            distributed_executor_backend=distributed_executor_backend,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=torch.cuda.get_device_capability()[0] >= 8,
            trust_remote_code=trust_remote_code,
            worker_extension_cls="psrl.workers.gen.vllm_extension.vLLMWorkerExtension",
            seed=kwargs.get("seed", 0),
            **lora_kwargs,
            **engine_kwargs,
        )

        if config.mode == "psrl_async":
            self.inference_engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**llm_kwargs))
        else:
            self.inference_engine = LLM(**llm_kwargs)

        # Offload vllm model to reduce peak memory usage
        if load_format == "dummy" and config.free_cache_engine:
            self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
            output_kind=RequestOutputKind.CUMULATIVE,
        )

        # we may detokenize the result all together later
        kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)) and k != "seed":
                kwargs[k] = config.get(k)

        psrl_logger.info(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        """Context manager to temporarily update sampling parameters."""
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)
            
    def pre_process_inputs(
        self, 
        prompts: DataProto,
        kwargs: dict
    ) -> Tuple[Union[PromptType, Sequence[PromptType]], dict[str, Any]]:
        """Pre-process the prompts to convert them into vLLM inputs."""
        # 1. remove left padding -> raw_prompt_ids
        # 2. concat raw_prompt_ids and raw_response_ids -> prompt_token_ids in `vllm_inputs`
        # 3. add multi_modal_data if exists
        # 4. sampling params configuration
        
        # TODO: 在外部移除 single-request 的 padding
        
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        batch_size = idx.size(0)
        
        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            # Remove the left padding in the prompt token_id
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError(f"vllm sharding manager is not work properly with "
                               f"{batch_size=} v.s. {len(non_tensor_batch['raw_prompt_ids'])=}.")

        raw_prompt_ids = non_tensor_batch["raw_prompt_ids"]
        if "raw_response_ids" in non_tensor_batch:
            raw_response_ids = non_tensor_batch["raw_response_ids"]
        else:
            raw_response_ids = np.fromiter(([] for _ in range(batch_size)), dtype=object)

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids_, raw_response_ids_, multi_modal_data in zip(
                raw_prompt_ids, raw_response_ids, non_tensor_batch["multi_modal_data"]
            ):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids_ + raw_response_ids_, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [{"prompt_token_ids": raw_prompt_ids_ + raw_response_ids_} for raw_prompt_ids_, raw_response_ids_ in zip(raw_prompt_ids, raw_response_ids)]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }
        else:
            kwargs = {
                "n": 1, # we repeat the request manually to support partial rollout
                "prompt_logprobs": 0 if self.config.interrupt_as_prompt else None,
            }
            
        return vllm_inputs, kwargs
            
    def post_process_outputs(
        self,
        prompts: DataProto,
        outputs: Union[Union[RequestOutput, PoolingRequestOutput], list[Union[RequestOutput, PoolingRequestOutput]]],
    ) -> DataProto:
        """Post-process the vllm outputs to convert them into DataProto."""
        # 1. collect response_ids, response_len, interrupted
        # 2. collect log_probs if required
        # 3. concat new response_ids to raw_response_ids, update response_unpadded_len and rollout_log_probs
        # 4. pad response to the right side
        # 5. construct position_ids and attention_mask (final)
        # 6. construct the final data proto
        
        # partial rollout 情况下需要 response_ids 和 log_probs -> outputs

        if isinstance(outputs, (RequestOutput, PoolingRequestOutput)):
            outputs = [outputs]
        
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        batch_size = idx.size(0)
        non_tensor_batch = prompts.non_tensor_batch
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]
        
        response = []
        response_unpadded_len = []
        interrupted = []
        rollout_log_probs = []
        for output in outputs:
            for sample_id in range(len(output.outputs)):
                response_ids = output.outputs[sample_id].token_ids
                response.append(response_ids)
                response_unpadded_len.append(len(response_ids))
                interrupted.append(output.outputs[sample_id].finish_reason == "abort")
                # if inference logprobs is required, we need to collect the log probabilities
                if (
                    self.config.enable_inference_engine_log_prob and
                    hasattr(output.outputs[sample_id], 'logprobs') and
                    output.outputs[sample_id].logprobs is not None
                ):
                    log_prob_list = []
                    if self.config.interrupt_as_prompt:
                        curr_response_len = non_tensor_batch.get("response_unpadded_len", 0)
                        # Collect log probs only when the request finished normally
                        # The response log probs are collected in two parts:
                        # 1. The log probs of the accumulated response tokens (in current prompt tokens)
                        # 2. The log probs of the current response tokens
                        if output.outputs[sample_id].finish_reason != "abort" and curr_response_len > 0:
                            # partial response log probs from prompt log probs
                            prompt_token_ids = output.prompt_token_ids
                            for i, logprob in enumerate(output.prompt_logprobs[-curr_response_len:]):
                                log_prob_list.append(logprob[prompt_token_ids[i - curr_response_len]].logprob)
                            # new response log probs from decode log probs
                            for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                                log_prob_list.append(logprob[response_ids[i]].logprob)
                    else:
                        # Response log probs from decode log probs
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            log_prob_list.append(logprob[response_ids[i]].logprob)
                    rollout_log_probs.append(log_prob_list)

        non_tensor_batch["interrupted"] = np.array(interrupted, dtype=bool)
        raw_response_ids = non_tensor_batch["raw_response_ids"]
        # Reconstruct the raw response ids by concatenating the previous raw response ids
        # with the new response ids.
        response = raw_response_ids + np.fromiter(response, dtype=object)
        non_tensor_batch["raw_response_ids"] = response
        if "response_unpadded_len" in non_tensor_batch:
            curr_response_unpadded_len = non_tensor_batch["response_unpadded_len"]
        else:
            curr_response_unpadded_len = [0] * batch_size
        response_unpadded_len = [
            curr_response_unpadded_len[i] + response_unpadded_len[i] for i in range(batch_size)
        ]
        non_tensor_batch["response_unpadded_len"] = np.array(response_unpadded_len, dtype=int)

        # TODO: optimize the DataProto construction to packing
        # Here we pad the response to the right side for both interrupted and completed requests.
        response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
        if self.config.enable_inference_engine_log_prob:
            if "rollout_log_probs" in non_tensor_batch:
                curr_rollout_log_probs = non_tensor_batch["rollout_log_probs"]
                curr_rollout_log_probs = curr_rollout_log_probs.reshape(batch_size, -1)
            else:
                curr_rollout_log_probs = np.empty((batch_size, 0), dtype=object)
            non_tensor_batch["rollout_log_probs"] = np.concatenate([curr_rollout_log_probs, np.array(rollout_log_probs, dtype=object).reshape(batch_size, -1)], axis=-1)

        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences (including left padding & right padding)
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)
    
    def add_requests(self, prompts: DataProto, **kwargs):
        """Add requests to the vLLM inference engine."""
        vllm_inputs, kwargs = self.pre_process_inputs(prompts, kwargs)
        parsed_vllm_inputs = cast(Union[PromptType, Sequence[PromptType]], vllm_inputs)
        if isinstance(parsed_vllm_inputs, (str, dict)):
            # Convert a single prompt to a list.
            parsed_vllm_inputs = [parsed_vllm_inputs]
        
        with self.update_sampling_params(**kwargs):
            for prompt in parsed_vllm_inputs:
                request_id = str(self.get_next_request_id())
                self.inference_engine.llm_engine.add_request(
                    request_id,
                    prompt,
                    self.sampling_params,
                    priority=0,
                )

    @deprecated("vllm_rollout.step_all is not supported.")
    @torch.no_grad()
    def step_all(self) -> list[Union[RequestOutput, PoolingRequestOutput]]:
        outputs: list[Union[RequestOutput, PoolingRequestOutput]] = []
        while self.inference_engine.llm_engine.has_unfinished_requests():
            step_outputs = self.inference_engine.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
        return sorted(outputs, key=lambda x: int(x.request_id))
    
    @deprecated("vllm_rollout.step is not supported.")
    @torch.no_grad()
    def step(self) -> list[Union[RequestOutput, PoolingRequestOutput]]:
        return self.inference_engine.llm_engine.step()

    @GPUMemoryLogger(role="vllm rollout spmd", logger=psrl_logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences from the prompts using vLLM."""
        vllm_inputs, kwargs = self.pre_process_inputs(prompts, kwargs)
        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            # the inference_engine will handle the request_id internally
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            return self.post_process_outputs(prompts, outputs)
    
    async def generate_sequence_task(
        self,
        idx: int,
        prompt_tokens: Union[Dict[str, Any], List[int]],
        sampling_params: SamplingParams = None,
        uid: Optional[str] = None
    ) -> Tuple[int, RequestOutput]:
        """Generate a single sequence asynchronously using vLLM."""
        if sampling_params is None:
            sampling_params = self.sampling_params
        if isinstance(prompt_tokens, list):
            prompt_tokens = {'prompt_token_ids': prompt_tokens}

        task = self.inference_engine.generate(
            prompt=TokensPrompt(**prompt_tokens),
            sampling_params=sampling_params,
            request_id=str(uuid.uuid4()) if uid is None else uid,
        )
        async for output in task:
            last_output = output
        return idx, last_output
    
    @GPUMemoryLogger(role="vllm stream rollout", logger=psrl_logger)
    @torch.no_grad()
    async def generate_sequences_async(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences from the prompts using vLLM asynchronously."""
        vllm_inputs, kwargs = self.pre_process_inputs(prompts, kwargs)
        sample_ids = prompts.non_tensor_batch.get("uid", None)
        curr_response_unpadded_len = prompts.non_tensor_batch.get("response_unpadded_len", [0] * len(vllm_inputs))
        assert sample_ids is not None, \
            "sample_ids must be provided in the prompts.non_tensor_batch"
        
        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            tasks = []
            for prompt_idx, (vllm_input, sample_id, curr_response_len) in enumerate(zip(vllm_inputs, sample_ids, curr_response_unpadded_len)):
                partial_kwargs = dict(
                    max_tokens=self.config.response_length - curr_response_len,
                )
                with self.update_sampling_params(**partial_kwargs):
                    tasks.append(
                        self.generate_sequence_task(
                            prompt_idx,
                            vllm_input,
                            self.sampling_params,
                            str(sample_id),
                        )
                    )
        
            completed_rollout = []
            for completed_task in asyncio.as_completed(tasks):
                prompt_idx, output = await completed_task
                completed_rollout.append(self.post_process_outputs(prompts[prompt_idx:prompt_idx+1], output))
        
        return DataProto.concat(completed_rollout)

    @GPUMemoryLogger(role="vllm stream rollout", logger=psrl_logger)
    @torch.no_grad()
    async def raw_generate_sequences_async(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences from the prompts using vLLM asynchronously w/o post-processing."""
        vllm_inputs, kwargs = self.pre_process_inputs(prompts, kwargs)
        sample_ids = prompts.non_tensor_batch.get("uid", None)
        curr_response_unpadded_len = prompts.non_tensor_batch.get("response_unpadded_len", [0] * len(vllm_inputs))
        assert sample_ids is not None, \
            "sample_ids must be provided in the prompts.non_tensor_batch"
        
        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            tasks = []
            for prompt_idx, (vllm_input, sample_id, curr_response_len) in enumerate(zip(vllm_inputs, sample_ids, curr_response_unpadded_len)):
                partial_kwargs = dict(
                    max_tokens=self.config.response_length - curr_response_len,
                )
                with self.update_sampling_params(**partial_kwargs):
                    tasks.append(
                        self.generate_sequence_task(
                            prompt_idx,
                            vllm_input,
                            self.sampling_params,
                            str(sample_id),
                        )
                    )
        
            vllm_outputs = []
            for completed_task in asyncio.as_completed(tasks):
                output = await completed_task
                vllm_outputs.append(output)
        
        return vllm_outputs

    async def interrupt_all_requests_async(self) -> int:
        """Interrupt all requests (both running and waiting) asynchronously."""
        # Get request IDs of all running and waiting requests from the scheduler
        request_ids_to_abort = await self.inference_engine.waiting_and_running_queue()
        interrupted_request_num = len(request_ids_to_abort)
        
        if interrupted_request_num > 0:
            await self.inference_engine.abort(request_ids_to_abort)
            psrl_logger.debug(f"Interrupted {interrupted_request_num} requests.")
        else:
            psrl_logger.debug("No requests to interrupt.")
            
        return interrupted_request_num

    async def interrupt_requests_async(self, request_ids):
        """Interrupt specific requests asynchronously."""
        if len(request_ids) > 0:
            request_ids = [str(request_id) for request_id in request_ids]
            await self.inference_engine.abort(request_ids)
    
    async def waiting_and_running_queue_size(self):
        """Get the size of waiting and running queues."""
        return len(await self.inference_engine.waiting_and_running_queue())
    
    async def get_engine_status(self):
        """Get the status of the vLLM engine."""
        if self.inference_engine is None:
            return {"status": "not initialized"}
        
        status = {
            "waiting_and_running_queue_size": await self.waiting_and_running_queue_size(),
        }
        return status