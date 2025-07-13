import logging
import os
import uuid
import asyncio
from contextlib import contextmanager
from copy import deepcopy
from collections.abc import Sequence
from typing import Any, Dict, Optional, Union, List, Tuple, cast

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
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
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))


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
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
        """
       
        # Monkey patch for vLLM to ensure RAY_ADDRESS is set in Ray actors.
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

        tensor_parallel_size = config.get("tensor_model_parallel_size", 1)
        pipeline_parallel_size = config.get("pipeline_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), "tensor parallel size should be less than or equal to the world size"
        model_parallel_size = tensor_parallel_size * pipeline_parallel_size
        if config.mode == "psrl_async" and model_parallel_size > 1:
            import os
            if os.environ.get("LOCAL_RANK") != '0':
                self.inference_engine = None
                return

        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), "disable CUDA graph (enforce_eager = False) if free cache engine"

        assert tensor_parallel_size <= torch.distributed.get_world_size(), "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)
        
        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(model_hf_config.llm_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(model_hf_config.text_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")

            assert max_position_embeddings >= config.prompt_length + config.response_length, "model context length should be greater than total sequence length"

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format
        if config.mode == "psrl_async":
            load_format = "auto"

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = {} if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
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
            distributed_executor_backend = None

        if config.mode == "psrl_async":
            self.inference_engine = AsyncLLM.from_engine_args(
                AsyncEngineArgs(
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
                    enable_prefix_caching=True,
                    trust_remote_code=trust_remote_code,
                    worker_extension_cls="psrl.workers.gen.vllm_extension.vLLMWorkerExtension",
                    seed=kwargs.get("seed", 0),
                    **lora_kwargs,
                    **engine_kwargs,
                )
            )
        else:
            if pipeline_parallel_size > 1:
                raise NotImplementedError("Pipeline parallel is not supported in synchronous LLM rollout yet.")

            self.inference_engine = LLM(
                model=model_path,
                enable_sleep_mode=True,
                tensor_parallel_size=tensor_parallel_size,
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
                enable_prefix_caching=True,
                trust_remote_code=trust_remote_code,
                worker_extension_cls="psrl.workers.gen.vllm_extension.vLLMWorkerExtension",
                seed=config.get("seed", 0),
                **lora_kwargs,
                **engine_kwargs,
            )

        # Offload vllm model to reduce peak memory usage
        if load_format == "dummy":
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
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)
        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
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
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)
            
    def pre_process_inputs(
        self, 
        prompts: DataProto,
        kwargs: dict
    ) -> Tuple[Union[PromptType, Sequence[PromptType]], dict[str, Any]]:
        """Pre-process the prompts to convert them into vLLM inputs."""
        
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        batch_size = idx.size(0)
        non_tensor_batch = prompts.non_tensor_batch
        
        if "raw_prompt_ids" not in non_tensor_batch:
            # Remove the left padding in the prompt token_id
            non_tensor_batch["raw_prompt_ids"] = np.array([_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError(f"vllm sharding manager is not work properly with {batch_size=} v.s. {len(non_tensor_batch['raw_prompt_ids'])=}.")

        raw_prompt_ids = non_tensor_batch["raw_prompt_ids"]
        if "raw_response_ids" in non_tensor_batch:
            raw_response_ids = non_tensor_batch["raw_response_ids"]
        else:
            raw_response_ids = np.fromiter(([] for _ in range(batch_size)), dtype=object)

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids_, raw_response_ids_, multi_modal_data in zip(raw_prompt_ids, raw_response_ids, non_tensor_batch["multi_modal_data"]):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids_ + raw_response_ids_, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [{"prompt_token_ids": raw_prompt_ids_ + raw_response_ids_} for raw_prompt_ids_, raw_response_ids_ in zip(raw_prompt_ids, raw_response_ids)]

        '''
        vllm_inputs include: prompt_token_ids, (multi_modal_data)
        '''
        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

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
            # TODO: try **
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
                interrupted.append(output.outputs[sample_id].finish_reason == "abort")
                response_unpadded_len.append(len(response_ids))
                if (
                    self.config.enable_inference_engine_log_prob and
                    hasattr(output.outputs[sample_id], 'logprobs') and
                    output.outputs[sample_id].logprobs is not None
                ):
                    log_prob_list = []
                    if self.config.interrupt_as_prompt:
                        curr_response_len = non_tensor_batch.get("response_unpadded_len", 0)
                        if output.outputs[sample_id].finish_reason != "abort" and curr_response_len > 0:
                            # partial response log probs from prompt log probs
                            prompt_token_ids = output.prompt_token_ids
                            for i, logprob in enumerate(output.prompt_logprobs[-curr_response_len:]):
                                log_prob_list.append(logprob[prompt_token_ids[i - curr_response_len]].logprob)
                            # new response log probs from decode log probs
                            for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                                log_prob_list.append(logprob[response_ids[i]].logprob)
                    else:
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            log_prob_list.append(logprob[response_ids[i]].logprob)
                    rollout_log_probs.append(log_prob_list)
        non_tensor_batch["interrupted"] = np.array(interrupted, dtype=bool)
        raw_response_ids = non_tensor_batch["raw_response_ids"]
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

        response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
        # response_unpadded_len = torch.tensor(response_unpadded_len).to(idx.device)
        if self.config.enable_inference_engine_log_prob:
            if "rollout_log_probs" in non_tensor_batch:
                curr_rollout_log_probs = non_tensor_batch["rollout_log_probs"]
            else:
                curr_rollout_log_probs = np.fromiter(([] for _ in range(batch_size)), dtype=object)
            non_tensor_batch["rollout_log_probs"] = curr_rollout_log_probs + rollout_log_probs
            # rollout_log_probs = pad_2d_list_to_length(rollout_log_probs, -1, max_length=self.config.response_length).to(idx.device)
            # rollout_log_probs = rollout_log_probs.to(torch.float32)

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
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                # "response_unpadded_lens": response_unpadded_len,
                "input_ids": seq,  # here input_ids become the whole sentences (including left padding & right padding)
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        # if "multi_modal_data" in prompts[0].keys() and np.all(non_tensor_batch["interrupted"]):
        #     non_tensor_batch["multi_modal_data"] = np.array([prompts[i]["multi_modal_data"]for i in range(batch_size)], dtype=object)
        
        # if self.config.enable_inference_engine_log_prob:
        #     # TODO: rollout_log_probs 是带 padding 的，需要除掉再 concat
        #     if prompts.batch.get("rollout_log_probs", None) is not None:
        #         # if the rollout_log_probs is already in the batch, we just update it
        #         batch['rollout_log_probs'] = torch.cat([prompts.batch['rollout_log_probs'], rollout_log_probs], dim=-1)
        #     else:
        #         # otherwise, we add it to the batch
        #         batch['rollout_log_probs'] = rollout_log_probs
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)
    
    def get_curr_request_id(self) -> int:
        """Get the current request ID."""
        return self.inference_engine.request_counter.counter
    
    def get_next_request_id(self) -> int:
        """Get the next request ID."""
        return next(self.inference_engine.request_counter)
    
    def add_requests(self, prompts: DataProto, **kwargs):
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

    @torch.no_grad()
    def step_all(self) -> list[Union[RequestOutput, PoolingRequestOutput]]:
        outputs: list[Union[RequestOutput, PoolingRequestOutput]] = []
        while self.inference_engine.llm_engine.has_unfinished_requests():
            step_outputs = self.inference_engine.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
        return sorted(outputs, key=lambda x: int(x.request_id))
    
    # TODO: align attention mask
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
        curr_response_unpadded_len = prompts.non_tensor_batch.get("response_unpadded_len", None)
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
            
            # tasks = [
            #     self.generate_sequence_task(
            #         prompt_idx,
            #         vllm_input,
            #         self.sampling_params,
            #         str(sample_id),
            #     ) for prompt_idx, (vllm_input, sample_id) in enumerate(zip(vllm_inputs, sample_ids))
            # ]
        
            completed_rollout = []
            for completed_task in asyncio.as_completed(tasks):
                prompt_idx, output = await completed_task
                completed_rollout.append(self.post_process_outputs(prompts[prompt_idx:prompt_idx+1], output))
        
        return DataProto.concat(completed_rollout)

    async def interrupt_all_requests_async(self) -> int:
        scheduler = self.inference_engine.llm_engine.scheduler
        request_ids_to_abort = []
        for request in scheduler.running:
            request_ids_to_abort.append(request.request_id)
        for request in scheduler.waiting:
            request_ids_to_abort.append(request.request_id)
        interrupted_request_num = len(request_ids_to_abort)
        
        if interrupted_request_num > 0:
            await self.inference_engine.abort_requests(request_ids_to_abort)
            psrl_logger.info(f"Interrupted {interrupted_request_num} requests.")
        else:
            psrl_logger.info("No requests to interrupt.")
            
        return interrupted_request_num

    async def interrupt_requests_async(self, request_ids):
        scheduler = self.inference_engine.llm_engine.scheduler
        if len(request_ids) > 0:
            request_ids = list(str(request_id) for request_id in request_ids)
            await self.inference_engine.abort_requests(request_ids)