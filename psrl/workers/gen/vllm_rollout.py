import logging
import os
import uuid
import asyncio

import numpy as np
from contextlib import contextmanager
from copy import deepcopy
from collections.abc import Sequence
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from typing import Any, Dict, Optional, Union, List, Tuple, cast

import torch
import torch.distributed
from torch.distributed.device_mesh import DeviceMesh

from vllm import LLM, SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.distributed import parallel_state as vllm_ps
from vllm.inputs import PromptType, TokensPrompt
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind

from verl import DataProto
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length

from psrl.workers.config import HFModelConfig, RolloutConfig
from psrl.utils.logger import deprecated
from psrl.workers.gen import StatCollector
from psrl.utils.dataset.utils import _pre_process_inputs

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)

class PSRL_vLLMRollout:
    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
        **kwargs,
    ):
        """
        Initialize PSRL vLLM rollout with specified configuration.

        Args:
            model_path: Path to the model to load
            config: Configuration object containing vLLM and PSRL settings
            tokenizer: Tokenizer instance for the model
            **kwargs: Additional keyword arguments (trust_remote_code, lora_kwargs, etc.)
        """

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
        
        model_path = model_config.local_path
        tokenizer = model_config.tokenizer
        model_hf_config = model_config.hf_config
        trust_remote_code = model_config.trust_remote_code
        lora_kwargs = (
            {"enable_lora": True, "max_loras": 1, "max_lora_rank": model_config.lora_rank}
            if model_config.lora_rank > 0
            else {}
        )

        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)
        
        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(
                model_hf_config.llm_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(
                model_hf_config.text_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")
            assert max_position_embeddings >= config.prompt_length + config.response_length, (
                "model context length should be greater than total sequence length"
            )
        else:
            # handle type where there's a length extend factor
            # see https://qwen.readthedocs.io/en/latest/deployment/vllm.html#extended-context-support
            # for using yarn as an example
            rope_scaling_factor = rope_scaling_config.get("factor", 1.0)

            assert (
                model_hf_config.max_position_embeddings * rope_scaling_factor
                >= config.prompt_length + config.response_length
            ), (
                "model context length should be greater than total sequence length, "
                + f"got rope_scaling_factor={rope_scaling_factor} and "
                + f"max_position_embeddings={model_hf_config.max_position_embeddings}"
            )
        
        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)
        
        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        # LoRA configuration
        lora_kwargs = (
            {"enable_lora": True, "max_loras": 1, "max_lora_rank": model_config.lora_rank}
            if model_config.lora_rank > 0
            else {}
        )
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = config.get("engine_kwargs", {}).get("vllm", {}) or {}
        
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        compilation_config = {}

        cudagraph_capture_sizes = config.get("cudagraph_capture_sizes")
        # enforce_eager must be False to use cudagraph
        if not config.enforce_eager and cudagraph_capture_sizes:
            if isinstance(cudagraph_capture_sizes, ListConfig):
                compilation_config["compilation_config"] = CompilationConfig(
                    level=CompilationLevel.PIECEWISE, cudagraph_capture_sizes=cudagraph_capture_sizes
                )
            else:
                logger.warning(f"cudagraph_capture_sizes must be a list, but got {cudagraph_capture_sizes}")


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
            enable_sleep_mode=False,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            distributed_executor_backend=distributed_executor_backend,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            max_num_seqs=config.max_num_seqs,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=config.enable_prefix_caching,
            trust_remote_code=trust_remote_code,
            logprobs_mode=config.logprobs_mode,
            worker_extension_cls="psrl.workers.gen.vllm_extension.vLLMWorkerExtension",
            seed=kwargs.get("seed", 0),
            **lora_kwargs,
            **engine_kwargs,
        )

        if config.mode == "psrl_async":
            engine_args = AsyncEngineArgs(**llm_kwargs)
            stat_loggers = None
            if not config.disable_log_stats and self.config.status_collection.enable:
                # Use custom stat loggers to collect engine stats
                vllm_config = engine_args.create_engine_config()
                status_queue = kwargs["status_queue"]
                stat_logger = StatCollector(vllm_config, engine_index=kwargs.get("instance_id", 0))
                stat_logger.init_output_queue(status_queue)
                stat_loggers = [stat_logger]
            self.inference_engine = AsyncLLM.from_engine_args(engine_args, stat_loggers=stat_loggers)
        else:
            self.inference_engine = LLM(**llm_kwargs)

        # NOTE(lhy): sleep mode is not supported when using NIXL
        # Because it will cause illegal memory registration
        '''
        # Offload vllm model to reduce peak memory usage
        if load_format == "dummy" and config.free_cache_engine:
            self.inference_engine.sleep(level=1)
        '''

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
            repetition_penalty=config.get("repetition_penalty", 1.0),
            output_kind=RequestOutputKind.CUMULATIVE,
        )

        # we may detokenize the result all together later
        kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)) and k != "seed" and k != "n":
                kwargs[k] = config.get(k)

        psrl_logger.info(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        """
        Context manager to temporarily update sampling parameters.
        
        This allows temporary modifications to sampling parameters for specific
        generation requests while preserving the original parameters.
        
        Args:
            **kwargs: Sampling parameters to temporarily override
            
        Yields:
            None: Context for temporary parameter usage
        """
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                old_sampling_params_args[key] = getattr(self.sampling_params, key)
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
        """
        Pre-process prompts to convert them into vLLM-compatible inputs.
        
        This method performs several transformations:
        1. Remove left padding from prompt token IDs
        2. Concatenate raw prompt IDs and response IDs for continuation
        3. Add multi-modal data if present (images, etc.)
        4. Configure sampling parameters based on mode (sampling/validation/greedy)
        
        Args:
            prompts: DataProto containing input prompts and metadata
            kwargs: Additional keyword arguments for processing
            
        Returns:
            Tuple of (vllm_inputs, sampling_kwargs) ready for vLLM generation
        """
        
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
            # TODO(verl): try **
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
    
    def post_process_outputs_lite(
        self,
        prompts: DataProto,
        outputs: Union[Union[RequestOutput, PoolingRequestOutput], list[Union[RequestOutput, PoolingRequestOutput]]],
    ) -> DataProto:
        """
        Post-process vLLM outputs to convert them back into DataProto format.
        """
        if not isinstance(outputs, list):
            outputs = [outputs]
        assert len(outputs) == len(prompts), "Mismatched batch size between prompts and VLLM outputs."

        batch_size = len(prompts)
        non_tensor_batch = prompts.non_tensor_batch

        response_ids_list = []
        response_len_list = []
        interrupted_list = []
        all_log_prob_list = []

        for i in range(batch_size):
            vllm_output = outputs[i]
            assert len(vllm_output.outputs) == 1, "RolloutRouter only supports single request generation."

            response_ids = vllm_output.outputs[0].token_ids
            response_len = len(response_ids)
            interrupted = vllm_output.outputs[0].finish_reason == "abort"

            response_ids_list.append(response_ids)
            response_len_list.append(response_len)
            interrupted_list.append(interrupted)

            log_prob_list = []
            # if inference logprobs is required, we need to collect the log probabilities
            if (
                self.config.enable_rollout_engine_log_prob and
                hasattr(vllm_output.outputs[0], 'logprobs') and
                vllm_output.outputs[0].logprobs is not None
            ):
                if self.config.interrupt_as_prompt:
                    curr_response_len = non_tensor_batch.get("response_unpadded_len", 0)
                    # Collect log probs only when the request finished normally
                    # The response log probs are collected in two parts:
                    # 1. The log probs of the accumulated response tokens (in current prompt tokens)
                    # 2. The log probs of the current response tokens
                    if not interrupted and curr_response_len > 0:
                        # partial response log probs from prompt log probs
                        prompt_token_ids = vllm_output.prompt_token_ids
                        for i, logprob in enumerate(vllm_output.prompt_logprobs[-curr_response_len:]):
                            log_prob_list.append(logprob[prompt_token_ids[i - curr_response_len]].logprob)
                        # new response log probs from decode log probs
                        for i, logprob in enumerate(vllm_output.outputs[0].logprobs):
                            log_prob_list.append(logprob[response_ids[i]].logprob)
                else:
                    # Response log probs from decode log probs
                    for i, logprob in enumerate(vllm_output.outputs[0].logprobs):
                        log_prob_list.append(logprob[response_ids[i]].logprob)
            all_log_prob_list.append(log_prob_list)
        
        # Consolidate batch results
        if "raw_response_ids" in non_tensor_batch:
            raw_response_ids = non_tensor_batch["raw_response_ids"]
        else:
            raw_response_ids = np.fromiter(([] for _ in range(batch_size)), dtype=object)

        raw_response_ids += np.fromiter(response_ids_list, dtype=object)
        non_tensor_batch["raw_response_ids"] = raw_response_ids

        if "response_unpadded_len" in non_tensor_batch:
            curr_response_unpadded_len = non_tensor_batch["response_unpadded_len"]
        else:
            curr_response_unpadded_len = [0] * batch_size
        response_unpadded_len = [curr_response_unpadded_len[i] + response_len_list[i] for i in range(batch_size)]
        non_tensor_batch["response_unpadded_len"] = np.array(response_unpadded_len, dtype=int)
        non_tensor_batch["interrupted"] = np.array(interrupted_list, dtype=bool)

        # Update rollout_log_probs
        if self.config.enable_rollout_engine_log_prob:
            if "rollout_log_probs" in non_tensor_batch:
                curr_rollout_log_probs = non_tensor_batch["rollout_log_probs"]
            else:
                curr_rollout_log_probs = np.fromiter(([] for _ in range(batch_size)), dtype=object)
            curr_rollout_log_probs += np.fromiter(all_log_prob_list, dtype=object)
            non_tensor_batch["rollout_log_probs"] = curr_rollout_log_probs

        batch = TensorDict(
            {
                "input_ids": prompts.batch["input_ids"], # [bs, prompt_length]
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)
            
    def post_process_outputs(
        self,
        prompts: DataProto,
        outputs: Union[Union[RequestOutput, PoolingRequestOutput], list[Union[RequestOutput, PoolingRequestOutput]]],
    ) -> DataProto:
        """
        Post-process vLLM outputs to convert them back into DataProto format.
        
        This method performs several transformations:
        1. Extract response token IDs, lengths, and interruption status
        2. Collect log probabilities if required
        3. Concatenate new response IDs to existing raw response IDs
        4. Apply right-side padding to responses
        5. Construct proper position IDs and attention masks
        6. Build final DataProto with updated tensors and metadata
        
        Args:
            prompts: Original input DataProto containing prompts
            outputs: vLLM generation outputs (single output or list of outputs)
            
        Returns:
            Updated DataProto with generated responses and proper formatting
        """

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
                    self.config.enable_rollout_engine_log_prob and
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
        if "raw_response_ids" in non_tensor_batch:
            raw_response_ids = non_tensor_batch["raw_response_ids"]
        else:
            raw_response_ids = np.fromiter(([] for _ in range(batch_size)), dtype=object)
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

        # TODO(linsh): optimize the DataProto construction to packing
        # Here we pad the response to the right side for both interrupted and completed requests.
        response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
        if self.config.enable_rollout_engine_log_prob:
            if "rollout_log_probs" in non_tensor_batch:
                curr_rollout_log_probs = non_tensor_batch["rollout_log_probs"]
            else:
                curr_rollout_log_probs = np.fromiter(([] for _ in range(batch_size)), dtype=object)
            curr_rollout_log_probs += np.fromiter(rollout_log_probs, dtype=object)
            non_tensor_batch["rollout_log_probs"] = curr_rollout_log_probs

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
        """
        Add generation requests to the vLLM inference engine.
        
        This method converts prompts to vLLM format and queues them for generation.
        Used primarily for async/streaming generation modes.
        
        Args:
            prompts: DataProto containing input prompts
            **kwargs: Additional parameters for generation
        """
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

    @deprecated("vllm_rollout.step_all is not used.")
    @torch.no_grad()
    def step_all(self) -> list[Union[RequestOutput, PoolingRequestOutput]]:
        outputs: list[Union[RequestOutput, PoolingRequestOutput]] = []
        while self.inference_engine.llm_engine.has_unfinished_requests():
            step_outputs = self.inference_engine.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
        return sorted(outputs, key=lambda x: int(x.request_id))
    
    @deprecated("vllm_rollout.step is not used.")
    @torch.no_grad()
    def step(self) -> list[Union[RequestOutput, PoolingRequestOutput]]:
        return self.inference_engine.llm_engine.step()

    @GPUMemoryLogger(role="vllm rollout spmd", logger=psrl_logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """
        Generate sequences from prompts using synchronous vLLM generation.
        
        This is the main synchronous generation method that:
        1. Pre-processes inputs for vLLM
        2. Runs generation with specified sampling parameters
        3. Post-processes outputs back to DataProto format
        
        Args:
            prompts: DataProto containing input prompts and metadata
            **kwargs: Additional generation parameters
            
        Returns:
            DataProto with generated sequences and updated metadata
        """
        vllm_inputs, kwargs = self.pre_process_inputs(prompts, kwargs)
        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            # the inference_engine will handle the request_id internally
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            return self.post_process_outputs_lite(prompts, outputs)
    
    @GPUMemoryLogger(role="vllm stream rollout", logger=psrl_logger)
    @torch.no_grad()
    async def generate_sequences_async(self, prompts: DataProto, **kwargs) -> DataProto:
        """
        Generate sequences from prompts using asynchronous vLLM generation.
        
        This method enables concurrent generation of multiple sequences with:
        1. Async task creation for each prompt
        2. Independent sampling parameter customization per prompt
        3. Support for partial rollout (continuation from previous responses)
        4. Efficient concurrent processing with asyncio
        
        Args:
            prompts: DataProto containing input prompts with required 'uid' field
            **kwargs: Additional generation parameters
            
        Returns:
            DataProto with concatenated results from all async generations
        """
        vllm_inputs, kwargs = self.pre_process_inputs(prompts, kwargs)
        sample_ids = prompts.non_tensor_batch.get("uid", None)
        curr_response_unpadded_len = prompts.non_tensor_batch.get("response_unpadded_len", [0] * len(vllm_inputs))
        assert sample_ids is not None, \
            "sample_ids must be provided in the prompts.non_tensor_batch"
        
        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            tasks = []
            for prompt_idx, (vllm_input, sample_id, curr_response_len) in enumerate(zip(vllm_inputs, sample_ids, curr_response_unpadded_len)):
                tasks.append(
                    self.generate_sequence_task(
                        prompt_idx,
                        vllm_input,
                        sampling_params=self.sampling_params,
                        uid=str(sample_id),
                        max_tokens=self.config.response_length - curr_response_len,
                    )
                )
        
            completed_rollout = []
            for completed_task in asyncio.as_completed(tasks):
                prompt_idx, output = await completed_task
                completed_rollout.append(self.post_process_outputs_lite(prompts[prompt_idx:prompt_idx+1], output))
        
        return DataProto.concat(completed_rollout)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=psrl_logger)
    @torch.no_grad()
    def raw_generate_sequences(self, prompts: DataProto, **kwargs):
        """Generate sequences from the prompts using vLLM without post-processing."""
        assert "response_unpadded_len" not in prompts.non_tensor_batch, \
            "partial rollout is currently not supported in sync mode"

        vllm_inputs, kwargs = self.pre_process_inputs(prompts, kwargs)
        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            # the inference_engine will handle the request_id internally
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            return outputs
    
    @GPUMemoryLogger(role="vllm stream rollout", logger=psrl_logger)
    @torch.no_grad()
    async def raw_generate_sequences_async(self, prompts: DataProto, **kwargs):
        """Generate sequences from the prompts using vLLM asynchronously without post-processing."""
        vllm_inputs, kwargs = self.pre_process_inputs(prompts, kwargs)
        sample_ids = prompts.non_tensor_batch.get("uid", None)
        curr_response_unpadded_len = prompts.non_tensor_batch.get("response_unpadded_len", [0] * len(vllm_inputs))
        assert sample_ids is not None, \
            "sample_ids must be provided in the prompts.non_tensor_batch"
        
        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            tasks = []
            for prompt_idx, (vllm_input, sample_id, curr_response_len) in enumerate(zip(vllm_inputs, sample_ids, curr_response_unpadded_len)):
                tasks.append(
                    self.generate_sequence_task(
                        prompt_idx,
                        vllm_input,
                        sampling_params=self.sampling_params,
                        uid=str(sample_id),
                        max_tokens=self.config.response_length - curr_response_len,
                    )
                )
        
            vllm_outputs = []
            for completed_task in asyncio.as_completed(tasks):
                output = await completed_task
                vllm_outputs.append(output)
        
        return vllm_outputs
    
    async def generate_sequence_task(
        self,
        idx: int,
        prompt_tokens: Union[Dict[str, Any], List[int]],
        sampling_params: Optional[SamplingParams] = None,
        uid: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[int, RequestOutput]:
        """
        Generate a single sequence asynchronously using vLLM.
        
        This method creates an async generation task for a single prompt and
        waits for completion, returning the final output.
        
        Args:
            idx: Index of the prompt in the batch
            prompt_tokens: Either token IDs list or dict with prompt data
            sampling_params: Sampling parameters for generation (optional)
            uid: Unique identifier for the request (optional)
            
        Returns:
            Tuple of (prompt_index, final_request_output)
        """
        if sampling_params is None:
            sampling_params = self.sampling_params
        if isinstance(prompt_tokens, list):
            prompt_tokens = {'prompt_token_ids': prompt_tokens}
        if max_tokens is not None:
            setattr(sampling_params, "max_tokens", int(max_tokens))

        task = self.inference_engine.generate(
            prompt=TokensPrompt(**prompt_tokens),
            sampling_params=sampling_params,
            request_id=str(uuid.uuid4()) if uid is None else uid,
        )
        async for output in task:
            last_output = output
        return idx, last_output

    async def interrupt_all_requests_async(self) -> int:
        """
        Interrupt all requests (both running and waiting) asynchronously.
        
        This method aborts all currently queued and executing requests in the
        vLLM engine, useful for emergency stops or model updates.
        
        Returns:
            Number of requests that were interrupted
        """
        interrupted_request_num = await self.inference_engine.abort_all()
        if interrupted_request_num > 0:
            psrl_logger.debug(f"Interrupted {interrupted_request_num} requests via abort_all().")
        else:
            psrl_logger.debug("No requests to interrupt via abort_all().")
        return interrupted_request_num

    async def interrupt_requests_async(self, request_ids):
        """
        Interrupt specific requests by their IDs asynchronously.
        
        This method aborts only the specified requests, allowing selective
        interruption based on staleness or other criteria.
        
        Args:
            request_ids: List of request IDs to interrupt
        """
        if len(request_ids) > 0:
            request_ids = [str(request_id) for request_id in request_ids]
            await self.inference_engine.abort(request_ids)

    async def get_engine_status(self):
        """
        Get the current status of the vLLM engine.
        
        This method returns information about the engine state, including
        queue sizes and other operational metrics.
        
        Returns:
            Dictionary containing engine status information
        """
        if self.inference_engine is None:
            return {"status": "not initialized"}
        
        status = {
            "waiting_and_running_queue_size": await self.waiting_and_running_queue_size(),
        }
        return status
