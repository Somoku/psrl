import asyncio
import json
import logging
import os
import time
import uuid
from collections.abc import Sequence
from pprint import pprint
from typing import Any, cast

import numpy as np
import torch
import vllm
import vllm.entrypoints.cli.serve
from omegaconf import DictConfig, ListConfig
from ray.util.queue import Queue as RayQueue
from tensordict import TensorDict
from verl import DataProto
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import PromptType, TokensPrompt
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.engine.async_llm import AsyncLLM

try:
    # https://github.com/vllm-project/vllm/commit/96b9aa5aa076e64c68765232aec343e4d0006e2a
    from vllm.config import CompilationMode

    _use_compilation_mode = True
except ImportError:
    from vllm.config import CompilationLevel

    _use_compilation_mode = False

from psrl.utils.dataset.utils import _pre_process_inputs
from psrl.utils.logger import deprecated
from psrl.utils.profiling.event_converter import events_to_profiling_records
from psrl.workers.config import HFModelConfig, RolloutConfig
from psrl.workers.gen import StatCollector

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class PSRL_vLLMRollout:
    def __init__(
        self,
        psrl_config: DictConfig,
        config: RolloutConfig,
        model_config: HFModelConfig,
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
        self.psrl_config = psrl_config
        self.config = config
        self.stat_collector = None
        self.is_validate = kwargs.get("is_validate", False)

        # Cached vLLM server initialization artifacts for HTTP serving.
        self._server_args = None

        tensor_parallel_size = config.get("tensor_model_parallel_size", 1)
        pipeline_parallel_size = config.get("pipeline_model_parallel_size", 1)
        model_parallel_size = tensor_parallel_size * pipeline_parallel_size
        expert_parallel_size = config.get("expert_parallel_size", 1)
        enable_expert_parallel = expert_parallel_size > 1

        enable_return_routed_experts = config.get("enable_rollout_routing_replay", False)

        # For model parallel, we only run the inference engine on the first rank.
        # The inner parallel workers are handled by vLLM + Ray.
        if model_parallel_size > 1:
            import os
            if os.environ.get("LOCAL_RANK") != "0":
                self.inference_engine = None
                return

        model_path = model_config.local_path
        tokenizer = model_config.tokenizer
        model_hf_config = model_config.hf_config
        trust_remote_code = model_config.trust_remote_code
        lora_kwargs = (
            {
                "enable_lora": True,
                "max_loras": 1,
                "max_lora_rank": model_config.lora_rank,
            }
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

        # Load dummy format for meta init mode to save init time
        load_format = (
            "dummy"
            if (config.load_format.startswith("dummy") or kwargs.get("init_mode", "full") == "empty")
            else config.load_format
        )

        # LoRA configuration
        lora_kwargs = (
            {
                "enable_lora": True,
                "max_loras": 1,
                "max_lora_rank": model_config.lora_rank,
            }
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
                compilation_args = {"cudagraph_capture_sizes": cudagraph_capture_sizes}
                if _use_compilation_mode:
                    compilation_args["mode"] = CompilationMode.VLLM_COMPILE
                else:
                    compilation_args["level"] = CompilationLevel.PIECEWISE
                compilation_config["compilation_config"] = CompilationConfig(**compilation_args)
            else:
                psrl_logger.warning(f"cudagraph_capture_sizes must be a list, but got {cudagraph_capture_sizes}")

        if model_parallel_size > 1:
            # Configure vLLM for tensor/pipeline parallelism within Ray
            # Reset CUDA_VISIBLE_DEVICES to allow vLLM to manage GPU assignment
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            distributed_executor_backend = "ray"
        else:
            distributed_executor_backend = None  # auto detect

        llm_kwargs = dict(
            enable_sleep_mode=self.is_validate,  # Only validation uses sleep mode for val/train switching
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            enable_expert_parallel=enable_expert_parallel,
            distributed_executor_backend=distributed_executor_backend,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            # max_seq_len_to_capture=max_model_len, # deprecated
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
            enable_return_routed_experts=enable_return_routed_experts,
            **compilation_config,
            **lora_kwargs,
            **engine_kwargs,
        )

        if self.config.prometheus.enable:
            assert self.psrl_config.server_rollout.enable, (
                "Prometheus monitoring requires server rollout to be enabled."
            )

            if self.config.prometheus.served_model_name:
                # Extract model name from path if it's a full path
                served_model_name = self.config.prometheus.served_model_name
                if "/" in served_model_name:
                    # If it's a full path, extract the last part as model name
                    served_model_name = served_model_name.split("/")[-1]
                llm_kwargs["served_model_name"] = served_model_name

        llm_kwargs["scheduler_cls"] = "psrl.workers.gen.rollout_scheduler.RolloutScheduler"
        max_num_waiting_reqs_after_preemption = psrl_config.routing_strategy.max_num_waiting_reqs_after_preemption
        llm_kwargs["additional_config"] = {
            "max_num_waiting_reqs_after_preemption": max_num_waiting_reqs_after_preemption,
            "max_model_len_used_in_estimation": max_model_len
            * psrl_config.routing_strategy.max_estimated_concurrent_seqs_per_instance,
        }

        # Initialize abort queue, events, and request ids
        self.scheduler_abort_queue = RayQueue()
        self.scheduler_abort_events = {}
        self.scheduler_abort_requests = set()
        self._scheduler_abort_processor_task = None

        if self.psrl_config.server_rollout.enable:
            # Build and cache OpenAI server args at engine init time.
            # GenWorker will reuse these to start an in-process OpenAI-compatible server
            # without re-parsing or drifting defaults.
            server_args = self._build_server_args(model_path=model_path, args=llm_kwargs)
            self._server_args = server_args

            engine_args = AsyncEngineArgs.from_cli_args(server_args)
            usage_context = UsageContext.OPENAI_API_SERVER
            vllm_config = engine_args.create_engine_config(usage_context=usage_context)
        else:
            llm_kwargs["model"] = model_path
            engine_args = AsyncEngineArgs(**llm_kwargs)
            usage_context = UsageContext.ENGINE_CONTEXT
            vllm_config = engine_args.create_engine_config()

        stat_loggers = None
        if not config.disable_log_stats and psrl_config.status_collection.enable:
            psrl_logger.info(f"Enable status collection for rollout instance {kwargs.get('instance_id', 0)}")
            # Use custom stat loggers to collect engine stats
            status_queue = kwargs["status_queue"]
            self.stat_collector = StatCollector(
                vllm_config,
                psrl_config,
                instance_id=kwargs.get("instance_id", 0),
            )
            self.stat_collector.begin_record()
            self.stat_collector.init_output_queue(status_queue)
            self.stat_collector.init_scheduler_abort_queue(self.scheduler_abort_queue)
            self.stat_collector.record_model_version_update(0)
            stat_loggers = [self.stat_collector]
        psrl_logger.info(f"Initialize AsyncLLM for rollout instance {kwargs.get('instance_id', 0)}")
        self.inference_engine = AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )

        self.pad_token_id = tokenizer.pad_token_id

        # Start abort processor task
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        self._scheduler_abort_processor_task = loop.create_task(self._scheduler_abort_processor_loop())
        self._scheduler_abort_processor_task.add_done_callback(
            lambda f: f.result()
        )  # To avoid silent error in async tasks

    def _build_server_args(self, model_path: str, args: dict[str, Any]):
        """Build a CLI-like args Namespace compatible with vLLM OpenAI server.

        Args:
            model_path: Path to the model to load
            args: Dictionary of vLLM engine arguments
        """
        server_args = ["serve", model_path]
        for k, v in args.items():
            if isinstance(v, bool):
                if v:
                    server_args.append(f"--{k}")
            elif v is not None:
                server_args.append(f"--{k}")
                # Use json.dumps for dict to ensure valid JSON format
                server_args.append(json.dumps(v) if isinstance(v, dict) else str(v))

        pprint(server_args)

        parser = FlexibleArgumentParser(description="vLLM CLI")
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        cmds = {}
        for cmd in vllm.entrypoints.cli.serve.cmd_init():
            cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
            cmds[cmd.name] = cmd

        args = parser.parse_args(args=server_args)
        args.model = getattr(args, "model_tag", model_path)
        if getattr(args, "subparser", None) in cmds:
            cmds[args.subparser].validate(args)
        return args

    @property
    def server_args(self):
        """Get the cached vLLM OpenAI server args."""
        return self._server_args

    async def _scheduler_abort_processor_loop(self):
        """Background loop that processes abort requests from the queue."""
        while True:
            # Wait for abort request from queue using async method (blocking wait)
            request_ids = await self.scheduler_abort_queue.get_async(block=True)
            if request_ids is None:  # Sentinel value to stop the loop
                break

            # Create a new event for this abort request
            for request_id in request_ids:
                self.scheduler_abort_requests.add(request_id)
            request_ids_tuple = tuple(request_ids)
            self.scheduler_abort_events[request_ids_tuple] = asyncio.Event()

            # Process the abort request
            await self.inference_engine.abort(request_ids)

            # Signal that this abort request is done and clean up
            self.scheduler_abort_events[request_ids_tuple].set()
            del self.scheduler_abort_events[request_ids_tuple]

    async def _wait_for_all_scheduler_abort_requests_processed(self):
        """Wait until the abort queue is empty and all pending abort requests are processed."""
        if self.scheduler_abort_queue is None:
            return
        # Wait until queue is empty
        while not self.scheduler_abort_queue.empty():
            await asyncio.sleep(0)

        # Wait for all pending abort requests to complete
        if self.scheduler_abort_events:
            events = list(self.scheduler_abort_events.values())
            await asyncio.gather(*[event.wait() for event in events], return_exceptions=True)

    def pre_process_inputs(self, prompts: DataProto) -> PromptType | Sequence[PromptType]:
        """
        Pre-process prompts to convert them into vLLM-compatible inputs.

        This method performs several transformations:
        1. Remove left padding from prompt token IDs
        2. Concatenate raw prompt IDs and response IDs for continuation
        3. Add multi-modal data if present (images, etc.)
        4. Configure sampling parameters based on mode (sampling/validation/greedy)

        Args:
            prompts: DataProto containing input prompts and metadata

        Returns:
            vllm_inputs ready for vLLM generation
        """

        batch_size = len(prompts)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            idx = prompts.batch["input_ids"]  # (bs, prompt_length)
            # Remove the left padding in the prompt token_id
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)],
                dtype=object,
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError(
                f"vllm sharding manager is not work properly with "
                f"{batch_size=} v.s. {len(non_tensor_batch['raw_prompt_ids'])=}."
            )

        if isinstance(non_tensor_batch["raw_prompt_ids"], np.ndarray):
            raw_prompt_ids = non_tensor_batch["raw_prompt_ids"].tolist()
        else:
            raw_prompt_ids = non_tensor_batch["raw_prompt_ids"]

        if "raw_response_ids" in non_tensor_batch:
            raw_response_ids = non_tensor_batch["raw_response_ids"]
            raw_response_ids = np.fromiter(raw_response_ids.tolist(), dtype=object)
        else:
            raw_response_ids = np.fromiter(([] for _ in range(batch_size)), dtype=object)

        if isinstance(raw_response_ids, np.ndarray):
            raw_response_ids = raw_response_ids.tolist()

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids_, raw_response_ids_, multi_modal_data in zip(
                raw_prompt_ids, raw_response_ids, non_tensor_batch["multi_modal_data"]
            ):
                vllm_inputs.append(
                    {
                        "prompt_token_ids": raw_prompt_ids_ + raw_response_ids_,
                        "multi_modal_data": multi_modal_data,
                    }
                )
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids_ + raw_response_ids_}
                for raw_prompt_ids_, raw_response_ids_ in zip(raw_prompt_ids, raw_response_ids)
            ]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

        return vllm_inputs

    def post_process_outputs(
        self,
        prompts: DataProto,
        outputs: (RequestOutput | PoolingRequestOutput | list[RequestOutput | PoolingRequestOutput]),
        accumulated_events: list | None = None,
        generation_start_wall_ts: float = 0.0,
    ) -> DataProto:
        """
        Post-process vLLM outputs to convert them back into DataProto format.

        This method performs several transformations:
        1. Extract response token IDs, lengths, and interruption status
        2. Collect log probabilities if required
        3. Concatenate new response IDs to existing raw response IDs
        4. Build final DataProto with updated tensors and metadata

        Args:
            prompts (DataProto): Original input DataProto containing prompts.
            outputs (RequestOutput | PoolingRequestOutput | list): vLLM generation outputs.
            accumulated_events (list | None): Profiling events accumulated during streaming generation.
            generation_start_wall_ts (float): Wall-clock timestamp when generation started.
            generation_end_wall_ts (float): Wall-clock timestamp when generation ended.

        Returns:
            DataProto: Updated DataProto with generated responses and proper formatting.
        """

        if not isinstance(outputs, list):
            outputs = [outputs]
        assert len(outputs) == len(prompts), "Mismatched batch size between prompts and VLLM outputs."

        batch_size = len(prompts)
        non_tensor_batch = prompts.non_tensor_batch
        uid_list = non_tensor_batch["uid"].tolist()

        response_ids_list = []
        response_len_list = []
        interrupted_list = []
        interrupted_by_scheduler_list = []
        all_log_prob_list = []
        routed_experts_list = []

        for i, uid in enumerate(uid_list):
            vllm_output = outputs[i]
            assert len(vllm_output.outputs) == 1, "RolloutRouter only supports single request generation."

            response_ids = vllm_output.outputs[0].token_ids
            response_len = len(response_ids)
            interrupted = vllm_output.outputs[0].finish_reason == "abort"

            response_ids_list.append(response_ids)
            response_len_list.append(response_len)
            interrupted_list.append(interrupted)
            if str(uid) in self.scheduler_abort_requests:
                assert interrupted, "Requests interrupted by the scheduler should also have finish_reason 'abort'."
                interrupted_by_scheduler_list.append(True)
                self.scheduler_abort_requests.remove(str(uid))
            else:
                psrl_logger.debug(
                    f"Request {uid} is not interrupted by the scheduler (not in {self.scheduler_abort_requests}). "
                    f"It is interrupted by the synchronization (i.e., partial rollout)."
                )
                interrupted_by_scheduler_list.append(False)

            log_prob_list = []
            # if inference logprobs is required, we need to collect the log probabilities
            if (
                self.psrl_config.log_prob.enable_rollout_engine_log_prob
                and hasattr(vllm_output.outputs[0], "logprobs")
                and vllm_output.outputs[0].logprobs is not None
            ):
                if self.psrl_config.partial_rollout.interrupt_as_prompt:
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

            routed_experts = None
            if self.config.enable_rollout_routing_replay:
                routed_experts = vllm_output.outputs[0].routed_experts
            routed_experts_list.append(routed_experts)

        # Consolidate batch results
        if "raw_response_ids" in non_tensor_batch:
            raw_response_ids = non_tensor_batch.pop("raw_response_ids")
            raw_response_ids = np.fromiter(raw_response_ids.tolist(), dtype=object)
        else:
            raw_response_ids = np.fromiter(([] for _ in range(batch_size)), dtype=object)

        raw_response_ids = raw_response_ids + np.fromiter(response_ids_list, dtype=object)
        non_tensor_batch["raw_response_ids"] = raw_response_ids

        if "response_unpadded_len" in non_tensor_batch:
            curr_response_unpadded_len = non_tensor_batch["response_unpadded_len"]
        else:
            curr_response_unpadded_len = [0] * batch_size
        response_unpadded_len = [curr_response_unpadded_len[i] + response_len_list[i] for i in range(batch_size)]
        non_tensor_batch["response_unpadded_len"] = np.array(response_unpadded_len, dtype=int)
        non_tensor_batch["interrupted"] = np.array(interrupted_list, dtype=bool)
        non_tensor_batch["interrupted_by_scheduler"] = np.array(interrupted_by_scheduler_list, dtype=bool)

        # Update rollout_log_probs
        if self.psrl_config.log_prob.enable_rollout_engine_log_prob:
            if "rollout_log_probs" in non_tensor_batch:
                curr_rollout_log_probs = non_tensor_batch.pop("rollout_log_probs")
                curr_rollout_log_probs = np.fromiter(curr_rollout_log_probs.tolist(), dtype=object)
            else:
                curr_rollout_log_probs = np.fromiter(([] for _ in range(batch_size)), dtype=object)
            curr_rollout_log_probs += np.fromiter(all_log_prob_list, dtype=object)
            non_tensor_batch["rollout_log_probs"] = curr_rollout_log_probs

        # process routed experts
        if self.config.enable_rollout_routing_replay:
            non_tensor_batch["routed_experts"] = np.fromiter(routed_experts_list, dtype=object)

        # --- Per-trajectory profiling data ---
        enable_profiling = self.psrl_config.profile.trajectory.enable
        if enable_profiling and accumulated_events:
            for i, uid in enumerate(uid_list):
                vllm_output = outputs[i] if isinstance(outputs, list) else outputs
                num_cached = getattr(vllm_output, "num_cached_tokens", 0) or 0
                total_seq_len = len(getattr(vllm_output, "prompt_token_ids", []) or []) + response_len_list[i]
                prefill_records, decode_records = events_to_profiling_records(
                    events=accumulated_events,
                    num_cached_tokens=num_cached,
                    total_seq_len=total_seq_len,
                    num_output_tokens=response_len_list[i],
                )
                non_tensor_batch["profiling_prefill_records"] = np.array(
                    [prefill_records], dtype=object
                )
                non_tensor_batch["profiling_decode_records"] = np.array(
                    [decode_records], dtype=object
                )
                non_tensor_batch["profiling_generation_start_wall_ts"] = np.array(
                    [generation_start_wall_ts], dtype=float
                )
                # NOTE(claude): generation_end_wall_ts is wall-clock only, used by
                # TurnProfilingCollector to estimate env turn duration between turns
                non_tensor_batch["profiling_generation_end_wall_ts"] = np.array(
                    [time.time()], dtype=float
                )

        batch = TensorDict(
            {
                "input_ids": prompts.batch["input_ids"],  # [bs, prompt_length]
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)

    @deprecated("vllm_rollout.add_requests is not used.")
    def add_requests(self, prompts: DataProto, sampling_params: dict[str, Any]):
        """
        Add generation requests to the vLLM inference engine.
        This method converts prompts to vLLM format and queues them for generation.

        Args:
            prompts: DataProto containing input prompts
            sampling_params: Sampling parameters for generation
        """
        vllm_inputs = self.pre_process_inputs(prompts)
        parsed_vllm_inputs = cast(PromptType | Sequence[PromptType], vllm_inputs)
        if isinstance(parsed_vllm_inputs, (str, dict)):
            # Convert a single prompt to a list.
            parsed_vllm_inputs = [parsed_vllm_inputs]

        sampling_params = SamplingParams(**sampling_params)
        for prompt in parsed_vllm_inputs:
            request_id = str(self.get_next_request_id())
            self.inference_engine.llm_engine.add_request(
                request_id,
                prompt,
                sampling_params,
                priority=0,
            )

    @torch.no_grad()
    async def generate_sequences_async(self, prompts: DataProto, sampling_params: dict[str, Any]) -> DataProto:
        """
        Generate sequences from prompts using asynchronous vLLM generation.

        This method enables concurrent generation of multiple sequences with:
        1. Async task creation for each prompt
        2. Independent sampling parameter customization per prompt
        3. Support for partial rollout (continuation from previous responses)
        4. Efficient concurrent processing with asyncio

        Args:
            prompts: DataProto containing input prompts with required 'uid' field
            sampling_params: Sampling parameters for generation

        Returns:
            DataProto with concatenated results from all async generations
        """
        vllm_inputs = self.pre_process_inputs(prompts)
        sample_ids = prompts.non_tensor_batch.get("uid", None)
        curr_response_unpadded_len = prompts.non_tensor_batch.get("response_unpadded_len", [0] * len(vllm_inputs))
        assert sample_ids is not None, "sample_ids must be provided in the prompts.non_tensor_batch"

        # users can customize different sampling_params at different run
        sampling_params = SamplingParams(**sampling_params)

        tasks = []
        for prompt_idx, (vllm_input, sample_id, curr_response_len) in enumerate(
            zip(vllm_inputs, sample_ids, curr_response_unpadded_len)
        ):
            tasks.append(
                self.generate_sequence_task(
                    prompt_idx,
                    vllm_input,
                    sampling_params=sampling_params,
                    uid=str(sample_id),
                    max_tokens=self.config.response_length - curr_response_len,
                )
            )

        completed_rollout = []
        for completed_task in asyncio.as_completed(tasks):
            prompt_idx, output, accumulated_events, gen_start_ts = await completed_task
            completed_rollout.append(
                self.post_process_outputs(
                    prompts[prompt_idx : prompt_idx + 1],
                    output,
                    accumulated_events=accumulated_events,
                    generation_start_wall_ts=gen_start_ts,
                )
            )

        return DataProto.concat(completed_rollout)

    @torch.no_grad()
    async def raw_generate_sequences_async(self, prompts: DataProto, sampling_params: dict[str, Any]):
        """Generate sequences from the prompts using vLLM asynchronously without post-processing."""
        vllm_inputs = self.pre_process_inputs(prompts)
        sample_ids = prompts.non_tensor_batch.get("uid", None)
        curr_response_unpadded_len = prompts.non_tensor_batch.get("response_unpadded_len", [0] * len(vllm_inputs))
        assert sample_ids is not None, "sample_ids must be provided in the prompts.non_tensor_batch"

        # users can customize different sampling_params at different run
        sampling_params = SamplingParams(**sampling_params)
        tasks = []
        for prompt_idx, (vllm_input, sample_id, curr_response_len) in enumerate(
            zip(vllm_inputs, sample_ids, curr_response_unpadded_len)
        ):
            tasks.append(
                self.generate_sequence_task(
                    prompt_idx,
                    vllm_input,
                    sampling_params=sampling_params,
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
        prompt_tokens: dict[str, Any] | list[int],
        sampling_params: SamplingParams,
        uid: str | None = None,
        max_tokens: int | None = None,
    ) -> tuple[int, RequestOutput, list, float]:
        """
        Generate a single sequence asynchronously using vLLM.

        This method creates an async generation task for a single prompt and
        waits for completion, returning the final output along with profiling data.

        Args:
            idx (int): Index of the prompt in the batch.
            prompt_tokens (dict[str, Any] | list[int]): Either token IDs list or dict with prompt data.
            sampling_params (SamplingParams): Sampling parameters for generation.
            uid (str | None): Unique identifier for the request.
            max_tokens (int | None): Maximum number of tokens to generate.

        Returns:
            tuple[int, RequestOutput, list, float]: (prompt_idx, final_request_output,
                accumulated_events, generation_start_wall_ts)
        """
        # Ensure all abort requests in the queue are processed before starting generation
        # NOTE(lhy): currently, only the preempted requests are put into the abort queue.
        # Other requests are aborted (e.g., partial rollout) directly by the scheduler.
        if self.scheduler_abort_queue is not None:
            await self._wait_for_all_scheduler_abort_requests_processed()

        if isinstance(prompt_tokens, list):
            prompt_tokens = {"prompt_token_ids": prompt_tokens}
        if max_tokens is not None:
            sampling_params.max_tokens = int(max_tokens)

        request_id = str(uuid.uuid4()) if uid is None else uid

        accumulated_events: list = []
        generation_start_wall_ts = time.time()

        task = self.inference_engine.generate(
            prompt=TokensPrompt(**prompt_tokens),
            sampling_params=sampling_params,
            request_id=request_id,
        )
        async for output in task:
            # Accumulate profiling events from each streaming output.
            if hasattr(output, "events") and output.events:
                accumulated_events.extend(output.events)
                output.events = None  # Avoid double-counting
            last_output = output

        return idx, last_output, accumulated_events, generation_start_wall_ts

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
