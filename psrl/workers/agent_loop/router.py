import os
import logging
import numpy as np
from tensordict import TensorDict
from omegaconf import DictConfig

import ray

from verl import DataProto

from psrl.workers.ps.request_status_tracker import RequestStatus
from psrl.utils.logger import DualOutputHandler, get_worker_info, log_single_event, EventType, deprecated

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

class RolloutRouter:
    def __init__(
        self,
        config: DictConfig,
        ps_manager_handle,
        gen_worker_handles: list[ray.actor.ActorHandle],
    ):
        self.config = config
        self.ps_manager_handle = ps_manager_handle
        self.gen_worker_handles = gen_worker_handles
        self.gen_worker_num = len(gen_worker_handles)
        
        # Engine status tracking
        # NOTE: The engine status will be updated by RolloutCoordinator
        # and can be accessed via get_engine_status() method
        self.latest_engine_status = {}
        
        # Build logger
        self.log_prefix = f"RolloutRouter"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))

    def update_engine_status(self, engine_status: dict):
        self.latest_engine_status = engine_status

    def _choose_gen_worker(self, request: DataProto) -> ray.actor.ActorHandle:
        min_request_num = float('inf')
        chosen_worker = None
        for instance_id in range(self.gen_worker_num):
            request_num = self.latest_engine_status.get("instances", {}).get(instance_id, {}).get("waiting_and_running_queue_size", 0)
            if request_num < min_request_num:
                min_request_num = request_num
                chosen_worker = self.gen_worker_handles[instance_id]
        return chosen_worker

    def _consolidate_response(
        self,
        prompt,
        vllm_output,
    ) -> DataProto:
        assert len(vllm_output.outputs) == 1, "RolloutRouter only supports single request generation."
        
        non_tensor_batch = prompt.non_tensor_batch
        response_ids = vllm_output.outputs[0].token_ids
        response_len = len(response_ids)
        interrupted = vllm_output.outputs[0].finish_reason == "abort"    
        # if inference logprobs is required, we need to collect the log probabilities
        if (
            self.config.enable_inference_engine_log_prob and
            hasattr(vllm_output.outputs[0], 'logprobs') and
            vllm_output.outputs[0].logprobs is not None
        ):
            log_prob_list = []
            if self.config.interrupt_as_prompt:
                curr_response_len = non_tensor_batch.get("response_unpadded_len", 0)
                # Collect log probs only when the request finished normally
                # The response log probs are collected in two parts:
                # 1. The log probs of the accumulated response tokens (in current prompt tokens)
                # 2. The log probs of the current response tokens
                if vllm_output.outputs[0].finish_reason != "abort" and curr_response_len > 0:
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
            rollout_log_probs = log_prob_list
        raw_response_ids = non_tensor_batch["raw_response_ids"]
        curr_response_ids = raw_response_ids + np.fromiter([response_ids], dtype=object)
        non_tensor_batch["raw_response_ids"] = curr_response_ids
        
        if "response_unpadded_len" in non_tensor_batch:
            curr_response_unpadded_len = non_tensor_batch["response_unpadded_len"]
        else:
            curr_response_unpadded_len = [0]
        response_unpadded_len = [curr_response_unpadded_len[0] + response_len]
        non_tensor_batch["response_unpadded_len"] = np.array(response_unpadded_len, dtype=int)
        non_tensor_batch["interrupted"] = np.array([interrupted], dtype=bool)
        
        batch = TensorDict(
            {
                "input_ids": prompt.batch["input_ids"],
            },
            batch_size=1,
        )
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompt.meta_info)

    async def generate(
        self,
        request: DataProto,
    ) -> DataProto:
        assert len(request) == 1, "RolloutRouter only supports single request generation."

        request_ids = request.get("uid", None)
        update_status_success = await self.ps_manager_handle.update_request_status.remote(
            request_ids.tolist(),
            RequestStatus.ROLLOUT_DISPATCHED,
        )
        if update_status_success[0]:
            request_id = request.non_tensor_batch["uid"][0]
            needed_model_version = request.non_tensor_batch["version_tag"][0]
            gen_worker = self._choose_gen_worker(request)
            await gen_worker.push_task.remote(request_id, needed_model_version)
            
            continue_generation = True
            while continue_generation:
                vllm_output, update_status = await gen_worker.generate_async.remote(request)
                
                request = self._consolidate_response(request, vllm_output)
                if not update_status or update_status == RequestStatus.RUNNING:
                    continue_generation = False
                elif update_status == RequestStatus.ROLLOUT_INTERRUPTED:
                    continue
            await gen_worker.pop_task.remote(request_id, needed_model_version)

            return request
        return None
