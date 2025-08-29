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
        rollout_wg_list,
    ):
        self.config = config
        self.ps_manager_handle = ps_manager_handle
        self.rollout_wg_list = rollout_wg_list
        self.rollout_wg_size = len(rollout_wg_list)

        self.rank_0_is_model_owner = (
            self.config.gen_actor_rollout_ref.rollout.tensor_model_parallel_size *
                self.config.gen_actor_rollout_ref.rollout.pipeline_model_parallel_size > 1 and
            self.config.gen_actor_rollout_ref.rollout.mode == "psrl_async"
        )
        
        # Engine status tracking
        # NOTE: The engine status will be updated by RolloutCoordinator
        # and can be accessed via get_engine_status() method
        self.latest_engine_status = {}

        self._gen_worker_idx = 0
        
        # Build logger
        self.log_prefix = f"RolloutRouter"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))

    def update_engine_status(self, engine_status: dict):
        self.latest_engine_status = engine_status

    def _choose_gen_worker(self, request: DataProto) -> int:
        # min_request_num = float('inf')
        # chosen_worker = 0
        # for instance_id in range(self.rollout_wg_size):
        #     request_num = self.latest_engine_status.get("instances", {}).get(instance_id, {}).get("waiting_and_running_queue_size", 0)
        #     if request_num < min_request_num:
        #         min_request_num = request_num
        #         chosen_worker = instance_id
        # return instance_id
        
        # use round robin
        chosen_worker = self._gen_worker_idx
        self._gen_worker_idx = (self._gen_worker_idx + 1) % self.rollout_wg_size
        # psrl_logger.debug(f"Chosen gen worker {chosen_worker} for request {request.non_tensor_batch.get('uid', 'unknown')}")
        return chosen_worker

    def _consolidate_responses(
        self,
        prompts: DataProto,
        vllm_outputs,
    ) -> DataProto:
        if not isinstance(vllm_outputs, list):
            vllm_outputs = [vllm_outputs]
        assert len(vllm_outputs) == len(prompts), "Mismatched batch size between prompts and VLLM outputs."

        batch_size = len(prompts)
        non_tensor_batch = prompts.non_tensor_batch

        response_ids_list = []
        response_len_list = []
        interrupted_list = []
        all_log_prob_list = []

        for i in range(batch_size):
            vllm_output = vllm_outputs[i]
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
                self.config.psrl.log_prob.enable_inference_engine_log_prob and
                hasattr(vllm_output.outputs[0], 'logprobs') and
                vllm_output.outputs[0].logprobs is not None
            ):
                if self.config.psrl.rollout_test.partial_rollout.interrupt_as_prompt:
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
        if self.config.psrl.log_prob.enable_inference_engine_log_prob:
            if "rollout_log_probs" in non_tensor_batch:
                curr_rollout_log_probs = non_tensor_batch["rollout_log_probs"]
                # curr_rollout_log_probs = curr_rollout_log_probs.reshape(batch_size, -1)
            else:
                curr_rollout_log_probs = np.fromiter(([] for _ in range(batch_size)), dtype=object)
            curr_rollout_log_probs += np.fromiter(all_log_prob_list, dtype=object)
            non_tensor_batch["rollout_log_probs"] = curr_rollout_log_probs

        batch = TensorDict(
            {
                "input_ids": prompts.batch["input_ids"],
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)

    def _consolidate_response(
        self,
        prompt,
        vllm_output,
    ) -> DataProto:
        assert len(vllm_output.outputs) == 1, "RolloutRouter only supports single request generation."
        
        batch_size = len(prompt)
        non_tensor_batch = prompt.non_tensor_batch
        response_ids = vllm_output.outputs[0].token_ids
        response_len = len(response_ids)
        interrupted = vllm_output.outputs[0].finish_reason == "abort"    
        # if inference logprobs is required, we need to collect the log probabilities
        if (
            self.config.psrl.log_prob.enable_inference_engine_log_prob and
            hasattr(vllm_output.outputs[0], 'logprobs') and
            vllm_output.outputs[0].logprobs is not None
        ):
            log_prob_list = []
            if self.config.psrl.rollout_test.partial_rollout.interrupt_as_prompt:
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

        if self.config.psrl.log_prob.enable_inference_engine_log_prob:
            if "rollout_log_probs" in non_tensor_batch:
                curr_rollout_log_probs = non_tensor_batch["rollout_log_probs"]
                curr_rollout_log_probs = curr_rollout_log_probs.reshape(batch_size, -1)
            else:
                curr_rollout_log_probs = np.empty((batch_size, 0), dtype=object)
            non_tensor_batch["rollout_log_probs"] = np.concatenate([curr_rollout_log_probs, np.array(rollout_log_probs, dtype=object).reshape(batch_size, -1)], axis=-1)
        
        batch = TensorDict(
            {
                "input_ids": prompt.batch["input_ids"],
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompt.meta_info)

    def generate(
        self,
        requests: DataProto,
    ):
        request_ids = requests.non_tensor_batch.get("uid", None)
        update_status_success = ray.get(self.ps_manager_handle.update_request_status.remote(
            request_ids.tolist(),
            RequestStatus.ROLLOUT_DISPATCHED,
        ))
        filtered_request_idxs = [i for i, success in enumerate(update_status_success) if success]
        if filtered_request_idxs:
            requests = requests[filtered_request_idxs]
            request_ids = requests.non_tensor_batch["uid"]
            needed_model_versions = requests.non_tensor_batch["version_tag"]
            # evenly dispatch to rollout workers
            filtered_requests_list = requests.chunk(self.rollout_wg_size)
            futures = []
            for i, requests in enumerate(filtered_requests_list):
                requests.non_tensor_batch["rollout_instance_id"] = np.array([i] * len(requests), dtype=int)
                futures.extend(
                    self.rollout_wg_list[i].execute_all_async("generate", requests)
                )
            rollout_results = ray.get(futures)
            # Process results as needed
            results = []
            for i in range(self.rollout_wg_size):
                requests = filtered_requests_list[i]
                vllm_outputs, filtered_request_idxs, update_statuses = rollout_results[i]
                filtered_requests = requests[filtered_request_idxs]
                assert (
                    update_statuses is not None and
                    all(update_status == RequestStatus.RUNNING for update_status in update_statuses)
                ), "Interruption is not implemented in batching mode"
                consolidated_results = self._consolidate_responses(filtered_requests, vllm_outputs)
                results.append(consolidated_results)
            return DataProto.concat(results)
        return None

    async def generate_async(
        self,
        request: DataProto,
    ) -> DataProto:
        assert len(request) == 1, "RolloutRouter only supports single request generation."

        request_ids = request.non_tensor_batch.get("uid", None)
        update_status_success = await self.ps_manager_handle.update_request_status.remote(
            request_ids.tolist(),
            RequestStatus.ROLLOUT_DISPATCHED,
        )
        if update_status_success[0]:
            request_id = request.non_tensor_batch["uid"][0]
            needed_model_version = request.non_tensor_batch["version_tag"][0]
            gen_worker_idx = self._choose_gen_worker(request)
            request.non_tensor_batch["rollout_instance_id"] = np.array([gen_worker_idx], dtype=int)
            # if self.rank_0_is_model_owner:
            #     await self.rollout_wg_list[gen_worker_idx].execute_rank_zero_async("push_task", request_id, needed_model_version)
            # else:
            #     await self.rollout_wg_list[gen_worker_idx].execute_all_async("push_task", request_id, needed_model_version)
            psrl_logger.debug(f"Before push task {request_id} to gen worker {gen_worker_idx} with model version {needed_model_version}")
            await self.rollout_wg_list[gen_worker_idx].execute_rank_zero_async("push_task", request_id, needed_model_version)
            psrl_logger.debug(f"Push task {request_id} to gen worker {gen_worker_idx} with model version {needed_model_version}")
            
            continue_generation = True
            while continue_generation:
                # if self.rank_0_is_model_owner:
                #     vllm_output, update_status = await self.rollout_wg_list[gen_worker_idx].execute_rank_zero_async("generate_async", request)
                # else:
                #     vllm_output, update_status = await self.rollout_wg_list[gen_worker_idx].execute_all_async("generate_async", request)
                psrl_logger.debug(f"Before generate_async for request {request_id} on gen worker {gen_worker_idx} with model version {needed_model_version}")
                vllm_output, update_status = await self.rollout_wg_list[gen_worker_idx].execute_rank_zero_async("generate_async", request)
                psrl_logger.debug(f"Received vllm output for request {request_id} on gen worker {gen_worker_idx} with update status {update_status} and model version {needed_model_version}")
                if vllm_output is None:
                    psrl_logger.debug(f"Before pop task {request_id} from gen worker {gen_worker_idx} with model version {needed_model_version}")
                    await self.rollout_wg_list[gen_worker_idx].execute_rank_zero_async("pop_task", request_id, needed_model_version)
                    psrl_logger.debug(f"Pop task {request_id} from gen worker {gen_worker_idx} with model version {needed_model_version}")
                    return None
                
                request = self._consolidate_responses(request, vllm_output)
                psrl_logger.debug(f"Consolidated response for request {request_id} on gen worker {gen_worker_idx} with model version {needed_model_version}")
                if update_status == RequestStatus.RUNNING:
                    continue_generation = False
                    psrl_logger.debug(f"continue_generation set to False for request {request_id} on gen worker {gen_worker_idx} with model version {needed_model_version}")
                elif update_status == RequestStatus.ROLLOUT_INTERRUPTED:
                    continue
    
            psrl_logger.debug(f"Generation completed for request {request_id} on gen worker {gen_worker_idx} with model version {needed_model_version}")
            # if self.rank_0_is_model_owner:
            #     await self.rollout_wg_list[gen_worker_idx].execute_rank_zero_async("pop_task", request_id, needed_model_version)
            # else:
            #     await self.rollout_wg_list[gen_worker_idx].execute_all_async("pop_task", request_id, needed_model_version)     
            psrl_logger.debug(f"Before pop task {request_id} from gen worker {gen_worker_idx} with model version {needed_model_version}")
            await self.rollout_wg_list[gen_worker_idx].execute_rank_zero_async("pop_task", request_id, needed_model_version)
            psrl_logger.debug(f"Pop task {request_id} from gen worker {gen_worker_idx} with model version {needed_model_version}")

            return request
        return None
