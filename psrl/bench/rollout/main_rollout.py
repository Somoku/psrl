import os
import socket
import random
import logging
import asyncio
import torch
import numpy as np
import time
import uuid
from typing import Dict, Any, List, Union
from omegaconf import OmegaConf

import ray
import hydra
from vllm import LLM, SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import PromptType, TokensPrompt
from vllm.outputs import RequestOutput

from verl import DataProto
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local

from psrl.utils.logger import DualOutputHandler, log_dual_events, EventType
from psrl.utils.dataset.data_processor import DataProcessor
from psrl.bench.rollout.stats_collector import StatCollector

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))

def seed_everything(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SimpleRolloutTester:
    """Simplified rollout performance tester using vLLM AsyncLLM directly."""
    
    def __init__(self, config, tokenizer, processor):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        
        # Build logger
        mode_prefix = "Real" if self.config.rollout_test.get("mode", "synthetic") == "real_data" else "Syn"
        self.log_prefix = f"{mode_prefix}_TP{self.config.rollout.tensor_parallel_size}_PP{self.config.rollout.pipeline_parallel_size}_B{self.config.rollout_test.batch_size}_P{self.config.data.max_prompt_length}_R{self.config.data.max_response_length}"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))
        psrl_logger.info(f"Initialized simple rollout tester.")
        
        # Initialize AsyncLLM
        self.llm = None
        
        # Data processor for real data mode
        if self.config.rollout_test.get("mode", "synthetic") == "real_data":
            psrl_logger.info("Initializing data processor for real data mode...")
            self.data_processor = DataProcessor.remote(
                self.config,
                self.tokenizer,
                self.processor,
                None,  # No PS manager needed for testing
                None,  # No data queue needed for testing
                process_mode="batch",
            )
        else:
            self.data_processor = None
        
        # Stats collector for monitoring
        self.stats_collector = None
    
    def init_llm(self):
        """Initialize vLLM AsyncLLM engine."""
        psrl_logger.info("Initializing vLLM AsyncLLM engine...")
        
        # Get model path
        local_path = copy_to_local(
            self.config.model.path, 
            use_shm=self.config.model.get("use_shm", False)
        )
        
        # Configure vLLM engine args
        rollout_config = self.config.rollout
        engine_kwargs = {
            "model": local_path,
            "tensor_parallel_size": rollout_config.tensor_parallel_size,
            "pipeline_parallel_size": rollout_config.pipeline_parallel_size,
            "dtype": rollout_config.dtype,
            "gpu_memory_utilization": rollout_config.gpu_memory_utilization,
            "max_model_len": rollout_config.get("max_model_len", self.config.data.max_prompt_length + self.config.data.max_response_length),
            "max_num_seqs": rollout_config.max_num_seqs,
            "max_num_batched_tokens": rollout_config.max_num_batched_tokens,
            "enable_chunked_prefill": rollout_config.enable_chunked_prefill,
            "enable_prefix_caching": rollout_config.enable_prefix_caching,
            "trust_remote_code": self.config.model.get("trust_remote_code", False),
            "seed": 0,
            "worker_extension_cls": "psrl.bench.rollout.vllm_extension.vLLMWorkerExtension",
            "additional_config": {
                "max_model_len_used_in_estimation": rollout_config.max_num_seqs * 32768 * 32,
            },
        }
        
        # Create AsyncLLM with stats collector
        engine_args = AsyncEngineArgs(**engine_kwargs)
        
        # Initialize stats collector if profiling is enabled
        stat_loggers = None
        if not rollout_config.get("disable_log_stats", True):
            psrl_logger.info("Initializing stats collector for monitoring...")
            vllm_config = engine_args.create_engine_config()
            psrl_logger.info(f"vllm config: {vllm_config}")
            self.stats_collector = StatCollector(
                vllm_config=vllm_config,
                engine_index=0,
                log_dir=self.config.rollout_test.get("profile_logs_dir", "./profile_logs"),
                log_file=self.config.rollout_test.get("profile_log_file", "profile"),
            )
            stat_loggers = [self.stats_collector]
            psrl_logger.info(f"Stats collector initialized, logs will be saved to: {self.stats_collector.log_file}")
        else:
            self.stats_collector = None
        
        self.llm = AsyncLLM.from_engine_args(engine_args, stat_loggers=stat_loggers)
        
        psrl_logger.info("vLLM AsyncLLM engine initialized successfully!")
    
    async def run(self, config):
        """Execute the rollout performance test."""
        from pprint import pprint
        
        print(f"SimpleRolloutTester hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        
        # Initialize LLM
        self.init_llm()
        
        # Run performance test
        await self.run_performance_test()
        
    def _create_test_data(self) -> List[Dict[str, Any]]:
        vllm_input_1 = {"prompt_token_ids": [self.tokenizer.pad_token_id] * 1024}
        vllm_input_2 = {"prompt_token_ids": [self.tokenizer.pad_token_id] * 1024 * 7}
        return [vllm_input_1] * 64 + [vllm_input_2] * 8

    def _create_synthetic_data(self, batch_size: int) -> List[Dict[str, Any]]:
        """Create synthetic test data with consistent prompt length."""
        # Create prompts with consistent length using padding tokens
        prompt_length = self.config.data.max_prompt_length
        
        # Pad to desired length
        padded_tokens = [self.tokenizer.pad_token_id] * prompt_length
        
        # Ensure prompt_token_ids is list[int] as required by vLLM
        if isinstance(padded_tokens, np.ndarray):
            padded_tokens = padded_tokens.tolist()
        elif not isinstance(padded_tokens, list):
            padded_tokens = list(padded_tokens)
        
        # Create vLLM-compatible input format
        vllm_input = {"prompt_token_ids": padded_tokens}
        
        # Create batch of identical prompts
        test_prompts = [vllm_input] * batch_size
        return test_prompts

    def _get_real_data_batch(self) -> List[Dict[str, Any]]:
        """Get real data batch from data processor."""
        try:
            # Get batch from data processor
            batch_data = ray.get(self.data_processor.get_single_controller_batch.remote("train"))
            batch = DataProto.from_single_dict(batch_data)
            
            # Extract prompts from batch and convert to vLLM format
            prompts = []
            for i in range(len(batch)):
                if "raw_prompt_ids" in batch.non_tensor_batch:
                    # Use existing token IDs
                    prompt_token_ids = batch.non_tensor_batch["raw_prompt_ids"][i]
                else:
                    # Fallback to tokenized input
                    input_ids = batch.batch["input_ids"][i]
                    prompt_token_ids = input_ids.tolist()
                
                # Ensure prompt_token_ids is list[int] as required by vLLM
                if isinstance(prompt_token_ids, np.ndarray):
                    prompt_token_ids = prompt_token_ids.tolist()
                elif not isinstance(prompt_token_ids, list):
                    prompt_token_ids = list(prompt_token_ids)
                
                # Create vLLM-compatible input format
                vllm_input = {"prompt_token_ids": prompt_token_ids}
                prompts.append(vllm_input)
            
            return prompts
            
        except Exception as e:
            psrl_logger.error(f"Failed to get real data batch: {e}")
            raise

    async def _generate_batch_async(self, prompts: List[Dict[str, Any]], synthetic_list: List[bool]) -> List[RequestOutput]:
        """Generate responses for a batch of prompts using AsyncLLM."""
        # Create sampling parameters with EOS token disabled to force max length generation
        sampling_params = [SamplingParams(
            temperature=self.config.rollout.temperature,
            top_p=self.config.rollout.get("top_p", 1.0),
            top_k=self.config.rollout.get("top_k", -1),
            max_tokens=self.config.data.max_response_length,
            stop=None,
            # Disable EOS token to force generation to max length
            ignore_eos=True if synthetic else False,
        ) for synthetic in synthetic_list]
        
        # Log generation start if stats collector is available
        if self.stats_collector is not None:
            for i, prompt_data in enumerate(prompts):
                # Calculate actual token length from prompt_token_ids
                prompt_token_ids = prompt_data["prompt_token_ids"]
                prompt_length = len(prompt_token_ids)
                self.stats_collector.log_generation_start(
                    request_id=f"batch_{i}",
                    prompt_length=prompt_length,
                    max_tokens=self.config.data.max_response_length
                )
        
        # Create async tasks with individual timing using asyncio.gather
        async def process_single_request(i: int, prompt_data: Dict[str, Any]) -> RequestOutput:
            """Process a single request with proper timing."""
            start_time = time.time()
            
            # Generate response using TokensPrompt
            final_output = None
            async for output in self.llm.generate(
                prompt=TokensPrompt(**prompt_data),
                sampling_params=sampling_params[i],
                request_id=str(uuid.uuid4()),
            ):
                final_output = output  # Keep the last output
            
            # Calculate actual generation time for this specific request
            generation_time = time.time() - start_time
            
            # Log generation end if stats collector is available
            if self.stats_collector is not None:
                generated_length = len(final_output.outputs[0].token_ids) if final_output.outputs else 0
                finish_reason = final_output.outputs[0].finish_reason if final_output.outputs else "unknown"
                self.stats_collector.log_generation_end(
                    request_id=f"batch_{i}",
                    generated_length=generated_length,
                    finish_reason=finish_reason,
                    generation_time=generation_time
                )
            
            return final_output
        
        # Process all requests concurrently with proper individual timing
        tasks = [process_single_request(i, prompt_data) for i, prompt_data in enumerate(prompts)]
        results = await asyncio.gather(*tasks)
        
        return results
    
    async def estimate_max_model_len(self):
        """Estimate the max model length."""
        max_model_len = await self.llm.collective_rpc(
            "estimate_max_model_len",
            args=(),
        )
        return max_model_len

    async def run_performance_test(self):
        """Run performance tests using AsyncLLM directly."""
        test_mode = self.config.rollout_test.get("mode", "synthetic")  # "synthetic" or "real_data"
        estimated_max_model_len = await self.estimate_max_model_len()
        psrl_logger.info(f"Estimated max model length: {estimated_max_model_len}")
        psrl_logger.info(f"Starting rollout performance test in {test_mode} mode...")
        
        # Run performance test
        num_iterations = self.config.rollout_test.num_iterations
        warmup_iterations = self.config.rollout_test.warmup_iterations
        batch_size = self.config.rollout_test.batch_size
        
        psrl_logger.info(f"Running {warmup_iterations} warmup iterations + {num_iterations} test iterations")
        
        # Warmup
        for i in range(warmup_iterations):
            psrl_logger.info(f"Warmup iteration {i+1}/{warmup_iterations}")
            try:
                # Run a small batch to warm up the model
                test_prompts = self._create_synthetic_data(16)
                synthetic_list = [False] * 16
                
                # Generate responses
                results = await self._generate_batch_async(test_prompts, synthetic_list)
                psrl_logger.debug(f"Warmup iteration {i+1} generated {len(results)} sequences")
            except Exception as e:
                psrl_logger.warning(f"Warmup failed: {e}")
                raise e
        
        # Log warmup completion if stats collector is available
        if self.stats_collector is not None:
            self.stats_collector.warmup_completion(warmup_iterations)
            psrl_logger.info("=" * 30)
            psrl_logger.info("WARMUP PHASE COMPLETED")
            psrl_logger.info("=" * 30)
            psrl_logger.info(f"Completed {warmup_iterations} warmup iterations")
            psrl_logger.info("Starting performance test phase with detailed monitoring...")
            psrl_logger.info("=" * 30)
        
        # Performance test
        times = []
        for i in range(num_iterations):
            psrl_logger.info(f"Test iteration {i+1}/{num_iterations}")
            
            start_time = time.time()
            try:
                if test_mode == "synthetic":
                    # test_prompts = self._create_test_data()
                    test_prompts = self._create_synthetic_data(batch_size)
                    synthetic_list = [True] * batch_size
                else:
                    test_prompts = self._get_real_data_batch()
                    long_prompt = {"prompt_token_ids": [self.tokenizer.pad_token_id] * 1024 * 10}
                    # test_prompts = [long_prompt] * 32 + test_prompts
                    # synthetic_list = [True] * 32 + [False] * batch_size
                    synthetic_list = [False] * batch_size
                
                # Generate responses
                results = await self._generate_batch_async(test_prompts, synthetic_list)
                psrl_logger.debug(f"Test iteration {i+1} generated {len(results)} sequences")
            except Exception as e:
                psrl_logger.error(f"Generation failed: {e}")
                raise e
            
            end_time = time.time()
            iteration_time = end_time - start_time
            times.append(iteration_time)
            psrl_logger.info(f"Iteration {i+1} took {iteration_time:.2f} seconds")
        
        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        psrl_logger.info("=" * 50)
        psrl_logger.info("ROLLOUT PERFORMANCE TEST RESULTS")
        psrl_logger.info("=" * 50)
        psrl_logger.info(f"Test mode: {test_mode}")
        psrl_logger.info(f"Batch size: {batch_size}")
        psrl_logger.info(f"Test iterations: {num_iterations}")
        psrl_logger.info(f"Average time per iteration: {avg_time:.2f} ± {std_time:.2f} seconds")
        psrl_logger.info(f"Min time: {min_time:.2f} seconds")
        psrl_logger.info(f"Max time: {max_time:.2f} seconds")
        psrl_logger.info(f"Throughput: {batch_size / avg_time:.2f} samples/second")
        
        # Print stats collector summary if available
        if self.stats_collector is not None:
            psrl_logger.info("=" * 30)
            psrl_logger.info("VLLM ENGINE STATISTICS")
            psrl_logger.info("=" * 30)
            summary_stats = self.stats_collector.get_summary_stats()
            for key, value in summary_stats.items():
                psrl_logger.info(f"{key}: {value}")
            psrl_logger.info("=" * 30)
        
        psrl_logger.info("=" * 50)


@hydra.main(config_path="config", config_name="rollout_test", version_base=None)
def main(config):
    import asyncio
    asyncio.run(run_rollout_test(config))

async def run_rollout_test(config) -> None:
    """Run the simplified rollout performance test."""
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with minimal configuration
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    # Download model checkpoint and initialize tokenizer/processor
    local_path = copy_to_local(
        config.model.path, 
        use_shm=config.model.get("use_shm", False)
    )
    
    # Initialize tokenizer and processor
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
    
    # Create and run the simple rollout tester
    tester = SimpleRolloutTester(config, tokenizer, processor)
    await tester.run(config)

    # Save timeline if specified
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)

if __name__ == "__main__":
    seed_everything(0)
    main()

