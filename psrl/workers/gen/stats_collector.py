import os
import time
import logging
import numpy as np
from datetime import datetime
from omegaconf import DictConfig
from dataclasses import dataclass
from typing import Optional, Final

from vllm.config import VllmConfig
from vllm.v1.metrics.loggers import StatLoggerBase
from vllm.v1.metrics.stats import IterationStats, SchedulerStats

from psrl.utils.logger import FileOnlyHandler


psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@dataclass
class EngineStats:
    # immutable properties
    instance_id: Final[int]
    model_version: Final[int]
    snapshot: Final[dict]
    
    # mutable properties
    _waiting_and_running_queue_size: int = 0  

    def __post_init__(self):
        num_waiting_reqs = self.snapshot.get("scheduler_stats", {}).get("num_waiting_reqs", 0)
        num_running_reqs = self.snapshot.get("scheduler_stats", {}).get("num_running_reqs", 0)
        self._waiting_and_running_queue_size = num_waiting_reqs + num_running_reqs

    @staticmethod
    def get_default_snapshot() -> dict:
        return {
            "timestamp": datetime.now().isoformat(),
            "total_elapsed_time": 0.0,
            "elapsed_time_since_last_record": 0.0,
            "scheduler_stats": {
                "num_running_reqs": 0,
                "num_waiting_reqs": 0,
                "kv_cache_usage": 0.0,
            },
        }

    # mutable operations
    def get_waiting_and_running_queue_size(self) -> int:
        return self._waiting_and_running_queue_size

    def increment_waiting_and_running_queue_size(self) -> None:
        self._waiting_and_running_queue_size += 1
        

class StatCollector(StatLoggerBase):
    """
    Collects and reports statistics from vLLM engine to the rollout coordinator.
    
    This class extends vLLM's StatLoggerBase to:
    - Monitor scheduler statistics (waiting/running request counts)
    - Detect changes in request queue status
    - Send status updates to output queue for coordinator consumption
    """
    
    def __init__(self, vllm_config: VllmConfig, psrl_config: DictConfig, instance_id: int = 0):
        """
        Initialize the StatCollector.
        
        Args:
            vllm_config: vLLM configuration object
            instance_id: Unique identifier for this engine instance
        """
        self.vllm_config = vllm_config
        self.psrl_config = psrl_config
        self.instance_id = instance_id
        
        self.model_version = 0
        self._begin_record = False
        self.start_time = None
        self.last_record_time = None
        self.last_push_to_queue_time = None
        
        # Build logger
        if self.psrl_config.status_collection.dump_logging_to_file_level != "none":
            self.log_prefix = f"StatCollector_I{self.instance_id}"
            psrl_logger.addHandler(FileOnlyHandler(self.psrl_config.logging_path, self.log_prefix))
            psrl_logger.info(f"Initialized StatCollector for instance {self.instance_id}.")
        
    def begin_record(self):
        """
        Begin recording statistics.
        """
        self._begin_record = True
        self.start_time = time.time()
        self.last_record_time = self.start_time
        self.last_push_to_queue_time = self.start_time
        
    def _make_snapshot_after_model_version_update(self) -> dict:
        curr_time = time.time()
        snapshot = EngineStats.get_default_snapshot()
        snapshot["total_elapsed_time"] = curr_time - self.start_time
        snapshot["elapsed_time_since_last_record"] = curr_time - self.last_record_time
        return snapshot

    def init_output_queue(self, output_queue):
        """
        Initialize the output queue for sending status updates.
        
        Args:
            output_queue: Ray queue for sending status updates to coordinator
        """
        self.output_queue = output_queue

    def record_model_version_update(self, model_version: int):
        """
        Set the model version for this engine instance.
        
        Args:
            model_version: Model version
        """
        self.model_version = model_version
        # We force a record to the output queue to ensure the coordinator knows the model version update immediately
        self.output_queue.put_nowait(EngineStats(
            instance_id=self.instance_id,
            model_version=self.model_version,
            snapshot=self._make_snapshot_after_model_version_update(),
        ))
        self.last_record_time = time.time()
        self.last_push_to_queue_time = self.last_record_time

    def record(
        self,
        scheduler_stats: Optional[SchedulerStats],
        iteration_stats: Optional[IterationStats],
        engine_idx: int = 0,
    ):
        """
        Record and process scheduler statistics from vLLM engine.
        
        This method is called by vLLM during inference to report statistics.
        When request counts change, it sends updates to the coordinator via output queue.
        
        Args:
            scheduler_stats: Statistics from vLLM scheduler (request counts, etc.)
            iteration_stats: Statistics from vLLM iteration
            engine_idx: Engine index (not currently used)
        """
        assert self.output_queue is not None, f"Output queue is not initialized"

        curr_time = time.time()
        
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "total_elapsed_time": curr_time - self.start_time,
            "elapsed_time_since_last_record": curr_time - self.last_record_time,
            "scheduler_stats": {
                "num_running_reqs": scheduler_stats.num_running_reqs,
                "num_waiting_reqs": scheduler_stats.num_waiting_reqs,
                "kv_cache_usage": scheduler_stats.kv_cache_usage,
            },
        }
       
        if iteration_stats:
            time_to_first_tokens_iter = getattr(iteration_stats, 'time_to_first_tokens_iter', [])
            num_prompt_reqs = len(time_to_first_tokens_iter)
            if len(time_to_first_tokens_iter) == 0:
                time_to_first_tokens_iter = [0.0]
            inter_token_latencies_iter = getattr(iteration_stats, 'inter_token_latencies_iter', [])
            num_generation_reqs = len(inter_token_latencies_iter)
            if len(inter_token_latencies_iter) == 0:
                inter_token_latencies_iter = [0.0]
            iteration_stats_entry = {
                "num_prompt_tokens": getattr(iteration_stats, 'num_prompt_tokens', 0),
                "num_generation_tokens": getattr(iteration_stats, 'num_generation_tokens', 0),
                "num_prompt_reqs": num_prompt_reqs,
                "num_generation_reqs": num_generation_reqs,
                "num_preempted_reqs": getattr(iteration_stats, 'num_preempted_reqs', 0),
                "num_finished_reqs": len(getattr(iteration_stats, 'finished_requests', [])),
                "max_time_to_first_tokens": np.max(time_to_first_tokens_iter),
                "max_inter_token_latencies": np.max(inter_token_latencies_iter),
                "avg_time_to_first_tokens": np.mean(time_to_first_tokens_iter),
                "avg_inter_token_latencies": np.mean(inter_token_latencies_iter),
            }
            snapshot["iteration_stats"] = iteration_stats_entry
        
        if self.psrl_config.status_collection.dump_logging_to_file_level != "none":
            if self.psrl_config.status_collection.dump_logging_to_file_level == "prompt":
                if iteration_stats and snapshot["iteration_stats"]["num_prompt_reqs"] > 0:
                    psrl_logger.info(f"Snapshot (model version {self.model_version}): {snapshot}")
            elif self.psrl_config.status_collection.dump_logging_to_file_level == "all":
                psrl_logger.info(f"Snapshot (model version {self.model_version}): {snapshot}")
        self.last_record_time = curr_time
            
        if curr_time - self.last_push_to_queue_time >= self.psrl_config.status_collection.engine_sync_interval_in_ms / 1000.0:
            self.output_queue.put_nowait(EngineStats(
                instance_id=self.instance_id,
                model_version=self.model_version,
                snapshot=snapshot,
            ))
            self.last_push_to_queue_time = curr_time

    def log_engine_initialized(self):
        """
        Log engine initialization information including cache configuration.
        
        This method is called when the vLLM engine completes initialization
        and reports the number of GPU blocks available for caching.
        """
        if self.vllm_config.cache_config.num_gpu_blocks:
            psrl_logger.info(f"Engine {self.instance_id}: vllm cache_config_info with initialization "
                             f"after num_gpu_blocks is: {self.vllm_config.cache_config.num_gpu_blocks}")
