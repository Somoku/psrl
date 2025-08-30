import os
import time
import logging
import numpy as np
from typing import Optional

from vllm.config import VllmConfig
from vllm.v1.metrics.loggers import StatLoggerBase
from vllm.v1.metrics.stats import IterationStats, SchedulerStats

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

class StatCollector(StatLoggerBase):
    """
    Collects and reports statistics from vLLM engine to the rollout coordinator.
    
    This class extends vLLM's StatLoggerBase to:
    - Monitor scheduler statistics (waiting/running request counts)
    - Detect changes in request queue status
    - Send status updates to output queue for coordinator consumption
    """
    
    def __init__(self, vllm_config: VllmConfig, engine_index: int = 0):
        """
        Initialize the StatCollector.
        
        Args:
            vllm_config: vLLM configuration object
            engine_index: Unique identifier for this engine instance
        """
        self.engine_index = engine_index
        self.vllm_config = vllm_config
        self.last_scheduler_stats = SchedulerStats()
        self.last_request_counts = (0, 0)

    def init_output_queue(self, output_queue):
        """
        Initialize the output queue for sending status updates.
        
        Args:
            output_queue: Ray queue for sending status updates to coordinator
        """
        self.output_queue = output_queue

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
            iteration_stats: Statistics from vLLM iteration (not currently used)
            engine_idx: Engine index (not currently used)
        """
        assert self.output_queue is not None, f"Output queue is not initialized"

        if scheduler_stats is not None:
            self.last_scheduler_stats = scheduler_stats
            curr_request_counts = (scheduler_stats.num_running_reqs, scheduler_stats.num_waiting_reqs)
            if self.last_request_counts != curr_request_counts:
                psrl_logger.debug(f"Update engine status to {curr_request_counts}")
                self.last_request_counts = curr_request_counts
                self.output_queue.put_nowait(
                    (self.engine_index, self.last_request_counts))

    def log_engine_initialized(self):
        """
        Log engine initialization information including cache configuration.
        
        This method is called when the vLLM engine completes initialization
        and reports the number of GPU blocks available for caching.
        """
        if self.vllm_config.cache_config.num_gpu_blocks:
            psrl_logger.info(
                "Engine %03d: vllm cache_config_info with initialization "
                "after num_gpu_blocks is: %d", self.engine_index,
                self.vllm_config.cache_config.num_gpu_blocks)
