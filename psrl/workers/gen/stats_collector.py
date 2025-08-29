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
    def __init__(self, vllm_config: VllmConfig, engine_index: int = 0):
        self.engine_index = engine_index
        self.vllm_config = vllm_config
        self.last_scheduler_stats = SchedulerStats()
        self.last_request_counts = (0, 0)

    def init_output_queue(self, output_queue):
        self.output_queue = output_queue

    def record(
        self,
        scheduler_stats: Optional[SchedulerStats],
        iteration_stats: Optional[IterationStats],
        engine_idx: int = 0,
    ):
        assert self.output_queue is not None, f"Output queue is not initialized"

        if scheduler_stats is not None:
            self.last_scheduler_stats = scheduler_stats
            psrl_logger.debug(f"Collector get {scheduler_stats=}")
            curr_request_counts = (scheduler_stats.num_running_reqs, scheduler_stats.num_waiting_reqs)
            if self.last_request_counts != curr_request_counts:
                psrl_logger.debug(f"Update stat to {curr_request_counts=}")
                self.last_request_counts = curr_request_counts
                self.output_queue.put_nowait(
                    (self.engine_index, self.last_request_counts))

    def log_engine_initialized(self):
        if self.vllm_config.cache_config.num_gpu_blocks:
            psrl_logger.info(
                "Engine %03d: vllm cache_config_info with initialization "
                "after num_gpu_blocks is: %d", self.engine_index,
                self.vllm_config.cache_config.num_gpu_blocks)