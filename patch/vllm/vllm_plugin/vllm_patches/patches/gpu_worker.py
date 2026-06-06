import logging
import os
from contextlib import AbstractContextManager, nullcontext

import torch
from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized
from vllm.tracing import instrument
from vllm.utils.mem_utils import format_gib
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu_worker import Worker

from vllm_patches.core import min_vllm_version, vLLMPatch

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@min_vllm_version("0.22.0")
class TMSWorkerPatch(vLLMPatch[Worker]):
    """
    Replace cuMemAllocator with torch_memory_saver
    for better memory management.

    Compatible with vLLM 0.22.0+
    """

    def sleep(self, level: int = 1) -> None:
        """Put the worker into sleep mode to free up GPU memory."""
        from torch_memory_saver import torch_memory_saver

        free_bytes_before_sleep = torch.cuda.mem_get_info()[0]

        # Save the buffers before level 2 sleep
        if level == 2:
            model = self.model_runner.model
            self._sleep_saved_buffers = {name: buffer.cpu().clone() for name, buffer in model.named_buffers()}

        torch_memory_saver.pause("weights")
        torch_memory_saver.pause("kv_cache")
        if os.environ.get("PSRL_VLLM_PATCHES", "") == "TMS:GRAPH":
            torch_memory_saver.pause("graph")

        free_bytes_after_sleep, total = torch.cuda.mem_get_info()
        freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        used_bytes = total - free_bytes_after_sleep
        assert freed_bytes >= 0, "Memory usage increased after sleeping."
        psrl_logger.info(
            "Sleep mode freed %.2f GiB memory, %.2f GiB memory is still in use.",
            format_gib(freed_bytes),
            format_gib(used_bytes),
        )

    def wake_up(self, tags: list[str] | None = None) -> None:
        """Wake up from sleep mode and restore memory."""
        from torch_memory_saver import torch_memory_saver

        free_bytes_before_wake_up = torch.cuda.mem_get_info()[0]

        for tag in tags:
            torch_memory_saver.resume(tag)

        # Restore the buffers after level 2 sleep
        if len(self._sleep_saved_buffers):
            model = self.model_runner.model
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}

        if tags is None or "kv_cache" in tags:
            self.model_runner.post_kv_cache_wake_up()

        free_bytes_after_wake_up, total = torch.cuda.mem_get_info()
        increased_bytes = free_bytes_before_wake_up - free_bytes_after_wake_up
        used_bytes = total - free_bytes_after_wake_up
        assert increased_bytes >= 0, "Memory usage increased after waking up."
        psrl_logger.info(
            "Wake up mode increased %.2f GiB memory, %.2f GiB memory is still in use.",
            format_gib(increased_bytes),
            format_gib(used_bytes),
        )

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        """Get the memory pool context manager if sleep mode is enabled."""
        if not self.vllm_config.model_config.enable_sleep_mode:
            return nullcontext()

        from torch_memory_saver import torch_memory_saver

        enable_weights_cpu_backup = self.vllm_config.additional_config.get("enable_weights_cpu_backup", False)
        return torch_memory_saver.region(
            tag=tag,
            enable_cpu_backup=enable_weights_cpu_backup,
        )

    @instrument(span_name="Allocate KV cache")
    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""

        # Update local config with adjusted num blocks after profiling,
        # so that it's available to the warmup stage.
        self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks

        # Init kv cache connector here, because it requires
        # `kv_cache_config`.
        # NOTE(Kuntai): This need to be done before `initialize_kv_cache`,
        # because `initialize_kv_cache` will inject kv cache groups not
        # related to kv cache connector (e.g. kv cache sharing layers).
        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)

        if self.vllm_config.model_config.enable_sleep_mode:
            from torch_memory_saver import torch_memory_saver

            with torch_memory_saver.region(tag="kv_cache"):
                self.model_runner.initialize_kv_cache(kv_cache_config)
        else:
            self.model_runner.initialize_kv_cache(kv_cache_config)

        if self.model_config.enable_return_routed_experts:
            self.model_runner.init_routed_experts_capturer()

        # Build KV-zero metadata outside the CuMem pool so the bookkeeping
        # GPU tensors (seg_addrs, block-id buffers) use the standard PyTorch
        # allocator and are not discarded during sleep/wake cycles.
        if kv_cache_config.needs_kv_cache_zeroing and hasattr(self.model_runner, "_init_kv_zero_meta"):
            self.model_runner._init_kv_zero_meta()
