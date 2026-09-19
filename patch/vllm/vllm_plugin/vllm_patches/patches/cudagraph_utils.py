import logging
import os

import torch
from torch_memory_saver import torch_memory_saver
from tqdm import tqdm
from vllm.compilation.counter import compilation_counter
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
from vllm.distributed.parallel_state import graph_capture, is_global_first_rank
from vllm.model_executor.offloader.base import get_offloader
from vllm.platforms import current_platform
from vllm.utils.torch_utils import current_stream
from vllm.v1.worker.gpu.cudagraph_utils import CreateForwardFn, CudaGraphManager

from vllm_patches.core import min_vllm_version, vLLMPatch

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@min_vllm_version("0.29.0")
class TMSCudaGraphManagerPatch(vLLMPatch[CudaGraphManager]):
    """Capture the v2 model runner's CUDA graphs inside a TMS region.

    `vllm/v1/worker/gpu/cudagraph_utils.py` belongs to the **v2** runner; the v1
    runner captures through `CUDAGraphWrapper` instead and is covered by
    `TMSCUDAGraphWrapperPatch`. This patch applies to both runners' needs
    indirectly: v2's `ModelCudaGraphManager` (and its speculator / dflash
    managers) build a `create_forward_fn` and delegate to this base
    `CudaGraphManager.capture`, which holds the only `torch.cuda.graph(...)` site
    in the v2 tree.

    Mirrors `CudaGraphManager.capture` from vLLM 0.29.0 with a single change:
    the FULL capture is entered through
    `torch_memory_saver.cuda_graph(..., tag="graph")` instead of
    `torch.cuda.graph(...)`, so the capture pool becomes part of the `graph`
    region and is released by the TMS sleep backend. Breakable PIECEWISE capture
    does not go through here and is covered by
    `vllm_patches.patches.breakable_cudagraph`.

    The body must be kept in step with vLLM on every version bump. The only
    intentional divergence is the capture context manager; everything else,
    including the profiling samples (`_max_full_descs_to_capture`,
    `_capture_mem_samples`), the breakable-CUDA-graph path and the graph-pool
    id selection, matches upstream.
    """

    @torch.inference_mode()
    def capture(
        self,
        create_forward_fn: CreateForwardFn,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        """Capture CUDA graphs.

        Args:
            create_forward_fn: Factory that prepares inputs (OUTSIDE graph) and
                returns a forward_fn. For FULL and breakable PIECEWISE modes,
                it is invoked once with warmup=True and again with warmup=False
                because attention backends may mutate or lazily initialize
                metadata during warmup.
        """
        with graph_capture(device=self.device):
            # Capture in order: PIECEWISE first, then FULL. PIECEWISE has larger
            # activations so FULL activations should fit in already allocated
            # buffers in the graph pool.
            for mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]:
                if mode not in self._capture_descs:
                    continue

                descs = self._capture_descs[mode]
                if mode == CUDAGraphMode.FULL and self._max_full_descs_to_capture is not None:
                    # Profiling only: capture a sample of the largest FULL
                    # graphs; the total cost is extrapolated from their
                    # per-graph memory deltas.
                    descs = descs[: self._max_full_descs_to_capture]
                if is_global_first_rank():
                    descs = tqdm(descs, desc=f"{progress_bar_desc} ({mode.name})")
                for desc in descs:
                    # Prepare inputs and get forward function
                    forward_fn = create_forward_fn(desc, warmup=True)

                    # Warmup
                    forward_fn(CUDAGraphMode.NONE)

                    # Capture
                    psrl_logger.debug("CG Capture: mode=%s, batch_desc=%s", desc.cg_mode.name, desc)
                    if desc.cg_mode == CUDAGraphMode.PIECEWISE and not self.use_breakable_cg:
                        forward_fn(CUDAGraphMode.PIECEWISE)
                    else:
                        # Capture with fresh attention state.
                        forward_fn = create_forward_fn(desc, warmup=False)
                        if desc.cg_mode == CUDAGraphMode.PIECEWISE:
                            forward_fn(CUDAGraphMode.PIECEWISE)
                            continue
                        assert desc not in self.graphs, f"Graph already captured for {desc}"
                        graph = torch.cuda.CUDAGraph()
                        # Sync offloader's copy stream before capture.
                        # Ensure any pre-capture prefetches from offloader are complete.
                        get_offloader().sync_prev_onload()
                        if self.pool is not None:
                            set_graph_pool_id(self.pool)
                        else:
                            set_graph_pool_id(current_platform.graph_pool_handle())
                        if self._capture_mem_samples is not None:
                            torch.accelerator.synchronize()
                            free_before = torch.accelerator.get_memory_info()[0]
                        with torch_memory_saver.cuda_graph(
                            graph,
                            pool=self.pool,
                            stream=current_stream(),
                            tag="graph",
                        ):
                            forward_fn(CUDAGraphMode.NONE)
                            # Join offloader's copy stream after forward to avoid
                            # unjoined stream error. The last layer's start_prefetch
                            # forks copy_stream, but wait_prefetch only happens in
                            # the next forward pass.
                            get_offloader().join_after_forward()
                        if self._capture_mem_samples is not None:
                            torch.accelerator.synchronize()
                            free_after = torch.accelerator.get_memory_info()[0]
                            self._capture_mem_samples.append(free_before - free_after)
                        self.graphs[desc] = graph
                        compilation_counter.num_cudagraph_captured += 1
        self._graphs_captured = True
