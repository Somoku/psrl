import logging
import os
from collections.abc import Callable

import torch
from torch_memory_saver import torch_memory_saver
from tqdm import tqdm
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import graph_capture, is_global_first_rank
from vllm.model_executor.offloader.base import get_offloader
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor, CudaGraphManager

from vllm_patches.core import min_vllm_version, vLLMPatch

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@min_vllm_version("0.18.0")
class TMSCudaGraphManagerPatch(vLLMPatch[CudaGraphManager]):
    """
    Replace `torch.cuda.graph()` with `torch_memory_saver.cuda_graph()`

    Compatible with vLLM 0.18.0+
    """

    @torch.inference_mode()
    def capture(
        self,
        create_forward_fn: Callable[[BatchExecutionDescriptor], Callable[[CUDAGraphMode], None]],
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        """Capture CUDA graphs.

        Args:
            create_forward_fn: Factory that prepares inputs (OUTSIDE graph) and
                returns a function that runs forward with a given CUDAGraphMode.
        """
        with graph_capture(device=self.device):
            # Capture in order: PIECEWISE first, then FULL. PIECEWISE has larger
            # activations so FULL activations should fit in already allocated
            # buffers in the graph pool.
            for mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]:
                if mode not in self._capture_descs:
                    continue

                descs = self._capture_descs[mode]
                if is_global_first_rank():
                    descs = tqdm(descs, desc=f"{progress_bar_desc} ({mode.name})")
                for desc in descs:
                    # Prepare inputs and get forward function
                    forward_fn = create_forward_fn(desc)

                    # Warmup
                    forward_fn(CUDAGraphMode.NONE)

                    # Capture
                    psrl_logger.debug("CG Capture: mode=%s, batch_desc=%s", desc.cg_mode.name, desc)
                    if desc.cg_mode == CUDAGraphMode.PIECEWISE:
                        forward_fn(CUDAGraphMode.PIECEWISE)
                    else:
                        assert desc not in self.graphs, f"Graph already captured for {desc}"
                        graph = torch.cuda.CUDAGraph()
                        # Sync offloader's copy stream before capture.
                        # Ensure any pre-capture prefetches from offloader are complete.
                        get_offloader().sync_prev_onload()
                        with torch_memory_saver.cuda_graph(graph, self.pool, tag="graph"):
                            forward_fn(CUDAGraphMode.NONE)
                            # Join offloader's copy stream after forward to avoid
                            # unjoined stream error. The last layer's start_prefetch
                            # forks copy_stream, but wait_prefetch only happens in
                            # the next forward pass.
                            get_offloader().join_after_forward()
                        self.graphs[desc] = graph
        self._graphs_captured = True
