import logging
import os
from contextlib import ExitStack

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

GRAPH_TAG = "graph"


def apply_breakable_cudagraph_patch() -> None:
    """Route breakable CUDA-graph capture into the TMS ``graph`` region.

    `BreakableCUDAGraphWrapper` is used by both model runners (v1 wraps the
    compiled model in it; v2 builds one from
    `ModelCudaGraphManager.init_breakable_cg_runner`). Its capture deliberately
    drives `CUDAGraph.capture_begin`/`capture_end` directly instead of going
    through `torch.cuda.graph()`, because `torch.cuda.graph()` runs
    `gc.collect()` and `empty_cache()` on every entry and a single capture
    session issues one begin/end pair per layer break (see
    `vllm/compilation/breakable_cudagraph.py`, `_capture`). That design also
    means these graphs bypass `TMSCUDAGraphWrapperPatch` and
    `TMSCudaGraphManagerPatch`, so their memory would never be released by the
    TMS `graph` region.

    torch_memory_saver exposes no API for segmented manual capture: its
    `cuda_graph()` is a drop-in replacement for `torch.cuda.graph()` and would
    reintroduce the per-segment `gc.collect()`/`empty_cache()` cost. Its
    `region(tag=...)` is the primitive that actually tags allocations (via the
    preload hook's region config), so this patch holds a `graph` region across
    the whole `BreakableCUDAGraphCapture` context -- the same scope TMS gives
    the region in its own `cuda_graph()` wrapper -- and leaves `_capture`'s
    single gc/empty_cache per session untouched.

    Only applied when `PSRL_VLLM_PATCHES` requests the `graph` region
    (`TMS:GRAPH`), matching the other CUDA-graph patches.
    """
    from torch_memory_saver import torch_memory_saver
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    _SENTINEL = "_psrl_breakable_cudagraph_patched"
    if getattr(BreakableCUDAGraphCapture, _SENTINEL, False):
        psrl_logger.debug("BreakableCUDAGraphCapture already patched, skipping.")
        return

    _original_enter = BreakableCUDAGraphCapture.__enter__
    _original_exit = BreakableCUDAGraphCapture.__exit__

    def __enter__(self):
        stack = ExitStack()
        stack.enter_context(torch_memory_saver.region(tag=GRAPH_TAG))
        self._psrl_tms_region_stack = stack
        try:
            return _original_enter(self)
        except BaseException:
            self._psrl_tms_region_stack = None
            stack.close()
            raise

    def __exit__(self, exc_type, exc, tb):
        try:
            return _original_exit(self, exc_type, exc, tb)
        finally:
            stack = getattr(self, "_psrl_tms_region_stack", None)
            if stack is not None:
                self._psrl_tms_region_stack = None
                stack.close()

    BreakableCUDAGraphCapture.__enter__ = __enter__
    BreakableCUDAGraphCapture.__exit__ = __exit__
    setattr(BreakableCUDAGraphCapture, _SENTINEL, True)

    psrl_logger.info("Patched BreakableCUDAGraphCapture to capture inside the TMS 'graph' region.")
