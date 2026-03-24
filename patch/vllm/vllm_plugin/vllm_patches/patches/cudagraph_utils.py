import logging
import os

import torch
import torch.nn as nn
from torch_memory_saver import torch_memory_saver
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.v1.attention.backends.utils import AttentionMetadataBuilder
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager, prepare_inputs_to_capture
from vllm.v1.worker.gpu.dp_utils import make_num_tokens_across_dp
from vllm.v1.worker.gpu.input_batch import InputBuffers

from vllm_patches.core import min_vllm_version, vLLMPatch

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@min_vllm_version("0.14.0")
class TMSCudaGraphManagerPatch(vLLMPatch[CudaGraphManager]):
    """
    Replace `torch.cuda.graph()` with `torch_memory_saver.cuda_graph()`

    Compatible with vLLM 0.14.0+
    """

    def capture_graph(
        self,
        num_tokens: int,
        model: nn.Module,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_metadata_builders: list[AttentionMetadataBuilder],
        kv_cache_config: KVCacheConfig,
    ) -> None:
        num_reqs = min(num_tokens, self.max_num_reqs)
        input_ids = input_buffers.input_ids[:num_tokens]
        if not self.uses_mrope:
            positions = input_buffers.positions[:num_tokens]
        else:
            positions = input_buffers.mrope_positions[:, :num_tokens]
        attn_metadata = prepare_inputs_to_capture(
            num_reqs,
            num_tokens,
            input_buffers,
            block_tables,
            attn_metadata_builders,
            self.max_model_len,
            kv_cache_config,
        )
        num_tokens_across_dp = make_num_tokens_across_dp(self.dp_size, num_tokens)

        # Warm up.
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            num_tokens_across_dp=num_tokens_across_dp,
        ):
            hidden_states = model(
                input_ids=input_ids,
                positions=positions,
            )
            if self.hidden_states is None:
                self.hidden_states = torch.empty_like(hidden_states)

        # Capture the graph.
        assert num_tokens not in self.graphs
        graph = torch.cuda.CUDAGraph()
        with (
            set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                num_tokens_across_dp=num_tokens_across_dp,
            ),
            torch_memory_saver.cuda_graph(graph, pool=self.pool, tag="graph"),
        ):
            hidden_states = model(
                input_ids=input_ids,
                positions=positions,
            )
            self.hidden_states[:num_tokens] = hidden_states
        self.graphs[num_tokens] = graph
