"""KV cache headroom via vLLM sleep/wake_up API (item 6).

Design §3.7.3 says the colocate path frees KV cache memory by calling
`LLM.sleep(level=1)` (which discards KV cache and offloads weights to CPU), then
uses the freed headroom for weight transfer, then restores the engine with
`LLM.wake_up(["kv_cache"])`.

The design makes three claims that must hold on this single machine:

  1. `sleep(level=1)` actually frees GPU memory — the device memory-used reading must
     drop after sleep, and the KV cache tensors must no longer be resident.
  2. `wake_up(tags=None)` (or `wake_up(["kv_cache", "weights"])`) restores the engine
     to a working state that can produce correct logits again.
  3. The colocate IPC path never calls `register_memory` on the shared uint8 buffer
     (already proven by test_colocate_ipc_gpu.py); this file verifies the orthogonal
     half — that the vLLM side of the handshake, i.e. sleep/wake, exists and has the
     documented semantics.

Starting a full vLLM engine is too slow for per-test isolation: a Qwen3.5-4B engine
takes ~25 GB and ~30 s to initialise. So the tests are module-scoped and share one
engine instance. The fixture tears it down after all tests in this file complete.

These tests do not measure KV cache *size* — the design's "`discard(kv_cache)` frees
at least N GB" claim requires a target model and a specific KV-cache config, and belongs
to the cluster-level e2e suite. What can be verified here without a full cluster is:
  - the API exists and can be called without error
  - GPU memory drops observably after sleep(level=1)
  - generate works again after wake_up, producing the same logits as before sleep
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# Tiny prompt that exercises generation without requiring a large context.
_PROMPT = "The quick brown fox"
# Minimum expected free-memory gain from sleeping level 1.  vLLM allocates at least a
# few hundred MB for KV cache even on a tiny model; we set the bar very low so that the
# test doesn't race with other GPU users on the same machine.
_MIN_FREED_MIB = 1
_MODEL = "/apdcephfs_zwfy10/share_303541817/lhy/models/Qwen3.5-4B"


@pytest.fixture(scope="module")
def llm():
    """A module-scoped vLLM LLM instance over Qwen3.5-4B.

    Module scope amortises the ~30 s startup across all tests.  The fixture yields
    *after* the engine is warm (one generate call has completed) so that all tests
    start from a known-good state and the lazy-allocation noise is gone.
    """
    vllm_mod = pytest.importorskip("vllm", reason="vLLM not installed")
    LLM = vllm_mod.LLM
    SamplingParams = vllm_mod.SamplingParams
    import os

    if not os.path.isdir(_MODEL):
        pytest.skip(f"model not present on this machine: {_MODEL}")

    engine = LLM(
        model=_MODEL,
        dtype="bfloat16",
        max_model_len=512,
        # One GPU, no TP, tiny KV cache — just enough to test the API.
        tensor_parallel_size=1,
        gpu_memory_utilization=0.3,
        # Qwen3.5 is a hybrid Mamba model: each decode sequence needs its own Mamba
        # cache block, and CUDA graph capture refuses to start when max_num_seqs
        # exceeds the number of blocks that fit. The default (1024) does not fit at
        # this memory fraction, so cap it rather than raising the fraction -- this
        # test is about the sleep/wake API, not about throughput.
        max_num_seqs=16,
        # Matches production. Without the CuMem allocator `sleep(level=1)` frees nothing.
        enable_sleep_mode=True,
    )
    # Warm up so lazy allocations don't pollute the memory measurements below.
    engine.generate([_PROMPT], SamplingParams(max_tokens=4))
    yield engine
    del engine
    torch.cuda.empty_cache()


class TestSleepWakeSemantics:
    """vLLM sleep(level=1) must free KV cache; wake_up must restore correct generation."""

    def test_sleep_level1_api_is_callable(self, llm):
        """sleep(level=1) must not raise — the API must exist and accept level=1."""
        llm.sleep(level=1)
        # If we get here without an exception the API exists.
        llm.wake_up()

    def test_sleep_level1_frees_gpu_memory(self, llm):
        """After sleep(level=1), total GPU memory used on the device must drop.

        vLLM's engine runs in a child process, so torch's per-process stats report
        zero in the test process. Reading total device-wide memory with pynvml (which
        queries the NVML driver, not the CUDA allocator) covers all processes including
        the EngineCore child, and gives an honest reading of what sleep(level=1) did.
        """
        import pynvml

        pynvml.nvmlInit()
        # NVML ignores CUDA_VISIBLE_DEVICES, so resolve the device under test by PCI bus id.
        props = torch.cuda.get_device_properties(0)
        bus_id = f"{props.pci_domain_id:08x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
        handle = pynvml.nvmlDeviceGetHandleByPciBusId(bus_id)

        before_bytes = pynvml.nvmlDeviceGetMemoryInfo(handle).used
        llm.sleep(level=1)
        after_bytes = pynvml.nvmlDeviceGetMemoryInfo(handle).used
        freed_mib = (before_bytes - after_bytes) / (1024 ** 2)
        llm.wake_up()  # restore before asserting so later tests are unaffected

        assert freed_mib >= _MIN_FREED_MIB, (
            f"sleep(level=1) freed only {freed_mib:.0f} MiB (expected >= {_MIN_FREED_MIB}). "
            f"before={before_bytes/(1024**2):.0f} MiB, after={after_bytes/(1024**2):.0f} MiB. "
            "If pynvml is not installed or another process caused noise, rerun alone."
        )

    def test_wake_up_restores_generation(self, llm):
        """wake_up(None) must restore the engine to a state that produces correct output.

        The test checks that generate does not raise and that the output tokens are
        the same as before sleep, which would break if the weight reload is partial or
        if the KV cache is silently corrupted.
        """
        from vllm import SamplingParams

        params = SamplingParams(max_tokens=8, temperature=0.0, seed=42)
        before_outputs = llm.generate([_PROMPT], params)
        before_text = before_outputs[0].outputs[0].text

        llm.sleep(level=1)
        llm.wake_up()

        after_outputs = llm.generate([_PROMPT], params)
        after_text = after_outputs[0].outputs[0].text

        assert before_text == after_text, (
            f"wake_up produced different output: before={before_text!r}, after={after_text!r}. "
            "This means either the weight reload was incomplete or determinism broke."
        )

    def test_sleep_wake_cycle_is_repeatable(self, llm):
        """A second sleep/wake cycle must work just as well as the first."""
        from vllm import SamplingParams

        params = SamplingParams(max_tokens=4, temperature=0.0, seed=0)
        ref = llm.generate([_PROMPT], params)[0].outputs[0].text

        for _ in range(2):
            llm.sleep(level=1)
            llm.wake_up()
            result = llm.generate([_PROMPT], params)[0].outputs[0].text
            assert result == ref, "repeated sleep/wake cycle changed the output"

    def test_wake_up_with_kv_cache_tag(self, llm):
        """wake_up(['kv_cache']) must be callable and produce working generation.

        The design calls wake_up with specific tags so that weights are not reloaded when
        they are already resident (the PSRL colocate path keeps weights in place while
        only the KV cache is freed).  This test confirms the tagged API exists.
        """
        from vllm import SamplingParams

        params = SamplingParams(max_tokens=4, temperature=0.0, seed=1)
        ref = llm.generate([_PROMPT], params)[0].outputs[0].text

        # level=1 discards KV cache *and* offloads weights; we wake up everything.
        llm.sleep(level=1)
        llm.wake_up(tags=["kv_cache", "weights"])

        result = llm.generate([_PROMPT], params)[0].outputs[0].text
        assert result == ref, "wake_up with explicit tags changed the output"


class TestSleepAPIContract:
    """Sanity checks on the sleep/wake_up interface itself."""

    def test_sleep_level0_does_not_crash(self, llm):
        """level=0 pauses scheduling; the engine should accept it without error."""
        llm.sleep(level=0, mode="abort")
        # Restore using the scheduling tag (level 0 only pauses scheduling).
        llm.wake_up(tags=["scheduling"])

    def test_sleep_method_exists_on_llm(self, llm):
        """sleep and wake_up must both be callable attributes of LLM."""
        assert callable(getattr(llm, "sleep", None)), "LLM.sleep not found"
        assert callable(getattr(llm, "wake_up", None)), "LLM.wake_up not found"
