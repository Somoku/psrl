# PSRL Patches

This directory contains patches for **verl** and **vLLM** libraries to enable distributed deployment of AsyncLLM with enhanced resource management and sampling capabilities.

## Overview

The patches provide the following enhancements:

### verl Patch (v0.4.1.x & v0.5.x)

- **Custom Resource Management**: Enables workers to define custom resource requirements, environment variables, and initialization parameters by implementing a `configure_worker` class method.
- **Distributed AsyncLLM Support**: Allows for flexible distributed deployment of AsyncLLM across multiple nodes
- **Enhanced Master Node Discovery**: Improved master address/port resolution for distributed setups (v0.5.x)
- **Protocol Optimization**: Fixed tensordict serialization issues for empty batches (v0.5.x)
- **Compatibility with Disaggregated Architecture**: Fix compatibility issues with disaggregated architectures in Megatron workers

### vLLM Patch (v0.9.0.1 & v0.10.0)

- **Post-Sampling Logprobs**: Introduces `use_post_sampling_logprobs` parameter in SamplingMetadata to allow returning logprobs computed after sampling for more accurate probability distributions
- **Enhanced Request Management**: Improved abort request handling and queue monitoring capabilities (v0.10.0)
- **Queue Monitoring**: Added methods to query running and waiting queue sizes (v0.10.0)
- **Batch Abort Support**: Enhanced abort functionality to handle multiple request IDs (v0.10.0)
- **Python 3.8+ Compatibility**: Replaces `list[int]` with `List[int]` type annotations for broader Python version support

## Supported Versions

- **verl**: v0.4.1.x, v0.5.x
- **vLLM**: v0.9.0.1, v0.10.0

## Installation and Usage

### Prerequisites

Both libraries must be installed in **editable mode** to apply the patches:

```bash
# Install verl in editable mode
pip install -e /path/to/verl

# Install vllm in editable mode  
pip install -e /path/to/vllm
```

### Applying the Patches

#### verl Patch

```bash
cd psrl/patch/verl
bash apply_patch.sh
```

#### vLLM Patch

```bash
cd psrl/patch/vllm
bash apply_patch.sh
```

### Force Application (if not in editable mode)

If you want to apply patches to non-editable installations (not recommended):

```bash
bash apply_patch.sh --force
```

## Patch Details

### verl Patch Features

The verl patch modifies `verl/single_controller/ray/base.py` to support:

1. **Dynamic Worker Configuration**: Workers can implement a `configure_worker` class method to specify:
   - Custom resource requirements (CPU, GPU)
   - Environment variables
   - Initialization parameters

2. **Enhanced Ray Scheduling**: Improved placement group scheduling with:
   - Better resource allocation
   - Child task capturing
   - Custom runtime environments

3. **Flexible Resource Management**: Support for heterogeneous worker configurations in distributed setups

**Example Usage:**

```python
class CustomWorker:
    @classmethod
    def configure_worker(cls, num_gpus, bundle_indices):
        resources = {"num_gpus": 2, "num_cpus": 4}
        env_vars = {"CUDA_VISIBLE_DEVICES": "0,1"}
        init_kwargs = {"custom_param": "value"}
        return resources, env_vars, init_kwargs
```

### vLLM Patch Features

The vLLM patch modifies sampling and engine components to support:

1. **Post-Sampling Logprobs**: New `use_post_sampling_logprobs` flag in `SamplingMetadata`
   - Enables computation of logprobs after sampling for more accurate probability distributions
   - Provides better control over when logprobs are calculated
   - Backward compatibility with existing sampling behavior

2. **Enhanced Request Management** (v0.10.0):
   - Improved abort request handling with proper cleanup
   - Support for aborting multiple requests simultaneously
   - Better request state tracking and output generation

3. **Queue Monitoring** (v0.10.0):
   - Added `running_queue()`, `waiting_queue()`, and `waiting_and_running_queue()` methods
   - Real-time visibility into engine request processing status
   - Useful for load balancing and performance monitoring

4. **Python 3.8+ Compatibility**:
   - Replaces `list[int]` with `List[int]` type annotations throughout the codebase
   - Ensures compatibility with Python versions that don't support PEP 585 syntax

**Example Usage:**

```python
from vllm.v1.sample.metadata import SamplingMetadata

# Enable post-sampling logprobs
sampling_metadata = SamplingMetadata(
    # ... other parameters
    use_post_sampling_logprobs=True
)

# Monitor queue status (v0.10.0)
running_count = await async_llm.running_queue()
waiting_count = await async_llm.waiting_queue()
total_count = await async_llm.waiting_and_running_queue()

# Abort multiple requests (v0.10.0)
await async_llm.abort(["request_1", "request_2", "request_3"])
```

## File Modifications

### verl Patches (v0.4.1.x & v0.5.x)

- `verl/single_controller/ray/base.py`: Enhanced worker configuration, Ray scheduling, and master node discovery
- `verl/single_controller/base/worker.py`: Improved master address/port resolution (v0.5.x)
- `verl/protocol.py`: Fixed tensordict serialization for empty batches (v0.5.x)
- `verl/workers/actor/dp_actor.py`: Improved data type handling and imports
- `verl/workers/megatron_workers.py`: Fixed compatibility issues with disaggregated architectures
- `verl/workers/fsdp_workers.py`: Enhanced distributed initialization (v0.5.x)

### vLLM Patches (v0.9.0.1 & v0.10.0)

**Common (both versions):**
- `vllm/v1/sample/metadata.py`: Added `use_post_sampling_logprobs` field
- `vllm/v1/sample/sampler.py`: Implemented conditional post-sampling logprobs computation
- `vllm/v1/worker/gpu_input_batch.py`: Updated metadata construction with new flag
- `vllm/v1/worker/gpu_model_runner.py`: Updated metadata construction with new flag

**v0.9.0.1 specific:**
- `vllm/model_executor/layers/fused_moe/fused_moe.py`: Updated type annotations from `list[int]` to `List[int]`
- `vllm/model_executor/layers/quantization/utils/fp8_utils.py`: Updated type annotations for block-wise quantization functions

**v0.10.0 specific:**
- `vllm/v1/core/sched/scheduler.py`: Enhanced request abort handling and tracking
- `vllm/v1/engine/async_llm.py`: Added queue monitoring methods and batch abort support
- `vllm/v1/engine/core.py`: Added queue size query methods
- `vllm/v1/engine/core_client.py`: Added async queue monitoring methods
- `vllm/v1/engine/output_processor.py`: Improved request state management
- `vllm/v1/metrics/loggers.py`: Enhanced logger factory handling

## Important Notes

- These patches are designed for specific versions and may not work with other versions
- **verl v0.5.x** includes significant improvements in distributed setup and protocol handling
- **vLLM v0.10.0** adds enhanced request management and queue monitoring capabilities
- Always backup your installation before applying patches
- Test thoroughly in your specific environment before production use
- The post-sampling logprobs feature is disabled by default to maintain backward compatibility
- Queue monitoring methods in vLLM v0.10.0 provide real-time insights into engine performance
