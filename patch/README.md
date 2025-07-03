# PSRL Patches

This directory contains patches for **veRL** and **vLLM** libraries to enable distributed deployment of AsyncLLM with enhanced resource management and sampling capabilities.

## Overview

The patches provide the following enhancements:

### veRL Patch (v0.4.1.x)
- **Custom Resource Management**: Enables workers to define custom resource requirements, environment variables, and initialization parameters by implementing a `configure_worker` class method.
- **Distributed AsyncLLM Support**: Allows for flexible distributed deployment of AsyncLLM across multiple nodes

### VLLM Patch (v0.9.0)
- **Post-Sampling Logprobs**: Introduces `use_post_sampling_logprobs` parameter in SamplingMetadata to allow returning logprobs computed after sampling for more accurate probability distributions
- **Python 3.8+ Compatibility**: Replaces `list[int]` with `List[int]` type annotations for broader Python version support

## Supported Versions

- **VERL**: v0.4.1.x
- **VLLM**: v0.9.0

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

#### VERL Patch

```bash
cd psrl/patch/verl
bash apply_patch.sh
```

#### VLLM Patch

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

### VERL Patch Features

The VERL patch modifies `verl/single_controller/ray/base.py` to support:

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

### VLLM Patch Features

The VLLM patch modifies sampling components to support:

1. **Post-Sampling Logprobs**: New `use_post_sampling_logprobs` flag in `SamplingMetadata`
   - Enables computation of logprobs after sampling for more accurate probability distributions
   - Provides better control over when logprobs are calculated
   - Backward compatibility with existing sampling behavior

2. **Python 3.8+ Compatibility**: 
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
```

## File Modifications

### VERL Patch
- `verl/single_controller/ray/base.py`: Enhanced worker configuration and Ray scheduling
- `verl/workers/actor/dp_actor.py`: Improved data type handling and imports

### VLLM Patch
- `vllm/model_executor/layers/fused_moe/fused_moe.py`: Updated type annotations from `list[int]` to `List[int]`
- `vllm/model_executor/layers/quantization/utils/fp8_utils.py`: Updated type annotations for block-wise quantization functions
- `vllm/v1/sample/metadata.py`: Added `use_post_sampling_logprobs` field
- `vllm/v1/sample/sampler.py`: Implemented conditional post-sampling logprobs computation
- `vllm/v1/worker/gpu_input_batch.py`: Updated metadata construction with new flag
- `vllm/v1/worker/gpu_model_runner.py`: Updated metadata construction with new flag

## Important Notes

- These patches are designed for specific versions and may not work with other versions
- Always backup your installation before applying patches
- Test thoroughly in your specific environment before production use
- The post-sampling logprobs feature is disabled by default to maintain backward compatibility
