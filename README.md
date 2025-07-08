# PSRL

PSRL is a post-training framework for LLMs that supports both synchronous and asynchronous training paradigms through staleness control, streaming rollout and parameter server architecture.

## Quick Start

### Installation

**Requirements:**

- GCC >= 9
- Python 3.9+
- PyTorch 2.7.1
- CUDA 12.8

```bash
# Use conda to manage the environment
conda create -n psrl python=3.10
conda activate psrl

# Install all dependencies (including Megatron)
# If you have an existing **editable** vLLM or veRL installation,
# you can pass in VLLM_PATH and VERL_PATH to the script.
USE_MEGATRON=1 bash scripts/install_env.sh

# Install PSRL
pip install -e .
```

### Training

You can find training examples in the `examples` directory. We provide examples for different training paradigms, including:

- **Algorithms**: PPO, GRPO
- **Training Backends**: FSDP, Megatron-LM
- **Rollout Modes**: Batch, Streaming
- **Synchronous/Asynchronous Training**: controlled by `staleness` parameter (0 for synchronous, >0 for asynchronous)
- **Rollout Strategies**: Heterogeneous, Homogeneous TP/PP

As an example, you can run the following command to start training:

```bash
# For PPO with FSDP backend, batch rollout, homogeneous TP/PP and synchronous training
bash examples/ppo_trainer/fsdp/train.sh
```

## Acknowledgements

PSRL is built upon the foundations of [verl](https://github.com/volcengine/verl), an open-source RLHF framework from ByteDance Seed.

We also appreciate all the pioneering and inspirable projects from the community, including but not limited to vLLM, OpenRLHF, AReaL and NeMo-RL.