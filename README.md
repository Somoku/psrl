# PSRL

PSRL is a post-training framework for LLMs that supports both synchronous and asynchronous training paradigms through staleness control, streaming rollout and parameter server architecture.

## Quick Start

### Installation

**Requirements:**

- GCC >= 9
- Python 3.10+
- PyTorch 2.9.0
- CUDA 12.8

```bash
# Use conda to manage the environment
conda create -n psrl python=3.11
conda activate psrl

# Install all dependencies (including NIXL and Megatron)
# If you have an existing **editable** vLLM or veRL installation,
# you can pass in VLLM_PATH and VERL_PATH to the `scripts/install_basic.sh`.
bash scripts/install_basic.sh
bash scripts/install_nixl.sh
bash scripts/install_megatron.sh
bash scripts/install_tms.sh

# Install PSRL
pip install -e .
```

### Training

You can find training examples in the `examples` directory. We provide examples for different training paradigms, including:

- **Algorithms**: PPO, GRPO
- **Training Backends**: FSDP, Megatron-LM
- **Synchronous/Asynchronous Training**: controlled by `staleness` parameter (0 for synchronous, >0 for asynchronous)
- **Rollout Strategies**: Heterogeneous, Homogeneous TP/PP

As an example, you can run the following command to start training:

```bash
# For PPO with FSDP backend, batch rollout, homogeneous TP/PP and synchronous training
bash examples/ppo_trainer/fsdp/train.sh
```

## Contributing to PSRL

PSRL is open to everyone and welcomes all kinds of contributions! Please feel free to submit an Issue or PR. Before contributing, please use [pre-commit](https://pre-commit.com/#usage) to lint and format the codebase:

```bash
uv pip install pre-commit
pre-commit install

# run pre-commit to ensure code style consistency
pre-commit run --all-files --show-diff-on-failure --color=always

# run pre-commit on staged files only
pre-commit run
```

## Acknowledgements

PSRL is built upon the foundations of [verl](https://github.com/volcengine/verl), an open-source RLHF framework from ByteDance Seed.

We also appreciate all the pioneering and inspirable projects from the community, including but not limited to vLLM, OpenRLHF, AReaL and NeMo-RL.