# Installation

This guide covers installing PSRL and all its optional components from source.

## Prerequisites

| Requirement | Minimum |
|---|---|
| GPU | NVIDIA GPU |
| OS | Ubuntu 22.04+ |
| CUDA | 12.8+ |
| Python | 3.11 |

## Create Conda Environment

```bash
conda create -n psrl python=3.11
conda activate psrl
```

## Install Components

PSRL is modular, install only what you need. The **Core** tab is mandatory, all other
tabs are optional extensions that can be installed in any order *after* Core.

:::{important}
`scripts/install_basic.sh` (Core) **must** be run first. The remaining install scripts
can be run in any order after Core completes.
:::

:::{tip}
**Recommended: Install everything for maximum performance.**
All optional components work together and are used in production deployments. Unless
you have strong constraints on disk space or build time, install them all:

```bash
bash scripts/install_basic.sh    # Core (required)
bash scripts/install_nixl.sh     # RDMA weight sync
bash scripts/install_megatron.sh # Megatron-LM backend (large models)
bash scripts/install_tms.sh      # GPU memory sharing (TMS)
bash scripts/install_lmcache.sh  # KV cache offloading
```
:::

::::{tab-set}

:::{tab-item} Core
:sync: core

Installs PyTorch 2.9, vLLM, veRL, and FlashAttention, the minimum stack required for
any PSRL training job.

```bash
bash scripts/install_basic.sh
```

This script pins compatible versions of all core dependencies, applies PSRL-specific
patches to the vendored vLLM and veRL copies under `third_party/`, and may take 10--20
minutes depending on network speed.
:::

:::{tab-item} NIXL
:sync: nixl

Installs the NIXL library and UCX for RDMA-based weight synchronization between training
and generation clusters. Required when using `ps_mode: nixl_cpu`.

```bash
bash scripts/install_nixl.sh
```

Requires InfiniBand/RoCE-capable NICs for multi-node GPU-direct transfers.
:::

:::{tab-item} Megatron
:sync: megatron

Installs Megatron-LM as an alternative training backend for large models.

```bash
bash scripts/install_megatron.sh
```

Use this when training with TP/PP/CP/EP via the Megatron backend.
:::

:::{tab-item} TMS
:sync: tms

Installs `torch_memory_saver` (TMS), a GPU memory manager that transparently swaps
idle tensors to CPU, enabling colocated rollout and training workers to share GPU
efficiently.

```bash
bash scripts/install_tms.sh
```
:::

:::{tab-item} LMCache
:sync: lmcache

Installs LMCache for KV cache offloading to CPU/disk and cross-instance P2P KV
transfer. Dramatically reduces re-prefill cost in multi-turn agentic RL workloads.

```bash
bash scripts/install_lmcache.sh
```
:::

::::

## Install PSRL Package

After installing the desired components, install the PSRL package itself in editable
mode:

```bash
python -m pip install -e .
```

## Verification

Run the checks below in sequence. Commands marked **(optional)** apply only if you
installed the corresponding component.

**Core**

```bash
# Python and CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# PSRL package
python -c "import psrl; print('PSRL:', psrl.__file__)"

# vLLM (patched third-party copy)
python -c "import vllm; print('vLLM:', vllm.__version__)"

# Ray
python -c "import ray; ray.init(); print('Ray resources:', ray.cluster_resources()); ray.shutdown()"

# veRL (patched third-party copy)
python -c "import verl; print('veRL:', verl.__file__)"

# FlashAttention
python -c "import flash_attn; print('FlashAttention:', flash_attn.__version__)"
```

**NIXL** *(optional, installed via `install_nixl.sh`)*

```bash
python -c "import nixl; print('NIXL OK')"
```

**Megatron-LM** *(optional, installed via `install_megatron.sh`)*

```bash
python -c "import megatron; print('Megatron-LM:', megatron.__file__)"
```

**TMS (torch_memory_saver)** *(optional, installed via `install_tms.sh`)*

```bash
python -c "import torch_memory_saver; print('TMS:', torch_memory_saver.__file__)"
```

**LMCache** *(optional, installed via `install_lmcache.sh`)*

```bash
python -c "import lmcache; print('LMCache:', lmcache.__file__)"
```

If all commands succeed without errors, your installation is ready. Proceed to the
{doc}`quickstart` to run your first training job.
