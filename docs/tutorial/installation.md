# Installation

PSRL is installed from source because its core runtime combines pinned PyTorch,
vLLM, veRL, SMG, NIXL, and optional KV-cache components.

## Prerequisites

| Requirement | Notes |
|---|---|
| OS | Ubuntu 22.04+ |
| GPU | NVIDIA GPU supported by the pinned PyTorch/CUDA stack |
| CUDA | CUDA 12.8-compatible driver/toolchain |
| Python | 3.12 recommended |
| Rust/Cargo | Required to build SMG's Rust gateway and Python binding |
| Build tools | Git, C/C++ toolchain, CMake/Ninja, and sufficient disk space |
| Multi-node network | InfiniBand/RoCE recommended for `nixl_cpu` and LMCache P2P |

Install Rust before running the core installer:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustc --version
cargo --version
```

## Create the Environment

```bash
conda create -n psrl python=3.12
conda activate psrl
```

All nodes in a Ray cluster must use the same environment and see the same PSRL
checkout or installed package.

## Core Installation

```bash
bash scripts/install_basic.sh
pip install .
```

`install_basic.sh` currently installs:

- PyTorch 2.11.0 for CUDA 12.8, Triton, TensorDict, FlashAttention, FlashInfer,
  Apex, and common Python dependencies.
- **SMG from the `psrl-dev` branch**, including the release Rust gateway, Python
  binding, gRPC protocol/client package, PSRL state protocol, and gRPC servicer.
- vLLM `releases/v0.22.0` and the PSRL vLLM patches.
- A pinned veRL checkout and the PSRL veRL patches.
- `torch_memory_saver`.

The installer clones sources under `third_party/`. Set `VLLM_PATH` or `VERL_PATH`
before running it to use an existing checkout.

## SMG Requirements

SMG is not optional in the current default architecture:

- `psrl.rollout_gateway.enable=True` launches an SMG Router process.
- Rollout instances register as gRPC workers, so SMG's gRPC client/proto and
  servicer packages must be installed.
- The PSRL worker selector uses the SMG `psrl-state` gRPC protocol to contact
  PSManager.
- SessionRouter/TITO-based agent loops require an SMG build with TITO enabled.

The core installer performs the equivalent of:

```bash
cd third_party/smg
cargo build --release
python -m pip install -e crates/grpc_client/python/
python -m pip install -e crates/psrl_state/python/
python -m pip install -e grpc_servicer/
cd bindings/python
maturin develop --features vendored-openssl
```

For SMG development, rebuild from the exact SMG checkout used by PSRL, then restart
the training job so the gateway subprocess loads the new binding.

## Optional Performance Components

Run these after the core installer:

::::{tab-set}

:::{tab-item} NIXL / RDMA

```bash
bash scripts/install_nixl.sh
```

Required for `psrl.ps_mode=nixl_cpu`. NIXL/UCX selects local shared-memory/IPC paths
where possible and RDMA-capable transports across nodes.
:::

:::{tab-item} Megatron
:sync: megatron

Installs Megatron-LM as an alternative training backend for large models.

```bash
bash scripts/install_megatron.sh
```

Required for Megatron-LM actor/critic training with TP, PP, CP, or EP.
:::

:::{tab-item} LMCache
:sync: lmcache

Installs LMCache for KV cache offloading to CPU/disk and cross-instance P2P KV
transfer. Dramatically reduces re-prefill cost in multi-turn agentic RL workloads.

```bash
bash scripts/install_lmcache.sh
```

Required for `psrl.lmcache.enable=True` and cross-instance KV transfer. P2P transfer
with `p2p_transfer_channel=nixl` also requires NIXL/UCX.
:::

::::

`torch_memory_saver` is installed by `install_basic.sh`; this checkout does not
provide a separate `install_tms.sh`.

The pinned veRL requirements install `TransferQueue==0.1.7` during the core
installation.

`SimpleStorage` is the default TransferQueue backend.

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
