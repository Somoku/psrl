# mini-SWE-agent Training Recipe

Train language models to solve software engineering tasks using reinforcement learning. This recipe integrates [mini-SWE-agent](https://github.com/SWE-agent/mini-SWE-agent) (v2) as a **Python library** with PSRL's trainer, enabling models to learn from interactive coding feedback in Docker-sandboxed environments.

## Overview

The training loop works as follows:

1. **Data**: Each training sample contains a problem statement and a reference patch.
2. **Rollout**: For each sample, mini-SWE-agent's `DefaultAgent` runs **in-process** (in a worker thread). It manages a Docker container and has the model interact with the codebase by running bash commands.
3. **Model Bridge**: A custom `_PSRLModel` (inheriting `LitellmTextbasedModel`) intercepts every `query()` call and routes generation through PSRL's vLLM rollout engine via thread-safe queues -- no HTTP proxy, no subprocess.
4. **Reward**: After the agent finishes, its generated patch is compared against the reference patch to produce a reward signal.
5. **Training**: PSRL applies GRPO policy gradient updates using the collected trajectories and rewards.

```
┌─────────────────────────────────────────────────────┐
│               PSRL GRPO Trainer                     │
│  (actor, ref model, vLLM rollout, reward scoring)   │
└──────────────────────┬──────────────────────────────┘
                       │  per-episode
          ┌────────────┴────────────┐
          │  MiniSWEAgentLoop.run() │
          │  (async event loop)     │
          └────────────┬────────────┘
                       │
     ┌─────────────────┼──────────────────┐
     │                 │                  │
     ▼                 ▼                  ▼
┌──────────┐   ┌──────────────┐   ┌────────────────────┐
│  Docker  │   │ _PSRLModel   │   │ DefaultAgent.run() │
│container │   │ .query()     │◄──│ (worker thread)    │
│ (sandbox)│   │ (queue bridge)│   └────────────────────┘
└──────────┘   └──────┬───────┘
                      │
               ┌──────┴───────┐
               │ vLLM generate│
               │ (on-policy)  │
               └──────┬───────┘
                      │
               ┌──────┴───────┐
               │ compute_score│
               │ (patch diff) │
               └──────────────┘
```

## Directory Structure

```
examples/mini_swe/
├── README.md                             # This file
├── config.py                             # Runtime config dataclasses (env, agent, sandbox)
├── reward.py                             # Patch-based reward function (compute_score)
├── config/
│   └── mini_swe_agent_config.yaml        # Agent loop config template
└── prepare/
    ├── prepare_data.py                   # Dataset generator (simple synthetic tasks)
    ├── bake_simple_repos.sh              # Bake repos into Docker image
    ├── simple_cases_train.json           # Training cases (40 synthetic bug-fix tasks)
    └── simple_cases_val.json             # Validation cases (12 synthetic bug-fix tasks)

# Core integration modules inside psrl/
psrl/workers/agent_loop/loops/mini_swe_agent_loop.py   # MiniSWEAgentLoop + _PSRLModel
psrl/workers/agent_loop/agent_data/mini_swe_agent_data.py  # MiniSWEAgentData
psrl/environments/mini_swe_env.py                      # MiniSWEEnvironment
```

## Prerequisites

### Hardware

- NVIDIA GPUs (tested on RTX 3090 24GB, A100, H100; 8x GPUs per node recommended)
- Sufficient disk space for model checkpoints (~50 GB per checkpoint)
- Docker installed and accessible on all worker nodes

### Software Dependencies

```bash
# 1. Create conda environment and install PSRL (from repo root)
conda create -n psrl python=3.11
conda activate psrl

bash scripts/install_basic.sh
bash scripts/install_nixl.sh
bash scripts/install_megatron.sh
bash scripts/install_tms.sh
bash scripts/install_lmcache.sh

python -m pip install -e .

# 2. Install mini-SWE-agent (used as a Python library, not just CLI)
python -m pip install mini-swe-agent
python -c "from minisweagent.agents.default import DefaultAgent; print('OK')"

# 3. Pull the Docker sandbox image
docker pull python:3.11-slim

# 4. Prepare model weights
ls /path/to/models/Qwen/Qwen3-4B-Instruct/config.json
```

### Pre-flight Check

```bash
nvidia-smi -L | wc -l                                   # expect: >= 1
python -c "from minisweagent.agents.default import DefaultAgent"  # no error
docker run --rm python:3.11-slim bash -c "echo ok"       # expect: ok
```

## Data Preparation

### Step 1: Bake repos into Docker image

```bash
bash prepare/bake_simple_repos.sh python:3.11-slim
```

> **Note**: The bake script contains commented-out proxy settings. Uncomment and edit if your network requires a proxy.

### Step 2: Generate parquet datasets

```bash
python prepare/prepare_data.py \
    --mode simple \
    --train_size 64 \
    --test_size 16 \
    --output_dir data/mini_swe_agent
```

## Training

A ready-to-run multi-node FSDP script is provided:

```bash
bash examples/mini_swe/fsdp_qwen_7b_dapo.sh
```

The script targets 4 nodes × 8 GPUs (2 nodes for rollout, 2 for training) with
Qwen2.5-7B-Instruct. Edit the cluster-layout variables at the top to match your
hardware.

Key parameters set by the script (passed to `psrl.trainer.main_ppo`):

```
gen_actor_rollout_ref.rollout.multi_turn.enable=True
gen_actor_rollout_ref.rollout.multi_turn.max_turns=30
gen_actor_rollout_ref.rollout.agent.agent_loop_config_path=<path>/mini_swe_agent_config.yaml
gen_actor_rollout_ref.rollout.agent.env.name=mini_swe_env
gen_actor_rollout_ref.rollout.agent.data.name=mini_swe_agent_data
custom_reward_function.path=<path>/examples/mini_swe/reward.py
custom_reward_function.name=compute_score
algorithm.adv_estimator=grpo
```

For a minimal single-node smoke test:

```bash
PSRL_PATH=$(python -c "import psrl; import os; print(os.path.dirname(os.path.dirname(psrl.__file__)))")

python -m psrl.trainer.main_ppo --config-path=./config --config-name='ppo_trainer' \
    data.train_files="['${PSRL_PATH}/examples/mini_swe/data/mini_swe_agent/train.parquet']" \
    data.val_files="['${PSRL_PATH}/examples/mini_swe/data/mini_swe_agent/test.parquet']" \
    data.prompt_key=prompt \
    data.return_raw_chat=True \
    data.max_prompt_length=2048 \
    data.max_response_length=16384 \
    gen_actor_rollout_ref.model.path=/path/to/model \
    gen_actor_rollout_ref.rollout.multi_turn.enable=True \
    gen_actor_rollout_ref.rollout.multi_turn.max_turns=30 \
    gen_actor_rollout_ref.rollout.agent.agent_loop_config_path=${PSRL_PATH}/examples/mini_swe/config/mini_swe_agent_config.yaml \
    gen_actor_rollout_ref.rollout.agent.env.name=mini_swe_env \
    gen_actor_rollout_ref.rollout.agent.data.name=mini_swe_agent_data \
    custom_reward_function.path=${PSRL_PATH}/examples/mini_swe/reward.py \
    custom_reward_function.name=compute_score \
    algorithm.adv_estimator=grpo \
    trainer.total_epochs=10 \
    trainer.project_name=mini_swe_agent \
    trainer.experiment_name=mini_swe_grpo
```

## Configuration

### Config hierarchy

```
MiniSWEAgentRuntimeConfig (dataclass defaults in config.py)
  └── mini_swe_agent_config.yaml (deployment overrides)
       └── extra_info per instance (runtime overrides via sandbox_overrides / agent_overrides)
```

### Agent loop config (`config/mini_swe_agent_config.yaml`)

`agent.system_template` and `agent.problem_template` are **required** — no hardcoded defaults exist. `build_runtime_config` raises `ValueError` if either is left empty after YAML merge.

```yaml
- name: mini_swe_agent
  _target_: psrl.workers.agent_loop.loops.mini_swe_agent_loop.MiniSWEAgentLoop
  sandbox_config:
    max_parallel_tasks_per_worker: 0   # 0 = unlimited
    environment:
      image: "python:3.11-slim"
      container_timeout: "2h"
  agent:
    system_template: |
      <your system prompt here>
    problem_template: |
      <your per-problem prompt here — use {{ task }}, {{ cwd }}, {{ system }}, etc.>
```

### Key config fields

| Field | Category | Description |
|-------|----------|-------------|
| `rollout.multi_turn.enable` | Required | Must be `True` for mini-SWE-agent |
| `rollout.multi_turn.max_turns` | Required | Max LLM generation turns per episode (also passed as agent `step_limit`) |
| `psrl.agentic_rl.trajectory_output.enable` | Infrastructure | Whether to write per-problem `.txt` trajectory files |
| `psrl.agentic_rl.trajectory_output.dir` | Infrastructure | Base output directory; trajectories are grouped by model version under `v{N}/` sub-directories. Defaults to `<psrl.logging_path>/trajectories` |
| `sandbox_config.max_parallel_tasks_per_worker` | Infrastructure | Concurrency limit per node; `0` = unlimited |
| `sandbox_config.environment.image` | Data-affine | Docker image for sandbox (default: `python:3.11-slim`) |
| `sandbox_config.environment.cwd` | Data-affine | Working directory inside container (default: `/testbed`) |
| `sandbox_config.environment.container_timeout` | Infrastructure | Max container lifetime (default: `2h`) |
| `sandbox_config.environment.env` | Infrastructure | Docker container env vars (default: disables pagers) |
| `sandbox_config.environment.run_args` | Infrastructure | Extra `docker run` flags |
| `agent.system_template` | **Required** | System prompt template (no default) |
| `agent.problem_template` | **Required** | Per-problem prompt template; maps to mini-swe-agent's `instance_template` kwarg (no default) |
| `agent.cost_limit` | Optional | LiteLLM cost limit per episode (default: `0.0` = unlimited) |

## Architecture: How It Works

### _PSRLModel (in-process model bridge)

`_PSRLModel` inherits `LitellmTextbasedModel` from mini-swe-agent and overrides only `query()`. When mini-swe-agent's `DefaultAgent` calls `model.query(messages)`:

1. `_PSRLModel.query()` puts messages into a `request_queue` (thread-safe).
2. The async generation loop (in the event loop thread) picks up the request.
3. PSRL's vLLM generates a response; the turn is recorded in `MiniSWEAgentData`.
4. The response text is put into `response_queue`.
5. `_PSRLModel.query()` returns the response to `DefaultAgent`.

Inherited from `LitellmTextbasedModel` (no custom code needed):
- `format_observation_messages()` -- renders Docker command output as observation messages.
- `_parse_actions()` -- extracts bash commands from `mswea_bash_command` fenced blocks.
- `format_message()` -- formats system/user messages.
- `get_template_vars()` / `serialize()` -- template variables and trajectory serialization.

### Thread bridging

mini-swe-agent's `DefaultAgent.run()` is synchronous. PSRL's rollout is async. The bridge:
- `DefaultAgent.run(task)` runs in a **dedicated thread pool** (`_AGENT_THREAD_POOL`, max 32 threads, separate from the default asyncio executor to prevent deadlocks with `asyncio.to_thread`).
- `DockerEnvironment` is created **in the worker thread** (its `__init__` starts a Docker container synchronously).
- The async generation loop polls `request_queue` with a 0.1 s sleep and feeds `response_queue`.
- `_PSRLModel.query()` blocks on `response_queue` with a 600 s timeout.

### Reward Function

Computes a patch-based reward with bash-tool-use shaping for `mini_swe_agent` and `mini_swe_agent_simple` data sources. Tool usage detection targets mini-SWE-agent v2 bash patterns (`sed -i`, `cat <<`, `tee`, `patch`, etc.).

| Condition | Score |
|-----------|-------|
| Exact patch match | `1.0` |
| Partial patch match | `0.10` -- `0.85` |
| Patch on wrong files | `0.05` |
| No patch, but edited correct file | `0.05` |
| No patch, but ran tests or python | `0.03` |
| No patch, but made edits (wrong file) | `0.02` |
| No patch, but explored correct file (cat/ls) | `0.02` |
| No patch, but explored code (cat/ls, wrong file) | `0.01` |
| Alignment failed / 0 turns / timeout | `0.0` |
| Long and fruitless (>=10 turns, no editor) | `-0.05` |
| Premature exit (<=2 turns, no tool use) | `-0.1` |

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `ImportError: minisweagent` | mini-swe-agent not installed | `pip install mini-swe-agent` |
| `docker: Cannot connect` | Docker not running | `sudo systemctl start docker` |
| No patch found | Agent didn't submit before turn limit | Increase `rollout.multi_turn.max_turns` |
| OOM during rollout | Too many concurrent containers | Reduce `max_parallel_tasks_per_worker` |
| Alignment failed every episode | Context truncation | Use `max_model_len` instead of message-level truncation |

### Emergency Cleanup

```bash
docker ps -q --filter "label=psrl.swe_problem_id" | xargs -r docker stop
ray stop --force
pkill -9 -f main_ppo
```

## Extending

### Custom Docker Image

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*
RUN python -m pip install pytest numpy
```

Set `sandbox_config.environment.image` in the YAML config.

### Custom Reward Functions

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    """Returns a float reward (typically in [-0.1, 1.0])."""
```

### Custom Prompt Templates

Override templates in `mini_swe_agent_config.yaml` via `agent.system_template` / `agent.problem_template`. Templates use Jinja2 with variables: `{{ task }}`, `{{ cwd }}`, `{{ system }}`, `{{ release }}`, `{{ version }}`, `{{ machine }}`.

The model must output bash commands in `mswea_bash_command` fenced blocks (text-based model class).
