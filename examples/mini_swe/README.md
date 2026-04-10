# mini-SWE-Agent Training Recipe

Train language models to solve software engineering tasks using reinforcement learning. This recipe integrates [mini-SWE-agent](https://github.com/SWE-agent/mini-SWE-agent) (v2) as the agent framework with PSRL, enabling models to learn from interactive coding feedback in Docker-sandboxed environments.

> This recipe is migrated from the SWE-agent of [verl](https://github.com/volcengine/verl). However, mini-SWE-agent is simpler, faster, and has no dependency on SWE-ReX.

## Overview

The training loop works as follows:

1. **Data**: Each training sample contains a problem statement (e.g. "fix the bug in calculator.py") and a reference patch.
2. **Rollout**: For each sample, `mini-swe-agent` is launched as a subprocess. It spins up a Docker container and has the model interact with the codebase by running bash commands.
3. **Model Proxy**: A lightweight HTTP server intercepts the agent's LLM API calls and routes them through PSRL's vLLM rollout engine, so every token the agent generates is on-policy.
4. **Reward**: After the agent finishes (or hits the turn limit), its generated patch is compared against the reference patch to produce a reward signal.
5. **Training**: PSRL applies GRPO policy gradient updates using the collected trajectories and rewards.

```
┌─────────────────────────────────────────────────────┐
│               PSRL GRPO Trainer                     │
│  (actor, ref model, vLLM rollout, reward scoring)   │
└──────────────────────┬──────────────────────────────┘
                       │  per-episode
          ┌────────────┴────────────┐
          │  MiniSWEAgentLoop.run() │
          └────────────┬────────────┘
                       │
     ┌─────────────────┼─────────────────┐
     │                 │                 │
     ▼                 ▼                 ▼
┌──────────┐   ┌─────────────┐   ┌──────────────────┐
│  Docker  │   │ ModelProxy  │   │ mini-swe-agent   │
│container │   │ (HTTP)      │◄──│ (subprocess)     │
│ /testbed │   └──────┬──────┘   └──────────────────┘
└──────────┘          │
                      ▼
              ┌───────────────┐
              │ vLLM generate │
              │ (on-policy)   │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ compute_score │
              │ (patch diff)  │
              └───────────────┘
```

## Directory Structure

```
examples/mini_swe/
├── README.md                             # This file
├── config.py                             # Runtime config dataclasses + per-instance YAML builder
├── model_proxy.py                        # HTTP proxy: mini-SWE-agent ↔ vLLM (mimics OpenAI API)
├── subprocess_runner.py                  # Runs `mini-swe-agent` subprocess + Docker cleanup
├── reward.py                             # Patch-based reward function (compute_score)
├── config/
│   └── mini_swe_agent_config.yaml        # Agent loop config template
└── prepare/
    ├── prepare_data.py                   # Dataset generator (simple synthetic tasks)
    ├── bake_simple_repos.sh              # Bake repos into Docker image
    ├── simple_cases_train.json           # Training cases (40 synthetic bug-fix tasks)
    └── simple_cases_val.json             # Validation cases (12 synthetic bug-fix tasks)

# Core integration modules inside psrl/
psrl/workers/agent_loop/loops/mini_swe_agent_loop.py   # MiniSWEAgentLoop (registered as "mini_swe_agent")
psrl/workers/agent_loop/agent_data/mini_swe_agent_data.py  # MiniSWEAgentData
psrl/environments/mini_swe_env.py                      # MiniSWEEnvironment (registered as "mini_swe_env")
psrl/utils/common/patch_extractor.py                   # PatchExtractor (trajectory JSON / git diff fallback)
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

# 2. Install mini-SWE-agent CLI
python -m pip install mini-swe-agent
which mini-swe-agent   # verify it is on PATH

# 3. Pull the Docker sandbox image
docker pull python:3.11-slim

# 4. Prepare model weights
ls /path/to/models/Qwen/Qwen3-4B-Instruct/config.json   # or your model of choice
```

### Pre-flight Check

```bash
nvidia-smi -L | wc -l                                   # expect: >= 1
mini-swe-agent --version                                 # expect: 2.x.x
docker run --rm python:3.11-slim bash -c "echo ok"       # expect: ok
docker ps                                                # verify Docker access
```

## Data Preparation

### Step 1: Bake repos into Docker image

The simple test cases (synthetic fix-the-bug tasks) need their repos baked into the Docker image before training. At rollout time the agent container must find a pre-initialised git repo (e.g. `/train_0/`) without cloning anything.

```bash
# From the repo root:
bash examples/mini_swe/prepare/bake_simple_repos.sh python:3.11-slim
```

This script:
1. Reads `simple_cases_train.json` and `simple_cases_val.json`
2. Creates a git repo for each case inside a running container (at `/train_0/`, `/val_0/`, etc.)
3. Commits the modified container as the new `python:3.11-slim` image

> **Note**: The bake script contains commented-out proxy settings (`http_proxy` / `https_proxy`). Uncomment and edit the proxy lines at the top of `bake_simple_repos.sh` if your network requires a proxy.

After baking, verify:

```bash
docker run --rm python:3.11-slim bash -c "ls /train_0/ && git -C /train_0 log --oneline"
```

### Step 2: Generate parquet datasets

```bash
python examples/mini_swe/prepare/prepare_data.py \
    --mode simple \
    --train_size 64 \
    --test_size 16 \
    --output_dir data/mini_swe_agent
```

This produces `data/mini_swe_agent/train.parquet` and `data/mini_swe_agent/test.parquet` with fields:

| Field | Description |
|-------|-------------|
| `prompt` | Minimal chat message containing the problem statement |
| `data_source` | `"mini_swe_agent_simple"` |
| `reward_model.ground_truth` | Expected patch (unified diff) |
| `extra_info.problem_statement` | Problem text for the agent |
| `extra_info.sandbox_overrides` | `use_preexisting_repo: true`, `preexisting_repo_name: "train_0"` etc. |
| `agent_name` | `"mini_swe_agent"` |

## Training

Training uses PSRL's Hydra-based trainer (`psrl.trainer.main_ppo`) with mini-SWE-specific overrides for the agent loop, environment, data handler, and reward function.

### Minimal training command

```bash
PSRL_PATH=$(python -c "import psrl; import os; print(os.path.dirname(os.path.dirname(psrl.__file__)))")

python -m psrl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files="['data/mini_swe_agent/train.parquet']" \
    data.val_files="['data/mini_swe_agent/test.parquet']" \
    data.prompt_key=prompt \
    data.return_raw_chat=True \
    data.max_prompt_length=2048 \
    data.max_response_length=16384 \
    \
    gen_actor_rollout_ref.model.path=/path/to/model \
    gen_actor_rollout_ref.rollout.multi_turn.enable=True \
    gen_actor_rollout_ref.rollout.multi_turn.max_turns=30 \
    gen_actor_rollout_ref.rollout.agent.agent_loop_config_path=${PSRL_PATH}/examples/mini_swe/config/mini_swe_agent_config.yaml \
    gen_actor_rollout_ref.rollout.agent.env.name=mini_swe_env \
    gen_actor_rollout_ref.rollout.agent.data.name=mini_swe_agent_data \
    \
    custom_reward_function.path=${PSRL_PATH}/examples/mini_swe/reward.py \
    custom_reward_function.name=compute_score \
    \
    trainer.total_epochs=10 \
    trainer.project_name=mini_swe_agent \
    trainer.experiment_name=mini_swe_grpo
```

### Full multi-node example

For a full multi-node setup, refer to `examples/retool/fsdp_qwen_7b_dapo.sh` as a template. The key difference is replacing the retool-specific overrides with the mini-SWE-specific ones listed above. Here is a representative snippet showing the important overrides:

```bash
# Replace retool tool config with mini-SWE agent loop:
#   (remove) gen_actor_rollout_ref.rollout.multi_turn.tool_config_path=...
#   (remove) gen_actor_rollout_ref.rollout.agent.env.name=tool_env
#   (remove) gen_actor_rollout_ref.rollout.agent.data.name=tool_agent_data

# Add mini-SWE-specific overrides:
gen_actor_rollout_ref.rollout.agent.agent_loop_config_path=${PSRL_PATH}/examples/mini_swe/config/mini_swe_agent_config.yaml
gen_actor_rollout_ref.rollout.agent.env.name=mini_swe_env
gen_actor_rollout_ref.rollout.agent.data.name=mini_swe_agent_data

custom_reward_function.path=${PSRL_PATH}/examples/mini_swe/reward.py
custom_reward_function.name=compute_score

data.train_files="['data/mini_swe_agent/train.parquet']"
data.val_files="['data/mini_swe_agent/test.parquet']"
```

### Important: History Processors

When running GRPO with this recipe, **do not use `last_n_observations` or `closed_window`** history processors.

mini-SWE-agent uses a completely linear history: every step just appends to the message list. If you override templates to use dynamic context truncation that rewrites past messages, it will break token-level trajectory alignment and cause KL divergence explosion.

If you need to limit context length, use `max_model_len` in the PSRL rollout config instead of message-level truncation.

## Configuration

### Config hierarchy

```
MiniSWEAgentRuntimeConfig (dataclass defaults in config.py)
  └── mini_swe_agent_config.yaml (deployment overrides)
       └── extra_info per instance (runtime overrides, data-affine fields only)
```

All defaults live in the `MiniSWEAgentRuntimeConfig` dataclass (`config.py`). The YAML file only needs to specify values that differ from those defaults.

### Agent loop config (`config/mini_swe_agent_config.yaml`)

```yaml
- name: mini_swe_agent
  _target_: psrl.workers.agent_loop.loops.mini_swe_agent_loop.MiniSWEAgentLoop
  sandbox_config:
    swe_agent_timeout: 1800
    max_parallel_tasks_per_worker: 4
    max_model_calls_per_instance: 15
    environment:
      environment_class: docker
      image: "python:3.11-slim"
  proxy_config:
    port: 0
    timeout: 600
  agent:
    step_limit: 15
  model:
    model_name: "openai/verl-model"
```

### Key config fields

| Field | Category | Description |
|-------|----------|-------------|
| `sandbox_config.swe_agent_timeout` | Infrastructure | Total per-instance timeout in seconds (default: 1800) |
| `sandbox_config.max_parallel_tasks_per_worker` | Infrastructure | Concurrency limit per node; `0` = unlimited |
| `sandbox_config.max_model_calls_per_instance` | Data-affine | Max LLM calls per episode (default: 15) |
| `sandbox_config.environment.image` | Data-affine | Docker image for sandbox (default: `python:3.11-slim`) |
| `sandbox_config.environment.env` | Infrastructure | Docker container env vars (default: disables pagers/progress bars) |
| `sandbox_config.environment.run_args` | Infrastructure | Extra `docker run` flags |
| `agent.step_limit` | Data-affine | Max agent steps passed to mini-swe-agent (default: 15) |
| `agent.system_template` | Data-affine | System prompt template |
| `agent.instance_template` | Data-affine | Per-instance prompt template (uses `{{ task }}` and `{{ cwd }}`) |
| `model.model_name` | Infrastructure | LiteLLM model name (default: `openai/verl-model`) |
| `model.model_class` | Infrastructure | LiteLLM model class (default: `litellm_textbased`) |
| `model.cost_tracking` | Infrastructure | LiteLLM cost tracking mode (default: `ignore_errors`) |
| `proxy_config.port` | Infrastructure | `0` = OS-assigned (recommended); fixed port also supported |
| `proxy_config.timeout` | Infrastructure | Proxy request timeout in seconds (default: 600) |

Data-affine fields can be overridden per-instance via `extra_info.sandbox_overrides` and `extra_info.agent_overrides` (set during data preparation).

### Generated mini-SWE-agent YAML (per instance)

For each episode, `config.build_mini_sweagent_yaml()` generates a temporary YAML file consumed by the `mini-swe-agent -c <yaml>` CLI:

```yaml
agent:
  step_limit: 15
  cost_limit: 0
  system_template: "..."
  instance_template: "..."

model:
  model_name: "openai/verl-model"
  model_class: "litellm_textbased"
  model_kwargs:
    api_base: "http://127.0.0.1:<proxy_port>/v1"
    api_key: "verl-mini-swe-agent-key"
    temperature: 0.0
    drop_params: true
  cost_tracking: "ignore_errors"

environment:
  environment_class: "docker"
  image: "python:3.11-slim"
  cwd: "/train_0"                              # dynamically set per instance
  env:
    PAGER: cat
    MANPAGER: cat
    LESS: "-R"
    PIP_PROGRESS_BAR: "off"
    TQDM_DISABLE: "1"
  run_args:
    - "--rm"
    - "--memory=8g"
    - "--network"
    - "host"
    - "--add-host"
    - "host.docker.internal:host-gateway"
    - "--label"
    - "psrl.swe_problem_id=<instance_id>"
  container_timeout: "2h"
```

The `--label psrl.swe_problem_id=<id>` is injected per-instance so that `cleanup_instance_containers()` can find and stop any residual containers after the episode ends.

## Key Components

### MiniSWEAgentLoop (`psrl/workers/agent_loop/loops/mini_swe_agent_loop.py`)

The core agent loop, registered with PSRL as `"mini_swe_agent"`. For each episode it:

1. Parses `extra_info` to get the problem statement and repo settings
2. Merges per-instance overrides with config defaults (`apply_data_overrides`)
3. Acquires a parallel-run slot (via `fcntl` file lock, if `max_parallel_tasks_per_worker > 0`)
4. Starts a `ModelProxy` HTTP server on a free port
5. Generates a temporary mini-swe-agent YAML config pointing at the proxy
6. Launches `mini-swe-agent` as an async subprocess
7. Intercepts each agent API call; sends to vLLM for on-policy generation
8. Extracts the final patch from the trajectory JSON and returns `AgentLoopOutput`

### ModelProxy (`examples/mini_swe/model_proxy.py`)

A lightweight aiohttp HTTP server that mimics the OpenAI Chat Completions API. mini-SWE-agent sends requests here (via LiteLLM), thinking it's talking to an LLM API. The proxy queues each request and blocks until PSRL's vLLM generates a response, then returns it as plain text content.

The recipe uses `model_class: litellm_textbased` so that mini-SWE-agent parses bash commands from text using regex (triple-backtick fenced blocks with `mswea_bash_command` marker) rather than OpenAI tool calls. This avoids the need for the proxy to return structured `tool_calls` JSON.

Port assignment: `port: 0` (default) lets the OS assign an available port per worker.

### SubprocessRunner (`examples/mini_swe/subprocess_runner.py`)

Manages the `mini-swe-agent` subprocess lifecycle:

```
mini-swe-agent -t "<problem>" -c <config.yaml> -o <output.json> -y --exit-immediately
```

- Applies SIGTERM -> SIGKILL escalation on timeout
- Captures stdout/stderr logs to `{output_dir}/{instance_id}.stdout.log`
- After completion (or cancellation), calls `PatchExtractor` to retrieve the patch

### PatchExtractor (`psrl/utils/common/patch_extractor.py`)

Extracts the generated patch using a fallback chain:

1. **Trajectory JSON** (`{output_dir}/{instance_id}.traj.json`): reads `data["info"]["submission"]`
2. **`git diff HEAD`** on `repo_path` (fallback)
3. **`git diff`** unstaged (last resort)

### MiniSWEEnvironment (`psrl/environments/mini_swe_env.py`)

Registered as `"mini_swe_env"`. Handles Docker container lifecycle (reset/close), temporary directories for per-instance working space, and patch extraction after the agent finishes.

### Reward Function (`examples/mini_swe/reward.py`)

Computes a patch-based reward with bash-tool-use shaping for `mini_swe_agent` data sources. Tool usage detection targets mini-SWE-agent v2 bash patterns (`sed -i`, `cat <<`, `tee`, `patch`, etc.) rather than the deprecated SWE-agent v1 tools.

| Condition | Score |
|-----------|-------|
| Exact patch match | `1.0` |
| Partial patch match | `0.10` -- `0.85` |
| Patch on wrong files | `0.05` |
| No patch, but edited correct file | `0.05` |
| No patch, but ran tests / python verification | `0.03` |
| No patch, but made edits on wrong file | `0.02` |
| No patch, but explored correct file | `0.02` |
| No patch, but explored code (cat/ls) | `0.01` |
| Alignment failed / 0 turns / timeout | `0.0` |
| Long and fruitless (>=10 turns, no edits) | `-0.05` |
| Premature exit (<=2 turns, no tool use) | `-0.1` |

## Docker Image

mini-SWE-agent manages Docker itself. When PSRL runs `mini-swe-agent` as a subprocess, mini-SWE-agent reads the YAML config and starts its own Docker container using `docker run`. PSRL never interacts with Docker directly.

The default image is `python:3.11-slim`. Any image with `bash` works. For the built-in simple test cases, repos are baked into the image (see [Data Preparation](#data-preparation)). For SWE-bench tasks, use the official SWE-bench Docker images.

### Runtime environment

| Setup | How sandbox reaches ModelProxy |
|-------|-------------------------------|
| **Bare metal** (recommended) | Sandbox uses `--network host`; proxy at `127.0.0.1` |
| **Docker-in-Docker** | Outer container must share Docker socket; proxy still at `127.0.0.1` via host networking |

For Docker-in-Docker, the outer `docker run` command needs:

```bash
docker run -it \
  --gpus all \
  --network host \
  --shm-size=32g \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /usr/bin/docker:/usr/bin/docker:ro \
  -v /path/to/data:/data \
  -v /path/to/models:/models \
  --entrypoint /bin/bash \
  <your-psrl-image>
```

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `mini-swe-agent not found` | Not installed or not on PATH | `python -m pip install mini-swe-agent` |
| `docker: Cannot connect to the Docker daemon` | Docker not running or no permission | `sudo systemctl start docker` / add user to `docker` group |
| No patch found | Agent didn't submit before step limit | Increase `agent.step_limit` or `sandbox_config.max_model_calls_per_instance` |
| Proxy timeout after N seconds | Agent stalled; Docker slow to start | Pre-pull image: `docker pull python:3.11-slim`; check `proxy_config.timeout` |
| OOM during rollout | Too many concurrent containers | Reduce `sandbox_config.max_parallel_tasks_per_worker` or add `--memory` limit |
| Alignment failed every episode | Context truncation rewriting history | Remove dynamic history processors; use `max_model_len` instead |
| Residual containers after crash | Cleanup didn't run | See emergency cleanup below |

### Emergency Cleanup

```bash
# Stop all mini-SWE-agent Docker containers from this recipe
docker ps -q --filter "label=psrl.swe_problem_id" | xargs -r docker stop

# Force stop Ray
ray stop --force

# Kill training process
pkill -9 -f main_ppo
```

## Extending

### Custom Docker Image

Replace `python:3.11-slim` with any image that has `bash`:

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*
RUN python -m pip install pytest numpy
```

Then set `sandbox_config.environment.image` in `mini_swe_agent_config.yaml`.

### SWE-bench Tasks

To train on real SWE-bench tasks, set per-instance `sandbox_overrides` to use the official SWE-bench Docker images:

```python
"sandbox_overrides": {
    "use_preexisting_repo": True,
    "preexisting_repo_name": "",  # repo is at /testbed in the image
},
"agent_overrides": {
    "step_limit": 30,
}
```

### Custom Reward Functions

Replace or extend `reward.py`. The function signature is:

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    """Returns a float reward (typically in [-0.1, 1.0])."""
```

### Custom Prompt Templates

Override templates globally in `mini_swe_agent_config.yaml` or per-instance via `extra_info.agent_overrides.system_template` / `instance_template`. The instance template uses Jinja2 with `{{ task }}` (problem statement) and `{{ cwd }}` (working directory, provided by mini-SWE-agent's environment) variables.

The system template must instruct the model to wrap bash commands in triple-backtick fenced blocks with the `mswea_bash_command` marker, since the recipe uses the text-based model class.
