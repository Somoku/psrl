# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Principles

Principle 1: Apply first-principles thinking. Do not assume that I always have a clear understanding of what I want or how to achieve it. Stay cautious and start from the fundamental needs and problem. If the motivation or objective is unclear, pause and discuss it with me. If the objective is clear but the path is not optimal, point that out and suggest a better approach.

Principle 2: When running scripts or inspecting the environment, please activate the conda environment by executing `source /jizhicfs/lhy/env/psrl_agent.sh`. All dependencies and packages are installed within this environment. For basic bash commands such as `grep`, `ls`, `find`, and `read`, you could execute them directly without asking for my permission.

Principle 3: At all times, including when you are reading my code, if you identify a better design, you could interrupt the current task to consider refactoring. Discuss the refactoring with me before proceeding.

Principle 4: At all times, add more comments and use assertions where appropriate. Ensure that any newly added code remains consistent with the existing style, including comments, logging formats, and the way exceptions and assertions are written.

## Project Overview

PSRL is a post-training framework for LLMs supporting asynchronous training paradigms via staleness control, streaming rollout, and a parameter server architecture. It is built on top of [verl](https://github.com/volcengine/verl) (ByteDance's RLHF framework) and uses [Ray](https://ray.io/) for distributed orchestration.

The main entry point is `psrl/trainer/main_ppo.py`, driven by [Hydra](https://hydra.cc/). Pass config overrides as `key=value` on the command line; configs live in `psrl/trainer/config/`.

## Configuration System

Hydra config hierarchy under `psrl/trainer/config/`:
- `ppo_trainer.yaml` — top-level config that composes sub-configs
- `psrl/` — PSRL-specific config (logging, async params)
- `actor/`, `rollout/`, `model/`, `data/`, `critic/`, `reward_model/` — component-level configs

Key top-level config groups: `train_actor_rollout_ref` (trainer-side), `gen_actor_rollout_ref` (rollout-side), `algorithm`, `psrl`.

## Third-Party Vendored Dependencies

- `third_party/vllm/` — vLLM with PSRL-specific patches (applied via `patch/vllm/`)
- `third_party/verl/` — verl RLHF framework (base classes, data utils, PPO core algos)
- `third_party/nixl/` — NIXL GPU-direct communication
- `third_party/torch_memory_saver/` — GPU memory management utilities
