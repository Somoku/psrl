# Project Contract

## ALWAYS

- Apply first-principles thinking. Do not assume that I always have a clear understanding of what I want or how to achieve it. Stay cautious and start from the fundamental needs and problem. If the motivation or objective is unclear, pause and discuss it with me. If the objective is clear but the path is not optimal, point that out and suggest a better approach. When you are reading my code, if you identify a better design, you could interrupt the current task to consider refactoring. Discuss the refactoring with me before proceeding.
- When running scripts or inspecting the environment, please activate the conda environment by executing `source /jizhicfs/lhy/env/psrl.sh`. All dependencies and packages are installed within this environment.

## Coding Conventions

Detailed coding conventions are maintained as separate reference files under `.claude/rules/`.
Claude must read and apply these guides when writing or modifying code.

## Compact Instructions

When compressing, preserve in priority order:

1. Architecture decisions (NEVER summarize)
2. Modified files and their key changes
3. Current verification status (pass/fail)
4. Open TODOs and rollback notes
5. Tool outputs (can delete, keep pass/fail only)

## Project Overview

PSRL is a post-training framework for LLMs supporting asynchronous training paradigms via staleness control, streaming rollout, and a parameter server architecture.

The main entry point is `psrl/trainer/main_ppo.py`, driven by [Hydra](https://hydra.cc/). Pass config overrides as `key=value` on the command line; configs live in `psrl/trainer/config/`.

## Third-Party Vendored Dependencies

- `third_party/vllm/` — vLLM with PSRL-specific patches (applied via `patch/vllm/`)
- `third_party/verl/` — verl RLHF framework (base classes, data utils, PPO core algos)
- `third_party/nixl/` — NIXL GPU-direct communication
- `third_party/torch_memory_saver/` — GPU memory management utilities
