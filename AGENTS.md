# Project Contract

## ALWAYS

- Apply first-principles thinking. Do not assume that I always have a clear understanding of what I want or how to achieve it. Stay cautious and start from the fundamental needs and problem. If the motivation or objective is unclear, pause and discuss it with me. If the objective is clear but the path is not optimal, point that out and suggest a better approach. When you are reading my code, if you identify a better design, you could interrupt the current task to consider refactoring. Discuss the refactoring with me before proceeding.
- When running scripts or inspecting the environment, please activate the conda environment by executing `source /jizhicfs/lhy/env/psrl.sh`. All dependencies and packages are installed within this environment.

## Coding Guidelines

Two reference files live under `.claude/`:

- **`.claude/coding-style.md`** — formatting rules, naming conventions, docstrings, logging, and annotation markers. Ordered by risk (silent-bug rules first).
- **`.claude/codebase-map.md`** — system architecture, directory tree, configuration hierarchy, quick-lookup indices, and import dependency graphs.

Claude must read and apply these guides when writing or modifying code.

## Compact Instructions

When compressing, preserve in priority order:

1. Architecture decisions (NEVER summarize)
2. Modified files and their key changes
3. Current verification status (pass/fail)
4. Open TODOs and rollback notes
5. Tool outputs (can delete, keep pass/fail only)