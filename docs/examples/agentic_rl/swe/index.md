# SWE-agent: Software Engineering RL

Train language models to solve **software engineering tasks** using reinforcement learning. This recipe integrates [mini-SWE-agent](https://github.com/SWE-agent/mini-SWE-agent) (v2) with PSRL's trainer, enabling models to learn from interactive coding feedback in Docker-sandboxed environments.

---

## Architecture

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
│(rollout) │   │(queue bridge)│   └────────────────────┘
└──────────┘   └──────┬───────┘
                      │
               ┌──────┴───────┐
               │ vLLM generate│
               └──────┬───────┘
                      │
               ┌──────┴────────────────┐
               │ grade_fresh_container │
               │ (fresh Docker, pytest)│
               └──────┬────────────────┘
                      │
               ┌──────┴───────┐
               │ compute_score│
               └──────────────┘
```

---

## Key Concept: `_PSRLModel`

`_PSRLModel` bridges mini-SWE-agent's synchronous `DefaultAgent` with PSRL's async vLLM rollout engine:

1. `DefaultAgent.run()` executes in a dedicated thread pool
2. When the agent needs a model response, `_PSRLModel.query()` puts messages into a thread-safe queue
3. The async generation loop picks up the request and routes it through PSRL's vLLM
4. The response is returned via another queue back to the blocking agent thread

No HTTP proxy or subprocess is needed, everything runs in-process.

---

## Data Paths

| Path | Dataset | Size | Docker Images | Reward |
|------|---------|------|---------------|--------|
| **Toy** | `simple_cases_*.json` | 40-64 tasks | `python:3.11-slim` (baked) | Patch-text overlap |
| **SWE-smith** | `SWE-bench/SWE-smith-py` | ~50k tasks | `swebench/swesmith.*` (per-problem) | F2P/P2P test execution |
| **SWE-Gym** | `SWE-Gym/SWE-Gym` | 2438 tasks | `xingyaoww/sweb.eval.*` (per-problem) | F2P/P2P test execution |

---

## Grading

For SWE-smith and SWE-Gym paths, after the agent submits a patch:

1. A **fresh Docker container** is started from the same per-problem image
2. The agent's patch is applied
3. FAIL_TO_PASS and PASS_TO_PASS tests are executed
4. The result determines the reward (`+1.0` resolved, `-1.0` failed)

This ensures grading is independent of any side effects from the agent's exploration.

```{seealso}
Full setup instructions, prerequisites, and troubleshooting are in [`examples/mini_swe/README.md`](https://github.com/lhy101/psrl/blob/main/examples/mini_swe/README.md).
```

```{toctree}
:maxdepth: 1
:hidden:

prepare
training
```
