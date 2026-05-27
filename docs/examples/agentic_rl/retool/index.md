# ReTool: Code Interpreter RL

Train language models to **strategically invoke a Python code interpreter** while solving hard math problems, using reinforcement learning. This recipe is inspired by [ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536) (ByteDance-Seed, 2025).

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│               PSRL GRPO / DAPO Trainer              │
│  (actor, vLLM rollout, reward scoring, TIS)         │
└──────────────────────┬──────────────────────────────┘
                       │  per-episode
          ┌────────────┴────────────┐
          │ MultiTurnAgentLoop.run()│
          │   (tool_env + hermes)   │
          └────────────┬────────────┘
                       │
     ┌─────────────────┼─────────────────────┐
     │                 │                     │
     ▼                 ▼                     ▼
┌──────────┐    ┌────────────────┐   ┌─────────────────────────┐
│   vLLM   │    │ ToolEnvironment│   │ CustomSandboxFusionTool │
│ generate │    │  .step(action) │◄──│   .async_forward(code)  │
└──────────┘    └──────┬─────────┘   └───────────┬─────────────┘
                       │                         │ HTTP POST
                       │                         ▼
                       │              ┌──────────────────────┐
                       │              │ SandboxFusion service│
                       │              │  (Docker Swarm, 8080)│
                       │              └──────────────────────┘
                       │
               ┌───────┴──────────────┐
               │ compute_score        │
               │ (math_dapo +         │
               │  tool-call shaping)  │
               └──────────────────────┘
```

---

## Key Concepts

**SandboxFusion**: A Docker Swarm service that runs code on every cluster node (port 8080). Each rollout worker POSTs Python code to `localhost:8080/run_code` and receives stdout/stderr/return_code.

**Hermes tool format**: The model emits tool calls in Hermes format (`<tool_call>...</tool_call>`), which PSRL's `HermesToolParser` routes to the registered `code_interpreter` tool.

**Reward shaping**: Correct answers get `+1.0`. Incorrect answers get a base of `-1.0` with a small tool-call bonus proportional to `num_turns`, floored at `-0.6`. This encourages the model to use the code interpreter even when heading toward a wrong answer.

---

## Prerequisites

- Docker daemon on every worker node
- SandboxFusion Swarm service deployed (port 8080 on all nodes)

```{seealso}
Full setup instructions including Docker image baking, Swarm deployment, and troubleshooting are in [`examples/retool/README.md`](https://github.com/lhy101/psrl/blob/main/examples/retool/README.md).
```

```{toctree}
:maxdepth: 1
:hidden:

prepare
training
```

