# Agentic RL

Train language models to interact with external tools and environments across multiple turns, learning strategies for when and how to invoke tools to solve complex tasks.

---

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`terminal;1.5em` ReTool: Math + Code Interpreter
:link: retool/index
:link-type: doc

Train models to strategically invoke a Python code interpreter while solving hard math problems. Inspired by ByteDance's ReTool paper.
:::

:::{grid-item-card} {octicon}`code-square;1.5em` SWE: Software Engineering
:link: swe/index
:link-type: doc

Train models to solve real-world software engineering tasks via interactive bash commands in Docker-sandboxed environments.
:::

::::

---

## What Makes Agentic RL Different?

Unlike single-turn RLVR, agentic RL involves:

- **Multi-turn interaction**: The model generates multiple responses, interleaved with environment observations (tool outputs, command results).
- **Tool-use learning**: The model must learn *when* to call a tool, *what* arguments to pass, and *how* to interpret the result.
- **Long-horizon credit assignment**: Rewards are sparse (end-of-episode), but the causal chain spans dozens of turns.
- **Sandboxed execution**: Each episode runs in an isolated Docker container, ensuring reproducibility and safety.

PSRL handles these challenges through its `AgentLoop` abstraction, which manages the
multi-turn conversation flow while integrating seamlessly with the asynchronous
training pipeline. The design is **inspired by veRL's agent loop**: see
[veRL Agentic RL Training](https://verl.readthedocs.io/en/latest/start/agentic_rl.html)
for the architectural blueprint we built on.

---

## Common Infrastructure

Every agentic recipe in PSRL is built from the same three pieces, kept fully generic
so new tasks can plug in without touching the loop itself. The full developer guide
lives at [`psrl/environments/README.md`](https://github.com/lhy101/psrl/blob/main/psrl/environments/README.md).
In short:

| Component | Role |
|-----------|------|
| `Environment[ObsType, ActType]` | The "world" the agent interacts with: implements `reset(task) → (obs, info)` and `step(action) → (obs, reward, done, info)`. `ObsType` / `ActType` can be any Python type (dict, string, custom dataclass). |
| `AgentData[ObsType, ActType]` | Adapter between **environment space** (obs/actions) and **model space** (token IDs and log-probs in `DataProto`). Also incrementally builds the `Trajectory` of `Step`s used for training. |
| `multi_turn_agent_loop` | Generic driver that alternates *env → AgentData → model → AgentData → env* until the episode terminates. It is task-agnostic and is **not modified** when adding new environments. |

To add a new agentic task you only implement and register a new `Environment` +
`AgentData` pair and select them via
`rollout.agent.env.name` / `rollout.agent.data.name` (globally) or per-request through
`DataProto.non_tensor_batch["env_class"]` / `["data_class"]`. The loop, the rollout
engine, and the rest of the training stack stay the same.

The two recipes below, [ReTool](retool/index) (Python code interpreter) and
[SWE-agent](swe/index) (Docker-sandboxed bash), are concrete instantiations of this
pattern.

---

## Common Launch-Script Knobs

Beyond the standard PSRL trainer flags (see [Configuration](../../tutorial/configuration)),
every agentic launch script sets the same handful of `rollout` flags to switch from
single-turn to multi-turn mode and bind the agent loop to a concrete environment.
Individual recipes only override the **values** (`env.name` / `data.name`, and any
recipe-specific extras such as a tool-config or agent-config YAML).

| Knob | Role |
|---|---|
| `*.rollout.multi_turn.enable=True` | Switch the rollout from single-turn to multi-turn mode |
| `*.rollout.multi_turn.max_turns=<N>` | Hard cap on turns per episode |
| `*.rollout.agent.env.name=<name>` | Picks the registered `Environment` (e.g. `tool_env`, `mini_swe_env`) |
| `*.rollout.agent.data.name=<name>` | Picks the registered `AgentData` (e.g. `tool_agent_data`, `mini_swe_agent_data`) |
| `data.return_raw_chat=True` | Keep raw chat messages so the agent loop can re-format them across turns |

The `*` placeholder stands for **both** `gen_actor_rollout_ref` (training-time
rollouts) **and** `train_actor_rollout_ref` (validation rollouts that run on the
training nodes), set these flags on both subtrees so the same agent behaviour holds
during evaluation.

Each recipe's `training.md` only documents the **recipe-specific** knobs on top of
this baseline.

```{toctree}
:maxdepth: 2
:hidden:

retool/index
swe/index
```
