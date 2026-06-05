# Examples

Production-ready training recipes demonstrating PSRL's capabilities across different RL paradigms.

---

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`verified;1.5em` RLVR
:link: rlvr/index
:link-type: doc

Reinforcement Learning from Verifiable Rewards. Train models on math and reasoning tasks with rule-based reward signals.

**Algorithms**: PPO, GRPO, DAPO
:::

:::{grid-item-card} {octicon}`tools;1.5em` Agentic RL
:link: agentic_rl/index
:link-type: doc

Multi-turn agent training with tool use. Models learn to invoke code interpreters and interact with real software environments.

**Recipes**: ReTool, SWE-agent
:::

:::{grid-item-card} {octicon}`law;1.5em` Generative Reward Model
:link: generative_reward_model/index
:link-type: doc

LLM-derived reward signals for open-ended tasks: LLM-as-a-Judge scoring and On-Policy Distillation. *(Under development)*
:::

::::

---

## Recipe Overview

| Recipe | Task Domain | Reward Type | Training Backend | Status |
|--------|------------|-------------|-----------------|--------|
| [DAPO](rlvr/dapo) | Math / Reasoning | Verifiable (boxed answer) | FSDP / Megatron | Ready |
| [PPO](rlvr/ppo) | General | Verifiable | FSDP / Megatron | TBD |
| [GRPO](rlvr/grpo) | General | Verifiable | FSDP / Megatron | TBD |
| [ReTool](agentic_rl/retool/index) | Math + Code Interpreter | Verifiable + tool-call shaping | FSDP / Megatron | Ready |
| [SWE-agent](agentic_rl/swe/index) | Software Engineering | Test execution (F2P/P2P) | FSDP / Megatron | Ready |
| [LLM-as-a-Judge](generative_reward_model/llm_as_a_judge) | General / Open-ended | Judge LLM score |, | TBD |
| [On-Policy Distillation](generative_reward_model/on_policy_distillation) | General / Open-ended | Teacher LLM token-level supervision |, | TBD |

```{toctree}
:maxdepth: 2
:hidden:

rlvr/index
agentic_rl/index
generative_reward_model/index
```
