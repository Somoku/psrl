# ReTool Training

Once the [data and SandboxFusion service](prepare) are in place, kick off training
with the smallest reference recipe, FSDP2 on Qwen2.5-7B, 4 nodes × 8 GPU:

```bash
bash examples/retool/fsdp_qwen_7b_dapo.sh          # default staleness=2
bash examples/retool/fsdp_qwen_7b_dapo.sh 3        # override staleness
```

Two Megatron paths (`megatron_qwen_32b_dapo.sh`, `megatron_qwen3_30b_dapo_small.sh`)
follow the same structure, switching paths only changes the cluster layout and
checkpoint backend, not the agentic-RL knobs documented below.

---

## ReTool-specific knobs

On top of the [common agentic-RL knobs](../index) (`multi_turn.enable`,
`multi_turn.max_turns`, `agent.env.name`, `agent.data.name`, `data.return_raw_chat`),
the launch script pins the agent loop to ReTool's tool-using environment and tells
vLLM to parse the model's output as Hermes-style tool calls.

| Knob | Value in `fsdp_qwen_7b_dapo.sh` | Role |
|---|---|---|
| `*.rollout.agent.env.name` | `tool_env` | Selects `ToolEnvironment` (`psrl/environments/tool_env.py`) |
| `*.rollout.agent.data.name` | `tool_agent_data` | Selects `ToolAgentData` (`psrl/workers/agent_loop/agent_data/tool_agent_data.py`) |
| `*.rollout.multi_turn.tool_config_path` | `examples/retool/sandbox_fusion_tool_config.yaml` | Registers `CustomSandboxFusionTool` as `code_interpreter` |
| `*.rollout.multi_turn.format` | `hermes` | Parse model output as Hermes `<tool_call>...</tool_call>` blocks |
| `data.custom_cls.path` / `.name` | `examples/retool/retool.py` / `CustomRLHFDataset` | Custom dataset class that appends the `\boxed{}` answer-format instruction |
| `custom_reward_function.path` / `.name` | `examples/retool/retool.py` / `compute_score` | Math-answer verification + tool-call reward shaping |

---

## Reward function

`compute_score` (in `examples/retool/retool.py`) verifies the model's `\boxed{...}`
answer via veRL's `math_dapo.compute_score` with `strict_box_verify=True`, then
shapes the reward:

| Outcome | Score |
|---|---|
| Correct `\boxed{answer}` | `+1.0` |
| Wrong answer (no tool calls) | `-1.0` |
| Wrong answer (with tool calls) | `-1.0` + `(num_turns - 2) / 2 * 0.1`, floored at `-0.6` |
| Malformed / no boxed answer | `0.0` |

The tool-call bonus encourages the model to actually invoke the code interpreter when
it would otherwise produce a wrong answer.

---

```{seealso}
Full launch-script parameter reference, tool registry schema, and troubleshooting:
[`examples/retool/README.md`](https://github.com/lhy101/psrl/blob/main/examples/retool/README.md).
```
