# SWE Agentic RL

This recipe trains a policy to solve software-engineering tasks with
[mini-SWE-agent](https://github.com/SWE-agent/mini-SWE-agent), Docker sandboxes,
fresh-container grading, and PSRL's asynchronous GRPO/DAPO training path.

The current integration treats mini-SWE-agent as a **black-box agent**. It uses its
normal Python bindings and OpenAI-compatible model client; PSRL does not replace the
model with the legacy `_PSRLModel` queue bridge.

---

## Current Runtime Path

```{mermaid}
sequenceDiagram
    participant ALW as AgentLoopWorker
    participant LOOP as MiniSWEAgentLoopV1
    participant SR as SessionRouter
    participant SMG as SMG + TITO
    participant V as vLLM
    participant A as mini-SWE-agent
    participant D as Docker sandbox
    participant G as Fresh grader container
    participant TQ as TransferQueue

    ALW->>LOOP: run(request)
    LOOP->>SR: create TITO session + routing headers
    LOOP->>A: run_agent(session-scoped API URL)
    A->>D: inspect, edit, and test repository
    A->>SR: OpenAI chat completion
    SR->>SMG: session request
    SMG->>V: route generation
    V-->>A: assistant turn
    Note over A,D: repeat until submit / max turns / timeout
    LOOP->>G: apply patch and execute F2P/P2P tests
    LOOP->>SR: fetch TITO session
    SR-->>LOOP: tokens, masks, logprobs, turn records
    LOOP->>TQ: finalized trajectory + patch + grader result
    LOOP->>SR: delete session
```

Per episode:

1. `MiniSWEEnvironment.reset()` validates dataset metadata and builds the
   per-problem sandbox configuration.
2. `MiniSWEAgentLoopV1` creates one SessionRouter/TITO session containing request,
   prompt, trajectory, validation, and model-version metadata.
3. `examples/mini_swe/runner.py` starts mini-SWE-agent in a worker thread. The agent
   owns its Docker interaction loop and calls the session-scoped OpenAI endpoint.
4. SessionRouter pins later turns to the selected rollout instance. SMG TITO records
   exact model-side training data while requests pass through vLLM.
5. After submission, the runner grades the patch in a fresh container when
   `swe_grader=swebench_fresh_container`.
6. The loop fetches TITO once, builds `prompt_ids`, `response_ids`,
   `response_mask`, rollout log-probabilities, and optional routed-expert tensors,
   then writes the finalized trajectory to TransferQueue.

The environment's command outputs are included in `response_ids` with mask `0`;
assistant-generated tokens use mask `1`. This preserves the full multi-turn context
while training only on policy tokens.

---

## Supported Data Paths

| Path | Dataset | Sandbox | Grading |
|---|---|---|---|
| SWE-smith-py | `SWE-bench/SWE-smith-py` | Per-problem `swebench/swesmith.*` image | Fresh container, generated F2P/P2P test spec |
| SWE-Gym | `SWE-Gym/SWE-Gym` | Per-problem `xingyaoww/sweb.eval.*` image | Fresh container, dataset-provided eval script |
| SWE-bench Verified | `SWE-bench/SWE-bench_Verified` | Per-problem SWE-bench image | Fresh container, usually used for validation |

:::{warning}
The current `MiniSWEEnvironment` requires `swe_grader`, `swe_problem`, and
`swe_problem_image` in every row. `prepare_simple_data.py` does not currently emit
that contract, so the toy launch path is not a valid end-to-end path until its
dataset schema or environment handling is updated.
:::

---

## Main Components

| Component | Responsibility |
|---|---|
| `MiniSWEAgentLoopV1` | Owns SessionRouter/TITO lifecycle and converts the black-box run into a PSRL trajectory |
| `MiniSWEEnvironment` | Parses parquet metadata, applies per-instance overrides, and performs safety-net Docker cleanup |
| `examples/mini_swe/runner.py` | Runs mini-SWE-agent through standard Python bindings and invokes fresh grading |
| `SessionRouter` | Preserves PSRL headers, session affinity, and session close/drain semantics |
| SMG TITO | Captures canonical model-side tokens, log-probabilities, masks, and turn boundaries |
| `MiniSWEAgentData` | Adds patch, grader result, turn count, and resolve-rate metadata |
| `examples/mini_swe/reward.py` | Converts grader output into binary or shaped training rewards |

The older `mini_swe_agent_loop.py` queue-bridge implementation remains in the tree
for compatibility, but current agent YAML files target `MiniSWEAgentLoopV1`.

---

## Reward Semantics

For SWE-smith, SWE-Gym, and Verified, grading uses a clean container created from the
same per-problem image:

1. Apply the submitted patch.
2. Reject policy-violating changes when configured by the grader.
3. Run FAIL_TO_PASS and PASS_TO_PASS tests.
4. Attach the structured grader result to the trajectory.

`compute_score` supports `binary`, `partial_credit`, `test_ratio`, and `shaped`
reward modes. `acc` is always the binary resolve indicator and should be used as the
primary evaluation metric.

```{toctree}
:maxdepth: 1
:hidden:

prepare
training
```
