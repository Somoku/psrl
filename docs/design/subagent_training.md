# Subagent-aware TITO training

Status: proposed

Scope: PSRL SessionRouter, TITO training data, agent loops, and the SMG gateway.

This page describes how one TITO session can hold a main agent and one or more
subagents, how each trajectory is marked trainable or context-only, and how a
shared prefix is trained exactly once.

## 1. Problem

A harness such as Codex can run a main agent and several subagents in one
conversation. The subagent calls currently share the session. TITO already
stores them as separate trajectories, but PSRL trains every trajectory in the
session with the same rule:

- Every assistant token on every trajectory is mask 1.
- Shared prefix turns appear in each trajectory's `turn_records`.
- The reward is broadcast to every output.
- There is no way to say "train the main agent, keep the subagent context, and
  do not train the subagent".

The goal is to make trainability a per-trajectory property while keeping one
session, one prefix tree, and one KV-cache lineage.

## 2. Current behavior

### 2.1 TITO snapshot

`third_party/smg/crates/tito/src/store.rs` already exposes:

```rust
pub struct Trajectory {
    pub trajectory_id: u64,
    pub accumulated_token_ids: Vec<u32>,
    pub turn_records: Vec<TurnRecord>,
}

pub struct TitoNodeMeta {
    pub id: u64,
    pub hash: String,
    pub stored: bool,
    pub parent: Option<u64>,
    pub finish_reason: String,
    pub truncated: bool,
    pub num_tokens: usize,
    pub trajectory_ids: Vec<u64>,
}

pub struct TitoLeafMeta {
    pub trajectory_id: u64,
    pub node_id: u64,
    pub parent: Option<u64>,
    pub path_node_ids: Vec<u64>,
}
```

`collect_records_for_leaf()` walks the leaf's parent chain and clones each
shared node's `TurnRecord` into every trajectory that passes through it.
`collect_tree_meta()` lists the trajectories that share a node in
`TitoNodeMeta::trajectory_ids`. There is no owner field, no capture mode, and no
role metadata.

### 2.2 PSRL extraction

`psrl/utils/tito/training_data.py::build_training_data()` treats one
trajectory as a linear stream:

- The first record's prompt becomes `prompt_ids`.
- Every later assistant output becomes `response_ids` with mask 1.
- Environment and tool tokens between turns become mask 0.
- `output_logprobs` is the only source of the generated token ids.

`psrl/workers/agent_loop/loops/session_agent_loop.py::get_training_data()`
builds one item per trajectory. `run()` returns a list of `TokenOutput` when the
session has more than one. `compute_advantage_for_multi_trajectories()` picks
the highest trajectory index in the session and broadcasts its advantage to all
rows.

### 2.3 Subagent detection

`psrl/workers/gen/session_router.py::is_subagent_request()` reads
`x-codex-turn-metadata` and checks `thread_source == "subagent"`. The router
uses that only to charge the turn budget separately. The header is forwarded to
SMG already because the router filters only hop-by-hop headers. SMG does not
store the role today.

### 2.4 Gaps

| Gap | Location |
|---|---|
| No role or agent identity on a trajectory | `Trajectory`, `TitoLeafMeta` |
| No owner on a shared node | `TitoNodeMeta`, `PrefixEntry` |
| No per-record capture mode | `TurnRecord` |
| Token-only records cannot be reconstructed without logprobs | `build_training_data()` |
| Shared prefix is trained once per trajectory | `build_training_data()` |
| Advantage picks the last trajectory, not the main one | `compute_advantage_for_multi_trajectories()` |
| No per-agent training policy | `psrl.agentic_rl` |

## 3. Goals and non-goals

### Goals

- Keep one TITO session for the main agent and its subagents.
- Mark each trajectory trainable or context-only.
- Keep the same prefix tree and KV lineage across agents.
- Train a shared prefix exactly once.
- Let an untrained subagent keep only the token ids needed for its next turn.
- Keep the session-level reward and the existing GRPO path.
- Preserve the current behavior when no role metadata exists.

### Non-goals

- Per-agent reward models or per-agent credit assignment beyond the session
  reward. This is a later extension.
- Changing how the harness spawns subagents. That remains a harness concern.
- Multi-session subagent routing. The design keeps one session.

## 4. Invariants

1. A session may contain many trajectories and many leaves. A trajectory is one
   root-to-leaf path.
2. Each stored node has exactly one `owner_trajectory_id`. The first successful
   store assigns it.
3. A node is trainable only when its owner is trainable and its capture is
   `Full`. A node without a training payload is context only.
4. A trainable trajectory trains its owned nodes. A node owned by another
   trajectory is context with mask 0.
5. A token is masked 1 at most once in a session. The owner's sample trains it
   and every other path masks it.
6. An untrained trajectory never becomes a training row. Its token ids remain in
   the session for inference.
7. The default mode trains only the main trajectory.
8. A session without role metadata keeps the current linear behavior.

## 5. Design

### 5.1 Data model in SMG

#### TurnRecord

```rust
pub enum TurnCapture {
    Full,
    Tokens,
}

pub struct TurnRecord {
    pub prompt_token_count: usize,
    pub output_token_ids: Option<Vec<u32>>,
    pub output_logprobs: Option<Vec<(f32, u32)>>,
    pub sampler_capture_width: u32,
    pub sampler_distribution: Option<SamplerDistribution>,
    pub routed_experts: Option<TurnRoutedExperts>,
    pub finish_reason: String,
    pub mismatch_report: Vec<MismatchEntry>,
    pub weight_version: Option<String>,

    pub owner_trajectory_id: u64,
    pub capture: TurnCapture,
}
```

`output_token_ids` is populated on every capture path. It is the authoritative
source of generated ids for a token-only record. With `Full` it duplicates the
ids already present in `output_logprobs`, which keeps the extraction uniform and
lets the boundary match work without logprobs. The duplicate is 4 bytes per
token. A later optimization can omit it for `Full` records if the wire size
matters.

`capture == Tokens` sets:

- `output_logprobs = None`
- `sampler_capture_width = 0`
- `sampler_distribution = None`
- `routed_experts = None`
- `output_token_ids = Some(generated_ids)`
- `finish_reason` and `prompt_token_count` retained

The turn also disables the sampler head request. SMG must not ask vLLM for
`sampler_top_logprobs` on a token-only turn, otherwise the capture is skipped
but the engine still pays for it.

#### PrefixEntry

```rust
pub struct PrefixEntry {
    pub token_ids: Arc<Vec<u32>>,
    pub reusable_prefix_token_ids: Option<Arc<Vec<u32>>>,
    pub parent_hash: Option<PrefixHash>,
    pub turn_record: TurnRecord,
    pub seq: u64,
    pub ends_tool_call: bool,

    pub owner_trajectory_id: u64,
    pub capture: TurnCapture,
}
```

`store_token_sequences()` sets `owner_trajectory_id = trajectory_id` when it
inserts a new node. If the same node is stored again:

- The first owner stays when its capture is `Full`.
- A `Tokens` owner may be upgraded to `Full` by a later trainable store, and the
  new owner and payload replace the old ones.
- A `Full` owner is never taken over by a `Tokens` store.

This keeps a trainable node trainable and prevents a later subagent call from
erasing its payload.

#### Trajectory metadata

```rust
pub struct TrajectoryMeta {
    pub thread_source: String,          // "main" or "subagent"
    pub agent_id: Option<String>,
    pub parent_trajectory_id: Option<u64>,
    pub capture: TurnCapture,
}
```

`SessionState` holds `trajectory_meta: HashMap<u64, TrajectoryMeta>`. The
preparation stage writes it when it resolves the trajectory id. The snapshot
copies it into each `Trajectory` and `TitoLeafMeta`:

```rust
pub struct Trajectory {
    pub trajectory_id: u64,
    pub accumulated_token_ids: Vec<u32>,
    pub turn_records: Vec<TurnRecord>,
    pub thread_source: String,
    pub agent_id: Option<String>,
    pub parent_trajectory_id: Option<u64>,
    pub capture: TurnCapture,
}
```

`parent_trajectory_id` comes from the normalized request header when the harness
provides one. Otherwise the snapshot derives it from the tree. For a leaf, walk
its path and find the nearest ancestor node whose `owner_trajectory_id` differs
from this leaf's id. That owner is the parent trajectory.

#### TitoNodeMeta

```rust
pub struct TitoNodeMeta {
    pub id: u64,
    pub hash: String,
    pub stored: bool,
    pub parent: Option<u64>,
    pub finish_reason: String,
    pub truncated: bool,
    pub num_tokens: usize,
    pub trajectory_ids: Vec<u64>,

    pub owner_trajectory_id: u64,
    pub capture: TurnCapture,
}
```

`trajectory_ids` keeps the sharing view. `owner_trajectory_id` identifies who
generated the node. A tool that joins snapshots by `hash` can use both.

#### Snapshot example

```json
{
  "session_id": "...",
  "max_trim_tokens": 1,
  "trajectories": [
    {
      "trajectory_id": 0,
      "thread_source": "main",
      "agent_id": "main",
      "parent_trajectory_id": null,
      "capture": "full",
      "accumulated_token_ids": [1, 2, 3, 4, 5],
      "records": [
        {
          "prompt_token_count": 2,
          "output_token_ids": [3],
          "output_logprobs": [[-0.1, 3]],
          "owner_trajectory_id": 0,
          "capture": "full",
          "finish_reason": "tool_calls"
        }
      ]
    },
    {
      "trajectory_id": 1,
      "thread_source": "subagent",
      "agent_id": "codex-thread-7",
      "parent_trajectory_id": 0,
      "capture": "tokens",
      "accumulated_token_ids": [1, 2, 3, 6, 7],
      "records": [
        {
          "prompt_token_count": 3,
          "output_token_ids": [6],
          "output_logprobs": null,
          "owner_trajectory_id": 1,
          "capture": "tokens",
          "finish_reason": "stop"
        }
      ]
    }
  ],
  "tree": {
    "nodes": [
      {"id": 0, "parent": null, "num_tokens": 3, "trajectory_ids": [0, 1],
       "owner_trajectory_id": 0, "capture": "full"},
      {"id": 1, "parent": 0, "num_tokens": 2, "trajectory_ids": [1],
       "owner_trajectory_id": 1, "capture": "tokens"}
    ],
    "leaves": [
      {"trajectory_id": 0, "node_id": 0, "path_node_ids": [0]},
      {"trajectory_id": 1, "node_id": 1, "path_node_ids": [0, 1]}
    ]
  }
}
```

In this example the subagent record starts at `prompt_token_count = 3`, so the
shared prefix node 0 is inside its prompt. The main trajectory owns node 0 and
trains it. The subagent owns node 1 but it is token-only, so it is context only.

### 5.2 Request headers

Four turn-level headers are introduced. The session policy is not a header. It
travels in the `POST /sessions` JSON body. None of the headers is read while
training data is built. They matter during rollout, because SMG persists their
values into the TITO snapshot and the extractor reads that snapshot.

The session body is:

```json
{
  "subagent_training": {
    "mode": "main_only",
    "train_agents": [],
    "capture": "auto",
    "shared_prefix": "owner",
    "reward": "session",
    "loss_weight": {"main": 1.0, "subagent": 1.0}
  }
}
```

The SessionRouter parses it, stores it in `SessionState`, and does not forward it
to SMG. An empty body uses the configured default. A body is a better fit than a
header because the policy is structured and session scoped.

| Header | From | To | Scope | Meaning |
|---|---|---|---|---|
| `x-psrl-agent-role` | SessionRouter | SMG | turn | `main`, `subagent`, `auxiliary`, or `compaction` |
| `x-psrl-agent-id` | SessionRouter | SMG | turn | stable agent identity |
| `x-psrl-parent-agent-id` | SessionRouter | SMG | turn | parent agent identity, empty for main |
| `x-psrl-tito-capture` | SessionRouter | SMG | turn | `full` or `tokens` |

The role domain has four values. `auxiliary` covers harness calls that are not
part of the agent conversation, such as title generation and side replies.
`compaction` covers a main-context rewrite. It is not a subagent.

#### Why the identity headers exist

Each harness already marks its own calls in its own way. Codex uses
`x-codex-turn-metadata.thread_source`. Claude Code uses
`x-claude-code-request-class`, `x-claude-code-agent-id`, and
`x-claude-code-parent-agent-id`. A third-party harness may have no marker at
all.

`x-psrl-agent-role`, `x-psrl-agent-id`, and `x-psrl-parent-agent-id` are the
normalization layer. The SessionRouter is the only component that knows the
harness dialect. SMG and the extractor see one vocabulary. This keeps SMG from
depending on Codex or Claude Code, and it gives one place to add a new harness.

The headers are turn scoped because the role is a property of the call, not of
the session. A session can mix main, subagent, auxiliary, and compaction calls.
The identity is turn scoped for the same reason.

#### Why the capture header exists

`x-psrl-tito-capture` is the storage decision for one turn. It tells SMG which
part of the turn to keep:

- `full`: keep `output_logprobs`, `sampler_distribution`, and
  `routed_experts`, in addition to the token ids. The node can train.
- `tokens`: keep the token ids, prompt boundary, output ids, and finish reason.
  Drop the training payload. The node can still be used as context for the next
  turn, but it can never train.

The SessionRouter computes it from the session policy and the role. SMG does not
need to know the training policy. It only needs to know what to store.

The header is part of the first landing. The first version implements both
`full` and `tokens`. A trainable agent gets `full`, and untrained agents and
auxiliary calls get `tokens`. A correctness run may force `full` for everyone
before `tokens` is enabled in production.

#### Header lifecycle

1. The agent loop sends the `subagent_training` object once, in the
   `POST /sessions` body. The harness CLI does not send it.
2. The harness CLI sends its own native per-call metadata. The SessionRouter
   reads it and normalizes it into `x-psrl-agent-role`, `x-psrl-agent-id`, and
   `x-psrl-parent-agent-id`.
3. The SessionRouter computes `x-psrl-tito-capture` from the session policy and
   the role, and sends it to SMG with the turn.
4. SMG stores the role, agent id, parent, owner, and capture into the TITO
   session. The extractor reads only the snapshot.

The harness must carry its native marker on every model call. It does not need
to send the `x-psrl-*` names itself. If it has no native marker, the router
cannot distinguish its subagents, and the strict default attributes every call
to the main agent.

The SessionRouter overwrites any client-supplied `x-psrl-agent-role`,
`x-psrl-agent-id`, `x-psrl-parent-agent-id`, and `x-psrl-tito-capture` with its
own normalized values. Only the native harness markers are trusted from the
client. This keeps a harness from declaring an auxiliary call trainable.

#### Native harness markers

| Harness | Native input | Notes |
|---|---|---|
| Codex | `x-codex-turn-metadata` | `thread_source` is `user` or `subagent`. A stable `thread_id` and parent id are used when present. |
| Claude Code | `x-claude-code-request-class`, `x-claude-code-agent-id`, `x-claude-code-parent-agent-id`, `x-claude-code-agent-type` | The request class is `main`, `subagent`, `workflow`, `compaction`, `auxiliary`, or `side_reply`. Agent id and parent id are emitted for non-main agents. |
| Generic | `x-psrl-agent-role`, `x-psrl-agent-id`, `x-psrl-parent-agent-id` | A harness without native metadata may send the normalized headers directly. |

The Claude Code runtime is pinned to 2.1.295 or newer. That version emits
`x-claude-code-request-class`, `x-claude-code-agent-type`,
`x-claude-code-agent-id`, and `x-claude-code-parent-agent-id`.

Claude Code gates `x-claude-code-request-class` and
`x-claude-code-agent-type` behind `CLAUDE_CODE_GATEWAY_HINT_HEADERS` or a
first-party base URL. A custom session router is not first-party, so the Claude
Code adapter must set `CLAUDE_CODE_GATEWAY_HINT_HEADERS=1` in the harness
environment.

Normalization rules:

- Codex `thread_source == "subagent"` maps to `subagent`. Any other value maps
  to `main`.
- Claude Code `request-class == "subagent"` or `"workflow"` maps to `subagent`.
  `"main"` maps to `main`. `"compaction"` maps to `compaction`.
  `"auxiliary"` and `"side_reply"` map to `auxiliary`.
- `agent_id` uses the native agent id or thread id when present. Otherwise the
  router uses `main` for the main thread and `subagent:<session turn index>` for
  a subagent.
- `parent_agent_id` uses the native parent id when present. Otherwise it uses
  the session's first main agent id.
- `x-psrl-tito-capture` comes from the session policy and the role. A trainable
  agent gets `full`. An untrained agent and every `auxiliary` call get `tokens`.
  A `compaction` call follows the main agent's trainability.
- A request without any role metadata is charged to the main agent, which is the
  stricter default.

SMG reads the headers in
`model_gateway/src/routers/grpc/regular/stages/chat/preparation.rs::try_tito()`.
The values live on `TitoRequestContext` and reach the response stage.

### 5.3 Capture policy

The SessionRouter owns the policy. It stores the session mode and the resolved
trainable agent ids in `SessionState`.

```text
role        = classify_request(request)   # main, subagent, auxiliary, compaction
agent_id    = normalized agent identity
trainable   = policy.is_trainable(role, agent_id)
capture     = "full" if trainable else "tokens"
```

Policy semantics:

- `off`: no trajectory is trained. This is an explicit diagnostic mode. A
  session with no trainable row is a setup error.
- `main_only`: the main agent is trainable. All subagents are `tokens`.
- `all`: the main agent and every subagent are `full`.
- `selected`: the main agent plus the configured agent ids or patterns are
  `full`. Everything else is `tokens`.
- `auxiliary` and `side_reply` calls are always `tokens` and never produce a
  training row.
- `compaction` calls follow the main agent's trainability. They are a context
  rewrite of the main path, not a separate agent.

The policy can be overridden per dataset row through
`extra_info.subagent_training`. The agent loop reads it and sends it in the
`POST /sessions` body when it creates the session.

The harness setting `harness.subagents_enabled` is separate. It controls whether
the CLI may spawn subagents. The training policy controls whether their
trajectories train.

### 5.4 Owner rule

A node's owner is the trajectory whose model call generated the node. The store
already receives the resolved `trajectory_id`, so it can write the owner at
insert time.

- New node: owner is the storing trajectory.
- Existing node with a `Full` owner: keep the owner and payload.
- Existing node with a `Tokens` owner and a new `Full` store: replace owner and
  payload with the trainable one.
- Existing node with a `Tokens` owner and a new `Tokens` store: keep the first
  owner.

A shared node's `trajectory_ids` list still contains every path that passes
through it. The owner is the one that trains it.

### 5.5 PSRL extraction

#### Selection

```python
def select_trainable_trajectories(session_data, policy) -> list[dict]:
    """Return the trajectories that may produce a training row."""
```

The function resolves the session's policy, reads each trajectory's
`thread_source`, `agent_id`, `parent_trajectory_id`, and `capture`, and returns
the trainable ones. The main trajectory is marked `session_primary=True`. If no
main trajectory exists, raise. If a trainable trajectory has `capture=tokens`,
raise when `require_primary_capture` is on.

Untrained trajectories stay in `session_snapshot` for the dump and for analysis.
They do not become `TokenOutput` rows.

#### Owner-aware builder

`build_training_data()` gains:

```python
def build_training_data(
    accumulated_token_ids: list[int],
    records: list[dict],
    *,
    trainable_trajectory_id: int,
    shared_prefix: str = "owner",
    ...
) -> dict:
```

For each record:

```text
env/user tokens between the previous cursor and this prompt end:
    response_ids += env_ids
    response_mask += 0
    logprobs += 0.0
    sampler_distribution += empty
    routed_experts += zero rows

output tokens:
    owner = record.get("owner_trajectory_id", trainable_trajectory_id)
    capture = record.get("capture", "full")
    train_here = (
        owner == trainable_trajectory_id
        and capture == "full"
        and (shared_prefix == "owner" or owner == trainable_trajectory_id)
    )
    response_ids += record output ids
    response_mask += 1 if train_here else 0
    logprobs += record logprobs if train_here else zeros
    sampler_distribution += record distribution if train_here else empty
    routed_experts += record payload if train_here else zero rows
```

`output ids` come from `record.output_token_ids` when present. The older
`output_logprobs[:, 1]` path remains as a fallback for snapshots without the new
field.

`shared_prefix="owner"` is the default and the only policy used by this design.
`shared_prefix="context"` exists only as a diagnostic fallback that masks every
shared node. It is not recommended.

`routed_experts` needs one change. `_assemble_routed_experts()` currently
requires a segment on every record once any record has one. It becomes:

- A `Full` record contributes its segment.
- A `Tokens` record contributes zero rows with the shape of the first `Full`
  segment.
- If no record has a segment, the whole tensor is `None`.

The trainer ignores the zero rows because their mask is 0. The tensor shape must
stay stable, so the first `Full` segment defines `(num_layers, top_k)` and
`dtype`.

#### TokenOutput metadata

Each trainable output carries:

```python
extra_fields = {
    "trajectory_id": ...,
    "agent_id": ...,
    "thread_source": ...,
    "parent_trajectory_id": ...,
    "trainable": True,
    "capture": "full",
    "session_primary": bool,
    "loss_weight": float,
}
```

#### Advantage

`compute_advantage_for_multi_trajectories()` prefers the row with
`session_primary=True` for the session advantage. If the field is absent, it
keeps the current behavior and uses the last row.

`session_primary` must travel in the TransferQueue non-tensor fields. The
trainer's field selection in `ray_trainer.py` gains it next to `uid` and
`parent_id`.

#### Loss and reward

- `reward=session` is the default. The harness computes one session reward and
  broadcasts it to every trainable row. A trainable subagent receives the same
  reward as the main agent.
- `loss_agg_mode=session-mean-token-mean` is recommended. `uid` identifies the
  session and `T_s` counts only mask 1 tokens. Shared prefix tokens appear in
  mask 1 only in the owner row, so they are counted once.
- `group_baseline_valid_only` excludes zero-mask rows from the group baseline.
- `n_trajectory` is the number of trainable rows, not the number of leaves.

Per-agent reward is a later extension. It would call the reward function once
per trainable output and pass `agent_id`. It is out of scope for this version.

### 5.6 Configuration

```yaml
psrl:
  agentic_rl:
    subagent_training:
      # off | main_only | all | selected
      mode: main_only
      # Used only by selected. Exact agent ids or simple prefix patterns.
      train_agents: []
      # auto resolves to full for trainable agents and tokens for the rest.
      capture: auto
      # owner is the only supported value for training.
      shared_prefix: owner
      # session | per_agent. per_agent is reserved.
      reward: session
      # Applied to the session-weighted loss. Both default to 1.0.
      loss_weight:
        main: 1.0
        subagent: 1.0
      # Fail when a trainable trajectory has no training payload.
      require_primary_capture: true
```

A dataset row may override the mode:

```json
{"extra_info": {"subagent_training": {"mode": "all"}}}
```

The agent loop sends the resolved policy in the `POST /sessions` body at session
creation. The same resolved policy is used at extraction time. If the two
disagree, `require_primary_capture` fails the sample instead of silently
training a context token.

### 5.7 Harness adaptation

The harness changes are additive. The session URL is already the only endpoint
the CLI talks to. The extra work is to enable subagents and to expose their
native role metadata.

#### Codex

- Enable subagent generation. `CodexHarness._model_catalog()` currently sets
  `multi_agent_version = None`, and `HarnessConfig.subagents_enabled` is not
  wired into the Codex adapter. Set the catalog's multi-agent fields to the
  values the installed CLI supports, and map `subagents_enabled` to them.
- Keep the provider on the session URL. Codex applies one `model_provider` to
  the main agent and its subagents, so no per-agent URL override is needed. A
  harness that lets a subagent select another provider must force the session
  provider.
- Keep `x-codex-turn-metadata` on every call. `thread_source` is the role
  marker. A stable `thread_id` and parent id improve agent selection, and the
  trajectory id plus the TITO tree are the fallback.
- The turn budget already counts subagent calls separately. No change.

#### Claude Code

- The harness runtime pins Claude Code 2.1.295 or newer. That version emits
  `x-claude-code-request-class`, `x-claude-code-agent-type`,
  `x-claude-code-agent-id`, and `x-claude-code-parent-agent-id`.
- Set `CLAUDE_CODE_GATEWAY_HINT_HEADERS=1` in the harness environment. The
  Claude Code adapter already writes `settings.json` under the sandbox home, so
  this is one entry in that file. Without it, the request-class and agent-type
  headers are gated off on a custom base URL.
- `subagents_enabled` already controls the `Agent` tool. With the flag on, the
  CLI emits `x-claude-code-request-class = subagent` or `workflow`, plus
  `x-claude-code-agent-id`, `x-claude-code-parent-agent-id`, and
  `x-claude-code-agent-type`.
- Keep the Messages session URL for the main agent and its subagents.
- `x-claude-code-request-class = auxiliary` or `side_reply` marks calls outside
  the agent conversation. They are captured as `tokens` and never trained.
- `x-claude-code-request-class = compaction` is a main-context rewrite. It
  follows the main agent's trainability and is part of the main path.
- The CLI's `isSidechain` and `parent_tool_use_id` are transcript-level markers.
  The router does not need them.

#### Generic harnesses

- Send `x-psrl-agent-role`, `x-psrl-agent-id`, and `x-psrl-parent-agent-id` on
  every model call, or a native marker the SessionRouter knows.
- The SessionRouter sets `x-psrl-tito-capture` from the policy. The harness does
  not compute it.
- Use the same session URL for every agent. A subagent that talks to another
  endpoint is not captured.
- A harness that cannot set per-call metadata keeps the strict default. Every
  call is attributed to the main agent, so only `main_only` is meaningful.

## 6. Behavior matrix

| Mode | Main capture | Subagent capture | Auxiliary capture | Training rows |
|---|---|---|---|---|
| `off` | tokens | tokens | tokens | none, session is rejected |
| `main_only` | full | tokens | tokens | main only |
| `all` | full | full | tokens | main plus every subagent |
| `selected` | full | full for selected, tokens for others | tokens | main plus selected agents |

A `compaction` call follows the main agent's row. It is a context rewrite of the
main path, not a separate agent.

For `main_only` and `selected`, an untrained subagent still keeps its token ids
and turn boundaries. Its next turn can continue from the same prefix and KV
cache.

## 7. Example

```text
P (main prompt)
└── M1 (main assistant, owner=main, full)
    ├── M2 (main continues, owner=main, full)
    └── S1 (subagent assistant, owner=subagent, tokens)
        └── S2 (subagent continues, owner=subagent, tokens)
```

`main_only`:

- One training row. `P` mask 0, `M1` mask 1, `M2` mask 1.
- `S1` and `S2` exist in the TITO snapshot for inference. They are not rows.

`all`, with `S1` and `S2` captured as `full`:

- Main row: `P` mask 0, `M1` mask 1, `M2` mask 1.
- Subagent row: `P` mask 0, `M1` mask 0 because the owner is main, `S1` mask 1,
  `S2` mask 1.
- `M1` is trained once.

`all`, with `S1` token-only and a later trainable child:

- The subagent row has `S1` mask 0 and its own later tokens mask 1. `S1` has no
  payload, so it can never be trained.

## 8. Failure modes

| Condition | Action |
|---|---|
| A trainable trajectory has a `Tokens` owned node | Raise when `require_primary_capture` is on |
| Two `Full` stores disagree on an owner | Keep the first owner and log a warning |
| No main trajectory in the session | Raise, because the session has no primary |
| The snapshot has no role or owner metadata | Use the legacy linear path |
| A `Tokens` record has no `output_token_ids` | Raise, because the boundary cannot be reconstructed |
| A shared node has `num_tokens` mismatch across snapshots | Raise, because the tree is inconsistent |
| `drop_dead_leaves` would drop a live subagent | Keep the leaf, because it is still pointed to by `trajectory_leaves` |
| The capture policy and the extraction policy disagree | `require_primary_capture` fails the sample |

## 9. Compatibility

- Old SMG snapshots have no role, owner, or capture fields. PSRL detects their
  absence and uses `build_training_data()` exactly as it does today. Every
  trajectory is trainable.
- Manual trajectory mode keeps working. The manual id is the owner, and the
  normalized headers still identify the role.
- Compaction branches are ordinary trajectories. The owner and capture fields
  apply to their nodes the same way.
- A harness that cannot mark subagent calls has all calls attributed to the main
  agent. This is the strict default and matches the current turn-budget rule.
- Claude Code auxiliary and side-reply calls are captured as `tokens` and never
  become rows. Compaction calls follow the main agent. The TITO tree already
  carries the compaction branch.
- The session API, the TITO tree shape, and the existing `x-codex-turn-metadata`
  input stay compatible.

## 10. Implementation phases

| Phase | Scope | Exit criterion |
|---|---|---|
| 0 | Contract: all four headers, session body, SMG fields, SessionRouter normalization, snapshot dump. Both `full` and `tokens` capture modes are implemented. | A Codex subagent episode shows role, owner, and capture in the snapshot. Existing tests pass. |
| 1 | First landing: `main_only`, all four headers active, `tokens` for untrained subagents and auxiliary calls, owner-aware masks, token-only extraction, routed-expert zero-row padding, tests. | A subagent episode produces one training row. An untrained subagent snapshot has no logprobs or routed experts, and its next turn still works. The main path has the correct masks. |
| 2 | `all` and `selected`, `session_primary`, multi-row advantage and loss | A shared prefix token is mask 1 exactly once. The session loss matches the expected total. |
| 3 | Per-agent reward and loss weights | Optional. Not required for the first landing. |

Phase 1 is the first landing. It covers all four headers, both capture modes,
and the token-only extraction path. There is no separate token-only phase.

## 11. Testing

SMG:

- Owner assignment on a new node, on a repeated `Full` store, and on a `Tokens`
  to `Full` upgrade.
- Token-only records omit logprobs, sampler distribution, and routed experts but
  keep `output_token_ids`.
- A shared node appears in `trajectory_ids` for every path and has one owner.
- The snapshot round-trips the new fields.
- `drop_dead_leaves` keeps live subagent leaves.

PSRL:

- `main_only` returns one row and keeps the subagent in the snapshot.
- `all` returns one row per trainable agent. The shared prefix is mask 1 only in
  the owner row.
- `selected` filters by agent id.
- A trainable node without `Full` capture fails when
  `require_primary_capture` is set.
- A token-only node is mask 0 in every sample.
- `session_primary` selects the main row for the advantage.
- A legacy snapshot without metadata matches the current extraction.
- `session-mean-token-mean` counts the shared prefix once.

Integration:

- Run one Codex main and subagent episode against a local SMG gateway.
- Inspect `GET /tito/sessions/{id}` for the tree, roles, owners, and capture.
- Run one training step with `main_only` and one with `all`.

## 12. Open questions

1. Does the Codex version in use put a stable `thread_id` and a parent thread id
   in `x-codex-turn-metadata`? If it does not, the fallback agent id is the
   trajectory id and the parent is derived from the tree.
2. Resolved: the harness runtime pins Claude Code 2.1.295 or newer. That version
   emits `x-claude-code-request-class`, `x-claude-code-agent-type`,
   `x-claude-code-agent-id`, and `x-claude-code-parent-agent-id`. The Claude
   Code adapter must set `CLAUDE_CODE_GATEWAY_HINT_HEADERS=1`.
3. Resolved: the session policy travels in the `POST /sessions` body, not in a
   header.
4. Does the deployed gateway rebuild from the vendored SMG source in
   `third_party/smg`? The implementation assumes it does.
5. Should `output_token_ids` be emitted for `Full` records as well? This design
   emits it to keep the extraction uniform. It can be dropped for `Full` records
   later if wire size matters.
