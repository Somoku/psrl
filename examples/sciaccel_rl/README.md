# SciAccel-RL: agentic RL on scientific-computing repair tasks

Trains a model to fix injected defects in real scientific simulation codebases
(LAPS, MITgcm, Athena++). Each task is a containerized Harbor episode: the agent
gets a repository and an instruction, edits source, and a verifier recompiles and
compares numerical output against reference frames.

This directory holds the whole pipeline. Read it top to bottom the first time.

| Path | Purpose |
|------|---------|
| [`prepare/`](prepare/) | Dataset construction, defect-line resolution, node provisioning |
| [`fsdp_qwen35_4b.sh`](fsdp_qwen35_4b.sh) | GRPO training entry point |
| [`eval_qwen35_4b.sh`](eval_qwen35_4b.sh) | Score a checkpoint through PSRL, no training |
| [`eval/`](eval/) | Standalone eval, independent of PSRL |
| [`agent_loop.py`](agent_loop.py), [`runner.py`](runner.py), [`agent.py`](agent.py) | Harbor integration |
| [`reward.py`](reward.py) | Verifier reward extraction |

---

## 0. Environment

Every command below assumes this first:

```bash
source /apdcephfs_zwfy10/share_303541817/lhy/env/psrl.sh
cd /apdcephfs_zwfy10_303541817/share_303541817/lhy/psrl
```

---

## 1. Get the task bank

```bash
git clone https://github.com/HHHHHejia/sciaccel-rl.git --branch easy-rl \
    /path/to/sciaccel-rl
```

That branch carries the `laps` environment already compiled. Additional
environments live in a separate, larger tree (`sciaccel-rl-full`); copy the ones
you want plus any source archive they share:

```bash
SRC=/apdcephfs_zwfy10_303541817/share_303541817/lhy/science_infra/sciaccel-rl-full
DST=/apdcephfs_zwfy10_303541817/share_303541817/lhy/science_infra/sciaccel-rl

cp -a $SRC/envs/mitgcm-biogeo  $DST/envs/
cp -a $SRC/envs/athena-gr      $DST/envs/
# athena-gr reads its pinned tarball from a sibling env, so that one directory
# has to come along even though the env itself is not used.
mkdir -p $DST/envs/athena-fft/env
cp -a $SRC/envs/athena-fft/env/source $DST/envs/athena-fft/env/
# The task compiler ships only in the full tree.
cp -a $SRC/utils/harbor $DST/utils/
```

### Compile authored tasks into Harbor tasks

**This step is mandatory for any env copied from the full tree, and it is the one
that is easy to miss.** `laps` is stored already compiled; the newer envs are
stored as authored *sources* that have no `environment/` directory. Harbor builds
its image from exactly that directory, so an uncompiled task fails at trial start
with `unable to prepare context: path .../environment not found`, after about two
seconds and with no other diagnostic.

#### There are two compilers, and they are not interchangeable

| Compiler | Scope | Dockerfiles come from |
|---|---|---|
| `utils/adapters/to_harbor.py` | **laps only** | templates hardcoded *inside the script* |
| `utils/harbor/to_harbor.py` | every other env | `envs/<env>/env/harbor/*.Dockerfile` |

This matters because the two carry their fixes in different places. The older
`adapters` version embeds its agent Dockerfile as a string literal, so its
`tmux`, `asciinema`, and apt-mirror handling apply **only to laps** and cannot
help another env. The newer `harbor` version is env-agnostic: it reads each env's
own templates, so a fix has to be made in `envs/<env>/env/harbor/` per env.

Consequence to know about: a fix that already exists for laps is not
automatically present in a newly copied env. Two were missing and had to be
ported by hand:

- **`tmux` in the agent image.** terminus-2 drives the container through a tmux
  pane, and the agent allowlist contains only model API hosts, so its runtime
  `apt install tmux` and build-from-source fallbacks both fail. The symptom is a
  flood of `Failed to install tmux from source` followed by `rollout_error` on
  every episode. `athena-gr` already shipped it; `mitgcm-biogeo` did not.
- **The apt mirror in the verifier's final stage.** See below.

Verify both after compiling a new env:

```bash
T=$(ls -d build/<env>/repair/easy/*/environment | head -1)
grep -c tmux $T/Dockerfile              # must be >= 1
docker build -t probe -f $T/Dockerfile $T && docker run --rm probe tmux -V
```

```bash
cd $DST
python utils/harbor/to_harbor.py --env envs/mitgcm-biogeo \
    --apt-mirror http://mirrors.tencentyun.com
python utils/harbor/to_harbor.py --env envs/athena-gr \
    --apt-mirror http://mirrors.tencentyun.com
```

**`--apt-mirror` is mandatory here, not optional.** It is baked into the generated
Dockerfiles at compile time, so omitting it cannot be corrected later without
recompiling. Every `apt-get update` in a task image then goes to `deb.debian.org`,
which measured **26 kB/s** from these nodes: 6.5 minutes to fetch the package index
alone. The build does not fail, it crawls, so the symptom is a warm pass that sits
at 0 completed tasks with `docker compose ... build` processes apparently hung.

The compiler writes to `build/<env>/<category>/<tier>/<task>/` and adds
`environment/`, `tests/`, and `solution/`. `build_dataset_v2.py` prefers
`build/<env>` when it exists and refuses to emit a dataset pointing at
uncompiled sources, so you cannot get this wrong silently twice.

---

## 2. Everything up to training, in one command

Steps 2 through 4 below have a hard dependency order, so they are also wrapped:

```bash
bash examples/sciaccel_rl/prepare/prepare_all.sh --dry-run   # see the plan first
bash examples/sciaccel_rl/prepare/prepare_all.sh
```

It runs **compile → resolve lines → build datasets → warm caches**, skips
already-compiled envs, drops the difficulty filter for `laps` (whose tasks predate
the tiers), probes each node's Docker before dispatching, and warms envs
sequentially per host so concurrency stays at the intended 12 rather than 12 times
the env count.

Re-run a single stage after a failure instead of redoing slow work:

```bash
bash examples/sciaccel_rl/prepare/prepare_all.sh --stages dataset,warm
bash examples/sciaccel_rl/prepare/prepare_all.sh --envs athena-gr --stages warm
```

The order is not cosmetic. A dataset built before `compile` points at task
directories with no `environment/`, and every episode dies about two seconds in
with `unable to prepare context`. A dataset built before `lines` silently drops
the line from every L1 hint for an env that records only the file, quietly
turning L1 into L2.

The rest of this section explains each stage for when you need to run one by hand.

---

## 3. Resolve defect line numbers (only for new envs)

The strongest hint level names the file **and line** of the defect. Envs disagree
on whether they record one: `laps` and `mitgcm-biogeo` store
`candidate.meta.line`, `athena-gr` stores only the file.

```bash
python examples/sciaccel_rl/prepare/resolve_defect_lines.py \
    --repo $DST --env laps --env mitgcm-biogeo --env athena-gr
```

This downloads each env's pinned upstream source, verifies its sha256, and finds
the line by matching the provenance `old` text block literally. It writes
`envs/<env>/factory/DEFECT_LINES.json`.

A recorded line always wins and is never cached, so the cache holds only what an
env is missing. The method is cross-validated where both exist: all 91
mitgcm-biogeo tasks with a recorded line resolve to exactly that line. The 8
`laps` offsets it reports are expected, not errors: the LAPS build inserts a
10-line instrumentation patch into `mhd.f90`, so its recorded lines are
post-patch while the resolver reads the pristine tree.

Re-run only when the task bank changes.

---

## 4. Build the hinted datasets

The hint is a localization aid appended to the instruction:

```
## Where to look

The defect is a single edit confined to:

    pkg/bling/bling_bio_nitrogen.F, line 1037

The change made there: clip a loop upper bound by one. No other file has been modified.
```

Three levels: **L1** file and line, **L2** file only, **L3** no hint (control).
Hints exist because an unhinted 4B model scored near zero: the task became
*finding* the defect in a large repository rather than *fixing* it.

```bash
R=$DST

# laps: no difficulty field on its tasks, so no filter
python -m examples.sciaccel_rl.prepare.build_dataset_v2 \
    --repo $R --out-dir examples/sciaccel_rl/data/v2_repair \
    --env laps --categories repair --hint-level all

# newer envs: tier-nested, so filter to the measured-easy set
python -m examples.sciaccel_rl.prepare.build_dataset_v2 \
    --repo $R --out-dir examples/sciaccel_rl/data/mitgcm-biogeo_repair_easy \
    --env mitgcm-biogeo --categories repair --difficulty easy --hint-level all

python -m examples.sciaccel_rl.prepare.build_dataset_v2 \
    --repo $R --out-dir examples/sciaccel_rl/data/athena-gr_repair_easy \
    --env athena-gr --categories repair --difficulty easy --hint-level all
```

Each run writes, per level: `L*_all.parquet`, `L*_train.parquet`,
`L*_val.parquet`, plus a shared unhinted `val.parquet`, `split.json`, and
`L*_stats.json`.

| Dataset | all | train | val |
|---|---|---|---|
| `v2_repair` (laps) | 99 | 85 | 14 |
| `mitgcm-biogeo_repair_easy` | 87 | 64 | 23 |
| `athena-gr_repair_easy` | 104 | 87 | 17 |

Two validation files exist on purpose. `L1_val.parquet` is **hinted**, matching
the training distribution, and is what you want for measuring training progress.
`val.parquet` is **unhinted**, and measures unaided localization, which a
hint-trained model was never asked to do. Comparing a hinted checkpoint against
`val.parquet` will look like a catastrophic regression that is really a task
change.

The split is stratified by `(category, family, tree)` and chosen by sorted task
name, so it is reproducible without a seed and identical across hint levels.

---

## 5. Warm the Docker image cache, on every node that runs episodes

Harbor containers run wherever the agent-loop process runs, and the buildkit
cache is node-local (`/var/lib/docker/buildkit`). A cold node spends 8-15 minutes
per task on its first build. Warm each node with the `nop` agent, which builds
and tears down without needing a GPU or a model:

```bash
R=/apdcephfs_zwfy10_303541817/share_303541817/lhy/psrl
for H in 28.58.246.40 28.59.83.117; do
  ssh -o BatchMode=yes "$H" "cd $R && nohup setsid bash examples/sciaccel_rl/eval/run_eval.sh \
      --agent nop \
      --dataset examples/sciaccel_rl/data/mitgcm-biogeo_repair_easy/L1_all.parquet \
      --output-dir /tmp/nop_warm_${H//./_} \
      --skip-gpu-tasks --max-per-instance 12 -n 12 \
      > /tmp/nop_warm.log 2>&1 < /dev/null &"
  sleep 2   # staggering matters: launching all at once has silently dropped a host
done
```

See [`prepare/NEW_NODE_RUNBOOK.md`](prepare/NEW_NODE_RUNBOOK.md) for provisioning
a brand-new node, which additionally needs registry mirrors and a buildkit proxy.

### Check Docker health first

A degraded daemon is the single most common cause of a stalled run, and it does
not announce itself. Before launching anything:

```bash
timeout 20 docker ps -q | wc -l      # hangs => daemon is wedged, do not launch
ps -eo comm | grep -c fuse-overlayfs # hundreds => orphaned mounts
uptime                               # load >> core count
```

If `docker ps` times out, that node cannot host episodes. Exclude it with
`AGENT_NODE_IPS` rather than waiting for it to recover.

---

## 6. Train

```bash
bash examples/sciaccel_rl/fsdp_qwen35_4b.sh
```

Defaults to `v2_repair` at `L1` on 3 nodes (8 generation + 16 training GPUs).
Useful overrides:

```bash
DATA_DIR=examples/sciaccel_rl/data/mitgcm-biogeo_repair_easy \
HINT_LEVEL=L1 \
AGENT_NODE_IPS=28.58.246.40,28.59.83.117 \
    bash examples/sciaccel_rl/fsdp_qwen35_4b.sh
```

| Variable | Meaning |
|---|---|
| `DATA_DIR` | Dataset directory. Becomes part of the experiment name. |
| `HINT_LEVEL` | `L1`, `L2`, or `L3`. Selects `${HINT_LEVEL}_train.parquet`. |
| `AGENT_NODE_IPS` | Nodes allowed to host containers. Empty means all alive nodes. |
| `OVERLONG_FILTERING` | DAPO overlong filtering, on by default. See below. |
| `GROUP_FILTER` | Drop zero-variance GRPO groups. Off by default. |
| `MAX_TURNS` | Turn cap, 50. Raising it needs `MAX_RESPONSE_LENGTH` raised too. |

### Metrics that matter

Watch these rather than `critic/score/mean` alone, which blends two populations
that move in opposite directions:

- `termination/finished/score_mean` — reward on episodes that actually finished
- `termination/max_turns_exceeded/fraction` — share cut off by the turn cap
- `group/zero_variance_fraction` — share of GRPO groups producing no gradient
- `rollout_corr/rollout_is_eff_sample_size` — rollout-vs-trainer agreement; a
  drop here means a weight-transfer problem, not an RL one

`OVERLONG_FILTERING=True` zeroes the loss mask of budget-truncated episodes while
keeping their reward in the GRPO baseline. Without it, a truncated episode is
graded as a policy failure even though the harness cut it off, and under
`token-mean` the cheapest way to shed that penalty is to emit shorter turns,
which spends the turn cap faster. That feedback loop was measured collapsing a
run from 0.573 at step 11 to 0.078 at step 16.

---

## 7. Evaluate

### Through PSRL, reusing the training topology

```bash
CKPT_PATH=examples/sciaccel_rl/ckpts/sciaccel_rl/<experiment>/global_step_30 \
    bash examples/sciaccel_rl/eval_qwen35_4b.sh
```

`val_only=True` returns straight after the initial validation, so nothing trains
and no optimizer step runs. `resume_mode=resume_path` loads the named checkpoint
rather than searching, which matters: `auto` would find nothing in the empty eval
directory and silently score the **base** model.

The checkpoint is FSDP-sharded across 16 ranks, so it must be scored on the same
`TRAIN_NNODES x TRAIN_NGPUS_PER_NODE` topology that wrote it.

### Standalone, without PSRL

Use [`eval/`](eval/) when you want a number without the training stack: it serves
the model with vLLM, drives Harbor's own `terminus-2` harness, and aggregates by
category and family.

```bash
bash examples/sciaccel_rl/eval/run_eval.sh \
    --model /apdcephfs_zwfy10_303541817/share_303541817/lhy/models/Qwen3.5-4B \
    --dataset examples/sciaccel_rl/data/mitgcm-biogeo_repair_easy/L1_val.parquet \
    --output-dir examples/sciaccel_rl/outputs/eval_biogeo_base \
    --skip-gpu-tasks
```

Serve with 4 replicas at TP=2 (`topology=fleet`). Do **not** use vLLM
`--data-parallel-size`: it is broken in this repo's patched vLLM.

Two agents need no model at all and are the fastest way to test infrastructure:

| `--agent` | Does | Use for |
|---|---|---|
| `nop` | Builds the environment, edits nothing | Warming the image cache; proving a node can build |
| `oracle` | Applies the known fix from `solution/` | Proving the verifier grades a correct patch |

`oracle` is the real end-to-end check. It should score near 1.0; anything lower
means the task or verifier is broken, not the model. Budget time for it: a single
MITgcm task takes about 8 minutes to build and up to 15 more in the verifier
(`verifier_timeout_sec: 900`).

---

## Gotchas worth knowing before you hit them

- **`/tmp` is node-local.** Put datasets on the shared filesystem before a
  cross-node run, or the remote node reads a stale copy and fails against paths
  that no longer exist.
- **Editing the `prompt` column does nothing.** Harbor re-reads `instruction.md`
  from disk. The hint reaches the model through `extra_info["hint"]`, which
  `runner.py` passes as Harbor `extra_instructions`. The `prompt` column is a
  record of the delivered text, not the delivery path.
- **`max_model_len` is the agent's budget, not slack.** It is forwarded to
  terminus-2 as `max_input_tokens`, so any headroom is headroom the agent will
  spend, and TITO then hands the trainer a response longer than
  `max_response_length`.
- **Dangling images accumulate over days** on fuse-overlayfs and eventually wedge
  the daemon. `docker image prune -f` was measured removing 0 of 5816 in 25
  minutes on a degraded daemon, while batched `docker rmi -f` cleared them in
  under 3.
