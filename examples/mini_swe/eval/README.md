# Standalone Evaluation and Model Serving

This directory packages every tool needed to evaluate SWE-bench / SWE-smith
outside the training loop, and to serve your own checkpoint as an
OpenAI-compatible endpoint for mini-swe-agent to drive.

| File | Purpose |
|------|---------|
| [`eval_swebench.py`](eval_swebench.py) | Single-host evaluation entry point (rollout + grading). Supports an HF dataset key *or* a prepared parquet file via `--dataset`. |
| [`eval_swebench_multinode.py`](eval_swebench_multinode.py) | Hash-shards a prepared parquet across hosts, fans `eval_swebench` out over ssh, merges per-shard artefacts into one output directory. Forwards `OPENAI_API_BASE` / `OPENAI_API_KEY` to every host. |
| [`serve_vllm.sh`](serve_vllm.sh) | Single-node vLLM OpenAI-compatible server wrapper with TP / PP / DP flags, tool-call-parser selection, health probe, and background launch. |
| [`serve_vllm_multinode.sh`](serve_vllm_multinode.sh) | Fans `serve_vllm.sh` to every host in a hosts file (data-parallel across hosts), writes `endpoints.txt` + a drop-in litellm proxy config that routes across every replica. |

The grader itself — [`../swebench_grader.py`](../swebench_grader.py) — stays at
the top level of `examples/mini_swe/` because it is shared between standalone
evaluation here and the post-rollout grading used inside the training loop.

---

## Standalone Evaluation

Evaluation runs independently of the training loop via `eval_swebench.py`.
This is used for:

- **Gold-patch sanity check** — verifies that Docker images and the grader are
  correctly configured before investing in a full training run.
- **Baseline measurement** — run on the initial checkpoint to establish a reference.
- **Checkpoint evaluation** — run on a saved checkpoint to produce leaderboard-ready
  `preds.json`.

### Gold-patch sanity check

Every SWE-bench SWE problem should resolve when the gold patch is applied. Use
this to confirm that images are present and the grading pipeline works end-to-end:

```bash
python -m examples.mini_swe.eval.eval_swebench \
    --gold-patches \
    --dataset examples/mini_swe/data/verified_subset_80/train.parquet \
    --subset-spec "^astropy" \
    --output-dir examples/mini_swe/output/eval/gold_sanity \
    --workers 4
```

Expect most of the problems are resolved, e.g, 68/80 (85.0%). SWE problems that fail consistently likely have fragile test environments; exclude them from future evaluation subsets if needed.

Heavy repos (`scikit-learn`, `psf/requests`, `matplotlib`) spend most of their
time on `pip install -e .` + the full PASS_TO_PASS suite and can legitimately
take 10–20 minutes each; keep `--grader-timeout` at the default `1800` (or
higher) and tune `--workers` upward until per-task `elapsed_s` starts to rise
(rough rule: `min(CPU_cores / 8, RAM_GiB / 12, 16)`).

### Multi-node evaluation

For larger subsets (e.g. the full 500-problem Verified split) fan the eval
out across every host listed in a hosts file. Each host is given a shard of
the parquet (bucketed by `hash(instance_id)`), runs `eval_swebench` locally
with its usual `--workers`, and the per-shard artefacts are merged into one
combined output directory:

```bash
python -m examples.mini_swe.eval.eval_swebench_multinode \
    --hosts ${PSRL_WORKSPACE}/hosts/32GPUs \
    --dataset examples/mini_swe/data/verified_subset_80/train.parquet \
    --output-dir examples/mini_swe/output/eval/gold_sanity_mn \
    --gold-patches \
    --workers-per-node 8 \
    --grader-timeout 1800 \
    --ssh-timeout 3600
```

Prerequisites are the same as `prepare/docker_scripts/load_all_nodes.sh`:
the repo, conda env, and output directory live on a shared FS; the target
hosts already have their shard's Docker images loaded (run
`load_all_nodes.sh` first); and passwordless ssh works from the launcher
to every host. Inspect `--dry-run` output first to review the exact ssh
commands that will be issued. Per-host stdout/stderr lands in
`<output-dir>/host_logs/`, raw per-shard artefacts in
`<output-dir>/host_output/<host>/`, and the merged `preds.json` /
`results.jsonl` / `summary.json` live at the top of `<output-dir>`
(per-instance directories are symlinked up from the host-specific output).

### Evaluating your own checkpoint

`eval_swebench.py` is **decoupled from the training stack**: it does not use
PSRL's agent loop at all. Internally it calls mini-swe-agent's `LitellmModel`,
which issues `litellm.completion(model=<prefixed_model_name>, ...)` for every
turn. The `openai/` prefix applied in [`eval_swebench.py`](eval_swebench.py)
means every request goes to whatever HTTP endpoint `OPENAI_API_BASE` points
at. So to evaluate your own checkpoint, serve it with **any OpenAI-compatible
server** (vLLM / sglang / TGI / llama.cpp server / litellm proxy) and point
the eval at it.

The two `serve_vllm*.sh` helpers in this directory wrap vLLM for this:

#### Single-node (TP / PP / DP on one box)

PSRL trains with the `mswea_bash_command` text-block format (not OpenAI
tool-calls), so vLLM should be started as a **plain text-completion server**
— no `--tool-call-parser`, no `--enable-auto-tool-choice`:

```bash
bash examples/mini_swe/eval/serve_vllm.sh \
    --checkpoint ${PSRL_WORKSPACE}/checkpoints/my-step-1000 \
    --served-model-name my-model \
    --port 8000 \
    --tp 4 \            # tensor parallel across 4 GPUs on this host
    --pp 1 \
    --dp 1              # vLLM in-server data-parallel replicas (>=0.6)
# No --tool-call-parser here.  The model was trained to output
# ```mswea_bash_command blocks, not OpenAI tool-call JSON.
```

`--tool-call-parser` is only needed when serving **external models** (GPT-4,
Claude, Llama3-Instruct, etc.) that natively output OpenAI tool-call JSON and
are evaluated with `--model-class litellm` (see "Evaluating external models"
below).

The script launches vLLM with `nohup setsid`, polls `/v1/models` until the
server reports ready, and writes `/tmp/vllm_<port>.{log,pid}` so you can
tail logs and kill cleanly. Add `--foreground` when debugging.

Then point the eval at it — the default `--model-class litellm_textbased`
matches the training-time action format:

```bash
export OPENAI_API_BASE=http://<serve_host>:8000/v1
export OPENAI_API_KEY=dummy   # vLLM ignores it, but litellm requires the field

python -m examples.mini_swe.eval.eval_swebench \
    --model my-model \
    --dataset examples/mini_swe/data/verified_subset_80/train.parquet \
    --output-dir examples/mini_swe/output/eval/my_step1000 \
    --workers 8 \
    --grader-timeout 1800
# --model-class litellm_textbased is the default; no need to specify it
# unless you want to override to 'litellm' for an external model.
```

#### Cross-node (DP across hosts)

For bigger throughput, run one full vLLM replica per host (data-parallel
across hosts):

```bash
bash examples/mini_swe/eval/serve_vllm_multinode.sh \
    --hosts ${PSRL_WORKSPACE}/hosts/32GPUs \
    --checkpoint ${PSRL_WORKSPACE}/checkpoints/my-step-1000 \
    --served-model-name my-model \
    --port 8000 \
    --tp 4 \
    --outdir examples/mini_swe/output/serve/my_step1000
# No --tool-call-parser for PSRL-trained models.
```

`serve_vllm_multinode.sh` writes `<outdir>/endpoints.txt` (one
`http://<host>:<port>` per healthy host) and `<outdir>/litellm_proxy.yaml`
(a litellm router config).

**Recommended: direct-to-localhost (no central proxy)**

Each eval shard calls its own local vLLM.  
This avoids the `litellm` CLI, which hangs on startup in environments with a corporate HTTP proxy:

```bash
export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY=dummy
export NO_PROXY="localhost,127.0.0.1"

python -m examples.mini_swe.eval.eval_swebench_multinode \
    --hosts ${PSRL_WORKSPACE}/hosts/32GPUs \
    --dataset examples/mini_swe/data/verified_subset_80/train.parquet \
    --output-dir examples/mini_swe/output/eval/my_step1000_mn \
    --model my-model \
    --workers-per-node 8 \
    --grader-timeout 1800 \
    --ssh-timeout 3600
```

`eval_swebench_multinode.py` auto-forwards `OPENAI_API_BASE`, `NO_PROXY` and
`OPENAI_API_KEY` to every remote host, so no per-host bashrc editing is
needed.  
Use `--forward-env NAME` to add extra vars or `--set-env NAME=VALUE`
to override.

**Alternative: litellm proxy (open-network environments)**

If there is no corporate proxy, a central litellm router can load-balance
across all replicas:

```bash
litellm --config <outdir>/litellm_proxy.yaml --port 4000 &
export OPENAI_API_BASE=http://<launcher_ip>:4000/v1
export OPENAI_API_KEY=dummy
# then run eval_swebench_multinode as above
```

#### Evaluating external models (optional: OpenAI / Claude / third-party)

For models that natively support OpenAI tool-calling (GPT-4o, Claude,
Llama3-Instruct, etc.) you need the tool-call parser on the vLLM side:

```bash
bash examples/mini_swe/eval/serve_vllm.sh \
    --checkpoint ${PSRL_WORKSPACE}/checkpoints/external-model \
    --served-model-name ext-model \
    --port 8001 \
    --tp 4 \
    --tool-call-parser hermes      # llama3_json / mistral / deepseek_v3 as needed
```

Then eval with `--model-class litellm`:

```bash
export OPENAI_API_BASE=http://<serve_host>:8001/v1
export OPENAI_API_KEY=dummy
python -m examples.mini_swe.eval.eval_swebench \
    --model ext-model \
    --model-class litellm \        # switches to the OpenAI tool-call path
    --dataset examples/mini_swe/data/verified_subset_80/train.parquet \
    --output-dir examples/mini_swe/output/eval/ext_model
```

#### Cross-node tensor parallelism

`serve_vllm_multinode.sh` does **not** do cross-node TP — it assumes each
host fits one full replica. If your model is so large that one host's GPUs
aren't enough, start a Ray cluster yourself (`ray start --head` on the head
node, `ray start --address=<head>:6379` on the workers), then use
`serve_vllm.sh` on the head node with
`--distributed-executor-backend ray --tp <total_gpus>`.

#### Stopping vLLM replicas

```bash
# Single host
kill "$(cat /tmp/vllm_8000.pid)"

# Every host in a hosts file
pssh -h ${PSRL_WORKSPACE}/hosts/32GPUs -i \
    "pkill -f 'vllm.entrypoints.openai.api_server.*--port 8000'"
```

### Checkpoint evaluation (HF dataset mode)

When you just want to run on a slice of the raw HF Verified split instead
of a prepared parquet:

```bash
python -m examples.mini_swe.eval.eval_swebench \
    --model /path/to/checkpoint \
    --dataset verified \
    --split test \
    --subset-spec "0:100" \
    --output-dir output/eval/step200 \
    --workers 8 \
    --max-turns 30
```

For a full leaderboard submission run omit `--subset-spec` to evaluate all 500
Verified SWE problems.

### Output artefacts

```
<output-dir>/
  preds.json          # { instance_id: {model_patch, model_name_or_path, ...} }
  summary.json        # { resolved, total, resolve_rate, avg_turns, elapsed_s, ... }
  results.jsonl       # One JSON object per line, per-SWE-problem result
  <instance_id>/      # One directory per SWE problem (named after its HF instance_id)
    traj.json         # Full conversation + exit status
    patch.diff        # Submitted patch
    grading.json      # Raw output of grade_fresh_container
```

`preds.json` is compatible with the official `swebench.harness.run_evaluation`
grader for leaderboard submission.

### In-training validation

During training, PSRL runs validation rollouts on `test_files` every `test_freq`
steps using the same `MiniSWEAgentLoop` and `compute_score` as training. The
`train/acc` and `val/acc` wandb metrics track resolve rate throughout training
without needing to invoke `eval_swebench.py`.
