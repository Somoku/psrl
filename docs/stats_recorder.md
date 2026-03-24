# Worker Stats Monitoring — Quick Reference

## What This Does

When enabled, `RolloutCoordinator` writes one JSONL file per vLLM replica instance to
`psrl.logging_path` (default `~/psrl_logs/`). Each file is a time-series of per-replica
stats snapshots, sampled every `interval_in_s` seconds (default 5s).

## How to Enable

In your training config or launch script, override:

```bash
psrl.status_collection.stats_recorder.enable=True
# Optional: adjust interval
psrl.status_collection.stats_recorder.interval_in_s=5.0
```

## Output Files

```
~/psrl_logs/
  stats_config.json            # run-level metadata (routing strategy, partial_rollout, etc.)
  stats_rollout-0_dp0.jsonl    # replica "rollout-0", data-parallel rank 0
  stats_rollout-1_dp0.jsonl    # replica "rollout-1", data-parallel rank 0
  ...
```

Each JSONL row:
```json
{
  "ts": "2026-03-23T10:15:03.421+00:00",
  "model_version": 42,
  "num_running_reqs": 8,
  "num_waiting_reqs": 2,
  "kv_cache_usage": 0.63,
  "generation_throughput": 412.5,
  "avg_ttft": 0.034,
  "avg_itl": 0.009
}
```

`avg_ttft` and `avg_itl` are `null` when no generation is active (e.g., during weight sync).

## Verifying Routing Policies

Load all instance files into pandas and compute derived metrics:

```python
import pandas as pd
import glob
import json
import os

# Load run config
with open(os.path.expanduser("~/psrl_logs/stats_config.json")) as f:
    run_cfg = json.load(f)
print(f"Routing strategy: {run_cfg['routing_strategy']}")

# Load all instance timelines
files = glob.glob(os.path.expanduser("~/psrl_logs/stats_*.jsonl"))
dfs = {}
for f in files:
    name = os.path.basename(f).replace("stats_", "").replace(".jsonl", "")
    dfs[name] = pd.read_json(f, lines=True)

# --- Load balance (request_num_balance) ---
# Merge on timestamp (rows are aligned since they share the same recorder tick)
combined = pd.concat([df.assign(instance=name) for name, df in dfs.items()])
pivot = combined.pivot_table(index="ts", columns="instance", values="num_running_reqs")
pivot["balance_ratio"] = pivot.max(axis=1) / pivot.mean(axis=1).replace(0, float("nan"))
print(pivot["balance_ratio"].describe())
# Good: mean ≈ 1.0, max close to 1.0

# --- Aggregate throughput (throughput_optimal) ---
throughput = combined.groupby("ts")["generation_throughput"].sum()
print(throughput.describe())

# --- Partial rollout sync rate ---
waiting = combined.pivot_table(index="ts", columns="instance", values="num_waiting_reqs")
all_idle = (waiting == 0).all(axis=1)
print(f"Fraction of ticks where all replicas idle: {all_idle.mean():.2%}")

# --- KV cache per replica ---
kv = combined.pivot_table(index="ts", columns="instance", values="kv_cache_usage")
kv.plot(title="KV cache utilization per replica")
```

## Notes

- Files are **overwritten** at the start of each training run (same `logging_path`).
- The feature adds negligible overhead: one dict read + one file write every N seconds inside an asyncio task.
- Disable with `psrl.status_collection.stats_recorder.enable=False` (default).
