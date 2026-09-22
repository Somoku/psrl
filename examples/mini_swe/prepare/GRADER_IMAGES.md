# Grading gate and problem-image repair

A prepared split is only usable if every row can be solved by its own gold patch.
When it cannot, the row still consumes rollout budget and reports a model failure
that no model can avoid, so `acc` stops meaning "the policy is weak".

Two tools keep a split honest:

- `prepare/gold_gate.py` grades each row with its own gold patch and freezes the
  result into the row.
- `prepare/docker_scripts/repair_grader_images.sh` bakes a per-problem image with
  the dependency a repository needs, then points the row at it.

Placeholders used below:

| Placeholder | Meaning |
|---|---|
| `${PSRL_WORKSPACE}` | Cluster workspace root, already exported by `env/`. |
| `<parquet>` | Prepared split, for example `examples/mini_swe/data/swe_gym_293/val.parquet`. |

## Quickstart

Run from the repository root, in this order:

```bash
# 1. Repair the images whose repository cannot import at all.
bash examples/mini_swe/prepare/docker_scripts/repair_grader_images.sh \
    --parquet examples/mini_swe/data/swe_gym_293/val.parquet

# 2. Gate the split. Prunes pass-to-pass entries the image cannot run and
#    writes <parquet-stem>.gold_report.json next to the parquet.
python -m examples.mini_swe.prepare.gold_gate \
    --parquet examples/mini_swe/data/swe_gym_293/val.parquet --prune-p2p
```

The gate exits non-zero while any row is unusable. Its report names the reason per
row, which is what tells you whether the remaining failures need another recipe or
a data fix. When the split is clean, the report reports `gold_ceiling=1.0`.

## Files

| Path | Purpose |
|---|---|
| `prepare/gold_gate.py` | Grade every row's gold patch, prune, and set `gold_ceiling`. |
| `prepare/retarget_problem_images.py` | Move a row onto a repaired problem image. |
| `prepare/docker_scripts/repair_grader_images.sh` | Bake and record the repaired images. |
| `<parquet-stem>.gold_report.json` | Per-row verdicts from the last gate run. |
| `<parquet-stem>.grader_images.tsv` | Repaired image map the gate and retarget share. |

## The gate

```bash
python -m examples.mini_swe.prepare.gold_gate --parquet <parquet> [options]
```

| Flag | Default | Effect |
|---|---|---|
| `--instances REGEX` | all rows | Gate only rows whose `instance_id` matches. |
| `--repeats N` | `1` | Gold runs per row. A test is pruned only if it fails in every run. |
| `--workers N` | `4` | Concurrent grader containers. |
| `--timeout S` | `900` | Grader eval-script budget. |
| `--prune-p2p` | off | Rewrite the parquet with the pruned expectations. |
| `--dry-run` | off | Grade and print, but write nothing. |
| `--allow-partial` | off | Exit 0 even when a row is unusable. |

What it writes, per row, under `extra_info.swe_problem`:

- `gold_ceiling`: `1.0` when the row resolves with its own gold patch, else `0.0`.
  The reward reports it as `val-aux/<dataset>/gold_ceiling/mean@1`, so read `acc`
  next to the best score the split allows.
- `pruned_pass_to_pass`: pass-to-pass tests dropped because the image cannot run
  them. They never become a regression signal again.

Only pass-to-pass expectations are pruned, and only those that fail under the gold
patch. A failing fail-to-pass test means the row itself is wrong, so the gate
fails instead of relaxing it.

## Repairing an image

An image is repaired when the repository cannot even be imported, for example a
`vtk` import that needs `libGL.so.1` or a repo using `np.Inf` under NumPy 2. The
script derives a derivative per base image and records the mapping:

```bash
bash examples/mini_swe/prepare/docker_scripts/repair_grader_images.sh \
    --parquet <parquet> [--instances REGEX] [--dry-run]
```

| Recipe | Repository | Repair |
|---|---|---|
| `numpy1` | `pvlib/pvlib-python` | Pin NumPy below 2 in the `testbed` env, then `import pvlib`. |
| `gl` | `pyvista/pyvista` | Install the GL/X libraries `vtk` loads, then `import pyvista`. |

A repository without a recipe is skipped on purpose, because its failure needs a
data fix rather than an image fix.

| Flag | Default | Effect |
|---|---|---|
| `--instances REGEX` | all rows | Repair only the images those rows use. |
| `--revision TAG` | `v1` | Bump to invalidate every previously repaired image. |
| `--namespace NAME` | `psrl/swebench-grader` | Tag namespace for the derivatives. |
| `--map FILE` | `<parquet-stem>.grader_images.tsv` | Where the `base<TAB>repaired` map is written. |
| `--save-tars` | off | Also `docker save` each derivative for the node fan-out. |
| `--no-apply` | off | Bake and record, but leave the parquet untouched. |
| `--force` | off | Rebuild even when the derivative already exists. |

Tags are content addressed on the base image, the recipe, and the revision, so
re-running the script skips everything already repaired. Each recipe ends by
importing the repository, and the commit is skipped when that import fails.

### Installing into a restricted network

The recipes use the default pip and apt endpoints. To route through a mirror, set
`PSRL_GRADER_PIP_ARGS` (extra `pip install` flags) or `PSRL_GRADER_IMAGE_DIR`, and
note that apt mirrors belong in the image build, not here.

### Multi-node runs

Grading runs on a validation instance, so every node that can host one needs the
derivative:

```bash
bash examples/mini_swe/prepare/docker_scripts/repair_grader_images.sh \
    --parquet <parquet> --save-tars

bash examples/mini_swe/prepare/docker_scripts/load_all_nodes.sh \
    --hosts ${PSRL_WORKSPACE}/hosts/16GPUs \
    --image-dir ${PSRL_WORKSPACE}/docker_images/swe_gym_293 \
    --images-list <parquet-stem>.grader_images.images.txt
```

## Gotchas

- **The gate keeps failing the same rows.** The image is still broken. Repair it,
  then re-run `--dry-run` to see the remaining reason per row.
- **A row resolves but `gold_ceiling` stays `0.0`.** The gate ran without
  `--prune-p2p`, so the frozen expectations were not written back.
- **`log_unparseable` on every run of one repository.** The repo cannot be
  imported at all, which no patch can fix. Add a recipe for it.
- **Grading still uses the old image after a repair.** The grader reads
  `extra_info.swe_problem_image`, which only changes when the retarget step runs.
  Use the script without `--no-apply`, or run
  `python -m examples.mini_swe.prepare.retarget_problem_images --parquet <parquet> --overrides <map>`.
- **The repaired image is missing on a worker node.** `docker load` the tar there,
  or re-run the fan-out above.
