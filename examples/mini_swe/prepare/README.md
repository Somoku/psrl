# Data Preparation for mini-SWE-agent RL Training

This directory contains everything needed to convert raw datasets into PSRL
training parquets and to warm up Docker image caches on cluster nodes.

Two independent data paths are supported:

- **Path A — Toy dataset**: 40–64 synthetic Python bug-fix tasks baked into
  a single `python:3.11-slim` image. Good for rapid iteration and smoke tests.
- **Path B — SWE-smith-py + SWE-bench Verified**: Real-world bugs from the
  SWE-smith-py collection (~51k SWE problems) for RL training, graded by running
  the actual test suite inside per-problem Docker images. SWE-bench Verified
  (500 SWE problems) is used for periodic validation.

---

## Files

| File | Description |
|------|-------------|
| `prepare_simple_data.py` | Toy dataset generator — produces `train.parquet` / `test.parquet` from `simple_cases_*.json` |
| `simple_cases_train.json` | 40 synthetic training bug-fix tasks |
| `simple_cases_val.json` | 12 synthetic validation bug-fix tasks |
| `prepare_swebench.py` | HF → parquet converter for SWE-smith-py and SWE-bench Verified/Lite/Full |
| `swebench_subsets.py` | Repo-balanced sampling helpers used by `prepare_swebench.py` |
| `docker_scripts/bake_simple_repos.sh` | Bakes toy repositories into a Docker image for Path A |
| `docker_scripts/prefetch_images.sh` | Pull per-SWE-problem images (skopeo-first, multi-mirror fallback, tar cache, `docker load`) |
| `docker_scripts/prefetch_example.sh` | Reference invocation that chains `prefetch_images.sh` + `load_all_nodes.sh` |
| `docker_scripts/probe_mirrors.sh` | Quickly check which public Docker Hub mirrors can serve a given image (uses `skopeo inspect`, no download) |
| `docker_scripts/load_all_nodes.sh` | `pssh` fan-out: on every host listed in a file, `docker load` every `*.tar` in a shared-FS image dir, with per-node parallelism and skip-if-already-loaded |
| `_prefetch_logs/` | One log file per image (kept by `prefetch_images.sh`) — header `已经拥有了` when cached, or a full per-mirror/per-attempt log when pulled |
| `_load_logs/<timestamp>/` | `pssh` per-host stdout / stderr from `load_all_nodes.sh` |

---

## Path A — Toy dataset (quick iteration)

### Step 1: Bake repos into a Docker image

```bash
# From examples/mini_swe/
bash prepare/docker_scripts/bake_simple_repos.sh python:3.11-slim
```

This creates a local Docker image (tagged `psrl-mini-swe:latest` by default) with
all toy repositories pre-installed under `/<split>_<idx>/`. The script may need
proxy settings uncommented if your network requires them.

### Step 2: Generate parquet datasets

```bash
python prepare/prepare_simple_data.py \
    --mode simple \
    --train_size 64 \
    --test_size 16 \
    --output_dir examples/mini_swe/data/mini_swe_agent
```

Output: `data/mini_swe_agent/train.parquet` and `data/mini_swe_agent/test.parquet`.

Each row contains a synthetic problem statement, a reference patch, and
`sandbox_overrides.use_preexisting_repo = True` pointing to the pre-baked repo.

---

## Path B — SWE-smith-py + SWE-bench Verified (real RL)

### Prerequisites

Extra Python packages are required for grading (not needed for Path A):

```bash
python -m pip install swebench==4.1.0 swesmith
```

### Step 1: Generate SWE-smith-py training data

SWE-smith-py has 50,908 SWE problems, each with a pre-built per-problem Docker
image.  A repo-balanced 1,000-problem subset is a good starting point (roughly
10 bugs per repository, ~100 unique images to pull):

```bash
python -m examples.mini_swe.prepare.prepare_swebench \
    --dataset smith \
    --split train \
    --total 1000 \
    --per-repo-k 10 \
    --output-dir examples/mini_swe/data/swe_smith_py_1k
```

For a larger scale (more diversity, more disk space):

```bash
# 5 000 SWE problems, ~20 per repo
python -m examples.mini_swe.prepare.prepare_swebench \
    --dataset smith --split train \
    --total 5000 --per-repo-k 20 \
    --output-dir examples/mini_swe/data/swe_smith_py_5k
```

The script always applies repo-balanced round-robin sampling so no single
repository dominates. Pass `--no-repo-balanced` to disable and simply truncate
to `--total` instead.

### Step 2: Generate SWE-bench Verified validation data

Training uses a small repo-balanced subset for fast `test_freq` evaluation.
The full 500-problem set is used for leaderboard-quality final evaluation.

```bash
# 80-problem subset — used as test_files during training
python -m examples.mini_swe.prepare.prepare_swebench \
    --dataset verified \
    --split test \
    --total 80 \
    --repo-balanced \
    --output-dir examples/mini_swe/data/verified_subset_80 \
    --output-filename train.parquet

# Full 500 — for standalone eval or leaderboard submission
python -m examples.mini_swe.prepare.prepare_swebench \
    --dataset verified \
    --split test \
    --output-dir examples/mini_swe/data/swe_bench_verified
```

Other available datasets (`--dataset` values):

| Key | HuggingFace path | Split | Notes |
|-----|-----------------|-------|-------|
| `smith` | `SWE-bench/SWE-smith-py` | `train` | 50,908 Python bugs, pre-built images |
| `verified` | `SWE-bench/SWE-bench_Verified` | `test` | 500 human-verified issues |
| `lite` | `SWE-bench/SWE-bench_Lite` | `test` | 300 issues, common benchmark |
| `full` | `SWE-bench/SWE-bench` | `test` | 2,294 issues |

#### Subsampling and filtering

`prepare_swebench.py` (used in Step 1 and Step 2 above) also accepts two independent
filter/sample mechanisms:

- `--subset-spec`: Applied first. Accepts a Python slice (`"0:500"`) or a regex
  matched against each SWE problem's `instance_id` field (e.g. `"^django"`).
- `--total` + `--repo-balanced`: Applied after `--subset-spec`. Round-robins
  across repositories alphabetically until `--total` SWE problems are collected.
  `--per-repo-k` adds a hard cap per repository before round-robin.

Example — first 200 Django SWE problems only:

```bash
python -m examples.mini_swe.prepare.prepare_swebench \
    --dataset smith --split train \
    --subset-spec "^django" \
    --total 200 \
    --output-dir examples/mini_swe/data/django_200
```

### Step 3: Pre-fetch Docker images (shared-FS cache)

Each SWE-smith-py SWE problem uses a dedicated Docker image (~2–5 GB per
image, 3 GB median). Images must be reachable from **every cluster node** that
runs rollout workers. The recommended flow is:

1. Pull every unique image **once** to a shared-FS directory as a
   `docker-archive` tar (`prefetch_images.sh`, this step).
2. Fan `docker load` out to every node over pssh (`load_all_nodes.sh`,
   [Step 4](#step-4-fan-out-to-all-cluster-nodes)).

That way Docker Hub is hit once per image, not `num_nodes` times.

#### 3.1 — Why not just `docker pull`?

On many clusters the docker daemon itself cannot reach `registry-1.docker.io`
(user-space proxy env vars like `http_proxy` are **not** inherited by
`dockerd`). `prefetch_images.sh` sidesteps this by using **skopeo** — a
user-space tool that honours `$http_proxy` / `$https_proxy` from
`/jizhicfs/lhy/env/psrl.sh` and writes straight to a local `docker-archive`
tar (or directly into `dockerd` via the Unix socket). No daemon proxy config
needed.

#### 3.2 — Dry-run

```bash
source /jizhicfs/lhy/env/psrl.sh   # sets http_proxy for skopeo
bash examples/mini_swe/prepare/docker_scripts/prefetch_images.sh \
    --parquet examples/mini_swe/data/swe_smith_py_1k/train.parquet \
    --dry-run
```

Prints every unique `extra_info.sandbox_overrides.environment.image` referenced
by the parquet.

#### 3.3 — Full pull

```bash
source /jizhicfs/lhy/env/psrl.sh
bash examples/mini_swe/prepare/docker_scripts/prefetch_images.sh \
    --parquet examples/mini_swe/data/swe_smith_py_1k/train.parquet \
    --image-dir /jizhicfs/lhy/docker_images/swe \
    --workers 4 \
    --retries 5 \
    --mirrors docker.xuanyuan.me,docker.1ms.run,docker.1panel.live,hub.rat.dev,dockerproxy.net,proxy.vvvv.ee,docker.xiaogenban1993.com,lispy.org,registry.cyou \
    --no-direct-fallback
```

A reference invocation that chains the full pull + `docker load` fan-out lives
at `docker_scripts/prefetch_example.sh`.

What each flag does:

| Flag | Effect |
|------|--------|
| `--image-dir DIR` | Save each image as `docker-archive:<DIR>/<image>.tar`. Cached tars are reused on re-run. |
| `--load` | (optional) After each successful pull, also `docker load -i` into the local `dockerd`. Safe to skip if you plan to load later via [Step 4](#step-4-fan-out-to-all-cluster-nodes). |
| `--workers N` | Parallel pulls. `4` is sane; heavy networks can push to `8–16`. |
| `--retries N` | Per `(image, mirror)` retry count on transient failures (`unexpected EOF`, blob `404` from a mirror's cold cache). |
| `--mirrors a,b,c` | Ordered fallback list. A pull failing on `a` transparently retries on `b`, etc. Mirrors are applied via `apply_dockerhub_mirror` (same logic as `scripts/docker/docker_install.sh`). |
| `--no-direct-fallback` | Don't try `docker.io` directly after all mirrors fail — useful on clusters where `registry-1.docker.io` is firewalled. |
| `--force` | Ignore cache (even if the tar is structurally complete) and re-pull. |
| `--images FILE` / `--only A,B` | Alternatives to `--parquet` — feed a hand-written image list, one image per line (`#` comments OK). Useful for re-running a curated subset. |
| `--log-dir DIR` | Default is `<prepare>/_prefetch_logs/`, one `.log` per image. |

#### 3.4 — Integrity, idempotency, and the log directory

- Every tar is validated with `tar -tf | grep manifest.json` on every run
  (the `verify_docker_archive` helper, ~40 ms per 3 GB tar). Truncated/EOF'd
  tars are treated as **missing** and re-pulled automatically.
- Failed/aborted runs **do not** leave zombie tars: partial tars are
  `rm -f`'d before every retry, after every failed mirror, on SIGINT/SIGTERM
  (via `trap`), and again in the "all mirrors failed" branch.
- Every image gets a log file in `_prefetch_logs/`:
  - **Cached**: header `已经拥有了 <image>` + verification timestamp.
  - **Pulled**: per-mirror, per-attempt output (`----- attempt N/M -----`).
  - **Failed**: final attempt's fatal error + note pointing at `.log` /
    `.log.load`.

#### 3.5 — Mirror hygiene

Public Docker Hub mirrors come and go almost monthly. The list above is the
one that actually works as of 2026-04. Before trusting a new mirror,
probe it first:

```bash
source /jizhicfs/lhy/env/psrl.sh
bash examples/mini_swe/prepare/docker_scripts/probe_mirrors.sh \
    swebench/swesmith.x86_64.paramiko_1776_paramiko.23f92003
```

Output:

```
MIRROR                            RESULT    MESSAGE
docker.xuanyuan.me                OK        manifest reachable
docker.1ms.run                    OK        manifest reachable
docker.1panel.live                FAIL      manifest unknown: ...
hub.rat.dev                       TIMEOUT   > 25s
...
```

`probe_mirrors.sh` only does `skopeo inspect` (manifest-level probe, ~seconds
per mirror) — no blobs are downloaded. Edit the `MIRRORS=(...)` array inside
the script to add/remove candidates.

Known-dead / do-not-use mirrors (as of 2026-04):

- `dockerpull.org` — GFW-blocked since 2025-12.
- `docker.hlmirror.com` — now paywalls pulls behind `mirror.houlang.cloud`.
- `docker.m.daocloud.io` — allow-list only; `swebench/*` is **not** in it.
- `docker.imgdb.de`, `hub.docker.io`, `aicarbon.xyz`, …  — abandoned.

#### 3.6 — Retrying only the images that failed

After a run, failing images are easy to list from the log dir:

```bash
cd examples/mini_swe/prepare/_prefetch_logs
for f in *.log; do
    tail -n 1 "$f" | grep -q FATA && \
        echo "${f%.log}" | sed 's|__|/|; s|__|:|'
done > /tmp/failed_images.txt
```

Feed the list back into `prefetch_images.sh` via `--images` (same flags as the
full pull, just swap `--parquet` for `--images`):

```bash
source /jizhicfs/lhy/env/psrl.sh
bash examples/mini_swe/prepare/docker_scripts/prefetch_images.sh \
    --images /tmp/failed_images.txt \
    --image-dir /jizhicfs/lhy/docker_images/swe \
    --workers 8 \
    --retries 10 \
    --mirrors docker.xuanyuan.me,docker.1ms.run,docker.1panel.live,hub.rat.dev,dockerproxy.net,proxy.vvvv.ee,docker.xiaogenban1993.com,lispy.org,registry.cyou \
    --no-direct-fallback
```

Already-cached tars are skipped automatically via `verify_docker_archive`, so
retrying is cheap — only the genuinely missing/truncated ones get re-pulled.

> **Disk budget**: The 1k smith subset uses ~131 unique images, total
> ~330–500 GB of `docker-archive` tars on the shared FS. The 5k subset roughly
> doubles that. Plan `/jizhicfs/lhy/docker_images/swe` capacity accordingly,
> and on each node reserve ~1–2× that again for `/var/lib/docker` after
> `docker load`.

---

### Step 4: Fan out to all cluster nodes

Once every image has a complete tar in `/jizhicfs/lhy/docker_images/swe/`, use
`load_all_nodes.sh` to `docker load` them on every host **in parallel** over
`pssh`. Because `/jizhicfs/` is mounted on every node, tars are loaded
**directly from the shared path** — no `scp`/`rsync` copy stage.

```bash
bash examples/mini_swe/prepare/docker_scripts/load_all_nodes.sh \
    --hosts     /jizhicfs/lhy/hosts/32GPUs \
    --image-dir /jizhicfs/lhy/docker_images/swe
```

Key defaults and flags:

| Flag | Default | Effect |
|------|---------|--------|
| `--hosts FILE` | required | One IP (or IP:port) per line. `#` comments and blank lines ignored. |
| `--image-dir DIR` | required | Directory containing `*.tar` files (created by Step 3). |
| `--images-list FILE` | — | Roll out a subset. Accepts either image refs (`swebench/xxx:latest`) or tar basenames (`swebench__xxx`). |
| `--parallel-per-node N` | `2` | `xargs -P` concurrency **on each node**. `docker load` is I/O bound; 2–4 is the sweet spot. |
| `--skip-existing` / `--force` | skip | Before loading, `tar -xOf <tar> manifest.json` extracts the `RepoTags[0]`; if `docker image inspect <tag>` finds the image already present, the tar is skipped on that node. `--force` disables skipping. |
| `--timeout S` | `7200` | `pssh -t`. |
| `--user USER` | — | `pssh -l USER`. Uses your default ssh config if unset. |
| `--outdir DIR` | `_load_logs/<ts>/` | Per-host stdout / stderr collection directory. |
| `--dry-run` | off | Prints the planned command and the first 10 hosts/tars, does nothing. |

After the run, the script prints a per-host summary:

```
--- summary ---
  28.49.196.175         loaded=131  skipped=0    failed=0
  29.162.234.163        loaded=0    skipped=131  failed=0   # already had them
  28.49.37.141          loaded=130  skipped=0    failed=1
  29.162.224.113        loaded=131  skipped=0    failed=0
```

Full per-host output is in `_load_logs/<timestamp>/stdout/<ip>` and
`_load_logs/<timestamp>/stderr/<ip>`.

#### 4.1 — Rolling out only a new subset

After adding more training data (e.g. you went from 1k → 5k), compute the
delta and feed it to `--images-list`:

```bash
# Build a subset file with only the NEW images:
diff <(ls /jizhicfs/lhy/docker_images/swe/*.tar | xargs -n1 basename -s .tar | sort) \
     <(previous_deployed_list.txt) \
    | grep '^<' | sed 's/^< //' > /tmp/new_images.txt

bash examples/mini_swe/prepare/docker_scripts/load_all_nodes.sh \
    --hosts /jizhicfs/lhy/hosts/128GPUs \
    --image-dir /jizhicfs/lhy/docker_images/swe \
    --images-list /tmp/new_images.txt
```

Old images already on every node are untouched (thanks to
`--skip-existing`).

#### 4.2 — Operational notes

- **I/O planning**: 4 nodes × 2 concurrent loads × ~500 MB/s on the shared FS
  read side is already ~4 GB/s of NFS/Ceph read traffic. Tune
  `--parallel-per-node` down if the FS saturates.
- **Disk on each node**: `/var/lib/docker` needs room for *every* image
  you plan to use during training, not just the currently running ones. Run
  `docker system df` on one node post-load to sanity-check.
- **Skip logic uses tags, not digests**: If a remote mirror changed what
  `swebench/foo:latest` points to, `--skip-existing` will still skip. Use
  `--force` if you specifically need to refresh.
- **Partial failures**: If one node reports `failed=K`, you can re-run with
  `--hosts` pointing to just that node — `--skip-existing` makes the retry
  cheap:
  ```bash
  echo 28.49.37.141 > /tmp/one_host
  bash examples/mini_swe/prepare/docker_scripts/load_all_nodes.sh \
      --hosts /tmp/one_host --image-dir /jizhicfs/lhy/docker_images/swe \
      --parallel-per-node 4
  ```

Running `docker_scripts/prefetch_images.sh` (Step 3) + `docker_scripts/load_all_nodes.sh` (Step 4) on a
new cluster is the complete image-warmup path. Containers that can't pull
their image at rollout time produce a zero-reward episode and waste the
rollout slot, so validate once with `docker run --rm <a-sample-image> true`
on every host before kicking off training.

---

## Parquet schema

Each output row produced by `prepare_swebench.py` contains:

| Field | Type | Description |
|-------|------|-------------|
| `prompt` | `list[dict]` | Single `[{"role": "user", "content": problem_statement}]`. Agent templates are applied at runtime. |
| `data_source` | `str` | `"swe_smith_py"` or `"swebench_verified"`. Determines which reward branch fires in `reward.py`. |
| `ability` | `str` | Always `"software_engineering"`. |
| `reward_model.style` | `str` | `"swebench_test_exec"`. Signals test-execution reward path. |
| `reward_model.ground_truth.instance_id` | `str` | HuggingFace `instance_id` (e.g. `django__django-11039`). The dict key is kept as `instance_id` because it is consumed by the upstream swebench / swesmith harnesses. |
| `reward_model.ground_truth.FAIL_TO_PASS` | `list[str]` | Tests that must go from failing to passing. |
| `reward_model.ground_truth.PASS_TO_PASS` | `list[str]` | Tests that must continue passing. |
| `reward_model.ground_truth.gold_patch` | `str` | Reference patch (for offline analysis only; not used in RL reward). |
| `extra_info.swe_problem_id` | `str` | The SWE problem's HuggingFace `instance_id`, used for logging and grader correlation. |
| `extra_info.swe_problem` | `dict` | Full HuggingFace dataset row for this SWE problem. Passed to `grade_fresh_container` for `make_test_spec` / `get_test_cmd`. |
| `extra_info.swe_problem_image` | `str` | Docker image name for this SWE problem. |
| `extra_info.swe_restore_tests` | `bool` | `True` for SWE-smith-py (must run `git checkout HEAD~1` to restore F2P test files). `False` for Verified. |
| `extra_info.swe_grader` | `str` | `"swebench_fresh_container"`. Activates post-rollout fresh-container grading in the agent loop. |
| `extra_info.sandbox_overrides.environment.image` | `str` | Per-SWE-problem image injected into `MiniEnvironmentConfig` at rollout time. |
| `extra_info.sandbox_overrides.environment.cwd` | `str` | Always `"/testbed"` for real SWE problems. |
| `agent_name` | `str` | `"mini_swe_agent"`. Selects `MiniSWEAgentLoop` in the agent loop registry. |
