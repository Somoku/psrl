# SWE Data Preparation

Each training path requires converting raw datasets into PSRL training parquets and warming up Docker image caches on all cluster nodes.

---

## Data Paths Summary

| Path | Dataset | Script | Docker Images | Disk Budget |
|------|---------|--------|---------------|-------------|
| **Toy** | `simple_cases_train.json` (40 tasks) | `prepare_simple_data.py` | `python:3.11-slim` (baked as `psrl-mini-swe:latest`) | Minimal |
| **SWE-smith** | [SWE-bench/SWE-smith-py](https://huggingface.co/datasets/SWE-bench/SWE-smith-py) (50k) | `prepare_swebench.py` | `swebench/swesmith.x86_64.*` (~3 GB each) | 500 GB-1 TB shared FS |
| **SWE-Gym** | [SWE-Gym/SWE-Gym](https://huggingface.co/datasets/SWE-Gym/SWE-Gym) (2438) | `prepare_swe_gym.py` | `xingyaoww/sweb.eval.x86_64.*` | 500-800 GB shared FS |

---

## Path A: Toy Dataset

Quick iteration and smoke tests. All bugs are synthetic and baked into a single Docker image.

```bash
# 1. Bake repos into Docker image
bash examples/mini_swe/prepare/docker_scripts/bake_simple_repos.sh python:3.11-slim

# 2. Generate parquets
python examples/mini_swe/prepare/prepare_simple_data.py \
    --mode simple --train_size 64 --test_size 16 \
    --output_dir examples/mini_swe/data/mini_swe_agent
```

---

## Path B: SWE-smith-py

Real-world GitHub bugs with per-problem Docker images. Supports repo-balanced subsampling.

```bash
# 1. Generate training parquet (1k subset)
python -m examples.mini_swe.prepare.prepare_swebench \
    --dataset smith --split train \
    --total 1000 --per-repo-k 10 \
    --output-dir examples/mini_swe/data/swe_smith_py_1k

# 2. Generate validation parquet
python -m examples.mini_swe.prepare.prepare_swebench \
    --dataset verified --split test \
    --total 80 --repo-balanced \
    --output-dir examples/mini_swe/data/verified_subset_80 \
    --output-filename train.parquet

# 3. Pre-fetch Docker images to shared FS
bash examples/mini_swe/prepare/docker_scripts/prefetch_images.sh \
    --parquet examples/mini_swe/data/swe_smith_py_1k/train.parquet \
    --image-dir /jizhicfs/lhy/docker_images/swe --workers 4

# 4. Fan out to all cluster nodes
bash examples/mini_swe/prepare/docker_scripts/load_all_nodes.sh \
    --hosts /jizhicfs/lhy/hosts/32GPUs \
    --image-dir /jizhicfs/lhy/docker_images/swe
```

---

## Path C: SWE-Gym

Real-world bugs with pre-computed eval scripts. No `git checkout HEAD~1` needed.

```bash
# Quick start (100 instances, no Fork dependency)
python -m examples.mini_swe.prepare.prepare_swe_gym \
    --dataset gym-subset \
    --output-dir examples/mini_swe/data/swe_gym_subset_100

# Full dataset (2438 instances, requires SWE-Bench-Fork 2.0.13)
python -m examples.mini_swe.prepare.prepare_swe_gym \
    --dataset gym \
    --output-dir examples/mini_swe/data/swe_gym_2438

# Pre-fetch images
bash examples/mini_swe/prepare/docker_scripts/swe_gym.sh
```

---

## Docker Image Workflow

The two-step workflow avoids hitting Docker Hub `N × nodes` times:

1. **Prefetch** (once): `prefetch_images.sh` pulls each unique image via `skopeo` to a shared-FS tar.
2. **Fan-out** (per cluster): `load_all_nodes.sh` does `docker load` on every node via `pssh`.

Both steps are idempotent, already-cached tars and already-loaded images are skipped automatically.

---

```{seealso}
Full instructions including mirror configuration, retry strategies, and disk planning: [`examples/mini_swe/prepare/README.md`](https://github.com/lhy101/psrl/blob/main/examples/mini_swe/prepare/README.md)
```
