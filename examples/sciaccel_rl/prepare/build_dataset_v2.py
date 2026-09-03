"""
Build stratified SciAccel v2 datasets from taxonomy metadata.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import tomllib

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))
if not psrl_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    psrl_logger.addHandler(_handler)
    psrl_logger.propagate = False

# Map each category to its verifier training reward.
_CATEGORY_REWARD_KEYS = {
    "repair": "reward_repair",
    "implementation": "reward_repair",
    "acceleration": "reward",
}

# The CUDA acceleration task uses its telemetry-gated reward.
_TASK_REWARD_KEY_OVERRIDES = {
    "laps-accel-cuda": "reward_gpu",
}

# The two acceleration tasks predate the taxonomy and carry no
# `[metadata.taxonomy]` section, so their group labels are assigned here.
_ACCELERATION_FAMILY = "accel"
_ACCELERATION_TREE = "both"

DATA_SOURCE = "sciaccel_rl"


def _load_canonical_rows(repo: Path, env: str) -> dict[str, dict[str, Any]]:
    """
    Load `tasks.jsonl` and index the canonical rows by task name.

    Only generated tasks (repair, implementation) have rows. The hand-authored
    acceleration tasks are absent. The rows carry the in-situ measured floor and
    the graded check list, which the compiled `task.toml` only summarizes.

    Args:
        repo (Path): sciaccel-rl repository root.
        env (str): Environment directory name under `envs/`.

    Returns:
        dict[str, dict[str, Any]]: Task name (unprefixed) to canonical row.
    """
    jsonl_path = repo / "envs" / env / "tasks.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Canonical rows not found: {jsonl_path}")

    rows: dict[str, dict[str, Any]] = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[row["task"]] = row
    psrl_logger.info(f"Loaded canonical rows from {jsonl_path!s}. Count: {len(rows)}.")
    return rows


def _discover_task_dirs(repo: Path, env: str, categories: list[str] | None) -> list[tuple[str, Path]]:
    """
    Find every compiled task directory, as (category, path) pairs.

    A compiled task directory is one containing a `task.toml`. Planned-but-empty
    category directories hold only a `.gitkeep` and are skipped.

    Args:
        repo (Path): sciaccel-rl repository root.
        env (str): Environment directory name under `envs/`.
        categories (list[str] | None): Categories to include. None means every
            category that has at least one compiled task.

    Returns:
        list[tuple[str, Path]]: Sorted (category, task_dir) pairs.
    """
    tasks_root = repo / "envs" / env / "tasks"
    if not tasks_root.is_dir():
        raise FileNotFoundError(f"Tasks root not found: {tasks_root}")

    found: list[tuple[str, Path]] = []
    for category_dir in sorted(p for p in tasks_root.iterdir() if p.is_dir()):
        category = category_dir.name
        if categories is not None and category not in categories:
            continue
        task_dirs = sorted(p for p in category_dir.iterdir() if p.is_dir() and (p / "task.toml").exists())
        if not task_dirs:
            psrl_logger.info(f"No compiled tasks in category={category!r}. Skipping.")
            continue
        found.extend((category, task_dir) for task_dir in task_dirs)

    if categories is not None:
        missing = set(categories) - {category for category, _ in found}
        if missing:
            raise ValueError(f"Requested categories have no compiled tasks: {sorted(missing)}.")
    return found


def _build_row(
    category: str,
    task_dir: Path,
    canonical_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Build one dataset row from a compiled task directory.

    The manifest is parsed with `tomllib` rather than scanned line by line: the
    v2 `task.toml` has several sections carrying a `name`-like key, so a textual
    scan picks up the wrong one.

    Args:
        category (str): Taxonomy category, from the parent directory name.
        task_dir (Path): Compiled task directory containing `task.toml`.
        canonical_rows (dict[str, dict[str, Any]]): Output of
            `_load_canonical_rows`, keyed by unprefixed task name.

    Returns:
        dict[str, Any]: One row for the output Parquet.
    """
    manifest = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
    instruction = (task_dir / "instruction.md").read_text(encoding="utf-8").strip()

    taxonomy = manifest.get("metadata", {}).get("taxonomy", {})
    environment = manifest.get("environment", {})
    task_name = manifest["task"]["name"]
    canonical = canonical_rows.get(task_dir.name, {})

    # Manifest and directory categories must agree.
    manifest_category = taxonomy.get("category")
    if manifest_category is not None and manifest_category != category:
        raise ValueError(
            f"Task {task_dir.name!r} sits under category {category!r} but its manifest "
            f"declares {manifest_category!r}. Recompile the task tree."
        )

    reward_key = _TASK_REWARD_KEY_OVERRIDES.get(task_dir.name)
    if reward_key is None:
        if category not in _CATEGORY_REWARD_KEYS:
            raise ValueError(
                f"No reward key known for category {category!r} (task {task_dir.name!r}). "
                f"Add it to _CATEGORY_REWARD_KEYS."
            )
        reward_key = _CATEGORY_REWARD_KEYS[category]

    # Prefer the funnel's in-situ measured floor over the manifest's
    # `floor_native_estimate`, which the task factory documents as advisory.
    floor = float(canonical.get("funnel", {}).get("floor", taxonomy.get("floor_native_estimate", 0.0)))

    # Resolve checks from canonical data, manifest metadata, then the checks directory.
    checks = list(canonical.get("checks") or taxonomy.get("affected_checks") or [])
    if not checks:
        checks_dir = task_dir / "environment" / "checks"
        checks = sorted(p.name for p in checks_dir.iterdir() if p.is_dir()) if checks_dir.is_dir() else []

    family = taxonomy.get("family", _ACCELERATION_FAMILY if category == "acceleration" else "")
    tree = taxonomy.get("tree", _ACCELERATION_TREE if category == "acceleration" else "")

    extra_info = {
        "task_path": str(task_dir.resolve()),
        "reward_key": reward_key,
        "task_name": task_name,
        "timeout_sec": float(manifest.get("agent", {}).get("timeout_sec", 3600.0)),
        "gpus": int(environment.get("gpus", 0)),
        "category": category,
        "family": family,
        "tree": tree,
        "mode": taxonomy.get("mode", ""),
        "floor": floor,
        "checks": checks,
        "network_mode": environment.get("network_mode", ""),
    }

    return {
        "prompt": [{"role": "user", "content": instruction}],
        "data_source": DATA_SOURCE,
        # Harbor supplies verifier reward while the schema still requires ground truth.
        "reward_model": {"style": "rule", "ground_truth": ""},
        "task_name": task_name,
        "category": category,
        "family": family,
        "tree": tree,
        "floor": floor,
        "extra_info": extra_info,
    }


def _stratified_val_names(df: pd.DataFrame, per_group: int) -> list[str]:
    """
    Pick validation tasks by taking the first `per_group` of every group.

    Grouping is by (category, family, tree), which is the sampling unit the
    sciaccel-rl README prescribes: same-family tasks share a debugging shape, so
    uniform sampling over-weights the large families (57 of 99 repair tasks are
    sign flips). Selection is by sorted task name so the split is reproducible
    without a random seed.

    Args:
        df (pd.DataFrame): The full dataset.
        per_group (int): Tasks to take from each group.

    Returns:
        list[str]: Sorted validation task names.
    """
    val_names: list[str] = []
    for _, group in df.groupby(["category", "family", "tree"], sort=True):
        val_names.extend(sorted(group["task_name"])[:per_group])
    return sorted(val_names)


def _build_stats(df: pd.DataFrame) -> dict[str, Any]:
    """
    Summarize task counts and floor ranges per group, for curriculum planning.

    Args:
        df (pd.DataFrame): The full dataset.

    Returns:
        dict[str, Any]: Totals plus per-category and per-group breakdowns.
    """
    by_group: dict[str, dict[str, Any]] = {}
    for (category, family, tree), group in df.groupby(["category", "family", "tree"], sort=True):
        floors = group["floor"].tolist()
        by_group[f"{category}/{family}/{tree}"] = {
            "n_tasks": len(group),
            "floor_min": round(min(floors), 6),
            "floor_max": round(max(floors), 6),
            "floor_mean": round(sum(floors) / len(floors), 6),
            "reward_key": sorted({row["reward_key"] for row in group["extra_info"]}),
        }

    return {
        "n_tasks": len(df),
        "by_category": {k: int(v) for k, v in df["category"].value_counts().sort_index().items()},
        "by_reward_key": {
            k: int(v)
            for k, v in pd.Series([row["reward_key"] for row in df["extra_info"]]).value_counts().sort_index().items()
        },
        "by_group": by_group,
    }


def build_datasets(
    repo_path: str,
    out_dir: str,
    env: str = "laps",
    categories: list[str] | None = None,
    val_per_group: int = 1,
) -> dict[str, Any]:
    """
    Build `all.parquet`, the train/val split, and the accompanying manifests.

    Args:
        repo_path (str): Path to the sciaccel-rl repository root.
        out_dir (str): Directory to write the artefacts into.
        env (str): Environment directory name under `envs/`.
        categories (list[str] | None): Categories to include. None means every
            category with at least one compiled task.
        val_per_group (int): Tasks to hold out per (category, family, tree) group.

    Returns:
        dict[str, Any]: The stats dict also written to `stats.json`.
    """
    repo = Path(repo_path).resolve()
    canonical_rows = _load_canonical_rows(repo, env)
    task_dirs = _discover_task_dirs(repo, env, categories)
    psrl_logger.info(f"Discovered compiled tasks under {repo / 'envs' / env / 'tasks'!s}. Count: {len(task_dirs)}.")

    counts: dict[str, int] = defaultdict(int)
    rows: list[dict[str, Any]] = []
    for category, task_dir in task_dirs:
        rows.append(_build_row(category, task_dir, canonical_rows))
        counts[category] += 1

    df = pd.DataFrame(rows)
    if df["task_name"].duplicated().any():
        duplicates = sorted(df.loc[df["task_name"].duplicated(), "task_name"])
        raise ValueError(f"Duplicate task names in the dataset: {duplicates}.")

    val_names = _stratified_val_names(df, val_per_group)
    val_df = df[df["task_name"].isin(val_names)].reset_index(drop=True)
    train_df = df[~df["task_name"].isin(val_names)].reset_index(drop=True)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "all.parquet", index=False)
    train_df.to_parquet(out / "train.parquet", index=False)
    val_df.to_parquet(out / "val.parquet", index=False)

    (out / "split.json").write_text(
        json.dumps(
            {
                "val_per_group": val_per_group,
                "group_key": ["category", "family", "tree"],
                "val": val_names,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    stats = _build_stats(df)
    stats["repo"] = str(repo)
    stats["env"] = env
    stats["n_train"] = len(train_df)
    stats["n_val"] = len(val_df)
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    psrl_logger.info(
        f"Wrote dataset={out / 'all.parquet'!s}. Total tasks: {len(df)}. "
        f"Train: {len(train_df)}. Validation: {len(val_df)}. Categories: {dict(counts)!r}."
    )
    return stats


def main() -> None:
    """
    CLI entry point.
    """
    parser = argparse.ArgumentParser(description="Build SciAccel-RL v2 datasets.")
    parser.add_argument("--repo", required=True, help="Path to the sciaccel-rl repo root.")
    parser.add_argument("--out-dir", required=True, help="Directory for the output artefacts.")
    parser.add_argument("--env", default="laps", help="Environment directory name under envs/.")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Taxonomy categories to include. Defaults to every non-empty category.",
    )
    parser.add_argument(
        "--val-per-group",
        type=int,
        default=1,
        help="Tasks held out per (category, family, tree) group.",
    )
    args = parser.parse_args()

    stats = build_datasets(
        repo_path=args.repo,
        out_dir=args.out_dir,
        env=args.env,
        categories=args.categories,
        val_per_group=args.val_per_group,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
