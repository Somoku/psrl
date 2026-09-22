"""
Retarget a prepared SWE split at repaired problem images.

A problem image can be missing a dependency its repository needs, and then the
gold patch cannot be graded on it. Repairing the image is only half the job: the
row still points at the broken original, so grading keeps failing. `--plan` lists
the images each selected instance needs, and `--overrides` applies the repaired
tags back to the parquet.

Usage::

    python -m examples.mini_swe.prepare.retarget_problem_images --parquet <val.parquet> --plan
    python -m examples.mini_swe.prepare.retarget_problem_images --parquet <val.parquet> --overrides <map.tsv>
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


def _row_problem(extra: dict[str, Any] | None) -> dict[str, Any]:
    problem = (extra or {}).get("swe_problem")
    return problem if isinstance(problem, dict) else {}


def _row_image(extra: dict[str, Any] | None) -> str:
    extra = extra or {}
    overrides = extra.get("sandbox_overrides") or {}
    return str(extra.get("swe_problem_image") or (overrides.get("environment") or {}).get("image") or "")


def plan_images(frame: pd.DataFrame, instances: str | None) -> list[tuple[str, str, list[str]]]:
    """
    Return the unique `(image, repo, instance_ids)` triples a repair must cover.

    Args:
        frame (pd.DataFrame): Prepared dataset with an `extra_info` column.
        instances (str | None): Regular expression matched against `instance_id`.

    Returns:
        list[tuple[str, str, list[str]]]: Triples sorted by image.

    Raises:
        ValueError: If one image is shared by rows from different repositories, or
            if a selected row carries no image.
    """
    pattern = re.compile(instances) if instances else None
    planned: dict[str, tuple[str, list[str]]] = {}
    for extra in frame["extra_info"]:
        problem = _row_problem(extra)
        instance_id = str(problem.get("instance_id") or "")
        if not instance_id or (pattern is not None and not pattern.search(instance_id)):
            continue
        image = _row_image(extra)
        repo = str(problem.get("repo") or "")
        if not image:
            raise ValueError(f"Row {instance_id!r} has no problem image to repair.")
        known_repo, instance_ids = planned.get(image, (repo, []))
        if repo and known_repo and repo != known_repo:
            raise ValueError(f"Image {image!r} is shared by {known_repo!r} and {repo!r}.")
        planned[image] = (known_repo or repo, [*instance_ids, instance_id])
    return sorted((image, repo, sorted(ids)) for image, (repo, ids) in planned.items())


def parse_overrides(path: Path) -> dict[str, str]:
    """
    Read a `base_image<TAB>repaired_image` map, skipping blanks and comments.
    """
    overrides: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.split("#", 1)[0].strip()
        if not stripped:
            continue
        parts = stripped.split("\t")
        if len(parts) != 2 or not all(part.strip() for part in parts):
            raise ValueError(f"Override line {number} must be 'base_image<TAB>repaired_image'.")
        overrides[parts[0].strip()] = parts[1].strip()
    if not overrides:
        raise ValueError(f"Override map {str(path)!r} is empty.")
    return overrides


def apply_overrides(
    frame: pd.DataFrame,
    overrides: dict[str, str],
    *,
    include_rollout_image: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Return `frame` pointed at the repaired images, plus what changed.

    The grader reads `swe_problem_image`, while the rollout reads
    `sandbox_overrides.environment.image`. Grading is what a broken dependency
    blocks, so only the grader image is retargeted unless asked otherwise.

    Args:
        frame (pd.DataFrame): Prepared dataset with an `extra_info` column.
        overrides (dict[str, str]): Base image to repaired image mapping.
        include_rollout_image (bool): Also retarget the rollout image.

    Returns:
        tuple[pd.DataFrame, dict[str, Any]]: The rewritten frame and a summary.
    """
    frame = frame.copy()
    summary: dict[str, Any] = {"rows": 0, "rollout_rows": 0, "applied": sorted(set(overrides.values()))}
    for index in frame.index:
        extra = copy.deepcopy(frame.at[index, "extra_info"]) or {}
        problem = _row_problem(extra)
        if not problem:
            continue
        repaired = overrides.get(_row_image(extra))
        if repaired is None:
            continue
        extra["swe_problem_image"] = repaired
        summary["rows"] += 1
        if include_rollout_image:
            environment = extra.setdefault("sandbox_overrides", {}).setdefault("environment", {})
            environment["image"] = repaired
            summary["rollout_rows"] += 1
        frame.at[index, "extra_info"] = extra
    return frame, summary


def write_frame(frame: pd.DataFrame, parquet: Path) -> None:
    """
    Write `frame` back to `parquet` through a temporary file.
    """
    temporary = parquet.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary)
    temporary.replace(parquet)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Point a prepared SWE split at repaired problem images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--instances", default=None, help="Regex filter on instance_id.")
    parser.add_argument("--plan", action="store_true", help="Print image, repo, and instances.")
    parser.add_argument("--overrides", type=Path, default=None, help="Override map to apply.")
    parser.add_argument("--include-rollout-image", action="store_true", help="Retarget the rollout image too.")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = pd.read_parquet(args.parquet)
    if args.plan:
        for image, repo, instance_ids in plan_images(frame, args.instances):
            print(f"{image}\t{repo}\t{','.join(instance_ids)}")
        return 0
    if args.overrides is None:
        raise SystemExit("Pass --plan to list images or --overrides to apply a repair map.")

    overrides = parse_overrides(args.overrides)
    updated, summary = apply_overrides(frame, overrides, include_rollout_image=args.include_rollout_image)
    print(
        f"Retargeted {summary['rows']} rows at {len(summary['applied'])} repaired images "
        f"(rollout image rows: {summary['rollout_rows']})."
    )
    unmatched = sorted(set(overrides) - {_row_image(extra) for extra in frame["extra_info"]})
    if unmatched:
        psrl_logger.warning(f"Override map has {len(unmatched)} base images absent from the split.")
    if args.dry_run:
        return 0
    if not summary["rows"]:
        raise SystemExit("No row matched the override map, so nothing was written.")
    write_frame(updated, args.parquet)
    print(f"Rewrote {args.parquet}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
