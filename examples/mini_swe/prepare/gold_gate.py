"""
Grading gate for a prepared SWE dataset.

Every row must be solvable by its own gold patch before the split is usable. The
gate grades the gold patch of each selected row with the same grader the model is
graded with, freezes the observed failures into the row, and fails when a row
still cannot resolve. Pruning the pass-to-pass entries the image cannot run is
what turns an environment-broken expectation into a usable sample, and
`gold_ceiling` records whether the row is usable at all.

Usage::

    python -m examples.mini_swe.prepare.gold_gate --parquet <val.parquet>
    python -m examples.mini_swe.prepare.gold_gate --parquet <val.parquet> --prune-p2p
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
from examples.mini_swe.grading.schema import GradingPlan, normalize_test_ids

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

GOLD_CEILING_KEY = "gold_ceiling"
PRUNED_P2P_KEY = "pruned_pass_to_pass"


def gold_ceiling(*, resolved_every_run: bool, flaky: bool) -> float:
    """
    Return the per-row ceiling: 1.0 only when every gold run resolves.
    """
    return 1.0 if resolved_every_run and not flaky else 0.0


def prunable_p2p(failed_by_run: list[list[str]]) -> list[str]:
    """
    Return the pass-to-pass tests that fail under gold in every run.

    A test that fails with the gold patch is an environment expectation the image
    cannot satisfy, so it can never be a regression signal. Requiring it to fail
    in every run keeps a single flaky observation from pruning a real test.
    """
    if not failed_by_run:
        return []
    common = set(failed_by_run[0])
    for failed in failed_by_run[1:]:
        common &= set(failed)
    return sorted(common)


def kept_p2p_failed(failed: list[str], pruned: list[str]) -> list[str]:
    """
    Return the observed pass-to-pass failures that pruning does not excuse.
    """
    pruned_set = set(pruned)
    return [test for test in failed if test not in pruned_set]


def run_resolves(run: dict[str, Any], pruned: list[str]) -> bool:
    """
    Return whether one gold run resolves once the pruned expectations are dropped.
    """
    if run.get("failure_reason"):
        return False
    return not run["f2p_failed"] and not kept_p2p_failed(run["p2p_failed"], pruned)


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Reduce the repeats of one instance into a gate verdict and a ceiling.
    """
    failed_by_run = [run["p2p_failed"] for run in runs]
    pruned = prunable_p2p(failed_by_run)
    resolved = [run_resolves(run, pruned) for run in runs]
    return {
        "repeats": len(runs),
        "runs": runs,
        "pruned_pass_to_pass": pruned,
        "resolved_runs": sum(resolved),
        "flaky": len(set(resolved)) > 1,
        "passed": all(resolved),
        "gold_ceiling": gold_ceiling(resolved_every_run=all(resolved), flaky=len(set(resolved)) > 1),
    }


def _grade_once(
    swe_problem: dict[str, Any],
    image: str,
    *,
    timeout: int,
    memory: str,
    attempt: int,
) -> dict[str, Any]:
    """
    Grade the gold patch of one row in a fresh container and scorecard it.
    """
    from examples.mini_swe.swebench_grader import grade_fresh_container

    instance_id = str(swe_problem.get("instance_id") or "unknown")
    started = time.monotonic()
    result = grade_fresh_container(
        swe_problem,
        str(swe_problem.get("patch") or ""),
        grader_kind="smith" if "swesmith" in image.lower() else "verified",
        image_name=image,
        timeout=timeout,
        swe_task_id=f"{instance_id}__gold{attempt}",
        memory=memory,
        grading_plan=GradingPlan.from_swe_problem(swe_problem),
    )
    return {
        "resolved": bool(result.get("resolved", False)),
        "failure_reason": result.get("failure_reason"),
        "f2p_pass": int(result.get("f2p_pass", 0)),
        "f2p_total": int(result.get("f2p_total", 0)),
        "p2p_pass": int(result.get("p2p_pass", 0)),
        "p2p_total": int(result.get("p2p_total", 0)),
        "f2p_failed": list(result.get("f2p_failed") or []),
        "p2p_failed": list(result.get("p2p_failed") or []),
        "elapsed_s": round(time.monotonic() - started, 1),
    }


def _grade_instance(
    swe_problem: dict[str, Any],
    image: str,
    *,
    repeats: int,
    timeout: int,
    memory: str,
) -> dict[str, Any]:
    """
    Grade one row `repeats` times, tolerating a raising run as a failed attempt.
    """
    runs: list[dict[str, Any]] = []
    for attempt in range(1, repeats + 1):
        try:
            runs.append(_grade_once(swe_problem, image, timeout=timeout, memory=memory, attempt=attempt))
        except Exception as exc:
            psrl_logger.exception(f"Gold grading raised for {swe_problem.get('instance_id')!r}.")
            expected_f2p = list(normalize_test_ids(swe_problem.get("FAIL_TO_PASS")))
            expected_p2p = list(normalize_test_ids(swe_problem.get("PASS_TO_PASS")))
            runs.append(
                {
                    "resolved": False,
                    "failure_reason": "grader_exception",
                    "error": f"{type(exc).__name__}: {exc}",
                    "f2p_pass": 0,
                    "f2p_total": len(expected_f2p),
                    "p2p_pass": 0,
                    "p2p_total": len(expected_p2p),
                    "f2p_failed": expected_f2p,
                    "p2p_failed": expected_p2p,
                    "elapsed_s": 0.0,
                }
            )
    summary = summarize_runs(runs)
    summary["instance_id"] = str(swe_problem.get("instance_id") or "unknown")
    summary["image"] = image
    return summary


def load_rows(parquet: Path, instances: str | None) -> list[tuple[int, dict[str, Any], str]]:
    """
    Return the (row index, swe problem, image) of every selected row.

    Args:
        parquet (Path): Prepared dataset to read.
        instances (str | None): Regular expression matched against `instance_id`.

    Returns:
        list[tuple[int, dict[str, Any], str]]: Selected rows in file order.

    Raises:
        ValueError: If a selected row has no gold patch, image, or eval script.
    """
    frame = pd.read_parquet(parquet)
    pattern = re.compile(instances) if instances else None
    selected: list[tuple[int, dict[str, Any], str]] = []
    for index, row in frame.iterrows():
        extra = row.get("extra_info") or {}
        problem = extra.get("swe_problem") or {}
        instance_id = str(problem.get("instance_id") or "")
        if pattern is not None and not pattern.search(instance_id):
            continue
        overrides = extra.get("sandbox_overrides") or {}
        image = str(extra.get("swe_problem_image") or (overrides.get("environment") or {}).get("image") or "")
        if not problem.get("patch"):
            raise ValueError(f"Row {instance_id!r} has no gold patch, so it cannot be gated.")
        if not image:
            raise ValueError(f"Row {instance_id!r} has no problem image, so it cannot be graded.")
        if not problem.get("eval_script"):
            raise ValueError(f"Row {instance_id!r} has no eval_script. Reprepare the dataset first.")
        selected.append((int(index), problem, image))
    if not selected:
        raise ValueError(f"No rows in {str(parquet)!r} matched instances={instances!r}.")
    return selected


def gate_split(
    parquet: Path,
    *,
    instances: str | None,
    repeats: int,
    workers: int,
    timeout: int,
    memory: str,
) -> dict[str, Any]:
    """
    Grade every selected row and return the report for the whole split.
    """
    rows = load_rows(parquet, instances)
    psrl_logger.info(f"Gating {len(rows)} rows from {parquet} with {workers} workers...")
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _grade_instance,
                problem,
                image,
                repeats=repeats,
                timeout=timeout,
                memory=memory,
            ): (index, problem.get("instance_id"))
            for index, problem, image in rows
        }
        results: list[dict[str, Any]] = []
        for future in futures:
            results.append(future.result())
    return {
        "parquet": str(parquet),
        "repeats": repeats,
        "elapsed_s": round(time.monotonic() - started, 1),
        "rows": results,
        "passed": all(item["passed"] for item in results),
        "gold_ceiling": sum(item["gold_ceiling"] for item in results) / len(results),
    }


def prune_frame(frame: pd.DataFrame, results: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Return `frame` with each gated row's frozen expectations and ceiling updated.
    """
    by_instance = {item["instance_id"]: item for item in results}
    frame = frame.copy()
    for index in frame.index:
        extra = copy.deepcopy(frame.at[index, "extra_info"]) or {}
        problem = extra.get("swe_problem")
        if not isinstance(problem, dict):
            continue
        verdict = by_instance.get(str(problem.get("instance_id") or ""))
        if verdict is None:
            continue
        # Pandas materializes a parquet list column as an array, so never test it
        # for truthiness before normalizing.
        expected = list(normalize_test_ids(problem.get("PASS_TO_PASS")))
        pruned = [test for test in verdict["pruned_pass_to_pass"] if test in set(expected)]
        if pruned:
            pruned_set = set(pruned)
            problem["PASS_TO_PASS"] = [test for test in expected if test not in pruned_set]
            problem[PRUNED_P2P_KEY] = sorted(set(normalize_test_ids(problem.get(PRUNED_P2P_KEY))) | pruned_set)
        problem[GOLD_CEILING_KEY] = verdict["gold_ceiling"]
        frame.at[index, "extra_info"] = extra
    return frame


def report_table(report: dict[str, Any]) -> str:
    """
    Render one line per gated row for the console.
    """
    lines = []
    for item in sorted(report["rows"], key=lambda row: row["instance_id"]):
        first = item["runs"][0]
        lines.append(
            f"{item['instance_id']:38} ceiling={item['gold_ceiling']:.1f} "
            f"runs={item['resolved_runs']}/{item['repeats']} flaky={str(item['flaky']):5} "
            f"f2p={first['f2p_pass']}/{first['f2p_total']} p2p={first['p2p_pass']}/{first['p2p_total']} "
            f"pruned={len(item['pruned_pass_to_pass'])} reason={first['failure_reason']}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate a prepared SWE split on its own gold patches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--instances", default=None, help="Regex filter on instance_id.")
    parser.add_argument("--repeats", type=int, default=1, help="Gold runs per row.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=900, help="Grader timeout in seconds.")
    parser.add_argument("--memory", default="", help="Grader container memory limit.")
    parser.add_argument("--report", default=None, type=Path)
    parser.add_argument("--prune-p2p", action="store_true", help="Rewrite the parquet with pruned P2P.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write the report or the parquet.")
    parser.add_argument("--allow-partial", action="store_true", help="Exit 0 even when a row fails.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = gate_split(
        args.parquet,
        instances=args.instances,
        repeats=args.repeats,
        workers=args.workers,
        timeout=args.timeout,
        memory=args.memory,
    )
    print(report_table(report))
    print(f"gold_ceiling={report['gold_ceiling']:.4f} passed={report['passed']} elapsed={report['elapsed_s']}s")
    needing_prune = [item["instance_id"] for item in report["rows"] if item["pruned_pass_to_pass"]]
    if needing_prune and not args.prune_p2p:
        psrl_logger.warning(
            f"{len(needing_prune)} rows resolve only after pruning pass-to-pass. "
            "Re-run with --prune-p2p to write the pruning into the parquet."
        )
    if args.dry_run:
        return 0 if report["passed"] or args.allow_partial else 1

    report_path = args.report or args.parquet.with_suffix(".gold_report.json")
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"Wrote {report_path}.")
    if args.prune_p2p:
        frame = prune_frame(pd.read_parquet(args.parquet), report["rows"])
        temporary = args.parquet.with_suffix(".parquet.tmp")
        frame.to_parquet(temporary)
        temporary.replace(args.parquet)
        print(f"Rewrote {args.parquet} with pruned pass-to-pass and gold_ceiling.")
    return 0 if report["passed"] or args.allow_partial else 1


if __name__ == "__main__":
    raise SystemExit(main())
