#!/usr/bin/env python3
"""Per-step GRPO metrics from two ablation logs, side by side.

    compare_ablation.py <arm-a-log> <arm-b-log> [--csv out.csv]

Each log is a driver log (`tee` of the example script). Training metrics come from the
`step:N - ...` lines and validation metrics ride along on the same line, because the
step strategy merges them into the metrics it logs for that step.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

#: Reported for every step in both arms, with the reason each one matters.
TRAIN_KEYS = (
    "critic/score/mean",
    "critic/score/max",
    "critic/rewards/mean",
    "critic/advantages/mean",
    "actor/entropy",
    "actor/pg_loss",
    "actor/grad_norm",
    "response_length/mean",
    "response_length/clip_ratio",
    "global_seqlen/mean",
    "perf/mfu/actor",
)

# `(?<!global_)` keeps `global_step:N` out; that field is a payload, not a header.
_STEP_RE = re.compile(r"(?<!global_)step:(\d+) - (.*)")
_KV_RE = re.compile(r"([A-Za-z0-9_/]+):(-?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)")


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


def parse(path: Path) -> tuple[dict[int, dict[str, str]], dict[int, dict[str, str]]]:
    """Return (train metrics by step, validation metrics by step)."""
    train: dict[int, dict[str, str]] = {}
    val: dict[int, dict[str, str]] = {}
    text = _strip_ansi(path.read_text(errors="replace"))
    for match in _STEP_RE.finditer(text):
        step = int(match.group(1))
        body = match.group(2)
        # A validation run appends its metrics to the same step's line.
        train[step] = {}
        val[step] = {}
        for key, value in _KV_RE.findall(body):
            if key.startswith("val-"):
                val[step][key] = value
            else:
                train[step][key] = value
    return train, val


def _num(raw: str | None) -> float | None:
    return float(raw) if raw is not None else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--csv")
    args = ap.parse_args()

    train_a, val_a = parse(Path(args.a))
    train_b, val_b = parse(Path(args.b))
    steps = sorted(set(train_a) | set(train_b))

    print(f"steps found: A={sorted(train_a)} B={sorted(train_b)}\n")
    header = f"{'metric':<32}{'step':>6}{'A':>16}{'B':>16}{'delta':>16}"
    print(header)
    print("-" * len(header))
    for key in TRAIN_KEYS:
        for step in steps:
            va = _num(train_a.get(step, {}).get(key))
            vb = _num(train_b.get(step, {}).get(key))
            if va is None and vb is None:
                continue
            delta = "" if va is None or vb is None else f"{vb - va:+.6g}"
            fa = "" if va is None else f"{va:.6g}"
            fb = "" if vb is None else f"{vb:.6g}"
            print(f"{key:<32}{step:>6}{fa:>16}{fb:>16}{delta:>16}")
        print()

    print("validation")
    print("-" * len(header))
    val_keys = sorted({k for d in (*val_a.values(), *val_b.values()) for k in d})
    for key in val_keys:
        for step in steps:
            va = _num(val_a.get(step, {}).get(key))
            vb = _num(val_b.get(step, {}).get(key))
            if va is None and vb is None:
                continue
            delta = "" if va is None or vb is None else f"{vb - va:+.6g}"
            fa = "" if va is None else f"{va:.6g}"
            fb = "" if vb is None else f"{vb:.6g}"
            print(f"{key:<32}{step:>6}{fa:>16}{fb:>16}{delta:>16}")

    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["step", "metric", "A", "B"])
            for key in TRAIN_KEYS:
                for step in steps:
                    writer.writerow(
                        [step, key, train_a.get(step, {}).get(key, ""), train_b.get(step, {}).get(key, "")]
                    )
            for key in val_keys:
                for step in steps:
                    writer.writerow([step, key, val_a.get(step, {}).get(key, ""), val_b.get(step, {}).get(key, "")])
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
