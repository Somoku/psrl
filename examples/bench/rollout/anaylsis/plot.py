#!/usr/bin/env python3
"""
plot_jsonl_simple.py

Simple: read a .jsonl file line-by-line, call a user-defined "processor" for each JSON object,
collect kept rows and plot multiple labeled time series.

REQUIREMENT: user must provide a processor function with signature:
    processor(obj: dict) -> (keep: bool, values: dict[str, float], x_value)

- keep: whether to keep this row
- values: mapping label -> numeric value for this row
- x_value: x-axis value for this row (can be number or datetime)
Note: this script preserves the original file order; it does NOT sort by x.
"""

from typing import Dict, Any, Tuple, Callable, Iterable, List
import json
from datetime import datetime
import math
import matplotlib.pyplot as plt
import argparse
import os

from processor import make_intertoken_indexed_processor, make_prompt_time_indexed_processor, make_generation_time_indexed_processor, Processor


def read_jsonl_lines(path: str) -> Iterable[Dict[str, Any]]:
    """Yield parsed JSON objects from a jsonl file, skipping empty/invalid lines."""
    with open(path, "r", encoding="utf-8") as f:
        for i, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except Exception as e:
                print(f"[warn] skip line {i}: invalid json ({e})")
                raise e


def collect_by_processor(path: str, processor: Processor) -> Dict[str, List[Tuple[Any, float]]]:
    """
    Iterate file, call processor for each JSON object.
    Return: dict[label] -> list of (x_value, y_value) preserving input order.
    """
    series: Dict[str, List[Tuple[Any, float]]] = {}
    for obj in read_jsonl_lines(path):
        try:
            keep, values, x = processor(obj)
        except Exception as e:
            print(f"[warn] processor raised error; skipping line: {e}")
            raise e
            continue

        if not keep:
            continue
        if not isinstance(values, dict):
            print("[warn] processor returned non-dict values; skipping line")
            continue

        for label, raw_v in values.items():
            try:
                v = float(raw_v)
                if math.isfinite(v):
                    series.setdefault(label, []).append((x, v))
            except Exception:
                # skip non-numeric label/value for this row
                pass

    return series


def plot_series(series: Dict[str, List[Tuple[Any, float]]],
                title: str = "Plot",
                xlabel: str = "x",
                ylabel: str = "value",
                out_png: str = None,
                show: bool = True):
    """
    Plot each label's points in the order they were collected (no sorting).
    If x-values are datetime objects, format the x-axis accordingly.
    """
    plt.figure(figsize=(10, 6))
    any_datetime = False

    for label, points in series.items():
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        if isinstance(xs[0], datetime):
            any_datetime = True
        plt.scatter(xs, ys, marker='.', label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend(loc="best")

    if any_datetime:
        try:
            import matplotlib.dates as mdates
            plt.gcf().autofmt_xdate()
        except Exception:
            pass

    if out_png:
        plt.savefig(out_png, bbox_inches="tight")
        print(f"Saved plot to {out_png}")
    if show:
        plt.show()
    plt.close()


# -------------------------
# Example processor (editable)
# -------------------------
#
# This example processor implements the required signature:
#   processor(obj) -> (keep: bool, values: dict, x_value)
#
# It:
#  - uses 'elapsed_time' as x
#  - extracts 'throughput_stats.total_throughput' and 'scheduler_stats.kv_cache_usage'
#  - keeps every row (return keep=True). Modify as needed.
#

def _get_by_dotted(d: Dict[str, Any], path: str, default=None):
    """Get nested value by dotted path (e.g. 'throughput_stats.total_throughput')."""
    cur = d
    for part in path.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def processor_example(obj: Dict[str, Any]) -> Tuple[bool, Dict[str, float], Any]:
    """
    Example processor:
    - keep every row
    - x_value = obj['elapsed_time'] (numeric). If missing, fall back to timestamp parsed as datetime.
    - returns two labels: 'total_throughput' and 'kv_cache_usage' (floats) if available.
    """
    # choose x
    x = obj.get("elapsed_time", None)
    if x is None:
        ts = obj.get("timestamp", None)
        if isinstance(ts, str):
            try:
                x = datetime.fromisoformat(ts)
            except Exception:
                x = ts  # keep raw string if can't parse
        else:
            x = None

    values: Dict[str, float] = {}
    ttp = _get_by_dotted(obj, "throughput_stats.total_throughput", None)
    if ttp is not None:
        try:
            values["total_throughput"] = float(ttp)
        except Exception:
            pass
    kv = _get_by_dotted(obj, "scheduler_stats.kv_cache_usage", None)
    if kv is not None:
        try:
            values["kv_cache_usage"] = float(kv)
        except Exception:
            pass

    # keep only if we have at least one numeric value
    keep = len(values) > 0
    return keep, values, x


# -------------------------
# Main (CLI)
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Simple jsonl -> multi-line plot using a user processor.")
    ap.add_argument("jsonl", help=".jsonl input path")
    ap.add_argument("--out", help="output PNG path", default="output.png")
    ap.add_argument("--title", help="plot title", default="Plot")
    ap.add_argument("--xlabel", help="x axis label", default="x")
    ap.add_argument("--ylabel", help="y axis label", default="value")
    args = ap.parse_args()

    if not os.path.exists(args.jsonl):
        print("jsonl file not found:", args.jsonl)
        return

    # Replace `processor_example` with your own `processor` function if desired.
    # processor = processor_example
    processor = make_intertoken_indexed_processor()
    processor = make_generation_time_indexed_processor()
    processor = make_prompt_time_indexed_processor()

    series = collect_by_processor(args.jsonl, processor)
    if not series:
        print("No data collected with current processor.")
        return

    plot_series(series, title=args.title, xlabel=args.xlabel, ylabel=args.ylabel, out_png=args.out, show=True)


if __name__ == "__main__":
    main()
