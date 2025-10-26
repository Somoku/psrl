#!/usr/bin/env python3
"""
Scan a folder (recursively) for .log files, extract batch size from filenames (matching B<digits>),
parse the metrics total_preempts, avg_prompt_throughput, avg_generation_throughput from each file,
aggregate values per batch (mean if multiple files for the same batch), and plot three subplots
(one per metric) with batch size on the x-axis.

Usage:
    python plot_summary.py /path/to/log/folder --out out.png

The script is robust to missing metrics and attempts UTF-8 then latin1 decoding when needed.
"""

import os
import re
import argparse
from collections import defaultdict
import math

import numpy as np
import matplotlib.pyplot as plt

# Regular expression to extract batch size from filename: B<digits>
RE_BATCH = re.compile(r'B(\d+)', re.IGNORECASE)

# Regular expressions for metric extraction (supports integer, float, scientific notation)
METRIC_KEYS = {
    'total_preempts': re.compile(r'total_preempts\s*:\s*([+-]?\d+)', re.IGNORECASE),
    'avg_prompt_throughput': re.compile(r'avg_prompt_throughput\s*:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', re.IGNORECASE),
    'avg_generation_throughput': re.compile(r'avg_generation_throughput\s*:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', re.IGNORECASE),
}


def parse_metrics_from_text(text):
    """
    Parse the three target metrics from file text.
    Returns a dict mapping metric -> float (or None if not found).
    """
    found = {}
    for k, pattern in METRIC_KEYS.items():
        m = pattern.search(text)
        if m:
            try:
                found[k] = float(m.group(1))
            except ValueError:
                found[k] = None
        else:
            found[k] = None
    return found


def scan_folder(folder):
    """
    Recursively scan `folder` for .log files. Files without a B<digits> token in the filename
    are skipped. Returns a dict: batch_size (int) -> list of metric dicts (one per file).
    """
    results = defaultdict(list)
    for root, dirs, files in os.walk(folder):
        for fn in files:
            if not fn.lower().endswith('.log'):
                continue
            m_batch = RE_BATCH.search(fn)
            if not m_batch:
                # skip files that don't contain a B<digits> token
                continue
            batch = int(m_batch.group(1))
            path = os.path.join(root, fn)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    txt = f.read()
            except UnicodeDecodeError:
                # fallback encoding
                with open(path, 'r', encoding='latin1') as f:
                    txt = f.read()
            metrics = parse_metrics_from_text(txt)
            metrics['_file'] = path
            results[batch].append(metrics)
    return results


def aggregate_results(results):
    """
    Aggregate per-batch lists of metric dicts into averaged values.
    Returns three dicts: batch -> mean_value (or None) for each metric, plus a summary dict.
    """
    batches = sorted(results.keys())
    agg_total_preempts = {}
    agg_avg_prompt = {}
    agg_avg_generation = {}
    summary = {}

    for b in batches:
        items = results[b]
        vals_preempts = [it['total_preempts'] for it in items if it['total_preempts'] is not None]
        vals_prompt = [it['avg_prompt_throughput'] for it in items if it['avg_prompt_throughput'] is not None]
        vals_gen = [it['avg_generation_throughput'] for it in items if it['avg_generation_throughput'] is not None]

        agg_total_preempts[b] = np.mean(vals_preempts) if vals_preempts else None
        agg_avg_prompt[b] = np.mean(vals_prompt) if vals_prompt else None
        agg_avg_generation[b] = np.mean(vals_gen) if vals_gen else None

        summary[b] = {
            'files': len(items),
            'missing_total_preempts': len(items) - len(vals_preempts),
            'missing_avg_prompt_throughput': len(items) - len(vals_prompt),
            'missing_avg_generation_throughput': len(items) - len(vals_gen),
        }

    return agg_total_preempts, agg_avg_prompt, agg_avg_generation, summary


def plot_three_subplots(agg_preempts, agg_prompt, agg_gen, out_file=None):
    """
    Plot three vertically stacked subplots (shared x-axis) for the three metrics.
    Batches with missing values for a metric are skipped for that metric's plot.
    """
    batches = sorted(set(list(agg_preempts.keys()) + list(agg_prompt.keys()) + list(agg_gen.keys())))

    def prepare(xs, d):
        x = []
        y = []
        for b in xs:
            v = d.get(b, None)
            if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                x.append(b)
                y.append(v)
        return x, y

    x_pre, y_pre = prepare(batches, agg_preempts)
    x_prm, y_prm = prepare(batches, agg_prompt)
    x_gen, y_gen = prepare(batches, agg_gen)

    fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)
    fig.suptitle('Metrics vs Batch Size', fontsize=16)

    ax = axes[0]
    ax.plot(x_pre, y_pre, marker='o', linestyle='-', label='total_preempts')
    ax.set_ylabel('total_preempts')
    ax.grid(True)
    ax.legend()

    ax = axes[1]
    ax.plot(x_prm, y_prm, marker='o', linestyle='-', label='avg_prompt_throughput')
    ax.set_ylabel('avg_prompt_throughput (samples/s)')
    ax.grid(True)
    ax.legend()

    ax = axes[2]
    ax.plot(x_gen, y_gen, marker='o', linestyle='-', label='avg_generation_throughput')
    ax.set_xlabel('Batch size')
    ax.set_ylabel('avg_generation_throughput')
    ax.grid(True)
    ax.legend()

    if batches:
        plt.xticks(batches)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if out_file:
        plt.savefig(out_file, dpi=200)
        print(f"Saved figure to: {out_file}")
    plt.show()


def main(folder, out_file=None):
    results = scan_folder(folder)
    if not results:
        print(f"No .log files with 'B<digits>' found under {folder}.")
        return
    agg_preempts, agg_prompt, agg_gen, summary = aggregate_results(results)

    print("Found batches:", sorted(results.keys()))
    for b in sorted(summary.keys()):
        s = summary[b]
        print(f"Batch {b}: {s['files']} files, missing: total_preempts={s['missing_total_preempts']}, "
              f"avg_prompt={s['missing_avg_prompt_throughput']}, avg_gen={s['missing_avg_generation_throughput']}")

    plot_three_subplots(agg_preempts, agg_prompt, agg_gen, out_file=out_file)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Plot total_preempts, avg_prompt_throughput, avg_generation_throughput by batch")
    ap.add_argument('folder', help='folder to scan (recursive)')
    ap.add_argument('--out', help='optional output image file path (png)', default='throughputs_by_batch.png')
    args = ap.parse_args()
    main(args.folder, out_file=args.out)
