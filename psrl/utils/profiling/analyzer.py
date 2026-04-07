import json
import logging
import os
from pathlib import Path

import numpy as np

from psrl.utils.profiling.records import TrajectoryProfilingData

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


def read_jsonl(path: str | Path) -> list[TrajectoryProfilingData]:
    """
    Read all trajectory profiling records from a JSONL file.

    Each line in the file must be a JSON object representing one
    TrajectoryProfilingData record. Lines that cannot be parsed are
    skipped with a warning. If the file is missing, an empty list is
    returned with a warning.

    Args:
        path (str | Path): Path to the JSONL file to read.

    Returns:
        list[TrajectoryProfilingData]: Parsed records in file order.
            Returns an empty list when the file is missing or contains
            no valid lines.
    """
    path = Path(path)

    if not path.exists():
        psrl_logger.warning(f"Profiling JSONL file not found: {path}.")
        return []

    records: list[TrajectoryProfilingData] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                data = json.loads(raw_line)
                records.append(TrajectoryProfilingData.from_dict(data))
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                psrl_logger.warning(
                    f"Skipping malformed line {lineno} in {path}: {e}."
                )

    psrl_logger.info(
        f"[read_jsonl] Loaded {len(records)} trajectory profiling records from {path}."
    )
    return records


def analyze(records: list[TrajectoryProfilingData]) -> dict:
    """
    Compute aggregate statistics across all trajectory profiling records.

    Before aggregation, compute_summary() is called on any record whose
    summary has not yet been populated, ensuring all derived fields are
    available.

    Args:
        records (list[TrajectoryProfilingData]): Trajectory records to
            analyze. May be empty.

    Returns:
        dict: Aggregate statistics, or an empty dict when no records are
            provided.
    """
    if not records:
        return {}

    # Ensure all summaries are computed
    for r in records:
        if not r.summary:
            r.compute_summary()

    num_trajectories = len(records)

    # Per-trajectory aggregates
    prefill_fractions = [r.summary.get("prefill_fraction", 0.0) for r in records]
    decode_fractions = [r.summary.get("decode_fraction", 0.0) for r in records]
    env_fractions = [r.summary.get("env_fraction", 0.0) for r in records]
    scheduler_wait_fractions = [r.summary.get("scheduler_wait_fraction", 0.0) for r in records]
    router_wait_fractions = [r.summary.get("router_wait_fraction", 0.0) for r in records]
    cache_hit_rates = [r.summary.get("avg_cache_hit_rate", 0.0) for r in records]
    avg_turn_durations = [r.summary.get("avg_turn_duration_s", 0.0) for r in records]
    avg_tokens = [r.summary.get("avg_num_generated_tokens", 0.0) for r in records]

    # Trigger breakdown across all trajectories
    total_triggers = {
        "initial": 0,
        "preempt_resume": 0,
        "partial_rollout_resume": 0,
        "internal_preempt_resume": 0,
    }
    for r in records:
        breakdown = r.summary.get("prefill_trigger_breakdown", {})
        for k in total_triggers:
            total_triggers[k] += breakdown.get(k, 0)

    # Per-turn prefill seqlen distribution
    all_prefill_seqlens = []
    for r in records:
        for turn in r.turn_records:
            for pr in turn.prefill_records:
                all_prefill_seqlens.append(pr.num_prefill_tokens)

    # Decode throughput (tokens/sec) using pure decode compute time.
    decode_throughputs = []
    for r in records:
        for turn in r.turn_records:
            if turn.decode_time_s > 0:
                decode_throughputs.append(turn.num_generated_tokens / turn.decode_time_s)

    return {
        "num_trajectories": num_trajectories,
        "avg_prefill_fraction": float(np.mean(prefill_fractions)),
        "avg_decode_fraction": float(np.mean(decode_fractions)),
        "avg_env_fraction": float(np.mean(env_fractions)),
        "avg_scheduler_wait_fraction": float(np.mean(scheduler_wait_fractions)),
        "avg_router_wait_fraction": float(np.mean(router_wait_fractions)),
        "avg_cache_hit_rate": float(np.mean(cache_hit_rates)),
        "std_cache_hit_rate": float(np.std(cache_hit_rates)),
        "avg_turn_duration_s": float(np.mean(avg_turn_durations)),
        "std_turn_duration_s": float(np.std(avg_turn_durations)),
        "avg_tokens_per_turn": float(np.mean(avg_tokens)),
        "prefill_trigger_breakdown": total_triggers,
        "prefill_seqlen_p50": (
            float(np.percentile(all_prefill_seqlens, 50))
            if all_prefill_seqlens
            else 0.0
        ),
        "prefill_seqlen_p90": (
            float(np.percentile(all_prefill_seqlens, 90))
            if all_prefill_seqlens
            else 0.0
        ),
        "prefill_seqlen_p99": (
            float(np.percentile(all_prefill_seqlens, 99))
            if all_prefill_seqlens
            else 0.0
        ),
        "avg_decode_throughput_tok_s": (
            float(np.mean(decode_throughputs)) if decode_throughputs else 0.0
        ),
    }
