"""
Build a training Parquet for SciAccel-RL tasks.

Reads instruction.md and task.toml from each task directory in the sciaccel-rl
repo and produces a single Parquet with one row per task.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TASK_REWARD_KEYS = {
    "laps-cpu": "reward",
    "laps-cuda": "reward_gpu",
}

TASK_GPUS = {
    "laps-cpu": 0,
    "laps-cuda": 1,
}

TASK_TIMEOUTS = {
    "laps-cpu": 3600.0,
    "laps-cuda": 14400.0,
}


def _read_instruction(task_dir: Path) -> str:
    """
    Read the agent instruction from a task directory.
    """
    instruction_path = task_dir / "instruction.md"
    return instruction_path.read_text(encoding="utf-8").strip()


def _parse_task_name(task_dir: Path) -> str:
    """
    Parse the task name from task.toml.
    """
    toml_path = task_dir / "task.toml"
    content = toml_path.read_text(encoding="utf-8")
    for line in content.splitlines():
        if line.strip().startswith("name"):
            parts = line.split("=", 1)
            if len(parts) == 2:
                return parts[1].strip().strip('"').strip("'")
    return f"sciaccel/{task_dir.name}"


def build_dataset(
    repo_path: str,
    output_path: str,
    tasks: list[str] | None = None,
) -> None:
    """
    Build a training Parquet from sciaccel-rl tasks.

    Args:
        repo_path: Path to the sciaccel-rl repository root.
        output_path: Where to write the output Parquet file.
        tasks: List of task directory names to include. Defaults to all known tasks.
    """
    repo = Path(repo_path)
    if tasks is None:
        tasks = list(TASK_REWARD_KEYS.keys())

    rows = []
    for task_name in tasks:
        task_dir = repo / "tasks" / task_name
        if not task_dir.exists():
            raise FileNotFoundError(f"Task directory not found: {task_dir}")

        instruction = _read_instruction(task_dir)
        parsed_name = _parse_task_name(task_dir)

        prompt = [{"role": "user", "content": instruction}]
        extra_info = {
            "task_path": str(task_dir),
            "reward_key": TASK_REWARD_KEYS.get(task_name, "reward"),
            "task_name": parsed_name,
            "timeout_sec": TASK_TIMEOUTS.get(task_name, 3600.0),
            "gpus": TASK_GPUS.get(task_name, 0),
        }

        rows.append(
            {
                "prompt": prompt,
                "extra_info": extra_info,
                "data_source": "sciaccel_rl",
                # Harbor supplies verifier reward at rollout time.
                # Ground truth remains required by the reward loop schema.
                "reward_model": {"style": "rule", "ground_truth": ""},
            }
        )

    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


def main():
    ap = argparse.ArgumentParser(description="Build SciAccel-RL training dataset.")
    ap.add_argument("--repo", required=True, help="Path to sciaccel-rl repo root.")
    ap.add_argument("--output", required=True, help="Output Parquet path.")
    ap.add_argument("--tasks", nargs="+", default=None, help="Task names to include.")
    args = ap.parse_args()

    build_dataset(repo_path=args.repo, output_path=args.output, tasks=args.tasks)
    print(f"Dataset written to {args.output}")


if __name__ == "__main__":
    main()
