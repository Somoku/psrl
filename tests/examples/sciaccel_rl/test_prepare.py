"""Tests for the SciAccel-RL dataset preparation script."""

import tempfile
from pathlib import Path

import pytest


SCIACCEL_RL_REPO = "/apdcephfs_zwfy10_303541817/share_303541817/lhy/science_infra/sciaccel-rl"


@pytest.mark.skipif(
    not Path(SCIACCEL_RL_REPO).exists(),
    reason="sciaccel-rl repo not available",
)
class TestBuildDataset:
    """Test Parquet generation from sciaccel-rl tasks."""

    def test_builds_parquet_with_both_tasks(self):
        import pandas as pd
        from examples.sciaccel_rl.prepare.build_dataset import build_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "train.parquet"
            build_dataset(
                repo_path=SCIACCEL_RL_REPO,
                output_path=str(out_path),
                tasks=["laps-cpu", "laps-cuda"],
            )
            assert out_path.exists()
            df = pd.read_parquet(out_path)
            assert len(df) == 2
            assert "prompt" in df.columns
            assert "extra_info" in df.columns
            assert "data_source" in df.columns

            cpu_row = df[df["extra_info"].apply(lambda x: x.get("task_name") == "sciaccel/laps-cpu")]
            assert len(cpu_row) == 1
            assert cpu_row.iloc[0]["extra_info"]["reward_key"] == "reward"
            assert cpu_row.iloc[0]["extra_info"]["gpus"] == 0

            cuda_row = df[df["extra_info"].apply(lambda x: x.get("task_name") == "sciaccel/laps-cuda")]
            assert len(cuda_row) == 1
            assert cuda_row.iloc[0]["extra_info"]["reward_key"] == "reward_gpu"
            assert cuda_row.iloc[0]["extra_info"]["gpus"] == 1

    def test_prompt_is_chat_format(self):
        import pandas as pd
        from examples.sciaccel_rl.prepare.build_dataset import build_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "train.parquet"
            build_dataset(
                repo_path=SCIACCEL_RL_REPO,
                output_path=str(out_path),
                tasks=["laps-cpu"],
            )
            df = pd.read_parquet(out_path)
            prompt = list(df.iloc[0]["prompt"])
            assert isinstance(prompt, list)
            assert prompt[0]["role"] == "user"
            assert "LAPS" in prompt[0]["content"]
