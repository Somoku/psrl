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


@pytest.fixture(scope="module")
def canonical_rows():
    """
    Index the canonical task rows by unprefixed task name.
    """
    import json

    path = Path(SCIACCEL_RL_REPO) / "envs" / "laps" / "tasks.jsonl"
    with open(path, encoding="utf-8") as f:
        return {row["task"]: row for row in (json.loads(line) for line in f if line.strip())}


@pytest.fixture(scope="module")
def hint_datasets():
    """
    Build every hint level once into a shared temporary directory.
    """
    import pandas as pd
    from examples.sciaccel_rl.prepare.build_dataset_v2 import HINT_LEVELS, build_datasets

    with tempfile.TemporaryDirectory() as tmpdir:
        frames = {}
        for level in HINT_LEVELS:
            build_datasets(
                repo_path=SCIACCEL_RL_REPO,
                out_dir=tmpdir,
                categories=["repair", "implementation"],
                hint_level=level,
            )
            frames[level] = pd.read_parquet(Path(tmpdir) / f"{level}_all.parquet")
            frames[f"{level}_train"] = pd.read_parquet(Path(tmpdir) / f"{level}_train.parquet")
        frames["val"] = pd.read_parquet(Path(tmpdir) / "val.parquet")
        yield frames


@pytest.fixture(scope="module")
def hint_train_frames(hint_datasets):
    """
    Expose the train splits under their bare level names.
    """
    return {level: hint_datasets[f"{level}_train"] for level in ("L1", "L2", "L3")}


@pytest.mark.skipif(
    not Path(SCIACCEL_RL_REPO).exists(),
    reason="sciaccel-rl repo not available",
)
class TestHintLevels:
    """Test the localization hint appended to repair task instructions."""

    def test_l3_carries_no_hint(self, hint_datasets):
        df = hint_datasets["L3"]
        assert len(df) == 143
        assert all(not row["hint"] for row in df["extra_info"])

    def test_only_repair_tasks_are_hinted(self, hint_datasets):
        for level in ("L1", "L2"):
            df = hint_datasets[level]
            hinted = {r["task_name"] for _, r in df.iterrows() if r["extra_info"]["hint"]}
            assert len(hinted) == 99
            categories = set(df[df["task_name"].isin(hinted)]["category"])
            assert categories == {"repair"}

    def test_hint_names_the_golden_patch_file(self, hint_datasets, canonical_rows):
        # A hint pointing anywhere but the edited file would teach the wrong search.
        for level in ("L1", "L2"):
            for _, row in hint_datasets[level].iterrows():
                hint = row["extra_info"]["hint"]
                if not hint:
                    continue
                canonical = canonical_rows[row["task_name"].split("/")[-1]]
                assert canonical["candidate"]["fix"]["edits"][0]["file"] in hint

    def test_l1_carries_the_line_and_l2_does_not(self, hint_datasets, canonical_rows):
        for _, row in hint_datasets["L1"].iterrows():
            hint = row["extra_info"]["hint"]
            if not hint:
                continue
            line = canonical_rows[row["task_name"].split("/")[-1]]["candidate"]["meta"].get("line")
            if line is not None:
                assert f"line {line}" in hint
        assert all("line " not in row["hint"] for row in hint_datasets["L2"]["extra_info"])

    def test_prompt_column_mirrors_the_hint(self, hint_datasets):
        # The column is documentation, because Harbor delivers the hint itself.
        for _, row in hint_datasets["L1"].iterrows():
            hint = row["extra_info"]["hint"]
            if hint:
                assert row["prompt"][0]["content"].endswith(hint)

    def test_validation_is_unhinted_at_every_level(self, hint_datasets):
        assert all(not row["hint"] for row in hint_datasets["val"]["extra_info"])

    def test_hint_level_is_recorded_on_every_row(self, hint_datasets):
        for level in ("L1", "L2", "L3"):
            assert all(row["hint_level"] == level for row in hint_datasets[level]["extra_info"])

    def test_unknown_level_is_rejected(self):
        from examples.sciaccel_rl.prepare.build_dataset_v2 import build_datasets

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Unknown hint_level"):
                build_datasets(repo_path=SCIACCEL_RL_REPO, out_dir=tmpdir, hint_level="L9")

    def test_train_split_interleaves_categories(self, hint_train_frames):
        # The bank is grouped by category on disk. A sequential sampler over that
        # order spent the first two steps entirely on unhinted restore tasks.
        df = hint_train_frames["L1"]
        batch = df.iloc[:16]
        assert set(batch["category"]) == {"repair", "implementation"}
        assert any(row["hint"] for row in batch["extra_info"])

    def test_first_batch_is_mostly_hinted(self, hint_train_frames):
        # Repair is 99 of 143 tasks, so a representative batch is majority hinted.
        batch = hint_train_frames["L1"].iloc[:16]
        assert sum(1 for row in batch["extra_info"] if row["hint"]) >= 8

    def test_repair_only_build_is_fully_hinted(self):
        # The default training set for the 4B run. Excised routine tasks are excluded
        # because a location hint cannot help restore a whole subroutine body.
        import pandas as pd
        from examples.sciaccel_rl.prepare.build_dataset_v2 import build_datasets

        with tempfile.TemporaryDirectory() as tmpdir:
            build_datasets(
                repo_path=SCIACCEL_RL_REPO,
                out_dir=tmpdir,
                categories=["repair"],
                hint_level="L1",
            )
            train = pd.read_parquet(Path(tmpdir) / "L1_train.parquet")
            val = pd.read_parquet(Path(tmpdir) / "val.parquet")

        assert set(train["category"]) == {"repair"}
        assert all(row["hint"] for row in train["extra_info"])
        assert all("line " in row["hint"] for row in train["extra_info"])
        # Validation stays unhinted so it measures the task as it really ships.
        assert not any(row["hint"] for row in val["extra_info"])
