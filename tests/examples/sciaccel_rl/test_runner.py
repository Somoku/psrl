"""Tests for sciaccel_rl Harbor runner."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


class TestHarborEpisodeResult:
    """Test the result dataclass."""

    def test_construction(self):
        from examples.sciaccel_rl.runner import HarborEpisodeResult

        result = HarborEpisodeResult(
            task_name="sciaccel/laps-cpu",
            reward=0.75,
            rewards={"reward": 0.75, "equivalence_pass": 0},
        )
        assert result.task_name == "sciaccel/laps-cpu"
        assert result.reward == 0.75
        assert result.exception is None

    def test_failed_episode(self):
        from examples.sciaccel_rl.runner import HarborEpisodeResult

        result = HarborEpisodeResult(
            task_name="sciaccel/laps-cpu",
            reward=0.0,
            rewards={},
            exception="container_timeout",
        )
        assert result.reward == 0.0
        assert result.exception == "container_timeout"


class TestRunHarborEpisode:
    """Test run_harbor_episode with mocked Harbor API."""

    def test_successful_episode(self):
        from examples.sciaccel_rl.config import SciAccelRuntimeConfig
        from examples.sciaccel_rl.runner import HarborEpisodeResult, run_harbor_episode

        mock_trial_result = MagicMock()
        mock_trial_result.task_name = "sciaccel/laps-cpu"
        mock_trial_result.trial_name = "trial-001"
        mock_trial_result.verifier_result = MagicMock()
        mock_trial_result.verifier_result.rewards = {"reward": 0.85, "equivalence_pass": 0}
        mock_trial_result.exception_info = None

        mock_job_result = MagicMock()
        mock_job_result.trial_results = [mock_trial_result]

        mock_job = AsyncMock()
        mock_job.run = AsyncMock(return_value=mock_job_result)

        config = SciAccelRuntimeConfig()

        with patch("examples.sciaccel_rl.runner.Job") as mock_job_cls:
            mock_job_cls.create = AsyncMock(return_value=mock_job)
            result = asyncio.run(
                run_harbor_episode(
                    task_path="/path/to/tasks/laps-cpu",
                    model_base_url="http://10.0.0.1:8000/sessions/abc/v1",
                    model_name="Qwen/Qwen3-8B",
                    config=config,
                )
            )
            job_config = mock_job_cls.create.await_args.args[0]

        assert isinstance(result, HarborEpisodeResult)
        assert result.reward == 0.85
        assert result.rewards["reward"] == 0.85
        assert result.exception is None
        assert job_config.agents[0].kwargs["record_terminal_session"] is False

    def test_episode_with_exception(self):
        from examples.sciaccel_rl.config import SciAccelRuntimeConfig
        from examples.sciaccel_rl.runner import run_harbor_episode

        mock_trial_result = MagicMock()
        mock_trial_result.task_name = "sciaccel/laps-cpu"
        mock_trial_result.trial_name = "trial-001"
        mock_trial_result.verifier_result = None
        mock_trial_result.exception_info = MagicMock()
        mock_trial_result.exception_info.exception_message = "build_failed"

        mock_job_result = MagicMock()
        mock_job_result.trial_results = [mock_trial_result]

        mock_job = AsyncMock()
        mock_job.run = AsyncMock(return_value=mock_job_result)

        config = SciAccelRuntimeConfig()

        with patch("examples.sciaccel_rl.runner.Job") as mock_job_cls:
            mock_job_cls.create = AsyncMock(return_value=mock_job)
            result = asyncio.run(
                run_harbor_episode(
                    task_path="/path/to/tasks/laps-cpu",
                    model_base_url="http://10.0.0.1:8000/sessions/abc/v1",
                    model_name="Qwen/Qwen3-8B",
                    config=config,
                )
            )

        assert result.reward == 0.0
        assert result.exception == "build_failed"
