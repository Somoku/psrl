"""
Harbor Job execution wrapper for SciAccel-RL.

Runs one Harbor episode (agent container + verifier) and returns the shaped
reward from the verifier. The model endpoint is pointed at PSRL's
SessionRouter so TITO captures all tokens.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlparse

from examples.sciaccel_rl.config import SciAccelRuntimeConfig
from harbor.job import Job
from harbor.models.job.config import AgentConfig, JobConfig, SourceJobConfig
from harbor.models.trial.config import TaskConfig

from psrl.utils.agent.thinking import MULTI_TRAJ, harness_extra_body

psrl_logger = logging.getLogger("psrl.sciaccel_rl.runner")
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@dataclass
class HarborEpisodeResult:
    """
    Result of one Harbor episode.
    """

    task_name: str
    reward: float
    rewards: dict = field(default_factory=dict)
    exception: str | None = None
    # Exception class name, kept separate because Harbor's own classes distinguish
    # cases the message cannot. `HarborExceptionClassifier` prefers this over text.
    exception_type: str | None = None


async def _regrade_from_artifacts(
    task_path: str,
    trial_uri: str | None,
    jobs_dir: str | Path,
    timeout_sec: float,
) -> dict:
    """
    Recover a verifier score for a trial whose verification was skipped.

    Harbor skips verification whenever the agent raises, so an episode that hit the
    context window reports no rewards at all. The artifacts survive, though:
    `_recover_outputs()` collects them even on the failure path. Harbor can grade
    exactly those with a `regrade` source job -- no agent, no live container -- which
    LAPS tasks support because they declare `[verifier] environment_mode = "separate"`.

    This runs even when the trial delivered nothing. An earlier version skipped that
    case to avoid "wasting" a verifier build confirming a zero, which was a false
    economy: an empty reward dict and a graded 0.0 are different facts. The verifier
    grades empty artifacts happily (measured: 31 s, full reward dict), and the graded
    result carries `floor`, which `reward_repair` needs to normalize -- and which
    differs per task (0.0 for restore, 0.5 for repair). Leaving the dict empty makes a
    harness failure indistinguishable from a real zero in every downstream log.

    Safe by construction: it runs only when the reward dict is empty, and any failure
    degrades to `{}` so a regrade problem can never take down a rollout. Costs ~30-50 s
    against a 3600 s agent budget.

    Args:
        task_path: Harbor task directory, providing the verifier to grade with.
        trial_uri: The failed trial's directory, as a `file://` URI or a path.
        jobs_dir: Root for the regrade job's own scratch directory.
        timeout_sec: Backstop for the regrade job.

    Returns:
        dict: The recovered reward dict, or empty when grading could not run at all.
    """
    if not trial_uri:
        return {}
    parsed = urlparse(str(trial_uri))
    trial_dir = Path(unquote(parsed.path)) if parsed.scheme == "file" else Path(str(trial_uri))

    try:
        config = JobConfig(
            jobs_dir=Path(jobs_dir) / "regrade",
            job_name=f"regrade_{trial_dir.name}_{uuid.uuid4().hex[:8]}",
            tasks=[TaskConfig(path=task_path)],
            source_jobs=[SourceJobConfig(action="regrade", type="local", path=trial_dir.parent.resolve())],
            n_concurrent_trials=1,
            quiet=True,
        )
        job = await Job.create(config)
        result = await asyncio.wait_for(job.run(), timeout=timeout_sec)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        psrl_logger.warning("Regrade from artifacts failed for %s: %s.", trial_dir.name, exc)
        return {}

    for trial in result.trial_results or []:
        if trial.verifier_result and trial.verifier_result.rewards:
            rewards = dict(trial.verifier_result.rewards)
            psrl_logger.info(
                "Recovered a verifier score by regrading %s: reward=%s.",
                trial_dir.name,
                rewards.get("reward"),
            )
            return rewards
    return {}


async def run_harbor_episode(
    task_path: str,
    model_base_url: str,
    model_name: str,
    config: SciAccelRuntimeConfig,
    needs_gpu: bool = False,
    session_id: str = "",
    max_model_len: int = 40960,
    max_turns: int | None = None,
    regrade_unverified: bool = True,
    actor_id: str = "",
    thinking_template: str = MULTI_TRAJ,
) -> HarborEpisodeResult:
    """
    Run one Harbor episode and return the verifier reward.

    Args:
        task_path: Path to the Harbor task directory (containing task.toml).
        model_base_url: SessionRouter session URL for the model endpoint.
        model_name: Model name for the agent (e.g. ``Qwen/Qwen3-8B``).
        config: Runtime config with Harbor and timeout settings.
        needs_gpu: Whether the task container needs GPU device access.
        session_id: TITO session ID, used as the job directory name for traceability.
        max_model_len: vLLM context window size, forwarded to terminus-2 as its
            ``model_info`` limits. Note this is metadata only on the litellm path:
            nothing sends it as a per-request ``max_tokens``, so generation is bounded
            by the server's own ``--max-model-len``. `finish_reason == "length"` from
            that boundary is what terminus-2 reports as ``OutputLengthExceededError``.
        max_turns: Hard cap on agent turns. Must be set: terminus-2 otherwise defaults
            to ``max_episodes=1000000``, so an agent that never calls the completion
            tool grinds on until the context window overflows.
        regrade_unverified: When the verifier never ran, try to recover a score from
            the artifacts the episode delivered before failing. Training an empty
            reward dict as 0.0 is an incorrect label, not just a missing one -- see
            ``_regrade_from_artifacts``.
        thinking_template: How the model's chain-of-thought is carried across turns.
            Decides the ``extra_body`` sent to the gateway: ``multi_thinking`` turns the
            reasoning parser off so <think> stays inline in ``content``,
            ``disable_thinking`` turns thinking off, and the two trajectory-oriented
            modes send nothing. See ``psrl/utils/agent/thinking.py``.

    Returns:
        HarborEpisodeResult with the verifier's shaped reward.
    """
    extra_body = harness_extra_body(thinking_template)
    env_kwargs: dict = {}
    extra_compose_paths: list[Path] = []
    if needs_gpu and config.harbor.gpu_compose_override:
        extra_compose_paths.append(Path(config.harbor.gpu_compose_override))

    if actor_id:
        import tempfile

        import yaml

        label_override = {
            "services": {
                "main": {
                    "labels": [f"psrl.actor_id={actor_id}"],
                },
            },
        }
        label_file = Path(tempfile.mktemp(suffix=".yaml", prefix="harbor-label-"))
        label_file.write_text(yaml.dump(label_override))
        extra_compose_paths.append(label_file)

    if extra_compose_paths:
        env_kwargs["extra_docker_compose"] = extra_compose_paths

    job_name = session_id or uuid.uuid4().hex[:12]
    job_dir = Path(config.harbor.jobs_dir) / job_name
    job_config = JobConfig(
        jobs_dir=job_dir,
        tasks=[TaskConfig(path=task_path)],
        agents=[
            AgentConfig(
                name=config.harbor.agent_name,
                model_name=f"openai/{model_name}",
                env={"OPENAI_API_KEY": "EMPTY"},
                # Let Harbor enforce the agent budget from OUTSIDE the container, the
                # one clock the agent cannot forge. Without this, Harbor falls back to
                # the task's own `[agent] timeout_sec` (3600s for laps-cpu) while the
                # `asyncio.wait_for` below fires at task_timeout_sec + 600 -- so PSRL
                # killed the Job before Harbor could stop the agent and grade it, and
                # an honestly-slow episode became `trajectory_timeout` with no reward.
                # Matches SkyRL's harbor_trial_config/default.yaml:34.
                override_timeout_sec=config.task_timeout_sec,
                kwargs={
                    "api_base": model_base_url,
                    "enable_summarize": False,
                    "collect_rollout_details": True,
                    # SciAccel task containers run without network access and their
                    # images do not include asciinema. Terminus-2 otherwise retries
                    # apt and pip in every episode before falling back to the tmux
                    # pane, adding setup latency and one misleading error per trial.
                    # TITO token capture and terminus_2.pane do not depend on this.
                    "record_terminal_session": False,
                    "temperature": 1.0,
                    # Without this terminus-2 falls back to max_episodes=1000000 and
                    # runs until the context window overflows. Matches SkyRL's
                    # harbor_trial_config/default.yaml (max_turns + suppression).
                    **({"max_turns": max_turns} if max_turns else {}),
                    "suppress_max_turns_warning": True,
                    "model_info": {
                        "max_input_tokens": max_model_len,
                        "max_output_tokens": max_model_len,
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                    "llm_kwargs": {
                        "timeout": 900,
                        "max_retries": 0,
                        # Whatever `thinking_template` requires of the gateway. Empty for
                        # the `multi_traj` / `longest_traj` modes, which leave SMG's
                        # defaults alone and accept the resulting trajectory forks.
                        # See psrl/utils/agent/thinking.py.
                        **({"extra_body": extra_body} if extra_body else {}),
                    },
                },
            )
        ],
        n_attempts=1,
        n_concurrent_trials=1,
        **({"environment": env_kwargs} if env_kwargs else {}),
    )

    # Backstop only: Harbor's own agent + verifier clocks (set above) should always
    # fire first and still produce a graded result. This guard exists for the case
    # where Harbor itself wedges, so it must be strictly larger than the sum of the
    # budgets it supervises -- agent + verifier + container build/teardown slack.
    timeout = config.task_timeout_sec + config.verifier_timeout_sec + 600.0

    # Single exception boundary. Harbor normally reports per-trial failures as data in
    # `TrialResult.exception_info`, but `Job.create` and `Job.run` can still raise for
    # job-level trouble (unreadable task.toml, docker unreachable, the backstop above).
    # Letting those propagate would put raw Harbor/litellm exceptions in front of the
    # agent loop, which is what forced the loop to string-match them one by one.
    # Everything below returns a `HarborEpisodeResult` so the loop's input is a closed
    # set. `CancelledError` is deliberately NOT caught: cancellation is control flow,
    # and swallowing it would strand the Harbor event-loop thread.
    try:
        job = await Job.create(job_config)
        result = await asyncio.wait_for(job.run(), timeout=timeout)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        psrl_logger.warning("Harbor job failed before returning a trial: %s.", exc, exc_info=True)
        return HarborEpisodeResult(
            task_name="unknown",
            reward=0.0,
            exception=str(exc) or type(exc).__name__,
            exception_type=type(exc).__name__,
        )

    if not result.trial_results:
        return HarborEpisodeResult(
            task_name="unknown",
            reward=0.0,
            exception="no_trials",
            exception_type="NoTrialsError",
        )

    tr = result.trial_results[0]
    rewards = dict(tr.verifier_result.rewards) if tr.verifier_result and tr.verifier_result.rewards else {}
    # Keep the exception type alongside the message. Harbor's own classes are the
    # reliable signal (`AgentTimeoutError` vs `AgentSetupTimeoutError` mean opposite
    # things for training), while the message has already been through litellm, which
    # drops both the class and SMG's `x-smg-error-code` header.
    exception = tr.exception_info.exception_message if tr.exception_info else None
    exception_type = tr.exception_info.exception_type if tr.exception_info else None

    # An empty reward dict is a MISSING measurement, not a zero. Harbor's trial body
    # is a bare sequence (`harbor/trial/single_step.py`): agent, collect artifacts,
    # verify, with no try/except between them, so any agent-side exception -- a
    # context overflow above all -- skips verification entirely. Training that as
    # `reward=0` is an incorrect label, not merely a missing one: the episode may have
    # delivered a partially-correct result that the ladder would have given credit for.
    if not rewards and regrade_unverified:
        rewards = await _regrade_from_artifacts(
            task_path=task_path,
            trial_uri=tr.trial_uri,
            jobs_dir=job_config.jobs_dir,
            timeout_sec=config.verifier_timeout_sec + 600.0,
        )

    return HarborEpisodeResult(
        task_name=tr.task_name,
        reward=float(rewards.get("reward", 0.0)),
        rewards=rewards,
        exception=exception,
        exception_type=exception_type,
    )
