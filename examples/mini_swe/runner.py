"""Run one mini-SWE-agent task through its standard Python bindings."""

from __future__ import annotations

import logging
import os
from typing import Any

os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")

psrl_logger = logging.getLogger(__name__)
_PROXY_ENV_KEYS = [
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "no_proxy",
    "NO_PROXY",
]


def _build_environment_config(payload: dict[str, Any]) -> dict[str, Any]:
    environment = dict(payload["runtime_config"]["sandbox_config"]["environment"])
    observation = payload["observation"]
    use_preexisting_repo = observation.get("use_preexisting_repo", True)
    preexisting_repo_name = observation.get("preexisting_repo_name", "")
    if use_preexisting_repo and preexisting_repo_name:
        environment["cwd"] = f"/{preexisting_repo_name}"

    run_args = list(environment.get("run_args", []))
    swe_task_id = observation.get("swe_task_id", "")
    if swe_task_id:
        run_args.extend(["--label", f"psrl.swe_task_id={swe_task_id}"])
    actor_id = payload.get("actor_id", "")
    if actor_id:
        run_args.extend(["--label", f"psrl.actor_id={actor_id}"])
    if not use_preexisting_repo and observation.get("repo_path"):
        run_args.extend(["--volume", f"{observation['repo_path']}:/testbed"])

    environment.update(
        {
            "environment_class": environment.get("environment_class", "docker"),
            "forward_env": _PROXY_ENV_KEYS,
            "run_args": run_args,
        }
    )
    environment.pop("grader_memory", None)
    return environment


def _build_model_config(payload: dict[str, Any]) -> dict[str, Any]:
    model_config = dict(payload["runtime_config"]["model"])
    model_kwargs = dict(payload["sampling_params"])
    top_k = model_kwargs.pop("top_k", None)
    if top_k is not None and int(top_k) >= 0:
        model_kwargs["extra_body"] = {"top_k": int(top_k)}
    model_kwargs.update(
        {
            "api_base": payload["base_url"],
            "api_key": "EMPTY",
            "timeout": payload["runtime_config"]["sandbox_config"]["rollout_turn_timeout"],
        }
    )
    model_config.update(
        {
            "model_name": payload["model"],
            "model_kwargs": model_kwargs,
            "cost_tracking": "ignore_errors",
        }
    )
    return model_config


def _grader_failure(swe_problem: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        "policy_violated": False,
        "policy_reasons": [],
        "resolved": False,
        "apply_ok": False,
        "f2p_pass": 0,
        "f2p_total": len(swe_problem.get("FAIL_TO_PASS", [])),
        "p2p_pass": 0,
        "p2p_total": len(swe_problem.get("PASS_TO_PASS", [])),
        "timeout": False,
        "error": error,
        "elapsed_s": 0.0,
        "output_tail": "",
        "resolved_by": "grader_error",
    }


def _grade_patch(payload: dict[str, Any], patch: str) -> dict[str, Any] | None:
    observation = payload["observation"]
    if not patch or observation.get("swe_grader") != "swebench_fresh_container":
        return None

    swe_problem = observation.get("swe_problem", {})
    swe_image = str(observation.get("swe_problem_image", "") or "")
    if not swe_problem or not swe_image:
        return _grader_failure(swe_problem, "missing_grader_input")

    try:
        from examples.mini_swe.swebench_grader import grade_fresh_container

        grader_kind = (
            "smith"
            if observation.get("swe_restore_tests", False)
            else "gym"
            if swe_problem.get("eval_script")
            else "verified"
        )
        grader_memory = payload["runtime_config"]["sandbox_config"]["environment"].get("grader_memory", "")
        return grade_fresh_container(
            swe_problem,
            patch,
            grader_kind,
            swe_image,
            900,
            observation.get("swe_task_id", ""),
            grader_memory,
        )
    except Exception as exc:
        return _grader_failure(swe_problem, str(exc))


def run_agent(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one task through mini-SWE-agent's standard Python bindings."""
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments import get_environment
    from minisweagent.models import get_model

    environment = None
    try:
        environment = get_environment(_build_environment_config(payload))
        model = get_model(config=_build_model_config(payload))
        agent_config = dict(payload["runtime_config"]["agent"])
        agent_config["instance_template"] = agent_config.pop("problem_template")
        agent_config["step_limit"] = payload["max_turns"]
        agent_config["output_path"] = None
        result = DefaultAgent(model, environment, **agent_config).run(payload["task"])
        patch = result.get("submission", "") or ""
        try:
            environment.cleanup()
            environment = None
        except Exception:
            psrl_logger.warning("mini-SWE-agent environment cleanup before grading failed.", exc_info=True)
        return {
            "exit_status": result.get("exit_status", ""),
            "submission": patch,
            "grader_result": _grade_patch(payload, patch),
        }
    except Exception as exc:
        psrl_logger.warning("mini-SWE-agent task failed: %s.", exc, exc_info=True)
        return {
            "exit_status": "error",
            "submission": "",
            "grader_result": None,
            "error": str(exc),
        }
    finally:
        if environment is not None:
            try:
                environment.cleanup()
            except Exception:
                psrl_logger.warning("mini-SWE-agent environment cleanup failed.", exc_info=True)
