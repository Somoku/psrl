"""
mini-SWE-Agent subprocess runner.

Manages the `mini-swe-agent` CLI subprocess lifecycle:
- Process creation and environment setup.
- Timeout handling with graceful SIGTERM -> SIGKILL escalation.
- stdout/stderr log capture.
- Patch extraction via `PatchExtractor`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

from psrl.utils.common.patch_extractor import PatchExtractor
from psrl.utils.common.docker_utils import cleanup_containers_by_label

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


# ---------------------------------------------------------------------------
# Docker container cleanup
# ---------------------------------------------------------------------------


async def cleanup_instance_containers(swe_problem_id: str) -> None:
    """
    Stop Docker containers belonging to a specific SWE problem run.

    Uses the `psrl.swe_problem_id` label to precisely target only this
    run's containers. Idempotent -- no-op if no containers exist.
    """
    await cleanup_containers_by_label("psrl.swe_problem_id", swe_problem_id)


async def execute_mini_swe_agent(
    *,
    config_path: str,
    problem_statement: str,
    swe_problem_id: str,
    output_dir: str,
    output_json_path: str,
    repo_path: str,
    exec_dir: str,
    swe_agent_timeout: int = 1800,
    proxy_port: int = 8080,
) -> str | None:
    """
    Execute mini-SWE-Agent CLI and return the generated patch.

    Args:
        config_path (str): Path to mini-SWE-agent YAML config file.
        problem_statement (str): The problem statement text.
        swe_problem_id (str): Unique SWE problem identifier.
        output_dir (str): Directory for mini-SWE-agent output / logs.
        output_json_path (str): Path for the trajectory JSON output file.
        repo_path (str): Path to the repository (for git diff fallback).
        exec_dir (str): Working directory for the subprocess.
        swe_agent_timeout (int): Overall timeout in seconds.
        proxy_port (int): `ModelProxy` port (for logging only).

    Returns:
        str | None: Generated patch string, or None on failure.
    """
    cmd = [
        "mini-swe-agent",
        "-t", problem_statement,
        "-c", config_path,
        "-o", output_json_path,
        "-y",                  # no confirmation prompts
        "--exit-immediately",  # don't wait after completion
    ]

    psrl_logger.info(f"[{swe_problem_id}] Executing mini-SWE-Agent (proxy port={proxy_port})...")

    env = os.environ.copy()
    process = None

    try:
        subprocess_start = time.time()
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=exec_dir,
        )
        psrl_logger.info(f"[{swe_problem_id}] Subprocess created (pid={process.pid}), waiting for completion...")

        # Wait with timeout.
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=swe_agent_timeout,
            )
        except asyncio.TimeoutError:
            elapsed = time.time() - subprocess_start
            psrl_logger.error(
                f"[{swe_problem_id}] mini-SWE-Agent timed out after {elapsed:.1f}s "
                f"(limit={swe_agent_timeout}s)."
            )
            # Graceful: SIGTERM first, escalate to SIGKILL after 15 s.
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=15.0)
                psrl_logger.info(f"[{swe_problem_id}] mini-SWE-Agent exited gracefully after SIGTERM.")
            except asyncio.TimeoutError:
                psrl_logger.warning(
                    f"[{swe_problem_id}] mini-SWE-Agent did not exit after SIGTERM, sending SIGKILL."
                )
                process.kill()
                await process.wait()
            return None

        subprocess_elapsed = time.time() - subprocess_start
        stdout_text = stdout.decode(errors="replace")
        stderr_text = stderr.decode(errors="replace")

        # Persist logs.
        _save_logs(output_dir, swe_problem_id, stdout_text, stderr_text)

        if process.returncode != 0:
            psrl_logger.error(
                f"[{swe_problem_id}] mini-SWE-Agent failed (rc={process.returncode}) "
                f"after {subprocess_elapsed:.1f}s."
            )
            psrl_logger.error(f"[{swe_problem_id}] stderr (last 2000): {stderr_text[-2000:]}")
            psrl_logger.error(f"[{swe_problem_id}] stdout (last 1000): {stdout_text[-1000:]}")
        else:
            psrl_logger.info(
                f"[{swe_problem_id}] mini-SWE-Agent subprocess completed successfully "
                f"in {subprocess_elapsed:.1f}s."
            )

        # Extract patch.
        extract_start = time.time()
        patch = await _extract_patch(output_dir, swe_problem_id, repo_path, output_json_path)
        psrl_logger.info(f"[{swe_problem_id}] Patch extraction took {time.time() - extract_start:.1f}s.")

        if patch:
            psrl_logger.info(f"[{swe_problem_id}] Successfully extracted patch ({len(patch)} chars).")
        else:
            psrl_logger.warning(f"[{swe_problem_id}] No patch found in mini-SWE-Agent output or git diff.")

        return patch

    except asyncio.CancelledError:
        psrl_logger.warning(f"[{swe_problem_id}] mini-SWE-Agent task cancelled, terminating subprocess...")
        if process is not None and process.returncode is None:
            try:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=15.0)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
            except Exception:
                pass
        # Still try to extract patch from output files even after cancellation.
        try:
            patch = await asyncio.wait_for(
                _extract_patch(output_dir, swe_problem_id, repo_path, output_json_path),
                timeout=10.0,
            )
            if patch:
                psrl_logger.info(f"[{swe_problem_id}] Extracted patch after cancellation ({len(patch)} chars).")
                return patch
        except Exception as exc:
            psrl_logger.debug(f"[{swe_problem_id}] Patch extraction after cancel failed: {exc}.")
        raise
    except FileNotFoundError:
        psrl_logger.error("mini-SWE-Agent not found. Please install it with: pip install mini-swe-agent.")
        return None
    except Exception as e:
        psrl_logger.exception(f"[{swe_problem_id}] Error running mini-SWE-Agent: {e}.")
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_logs(output_dir: str, swe_problem_id: str, stdout_text: str, stderr_text: str) -> None:
    """
    Persist subprocess stdout/stderr to files.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        stdout_path = os.path.join(output_dir, f"{swe_problem_id}.stdout.log")
        stderr_path = os.path.join(output_dir, f"{swe_problem_id}.stderr.log")
        with open(stdout_path, "w", encoding="utf-8") as f:
            f.write(stdout_text)
        with open(stderr_path, "w", encoding="utf-8") as f:
            f.write(stderr_text)
        psrl_logger.info(
            f"[{swe_problem_id}] Saved mini-SWE-Agent subprocess logs: "
            f"stdout={stdout_path}, stderr={stderr_path}."
        )
    except Exception as e:
        psrl_logger.warning(f"[{swe_problem_id}] Failed to save subprocess logs: {e}.")


async def _extract_patch(
    output_dir: str,
    swe_problem_id: str,
    repo_path: str,
    trajectory_json_path: str | None = None,
) -> str | None:
    """
    Extract patch via `PatchExtractor` (trajectory JSON -> git diff fallback).
    """
    extractor = PatchExtractor(
        output_dir=output_dir,
        swe_problem_id=swe_problem_id,
        repo_path=repo_path,
        trajectory_json_path=trajectory_json_path,
    )
    return await extractor.extract()
