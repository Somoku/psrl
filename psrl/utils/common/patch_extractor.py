"""
Unified Patch Extractor.

Provides a single, clean interface for extracting patches from agent runs.
Tries multiple strategies in order:
1. Trajectory JSON: info.submission field
2. git diff HEAD (staged + unstaged)
3. git diff (unstaged only)
"""

import asyncio
import json
import logging
import os

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class PatchExtractor:
    """
    Unified patch extraction utility for mini-SWE-agent.

    Simplifies patch extraction by trying multiple strategies
    in a clean, testable way.
    """

    def __init__(
        self,
        output_dir: str,
        swe_problem_id: str,
        repo_path: str | None = None,
        trajectory_json_path: str | None = None,
    ):
        """
        Initialize patch extractor.

        Args:
            output_dir (str): mini-SWE-agent output directory.
            swe_problem_id (str): SWE problem identifier.
            repo_path (str | None): Optional repository path for git diff fallback.
            trajectory_json_path (str | None): Explicit path to the trajectory JSON
                file. If None, defaults to ``{output_dir}/{swe_problem_id}.traj.json``.
        """
        self.output_dir = output_dir
        self.swe_problem_id = swe_problem_id
        self.repo_path = repo_path
        self.trajectory_json_path = trajectory_json_path or os.path.join(
            output_dir, f"{swe_problem_id}.traj.json",
        )

    async def extract(self) -> str | None:
        """
        Extract patch using multiple strategies.

        Returns:
            str | None: Patch content string or None if no patch found.
        """
        # Strategy 1: Try trajectory JSON info.submission.
        patch = await self._try_trajectory_json()
        if patch:
            psrl_logger.info(f"Extracted patch from trajectory JSON ({len(patch)} chars).")
            return patch

        # Strategy 2: Fallback to git diff.
        if self.repo_path:
            patch = await self._try_git_diff()
            if patch:
                psrl_logger.info(f"Extracted patch from git diff ({len(patch)} chars).")
                return patch

        psrl_logger.warning("No patch found via any strategy.")
        return None

    async def _try_trajectory_json(self) -> str | None:
        """
        Try to read patch from mini-SWE-agent trajectory JSON file.

        mini-SWE-agent writes a JSON trajectory file with structure:
        ``{"info": {"submission": "<patch>", "exit_status": "..."}, "messages": [...]}``.
        """
        json_path = self.trajectory_json_path
        if not os.path.exists(json_path):
            psrl_logger.debug(f"Trajectory JSON not found: {json_path}.")
            return None

        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            submission = data.get("info", {}).get("submission", None)
            if submission and isinstance(submission, str) and submission.strip():
                psrl_logger.debug(f"Read patch from trajectory JSON: {json_path}.")
                return submission.strip()
        except Exception as e:
            psrl_logger.error(f"Failed to read trajectory JSON {json_path}: {e}.")

        return None

    async def _try_git_diff(self) -> str | None:
        """
        Try to extract patch using git diff.
        """
        if not self.repo_path or not os.path.isdir(self.repo_path):
            return None

        if not os.path.isdir(os.path.join(self.repo_path, ".git")):
            psrl_logger.debug(f"Not a git repository: {self.repo_path}.")
            return None

        # Try git diff HEAD first (includes staged + unstaged).
        patch = await self._run_git_diff("HEAD")
        if patch:
            return patch

        # Try git diff (unstaged only).
        patch = await self._run_git_diff()
        return patch

    async def _run_git_diff(self, ref: str | None = None) -> str | None:
        """
        Run git diff command.
        """
        cmd = ["git", "diff"]
        if ref:
            cmd.append(ref)

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30.0)

            if process.returncode == 0 and stdout:
                patch = stdout.decode("utf-8", errors="replace").strip()
                if patch:
                    ref_str = f" {ref}" if ref else ""
                    psrl_logger.debug(f"git diff{ref_str} returned {len(patch)} chars.")
                    return patch
        except asyncio.TimeoutError:
            psrl_logger.error("git diff timed out.")
        except Exception as e:
            psrl_logger.error(f"git diff failed: {e}.")

        return None
