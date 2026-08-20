"""Task-specific helpers shared by sandboxed coding harnesses."""

from psrl.sandbox import SandboxSession


def build_harness_prompt(problem_statement: str) -> str:
    """Render the same task boundary used by the native mini-SWE prompt.

    The native ``problem_template`` starts the task with ``<pr_description>``.
    Keep the task payload itself identical for Claude Code/Codex; their
    interaction protocol belongs in the adapter's system prompt instead of
    being mixed into the dataset problem text.
    """
    statement = problem_statement.strip()
    return (
        "<pr_description>\n"
        f"{statement}\n"
        "</pr_description>\n\n"
        "Implement the required changes in the current repository and verify the fix."
    )


async def collect_git_patch(session: SandboxSession, workdir: str, timeout_s: float = 60.0) -> str:
    """Collect tracked and untracked changes as one grader-compatible patch."""
    command = "git add -N . && git diff --binary HEAD -- ."
    result = await session.exec(command, cwd=workdir, timeout_s=timeout_s)
    if result.exit_code != 0:
        raise RuntimeError(f"Could not collect harness patch: {result.stderr.strip()}")
    return result.stdout
