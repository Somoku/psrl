"""Task-specific helpers shared by sandboxed coding harnesses."""

from psrl.sandbox import SandboxSession


def build_harness_prompt(problem_statement: str) -> str:
    """Wrap a SWE problem in instructions suitable for autonomous coding CLIs."""
    return (
        "Work on the software task below in the current repository. Inspect the code, implement a robust fix, "
        "and run relevant tests. Do not merely describe a solution; edit the working tree.\n\n"
        f"{problem_statement.strip()}"
    )


async def collect_git_patch(session: SandboxSession, workdir: str, timeout_s: float = 60.0) -> str:
    """Collect tracked and untracked changes as one grader-compatible patch."""
    command = "git add -N . && git diff --binary HEAD -- ."
    result = await session.exec(command, cwd=workdir, timeout_s=timeout_s)
    if result.exit_code != 0:
        raise RuntimeError(f"Could not collect harness patch: {result.stderr.strip()}")
    return result.stdout
