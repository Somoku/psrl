"""Task-specific helpers shared by sandboxed coding harnesses."""

import logging
import shlex

from psrl.sandbox import SandboxSession

psrl_logger = logging.getLogger(__file__)

# Staged inside the sandbox, then fetched through the file API. The exec response is
# capped at `max_exec_output_bytes`, and a patch is user-controlled data that has no
# business going through a diagnostic channel: a truncated patch would grade an
# incomplete solution and report the result as if it were measured.
_HARNESS_PATCH_PATH = "/tmp/psrl-harness.patch"
# A repository read by git, not an interactive command, so one generous call is enough
# and every retry cost is the agent's work being wasted.
_PATCH_COLLECT_TIMEOUT_S = 600.0
# Size at which a patch stops being a code diff. The patch is still collected in full,
# because only the grader may judge it, but a repository that the agent filled with
# build output is worth reporting before it reaches the grading container.
_PATCH_SIZE_WARNING_BYTES = 64 * 1024 * 1024


def build_harness_prompt(problem_statement: str) -> str:
    """Render task and integrity constraints as the harness user prompt.

    The native ``problem_template`` starts the task with ``<pr_description>``.
    Keep that task boundary while leaving Claude Code's stock system prompt
    intact. Integrity constraints are part of the task instruction rather than
    a replacement system policy.
    """
    statement = problem_statement.strip()
    return (
        "<pr_description>\n"
        f"{statement}\n"
        "</pr_description>\n\n"
        "Implement the required changes in the current repository and verify the fix.\n\n"
        "Integrity rules:\n"
        "- Do not modify tests, pytest configuration, or evaluation harness files.\n"
        "- Do not retrieve a solution, patch, commit, or pull request from the task repository or its mirrors.\n"
        "- Do not create nested git repositories (directories containing a .git) inside the working directory; "
        "if you need a scratch repository to reproduce the issue, create it under /tmp instead."
    )


async def collect_git_patch(
    session: SandboxSession,
    workdir: str,
    base_commit: str | None = None,
    timeout_s: float = _PATCH_COLLECT_TIMEOUT_S,
) -> str:
    """Collect staged, unstaged, and untracked changes as one git patch.

    The patch is written to a file inside the sandbox and fetched with `read_bytes`,
    never returned as command output. Command output travels through a capped response
    channel, and the cap is a diagnostic safety limit: an oversized patch used to fail
    the episode and abort its whole training group, when the patch itself was complete
    and gradeable. Reading it back as a file keeps the full patch intact.

    Robust against nested git repositories that an agent may have created inside
    the workdir (e.g. ``test_repo/`` used to reproduce a DVC bug): a nested repo
    without a checked-out commit makes ``git add -A`` fail with
    ``error: 'test_repo/' does not have a commit checked out``. This collector
    never stages anything. It emits ``git diff --cached`` (index vs base/HEAD)
    plus ``git diff`` (worktree vs index) for tracked changes, then adds
    untracked files as ``new file`` diffs while excluding nested repositories.
    """
    base_arg = f"{shlex.quote(base_commit)} --" if base_commit else "--"
    patch_path = shlex.quote(_HARNESS_PATCH_PATH)
    script = (
        "set -u\n"
        f"out={patch_path}\n"
        ': > "$out"\n'
        # staged changes (index vs base / HEAD)
        f'git diff --cached --binary --submodule=diff {base_arg} >> "$out" 2>/dev/null || true\n'
        # unstaged changes (worktree vs index)
        'git diff --binary --submodule=diff -- >> "$out" 2>/dev/null || true\n'
        # untracked files, excluding nested git repositories (agent scratch repos)
        "git ls-files --others --exclude-standard -z | while IFS= read -r -d '' f; do\n"
        '    [ -z "$f" ] && continue\n'
        '    case "$f" in */) continue ;; esac\n'
        "    skip=0\n"
        '    d="$(dirname "$f")"\n'
        '    while [ "$d" != "." ] && [ "$d" != "/" ]; do\n'
        '        if [ -e "$d/.git" ]; then skip=1; break; fi\n'
        '        d="$(dirname "$d")"\n'
        "    done\n"
        '    [ "$skip" = 1 ] && continue\n'
        '    git diff --no-index --binary /dev/null "$f" >> "$out" 2>/dev/null || true\n'
        "done\n"
        'wc -c < "$out"\n'
    )
    result = await session.exec(script, cwd=workdir, timeout_s=timeout_s)
    if result.exit_code != 0:
        raise RuntimeError(f"Could not collect harness patch: {result.stderr.strip()}")
    try:
        payload = await session.read_bytes(_HARNESS_PATCH_PATH)
    except Exception as exc:
        raise RuntimeError(
            f"Could not read the collected harness patch from {_HARNESS_PATCH_PATH!r}: {exc!r}"
        ) from exc
    if len(payload) >= _PATCH_SIZE_WARNING_BYTES:
        psrl_logger.warning(
            "Harness patch for task %r is %d bytes. Grading it in full, but a patch this size is "
            "usually repository build output rather than a solution.",
            workdir,
            len(payload),
        )
    return payload.decode(errors="replace")
