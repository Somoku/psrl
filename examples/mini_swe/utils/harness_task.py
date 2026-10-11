"""Task-specific helpers shared by sandboxed coding harnesses."""

import logging
import shlex

from psrl.sandbox import SandboxSession, SandboxSessionLostError

psrl_logger = logging.getLogger(__file__)

# Staged inside the sandbox, then fetched through the file API. The exec response is
# capped at `max_exec_output_bytes`, and a patch is user-controlled data that has no
# business going through a diagnostic channel: a truncated patch would grade an
# incomplete solution and report the result as if it were measured.
_HARNESS_PATCH_PATH = "/tmp/psrl-harness.patch"
# Collection diagnostics, written by the script and read back through the file API.
# The sandbox exec channel does not carry stderr on every backend (the p3b node agent
# returns an empty stderr), so a failing collection would otherwise report only an
# exit code with no way to tell which step failed.
_HARNESS_DIAG_PATH = "/tmp/psrl-harness-collect.diag"
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
    diag_path = shlex.quote(_HARNESS_DIAG_PATH)
    workdir_arg = shlex.quote(workdir)
    # Per-file body for the untracked pass. It is executed by `sh` (via `xargs -0`),
    # not by a `while read -d` loop: the sandbox runs commands with `/bin/sh`, and
    # dash's `read` has no `-d`, so the bash form silently skipped every untracked
    # file while still exiting 0.
    untracked_body = (
        "for f do\n"
        '    [ -z "$f" ] && continue\n'
        '    case "$f" in */) continue ;; esac\n'
        "    skip=0\n"
        '    d=$(dirname "$f")\n'
        '    while [ "$d" != "." ] && [ "$d" != "/" ]; do\n'
        '        if [ -e "$d/.git" ]; then skip=1; break; fi\n'
        '        d=$(dirname "$d")\n'
        "    done\n"
        '    if [ "$skip" = 1 ]; then continue; fi\n'
        # `--no-index` exits 1 when the files differ, which is the normal case here.
        '    git diff --no-index --binary /dev/null -- "$f" >> "$out" 2>>"$diag" || true\n'
        "done\n"
    )
    script = (
        "set -u\n"
        f"out={patch_path}\n"
        f"diag={diag_path}\n"
        "export out diag\n"
        ': > "$out"\n'
        ': > "$diag"\n'
        # Self-report so a non-zero exit is diagnosable without an stderr channel.
        "{\n"
        '    echo "shell=$0"\n'
        '    echo "cwd=$(pwd)"\n'
        f'    echo "wanted_cwd={workdir}"\n'
        "    for tool in git wc dirname xargs tr; do\n"
        '        printf "%s=%s\\n" "$tool" "$(command -v "$tool" 2>/dev/null || echo MISSING)"\n'
        "    done\n"
        f"    ls -ld {workdir_arg} 2>&1 || true\n"
        '} >> "$diag" 2>&1\n'
        # staged changes (index vs base / HEAD)
        f'git diff --cached --binary --submodule=diff {base_arg} >> "$out" 2>>"$diag" '
        '|| echo "cached_diff_rc=$?" >> "$diag"\n'
        # unstaged changes (worktree vs index)
        'git diff --binary --submodule=diff -- >> "$out" 2>>"$diag" '
        '|| echo "worktree_diff_rc=$?" >> "$diag"\n'
        'echo "untracked=$(git ls-files --others --exclude-standard -z | tr -cd "\\000" | wc -c)" '
        '>> "$diag"\n'
        # untracked files, excluding nested git repositories (agent scratch repos)
        f"git ls-files --others --exclude-standard -z | xargs -0 sh -c {shlex.quote(untracked_body)} sh\n"
        'echo "collector=done" >> "$diag"\n'
        'wc -c < "$out"\n'
    )
    try:
        result = await session.exec(script, cwd=workdir, timeout_s=timeout_s)
    except SandboxSessionLostError:
        # The sandbox is gone, so there is no patch to collect and never will be. Reported
        # as itself rather than as a patch-collection failure: the worker classifies it as
        # an infrastructure fault, which replaces the group without counting the episode
        # against the harness. Raising a bare RuntimeError here instead is what made a
        # released lease look like a broken collector.
        psrl_logger.warning(
            "Sandbox was lost before the harness patch could be collected, so this episode has no patch.",
            exc_info=True,
        )
        raise
    if result.exit_code != 0:
        diagnostics = await _read_collector_diagnostics(session)
        raise RuntimeError(
            f"Could not collect harness patch (exit {result.exit_code}, cwd={workdir!r}): "
            f"{result.stderr.strip() or '<backend reported no stderr>'}\n"
            f"collector diagnostics:\n{diagnostics}"
        )
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


async def _read_collector_diagnostics(session: SandboxSession) -> str:
    """Read the collector's self-report after a non-zero exit.

    Read as a file rather than from the exec response because the sandbox exec channel
    does not carry stderr on every backend. Failure to read is itself diagnostic, so it
    is reported inline instead of masking the collection error.
    """
    try:
        payload = await session.read_bytes(_HARNESS_DIAG_PATH)
    except Exception:
        psrl_logger.warning(
            "Could not read the patch-collector diagnostics from %r.", _HARNESS_DIAG_PATH, exc_info=True
        )
        return "<unavailable>"
    return payload.decode(errors="replace").strip() or "<empty>"
