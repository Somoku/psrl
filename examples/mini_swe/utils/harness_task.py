"""Task-specific helpers shared by sandboxed coding harnesses."""

import logging
import shlex
from pathlib import PurePosixPath

from psrl.sandbox import (
    SandboxCollectorError,
    SandboxExitReason,
    SandboxPathNotFoundError,
    SandboxSession,
    SandboxSessionLostError,
    SandboxStatus,
)

psrl_logger = logging.getLogger(__file__)

# Staged inside the sandbox, then fetched through the file API. The exec response is
# capped at `max_exec_output_bytes`, and a patch is user-controlled data that has no
# business going through a diagnostic channel: a truncated patch would grade an
# incomplete solution and report the result as if it were measured.
_HARNESS_PATCH_NAME = "psrl-harness.patch"
# Collection diagnostics, written by the script and read back through the file API.
# The sandbox exec channel now carries stderr on the p3b backend, but the
# self-report also names which tool or directory was missing at each step, which
# a stderr tail does not, and it survives on backends whose exec channel still
# carries no stderr.
_HARNESS_DIAG_NAME = "psrl-harness-collect.diag"
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
    *,
    staging_dir: str,
    timeout_s: float = _PATCH_COLLECT_TIMEOUT_S,
) -> str:
    """Collect staged, unstaged, and untracked changes as one git patch.

    The patch is written to a file inside the sandbox and fetched with `read_bytes`,
    never returned as command output. Command output travels through a capped response
    channel, and the cap is a diagnostic safety limit: an oversized patch used to fail
    the episode and abort its whole training group, when the patch itself was complete
    and gradeable. Reading it back as a file keeps the full patch intact.

    ``staging_dir`` is where the patch and the collector's self-report are written. It is
    the harness's own state directory rather than a world-writable convention like
    ``/tmp``, for two reasons: the harness prompt tells the agent to use ``/tmp`` as its
    scratch space, so a framework payload staged there competes with agent-controlled
    files; and a sandbox whose ``/tmp`` is gone - which happens on a node under enough
    pressure that the container's filesystem is disturbed - reported a collectable patch
    as a failed collection, costing the episode and then its whole group. The script
    re-asserts the directory either way, so a directory the agent removed costs nothing.

    Robust against nested git repositories that an agent may have created inside
    the workdir (e.g. ``test_repo/`` used to reproduce a DVC bug): a nested repo
    without a checked-out commit makes ``git add -A`` fail with
    ``error: 'test_repo/' does not have a commit checked out``. This collector
    never stages anything. It emits ``git diff --cached`` (index vs base/HEAD)
    plus ``git diff`` (worktree vs index) for tracked changes, then adds
    untracked files as ``new file`` diffs while excluding nested repositories.
    """
    patch_path = str(PurePosixPath(staging_dir) / _HARNESS_PATCH_NAME)
    diag_path = str(PurePosixPath(staging_dir) / _HARNESS_DIAG_NAME)
    base_arg = f"{shlex.quote(base_commit)} --" if base_commit else "--"
    patch_arg = shlex.quote(patch_path)
    diag_arg = shlex.quote(diag_path)
    staging_arg = shlex.quote(staging_dir)
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
        f"out={patch_arg}\n"
        f"diag={diag_arg}\n"
        "export out diag\n"
        # The staging directory belongs to the harness, but nothing here may assume the
        # step that created it still ran or still holds: the agent has a shell in this
        # container too. One builtin buys independence from any directory it removed.
        f"mkdir -p {staging_arg}\n"
        ': > "$out"\n'
        ': > "$diag"\n'
        # Self-report so a non-zero exit is diagnosable without an stderr channel.
        "{\n"
        '    echo "shell=$0"\n'
        '    echo "cwd=$(pwd)"\n'
        f'    echo "wanted_cwd={workdir}"\n'
        f'    echo "staging_dir={staging_dir}"\n'
        "    for tool in git wc dirname xargs tr; do\n"
        '        printf "%s=%s\\n" "$tool" "$(command -v "$tool" 2>/dev/null || echo MISSING)"\n'
        "    done\n"
        f"    ls -ld {workdir_arg} {staging_arg} 2>&1 || true\n"
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
        diagnostics, unavailable = await _read_collector_diagnostics(session, diag_path)
        detail = (
            f"Could not collect harness patch (exit {result.exit_code}, cwd={workdir!r}): "
            f"{result.stderr.strip() or '<backend reported no stderr>'}\n"
            f"collector diagnostics:\n{diagnostics}"
        )
        # The collector's exit code and its own diagnostics are the only evidence
        # available for telling these apart, and they lead to different caller
        # responses:
        #
        # - 137 is SIGKILL. If the sandbox is no longer running, the kernel killed
        #   it for memory and the episode is a resource loss; if it is still
        #   running, a child was killed and the sandbox is usable.
        # - 126/127 mean the shell could not execute the collector's tools. The
        #   sandbox's execution environment is not what the harness needs.
        # - A diagnostics file that cannot be read at all, after a non-137 exit,
        #   means the collector's filesystem or shell never worked. Reporting the
        #   episode as a collector error would blame the harness for a broken
        #   sandbox.
        if result.exit_code == 137:
            if not await _sandbox_is_running(session):
                raise SandboxSessionLostError(
                    detail,
                    exit_reason=SandboxExitReason.OOM,
                )
            raise SandboxCollectorError(detail, exit_code=137, diagnostics=diagnostics)
        if unavailable:
            raise SandboxSessionLostError(
                f"Harness patch collection could not run in the sandbox: {detail}",
                exit_reason=SandboxExitReason.CONTAINER_EXITED,
            )
        if result.exit_code in (126, 127):
            raise SandboxCollectorError(
                detail, exit_code=result.exit_code, diagnostics=diagnostics
            )
        raise RuntimeError(detail)
    try:
        payload = await session.read_bytes(patch_path)
    except Exception as exc:
        raise RuntimeError(f"Could not read the collected harness patch from {patch_path!r}: {exc!r}") from exc
    if len(payload) >= _PATCH_SIZE_WARNING_BYTES:
        psrl_logger.warning(
            "Harness patch for task %r is %d bytes. Grading it in full, but a patch this size is "
            "usually repository build output rather than a solution.",
            workdir,
            len(payload),
        )
    return payload.decode(errors="replace")


async def _sandbox_is_running(session: SandboxSession) -> bool:
    """Report whether the sandbox survived a collector that was SIGKILLed.

    A 137 exit can be the sandbox dying or a child process being killed for
    memory inside a healthy one. The distinction decides whether the episode is
    a resource loss or a collector fault, so it is asked rather than guessed. A
    status call that fails is treated as "not running": the caller then replaces
    the sandbox, which is the safe direction when its state cannot be read.
    """
    try:
        return await session.status() == SandboxStatus.RUNNING
    except Exception:  # noqa: BLE001 - an unreadable status is not evidence of health
        return False


async def _read_collector_diagnostics(
    session: SandboxSession, diag_path: str
) -> tuple[str, bool]:
    """Read the collector's self-report after a non-zero exit.

    Returns the text and whether the report was *unavailable* rather than simply
    empty. The second value is what lets a caller distinguish "the collector ran
    and said why it failed" from "the sandbox could not produce the file at
    all" -- two failures that used to be the same string, and the second of which
    points at the sandbox rather than at the collector.

    Read as a file rather than from the exec response because the sandbox exec
    channel does not carry stderr on every backend. Failure to read is itself
    diagnostic, so it is reported inline instead of masking the collection error.
    """
    try:
        payload = await session.read_bytes(diag_path)
    except SandboxPathNotFoundError as exc:
        # A missing file in a live sandbox. The collector creates this file in
        # its first lines, so its absence means the script never got that far.
        return f"<missing: the collector never created its diagnostics file: {exc!r}>", True
    except Exception as exc:  # noqa: BLE001 - any read failure is diagnostic here
        # The file is the only record of which tool the sandbox was missing, so losing it
        # loses the diagnosis. `read_bytes` is one way to reach it and not the only one:
        # fall back to the exec channel, which survives on backends whose file reads do
        # not, and report both failures inline rather than in a log line nobody reads.
        read_error = f"read_bytes failed: {exc!r}"
        try:
            shown = await session.exec(f"cat {shlex.quote(diag_path)}", timeout_s=30)
        except Exception as exec_exc:
            return f"<unavailable: {read_error}; cat failed: {exec_exc!r}>", True
        if shown.exit_code != 0:
            return f"<unavailable: {read_error}; cat exited {shown.exit_code}>", True
        text = (shown.stdout or "").strip()
        return (text or f"<empty; {read_error}>"), not bool(text)
    text = payload.decode(errors="replace").strip()
    return (text or "<empty>"), not bool(text)
