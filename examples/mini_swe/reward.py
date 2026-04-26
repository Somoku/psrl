"""
mini-SWE-agent Reward Function for PSRL.

Reward structure for mini_swe_agent data sources (toy / simple-test):
  1.0       — exact patch match
  0.10-0.85 — partial patch match (file overlap + line similarity)
  0.05      — patch generated but wrong files / no patch but edited correct file
  0.03      — no patch, but ran tests or python verification
  0.02      — no patch, but model made edits (bash edits on wrong file)
  0.01      — no patch, but model explored code (cat/ls used)
  0.0       — no patch and no meaningful tool usage / 0 turns (timeout)
 -0.05      — long and fruitless (>=10 turns, no patch, no editor)
 -0.1       — premature submit without any tool usage (1-2 turns)

Reward structure for swebench_verified / swe_smith_py data sources:
  +1.0  — all FAIL_TO_PASS pass AND all PASS_TO_PASS still pass (resolved)
   0.0  — patch policy violated (modifies test / config files) [OpenClaw L990]
  -1.0  — patch submitted but not resolved [OpenClaw L1093-1095]
   0.0  — no patch submitted (not penalised; policy-violation reward)

The `acc` field (0/1 float, set in agent_data.finalize_output) is emitted
alongside `score` on wandb to track resolve_rate separately from the shaped
training signal.
"""

import logging
import os
import re
from typing import Any

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


# ---------------------------------------------------------------------------
# Patch comparison helpers
# ---------------------------------------------------------------------------


def normalize_patch(patch: str) -> str:
    """
    Normalize a patch string for comparison.
    """
    if not patch:
        return ""
    lines = [line.rstrip() for line in patch.strip().split("\n")]
    normalized_lines = []
    for line in lines:
        if line.startswith("index "):
            continue
        if not line.strip():
            continue
        normalized_lines.append(line)
    return "\n".join(normalized_lines)


def _extract_changed_files(patch: str) -> set[str]:
    """
    Extract set of changed files from a patch.
    """
    if not patch:
        return set()
    pattern = r"diff --git a/(.+?) b/(.+)"
    matches = re.findall(pattern, patch)
    return {match[1] for match in matches}


def _extract_changed_lines(patch: str) -> set[str]:
    """
    Extract the set of added/removed content lines from a patch.
    """
    lines: set[str] = set()
    if not patch:
        return lines
    for raw in patch.split("\n"):
        stripped = raw.strip()
        if stripped.startswith(
            ("+++", "---", "@@", "diff ", "index ", "similarity", "rename", "new file", "deleted")
        ):
            continue
        if stripped.startswith(("+", "-")):
            lines.add(stripped[1:].strip())
    return lines


def compare_patches(generated: str, expected: str) -> float:
    """
    Fine-grained patch comparison with line-level similarity.

    Scoring:
    - 0.0:  no patch generated
    - 0.05: patch generated but wrong files
    - 0.10 - 0.85: partial match (file overlap + line similarity)
    - 1.0:  exact match (after normalization)
    """
    if not generated:
        return 0.0

    gen_normalized = normalize_patch(generated)
    exp_normalized = normalize_patch(expected)

    if gen_normalized == exp_normalized:
        return 1.0

    gen_files = _extract_changed_files(generated)
    exp_files = _extract_changed_files(expected)

    if not exp_files:
        return 0.05 if gen_files else 0.0

    file_overlap = len(gen_files & exp_files) / len(exp_files)

    if file_overlap == 0:
        return 0.05

    gen_lines = _extract_changed_lines(generated)
    exp_lines = _extract_changed_lines(expected)
    if exp_lines:
        line_sim = len(gen_lines & exp_lines) / len(exp_lines)
    else:
        line_sim = 0.0

    combined = 0.4 * file_overlap + 0.6 * line_sim
    score = 0.10 + combined * 0.75
    return min(score, 0.85)


def _detect_tool_usage(solution_str: str) -> dict[str, bool]:
    """
    Detect which bash tools the model used from the decoded response.

    Heuristics target mini-SWE-agent v2 patterns (plain bash commands)
    rather than SWE-agent v1 tools (``str_replace``, ``submit``).
    """
    text = solution_str or ""
    return {
        "used_editor": bool(
            re.search(r"sed\s+-i", text)
            or "cat <<" in text
            or re.search(r"(echo|printf)\s+.*>", text)
            or "tee " in text
            or "patch " in text
        ),
        "used_cat": "cat " in text and "cat <<" not in text,
        "used_ls": "ls " in text or "ls\n" in text,
        "used_submit": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in text,
        "used_python": "python " in text or "python3 " in text,
        "used_test": "pytest" in text or "unittest" in text,
    }


def _targeted_correct_file(solution_str: str, expected_patch: str) -> bool:
    """
    Check if the model interacted with the correct file(s) from the expected patch.
    """
    target_files = _extract_changed_files(expected_patch)
    if not target_files:
        return False
    text = solution_str or ""
    return any(f in text for f in target_files)


# ---------------------------------------------------------------------------
# PSRL-compatible compute_score entry point
# ---------------------------------------------------------------------------


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **kwargs,
) -> float | dict[str, Any]:
    """
    Custom reward function for mini-SWE-agent with tool-use shaping.

    Returns:
        float: For toy data sources (``mini_swe_agent_simple``, ``mini_swe_agent``),
            returns a plain float reward in the range [-0.1, 1.0].
        dict[str, Any]: For SWE-bench data sources (``swebench_verified``,
            ``swe_smith_py``), returns ``{"score": float, "acc": float}`` so that
            `DAPORewardLoopManager` emits both the shaped training signal and the
            0/1 resolve_rate metric to wandb separately.
    """
    # --- SWE-bench Verified / SWE-smith-py: test-execution reward ---
    if data_source in ("swebench_verified", "swe_smith_py"):
        if extra_info is None:
            extra_info = {}
        grader_result: dict[str, Any] = extra_info.get("grader_result", {}) or {}

        # Treat num_turns == 0 as ABORTED (same as the simple branch).
        num_turns = int(extra_info.get("num_turns", 0) or 0)
        if num_turns == 0:
            psrl_logger.debug("[swe reward] score=0.0, acc=0.0 (0 turns / aborted).")
            return {"score": 0.0, "acc": 0.0}

        policy_violated = bool(grader_result.get("policy_violated", False))
        resolved = bool(grader_result.get("resolved", False))

        # Outcome-reward convention aligned with OpenClaw-RL (L1093-1095):
        # {-1, +1} with policy violations treated as 0 (not -1, OpenClaw L990).
        if policy_violated:
            reasons = grader_result.get("policy_reasons", [])
            psrl_logger.debug(
                f"[swe reward] score=0.0, acc=0.0 (policy violated), reasons={reasons!r}."
            )
            return {"score": 0.0, "acc": 0.0}

        if resolved:
            psrl_logger.debug("[swe reward] score=+1.0, acc=1.0 (resolved).")
            return {"score": 1.0, "acc": 1.0}

        # Not resolved and no grader result (grader was not invoked, e.g. no patch).
        if not grader_result:
            psrl_logger.debug(
                "[swe reward] score=-1.0, acc=0.0 (no grader result, patch not evaluated)."
            )
            return {"score": -1.0, "acc": 0.0}

        psrl_logger.debug(
            f"[swe reward] score=-1.0, acc=0.0 (not resolved), "
            f"apply_ok={grader_result.get('apply_ok')}, "
            f"f2p={grader_result.get('f2p_pass')}/{grader_result.get('f2p_total')}."
        )
        return {"score": -1.0, "acc": 0.0}

    # --- Toy / simple-test data sources: patch-overlap shaping ---
    if data_source not in ("mini_swe_agent_simple", "mini_swe_agent"):
        # Fallback for unknown data sources: return 0.0.
        return 0.0

    generated_patch = None
    num_turns = 0
    alignment_failed = False
    alignment_failure_reason = ""
    if extra_info is not None:
        generated_patch = extra_info.get("patch", None)
        num_turns = int(extra_info.get("num_turns", 0) or 0)
        alignment_failed = bool(extra_info.get("alignment_failed", False))
        alignment_failure_reason = str(extra_info.get("alignment_failure_reason", "") or "")

    if isinstance(ground_truth, dict):
        expected_patch = ground_truth.get("gold_patch") or ground_truth.get("ground_truth") or ""
    else:
        expected_patch = ground_truth or ""

    # Timeout: agent loop never started (Docker/connection failure).
    if num_turns == 0:
        psrl_logger.debug("[mini-SWE-agent reward] score=0.00 (0 turns / timeout).")
        return 0.0

    if alignment_failed:
        psrl_logger.debug(
            "[mini-SWE-agent reward] score=0.00 (alignment failed), "
            f"turns={num_turns}, reason={alignment_failure_reason or 'unknown'!r}."
        )
        return 0.0

    tools = _detect_tool_usage(solution_str)
    hit_correct_file = _targeted_correct_file(solution_str, expected_patch)

    # Patch was generated — use patch comparison.
    if generated_patch:
        score = compare_patches(generated_patch, expected_patch)
        psrl_logger.debug(
            f"[mini-SWE-agent reward] score={score:.2f}, patch_len={len(generated_patch)}, "
            f"turns={num_turns}, correct_file={hit_correct_file}, tools={tools}."
        )
        return score

    # No patch — shaped reward based on tool usage (graduated).
    if tools["used_editor"] and hit_correct_file:
        score = 0.05
    elif tools["used_python"] or tools["used_test"]:
        score = 0.03
    elif tools["used_editor"]:
        score = 0.02
    elif (tools["used_cat"] or tools["used_ls"]) and hit_correct_file:
        score = 0.02
    elif tools["used_cat"] or tools["used_ls"]:
        score = 0.01
    elif num_turns <= 2:
        score = -0.1
    else:
        score = 0.0

    # Long and fruitless: many turns but never even tried editing.
    if num_turns >= 10 and not tools["used_editor"] and score >= 0.0:
        score = -0.05

    psrl_logger.debug(
        f"[mini-SWE-agent reward] score={score:.2f} (no patch), turns={num_turns}, "
        f"correct_file={hit_correct_file}, tools={tools}."
    )
    return score
