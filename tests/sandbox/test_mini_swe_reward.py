"""Reward compatibility tests for Mini-SWE recipes."""

from examples.mini_swe.reward import compute_score


def _extra_info(*, resolved: bool, gold_ceiling: float | None = None, failure_reason: str | None = None) -> dict:
    info: dict = {
        "num_turns": 3,
        "patch": "diff --git a/a.py b/a.py\n",
        "grader_result": {
            "resolved": resolved,
            "apply_ok": True,
            "f2p_pass": int(resolved),
            "f2p_total": 1,
            "failure_reason": failure_reason,
        },
    }
    if gold_ceiling is not None:
        info["swe_problem"] = {"gold_ceiling": gold_ceiling}
    return info


def test_legacy_binary_remains_signed_for_existing_recipes() -> None:
    failure = compute_score("swe_gym", "", {}, _extra_info(resolved=False), reward_mode="binary")

    assert failure["score"] == -1.0
    assert failure["acc"] == 0.0


def test_reward_reports_the_rows_gold_ceiling() -> None:
    # `acc` is only readable next to the best score the split allows, so every
    # graded sample carries the ceiling the gate froze into its row.
    lowered = compute_score("swe_gym", "", {}, _extra_info(resolved=False, gold_ceiling=0.0))
    raised = compute_score("swe_gym", "", {}, _extra_info(resolved=True, gold_ceiling=1.0))

    assert lowered["gold_ceiling"] == 0.0
    assert raised["gold_ceiling"] == 1.0
    assert raised["score"] == 1.0


def test_reward_flags_infrastructure_grading_failures() -> None:
    # An unreadable log scores 0, so it has to be distinguishable from a wrong
    # answer instead of silently reading as a model failure.
    broken = compute_score("swe_gym", "", {}, _extra_info(resolved=False, failure_reason="log_unparseable"))
    scored = compute_score("swe_gym", "", {}, _extra_info(resolved=False))

    assert broken["grading_failed"] == 1.0
    assert scored["grading_failed"] == 0.0
    assert broken["acc"] == scored["acc"] == 0.0


def test_ungated_row_reports_an_optimistic_ceiling() -> None:
    # A row prepared before the gate carries no ceiling, so assume it is solvable
    # rather than reporting a fabricated 0.0.
    result = compute_score("swe_gym", "", {}, _extra_info(resolved=False))

    assert result["gold_ceiling"] == 1.0
