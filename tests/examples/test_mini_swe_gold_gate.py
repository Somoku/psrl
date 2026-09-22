"""Gold-gate verdicts for a prepared SWE split."""

import numpy as np
import pandas as pd
import pytest
from examples.mini_swe.prepare.gold_gate import (
    GOLD_CEILING_KEY,
    PRUNED_P2P_KEY,
    load_rows,
    prunable_p2p,
    prune_frame,
    run_resolves,
    summarize_runs,
)

pytestmark = pytest.mark.cpu_test


def _run(f2p_failed: list[str], p2p_failed: list[str], failure_reason: str | None = None) -> dict:
    return {"f2p_failed": f2p_failed, "p2p_failed": p2p_failed, "failure_reason": failure_reason}


def test_prunable_p2p_keeps_only_tests_that_fail_in_every_run() -> None:
    # One flaky observation must not prune a real expectation.
    assert prunable_p2p([["a", "b"], ["b", "c"], ["b"]]) == ["b"]
    assert prunable_p2p([["a"], ["b"]]) == []
    assert prunable_p2p([]) == []


def test_a_pruned_failure_no_longer_blocks_resolution() -> None:
    # This is the pydicom case: the image cannot satisfy two of the frozen
    # pass-to-pass expectations, so they must stop deciding the reward.
    assert run_resolves(_run([], ["env_broken"]), ["env_broken"]) is True
    assert run_resolves(_run([], ["env_broken", "regressed"]), ["env_broken"]) is False


def test_an_infrastructure_failure_is_never_usable() -> None:
    assert run_resolves(_run([], [], "log_unparseable"), []) is False
    verdict = summarize_runs([_run([], [], "log_unparseable")])

    assert verdict["passed"] is False
    assert verdict["gold_ceiling"] == 0.0
    assert verdict["pruned_pass_to_pass"] == []


def test_a_consistent_success_sets_the_ceiling() -> None:
    verdict = summarize_runs([_run([], [], None), _run([], [], None)])

    assert verdict["passed"] is True
    assert verdict["flaky"] is False
    assert verdict["gold_ceiling"] == 1.0


def test_disagreeing_repeats_are_flaky_and_unusable() -> None:
    # A sample whose gold patch resolves sometimes corrupts the reward signal, so
    # flakiness has to lower the ceiling instead of averaging away.
    verdict = summarize_runs([_run([], [], None), _run(["f2p_failed"], [], None)])

    assert verdict["flaky"] is True
    assert verdict["passed"] is False
    assert verdict["gold_ceiling"] == 0.0


def test_a_failing_gold_patch_is_not_usable() -> None:
    verdict = summarize_runs([_run(["f2p_failed"], [], None)])

    assert verdict["passed"] is False
    assert verdict["gold_ceiling"] == 0.0


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "extra_info": [
                {
                    "swe_problem_image": "repo/img:1",
                    "swe_problem": {
                        "instance_id": "a__a-1",
                        "repo": "a/a",
                        "patch": "diff --git a/x b/x",
                        "eval_script": "pytest",
                        "FAIL_TO_PASS": ["t_f2p"],
                        "PASS_TO_PASS": ["t_ok", "t_env"],
                    },
                }
            ]
        }
    )


def test_prune_frame_drops_the_unrunnable_tests_and_records_the_ceiling() -> None:
    verdict = {
        "instance_id": "a__a-1",
        "pruned_pass_to_pass": ["t_env"],
        "gold_ceiling": 1.0,
    }

    updated = prune_frame(_frame(), [verdict])
    problem = updated.at[0, "extra_info"]["swe_problem"]

    assert problem["PASS_TO_PASS"] == ["t_ok"]
    assert problem[PRUNED_P2P_KEY] == ["t_env"]
    assert problem[GOLD_CEILING_KEY] == 1.0
    assert problem["FAIL_TO_PASS"] == ["t_f2p"]


def test_prune_frame_does_not_mutate_the_input_frame() -> None:
    frame = _frame()

    prune_frame(frame, [{"instance_id": "a__a-1", "pruned_pass_to_pass": ["t_env"], "gold_ceiling": 1.0}])

    assert frame.at[0, "extra_info"]["swe_problem"]["PASS_TO_PASS"] == ["t_ok", "t_env"]


def test_prune_frame_ignores_pruned_tests_outside_the_row() -> None:
    updated = prune_frame(
        _frame(),
        [{"instance_id": "a__a-1", "pruned_pass_to_pass": ["not_expected"], "gold_ceiling": 1.0}],
    )
    problem = updated.at[0, "extra_info"]["swe_problem"]

    assert problem["PASS_TO_PASS"] == ["t_ok", "t_env"]
    assert PRUNED_P2P_KEY not in problem


def test_prune_frame_survives_a_parquet_round_trip(tmp_path) -> None:
    # Pandas reads a parquet list column back as an array, and `array or []`
    # raises. The gate only ever sees the round-tripped frame.
    parquet = tmp_path / "split.parquet"
    _frame().to_parquet(parquet)
    frame = pd.read_parquet(parquet)
    assert isinstance(frame.at[0, "extra_info"]["swe_problem"]["PASS_TO_PASS"], np.ndarray)

    updated = prune_frame(
        frame,
        [{"instance_id": "a__a-1", "pruned_pass_to_pass": ["t_env"], "gold_ceiling": 1.0}],
    )

    updated.to_parquet(tmp_path / "pruned.parquet")
    reread = pd.read_parquet(tmp_path / "pruned.parquet").at[0, "extra_info"]["swe_problem"]
    assert list(reread["PASS_TO_PASS"]) == ["t_ok"]
    assert list(reread[PRUNED_P2P_KEY]) == ["t_env"]
    assert reread[GOLD_CEILING_KEY] == 1.0


def test_load_rows_rejects_a_row_without_a_gold_patch(tmp_path) -> None:
    frame = _frame()
    frame.at[0, "extra_info"]["swe_problem"]["patch"] = ""
    parquet = tmp_path / "split.parquet"
    frame.to_parquet(parquet)

    with pytest.raises(ValueError, match="no gold patch"):
        load_rows(parquet, None)


def test_load_rows_filters_by_instance_id(tmp_path) -> None:
    frame = _frame()
    parquet = tmp_path / "split.parquet"
    frame.to_parquet(parquet)

    assert [problem["instance_id"] for _, problem, _ in load_rows(parquet, "a__a-1")] == ["a__a-1"]
    with pytest.raises(ValueError, match="No rows"):
        load_rows(parquet, "b__b-1")
