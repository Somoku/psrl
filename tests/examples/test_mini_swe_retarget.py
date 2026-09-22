"""Retargeting a prepared SWE split at repaired problem images."""

import pandas as pd
import pytest
from examples.mini_swe.prepare.retarget_problem_images import (
    apply_overrides,
    parse_overrides,
    plan_images,
)

pytestmark = pytest.mark.cpu_test


def _row(instance_id: str, image: str, repo: str) -> dict:
    return {
        "swe_problem_image": image,
        "sandbox_overrides": {"environment": {"image": image}},
        "swe_problem": {"instance_id": instance_id, "repo": repo},
    }


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "extra_info": [
                _row("a__a-1", "repo/a:1", "a/a"),
                _row("a__a-2", "repo/a:1", "a/a"),
                _row("b__b-1", "repo/b:1", "b/b"),
            ]
        }
    )


def test_plan_groups_instances_behind_one_image() -> None:
    assert plan_images(_frame(), None) == [
        ("repo/a:1", "a/a", ["a__a-1", "a__a-2"]),
        ("repo/b:1", "b/b", ["b__b-1"]),
    ]


def test_plan_honours_the_instance_filter() -> None:
    assert plan_images(_frame(), "a__a-2") == [("repo/a:1", "a/a", ["a__a-2"])]


def test_plan_rejects_a_shared_image_from_two_repositories() -> None:
    frame = _frame()
    frame.at[2, "extra_info"]["swe_problem_image"] = "repo/a:1"
    frame.at[2, "extra_info"]["sandbox_overrides"]["environment"]["image"] = "repo/a:1"

    with pytest.raises(ValueError, match="shared by"):
        plan_images(frame, None)


def test_plan_requires_an_image_on_every_selected_row() -> None:
    frame = _frame()
    frame.at[0, "extra_info"]["swe_problem_image"] = ""
    frame.at[0, "extra_info"]["sandbox_overrides"]["environment"]["image"] = ""

    with pytest.raises(ValueError, match="no problem image"):
        plan_images(frame, "a__a-1")


def test_plan_falls_back_to_the_rollout_image_field() -> None:
    frame = _frame()
    frame.at[0, "extra_info"]["swe_problem_image"] = ""

    assert plan_images(frame, "a__a-1") == [("repo/a:1", "a/a", ["a__a-1"])]


def test_overrides_retarget_the_grader_image_only_by_default() -> None:
    # Grading is what a missing dependency blocks, so the rollout image keeps
    # pointing at the base until the caller asks for it.
    updated, summary = apply_overrides(_frame(), {"repo/a:1": "repaired/a:9"}, include_rollout_image=False)
    extra = updated.at[0, "extra_info"]

    assert extra["swe_problem_image"] == "repaired/a:9"
    assert extra["sandbox_overrides"]["environment"]["image"] == "repo/a:1"
    assert updated.at[1, "extra_info"]["swe_problem_image"] == "repaired/a:9"
    assert updated.at[2, "extra_info"]["swe_problem_image"] == "repo/b:1"
    assert summary["rows"] == 2
    assert summary["rollout_rows"] == 0


def test_overrides_can_retarget_the_rollout_image_too() -> None:
    updated, summary = apply_overrides(_frame(), {"repo/a:1": "repaired/a:9"}, include_rollout_image=True)

    assert updated.at[0, "extra_info"]["sandbox_overrides"]["environment"]["image"] == "repaired/a:9"
    assert summary["rollout_rows"] == 2


def test_parse_overrides_reads_a_tab_separated_map(tmp_path) -> None:
    path = tmp_path / "map.tsv"
    path.write_text("# base\trepaired\nrepo/a:1\trepaired/a:9\n\n")

    assert parse_overrides(path) == {"repo/a:1": "repaired/a:9"}


@pytest.mark.parametrize("line", ["repo/a:1", "repo/a:1\t", "\tpaired", "a\tb\tc"])
def test_parse_overrides_rejects_malformed_lines(tmp_path, line: str) -> None:
    path = tmp_path / "map.tsv"
    path.write_text(f"{line}\n")

    with pytest.raises(ValueError):
        parse_overrides(path)
