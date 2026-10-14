"""The response-length contract: every response-aligned field agrees per row.

A batch pads each field to that field's own maximum, so a producer that builds a field from
the wrong length changes a width and nothing else. These tests pin that the shared check names
such a field at the boundary it crossed, and that it does not flag a field which is
legitimately padded wider than the mask.
"""

import pytest

torch = pytest.importorskip("torch")

from psrl.utils.transferqueue_utils import (  # noqa: E402
    assert_response_aligned_lengths,
    response_lengths,
)

pytestmark = pytest.mark.cpu_test


def _jagged(lengths: list[int]) -> "torch.Tensor":
    return torch.nested.as_nested_tensor([torch.arange(length) for length in lengths], layout=torch.jagged)


def test_matching_ragged_fields_pass() -> None:
    fields = {"responses": _jagged([3, 5]), "response_mask": _jagged([3, 5])}

    lengths = assert_response_aligned_lengths(fields, stage="test")

    assert lengths.tolist() == [3, 5]


def test_a_diverging_field_is_named_with_its_offending_rows() -> None:
    fields = {"responses": _jagged([3, 5]), "rm_scores": _jagged([3, 4])}

    with pytest.raises(AssertionError) as excinfo:
        assert_response_aligned_lengths(fields, stage="test")

    message = str(excinfo.value)
    assert "rm_scores" in message
    assert "(1, 5, 4)" in message


def test_a_padded_field_narrower_than_the_longest_response_is_flagged() -> None:
    """The production shape: one trajectory's length used for a whole session's rewards."""
    fields = {"responses": _jagged([3, 5]), "rm_scores": torch.zeros(2, 4)}

    with pytest.raises(AssertionError) as excinfo:
        assert_response_aligned_lengths(fields, stage="test")

    message = str(excinfo.value)
    assert "rm_scores" in message
    assert "(4, 5)" in message


def test_a_padded_field_at_least_as_wide_as_the_longest_response_passes() -> None:
    """A worker output padded to `responses` is legitimately wider than the mask."""
    fields = {"responses": _jagged([3, 5]), "log_probs": torch.zeros(2, 7)}

    assert assert_response_aligned_lengths(fields, stage="test").tolist() == [3, 5]


def test_an_absent_field_is_skipped() -> None:
    fields = {"responses": _jagged([3, 5]), "ref_log_prob": None, "values": None}

    assert assert_response_aligned_lengths(fields, stage="test").tolist() == [3, 5]


def test_single_trajectory_fields_compare_one_dimensionally() -> None:
    """The agent loop checks a trajectory before the batch pads it."""
    fields = {"responses": torch.zeros(4), "response_mask": torch.zeros(4), "loss_mask": torch.zeros(4)}

    assert assert_response_aligned_lengths(fields, stage="test").tolist() == [4]


def test_a_short_single_trajectory_field_is_flagged() -> None:
    fields = {"responses": torch.zeros(4), "loss_mask": torch.zeros(3)}

    with pytest.raises(AssertionError) as excinfo:
        assert_response_aligned_lengths(fields, stage="test")

    assert "loss_mask" in str(excinfo.value)


def test_a_list_of_rows_reads_as_lengths() -> None:
    fields = {"responses": [torch.zeros(3), torch.zeros(5)], "rm_scores": [torch.zeros(3), torch.zeros(5)]}

    assert assert_response_aligned_lengths(fields, stage="test").tolist() == [3, 5]


def test_a_missing_reference_is_reported() -> None:
    with pytest.raises(AssertionError, match="reference field"):
        assert_response_aligned_lengths({"response_mask": _jagged([3, 5])}, stage="test")


def test_a_different_row_count_is_reported() -> None:
    with pytest.raises(AssertionError, match="rows but"):
        assert_response_aligned_lengths({"responses": _jagged([3, 5]), "rm_scores": _jagged([3])}, stage="test")


def test_a_padded_rectangle_reports_no_per_row_length() -> None:
    assert response_lengths(torch.zeros(2, 4)) is None
    assert response_lengths(None) is None
