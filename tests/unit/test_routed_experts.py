import numpy as np
import pytest
import torch
from psrl.utils.routed_experts import canonicalize_routed_experts, validate_routed_experts_array


def test_canonicalize_compact_uint16_adds_only_final_filler():
    compact = np.arange(4 * 2 * 3, dtype=np.uint16).reshape(4, 2, 3)

    result = canonicalize_routed_experts(compact, prompt_length=3, response_length=2)

    assert result.shape == (5, 2, 3)
    assert result.dtype == torch.int16
    torch.testing.assert_close(result[:-1], torch.from_numpy(compact.astype(np.int16)))
    assert torch.count_nonzero(result[-1]) == 0


def test_canonicalize_accepts_full_length_coverage():
    """Full-length input is accepted, and the never-forwarded final row stays filler."""
    compact = np.ones((5, 1, 1), dtype=np.uint8)

    result = canonicalize_routed_experts(compact, prompt_length=3, response_length=2)

    assert result.shape == (5, 1, 1)
    assert torch.count_nonzero(result[:-1]) == 4
    assert torch.count_nonzero(result[-1]) == 0


def test_canonicalize_returns_none_without_payload():
    assert canonicalize_routed_experts(None, prompt_length=3, response_length=2) is None


def test_canonicalize_rejects_row_count_mismatch():
    compact = np.ones((3, 1, 1), dtype=np.uint8)

    with pytest.raises(ValueError, match="cover every forwarded"):
        canonicalize_routed_experts(compact, prompt_length=3, response_length=2)


def test_canonicalize_rejects_tensor_ids_outside_int16():
    compact = torch.full((4, 1, 1), np.iinfo(np.int16).max + 1, dtype=torch.int32)

    with pytest.raises(ValueError, match="non-negative int16"):
        canonicalize_routed_experts(compact, prompt_length=3, response_length=2)


def test_validate_rejects_non_contiguous_array():
    non_contiguous = np.zeros((4, 2, 3), dtype=np.uint8)[:, :, ::-1]

    with pytest.raises(ValueError, match="C-contiguous"):
        validate_routed_experts_array(non_contiguous)


def test_validate_accepts_c_contiguous_singleton_dimension_with_arbitrary_stride():
    storage = np.arange(12, dtype=np.uint8)
    routed_experts = np.lib.stride_tricks.as_strided(
        storage,
        shape=(4, 1, 3),
        strides=(3, 99, 1),
    )
    assert routed_experts.flags.c_contiguous

    assert validate_routed_experts_array(routed_experts) is routed_experts


def test_validate_rejects_wrong_dtype_and_shape():
    with pytest.raises(ValueError, match="3D"):
        validate_routed_experts_array(np.zeros((4, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="uint8 or uint16"):
        validate_routed_experts_array(np.zeros((4, 2, 3), dtype=np.int16))


def test_validate_rejects_uint16_values_that_overflow_verl_int16():
    routed_experts = np.array([[[np.iinfo(np.int16).max + 1]]], dtype=np.uint16)

    with pytest.raises(ValueError, match="cannot represent"):
        validate_routed_experts_array(routed_experts)
