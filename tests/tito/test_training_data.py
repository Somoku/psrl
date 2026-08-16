"""Tests for TITO training data builder."""

import base64
import io

import numpy as np
import pytest
import torch
from psrl.utils.tito.training_data import build_training_data


def _npy_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode()


def test_single_turn_no_trim():
    """Single turn: no env tokens, no trim needed."""
    accumulated = [1, 2, 3, 10, 11, 12]
    records = [
        {
            "prompt_token_count": 3,
            "output_logprobs": [[-0.5, 10], [-0.3, 11], [-0.1, 12]],
            "finish_reason": "stop",
        }
    ]
    result = build_training_data(accumulated, records)
    assert result["prompt_ids"] == [1, 2, 3]
    assert result["response_ids"] == [10, 11, 12]
    assert result["response_mask"] == [1, 1, 1]
    assert result["logprobs"] == [-0.5, -0.3, -0.1]


def test_two_turns_with_env_tokens():
    """Two turns with environment response between them."""
    accumulated = [1, 2, 3, 10, 11, 20, 21, 30, 31]
    records = [
        {
            "prompt_token_count": 3,
            "output_logprobs": [[-0.5, 10], [-0.3, 11]],
            "finish_reason": "tool_calls",
        },
        {
            "prompt_token_count": 7,
            "output_logprobs": [[-0.2, 30], [-0.1, 31]],
            "finish_reason": "stop",
        },
    ]
    result = build_training_data(accumulated, records)
    assert result["prompt_ids"] == [1, 2, 3]
    assert result["response_ids"] == [10, 11, 20, 21, 30, 31]
    assert result["response_mask"] == [1, 1, 0, 0, 1, 1]
    assert result["logprobs"] == [-0.5, -0.3, 0.0, 0.0, -0.2, -0.1]


def test_trailing_trim():
    """GLM47-style trailing stop token gets trimmed on non-last turn."""
    accumulated = [1, 2, 10, 11, 20, 30, 31]
    records = [
        {
            "prompt_token_count": 2,
            "output_logprobs": [[-0.5, 10], [-0.3, 11], [-0.9, 99]],
            "finish_reason": "tool_calls",
        },
        {
            "prompt_token_count": 5,
            "output_logprobs": [[-0.2, 30], [-0.1, 31]],
            "finish_reason": "stop",
        },
    ]
    result = build_training_data(accumulated, records, max_trim_tokens=1)
    assert result["response_ids"] == [10, 11, 20, 30, 31]
    assert result["response_mask"] == [1, 1, 0, 1, 1]
    assert result["logprobs"] == [-0.5, -0.3, 0.0, -0.2, -0.1]


def test_empty_records():
    result = build_training_data([], [])
    assert result["prompt_ids"] == []
    assert result["response_ids"] == []
    assert result["num_turns"] == 0


def test_no_logprobs_is_rejected():
    """Training reconstruction fails closed when capture omitted logprobs."""
    accumulated = [1, 2, 3]
    records = [
        {
            "prompt_token_count": 2,
            "output_logprobs": None,
            "finish_reason": "stop",
        }
    ]
    with pytest.raises(ValueError, match="Missing output_logprobs"):
        build_training_data(accumulated, records)


def test_output_token_mismatch_is_rejected():
    accumulated = [1, 2, 3]
    records = [
        {
            "prompt_token_count": 2,
            "output_logprobs": [[-0.1, 4]],
            "finish_reason": "stop",
        }
    ]
    with pytest.raises(ValueError, match="output tokens diverge"):
        build_training_data(accumulated, records)


def test_unconsumed_accumulated_tokens_are_rejected():
    accumulated = [1, 2, 3, 4]
    records = [
        {
            "prompt_token_count": 2,
            "output_logprobs": [[-0.1, 3]],
            "finish_reason": "stop",
        }
    ]
    with pytest.raises(ValueError, match="did not consume"):
        build_training_data(accumulated, records)


def test_routed_experts_none_when_absent():
    """No routed_experts in records → None (backward compatible)."""
    accumulated = [1, 2, 3, 10, 11, 12]
    records = [
        {
            "prompt_token_count": 3,
            "output_logprobs": [[-0.5, 10], [-0.3, 11], [-0.1, 12]],
            "finish_reason": "stop",
        }
    ]
    assert build_training_data(accumulated, records)["routed_experts"] is None


def test_routed_experts_cross_turn_assembly():
    """Per-turn RE blobs are placed at their absolute positions and tiled gap-free.

    Two turns; assembled length = 3 prompt + 6 response = 9 tokens, but the final
    sampled token has no RE, so the tensor has 8 rows.  Cross-turn prompt_start:
    turn 1 covers [0, 5), turn 2 covers [5, 8) — together filling all 8 rows.
    """
    accumulated = [1, 2, 3, 10, 11, 20, 21, 30, 31]
    num_layers, top_k = 2, 3
    t1 = np.arange(5 * num_layers * top_k, dtype=np.uint8).reshape(5, num_layers, top_k)
    t2 = np.arange(100, 100 + 3 * num_layers * top_k, dtype=np.uint8).reshape(3, num_layers, top_k)
    records = [
        {
            "prompt_token_count": 3,
            "output_logprobs": [[-0.5, 10], [-0.3, 11]],
            "finish_reason": "tool_calls",
            "routed_experts": {
                "data": _npy_b64(t1),
                "num_layers": num_layers,
                "top_k": top_k,
                "dtype": "uint8",
                "prompt_start": 0,
            },
        },
        {
            "prompt_token_count": 7,
            "output_logprobs": [[-0.2, 30], [-0.1, 31]],
            "finish_reason": "stop",
            "routed_experts": {
                "data": _npy_b64(t2),
                "num_layers": num_layers,
                "top_k": top_k,
                "dtype": "uint8",
                "prompt_start": 5,
            },
        },
    ]
    re = build_training_data(accumulated, records)["routed_experts"]
    # Canonical replay data covers all 9 tokens. The final never-forwarded row is filler.
    assert re.shape == (9, num_layers, top_k)
    assert re.dtype == torch.int16
    torch.testing.assert_close(re[0:5], torch.from_numpy(t1).to(torch.int16))
    torch.testing.assert_close(re[5:8], torch.from_numpy(t2).to(torch.int16))
    assert torch.count_nonzero(re[-1]) == 0


def test_routed_experts_ignores_prompt_start_gap():
    accumulated = [1, 2, 3, 10, 11]
    segment = np.ones((4, 1, 1), dtype=np.uint8)
    records = [
        {
            "prompt_token_count": 3,
            "output_logprobs": [[-0.5, 10], [-0.3, 11]],
            "finish_reason": "stop",
            "routed_experts": {
                "data": _npy_b64(segment),
                "num_layers": 1,
                "top_k": 1,
                "dtype": "uint8",
                "prompt_start": 1,
            },
        }
    ]

    routed_experts = build_training_data(accumulated, records)["routed_experts"]
    torch.testing.assert_close(routed_experts[:-1], torch.from_numpy(segment).to(torch.int16))


def test_routed_experts_ignores_prompt_start_overlap():
    accumulated = [1, 2, 10, 20, 30]
    first = np.ones((3, 1, 1), dtype=np.uint16)
    second = np.full((1, 1, 1), 2, dtype=np.uint16)
    records = [
        {
            "prompt_token_count": 2,
            "output_logprobs": [[-0.5, 10]],
            "finish_reason": "tool_calls",
            "routed_experts": {
                "data": _npy_b64(first),
                "num_layers": 1,
                "top_k": 1,
                "dtype": "uint16",
                "prompt_start": 0,
            },
        },
        {
            "prompt_token_count": 4,
            "output_logprobs": [[-0.2, 30]],
            "finish_reason": "stop",
            "routed_experts": {
                "data": _npy_b64(second),
                "num_layers": 1,
                "top_k": 1,
                "dtype": "uint16",
                "prompt_start": 2,
            },
        },
    ]

    routed_experts = build_training_data(accumulated, records)["routed_experts"]
    expected = np.concatenate((first, second), axis=0)
    torch.testing.assert_close(routed_experts[:-1], torch.from_numpy(expected).to(torch.int16))


def test_routed_experts_rejects_metadata_mismatch():
    accumulated = [1, 2, 10]
    segment = np.ones((2, 1, 1), dtype=np.uint8)
    records = [
        {
            "prompt_token_count": 2,
            "output_logprobs": [[-0.5, 10]],
            "finish_reason": "stop",
            "routed_experts": {
                "data": _npy_b64(segment),
                "num_layers": 2,
                "top_k": 1,
                "dtype": "uint8",
                "prompt_start": 0,
            },
        }
    ]

    with pytest.raises(ValueError, match="metadata mismatch"):
        build_training_data(accumulated, records)
