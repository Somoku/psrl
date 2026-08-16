"""Build RL training data from a TITO trajectory.

Converts accumulated_token_ids + per-turn records (from SMG GET /tito/sessions)
into the canonical prompt, response, mask, and log-probability fields.
"""

import base64
import io
import logging
import os

import numpy as np
import torch

from psrl.utils.routed_experts import canonicalize_routed_experts, validate_routed_experts_array

psrl_logger = logging.getLogger(__name__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


def _assemble_routed_experts(
    records: list[dict],
    prompt_length: int,
    response_length: int,
) -> torch.Tensor | None:
    """Concatenate per-turn segments into the canonical tensor."""
    if not any(record.get("routed_experts") is not None for record in records):
        return None

    expected_shape: tuple[int, int] | None = None
    expected_dtype: np.dtype | None = None
    segments: list[np.ndarray] = []
    for turn_index, record in enumerate(records):
        routed_experts = record.get("routed_experts")
        if routed_experts is None:
            raise ValueError(f"Missing routed_experts segment at TITO turn {turn_index}.")
        encoded = base64.b64decode(routed_experts["data"])
        array = np.load(io.BytesIO(encoded))
        array = validate_routed_experts_array(array)

        metadata_shape = (int(routed_experts["num_layers"]), int(routed_experts["top_k"]))
        metadata_dtype = np.dtype(routed_experts["dtype"])
        if metadata_shape != array.shape[1:] or metadata_dtype != array.dtype:
            raise ValueError(
                f"routed_experts metadata mismatch at TITO turn {turn_index}: "
                f"metadata shape/dtype={metadata_shape!r}/{metadata_dtype!r}, "
                f"array shape/dtype={array.shape[1:]!r}/{array.dtype!r}."
            )
        if expected_shape is None:
            expected_shape = array.shape[1:]
            expected_dtype = array.dtype
        elif array.shape[1:] != expected_shape or array.dtype != expected_dtype:
            raise ValueError(
                f"routed_experts shape/dtype changed at TITO turn {turn_index}: "
                f"expected {expected_shape!r}/{expected_dtype!r}, got {array.shape[1:]!r}/{array.dtype!r}."
            )
        segments.append(array)

    compact = np.concatenate(segments, axis=0)
    return canonicalize_routed_experts(compact, prompt_length, response_length)


def build_training_data(
    accumulated_token_ids: list[int],
    records: list[dict],
    max_trim_tokens: int = 0,
    prompt_ids_override: list[int] | None = None,
) -> dict:
    """Convert one TITO trajectory into canonical training data.

    Args:
        accumulated_token_ids: Full token sequence from TITO store (prompt + all turns).
        records: Per-turn records, each with:
            - prompt_token_count: int
            - output_logprobs: list of [logprob, token_id] pairs (or None)
            - finish_reason: str
        max_trim_tokens: Maximum number of trailing boundary tokens allowed to be trimmed
            on non-last turns.  Must be 0 for the last turn (boundary tokens are part of
            the final output).  Sourced from the SMG GET endpoint ``max_trim_tokens``
            field (0 = DefaultAdapter; 1 = Qwen3/GLM4.7).  A ``ValueError`` is raised
            if the actual trim count exceeds this limit, which indicates a TITO merge
            bug rather than a normal boundary-token situation.
        prompt_ids_override: Optional initial prompt token ids rendered by the Python/verl
            path.  When provided, these ids are used as ``prompt_ids`` instead of slicing
            TITO ``accumulated_token_ids`` by the first record's ``prompt_token_count``.

    Returns:
        Dict with keys: prompt_ids, response_ids, response_mask, logprobs, num_turns.
    """
    if not records:
        return {
            "prompt_ids": [],
            "response_ids": [],
            "response_mask": [],
            "logprobs": [],
            "routed_experts": None,
            "num_turns": 0,
        }

    if not isinstance(max_trim_tokens, int) or max_trim_tokens < 0:
        raise ValueError(f"max_trim_tokens must be a non-negative integer, got {max_trim_tokens!r}.")

    all_response_ids: list[int] = []
    all_response_mask: list[int] = []
    all_logprobs: list[float] = []

    cursor = 0
    total_acc_len = len(accumulated_token_ids)
    for turn_index, record in enumerate(records):
        prompt_len = int(record.get("prompt_token_count", 0))
        if prompt_len < cursor or prompt_len > total_acc_len:
            raise ValueError(
                f"Invalid prompt boundary at TITO turn {turn_index}: "
                f"cursor={cursor}, prompt_token_count={prompt_len}, accumulated_length={total_acc_len}."
            )

        if turn_index > 0:
            env_ids = accumulated_token_ids[cursor:prompt_len]
            all_response_ids.extend(env_ids)
            all_response_mask.extend([0] * len(env_ids))
            all_logprobs.extend([0.0] * len(env_ids))

        raw_logprobs = record.get("output_logprobs", [])
        output_ids: list[int] = []
        output_logprobs: list[float] = []
        for pair in raw_logprobs:
            output_logprobs.append(float(pair[0]))
            output_ids.append(int(pair[1]))

        matched = 0
        for token_id in output_ids:
            position = prompt_len + matched
            if position >= total_acc_len or accumulated_token_ids[position] != token_id:
                break
            matched += 1

        is_last = turn_index == len(records) - 1
        trim_count = len(output_ids) - matched
        allowed_trim = 0 if is_last else max_trim_tokens
        if trim_count > allowed_trim:
            raise ValueError(
                f"TITO output tokens diverge at turn {turn_index}: "
                f"trim_count={trim_count} exceeds allowed={allowed_trim}."
            )

        output_ids = output_ids[:matched]
        output_logprobs = output_logprobs[:matched]
        all_response_ids.extend(output_ids)
        all_response_mask.extend([1] * matched)
        all_logprobs.extend(output_logprobs)
        cursor = prompt_len + matched

    if cursor != total_acc_len:
        raise ValueError(
            f"TITO reconstruction did not consume accumulated tokens: cursor={cursor}, "
            f"accumulated_length={total_acc_len}."
        )
    if not (len(all_response_ids) == len(all_response_mask) == len(all_logprobs)):
        raise ValueError("TITO response token, mask, and logprob lengths are not aligned.")

    first_prompt_len = records[0]["prompt_token_count"]
    prompt_ids = accumulated_token_ids[:first_prompt_len]
    if prompt_ids_override is not None:
        if len(prompt_ids_override) != first_prompt_len:
            raise ValueError(
                "prompt_ids_override length must match the captured TITO prompt length: "
                f"got {len(prompt_ids_override)} and {first_prompt_len}."
            )
        prompt_ids = list(prompt_ids_override)
    routed_experts = _assemble_routed_experts(records, len(prompt_ids), len(all_response_ids))

    psrl_logger.debug(
        "[TITO build_training_data] prompt_len=%d tito_prompt_len=%d response_len=%d "
        "mask_sum=%d logprobs_len=%d num_turns=%d total_acc_len=%d re_tokens=%s",
        len(prompt_ids),
        first_prompt_len,
        len(all_response_ids),
        sum(all_response_mask),
        len(all_logprobs),
        len(records),
        total_acc_len,
        None if routed_experts is None else routed_experts.shape[0],
    )

    if not all_response_ids and records:
        psrl_logger.error(
            "[TITO] build_training_data: response_ids empty but num_turns=%d! "
            "records=%s, accumulated_len=%d, prompt_len=%d",
            len(records),
            [
                {
                    "prompt_token_count": r.get("prompt_token_count"),
                    "output_logprobs_len": len(r.get("output_logprobs") or []),
                    "output_logprobs_type": type(r.get("output_logprobs")).__name__,
                    "finish_reason": r.get("finish_reason"),
                }
                for r in records
            ],
            total_acc_len,
            len(prompt_ids),
        )

    return {
        "prompt_ids": prompt_ids,
        "response_ids": all_response_ids,
        "response_mask": all_response_mask,
        "logprobs": all_logprobs,
        "routed_experts": routed_experts,
        "num_turns": len(records),
    }
