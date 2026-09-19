"""Build RL training data from a TITO trajectory.

Converts accumulated_token_ids + per-turn records (from SMG GET /tito/sessions)
into the canonical prompt, response, mask, and log-probability fields.
"""

from __future__ import annotations

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
    """Concatenate per-turn routed-expert segments into the canonical int16 tensor."""
    if not any(record.get("routed_experts") is not None for record in records):
        return None

    expected_shape: tuple[int, ...] | None = None
    expected_dtype: np.dtype | None = None
    segments: list[np.ndarray] = []
    for turn_index, record in enumerate(records):
        routed_experts = record.get("routed_experts")
        if routed_experts is None:
            raise ValueError(f"Missing routed_experts segment at TITO turn {turn_index}.")
        array = np.load(io.BytesIO(base64.b64decode(routed_experts["data"])))
        array = validate_routed_experts_array(array)

        metadata_shape = (int(routed_experts["num_layers"]), int(routed_experts["top_k"]))
        metadata_dtype = np.dtype(routed_experts["dtype"])
        if metadata_shape != array.shape[1:] or metadata_dtype != array.dtype:
            raise ValueError(
                f"routed_experts metadata mismatch at TITO turn {turn_index}, "
                f"metadata shape/dtype={metadata_shape!r}/{metadata_dtype!r}, "
                f"array shape/dtype={array.shape[1:]!r}/{array.dtype!r}."
            )
        if expected_shape is None:
            expected_shape = array.shape[1:]
            expected_dtype = array.dtype
        elif array.shape[1:] != expected_shape or array.dtype != expected_dtype:
            raise ValueError(
                f"routed_experts shape or dtype changed at TITO turn {turn_index}, "
                f"expected {expected_shape!r}/{expected_dtype!r}, "
                f"got {array.shape[1:]!r}/{array.dtype!r}."
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
        max_trim_tokens: Maximum number of trailing boundary tokens tolerated on non-last turns.
            A final turn must align with the accumulated stream exactly. Sourced from the SMG
            session's ``max_trim_tokens`` field. A divergence beyond the ceiling raises
            ``ValueError`` instead of being clamped, because it indicates a TITO merge bug rather
            than a normal boundary-token situation.
        prompt_ids_override: Optional initial prompt token ids rendered by the Python/verl
            path. When provided, these ids are used as ``prompt_ids`` instead of slicing
            TITO ``accumulated_token_ids`` by the first record's ``prompt_token_count``.

    Returns:
        Dict with keys: prompt_ids, response_ids, response_mask, logprobs, num_turns.

    Raises:
        ValueError: If a turn contributed tokens to ``accumulated_token_ids`` but its
            record carries no ``output_logprobs``, or if the fields appended per turn
            drift apart. Both indicate a gateway/TITO contract violation.
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

    all_response_ids: list[int] = []
    all_response_mask: list[int] = []
    all_logprobs: list[float] = []

    cursor = 0
    total_acc_len = len(accumulated_token_ids)
    for i, record in enumerate(records):
        prompt_len = record["prompt_token_count"]
        raw_lps = record.get("output_logprobs") or []

        # Environment/user tokens between previous cursor and this turn's prompt end
        if cursor > 0 and prompt_len > cursor:
            env_ids = accumulated_token_ids[cursor:prompt_len]
            all_response_ids.extend(env_ids)
            all_response_mask.extend([0] * len(env_ids))
            all_logprobs.extend([0.0] * len(env_ids))

            psrl_logger.debug(
                "[TITO turn %d] env_ids: cursor=%d, prompt_len=%d, env_count=%d, env_ids[:5]=%s",
                i,
                cursor,
                prompt_len,
                len(env_ids),
                str(env_ids[:5]),
            )

        # Assistant output tokens. `output_logprobs` is the only source of the sampled tokens, so
        # a turn with accumulated tokens beyond its prompt but no logprobs cannot be reconstructed.
        output_ids = [int(pair[1]) for pair in raw_lps]
        output_logprobs = [float(pair[0]) for pair in raw_lps]
        is_last = i == len(records) - 1
        if not output_ids and prompt_len < total_acc_len:
            raise ValueError(f"Missing output_logprobs at TITO turn {i}.")

        # Greedy match the sampled output against the authoritative stream. The divergent suffix is
        # the boundary tokens the template re-renders as the next turn's prefix, so it must be trimmed.
        matched = 0
        for j, token_id in enumerate(output_ids):
            position = prompt_len + j
            if position < total_acc_len and token_id == accumulated_token_ids[position]:
                matched += 1
            else:
                break
        trim_count = len(output_ids) - matched
        allowed = 0 if is_last else max_trim_tokens
        if trim_count > allowed:
            raise ValueError(
                f"TITO output tokens diverge at turn {i}: trim_count={trim_count} exceeds "
                f"allowed={allowed} (is_last={is_last}, max_trim_tokens={max_trim_tokens}). "
                f"output_ids[-3:]={output_ids[-3:]}, "
                f"accumulated[{prompt_len + matched}:{prompt_len + matched + 3}]="
                f"{accumulated_token_ids[prompt_len + matched : prompt_len + matched + 3]}"
            )
        if trim_count > 0:
            output_ids = output_ids[:matched]
            output_logprobs = output_logprobs[:matched]
            psrl_logger.debug(
                "[TITO turn %d] trimmed %d trailing boundary tokens, remaining output_len=%d",
                i,
                trim_count,
                len(output_ids),
            )

        all_response_ids.extend(output_ids)
        all_response_mask.extend([1] * len(output_ids))
        all_logprobs.extend(output_logprobs)

        cursor = prompt_len + len(output_ids)

    if cursor != total_acc_len:
        raise ValueError(
            "TITO reconstruction did not consume the accumulated token stream: "
            f"cursor={cursor}, accumulated_length={total_acc_len}."
        )

    first_prompt_len = records[0]["prompt_token_count"]
    prompt_ids = (
        list(prompt_ids_override) if prompt_ids_override is not None else accumulated_token_ids[:first_prompt_len]
    )
    routed_experts = _assemble_routed_experts(records, len(prompt_ids), len(all_response_ids))

    # These fields are extended together and indexed interchangeably downstream.
    if not (len(all_response_ids) == len(all_response_mask) == len(all_logprobs)):
        raise AssertionError(
            f"[TITO build_training_data] length drift over {len(records)} turns: "
            f"response_ids={len(all_response_ids)} response_mask={len(all_response_mask)} "
            f"logprobs={len(all_logprobs)}. These are appended in lockstep per turn, so a "
            "mismatch means one append site diverged from the others."
        )

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
