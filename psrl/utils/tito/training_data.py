"""Build RL training arrays from TITO session data.

Converts accumulated_token_ids + per-turn records (from SMG GET /v1/tito/sessions)
into prompt_ids, response_ids, response_mask, and logprobs arrays.
"""

from __future__ import annotations


def build_training_arrays(
    accumulated_token_ids: list[int],
    records: list[dict],
) -> dict:
    """Convert TITO session data into training arrays.

    Args:
        accumulated_token_ids: Full token sequence from TITO store (prompt + all turns).
        records: Per-turn records, each with:
            - prompt_token_count: int
            - output_logprobs: list of [logprob, token_id] pairs (or None)
            - finish_reason: str

    Returns:
        Dict with keys: prompt_ids, response_ids, response_mask, logprobs, num_turns.
    """
    if not records:
        return {
            "prompt_ids": [],
            "response_ids": [],
            "response_mask": [],
            "logprobs": [],
            "num_turns": 0,
        }

    all_response_ids: list[int] = []
    all_response_mask: list[int] = []
    all_logprobs: list[float] = []

    cursor = 0
    for i, record in enumerate(records):
        prompt_len = record["prompt_token_count"]
        raw_lps = record.get("output_logprobs") or []

        # Environment/user tokens between previous cursor and this turn's prompt end
        if cursor > 0 and prompt_len > cursor:
            env_ids = accumulated_token_ids[cursor:prompt_len]
            all_response_ids.extend(env_ids)
            all_response_mask.extend([0] * len(env_ids))
            all_logprobs.extend([0.0] * len(env_ids))

        # Assistant output tokens
        output_ids = [int(pair[1]) for pair in raw_lps]
        output_logprobs = [float(pair[0]) for pair in raw_lps]

        # Trailing trim for non-last turns: greedy match against accumulated
        is_last = i == len(records) - 1
        if not is_last and output_ids:
            matched = 0
            for j, tid in enumerate(output_ids):
                pos = prompt_len + j
                if pos < len(accumulated_token_ids) and tid == accumulated_token_ids[pos]:
                    matched += 1
                else:
                    break
            trim_count = len(output_ids) - matched
            if trim_count > 0:
                output_ids = output_ids[:matched]
                output_logprobs = output_logprobs[:matched]

        all_response_ids.extend(output_ids)
        all_response_mask.extend([1] * len(output_ids))
        all_logprobs.extend(output_logprobs)

        cursor = prompt_len + len(output_ids)

    first_prompt_len = records[0]["prompt_token_count"]
    prompt_ids = accumulated_token_ids[:first_prompt_len]

    return {
        "prompt_ids": prompt_ids,
        "response_ids": all_response_ids,
        "response_mask": all_response_mask,
        "logprobs": all_logprobs,
        "num_turns": len(records),
    }
