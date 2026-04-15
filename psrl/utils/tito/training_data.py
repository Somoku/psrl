"""Build RL training arrays from TITO session data.

Converts accumulated_token_ids + per-turn records (from SMG GET /v1/tito/sessions)
into prompt_ids, response_ids, response_mask, and logprobs arrays.
"""

from __future__ import annotations


def build_training_arrays(
    accumulated_token_ids: list[int],
    records: list[dict],
    max_trim_tokens: int = 0,
) -> dict:
    """Convert TITO session data into training arrays.

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

        # Trailing trim for non-last turns: greedy match against accumulated.
        # The last turn never trims — boundary tokens are part of the final output.
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

            # Validate against the model-specific ceiling.
            # Last turn: no trimming ever allowed (allowed = 0).
            # Non-last turn: at most max_trim_tokens (typically 0 or 1).
            allowed = max_trim_tokens  # is_last already guarded above
            if trim_count > allowed:
                raise ValueError(
                    f"TITO trailing trim overflow at turn {i}: "
                    f"trim_count={trim_count} exceeds allowed={allowed} "
                    f"(max_trim_tokens={max_trim_tokens}). "
                    f"output_ids[-3:]={output_ids[-3:]}, "
                    f"accumulated[{prompt_len + matched}:{prompt_len + matched + 3}]="
                    f"{accumulated_token_ids[prompt_len + matched : prompt_len + matched + 3]}"
                )

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
