"""Bounded observation text shared by every backend.

A command's output is an observation a model reads, so it needs a budget that
keeps both ends. The head carries the command echo and the early errors, and the
tail carries the final result, which is what a task actually failed on.
"""

from __future__ import annotations

TRUNCATION_NOTICE = "\n... {omitted} characters omitted ...\n"


def truncate_observation(text: str, max_chars: int) -> str:
    """
    Cap an observation to a character budget, keeping the head and the tail.

    A command is not expected to bound its own output, so a single step can emit
    enough text to exhaust the model's context in one turn.

    Args:
        text (str): Raw observation text.
        max_chars (int): Budget for each side of the elision. The notice is
            deducted from this budget, so the result is at most
            `max_chars * 2` characters. Zero or less disables truncation.

    Returns:
        str: The original text, or a head plus notice plus tail within the budget.
    """
    if max_chars <= 0 or len(text) <= max_chars * 2:
        return text

    total_budget = max_chars * 2
    # The notice length stabilizes once the omitted digit count is known.
    placeholder = TRUNCATION_NOTICE.format(omitted=len(text))
    per_side = max(0, (total_budget - len(placeholder)) // 2)
    omitted = len(text) - per_side * 2
    notice = TRUNCATION_NOTICE.format(omitted=omitted)
    head = text[:per_side] if per_side > 0 else ""
    tail = text[-per_side:] if per_side > 0 else ""
    return head + notice + tail
