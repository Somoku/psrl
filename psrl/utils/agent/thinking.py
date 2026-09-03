"""Chain-of-thought handling modes for multi-turn agentic rollouts.

TITO chains a session's turns into a single trajectory by hashing each assistant
message, ``reasoning_content`` included (smg ``crates/tito/src/normalizer.rs``).
Harness clients such as terminus-2 replay history as ``{role, content}`` only, so
once the gateway splits ``<think>`` out into ``reasoning_content`` the replayed
prefix stops hashing onto the stored leaf and SMG forks a fresh trajectory on
every turn (``crates/tito/src/store.rs`` ``resolve_trajectory_id``).

This module is the single place that names the four coherent responses to that and
the invariants each one implies, so the agent loop (which decides how many
trajectories to keep) and the harness runner (which decides what to send the
gateway) cannot drift apart. See ``psrl/trainer/config/psrl/agentic_rl.yaml`` for
the user-facing description and ``examples/sciaccel_rl/knowledge.md`` for the
measurements behind it.
"""

from __future__ import annotations

# Accept the forks and train every trajectory. Each one is internally consistent
# (its prompt and response come from the same request), so a fork costs cross-turn
# context, not correctness.
MULTI_TRAJ = "multi_traj"

# Same rollout as MULTI_TRAJ, but keep only the session's longest trajectory.
# Trades data volume for one clean sample per episode.
LONGEST_TRAJ = "longest_traj"

# Keep the CoT inline in `content` so nothing forks: reasoning parser OFF plus an
# accumulating chat template that replays every prior turn's <think> block. One
# session yields one trajectory whose prompt carries several CoT segments.
MULTI_THINKING = "multi_thinking"

# Turn thinking off entirely. No CoT to split, so nothing forks.
DISABLE_THINKING = "disable_thinking"

SUPPORTED_THINKING_TEMPLATES = frozenset(
    {MULTI_TRAJ, LONGEST_TRAJ, MULTI_THINKING, DISABLE_THINKING}
)

# Modes that guarantee a session produces exactly one trajectory, because the
# assistant message the client replays is byte-identical to the one TITO stored.
# The agent loop may keep every trajectory it is given under these modes: a fork
# here means an invariant broke, not that forking is expected.
_SINGLE_TRAJECTORY_MODES = frozenset({MULTI_THINKING, DISABLE_THINKING})

# Modes that require the gateway's reasoning parser to stay OFF so the full
# generated text, the model's own `</think>` included, arrives inline in `content`.
_INLINE_REASONING_MODES = frozenset({MULTI_THINKING})

# Modes that require the model's thinking toggle to be OFF.
_THINKING_DISABLED_MODES = frozenset({DISABLE_THINKING})


def validate_thinking_template(mode: str) -> str:
    """Return ``mode`` if it names a supported policy, else raise ``ValueError``."""
    if mode not in SUPPORTED_THINKING_TEMPLATES:
        raise ValueError(
            f"psrl.agentic_rl.thinking_template must be one of "
            f"{sorted(SUPPORTED_THINKING_TEMPLATES)}, got {mode!r}."
        )
    return mode


def keeps_single_trajectory(mode: str) -> bool:
    """Whether ``mode`` is expected to yield exactly one trajectory per session."""
    return validate_thinking_template(mode) in _SINGLE_TRAJECTORY_MODES


def keeps_longest_trajectory_only(mode: str) -> bool:
    """Whether the agent loop should discard all but the longest trajectory."""
    return validate_thinking_template(mode) == LONGEST_TRAJ


def wants_inline_reasoning(mode: str) -> bool:
    """Whether the gateway's reasoning parser must be disabled for ``mode``."""
    return validate_thinking_template(mode) in _INLINE_REASONING_MODES


def wants_thinking_disabled(mode: str) -> bool:
    """Whether the model's thinking toggle must be turned off for ``mode``."""
    return validate_thinking_template(mode) in _THINKING_DISABLED_MODES


def harness_extra_body(mode: str) -> dict[str, object]:
    """Build the OpenAI ``extra_body`` a harness must send the gateway for ``mode``.

    ``separate_reasoning`` defaults to true in SMG (``crates/protocols/src/chat.rs``),
    so MULTI_THINKING has to switch it off explicitly to keep ``<think>`` in
    ``content``. ``enable_thinking`` rides in ``chat_template_kwargs`` because it is
    consumed by the Jinja template rather than the gateway.

    Returns:
        dict: Merge into the harness's ``extra_body``. Empty for the modes that
        leave gateway behavior at its defaults.
    """
    extra_body: dict[str, object] = {}
    if wants_inline_reasoning(mode):
        extra_body["separate_reasoning"] = False
    if wants_thinking_disabled(mode):
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    return extra_body


def select_trajectories(mode: str, training_data: list[dict]) -> list[dict]:
    """Apply ``mode``'s trajectory-retention policy to one session's trajectories.

    Args:
        mode: A supported ``thinking_template`` value.
        training_data: Every trajectory TITO recorded for the session, in the order
            the store returned them.

    Returns:
        list[dict]: The trajectories to train on. Length 1 for LONGEST_TRAJ whenever
        the input is non-empty, otherwise the input unchanged.
    """
    if not keeps_longest_trajectory_only(mode) or len(training_data) <= 1:
        return training_data
    # Longest by trained tokens, not by turn count: a single long turn carries more
    # signal than several truncated ones. `max` keeps the earliest on a tie, which
    # prefers the trajectory that started at the session root.
    return [max(training_data, key=lambda item: len(item.get("response_ids") or []))]


__all__ = [
    "MULTI_TRAJ",
    "LONGEST_TRAJ",
    "MULTI_THINKING",
    "DISABLE_THINKING",
    "SUPPORTED_THINKING_TEMPLATES",
    "validate_thinking_template",
    "keeps_single_trajectory",
    "keeps_longest_trajectory_only",
    "wants_inline_reasoning",
    "wants_thinking_disabled",
    "harness_extra_body",
    "select_trajectories",
]
