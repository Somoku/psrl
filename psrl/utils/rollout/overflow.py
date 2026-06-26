"""Detect vLLM prompt-overflow 400 errors and convert them to a terminal exception.

litellm wraps vLLM's "prompt is too long" 400 as a generic ``BadRequestError``
because the wording differs from what litellm expects (``"is longer than the
model's context length"`` vs vLLM's ``"is longer than the maximum model length
of"``). Without intervention, the ``tenacity`` retry loop in
``minisweagent.models.LitellmModel.query`` retries indefinitely until the
episode times out.

This module provides:

- ``PromptOverflowError`` — a terminal exception that aborts the retry loop.
- ``ensure_overflow_handling(model)`` — instance-level patch (idempotent).
- ``handle_prompt_overflow(cls)`` — class decorator equivalent.
"""

from __future__ import annotations

import functools
from typing import TypeVar

_T = TypeVar("_T")

_VLLM_OVERFLOW_MARKERS = (
    "maximum model length",
    "decoder prompt",
)


class PromptOverflowError(Exception):
    """The prompt for a turn exceeded the rollout engine's context window.

    Turns produced before the overflow are valid training data; the agent loop
    should recover them and treat the trajectory as a normal max-length
    termination rather than a fatal rollout error.
    """


def _is_vllm_overflow(exc: Exception) -> bool:
    """Return whether *exc* is a vLLM context-window overflow (HTTP 400)."""
    message = str(exc).lower()
    return any(marker in message for marker in _VLLM_OVERFLOW_MARKERS)


def ensure_overflow_handling(model: _T) -> _T:
    """Apply vLLM prompt-overflow detection to a model instance (idempotent).

    Wraps ``model._query`` so that a vLLM 400 overflow is converted to
    ``PromptOverflowError`` and registered in ``model.abort_exceptions`` to
    short-circuit the tenacity retry loop.
    """
    if PromptOverflowError in model.abort_exceptions:
        return model

    original_query = model._query

    @functools.wraps(original_query)
    def _wrapped_query(messages, **kwargs):
        try:
            return original_query(messages, **kwargs)
        except PromptOverflowError:
            raise
        except Exception as exc:
            if _is_vllm_overflow(exc):
                raise PromptOverflowError(str(exc)) from exc
            raise

    model._query = _wrapped_query
    model.abort_exceptions = [*model.abort_exceptions, PromptOverflowError]
    return model


def handle_prompt_overflow(cls):
    """Class decorator: adds vLLM overflow detection to a LitellmModel subclass.

    Equivalent to calling ``ensure_overflow_handling(instance)`` after every
    ``__init__``, but applied statically at class-definition time.
    """
    original_init = cls.__init__
    original_query = cls._query

    @functools.wraps(original_init)
    def _patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if PromptOverflowError not in self.abort_exceptions:
            self.abort_exceptions = [*self.abort_exceptions, PromptOverflowError]

    @functools.wraps(original_query)
    def _patched_query(self, messages, **kwargs):
        try:
            return original_query(self, messages, **kwargs)
        except PromptOverflowError:
            raise
        except Exception as exc:
            if _is_vllm_overflow(exc):
                raise PromptOverflowError(str(exc)) from exc
            raise

    cls.__init__ = _patched_init
    cls._query = _patched_query
    return cls
