"""
XML Function-Calling Model for standalone eval.

Extends ``LitellmTextbasedModel`` to handle SWE-agent-LM's XML function calling
format (``<function=NAME>...</function>``) by translating tool calls into bash
commands via ``psrl.tools.tool_parser.xml_fc_tool_parser.parse_xml_fc_to_bash``.

Usage in eval:
    --model-class examples.mini_swe.eval.xml_fc_model.XmlFcModel
"""

from __future__ import annotations

import logging
import os

from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_textbased_model import (
    LitellmTextbasedModel,
    LitellmTextbasedModelConfig,
)

psrl_logger = logging.getLogger(__name__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

# Substrings emitted by vLLM when the prompt exceeds the engine context window.
# litellm does NOT recognize this wording as a context-window error (its known
# substrings expect "is longer than the model's context length", whereas vLLM
# says "is longer than the maximum model length of"), so it wraps the 400 as a
# generic BadRequestError that tenacity would otherwise retry until the episode
# times out. We detect it explicitly and short-circuit instead.
_VLLM_OVERFLOW_MARKERS = (
    "maximum model length",
    "decoder prompt",
)


class PromptOverflowError(Exception):
    """The prompt for a turn exceeded the rollout engine's context window.

    Turns produced before the overflow are valid training data; the agent loop
    recovers them from the TITO session and treats the trajectory as a normal
    max-length termination. Raised from ``XmlFcModel._query`` so it propagates
    out of ``DefaultAgent.run`` to the mini-SWE-agent runner.
    """


def _is_vllm_overflow(exc: Exception) -> bool:
    """Return whether the exception is a vLLM context-window overflow (400)."""
    message = str(exc).lower()
    return any(marker in message for marker in _VLLM_OVERFLOW_MARKERS)


class XmlFcModelConfig(LitellmTextbasedModelConfig):
    """Config for XmlFcModel — same as textbased but with XML fc defaults."""

    action_regex: str = "__XML_FUNCTION_CALLING__"
    format_error_template: str = (
        "Your last output did not include a valid function call. "
        "Please make sure your output includes exactly ONE function call.\n"
        "If you think you have already resolved the issue, please submit your changes "
        "by calling the `submit` function.\n"
        "If you think you cannot solve the problem, please call `submit`.\n"
        "Otherwise, please continue with a new tool call using the correct format:\n\n"
        "<function=function_name>\n"
        "<parameter=parameter_name>value</parameter>\n"
        "</function>"
    )


class XmlFcModel(LitellmTextbasedModel):
    """
    LitellmTextbasedModel subclass that parses SWE-agent-LM XML function calls.

    Instead of regex-extracting a bash code block, it uses
    ``parse_xml_fc_to_bash()`` to translate ``<function=bash>``,
    ``<function=str_replace_editor>``, and ``<function=submit>`` calls into
    executable bash commands.
    """

    def __init__(self, **kwargs):
        """Initialize with XmlFcModelConfig."""
        super(LitellmTextbasedModel, self).__init__(config_class=XmlFcModelConfig, **kwargs)
        # PromptOverflowError is terminal: a turn that overflows the context
        # window will overflow on every retry with identical messages, so it
        # must abort the tenacity retry loop in LitellmModel.query immediately
        # rather than be retried until the episode times out.
        self.abort_exceptions = [*self.abort_exceptions, PromptOverflowError]

    def _query(self, messages: list[dict[str, str]], **kwargs):
        """Query the engine, converting vLLM context overflows into a terminal
        ``PromptOverflowError`` so they are neither retried nor misclassified.

        Only the vLLM overflow 400 is short-circuited; every other error is
        re-raised unchanged so the inherited retry logic still handles transient
        failures (timeouts, 5xx, rate limits, etc.).
        """
        try:
            return super()._query(messages, **kwargs)
        except Exception as exc:
            if _is_vllm_overflow(exc):
                raise PromptOverflowError(str(exc)) from exc
            raise

    def _parse_actions(self, response) -> list[dict]:
        """
        Parse XML function-calling format from model response.

        Delegates to ``parse_xml_fc_to_bash`` which handles:
        - ``<function=bash>`` — direct command passthrough
        - ``<function=submit>`` — submission command
        - ``<function=str_replace_editor>`` — translated to bash equivalents
        - Degraded patterns (markdown fences, etc.)

        Raises:
            FormatError: When no valid action is found.
        """
        from psrl.tools.tool_parser.xml_fc_tool_parser import parse_xml_fc_to_bash

        content = response.choices[0].message.content or ""

        cmd = parse_xml_fc_to_bash(content)
        if cmd is not None:
            return [{"command": cmd}]

        # No valid action found — raise FormatError so the agent retries.
        raise FormatError(
            {
                "role": "user",
                "content": self.config.format_error_template,
                "extra": {
                    "interrupt_type": "FormatError",
                    "n_actions": 0,
                    "model_response": content,
                },
            }
        )
