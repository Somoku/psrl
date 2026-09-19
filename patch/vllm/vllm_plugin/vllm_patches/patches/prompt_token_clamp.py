import logging
import os

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


def apply_prompt_token_clamp_patch() -> None:
    """Keep per-source prompt-token counters non-negative.

    ``PrometheusStatLogger`` feeds ``PromptTokenStats.get_by_source`` straight
    into a Prometheus ``Counter.inc()``, which rejects a negative value with a
    ``ValueError`` and would otherwise take down the EngineCore. When an
    externally loaded KV block fails to load, the scheduler rewrites
    ``request.num_computed_tokens`` downwards to the longest valid prefix, so a
    per-source count can be reported smaller than what was already added for the
    same iteration.

    Clamping at the getter keeps the metric monotonic and the engine alive;
    correctness is unaffected because the affected tokens are recomputed
    regardless. Only the reported metric value is clamped.

    The v0.29 accounting is additive-only and ``PrefillStats`` asserts
    ``num_external_computed_tokens > 0``, so this is defensive: remove it if a
    repro shows the negative delta can no longer occur.
    """
    _patch_get_by_source()


def _patch_get_by_source() -> None:
    from vllm.v1.metrics.stats import PromptTokenStats

    _SENTINEL = "_psrl_prompt_token_clamp_patched"
    if getattr(PromptTokenStats, _SENTINEL, False):
        psrl_logger.debug("PromptTokenStats.get_by_source already patched, skipping.")
        return

    _original = PromptTokenStats.get_by_source

    def get_by_source(self, source: str) -> int:
        return max(0, _original(self, source))

    PromptTokenStats.get_by_source = get_by_source
    setattr(PromptTokenStats, _SENTINEL, True)

    psrl_logger.info("Patched PromptTokenStats.get_by_source to clamp at zero.")
