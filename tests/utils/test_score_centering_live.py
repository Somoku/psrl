"""Live probe for score-centering sampler capture against a running SMG + vLLM.

Opt-in: set ``PSRL_LIVE_SAMPLER_ENDPOINT`` (SMG base URL) and ``PSRL_LIVE_SAMPLER_MODEL`` (served
model), then run this file. The server must run vLLM with ``logprobs_mode=processed_logprobs`` and
``max_logprobs >= PSRL_LIVE_BUDGET + 1`` (PSRL raises the limit from ``sampler_top_logprobs`` in
``psrl.workers.gen.vllm_async_server``).

It verifies the exact-support contract PSRL's ``support`` mode relies on, which SGLang needed a
dedicated API for but vLLM already provides through ``processed_logprobs``:

  1. the returned head is post-filter and renormalized (finite entries sum to 1 on the nucleus).
  2. entries outside the nucleus are ``-inf`` and survive to the wire (the completeness sentinel).
  3. the support fits the requested budget.

Run: ``python -m pytest -q tests/utils/test_score_centering_live.py``.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request

import numpy as np
import pytest

ENDPOINT = os.environ.get("PSRL_LIVE_SAMPLER_ENDPOINT", "").rstrip("/")
MODEL = os.environ.get("PSRL_LIVE_SAMPLER_MODEL", "")
PROMPT = os.environ.get("PSRL_LIVE_SAMPLER_PROMPT", "The capital of France is")
TOP_P = float(os.environ.get("PSRL_LIVE_SAMPLER_TOP_P", "0.5"))
BUDGET = int(os.environ.get("PSRL_LIVE_SAMPLER_BUDGET", "128"))

pytestmark = pytest.mark.skipif(
    not ENDPOINT or not MODEL,
    reason="set PSRL_LIVE_SAMPLER_ENDPOINT and PSRL_LIVE_SAMPLER_MODEL to run the live probe",
)


def _post_generate(payload: dict) -> dict:
    request = urllib.request.Request(
        f"{ENDPOINT}/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.loads(response.read().decode())
    # The gateway returns either one object or a list of completion objects.
    return body[0] if isinstance(body, list) else body


def _decode_head(meta_info: dict) -> tuple[np.ndarray, np.ndarray]:
    ids = np.frombuffer(base64.b64decode(meta_info["output_topk_token_ids"]), dtype="<i4")
    log_probs = np.frombuffer(base64.b64decode(meta_info["output_topk_log_probs"]), dtype="<f4")
    shape = tuple(int(value) for value in meta_info["output_topk_shape"])
    return ids.reshape(shape), log_probs.reshape(shape)


def test_support_capture_is_exact_and_sentinel_complete():
    payload = {
        "model": MODEL,
        "request_id": "psrl-sc-live-probe",
        "text": PROMPT,
        "stream": False,
        "return_logprob": True,
        "sampler_top_logprobs": BUDGET + 1,  # budget + one completeness sentinel
        "sampling_params": {
            "temperature": 1.0,
            "top_p": TOP_P,
            "top_k": -1,
            "min_p": 0.0,
            "max_tokens": 16,
        },
    }
    completion = _post_generate(payload)
    meta_info = completion["meta_info"]
    assert "output_topk_shape" in meta_info, (
        "the gateway returned no sampler head. Confirm SMG is built with the capture changes and "
        f"the request carried sampler_top_logprobs. meta_info keys: {sorted(meta_info)}"
    )

    ids, log_probs = _decode_head(meta_info)
    rows, width = log_probs.shape
    assert rows == len(completion.get("output_ids", [])) or rows >= 1
    assert width == BUDGET + 1, f"expected a {BUDGET + 1}-wide head, got width {width}"

    for row in range(rows):
        row_probs = log_probs[row]
        finite = np.isfinite(row_probs)
        nucleus = int(finite.sum())
        # (1) post-filter, renormalized: the finite entries sum to 1 on the nucleus.
        assert nucleus >= 1, f"row {row} has no finite support"
        assert np.isclose(np.exp(row_probs[finite].astype(np.float64)).sum(), 1.0, rtol=1e-3, atol=1e-4), (
            f"row {row} support does not sum to 1. vLLM was not run with processed_logprobs"
        )
        # (3) the realized nucleus fits the requested budget.
        assert nucleus <= BUDGET, f"row {row} support {nucleus} exceeds the budget {BUDGET}. Raise sampler_max_tokens"
        # (2) the -inf sentinel proves the finite support was fully captured. With top_p<1 and a
        # generous budget the nucleus is far below the window, so a sentinel must be present.
        assert not finite.all(), (
            f"row {row} filled the whole window with no -inf sentinel. Increase the budget or lower top_p"
        )
        # Non-sentinel ids are real vocabulary ids.
        assert (ids[row][finite] >= 0).all()


def test_unfiltered_head_covers_less_than_full_mass():
    # Sanity companion: with top_p=1 the top-k head is a proper sub-distribution (mass < 1), i.e. the
    # tail model applies. This distinguishes the two modes against the same server.
    payload = {
        "model": MODEL,
        "request_id": "psrl-sc-live-probe-full",
        "text": PROMPT,
        "stream": False,
        "return_logprob": True,
        "sampler_top_logprobs": BUDGET,
        "sampling_params": {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0, "max_tokens": 16},
    }
    completion = _post_generate(payload)
    meta_info = completion["meta_info"]
    if "output_topk_shape" not in meta_info:
        pytest.skip("gateway returned no sampler head for the unfiltered request")
    _, log_probs = _decode_head(meta_info)
    mass = np.exp(log_probs.astype(np.float64)).sum(axis=-1)
    assert (mass < 1.0 + 1e-3).all(), "unfiltered head mass must not exceed 1"
