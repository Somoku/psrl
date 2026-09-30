import base64

import numpy as np
import pytest
from psrl.utils.sampler_distribution import SamplerDistribution, configure_capture
from psrl.utils.tito.training_data import build_training_data


def wire(ids, probabilities, complete=False):
    return {
        "offsets": [0, len(ids)],
        "token_ids": base64.b64encode(np.asarray(ids, dtype="<i4").tobytes()).decode(),
        "log_probs": base64.b64encode(np.log(np.asarray(probabilities, dtype="<f4")).tobytes()).decode(),
        "complete": [complete],
    }


def test_sparse_layout_has_no_prompt_or_environment_entries():
    head = SamplerDistribution.from_wire(wire([2, 8], [0.4, 0.3]), [[np.log(0.4), 2]])
    support = SamplerDistribution.from_wire(wire([3], [1], True), [[0, 3]])
    merged = SamplerDistribution.concatenate([head, SamplerDistribution.empty(2), support])
    merged.validate_capture("topk", 2, [1, 0, 0, 1])
    fields = merged.as_tensor_fields(prompt_length=3)
    assert fields["sampler_offsets"].tolist() == [0, 0, 0, 2, 2, 2, 3, 3]
    assert fields["sampler_token_ids"].tolist() == [2, 8, 3]
    assert fields["sampler_complete"].tolist() == [False, False, False, False, False, True, False]
    assert merged.prefix(3).token_ids.tolist() == [2, 8]


def test_support_budget_never_silently_truncates():
    head = SamplerDistribution.from_wire(wire([2], [0.999999]), [[np.log(0.999999), 2]])
    with pytest.raises(ValueError, match="capture budget"):
        head.validate_capture("support", 128, [1])


@pytest.mark.parametrize(
    "payload,sampled,match",
    [
        (wire([2, 2], [0.3, 0.4]), [[np.log(0.3), 2]], "Duplicate"),
        (wire([2], [0.3], True), [[np.log(0.3), 2]], "mass"),
        (wire([2], [1], True), [[0, 3]], "omits"),
        (wire([2], [0.3]), [[-1, 2]], "disagree"),
        (wire([2], [0.3]), [[float("nan"), 3]], "Invalid sampled"),
    ],
)
def test_invalid_wire_fails_closed(payload, sampled, match):
    with pytest.raises(ValueError, match=match):
        SamplerDistribution.from_wire(payload, sampled)


def test_tito_trim_and_multiturn_environment_alignment():
    first = wire([10, 99], [1, 1])
    first["offsets"] = [0, 1, 2]
    first["complete"] = [False, False]
    records = [
        {"prompt_token_count": 2, "output_logprobs": [[0, 10], [0, 99]], "sampler_distribution": first},
        {"prompt_token_count": 4, "output_logprobs": [[0, 20]], "sampler_distribution": wire([20], [1], True)},
    ]
    data = build_training_data([1, 2, 10, 15, 20], records, max_trim_tokens=1)
    distribution = data["sampler_distribution"]
    assert data["response_mask"] == [1, 0, 1]
    assert distribution.offsets.tolist() == [0, 1, 1, 2]
    assert distribution.token_ids.tolist() == [10, 20]


def test_missing_capture_is_not_silently_disabled():
    records = [{"prompt_token_count": 1, "output_logprobs": [[0, 2]], "sampler_capture_width": 128}]
    with pytest.raises(ValueError, match="Missing sampler distribution"):
        build_training_data([1, 2], records)


def test_capture_does_not_duplicate_head_in_agent_response():
    body = {"top_p": 0.9, "temperature": 0.7}
    configure_capture(body, "support", 128)
    assert body["top_logprobs"] == 0
    assert body["sampler_top_logprobs"] == 129
    with pytest.raises(ValueError, match="Truncated sampling"):
        configure_capture(body, "topk", 128)
