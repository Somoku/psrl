"""
Exercise infrastructure hooks without initializing models or accelerator runtimes.
"""

import ast
import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from psrl.utils.sampler_training import LossWithLogitsProcessor, query_sampler_log_probs

ROOT = Path(__file__).resolve().parents[2]


def load_method(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def nested(tensors):
    return torch.nested.nested_tensor(tensors, layout=torch.jagged)


def data_and_processor():
    data = {
        "input_ids": nested([torch.tensor([0, 1, 2]), torch.tensor([1, 2])]),
        "sampler_offsets": nested([torch.tensor([0, 0, 2, 2]), torch.tensor([0, 1, 1])]),
        "sampler_token_ids": nested([torch.tensor([1, 3]), torch.tensor([0])]),
        "sampler_log_probs": nested([torch.tensor([-1.2, -2.3]), torch.tensor([-0.7])]),
    }

    def processor(logits, data, row_ids, process_group):
        rows, train_lp, sampler_lp = query_sampler_log_probs(logits, data, row_ids, process_group=process_group)
        correction = torch.zeros(row_ids.numel(), dtype=train_lp.dtype)
        correction.scatter_add_(0, rows, train_lp * sampler_lp.exp())
        return {"auxiliary": correction.reshape(row_ids.shape)}

    return data, LossWithLogitsProcessor(lambda **kwargs: None, processor)


def test_sparse_query_preserves_cp_order_and_padding():
    data, loss = data_and_processor()
    logits = torch.randn(1, 5, 7, requires_grad=True)
    rows = torch.tensor([[4, 3, -1, 1, 0]])
    output = loss.logits_processor(logits, data, rows, None)["auxiliary"]
    expected = logits.new_zeros(1, 5)
    expected[0, 1] = logits[0, 1].log_softmax(-1)[0] * torch.exp(torch.tensor(-0.7))
    expected[0, 3] = (logits[0, 3].log_softmax(-1)[[1, 3]] * torch.tensor([-1.2, -2.3]).exp()).sum()
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(
        torch.autograd.grad(output.sum(), logits)[0], torch.autograd.grad(expected.sum(), logits)[0]
    )


@pytest.mark.skipif(
    not hasattr(torch.nested, "nested_tensor_from_jagged"),
    reason="Requires the veRL nested tensor API in newer PyTorch.",
)
def test_fsdp_auxiliary_processor_and_label_gradient():
    data, loss = data_and_processor()
    data["calculate_entropy"] = True
    entropy = lambda logits: -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)
    calls = []

    def logprobs(logits, labels, inplace_backward=True):
        calls.append(inplace_backward)
        return logits.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)

    method = load_method(
        "third_party/verl/verl/workers/engine/fsdp/transformer_impl.py",
        "prepare_model_outputs",
        {
            "torch": torch,
            "tu": SimpleNamespace(get_non_tensor_data=lambda data, key, default=None: data.get(key, default)),
            "DatasetPadMode": SimpleNamespace(NO_PADDING="no_padding"),
            "DTensor": type(None),
            "_scale_logits_by_temperature": lambda logits, temperature, **kwargs: logits,
            "logprobs_from_logits": logprobs,
        },
    )
    engine = SimpleNamespace(
        use_ulysses_sp=False,
        engine_config=SimpleNamespace(entropy_checkpointing=False, entropy_from_logits_with_chunking=False),
        compute_entropy_from_logits=entropy,
        _get_packed_pad_size=lambda length: 0,
        _gather_and_unpad_packed=lambda tensor, padding: tensor,
    )
    logits = torch.randn(1, 5, 7, requires_grad=True)
    labels = torch.tensor([1, 2, 1, 2, 0])
    output = method(
        engine,
        SimpleNamespace(logits=logits),
        {"input_ids_rmpad_rolled": labels, "temperature_rmpad": torch.ones(5), "pad_size": 0},
        data,
        loss,
    )
    actual = sum(output[key].values().sum() for key in ("log_probs", "entropy", "auxiliary"))
    expected = (
        logits[0].log_softmax(-1).gather(-1, labels[:, None]).sum()
        + entropy(logits).sum()
        + loss.logits_processor(logits, data, torch.arange(5)[None], None)["auxiliary"].sum()
    )
    assert calls == [False]
    torch.testing.assert_close(torch.autograd.grad(actual, logits)[0], torch.autograd.grad(expected, logits)[0])


class MutatingCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels):
        ctx.save_for_backward(logits, logits.softmax(-1), labels)
        return logits.log_softmax(-1).gather(-1, labels[..., None]).squeeze(-1)

    @staticmethod
    def backward(ctx, grad):
        logits, probs, labels = ctx.saved_tensors
        logits.zero_()
        probs.neg_().scatter_add_(-1, labels[..., None], torch.ones_like(labels[..., None], dtype=probs.dtype))
        return probs * grad[..., None], None


def test_megatron_processor_survives_mutating_cross_entropy_backward():
    data, loss = data_and_processor()
    method = load_method(
        "third_party/verl/verl/workers/engine/megatron/transformer_impl.py",
        "_lm_head_logits_processor",
        {
            "torch": torch,
            "mpu": SimpleNamespace(get_tensor_model_parallel_group=lambda: None),
            "vocab_parallel_log_probs_from_logits": MutatingCrossEntropy.apply,
        },
    )
    source = torch.randn(1, 5, 7, requires_grad=True)
    labels = torch.tensor([[1, 2, 1, 2, 0]])
    output = method(
        SimpleNamespace(),
        source * 1,
        labels,
        torch.ones(1, 5),
        calculate_sum_pi_squared=False,
        calculate_entropy=False,
        distillation_use_topk=False,
        distillation_only=False,
        logits_processor_func=loss,
        batch=data,
        data_format="thd",
        auxiliary_row_ids=torch.arange(5)[None] + 1,
    )
    actual = output["log_probs"].sum() + output["auxiliary"].sum()
    expected = source.log_softmax(-1).gather(-1, labels[..., None]).sum()
    expected = expected + loss.logits_processor(source, data, torch.arange(5)[None], None)["auxiliary"].sum()
    torch.testing.assert_close(torch.autograd.grad(actual, source)[0], torch.autograd.grad(expected, source)[0])


@pytest.mark.parametrize(
    "validate,protocol,width,status",
    [(False, "chat", 129, 200), (True, "chat", 0, 200), (False, "responses", None, 400)],
)
def test_session_router_capture_and_validation(validate, protocol, width, status):
    from psrl.utils.sampler_distribution import configure_capture

    class Response:
        def __init__(self, status_code, content):
            self.status_code = status_code
            self.content = content

    async def state(sid):
        return SimpleNamespace(headers={"x-is-validate": str(validate).lower()})

    async def body():
        return {"stream": True, "temperature": 0.7, "top_p": 0.9}

    method = load_method(
        "psrl/workers/gen/session_router.py",
        "prepare_protocol_request",
        {"json": json, "JSONResponse": Response, "configure_capture": configure_capture},
    )
    router = SimpleNamespace(_ensure_state=state, sampler_distribution="support", sampler_max_tokens=128)
    result = asyncio.run(method(router, SimpleNamespace(json=body), protocol, "session"))
    if status == 400:
        assert result.status_code == 400, "Unsupported protocols must reject capture."
    else:
        payload, streamed = result
        payload = json.loads(payload)
        assert streamed, "The client streaming preference must survive buffering."
        assert not payload["stream"], "The upstream must complete one TITO turn."
        assert payload["sampler_top_logprobs"] == width, "Validation must disable sampler capture."
        assert payload["top_logprobs"] == 0, "Agent responses must not duplicate captured heads."


def test_synthetic_padding_has_empty_sampler_buffers():
    method = load_method(
        "third_party/verl/verl/trainer/ppo/padding_utils.py",
        "construct_minimal_padding_template",
        {
            "torch": torch,
            "copy": copy,
            "SYNTHETIC_PADDING_SEQ_LEN": 128,
            "build_padding_position_ids": lambda source, mask: torch.arange(len(mask)),
            "build_padding_routed_experts": lambda source, length: None,
        },
    )
    source = {
        "response_mask": torch.ones(3),
        "sampler_offsets": torch.tensor([0, 2, 2, 2, 2]),
        "sampler_token_ids": torch.tensor([2, 4], dtype=torch.int32),
        "sampler_log_probs": torch.tensor([-1.0, -2.0]),
        "sampler_complete": torch.zeros(4, dtype=torch.bool),
    }
    padding, tag = method(source, {}, 0, 99)
    assert padding["sampler_offsets"].shape == (129,), "Offsets must cover every padded prediction row."
    assert padding["sampler_offsets"].count_nonzero() == 0, "Synthetic rows must have no entries."
    assert padding["sampler_token_ids"].numel() == 0, "Synthetic samples must not copy real token IDs."
    assert padding["sampler_log_probs"].numel() == 0, "Synthetic samples must not copy sampler probabilities."
    assert tag["is_padding"], "Synthetic samples must be marked as padding."


@pytest.mark.parametrize("capture", [True, False])
def test_session_capture_rejects_a_gateway_without_distributions(capture):
    async def snapshot(session_id):
        return {"trajectories": [{"response_ids": [2]}]}

    method = load_method("psrl/workers/agent_loop/loops/session_agent_loop.py", "get_training_data", {})
    loop = SimpleNamespace(
        get_session_data=snapshot,
        build_training_data_from_trajectory=lambda trajectory, session: trajectory,
        _capture_sessions={"session"} if capture else set(),
    )
    if capture:
        with pytest.raises(ValueError, match="no distribution"):
            asyncio.run(method(loop, "session"))
    else:
        assert asyncio.run(method(loop, "session")) == [{"response_ids": [2]}], "Disabled capture needs no heads."


def test_microbatch_reordering_preserves_independent_sparse_lengths():
    method = load_method(
        "third_party/verl/verl/utils/tensordict_utils.py",
        "index_select_tensor_dict",
        {
            "torch": torch,
            "nested_tensor_from_tensor_list": lambda tensors, ragged_idx: nested(tensors),
            "TensorDict": lambda source, batch_size: source,
        },
    )
    data, loss = data_and_processor()
    reordered = method(data, [1, 0])
    logits = torch.randn(1, 5, 7, requires_grad=True)
    output = loss.logits_processor(logits, reordered, torch.arange(5)[None], None)["auxiliary"]
    expected = logits.new_zeros(1, 5)
    expected[0, 0] = logits[0, 0].log_softmax(-1)[0] * torch.exp(torch.tensor(-0.7))
    expected[0, 3] = (logits[0, 3].log_softmax(-1)[[1, 3]] * torch.tensor([-1.2, -2.3]).exp()).sum()
    torch.testing.assert_close(output, expected)
