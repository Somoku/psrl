import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from psrl.utils.sparse_logprobs import select_sparse_rows, sparse_log_probs


@pytest.mark.parametrize("chunk_size", [1, 3, 128])
def test_sparse_values_and_gradient_match_dense(chunk_size):
    torch.manual_seed(123)
    logits = torch.randn(7, 13, dtype=torch.float64, requires_grad=True)
    rows = torch.tensor([0, 0, 2, 2, 2, 6])
    ids = torch.tensor([0, 12, 4, 4, 8, 7])
    weights = torch.randn(len(rows), dtype=torch.float64)
    actual = sparse_log_probs(logits, rows, ids, chunk_size=chunk_size)
    expected = logits.log_softmax(-1)[rows, ids]
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad((actual * weights).sum(), logits)[0]
    expected_grad = torch.autograd.grad((expected * weights).sum(), logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_padded_vocabulary_and_shared_entropy():
    logits = torch.randn(4, 11, requires_grad=True)
    rows, ids = torch.tensor([0, 3]), torch.tensor([6, 0], dtype=torch.int32)
    reference = logits.detach().clone().requires_grad_()
    actual = sparse_log_probs(logits, rows, ids, vocab_size=7, chunk_size=2)
    expected = reference[:, :7].log_softmax(-1)[rows, ids.long()]
    entropy = -(logits.softmax(-1) * logits.log_softmax(-1)).sum()
    reference_entropy = -(reference.softmax(-1) * reference.log_softmax(-1)).sum()
    (actual.sum() + entropy).backward()
    (expected.sum() + reference_entropy).backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(logits.grad, reference.grad)


def test_select_zigzag_rows_with_empty_environment_tokens():
    offsets = torch.tensor([0, 2, 2, 3, 7, 7, 9])
    rows, entries = select_sparse_rows(offsets, torch.tensor([0, 1, 5, 4]))
    assert rows.tolist() == [0, 0, 2, 2]
    assert entries.tolist() == [0, 1, 7, 8]


def test_empty_support_has_zero_gradient():
    logits = torch.randn(3, 7, requires_grad=True)
    empty = torch.empty(0, dtype=torch.long)
    sparse_log_probs(logits, empty, empty).sum().backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


def _parallel_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(42)
        full = torch.randn(6, 14, dtype=torch.float64, requires_grad=True)
        local = full.detach()[:, rank * 7 : (rank + 1) * 7].clone().requires_grad_()
        rows = torch.tensor([0, 0, 2, 4, 5, 5])
        ids = torch.tensor([0, 9, 8, 6, 8, 12])
        weights = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0, 6.0], dtype=torch.float64)
        actual = sparse_log_probs(local, rows, ids, process_group=dist.group.WORLD, vocab_size=13, chunk_size=2)
        expected = full[:, :13].log_softmax(-1)[rows, ids]
        (actual * weights).sum().backward()
        (expected * weights).sum().backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(local.grad, full.grad[:, rank * 7 : (rank + 1) * 7])
    finally:
        dist.destroy_process_group()


def test_tensor_parallel_matches_full_vocabulary(tmp_path):
    mp.spawn(_parallel_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True)
