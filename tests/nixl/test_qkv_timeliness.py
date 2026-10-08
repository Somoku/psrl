"""Timeliness of ConcatenatedQKV layout conversion (item 18).

Design §3.7.2.3 describes the `ConcatenatedQKVSync` lifecycle hook.  For models using
`ConcatenatedQKVMapping` (Megatron-style interleaved Q/K/V), the Megatron parameter
stores Q, K, V interleaved in a single buffer.  Before a push, `before_push` re-slices
that buffer and writes a HF-ordered cat(Q,K,V) into the PSRL state dict.  After the
push, `after_push` does nothing (the state dict copy is gone).

The timeliness claim is: if `before_push` is skipped (the optimizer step already mutated
`source_param` since the last cat), the rollout reads stale bytes.  The test verifies
that this staleness is detectable: the diff must exceed zero by more than floating-point
noise, which is the precondition for the design's assertion that "the cat must happen at
the push boundary, not earlier".

What makes this non-trivial is that `state_dict[key]` is an alias created at conversion
time: it holds the cat value from `_convert_concatenated_qkv_parameter`, not a live
view of `source_param`.  After an optimizer step mutates `source_param`, the state dict
entry is stale.  `before_push` refreshes it; skipping `before_push` surfaces the delta.

These are CPU tests: the QKV slicing is all Python/torch math, no GPU or NIXL needed.
"""

from __future__ import annotations

import torch
import pytest

from psrl.utils.converter.param_sync import ConcatenatedQKVSync


pytestmark = pytest.mark.cpu_test


# ---------------------------------------------------------------------------
# Minimal helpers that reproduce what megatron_converter does
# ---------------------------------------------------------------------------

def _make_megatron_qkv(num_heads: int, num_kv_heads: int, head_size: int, tp_size: int) -> torch.Tensor:
    """Build a Megatron-style interleaved Q/K/V parameter (CPU).

    `slice_qkv_proj_megatron` reshapes the buffer to
    ``(num_split_heads // tp_size, q_len + k_len + v_len, hidden)`` where
    ``num_split_heads = gcd(num_heads, num_kv_heads)`` and the per-group lengths are
    ``head_size * n / num_split_heads`` for n in (num_heads, num_kv_heads, num_kv_heads).
    The flat row count must equal the product of the first two dims, so it is derived
    here rather than guessed -- an independently invented layout fails the slicer's own
    shape assertion.
    """
    import math

    num_split_heads = math.gcd(num_heads, num_kv_heads)
    q_len = head_size * num_heads // num_split_heads
    k_len = head_size * num_kv_heads // num_split_heads
    v_len = head_size * num_kv_heads // num_split_heads
    rows = (num_split_heads // tp_size) * (q_len + k_len + v_len)
    hidden = num_heads * head_size
    return torch.nn.Parameter(torch.randn(rows, hidden))


def _make_state_dict_entry(
    sync: ConcatenatedQKVSync,
    source_param: torch.Tensor,
) -> torch.Tensor:
    """Produce the tensor `_convert_concatenated_qkv_parameter` puts in the state dict.

    Mirrors the production expression: the slicer returns 3-D views whose middle dim
    differs between Q and K/V under GQA, so the concat is group-local (``dim=1``) and the
    result is flattened to the 2-D ``(rows, hidden)`` form a HF checkpoint stores.
    """
    from psrl.utils.converter.model_mappings import slice_qkv_proj_megatron

    q, k, v = slice_qkv_proj_megatron(
        fused_param=source_param,
        num_heads=sync.num_heads,
        num_kv_heads=sync.num_kv_heads,
        head_size=sync.head_size,
        attn_output_gate=False,
        tp_size=sync.tp_size,
    )
    return torch.cat((q, k, v), dim=1).reshape(-1, source_param.size(1)).clone()


class TestConcatenatedQKVSyncBasics:
    """The before_push/after_push/after_pull contract of ConcatenatedQKVSync."""

    @pytest.fixture
    def sync_and_state_dict(self):
        """A minimal ConcatenatedQKVSync over a synthetic Megatron QKV parameter."""
        num_heads = 8
        num_kv_heads = 8
        head_size = 64
        tp_size = 1

        source_param = _make_megatron_qkv(num_heads, num_kv_heads, head_size, tp_size)
        sync = ConcatenatedQKVSync(
            key="qkv",
            megatron_name="attention.query_key_value.weight",
            source_param=source_param,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            tp_size=tp_size,
        )
        state_dict_entry = _make_state_dict_entry(sync, source_param)
        state_dict = {"qkv": state_dict_entry}
        return sync, state_dict, source_param

    def test_before_push_updates_state_dict_from_source_param(self, sync_and_state_dict):
        """before_push must refresh state_dict['qkv'] from source_param.

        This is the core timeliness guarantee: whatever the optimizer last wrote into
        source_param must appear in the state dict at push time.
        """
        sync, state_dict, source_param = sync_and_state_dict

        # Simulate an optimizer step that mutates source_param.
        with torch.no_grad():
            source_param.add_(0.5)

        old_bytes = state_dict["qkv"].clone()
        sync.before_push(state_dict)
        new_bytes = state_dict["qkv"]

        assert not torch.equal(new_bytes, old_bytes), (
            "before_push did not update the state dict after source_param mutation; "
            "a consumer reading state_dict['qkv'] would get stale bytes"
        )

    def test_after_push_is_a_no_op(self, sync_and_state_dict):
        """after_push must not change the state dict — it has nothing to write back."""
        sync, state_dict, _source_param = sync_and_state_dict

        pre = state_dict["qkv"].clone()
        sync.after_push(state_dict)
        post = state_dict["qkv"]

        assert torch.equal(pre, post), (
            "after_push mutated the state dict; it should be a no-op"
        )

    def test_before_push_missing_key_is_silent(self, sync_and_state_dict):
        """before_push must not raise when the key is absent (e.g., TP rank holds a shard)."""
        sync, _state_dict, _source_param = sync_and_state_dict
        # No key in this dict — must complete without raising.
        sync.before_push({"other_key": torch.zeros(4, 4)})


class TestStaleQKVIsDetectable:
    """The central claim of item 18: stale QKV bytes differ measurably from fresh ones.

    The test simulates what happens when the optimizer step happens between the initial
    conversion (which writes an initial cat into the state dict) and the push.  If
    `before_push` is skipped, the state dict holds the pre-optimizer bytes; the
    difference must be non-zero and above noise.
    """

    @pytest.mark.parametrize("delta", [0.1, 1.0])
    @pytest.mark.parametrize("heads, kv_heads", [(8, 8), (32, 8)], ids=["mha", "gqa"])
    def test_skipping_before_push_leaves_stale_bytes(self, delta, heads, kv_heads):
        """After an optimizer step of magnitude `delta`, the stale diff must exceed noise.

        The diff being zero would mean either the state dict was a live view (not a
        clone) of source_param, or the before_push refresh is always applied regardless —
        either would break the semantics the test is designed to catch.
        """
        num_heads, num_kv_heads, head_size, tp_size = heads, kv_heads, 64, 1
        source_param = _make_megatron_qkv(num_heads, num_kv_heads, head_size, tp_size)
        sync = ConcatenatedQKVSync(
            key="qkv",
            megatron_name="attention.query_key_value.weight",
            source_param=source_param,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            tp_size=tp_size,
        )
        # Capture the initial state dict entry (pre-optimizer-step).
        stale_entry = _make_state_dict_entry(sync, source_param)

        # Simulate optimizer step.
        with torch.no_grad():
            source_param.add_(delta)

        # Fresh entry obtained via before_push.
        fresh_entry = stale_entry.clone()
        sync.before_push({"qkv": fresh_entry})

        diff_norm = (fresh_entry - stale_entry).norm().item()
        assert diff_norm > 0.01, (
            f"stale diff norm={diff_norm:.6f} is indistinguishable from zero (delta={delta}); "
            "before_push is not refreshing the QKV entry, so staleness would be silent"
        )

    @pytest.mark.parametrize("heads, kv_heads", [(8, 8), (32, 8)], ids=["mha", "gqa"])
    def test_fresh_entry_matches_source_param_after_step(self, heads, kv_heads):
        """The refreshed state dict entry must reflect the post-step source_param content.

        This is the positive form of the staleness test: not only must stale != fresh,
        but fresh must actually encode the latest source_param values.
        """
        from psrl.utils.converter.model_mappings import slice_qkv_proj_megatron

        num_heads, num_kv_heads, head_size, tp_size = heads, kv_heads, 64, 1
        source_param = _make_megatron_qkv(num_heads, num_kv_heads, head_size, tp_size)
        sync = ConcatenatedQKVSync(
            key="qkv",
            megatron_name="attention.query_key_value.weight",
            source_param=source_param,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            tp_size=tp_size,
        )
        state_dict = {"qkv": _make_state_dict_entry(sync, source_param)}

        with torch.no_grad():
            source_param.mul_(2.0)

        sync.before_push(state_dict)

        q, k, v = slice_qkv_proj_megatron(
            fused_param=source_param,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            attn_output_gate=False,
            tp_size=tp_size,
        )
        expected = torch.cat((q, k, v), dim=1).reshape(-1, source_param.size(1))
        assert torch.allclose(state_dict["qkv"], expected), (
            "after before_push, state_dict['qkv'] does not match the current source_param; "
            "the refresh is computing from stale data rather than the live parameter"
        )


class TestAfterPullRestoresSourceParam:
    """after_pull must write the pulled HF QKV back into source_param (Megatron form)."""

    @pytest.mark.parametrize("heads, kv_heads", [(8, 8), (32, 8)], ids=["mha", "gqa"])
    def test_after_pull_updates_source_param(self, heads, kv_heads):
        """A state dict modified by the PS pull must be reflected in source_param.

        Driven through `ParamSyncPlan` rather than by calling `after_pull` directly:
        the plan wraps every hook in `torch.no_grad()`, and `after_pull` writes into a
        leaf `nn.Parameter` in place, which autograd rejects outside that context. Going
        through the plan is also what production does, so this exercises the real path.
        """
        from psrl.utils.converter.param_sync import ParamSyncPlan

        num_heads, num_kv_heads, head_size, tp_size = heads, kv_heads, 64, 1
        source_param = _make_megatron_qkv(num_heads, num_kv_heads, head_size, tp_size)
        sync = ConcatenatedQKVSync(
            key="qkv",
            megatron_name="attention.query_key_value.weight",
            source_param=source_param,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            tp_size=tp_size,
        )
        # Generate a different QKV from a different source and treat it as a pull result.
        other_source = _make_megatron_qkv(num_heads, num_kv_heads, head_size, tp_size)
        pulled_entry = _make_state_dict_entry(sync, other_source)
        state_dict = {"qkv": pulled_entry}

        plan = ParamSyncPlan()
        plan.add(sync)

        original_norm = source_param.norm().item()
        plan.after_pull(state_dict)

        new_norm = source_param.norm().item()
        assert abs(new_norm - original_norm) > 0.01, (
            "after_pull did not modify source_param; a PS pull of new weights would be lost"
        )

    @pytest.mark.parametrize("heads, kv_heads", [(8, 8), (32, 8)], ids=["mha", "gqa"])
    def test_after_pull_round_trips_through_before_push(self, heads, kv_heads):
        """push(cat) then pull(un-cat) must return the original Megatron parameter.

        This is the real correctness property of the pair: the HF-ordered tensor the
        rollout sees is a faithful re-encoding, so pulling it back reconstructs the
        interleaved buffer the trainer started from. A dimension permuted in only one of
        the two directions would pass each test individually and fail here.
        """
        from psrl.utils.converter.param_sync import ParamSyncPlan

        num_heads, num_kv_heads, head_size, tp_size = heads, kv_heads, 64, 1
        source_param = _make_megatron_qkv(num_heads, num_kv_heads, head_size, tp_size)
        sync = ConcatenatedQKVSync(
            key="qkv",
            megatron_name="attention.query_key_value.weight",
            source_param=source_param,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            tp_size=tp_size,
        )
        state_dict = {"qkv": _make_state_dict_entry(sync, source_param)}
        expected = source_param.detach().clone()

        plan = ParamSyncPlan()
        plan.add(sync)
        plan.before_push(state_dict)
        plan.after_pull(state_dict)

        assert torch.allclose(source_param.detach(), expected, atol=1e-6), (
            "before_push followed by after_pull did not reconstruct the original "
            "Megatron parameter; the two directions disagree on the QKV layout"
        )
