"""Exposed tensors stay correct between pulls under direct supply (design §3.7.2.3).

The existing sync actions are a *pair*: `before_push` puts a tensor into the form readers
expect, `after_push` puts it back. That works when the reader is a PS, because the push is
a bounded interval and nothing reads outside it.

Direct supply removes the interval. Rollouts read the train ranks' registered tensors
whenever they pull, so there is no moment at which reverting is safe, and no moment at
which "the exposed value is stale" is detectable -- a pull against a reverted tensor
succeeds and returns wrong numbers. That is the failure these tests exist to prevent, and
it is why the exposed value needs its own storage rather than borrowing the parameter.

`ZeroCenteredGammaSync` is the one action that reverts; `DTypeCastSync` and
`ConcatenatedQKVSync` already write into separate tensors and leave `after_push` empty.
"""

from __future__ import annotations

import pytest
import torch

from psrl.utils.converter.param_sync import (
    ParamSyncPlan,
    ResidentZeroCenteredGammaSync,
    ZeroCenteredGammaSync,
)

pytestmark = pytest.mark.cpu_test

_KEY = "model.layers.0.input_layernorm.weight"


def _zero_centred_param() -> torch.Tensor:
    """A Megatron gamma, which stores `gamma - 1`."""
    return torch.tensor([-0.25, 0.5, 0.0], dtype=torch.float32)


class TestTheBorrowingActionIsUnsafeForDirectSupply:
    """Why a new action was needed, stated as a test rather than as a comment."""

    def test_it_reverts_the_tensor_readers_would_be_holding(self):
        param = _zero_centred_param()
        state_dict = {_KEY: param}
        plan = ParamSyncPlan()
        plan.add(ZeroCenteredGammaSync(_KEY))

        plan.before_push(state_dict)
        exposed_during_push = state_dict[_KEY].clone()
        plan.after_push(state_dict)

        assert torch.equal(exposed_during_push, param + 1 - 1 + 1)
        # After the revert the registered tensor no longer holds the exposed value, so a
        # rollout reading now gets gamma-1 where it expects gamma.
        assert not torch.equal(state_dict[_KEY], exposed_during_push)

    def test_it_aliases_the_train_parameter(self):
        """The aliasing is the mechanism *and* the problem: one tensor, two meanings."""
        param = _zero_centred_param()
        state_dict = {_KEY: param}
        ZeroCenteredGammaSync(_KEY).before_push(state_dict)

        assert state_dict[_KEY] is param


class TestTheResidentActionKeepsBothViewsCorrect:
    def test_the_exposed_copy_is_separate_storage(self):
        param = _zero_centred_param()
        exposed = torch.empty_like(param)
        state_dict = {_KEY: exposed}

        ResidentZeroCenteredGammaSync(key=_KEY, source_param=param).before_push(state_dict)

        assert state_dict[_KEY] is not param

    def test_the_trainer_still_sees_a_zero_centred_parameter(self):
        """A resident `+1` on the real parameter would corrupt the trainer's forward."""
        param = _zero_centred_param()
        original = param.clone()
        state_dict = {_KEY: torch.empty_like(param)}

        ResidentZeroCenteredGammaSync(key=_KEY, source_param=param).before_push(state_dict)

        assert torch.equal(param, original)

    def test_the_exposed_value_survives_after_push(self):
        """The property direct supply needs: correct at an arbitrary later read."""
        param = _zero_centred_param()
        state_dict = {_KEY: torch.empty_like(param)}
        action = ResidentZeroCenteredGammaSync(key=_KEY, source_param=param)

        action.before_push(state_dict)
        action.after_push(state_dict)

        assert torch.equal(state_dict[_KEY], param + 1)

    def test_a_later_version_refreshes_the_copy(self):
        """Without this the rollouts read version N's gamma forever, and nothing fails."""
        param = _zero_centred_param()
        state_dict = {_KEY: torch.empty_like(param)}
        action = ResidentZeroCenteredGammaSync(key=_KEY, source_param=param)
        action.before_push(state_dict)

        param.add_(10.0)  # stand in for an optimizer step
        action.before_push(state_dict)

        assert torch.equal(state_dict[_KEY], param + 1)

    def test_repeated_publishes_do_not_accumulate(self):
        """`add_(1)` on the exposed copy would drift by one per step if not overwritten."""
        param = _zero_centred_param()
        state_dict = {_KEY: torch.empty_like(param)}
        action = ResidentZeroCenteredGammaSync(key=_KEY, source_param=param)

        for _ in range(5):
            action.before_push(state_dict)

        assert torch.equal(state_dict[_KEY], param + 1)

    def test_a_pull_restores_the_zero_centred_form(self):
        """The receiving side holds gamma; the trainer's parameter must go back to gamma-1."""
        param = _zero_centred_param()
        incoming = torch.tensor([3.0, 4.0, 5.0], dtype=torch.float32)
        state_dict = {_KEY: incoming}

        ResidentZeroCenteredGammaSync(key=_KEY, source_param=param).after_pull(state_dict)

        assert torch.equal(param, incoming - 1)

    def test_a_missing_key_is_tolerated(self):
        """Pipeline stages hold different parameters, so absence is normal, not an error."""
        param = _zero_centred_param()
        action = ResidentZeroCenteredGammaSync(key=_KEY, source_param=param)

        action.before_push({})
        action.after_push({})
        action.after_pull({})


class TestTheConverterChoosesByMode:
    def test_direct_supply_selects_the_resident_action(self):
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[2]
            / "psrl/utils/converter/megatron_converter.py"
        ).read_text()
        at = source.index("if self.resident_exposure:")
        branch = source[at : at + 600]

        assert "ResidentZeroCenteredGammaSync" in branch
        assert "torch.empty_like(param)" in branch

    def test_the_worker_asks_the_mode_rather_than_hardcoding(self):
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[2]
            / "psrl/workers/train/engine_train_worker.py"
        ).read_text()

        assert "resident_exposure=not has_ps_data_plane(" in source
