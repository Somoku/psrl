"""Colocate switch order and stage-offload policy (design §3.7.3), on CPU.

Both encode a decision that, wrong, corrupts silently: reading a trainer that already
left, or releasing the only copy of something.
"""

from __future__ import annotations

import pytest

from psrl.trainer.ppo.colocate_switch import Step, check_order, to_rollout_steps
from psrl.utils.nixl.stage_offload import stage_policy

pytestmark = pytest.mark.cpu_test


class TestSwitchOrder:
    @pytest.mark.parametrize("has_ps", [True, False])
    def test_both_plans_satisfy_the_invariants(self, has_ps):
        check_order(to_rollout_steps(has_ps_data_plane=has_ps), trainer_is_source=not has_ps)

    def test_direct_supply_pulls_before_the_trainer_leaves(self):
        """The bug this replaces: the trainer slept first, and the validate instances
        then read a source whose memory was deregistered and whose bytes were gone."""
        steps = to_rollout_steps(has_ps_data_plane=False)

        assert steps.index(Step.VALIDATE_PULL) < steps.index(Step.TRAINER_OFFLOAD)

    def test_direct_supply_allocates_kv_only_after_the_trainer_leaves(self):
        """vLLM sized its KV cache against the whole card; waking it with the trainer
        resident asks for memory that is not free."""
        steps = to_rollout_steps(has_ps_data_plane=False)

        assert Step.VALIDATE_WAKE_ALL not in steps
        assert steps.index(Step.TRAINER_OFFLOAD) < steps.index(Step.VALIDATE_WAKE_KV)

    def test_the_old_order_is_rejected_under_direct_supply(self):
        old = to_rollout_steps(has_ps_data_plane=True)

        with pytest.raises(ValueError, match="reads the trainer after"):
            check_order(old, trainer_is_source=True)

    def test_a_parameter_server_keeps_the_trainer_first_order(self):
        """The validate instance gets its whole KV budget from the start; nothing reads
        the trainer, so there is no reason to keep it resident."""
        steps = to_rollout_steps(has_ps_data_plane=True)

        assert steps[0] is Step.TRAINER_OFFLOAD

    def test_resuming_without_a_kv_cache_is_rejected(self):
        steps = [
            Step.VALIDATE_WAKE_WEIGHTS,
            Step.VALIDATE_PUBLISH,
            Step.VALIDATE_PULL,
            Step.TRAINER_OFFLOAD,
            Step.VALIDATE_RESUME,
        ]

        with pytest.raises(ValueError, match="KV cache"):
            check_order(steps, trainer_is_source=True)

    def test_pulling_before_publishing_is_rejected(self):
        """The new addresses must reach the planner's view before anyone reads into them."""
        steps = [Step.VALIDATE_WAKE_WEIGHTS, Step.VALIDATE_PULL, Step.VALIDATE_PUBLISH]

        with pytest.raises(ValueError, match="published"):
            check_order(steps, trainer_is_source=True)


class TestStagePolicy:
    def test_direct_supply_preserves_weights(self):
        """The trainer is the only copy; releasing it is data loss."""
        policy = stage_policy(has_ps_data_plane=False, offload_optimizer=True, tms_pause=False)

        assert policy.preserve_weights

    def test_a_parameter_server_lets_weights_go(self):
        """The switch-back pulls them; a host copy would be 2x PCIe for nothing."""
        policy = stage_policy(has_ps_data_plane=True, offload_optimizer=True, tms_pause=False)

        assert not policy.preserve_weights

    @pytest.mark.parametrize("has_ps", [True, False])
    def test_the_optimizer_is_preserved_in_every_mode(self, has_ps):
        """fp32 masters and Adam moments exist nowhere else, PS or not."""
        policy = stage_policy(has_ps_data_plane=has_ps, offload_optimizer=True, tms_pause=False)

        assert policy.offload_optimizer

    def test_opting_out_of_optimizer_offload_is_honoured_without_tms(self):
        policy = stage_policy(has_ps_data_plane=True, offload_optimizer=False, tms_pause=False)

        assert not policy.offload_optimizer

    def test_a_tms_pause_overrides_the_opt_out(self):
        """TMS is configured without CPU backup, so a pause drops the optimizer and the
        next step writes the lost fp32 masters over the model. Honouring the opt-out
        there would be honouring it into silent corruption."""
        policy = stage_policy(has_ps_data_plane=True, offload_optimizer=False, tms_pause=True)

        assert policy.offload_optimizer
