"""`ps_mode=nixl_direct` in production, the prerequisite for removing the PS (D3/T2).

Direct supply was reachable only from the e2e harness: `direct_supply` is a test config
key, and `ray_trainer.py` dispatched on `ps_mode`, which had no value meaning "no PS".
Deleting `PSStorageWorker` before that gap closed would have left production with no
working transport -- the harness would still pass, because it never consults `ps_mode`.

So the deletion is gated on this mode existing, being validated, and refusing the
configurations that assume a PS holds weights. These tests cover the gate; removing the
worker itself is the step after.

Source-level rather than behavioural, for the same reason as the switch-back tests:
`ray_trainer` cannot be imported here (a pre-existing `verl.trainer.ppo.metric_utils`
skew), and the property is a dispatch branch that reading the file pins exactly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_test

_REPO = Path(__file__).resolve().parents[2]


def _source(relative: str) -> str:
    return (_REPO / relative).read_text()


class TestTheModeIsDispatched:
    def test_the_trainer_has_a_nixl_direct_branch(self):
        source = _source("psrl/trainer/ppo/ray_trainer.py")

        assert 'self.config.psrl.ps_mode == "nixl_direct"' in source

    def test_the_branch_creates_no_ps_workers(self):
        """The whole point: no PS data plane. A branch that still built one is a no-op.

        Comments are stripped before the check: the branch explains *why* it creates no
        `PSStorageWorker`, and a substring search over the raw text would read that
        explanation as the thing it forbids.
        """
        source = _source("psrl/trainer/ppo/ray_trainer.py")
        start = source.index('if self.config.psrl.ps_mode == "nixl_direct":')
        branch = source[start : source.index("elif", start)]
        code = "\n".join(
            line for line in branch.splitlines() if not line.strip().startswith("#")
        )

        assert "PSWorkerGroup(" not in code
        assert "PSStorageWorker(" not in code
        assert "create_ps_worker_group" not in code

    def test_the_branch_is_checked_before_the_ps_modes(self):
        """Order matters: a later branch would be shadowed by the PS-mode conditions."""
        source = _source("psrl/trainer/ppo/ray_trainer.py")
        direct_at = source.index('self.config.psrl.ps_mode == "nixl_direct"')
        ps_at = source.index('self.config.psrl.ps_mode == "cpu" or')

        assert direct_at < ps_at


class TestTheModeIsValidated:
    def test_config_validation_knows_the_mode(self):
        source = _source("psrl/utils/config.py")

        assert 'config.psrl.ps_mode == "nixl_direct"' in source

    def test_it_inherits_the_nixl_checks(self):
        """Direct supply is still NIXL, so FSDP1 and the offload flags still apply."""
        source = _source("psrl/utils/config.py")
        start = source.index('if config.psrl.ps_mode == "nixl_direct":')
        block = source[start : source.index("if config.psrl.ps_mode == \"nixl_cpu\"", start)]

        assert "_validate_nixl_offload_flags" in block
        assert "_validate_nixl_lora" in block
        assert "FSDP1 is not supported" in block

    def test_it_refuses_lmcache_p2p(self):
        """p2p routes through a PS, so pairing it with direct supply cannot work.

        Refusing at config time rather than mid-transfer, because the failure would
        otherwise surface as a routing error far from its cause.
        """
        source = _source("psrl/utils/config.py")
        start = source.index('if config.psrl.ps_mode == "nixl_direct":')
        block = source[start : start + 1400]

        assert "enable_p2p" in block
        assert "nixl_direct" in block


class TestTheDeletionGateIsRecorded:
    def test_the_branch_says_why_it_blocks_the_deletion(self):
        """Whoever removes the PS next needs to know this had to come first."""
        source = _source("psrl/trainer/ppo/ray_trainer.py")
        start = source.index('if self.config.psrl.ps_mode == "nixl_direct":')
        branch = source[start : start + 1200]

        assert "D3" in branch or "T2" in branch

    def test_the_ps_worker_still_exists(self):
        """Fails once the PS is removed, which is the signal to retire this gate.

        Deliberately a failing-on-success test: it marks the boundary between "direct
        supply is available" and "the PS is gone", so the two steps cannot be conflated.
        """
        assert (_REPO / "psrl" / "workers" / "ps" / "ps_storage_worker.py").is_file(), (
            "PSStorageWorker has been removed; D3/T2 are complete and this gate test "
            "should be deleted along with the PS-mode branches it guards"
        )


class TestEveryPsWorkerGroupAccessIsGuarded:
    """The bug this class exists for: `self.ps_wg` read in a NIXL-gated block.

    `ps_mode.py` separated "uses NIXL" from "has a PS data plane" precisely so direct
    supply could take the NIXL paths without the PS ones. The separation does nothing
    unless every `ps_wg` access actually asks the second question -- and one that did not
    produced `'PSRL_RayPPOTrainer' object has no attribute 'ps_wg'` in production, after
    Ray was up, the servers had registered and the NIXL clients had initialised.

    Source-level because the attribute only exists on a live trainer with a PS, so the
    failure cannot be reached without a cluster. What a reader can check is that no access
    sits outside a `has_ps_data_plane` guard.
    """

    _GUARDS = ("has_ps_data_plane", "requires_ps_workers")

    def _guarded_lines(self, tree, source: str) -> set[int]:
        """Line numbers of `ps_wg` accesses that some enclosing construct makes safe.

        Scope, not proximity: an access is safe if it sits inside an `if` whose test asks
        the PS-data-plane question, or if `self.ps_wg` was assigned earlier in the same
        block. A line-distance heuristic gets both wrong -- the real guards here sit 15
        lines up, and a nearby guard in a *sibling* branch would wrongly look like cover.
        """
        import ast

        safe: set[int] = set()

        def accesses_within(node) -> set[int]:
            return {
                child.lineno
                for child in ast.walk(node)
                if isinstance(child, ast.Attribute)
                and child.attr == "ps_wg"
                and isinstance(child.value, ast.Name)
                and child.value.id == "self"
            }

        for node in ast.walk(tree):
            # `if guard: ...` and `X if guard else Y` both cover their access; only the
            # first is an `ast.If`, and missing the second reads a correct guard as absent.
            if isinstance(node, (ast.If, ast.IfExp)):
                test = ast.get_source_segment(source, node.test) or ""
                if any(guard in test for guard in self._GUARDS):
                    # `if not has_ps_data_plane(...): ... else: <PS work>` is the same
                    # guard with the branches swapped. Covering only `body` would read
                    # the PS branch as unguarded and the direct-supply branch as safe --
                    # exactly backwards.
                    negated = isinstance(node.test, ast.UnaryOp) and isinstance(
                        node.test.op, ast.Not
                    )
                    if isinstance(node, ast.IfExp):
                        covered = [node.orelse if negated else node.body]
                    else:
                        covered = node.orelse if negated else node.body
                    for statement in covered:
                        safe |= accesses_within(statement)

            # A block that builds the group covers the rest of that block.
            for field in ("body", "orelse", "finalbody"):
                block = getattr(node, field, None)
                if not isinstance(block, list):
                    continue
                built_at = None
                for statement in block:
                    if built_at is not None:
                        safe |= accesses_within(statement)
                        continue
                    targets = getattr(statement, "targets", [])
                    for target in targets:
                        segment = ast.get_source_segment(source, target) or ""
                        if segment.strip() == "self.ps_wg":
                            built_at = statement.lineno
        return safe

    def test_no_access_sits_outside_a_data_plane_guard(self):
        import ast

        source = _source("psrl/trainer/ppo/ray_trainer.py")
        tree = ast.parse(source)
        safe = self._guarded_lines(tree, source)

        unguarded = sorted(
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr == "ps_wg"
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.lineno not in safe
            and not isinstance(node.ctx, ast.Store)
        )

        lines = source.splitlines()
        detail = [f"{n}: {lines[n - 1].strip()[:70]}" for n in unguarded]

        assert not unguarded, (
            f"these `ps_wg` accesses are governed by no PS-data-plane guard and would "
            f"raise AttributeError under ps_mode=nixl_direct: {detail}"
        )

    def test_the_predicate_is_imported(self):
        """A guard spelled with the wrong predicate reintroduces the bug."""
        source = _source("psrl/trainer/ppo/ray_trainer.py")

        assert "has_ps_data_plane" in source
        assert "from psrl.utils.ps_mode import" in source

    def test_the_agent_count_excludes_absent_ps_workers(self):
        """An over-count does not fail here; it hangs the meta server handshake.

        Worth pinning separately from the attribute error: this one has no traceback at
        the cause, only a timeout waiting for agents that were never going to register.
        """
        source = _source("psrl/trainer/ppo/ray_trainer.py")
        at = source.index("expected_nixl_client_agents =")
        preceding = source[max(0, at - 700) : at]

        assert "has_ps_data_plane" in preceding


class TestTheGenSidePullResolvesTrainRanks:
    """Direct supply must still answer "what do I read", not skip the question.

    The gen workers resolve pull sources by asking the PSManager, in every mode. An early
    version of this branch skipped the bind entirely, on the reasoning that there is no PS
    worker group -- and the pull then failed inside `get_ps_nixl_agent_names`, which
    asserts rather than returning an empty list. The fix is a different *answer* under the
    same names, which is what keeps one pull path on the gen side.
    """

    def test_the_manager_can_bind_train_ranks_as_sources(self):
        source = _source("psrl/workers/ps/ps_manager.py")

        assert "def bind_direct_supply_sources" in source

    def test_the_three_name_lists_coincide_for_train_ranks(self):
        """A train rank is its own agent; a PS worker is not, and that is the difference.

        A PS worker runs separate push-side and pull-side clients under one agent, so its
        three lists differ. Deriving the gen list from anything but the train client name
        would send rollouts to an agent that never registered.
        """
        source = _source("psrl/workers/ps/ps_manager.py")
        at = source.index("def bind_direct_supply_sources")
        body = source[at : at + 1800]

        assert "train_client_name(rank)" in body
        assert body.count("list(self.ps_nixl_agent_names)") == 2

    def test_the_trainer_binds_sources_in_both_modes(self):
        """Only the answer differs; skipping the bind is what broke the pull."""
        source = _source("psrl/trainer/ppo/ray_trainer.py")

        assert "bind_direct_supply_sources" in source
        assert "bind_ps_worker_group" in source

    def test_the_train_ranks_do_not_pull_from_themselves(self):
        """They are the source. Pulling resolves PS names that do not exist."""
        source = _source("psrl/trainer/ppo/ray_trainer.py")

        for index in range(len(source)):
            index = source.find('execute_all_async("pull_model")', index)
            if index == -1:
                break
            preceding = source[max(0, index - 900) : index]
            assert "has_ps_data_plane" in preceding, (
                "an actor pull_model is not gated on the PS data plane; under direct "
                "supply it asserts inside get_ps_nixl_agent_names"
            )
            index += 1


class TestTheProgressThreadIsOnWhenTrainRanksServeReads:
    """The step-2 hang: a transfer that commits and then never completes.

    Under direct supply a train rank is read by rollouts at arbitrary times, including
    while it is computing. `enable_prog_thread=False` leaves nothing advancing the UCX
    worker, so the reader's transfer sits in `PROC` until its timeout -- surfacing as
    "Timeout waiting for transfer (..., READ, ...) to NIXLTrainClient_N", on the *reader*,
    with nothing logged on the rank that failed to serve it.

    It passed step 1 and failed step 2, because the first pull happens while the trainer
    is otherwise idle. That is what makes it worth a test: "it worked once" is not
    evidence here.
    """

    def test_direct_supply_forces_the_thread_on(self):
        source = _source("psrl/workers/train/engine_train_worker.py")

        assert "serves_direct_reads or not self.psrl_config.checkpoint.use_dcp_save" in source

    def test_the_conflicting_combination_is_refused(self):
        """DCP wants it off, direct supply wants it on; picking silently hides the gap."""
        source = _source("psrl/utils/config.py")
        at = source.index('if config.psrl.ps_mode == "nixl_direct":')
        block = source[at : at + 3000]

        assert "use_dcp_save" in block

    def test_the_rule_lives_where_the_whole_config_is_visible(self):
        """The worker sees only its own actor config, so it cannot ask about saving.

        A first attempt put this in `engine_train_worker.__init__` and read
        `self.config.trainer.save_freq`, which raised `ConfigAttributeError` inside actor
        construction -- surfacing as `ActorDiedError` during `init_model`, far from the
        config decision that caused it.
        """
        worker = _source("psrl/workers/train/engine_train_worker.py")

        assert "self.config.trainer.save_freq" not in worker

    def test_saving_works_through_the_per_rank_manager(self):
        """Direct supply must not force `save_freq=-1`: 65 of 75 example scripts save.

        `use_dcp_save=False` is mandatory here (the progress thread has to stay up), so
        without a non-dist saver every saving script would be unrunnable under direct
        supply -- and the failure would land at the first `save_freq` boundary, hours in.
        """
        worker = _source("psrl/workers/train/engine_train_worker.py")

        assert "_maybe_use_per_rank_checkpointing" in worker
        assert "PSRL_PerRankMegatronCheckpointManager" in worker

    def test_the_resharding_cost_is_written_down(self):
        """Per-rank files cannot be resumed at a different world size; that must be said."""
        manager = _source("psrl/utils/checkpoint/megatron_manager.py")

        assert "resharded" in manager or "world size" in manager

    def test_the_example_script_sets_a_runnable_combination(self):
        """`use_dcp_save=False` is mandatory under direct supply, so the script must set it.

        `save_freq` is deliberately *not* pinned: per-rank checkpointing makes saving
        work, and the script leaves it overridable so a run can exercise that path.
        """
        script = _source("examples/grpo_trainer/qwen3_8b_megatron_direct.sh")

        assert "psrl.ps_mode=nixl_direct" in script
        assert "psrl.checkpoint.use_dcp_save=False" in script
