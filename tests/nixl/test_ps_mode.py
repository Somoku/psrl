"""What each `ps_mode` means, and that every guard asks the right question (D3, T2).

The modes used to be tested inline as `ps_mode == "nixl_cpu" or ps_mode == "nixl_gpu"`,
twelve times across six files. Adding `nixl_direct` therefore became a search problem:
config validation and the trainer's worker setup accepted it while ten other guards still
rejected it, so the mode existed without being usable -- and nothing said so, because each
guard failed on its own terms far from the decision that introduced the mode.

The distinction those guards were conflating is the one these tests pin: **"does the
transport use NIXL"** (true of direct supply) versus **"do PS workers hold the weights"**
(false of it). Most sites meant the former and were written as the latter.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from psrl.utils.ps_mode import (
    ALL_MODES,
    has_ps_data_plane,
    requires_ps_workers,
    uses_nixl,
    validate_mode,
)

pytestmark = pytest.mark.cpu_test

_REPO = Path(__file__).resolve().parents[2]


class TestTheTruthTable:
    @pytest.mark.parametrize(
        ("mode", "nixl", "ps_plane"),
        [
            ("cpu", False, True),
            ("cpu_ref", False, True),
            ("nixl_cpu", True, True),
            ("nixl_gpu", True, True),
            ("nixl_direct", True, False),
        ],
    )
    def test_each_mode_answers_both_questions(self, mode, nixl, ps_plane):
        assert uses_nixl(mode) is nixl
        assert has_ps_data_plane(mode) is ps_plane

    def test_direct_supply_is_the_only_nixl_mode_without_a_ps(self):
        """The property the whole item turns on."""
        nixl_without_ps = {
            mode for mode in ALL_MODES if uses_nixl(mode) and not has_ps_data_plane(mode)
        }

        assert nixl_without_ps == {"nixl_direct"}

    def test_direct_supply_needs_no_ps_workers(self):
        assert not requires_ps_workers("nixl_direct")
        assert requires_ps_workers("nixl_cpu")

    def test_an_unknown_mode_is_rejected_rather_than_defaulted(self):
        """Silently taking the else-branch is how a typo becomes a wrong transport."""
        with pytest.raises(ValueError, match="unknown ps_mode"):
            validate_mode("nixl_dircet")

    def test_every_predicate_rejects_an_unknown_mode(self):
        for predicate in (uses_nixl, has_ps_data_plane, requires_ps_workers):
            with pytest.raises(ValueError):
                predicate("nope")


class TestTheGuardsWereActuallyReplaced:
    """A predicate nobody calls leaves the duplication in place."""

    GUARDED = (
        "psrl/trainer/ppo/ray_trainer.py",
        "psrl/workers/train/base_train_worker.py",
        "psrl/workers/gen/vllm_async_server.py",
        "psrl/workers/ps/ps_manager.py",
    )

    def test_the_duplicated_idiom_is_gone(self):
        """One exception remains and is legitimate; everything else must use a predicate."""
        offenders = []
        for relative in self.GUARDED:
            source = (_REPO / relative).read_text()
            for match in re.finditer(
                r'ps_mode == "nixl_cpu" or [^\n]*ps_mode == "nixl_gpu"', source
            ):
                offenders.append(f"{relative}:{source[: match.start()].count(chr(10)) + 1}")

        assert not offenders, (
            f"these sites still test the modes inline and would reject nixl_direct: "
            f"{offenders}"
        )

    def test_the_remaining_nixl_cpu_test_is_placement_specific(self):
        """`nixl_cpu` alone is still meaningful: it decides *where* PS workers go.

        Keeping it is correct -- the question there is host-versus-device placement, not
        whether a PS exists -- so this test records why it survived the sweep rather than
        leaving it to look like an oversight.
        """
        source = (_REPO / "psrl/trainer/ppo/ray_trainer.py").read_text()
        remaining = re.findall(r'ps_mode == "nixl_cpu"(?! or)', source)

        assert len(remaining) == 1
        at = source.index('ps_mode == "nixl_cpu"')
        following = source[at : at + 300]
        assert "ps_node_ids" in following or "deployed" in following

    def test_each_guarded_file_uses_a_predicate(self):
        for relative in self.GUARDED:
            source = (_REPO / relative).read_text()
            assert "from psrl.utils.ps_mode import" in source, relative


class TestIntentIsDistinguishable:
    def test_push_is_gated_on_the_data_plane_not_on_nixl(self):
        """Pushing to a PS is meaningless without one, however the transport works."""
        source = (_REPO / "psrl/workers/train/base_train_worker.py").read_text()
        at = source.index("self.nixl_push_model()")
        preceding = source[max(0, at - 600) : at]

        assert "has_ps_data_plane" in preceding

    def test_pull_is_gated_on_nixl_not_on_the_data_plane(self):
        """A rollout pull works whether the source is a PS or a train rank."""
        source = (_REPO / "psrl/workers/gen/vllm_async_server.py").read_text()
        at = source.index("await self.nixl_pull_model()")
        preceding = source[max(0, at - 300) : at]

        assert "uses_nixl" in preceding
