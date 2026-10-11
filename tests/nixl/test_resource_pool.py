"""Fractional GPU bundles survive the move off the patched verl checkout.

`colocate_validate_and_train` places a train worker and a validate worker on one physical
GPU by requesting unequal fractions of it (0.9 / 0.1). That capability was a patch inside
the vendored verl checkout, which accepted it as
`RayResourcePool(resource_num_per_bundle=...)`. Upstream verl has no such parameter, so
after the verl bump the call site kept passing an argument the signature no longer took.

The failure mode is what makes this worth pinning: a `TypeError` raised from
`init_workers()` -- after Ray was up, the dataset loaded and the actors placed, minutes
into a run, in a traceback naming verl rather than the placement decision behind it.

These tests check the bundle dict and the derived worker request without a Ray cluster,
which is where the fraction has to appear; whether Ray then honours it is Ray's contract.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_test

_REPO = Path(__file__).resolve().parents[2]


def _source(relative: str) -> str:
    return (_REPO / relative).read_text()


class TestTheFractionReachesTheBundle:
    def test_the_bundle_requests_the_configured_fraction(self):
        """The one line that differs from verl's implementation."""
        source = _source("psrl/trainer/ppo/resource_pool.py")

        assert "bundle[device_name] = self.resource_num_per_bundle" in source

    def test_a_whole_device_is_the_default(self):
        """Anything else would silently change placement for every non-colocated pool."""
        source = _source("psrl/trainer/ppo/resource_pool.py")

        assert "resource_num_per_bundle: float = 1.0" in source

    def test_the_subpool_stores_the_fraction(self):
        """The elastic path reads it back off the parent to build the next subgroup.

        A subpool is handed its parent's placement groups and never builds bundles, so
        this value is never applied there -- but dropping it would turn a fractional
        parent into whole-GPU children one construction later.
        """
        source = _source("psrl/trainer/ppo/resource_pool.py")
        at = source.index("class PSRL_SubRayResourcePool")

        assert "self.resource_num_per_bundle = resource_num_per_bundle" in source[at:]


class TestTheWorkerRequestMatchesTheBundle:
    """The half that a fractional bundle alone does not buy.

    Bundles of `GPU: 0.9` with workers asking for `1.0` fail in Ray with a message that
    names the bundle list, not the fraction -- so the two must derive from one value.
    """

    def test_the_view_makes_verls_arithmetic_land_on_the_fraction(self):
        """verl computes `1 / max_colocate_count`; that must come out as the fraction."""
        from psrl.trainer.ppo.resource_pool import _FractionalPoolView

        class _Pool:
            max_colocate_count = 1
            use_gpu = True

        for fraction in (0.9, 0.1, 0.5):
            view = _FractionalPoolView(_Pool(), fraction)

            assert 1 / view.max_colocate_count == pytest.approx(fraction)

    def test_it_divides_by_colocate_count_as_verl_does(self):
        """With colocation the fraction is shared, exactly as the old parameter did."""
        from psrl.trainer.ppo.resource_pool import _FractionalPoolView

        class _Pool:
            max_colocate_count = 2
            use_gpu = True

        view = _FractionalPoolView(_Pool(), 0.9)

        assert 1 / view.max_colocate_count == pytest.approx(0.9 / 2)

    def test_other_attributes_pass_through(self):
        """A view that shadowed anything else would silently change placement."""
        from psrl.trainer.ppo.resource_pool import _FractionalPoolView

        class _Pool:
            max_colocate_count = 1
            use_gpu = True
            store = [4]

        view = _FractionalPoolView(_Pool(), 0.9)

        assert view.use_gpu is True
        assert view.store == [4]

    def test_a_whole_device_is_left_untouched(self):
        """No view at all when the fraction is 1.0, so the default path stays verl's."""
        source = _source("psrl/trainer/ppo/resource_pool.py")
        at = source.index("def _create_worker")

        assert "if fraction != 1.0:" in source[at:]


class TestNoSiteUsesTheVerlClassesDirectly:
    """A missed call site fails the same way the original did, just later."""

    @pytest.mark.parametrize(
        "relative",
        ["psrl/trainer/ppo/utils.py", "psrl/trainer/ppo/ray_trainer.py"],
    )
    def test_construction_goes_through_the_brianparker325085_classes(self, relative):
        """Matches whole identifiers, so `PSRL_SubRayResourcePool` is not read as verl's.

        A plain substring search cannot tell them apart -- verl's names are suffixes of
        psrl's -- which is the kind of near-miss that makes a guard test pass while the
        thing it guards is broken.
        """
        source = _source(relative)

        bare = re.findall(
            r"(?<![\w.])((?:Sub)?RayResourcePool|RayWorkerGroup)\s*\(", source
        )

        assert not bare, (
            f"{relative} constructs verl's {set(bare)} directly; those signatures do "
            f"not carry the GPU fraction"
        )


class TestTheReasonIsRecorded:
    def test_the_module_says_why_brianparker325085_owns_this(self):
        """Without the why, this reads as a gratuitous fork of a verl class."""
        source = _source("psrl/trainer/ppo/resource_pool.py")

        assert "colocate_validate_and_train" in source
        assert "patch" in source

    def test_the_inequality_is_explained_not_just_stated(self):
        """0.9/0.1 looks arbitrary; that it must be *unequal* is the mechanism."""
        source = _source("psrl/trainer/ppo/resource_pool.py")

        assert "0.5" in source
