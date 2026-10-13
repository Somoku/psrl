"""The four-dimensional configuration matrix and its CI tiers (implementation item T4).

The design does not run a full Cartesian product -- backend x precision x placement x
parallelism is hundreds of combinations and most add no information. It names three tiers
by risk, and the value of writing them down is that "which configurations are covered" is
otherwise a question nobody can answer without reading every YAML.

These tests check the matrix against the configs that exist, so a tier entry with no
runnable config fails here rather than being discovered when someone tries to run CI.
They deliberately do *not* run the configurations: that is a multi-node job measured in
tens of minutes, and its result is recorded in the implementation checklist.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from omegaconf import OmegaConf

pytestmark = pytest.mark.cpu_test

CONFIG_DIR = Path(__file__).resolve().parent / "config"


@dataclass(frozen=True)
class Combination:
    """One cell of the matrix, as the design names it."""

    backend: str
    train_dtype: str
    rollout_dtype: str
    placement: str
    tier: str
    #: Config file implementing it, or None when it is not yet runnable.
    config: str | None
    why: str = ""


#: Every cell the design lists, with the config that realises it.
#:
#: `config=None` is a deliberate record of a gap rather than an omission: the FP8 cells
#: need the cast path (A9 is verified at unit level but not wired into the harness), and
#: the FSDP2 cells need an FSDP2 e2e config. Writing them down keeps the matrix honest
#: about what is *not* covered.
MATRIX: tuple[Combination, ...] = (
    Combination(
        "megatron", "bf16", "fp8", "disaggregate", "every-ci",
        None, "cast path not wired into the harness (A9 is unit-verified only)",
    ),
    Combination(
        "fsdp2", "fp32", "bf16", "disaggregate", "every-ci",
        None, "no FSDP2 e2e config yet; converter is verified under torchrun",
    ),
    Combination(
        "megatron", "bf16", "bf16", "colocate", "every-ci",
        "nixl_e2e_colocate.yaml", "timing-exclusion path",
    ),
    Combination(
        "megatron", "bf16", "bf16", "disaggregate", "daily",
        "nixl_e2e_5node_direct.yaml", "TP mismatch exercises shard algebra",
    ),
    Combination(
        "fsdp2", "bf16", "bf16", "disaggregate", "daily",
        "nixl_e2e_5node_fsdp2.yaml",
        "the cell that makes the version barrier (E1) mandatory",
    ),
    Combination(
        "megatron", "fp8", "fp8", "colocate", "daily",
        None, "FP8-native; needs TE fp8_param plus the cast path",
    ),
)


def _load(name: str):
    return OmegaConf.load(CONFIG_DIR / name)


class TestEveryNamedConfigExists:
    @pytest.mark.parametrize(
        "combination", [c for c in MATRIX if c.config], ids=lambda c: c.config
    )
    def test_the_config_file_is_present(self, combination: Combination):
        assert (CONFIG_DIR / combination.config).is_file()

    @pytest.mark.parametrize(
        "combination", [c for c in MATRIX if c.config], ids=lambda c: c.config
    )
    def test_the_config_matches_the_cell_it_claims(self, combination: Combination):
        """A config drifting from its matrix row would silently change what CI covers."""
        cfg = _load(combination.config)

        assert cfg.test.train_engine_type == combination.backend
        assert cfg.model.train_dtype.startswith(combination.train_dtype[:2])
        assert cfg.model.gen_dtype.startswith(combination.rollout_dtype[:2])
        is_colocate = bool(cfg.test.get("colocate", False))
        assert is_colocate == (combination.placement == "colocate")


class TestGapsAreRecordedNotHidden:
    def test_every_uncovered_cell_says_why(self):
        """An empty config slot must carry its reason, or it reads as an oversight."""
        for combination in MATRIX:
            if combination.config is None:
                assert combination.why, f"{combination} has no config and no reason"

    def test_the_colocate_cell_is_covered(self):
        """T3's output: the timing-exclusion path is the one every-ci cell that runs."""
        colocate = [c for c in MATRIX if c.placement == "colocate" and c.tier == "every-ci"]

        assert colocate and all(c.config for c in colocate)

    def test_at_least_one_cell_per_placement_is_runnable(self):
        """Both placements must be exercisable, or §3.7's conclusions stay untested."""
        for placement in ("disaggregate", "colocate"):
            runnable = [c for c in MATRIX if c.placement == placement and c.config]
            assert runnable, f"no runnable config for {placement}"


class TestTiersArePartitioned:
    def test_every_cell_has_a_known_tier(self):
        for combination in MATRIX:
            assert combination.tier in ("every-ci", "daily", "pre-release")

    def test_the_every_ci_tier_stays_small(self):
        """It runs on every change, so growth here is a direct cost on iteration speed."""
        every_ci = [c for c in MATRIX if c.tier == "every-ci"]

        assert len(every_ci) <= 4, (
            f"{len(every_ci)} configurations in the per-change tier; the design names "
            f"three, and each one is a multi-node run"
        )


class TestTheTwoDisaggregateConfigsDifferByOneThing:
    """The PS and direct-supply configs must stay comparable.

    They exist to be diffed: if they differed in more than the supply mode, a performance
    difference between them could not be attributed.
    """

    def test_they_differ_only_in_supply_related_keys(self):
        ps = OmegaConf.to_container(_load("nixl_e2e_5node.yaml"), resolve=False)
        direct = OmegaConf.to_container(_load("nixl_e2e_5node_direct.yaml"), resolve=False)

        differing = {
            key
            for key in set(ps["test"]) | set(direct["test"])
            if ps["test"].get(key) != direct["test"].get(key)
        }

        assert differing <= {"direct_supply"}, (
            f"the two configs differ in {sorted(differing)}; a measured difference "
            f"between them could not be attributed to the supply mode alone"
        )


class TestEveryMatrixBackendIsOneTheHarnessAccepts:
    """The matrix names backends the harness has to build.

    `fsdp2` and `fsdp` were different spellings of the same thing -- the harness's
    `fsdp` branch calls `fully_shard`, and FSDP1 is rejected outright by config
    validation -- so the matrix's `fsdp2` cell had no config while the capability
    existed. A name the harness does not accept fails at startup, deep in a ~30-minute
    multi-node run, rather than here.
    """

    def _harness_backends(self) -> set[str]:
        from pathlib import Path
        import re

        source = (
            Path(__file__).resolve().parent / "test_nixl_e2e.py"
        ).read_text()
        match = re.search(r"engine_type in \[(.*?)\]", source, re.S)

        assert match, "could not find the harness's accepted engine types"
        return set(re.findall(r'"([a-z0-9_]+)"', match.group(1)))

    def test_every_named_backend_is_accepted(self):
        accepted = self._harness_backends()
        named = {combination.backend for combination in MATRIX}

        assert named <= accepted, (
            f"the matrix names backends the harness rejects: {sorted(named - accepted)}"
        )

    def test_the_check_reads_the_real_allowlist(self):
        """Guard against the regex silently matching nothing and the test passing."""
        accepted = self._harness_backends()

        assert "megatron" in accepted


class TestTheFsdpAliasIsHandledAtEveryBranch:
    """`fsdp` and `fsdp2` must move together everywhere, not just where I remembered.

    Adding the alias only to the model-construction branch left `protocol` falling through
    every `elif`, which surfaced on the cluster as `UnboundLocalError: state_dict` after
    ~16 minutes of startup. An equality test against the bare string is the shape that
    misses; a set membership test is the shape that does not.
    """

    def _harness_source(self) -> str:
        from pathlib import Path

        return (Path(__file__).resolve().parent / "test_nixl_e2e.py").read_text()

    def test_no_branch_tests_the_bare_fsdp_name(self):
        import re

        offenders = re.findall(
            r'engine_type == "fsdp"|engine_type != "fsdp"', self._harness_source()
        )

        assert not offenders, (
            f'{len(offenders)} branch(es) test `engine_type == "fsdp"`, which excludes the '
            f'`fsdp2` spelling the config matrix uses; use a set membership test'
        )

    def test_fsdp2_reaches_the_same_branches_as_fsdp(self):
        """Every tuple that names `fsdp` must also name `fsdp2`."""
        import re

        source = self._harness_source()
        tuples = re.findall(r"engine_type in \(([^)]*)\)", source)

        assert tuples, "found no membership tests to check"
        for group in tuples:
            names = set(re.findall(r'"([a-z0-9_]+)"', group))
            if "fsdp" in names:
                assert "fsdp2" in names, f"branch {group!r} names fsdp but not fsdp2"


class TestEveryHarnessConfigHasATopLevelNixlSection:
    """
    The harness reads `cfg.nixl` whole, so a lost `nixl:` header silently moves every option under `network:`.
    """

    @pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("nixl_e2e*.yaml")), ids=lambda p: p.name)
    def test_nixl_options_live_under_nixl(self, path):
        cfg = OmegaConf.load(path)
        assert "nixl" in cfg, f"{path.name} has no top level nixl section."
        assert "server_mode" in cfg.nixl, f"{path.name} is missing nixl.server_mode."
        stray = {"handshake_timeout_s", "server_mode", "peer_diffusion"} & set(cfg.get("network", {}))
        assert not stray, f"{path.name} parses NIXL options {sorted(stray)} under network."
