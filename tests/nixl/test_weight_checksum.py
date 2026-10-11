"""The digest has to catch the corruption RDMA actually produces (§3.7).

Direct supply deletes the middle observation point: with a PS you could ask "does the PS
copy match the rollout copy" and separate a transport fault from a training fault. Reading
train ranks directly removes that, and the faults it replaces it with are silent -- the
receiving buffer is pre-allocated and correctly shaped, so a short, misplaced or stale
landing yields a tensor that is valid and wrong.

So these tests are mostly adversarial: each one constructs a corruption that a *weaker*
digest would miss, and asserts this one sees it. A checksum that passes "same tensor
hashes the same" and nothing else would give false confidence, which is worse than having
no check at all.
"""

from __future__ import annotations

import pytest
import torch

from psrl.utils.nixl.weight_checksum import (
    ChecksumReport,
    checksum_state_dict,
    sample_keys,
    tensor_digest,
)

pytestmark = pytest.mark.cpu_test


class TestItCatchesCorruptionAWeakerDigestWouldMiss:
    def test_a_permuted_shard_is_detected(self):
        """A plain sum is order-blind; re-sharding across TP degrees can permute."""
        original = torch.tensor([1.0, 2.0, 3.0, 4.0])
        permuted = torch.tensor([4.0, 3.0, 2.0, 1.0])

        assert tensor_digest(original) != tensor_digest(permuted)

    def test_a_swapped_pair_is_detected(self):
        """The minimal permutation, which a sum and a mean both miss entirely."""
        original = torch.tensor([1.0, 2.0, 3.0, 4.0])
        swapped = torch.tensor([2.0, 1.0, 3.0, 4.0])

        assert tensor_digest(original) != tensor_digest(swapped)

    def test_a_single_low_bit_flip_is_detected(self):
        """FP8's low bits are where precision lives; a float-sum digest can round them away."""
        original = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        flipped = original.clone()
        flipped[1] = torch.frombuffer(
            (torch.tensor([2.0], dtype=torch.float32).view(torch.int32) ^ 1)
            .to(torch.int32)
            .numpy()
            .tobytes(),
            dtype=torch.float32,
        )[0]

        assert tensor_digest(original) != tensor_digest(flipped)

    def test_a_truncated_landing_is_detected(self):
        """A descriptor that lands short leaves a zero tail rather than failing."""
        full = torch.tensor([1.0, 2.0, 3.0, 4.0])
        short = torch.tensor([1.0, 2.0, 0.0, 0.0])

        assert tensor_digest(full) != tensor_digest(short)

    def test_a_stale_version_is_detected(self):
        """The failure mode of a missed publish: a valid tensor from the previous step."""
        current = torch.tensor([1.0, 2.0, 3.0])
        stale = current + 0.001

        assert tensor_digest(current) != tensor_digest(stale)

    def test_zeros_and_empty_do_not_collide(self):
        """An unfilled buffer must not hash like an absent one."""
        assert tensor_digest(torch.zeros(4)) != tensor_digest(torch.empty(0))


class TestItIsStableWhereItMustBe:
    def test_the_same_tensor_hashes_the_same(self):
        tensor = torch.randn(64, 32)

        assert tensor_digest(tensor) == tensor_digest(tensor.clone())

    def test_a_non_contiguous_view_hashes_as_its_values(self):
        """Strided shards are the norm here; a digest that depended on layout would
        report a false mismatch on every row-parallel tensor."""
        base = torch.randn(8, 8)
        strided = base[:, ::2]

        assert tensor_digest(strided) == tensor_digest(strided.contiguous())

    def test_large_tensors_stay_in_exact_range(self):
        """Without the modulus, float64 silently loses integer precision past 2^53 and
        distinct tensors begin to collide."""
        big = torch.ones(1 << 20, dtype=torch.float32)
        other = big.clone()
        other[1 << 19] = 2.0

        assert tensor_digest(big) != tensor_digest(other)


class TestTheReportNamesWhatDiverged:
    def test_matching_reports_compare_clean(self):
        state = {"a": torch.randn(4), "b": torch.randn(4)}
        left = checksum_state_dict(state, version=3)
        right = checksum_state_dict(state, version=3)

        assert left.compare(right) == []

    def test_a_mismatch_names_the_key(self):
        """"The model differs" is not actionable; the tensor that diverged is."""
        left = checksum_state_dict({"a": torch.ones(4), "b": torch.ones(4)}, version=3)
        right = checksum_state_dict({"a": torch.ones(4), "b": torch.zeros(4)}, version=3)

        assert left.compare(right) == ["b"]

    def test_a_missing_key_counts_as_differing(self):
        """The two sides derive key sets independently; absence is a real fault."""
        left = checksum_state_dict({"a": torch.ones(4), "b": torch.ones(4)}, version=1)
        right = checksum_state_dict({"a": torch.ones(4)}, version=1)

        assert left.compare(right) == ["b"]

    def test_comparing_across_versions_is_refused(self):
        """Otherwise a normal weight update reads as transport corruption."""
        left = checksum_state_dict({"a": torch.ones(4)}, version=1)
        right = checksum_state_dict({"a": torch.ones(4)}, version=2)

        with pytest.raises(ValueError, match="across versions"):
            left.compare(right)

    def test_the_version_travels_with_the_digest(self):
        report = checksum_state_dict({"a": torch.ones(4)}, version=7)

        assert report.version == 7


class TestSamplingIsComparableAcrossRanks:
    def test_it_is_deterministic(self):
        """Ranks choose without communicating; disagreement makes digests incomparable."""
        state = {f"layer.{i}.weight": torch.randn(2) for i in range(50)}

        assert sample_keys(state, 5) == sample_keys(state, 5)

    def test_it_does_not_depend_on_insertion_order(self):
        """State dicts are built independently on each rank and need not share order."""
        forward = {f"layer.{i}": torch.randn(2) for i in range(20)}
        backward = {f"layer.{i}": forward[f"layer.{i}"] for i in reversed(range(20))}

        assert sample_keys(forward, 4) == sample_keys(backward, 4)

    def test_it_spreads_across_the_model(self):
        """Faults cluster by layer family; the first N keys would miss a late-block bug."""
        state = {f"layer.{i:03d}": torch.randn(2) for i in range(100)}

        chosen = sample_keys(state, 4)
        indices = [int(key.split(".")[1]) for key in chosen]

        assert max(indices) > 50, f"sampled only the early layers: {chosen}"

    def test_asking_for_more_than_exists_returns_everything(self):
        state = {"a": torch.randn(2), "b": torch.randn(2)}

        assert sample_keys(state, 10) == ["a", "b"]

    def test_zero_samples_nothing(self):
        assert sample_keys({"a": torch.randn(2)}, 0) == []

    def test_a_sampled_report_only_covers_the_sample(self):
        state = {"a": torch.ones(2), "b": torch.ones(2), "c": torch.ones(2)}

        report = checksum_state_dict(state, version=1, keys=["a", "c"])

        assert set(report.digests) == {"a", "c"}


class TestItCostsNothingWhenNotAsked:
    def test_an_absent_key_is_skipped_rather_than_raising(self):
        """Pipeline stages hold different parameters; absence is normal on some ranks."""
        report = checksum_state_dict({"a": torch.ones(2)}, version=1, keys=["a", "missing"])

        assert set(report.digests) == {"a"}

    def test_report_is_hashable_and_frozen(self):
        """It crosses a Ray actor boundary; accidental mutation would be invisible."""
        report = ChecksumReport(version=1, digests={})

        with pytest.raises(Exception):
            report.version = 2  # type: ignore[misc]
