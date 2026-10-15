"""The reference policy as a swapped-in snapshot of the actor (design §3.10.2), on one GPU.

The properties that make the swap a faithful stand-in for a separate ref model:

* a forward pass under the swap equals a forward pass of an untouched copy of the
  initial model, bit for bit;
* the actor is exactly itself afterwards, including when the body raises;
* training continues identically, so the swap cannot leak into the optimizer.
"""

from __future__ import annotations

import copy

import pytest
import torch

from psrl.utils.nixl.stage_offload import StorageOffloader
from psrl.utils.nixl.weight_backup import WeightBackup

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _model():
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Linear(128, 256), torch.nn.GELU(), torch.nn.Linear(256, 64)
    ).cuda().to(torch.bfloat16)


def _train_step(model, optimizer, batch):
    optimizer.zero_grad()
    model(batch).float().pow(2).mean().backward()
    optimizer.step()


class TestTheRefForwardIsTheInitialModel:
    def test_logits_under_the_swap_equal_a_frozen_copy(self):
        model = _model()
        ref_copy = copy.deepcopy(model)
        backup = WeightBackup(model.parameters())
        backup.capture("ref")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        batch = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16)
        for _ in range(3):
            _train_step(model, optimizer, batch)
        backup.mark_modified()

        with torch.no_grad(), backup.swapped("ref"):
            under_swap = model(batch)
        expected = ref_copy(batch)

        assert torch.equal(under_swap, expected)

    def test_the_actor_is_itself_afterwards(self):
        model = _model()
        backup = WeightBackup(model.parameters())
        backup.capture("ref")
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1)
        backup.mark_modified()
        actor = [p.clone() for p in model.parameters()]

        with backup.swapped("ref"):
            pass

        for got, want in zip(model.parameters(), actor):
            assert torch.equal(got, want)

    def test_a_raising_body_still_restores_the_actor(self):
        """Otherwise the next optimizer step would train the reference policy."""
        model = _model()
        backup = WeightBackup(model.parameters())
        backup.capture("ref")
        with torch.no_grad():
            for p in model.parameters():
                p.mul_(3)
        backup.mark_modified()
        actor = [p.clone() for p in model.parameters()]

        with pytest.raises(ValueError), backup.swapped("ref"):
            raise ValueError("forward failed")

        for got, want in zip(model.parameters(), actor):
            assert torch.equal(got, want)


class TestTrainingIsUnaffected:
    def test_a_run_with_ref_passes_matches_one_without(self):
        """Adam's state lives outside the swapped storages; only the weights move, and
        they come back exactly. Any leak shows up as a diverging trajectory."""
        batch = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16)
        plain, swapped = _model(), _model()
        opt_plain = torch.optim.Adam(plain.parameters(), lr=1e-3)
        opt_swapped = torch.optim.Adam(swapped.parameters(), lr=1e-3)
        backup = WeightBackup(swapped.parameters())
        backup.capture("ref")

        for _ in range(4):
            _train_step(plain, opt_plain, batch)
            _train_step(swapped, opt_swapped, batch)
            backup.mark_modified()
            with torch.no_grad(), backup.swapped("ref"):
                swapped(batch)

        for a, b in zip(plain.parameters(), swapped.parameters()):
            assert torch.equal(a, b)


class TestCapturesAreSkippedWhenNothingChanged:
    def test_a_second_swap_without_a_step_copies_nothing_out(self):
        model = _model()
        backup = WeightBackup(model.parameters())
        backup.capture("ref")
        backup.mark_modified()

        with backup.swapped("ref") as first:
            pass
        with backup.swapped("ref") as second:
            pass

        assert first.captured_bytes == backup.nbytes
        assert second.captured_bytes == 0

    def test_a_step_between_swaps_forces_a_fresh_capture(self):
        model = _model()
        backup = WeightBackup(model.parameters())
        backup.capture("ref")
        with backup.swapped("ref"):
            pass
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.5)
        backup.mark_modified()
        expected = [p.clone() for p in model.parameters()]

        with backup.swapped("ref") as stats:
            pass

        assert stats.captured_bytes == backup.nbytes
        for got, want in zip(model.parameters(), expected):
            assert torch.equal(got, want)


class TestInteractionWithStageOffload:
    def test_swapping_while_offloaded_is_refused(self):
        """An offloaded storage has size zero; a swap would copy nothing and report
        success with the wrong weights resident."""
        model = _model()
        backup = WeightBackup(model.parameters())
        backup.capture("ref")
        stage = StorageOffloader.from_tensors(model.parameters())
        stage.offload()

        with pytest.raises(RuntimeError, match="offloaded"), backup.swapped("ref"):
            pass
        stage.restore()

    def test_the_backup_survives_a_stage_round_trip(self):
        """Offload reallocates on restore; the backup tracks storages, not addresses."""
        model = _model()
        backup = WeightBackup(model.parameters())
        backup.capture("ref")
        ref_copy = copy.deepcopy(model)
        stage = StorageOffloader.from_tensors(model.parameters())
        stage.offload()
        stage.restore()
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1)
        backup.mark_modified()
        batch = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)

        with torch.no_grad(), backup.swapped("ref"):
            assert torch.equal(model(batch), ref_copy(batch))


class TestCost:
    def test_a_swap_runs_near_pcie_bandwidth(self):
        """One D2H plus two H2D of the shard; on H20 each direction is ~50 GB/s alone."""
        big = [torch.randn(1 << 28, device="cuda", dtype=torch.bfloat16) for _ in range(4)]  # 2 GiB
        backup = WeightBackup(big)
        backup.capture("ref")
        backup.mark_modified()
        with backup.swapped("ref"):
            pass  # warm the actor buffer
        backup.mark_modified()

        with backup.swapped("ref") as stats:
            pass

        moved = stats.captured_bytes + stats.loaded_bytes
        gbps = moved / stats.seconds / 1e9
        print(f"\nswap moved {moved / 2**30:.1f} GiB in {stats.seconds:.3f}s ({gbps:.1f} GB/s)")
        assert gbps > 25
