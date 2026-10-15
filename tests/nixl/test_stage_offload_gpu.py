"""Stage-boundary offload: weights leave the GPU and come back intact (design §3.7.3).

Three claims, each of which the old sleep path violated or never checked:

* **Every view survives.** Megatron's parameters are views into DDP buffers and the
  weight transport holds views into those. After a round trip they must read the
  restored bytes without anyone rebuilding them.
* **The bytes are bit-exact.** The old path restored the allocation only; under direct
  supply nothing refilled it, so the next step trained on garbage.
* **The memory actually returns to the device.** Releasing into PyTorch's caching
  allocator frees nothing for vLLM, which allocates outside it.

Plus the cost, measured against PCIe: the design budgets ~0.3 s each way for 18 GB.
"""

from __future__ import annotations

import pytest
import torch

from psrl.utils.nixl.stage_offload import StorageOffloader

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

GiB = 1 << 30


def _ddp_like(nbytes: int, dtype=torch.bfloat16):
    """A flat buffer with parameter views carved out of it, as Megatron's DDP lays out."""
    numel = nbytes // torch.tensor([], dtype=dtype).element_size()
    buffer = torch.randn(numel, device="cuda", dtype=torch.float32).to(dtype)
    quarter = numel // 4
    params = [buffer[i * quarter : (i + 1) * quarter].view(-1, 64) for i in range(4)]
    return buffer, params


class TestViewsSurviveTheRoundTrip:
    def test_parameter_views_read_the_restored_bytes(self):
        buffer, params = _ddp_like(64 << 20)
        expected = [p.clone() for p in params]
        offloader = StorageOffloader.from_tensors([buffer])

        offloader.offload()
        assert buffer.untyped_storage().nbytes() == 0
        offloader.restore()

        for got, want in zip(params, expected):
            assert torch.equal(got, want)

    def test_a_derived_view_held_elsewhere_survives(self):
        """The weight transport's `unified_state_dict` is a view of a view. Moving the
        module instead would leave it pinned to the old GPU storage."""
        buffer, params = _ddp_like(64 << 20)
        derived = params[2].t()[3:17]
        expected = derived.clone()
        offloader = StorageOffloader.from_tensors(params)

        offloader.offload()
        offloader.restore()

        assert torch.equal(derived, expected)

    def test_fp8_bytes_are_bit_exact(self):
        """FP8 has NaN encodings that a float comparison would mishandle; compare bits."""
        raw = torch.randint(0, 256, (32 << 20,), dtype=torch.uint8, device="cuda")
        as_fp8 = raw.view(torch.float8_e4m3fn)
        expected = raw.clone()
        offloader = StorageOffloader.from_tensors([as_fp8])

        offloader.offload()
        offloader.restore()

        assert torch.equal(raw, expected)

    def test_shared_storage_is_copied_once(self):
        buffer, params = _ddp_like(16 << 20)

        offloader = StorageOffloader.from_tensors([buffer, *params])

        assert offloader.nbytes == buffer.untyped_storage().nbytes()


class TestMemoryIsReturned:
    def test_the_driver_sees_the_bytes_come_back(self):
        """`mem_get_info` is what vLLM sizes its KV cache from; the caching allocator's
        own counters would show the release even when the driver does not."""
        torch.cuda.empty_cache()
        buffer, _ = _ddp_like(2 * GiB)
        offloader = StorageOffloader.from_tensors([buffer])

        stats = offloader.offload()

        assert stats.freed >= 0.95 * 2 * GiB, f"only {stats.freed / GiB:.2f} GiB returned"
        offloader.restore()

    def test_discard_mode_allocates_no_host_memory(self):
        """With a parameter server the bytes come back by pull; copying them to host
        would be wasted PCIe time."""
        buffer, _ = _ddp_like(64 << 20)
        offloader = StorageOffloader.from_tensors([buffer], preserve=False)

        offloader.offload()
        offloader.restore()

        assert all(entry.host is None for entry in offloader._entries)
        assert buffer.untyped_storage().nbytes() == 64 << 20


class TestPinnedBuffersAreReused:
    def test_the_second_switch_reuses_the_first_switchs_host_buffer(self):
        """Allocating per switch costs seconds of `cudaHostAlloc` and briefly doubles
        host usage -- the cgroup OOM verl's own offload hit."""
        buffer, _ = _ddp_like(64 << 20)
        offloader = StorageOffloader.from_tensors([buffer])

        offloader.offload()
        first = offloader._entries[0].host.data_ptr()
        offloader.restore()
        offloader.offload()
        second = offloader._entries[0].host.data_ptr()
        offloader.restore()

        assert first == second

    def test_weights_change_between_switches_and_both_survive(self):
        """Training writes in between; the reused buffer must hold the *new* bytes."""
        buffer, _ = _ddp_like(16 << 20)
        offloader = StorageOffloader.from_tensors([buffer])
        offloader.offload()
        offloader.restore()

        buffer.mul_(0.5).add_(1)
        expected = buffer.clone()
        offloader.offload()
        offloader.restore()

        assert torch.equal(buffer, expected)


class TestHostBuffer:
    def test_host_usage_is_the_requested_size_not_a_power_of_two(self):
        """`pin_memory=True` rounds 9 GiB up to 16 GiB (measured). Per rank, eight ranks
        a node, that rounding alone can exceed a pod's host memory."""
        from psrl.utils.nixl.stage_offload import pinned_host_buffer

        def rss_mib():
            with open("/proc/self/status") as f:
                return int(f.read().split("VmRSS:")[1].split()[0]) // 1024

        before = rss_mib()
        buf = pinned_host_buffer((3 * GiB) + 12345)
        grew = rss_mib() - before

        assert buf.is_pinned()
        assert buf.numel() == 3 * GiB + 12345
        assert grew < 3.2 * 1024, f"host usage grew {grew} MiB for a 3 GiB request"

    def test_pytorch_sees_it_as_pinned(self):
        """Registering an interior view left `is_pinned()` False (measured), because
        PyTorch asks about the storage base -- and then treats copies as pageable."""
        from psrl.utils.nixl.stage_offload import pinned_host_buffer

        assert pinned_host_buffer(1 << 20).is_pinned()


class TestOptimizerState:
    def _trained_adam(self):
        model = torch.nn.Linear(256, 256, device="cuda")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model(torch.randn(8, 256, device="cuda")).sum().backward()
        optimizer.step()
        return model, optimizer

    def test_moments_and_step_survive_and_training_continues_identically(self):
        """The check that matters is behavioural: a step after the round trip must equal
        a step taken without one. Comparing tensors alone would miss a moment left out."""
        from psrl.utils.nixl.stage_offload import optimizer_tensors

        torch.manual_seed(0)
        model_a, opt_a = self._trained_adam()
        torch.manual_seed(0)
        model_b, opt_b = self._trained_adam()

        offloader = StorageOffloader.from_tensors(optimizer_tensors(opt_b))
        offloader.offload()
        offloader.restore()

        batch = torch.randn(8, 256, device="cuda")
        for model, opt in ((model_a, opt_a), (model_b, opt_b)):
            opt.zero_grad()
            model(batch).sum().backward()
            opt.step()
        assert torch.equal(model_a.weight, model_b.weight)

    def test_a_storage_two_offloaders_could_claim_goes_to_one(self):
        """FSDP's optimizer param groups *are* the model parameters. Two offloaders on
        one storage would have the second save zero bytes over the first's."""
        from psrl.utils.nixl.stage_offload import optimizer_tensors, storage_key

        model, optimizer = self._trained_adam()
        weights = StorageOffloader.from_tensors(model.parameters())
        opt = StorageOffloader.from_tensors(optimizer_tensors(optimizer), exclude=weights.keys)

        assert not (weights.keys & opt.keys)
        assert storage_key(model.weight) in weights.keys

    def test_state_created_after_construction_is_picked_up_by_retarget(self):
        """Adam creates its moments on the first step, so an offloader built at boot sees
        none; retargeting must find them and keep existing pinned buffers."""
        from psrl.utils.nixl.stage_offload import optimizer_tensors

        model = torch.nn.Linear(64, 64, device="cuda")
        optimizer = torch.optim.Adam(model.parameters())
        offloader = StorageOffloader.from_tensors(optimizer_tensors(optimizer))
        offloader.offload()
        offloader.restore()
        kept = {e.storage._cdata: e.host.data_ptr() for e in offloader._entries}

        model(torch.randn(2, 64, device="cuda")).sum().backward()
        optimizer.step()
        offloader.retarget(optimizer_tensors(optimizer))

        assert len(offloader._entries) > len(kept)
        for entry in offloader._entries:
            if entry.storage._cdata in kept:
                assert entry.host.data_ptr() == kept[entry.storage._cdata]


class TestGradients:
    def test_discarded_gradients_come_back_zeroed(self):
        grad = torch.randn(1 << 20, device="cuda")
        offloader = StorageOffloader.from_tensors([grad], preserve=False, zero_on_restore=True)

        offloader.offload()
        offloader.restore()

        assert int(torch.count_nonzero(grad)) == 0


class TestMisuseIsRefused:
    def test_a_second_offload_without_restore_raises(self):
        """It would copy the zero-size storages over the saved bytes."""
        buffer, _ = _ddp_like(1 << 20)
        offloader = StorageOffloader.from_tensors([buffer])
        offloader.offload()

        with pytest.raises(RuntimeError, match="twice"):
            offloader.offload()
        offloader.restore()

    def test_releasing_host_buffers_while_offloaded_raises(self):
        buffer, _ = _ddp_like(1 << 20)
        offloader = StorageOffloader.from_tensors([buffer])
        offloader.offload()

        with pytest.raises(RuntimeError, match="lose the weights"):
            offloader.release_host()
        offloader.restore()


class TestCostAgainstPcie:
    """The design budgets ~0.3 s each way for 18 GB, i.e. ~60 GB/s, the PCIe 5 x16
    ceiling. A single process on H20 measures this; the eight-process concurrent figure,
    where host memory bandwidth and NUMA placement start to matter, is in
    `bench_stage_offload.py`."""

    def test_offload_and_restore_run_near_pcie_bandwidth(self, record_property):
        buffer, _ = _ddp_like(8 * GiB)
        offloader = StorageOffloader.from_tensors([buffer])
        # First switch pays for pinning; measure the steady state, as production sees it.
        offloader.offload()
        offloader.restore()

        down = offloader.offload()
        up = offloader.restore()

        record_property("d2h_gbps", round(down.gbps, 1))
        record_property("h2d_gbps", round(up.gbps, 1))
        print(f"\nD2H {down.gbps:.1f} GB/s, H2D {up.gbps:.1f} GB/s over {down.bytes / GiB:.0f} GiB")
        # A floor, not the target: well below PCIe 5 means a non-pinned or synchronous
        # copy slipped in, which is the regression this guards.
        assert down.gbps > 20 and up.gbps > 20
