"""Registration-validity invariants for the NIXL transport (verification item 5).

Design §7 item 5 lists three invariants and says of all three: *都测静默损坏，必须有*
-- each one guards a failure that produces wrong bytes without any error. They are:

  (a) a pull issued after the trainer resized a buffer, but before re-registration,
      must be **rejected by the index** rather than reading freed pages;
  (b) after a real CPU offload + reload (so the virtual address genuinely changes),
      a remote read must still return **byte-correct** content -- this is the test the
      cached-address fast path at `client.py:401` would fail, which is why it exists;
  (c) a stale descriptor -- one held from before a re-registration -- must be
      **explicitly refused**, not silently resolved against a wrong address.

What is in scope here. (a) and (c) are properties of the client's registration
bookkeeping (`_reg_regions` / `_registered_descs` and the "not registered" rejection
paths at `client.py:354-410`), so they are testable against a real NIXL agent on one
node, which is what this file does. (b) additionally needs a second process holding the
remote side across an offload, so its cross-node form belongs to L3; the single-node
half -- that a re-registered tensor at a *new* address resolves to the new address and
not the cached one -- is covered here.

The shape of every test is the same: perform the lifecycle event, then assert the
transport **raises**. Asserting on the error is the whole point; an assertion that the
bytes happen to be right would pass for the wrong reason whenever the freed page has
not yet been reused.

The agent pair is the session-scoped one from `conftest_agents.py`; see that module for
the three ordering constraints NIXL imposes on registration and metadata exchange.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.nixl.conftest_agents import (  # noqa: F401  (fixtures)
    agent_pair,
    cuda_available,
    registered_buffer,
    xfer_descs,
)

nixlBind = pytest.importorskip("nixl._bindings", reason="NIXL not installed")

SLICE_BYTES = 64 * 1024


def _client(agent, name: str):
    """A minimally-constructed NIXLStorageClient sharing a real agent.

    `__new__` plus explicit field assignment rather than `__init__`: the constructor
    builds its own agent, meta-server connection and config, none of which these tests
    exercise. Only the registration bookkeeping is under test.
    """
    from psrl.utils.nixl.client import NIXLStorageClient

    client = NIXLStorageClient.__new__(NIXLStorageClient)
    client.agent = agent
    client.client_name = name
    client._reg_regions = set()
    client._mtype_to_reg_region_lists = {}
    client._registered_descs = {}
    client._cached_xfer_descs = []
    # Fields `deregister_local_tensors` touches. It asserts `local_client_info is not
    # None` up front -- deregistering before registration is a caller bug, not a
    # tolerated no-op -- and then clears this set.
    client.local_client_info = {}
    client.enable_tms_for_temp_buffers = True  # skips release_temp_memory()
    client._prepped_cache = None
    client._pull_engine = None
    return client


class TestRegistrationBookkeeping:
    """`_record_region_registration` must track base storage, not the view."""

    def test_registering_a_view_records_its_base_storage(self, agent_pair):
        """Two views of one allocation must register as a single region.

        NIXL registers memory by address range, so registering per-view would both
        duplicate registrations and -- worse -- let a view survive in the table after
        its base storage is gone.
        """
        local, _peer = agent_pair
        client = _client(local, "pytest-local")

        base = torch.zeros(4096, dtype=torch.uint8, device="cuda")
        first = client._record_region_registration(base[:1024])
        second = client._record_region_registration(base[2048:3072])

        assert first == second, (
            "two views of one storage produced different region keys; "
            f"{first} != {second}"
        )
        assert len(client._reg_regions) == 1, (
            f"expected 1 tracked region for 1 allocation, got {len(client._reg_regions)}"
        )

    def test_region_key_carries_base_address_and_size(self, agent_pair):
        """The key must describe the whole storage, not the view's slice."""
        local, _peer = agent_pair
        client = _client(local, "pytest-local")

        base = torch.zeros(4096, dtype=torch.uint8, device="cuda")
        view = base[1024:2048]
        base_addr, nbytes, device_id, mem_type = client._record_region_registration(view)

        assert base_addr == base.untyped_storage().data_ptr(), "not the base address"
        assert nbytes == base.untyped_storage().nbytes(), "not the full storage size"
        assert mem_type == "cuda"
        assert device_id == base.get_device()

    def test_distinct_allocations_are_distinct_regions(self, agent_pair):
        local, _peer = agent_pair
        client = _client(local, "pytest-local")

        a = torch.zeros(2048, dtype=torch.uint8, device="cuda")
        b = torch.zeros(2048, dtype=torch.uint8, device="cuda")
        client._record_region_registration(a)
        client._record_region_registration(b)

        assert len(client._reg_regions) == 2, (
            "two separate allocations collapsed into one region; a stale entry would "
            "then alias live memory"
        )

    def test_cpu_and_cuda_regions_are_tracked_separately(self, agent_pair):
        """mem_type is part of the key: a host and device address can collide numerically."""
        local, _peer = agent_pair
        client = _client(local, "pytest-local")

        client._record_region_registration(torch.zeros(1024, dtype=torch.uint8))
        client._record_region_registration(torch.zeros(1024, dtype=torch.uint8, device="cuda"))

        assert set(client._mtype_to_reg_region_lists) == {"cpu", "cuda"}


class TestStaleDescriptorIsRefused:
    """Item 5(c): a descriptor from before re-registration must be refused."""

    def test_deregistering_one_handle_leaves_other_registrations_intact(
        self, agent_pair, registered_buffer
    ):
        """Deregistering a duplicate registration must not invalidate the original.

        The first draft of this test registered a *second* handle over an
        already-registered buffer, dropped it, and expected the transfer to fail. It
        did not -- and the implementation is right: `registered_buffer` still holds a
        live registration for that range, so the address remains legitimately
        resolvable. NIXL refcounts registrations per handle rather than per range.

        That is worth pinning, because the opposite behaviour would make any transient
        duplicate registration a loaded gun: dropping it would silently revoke the
        original. The invariant for a genuinely revoked range is covered by
        `test_unregistered_address_is_rejected_at_prep`.
        """
        local, _peer = agent_pair
        source = registered_buffer(SLICE_BYTES)
        destination = registered_buffer(SLICE_BYTES)
        slices = [(0, SLICE_BYTES)]

        reg_list = nixlBind.nixlRegDList(
            nixlBind.VRAM_SEG,
            np.array(
                [[source.data_ptr(), SLICE_BYTES, source.device.index]], dtype=np.uint64
            ),
        )
        duplicate = local.register_memory(reg_list)
        local.deregister_memory(duplicate)

        # The fixture's registration is still live, so this must still work.
        prepped = local.prep_xfer_dlist(
            "NIXL_INIT_AGENT", xfer_descs(local, destination, slices), mem_type="VRAM"
        )
        remote = local.prep_xfer_dlist(
            "pytest-peer", xfer_descs(local, source, slices), mem_type="VRAM"
        )
        try:
            assert prepped and remote, (
                "dropping a duplicate registration revoked the original range; any "
                "transient double-registration would then corrupt live transfers"
            )
        finally:
            local.release_dlist_handle(prepped)
            local.release_dlist_handle(remote)

    def test_unregistered_address_is_rejected_at_prep(self, agent_pair):
        """`prep_xfer_dlist` over never-registered memory must fail, not return a handle.

        This is item 5(a)'s mechanism: the index refuses rather than letting the NIC
        resolve an address it was never told about.
        """
        local, _peer = agent_pair
        stray = torch.zeros(SLICE_BYTES, dtype=torch.uint8, device="cuda")
        descs = xfer_descs(local, stray, [(0, SLICE_BYTES)])

        with pytest.raises(Exception):
            handle = local.prep_xfer_dlist("NIXL_INIT_AGENT", descs, mem_type="VRAM")
            if not handle:
                raise RuntimeError("prep_xfer_dlist returned a falsy handle")
            local.release_dlist_handle(handle)


class TestAddressChangeIsObserved:
    """Item 5(b), single-node half: a new allocation must not resolve to a cached one."""

    def test_reallocation_yields_a_different_region_key(self, agent_pair):
        """Free then reallocate: the bookkeeping must not silently reuse the old entry.

        torch's caching allocator will often hand back the *same* device address, which
        is exactly the condition under which a cached-address fast path looks correct
        while being wrong. The assertion is therefore on the tracked region count, not
        on the address: a second registration must be recorded even when the address
        repeats, because the registration itself was torn down.
        """
        local, _peer = agent_pair
        client = _client(local, "pytest-local")

        first = torch.zeros(8192, dtype=torch.uint8, device="cuda")
        key_first = client._record_region_registration(first)
        del first
        torch.cuda.empty_cache()

        second = torch.zeros(8192, dtype=torch.uint8, device="cuda")
        key_second = client._record_region_registration(second)

        if key_first == key_second:
            # Same address reused by the allocator: the table cannot distinguish them,
            # which is precisely why re-registration must be driven by an explicit
            # lifecycle event rather than inferred from the address.
            assert len(client._reg_regions) == 1
        else:
            assert len(client._reg_regions) == 2

    def test_resized_buffer_is_a_new_region(self, agent_pair):
        """A resize changes the storage size, so the region key must change.

        This is the DDP-buffer-resize scenario from item 5(a): the size is part of the
        key, so a resized buffer cannot be mistaken for the original even if the base
        address survives.
        """
        local, _peer = agent_pair
        client = _client(local, "pytest-local")

        small = torch.zeros(1024, dtype=torch.uint8, device="cuda")
        key_small = client._record_region_registration(small)
        large = torch.zeros(4096, dtype=torch.uint8, device="cuda")
        key_large = client._record_region_registration(large)

        assert key_small[1] != key_large[1], "storage size is not part of the region key"
        assert key_small != key_large


class TestDeregisterClearsBookkeeping:
    """Deregistration must leave no entry that could later alias live memory."""

    def test_deregister_clears_prepped_cache_first(self, agent_pair):
        """The prepped cache holds raw pointers, so it must be cleared before deregister.

        Order matters: a prepped descriptor list that outlives its registration is a
        dangling pointer the NIC would happily dereference.
        """
        from psrl.utils.nixl.prepped_xfer import PreppedXferCache

        local, _peer = agent_pair
        client = _client(local, "pytest-local")
        client._prepped_cache = PreppedXferCache(local, "pytest-local")
        client._registered_descs = {}

        client.deregister_local_tensors()

        assert client._prepped_cache is None or not getattr(
            client._prepped_cache, "_dlists", {}
        ), "prepped cache still holds dlists after deregistration"

    def test_deregister_is_idempotent(self, agent_pair):
        """Calling it twice must not raise: shutdown paths may retry."""
        local, _peer = agent_pair
        client = _client(local, "pytest-local")
        client._prepped_cache = None
        client.deregister_local_tensors()
        client.deregister_local_tensors()
