"""Revocation makes a descriptor unusable rather than merely unrouted (item E4).

This is the claim the whole shrink path rests on, and it is the one thing the CPU suite
cannot check: `Membership` can refuse to release memory until every holder has acked,
but whether the ack *means* anything is a property of NIXL.

The design (§3.4.5) assumed it had to be arranged -- "撤销 holder 资格" stops new reads
being *routed* to a departing member, and the fear was that a retry or a bug could still
post an old descriptor and read freed pages successfully, the same silent corruption
`registration_epoch` exists for. Measured here, that fear is unfounded *provided the
eviction runs*: NIXL holds the remote section weakly and checks
`remoteSection.expired()` in `makeXferReq`, `postXferReq` and `getXferStatus`, so every
route to a revoked peer fails loudly with NIXL_ERR_NOT_FOUND.

Which is why PSRL needs no epoch of its own on this path -- only the guarantee that
every holder ran the eviction before the memory goes, which is what the ack provides.

Dedicated agents rather than the session-scoped `agent_pair`: this test's whole subject
is `remove_remote_agent`, and `conftest_agents` states that the shared pair must not be
torn down by a test. Agent construction dominates the runtime (see that file for the
`getifaddrs` diagnosis).
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

nixlBind = pytest.importorskip("nixl._bindings", reason="NIXL not installed")
_api = pytest.importorskip("nixl._api", reason="NIXL not installed")

from psrl.utils.nixl.leave_protocol import revoke_descriptors  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

NBYTES = 1 << 20


def _reg_list(tensor: torch.Tensor):
    descs = np.zeros((1, 3), dtype=np.uint64)
    descs[0] = (tensor.data_ptr(), tensor.numel(), tensor.device.index)
    return nixlBind.nixlRegDList(nixlBind.VRAM_SEG, descs)


def _xfer_descs(agent, tensor: torch.Tensor):
    descs = np.zeros((1, 3), dtype=np.uint64)
    descs[0] = (tensor.data_ptr(), tensor.numel(), tensor.device.index)
    return agent.get_xfer_descs(descs, mem_type="VRAM")


SOURCE_NAME = "pytest-revoke-source"


class _Pair:
    """Two agents and a `link()` that hands back a freshly connected pair of dlists.

    Each test revokes, so none may inherit the previous one's link. Rebuilding is not
    merely hygiene: a prepped dlist is bound to the remote *registration generation*, so
    a dlist carried across an evict/reload would be stale in a way that looks exactly
    like the fault under test and would make these assertions pass for the wrong reason.
    """

    def __init__(self, source, reader, src, dst):
        self.source, self.reader = source, reader
        self.src, self.dst = src, dst

    def link(self):
        """Ensure the reader knows the source, and return (local, remote) dlists.

        Evict first: `loadRemoteMD` answers NIXL_ERR_NOT_ALLOWED for a peer that is
        already known, so an unconditional add would fail whenever the previous test
        left the link up.
        """
        try:
            self.reader.remove_remote_agent(SOURCE_NAME)
        except nixlBind.nixlNotFoundError:
            pass
        self.reader.add_remote_agent(self.source.get_agent_metadata())
        return (
            self.reader.prep_xfer_dlist("NIXL_INIT_AGENT", _xfer_descs(self.reader, self.dst), mem_type="VRAM"),
            self.reader.prep_xfer_dlist(SOURCE_NAME, _xfer_descs(self.reader, self.src), mem_type="VRAM"),
        )


@pytest.fixture(scope="module")
def pair():
    """Dedicated agents. Construction dominates runtime, so they are built once."""
    config = _api.nixl_agent_config(enable_prog_thread=True)
    source = _api.nixl_agent(SOURCE_NAME, config)
    reader = _api.nixl_agent("pytest-revoke-reader", config)

    src = torch.arange(NBYTES, dtype=torch.uint8, device="cuda")
    dst = torch.zeros(NBYTES, dtype=torch.uint8, device="cuda")
    handles = [source.register_memory(_reg_list(src)), reader.register_memory(_reg_list(dst))]
    source.add_remote_agent(reader.get_agent_metadata())

    yield _Pair(source, reader, src, dst)

    for agent, handle in ((source, handles[0]), (reader, handles[1])):
        try:
            agent.deregister_memory(handle)
        except Exception:
            pass


def _read(reader, local, remote, tag: bytes, timeout: float = 30.0) -> str:
    """Post one READ and return its final state."""
    request = reader.make_prepped_xfer("READ", local, [0], remote, [0], tag)
    state = reader.transfer(request)
    start = time.time()
    while state not in ("DONE", "ERR"):
        state = reader.check_xfer_state(request)
        if time.time() - start > timeout:
            raise TimeoutError(f"read stuck in {state}")
    return state


class TestRevocationInvalidatesDescriptors:
    def test_the_read_works_before_revocation(self, pair):
        """Establishes that what the later tests observe is the revocation, not a
        broken setup -- a transfer that never worked would "fail" afterwards too."""
        local, remote = pair.link()
        pair.dst.zero_()

        assert _read(pair.reader, local, remote, b"baseline") == "DONE"
        torch.cuda.synchronize()
        assert torch.equal(pair.src, pair.dst)

    def test_a_request_built_after_revocation_is_refused(self, pair):
        """The prepped dlists survive the eviction as Python objects; what does not
        survive is the remote registration they point into."""
        local, remote = pair.link()

        revoke_descriptors(pair.reader, SOURCE_NAME)

        with pytest.raises(nixlBind.nixlNotFoundError):
            pair.reader.make_prepped_xfer("READ", local, [0], remote, [0], b"after-revoke")

    def test_a_request_built_before_revocation_cannot_be_reposted(self, pair):
        """The dangerous one. This handle was valid when created, so nothing at the
        call site distinguishes it from a live request -- exactly the retry path §3.4.5
        worried about. NIXL refuses the post instead of reading freed pages."""
        local, remote = pair.link()
        request = pair.reader.make_prepped_xfer("READ", local, [0], remote, [0], b"pre-revoke")
        revoke_descriptors(pair.reader, SOURCE_NAME)
        pair.dst.zero_()

        with pytest.raises(nixlBind.nixlNotFoundError):
            pair.reader.transfer(request)

        torch.cuda.synchronize()
        assert int(pair.dst.max()) == 0, "a revoked handle moved bytes"

    def test_revoking_twice_and_revoking_a_stranger_are_both_benign(self, pair):
        """Idempotence matters because a lost ack makes the server re-send, and the
        second eviction must not turn a completed leave into a failure.

        Measured here and the reason `revoke_descriptors` returns nothing: this build
        reports success for an already-evicted peer *and* for a name that was never
        loaded at all, so "did I hold it" has no answer at this layer.
        """
        pair.link()

        revoke_descriptors(pair.reader, SOURCE_NAME)
        revoke_descriptors(pair.reader, SOURCE_NAME)
        revoke_descriptors(pair.reader, "pytest-revoke-never-existed")

    def test_the_peer_can_be_reloaded_and_works_again(self, pair):
        """Revocation must be a lifecycle step, not a one-way door: a rejoin after a
        transient failure is routine, and it arrives as fresh metadata for a name the
        reader has evicted."""
        pair.link()
        revoke_descriptors(pair.reader, SOURCE_NAME)

        local, remote = pair.link()
        pair.dst.zero_()

        assert _read(pair.reader, local, remote, b"rejoined") == "DONE"
        torch.cuda.synchronize()
        assert torch.equal(pair.src, pair.dst)
