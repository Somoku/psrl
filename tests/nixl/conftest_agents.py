"""Shared NIXL agent fixtures for single-node multi-GPU tests.

Agent construction costs ~60s on a shared container host: `uct_tcp_query_devices`
calls `getifaddrs()` once per network interface, and this host has ~460 of them (see
`bench_nixl_prepped_xfer.py` for the full diagnosis). That makes per-test agents
untenable -- a four-dimensional parameter matrix would spend hours in setup.

So agents are session-scoped and shared. Tests must therefore not leave an agent in a
modified state: register memory through `registered_buffer`, which deregisters on exit,
and never call `remove_remote_agent` on the shared pair.
"""

from __future__ import annotations

import contextlib
import os
import time

import numpy as np
import pytest
import torch

nixlBind = pytest.importorskip("nixl._bindings", reason="NIXL not installed")
_api = pytest.importorskip("nixl._api", reason="NIXL not installed")

AGENT_SETUP_BUDGET_S = 180.0


@pytest.fixture(scope="session")
def cuda_available() -> bool:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    return True


@pytest.fixture(scope="session")
def agent_pair(cuda_available):
    """Two in-process NIXL agents that can address each other.

    NIXL rejects an agent as its own remote ("remote agent name same as local"), so a
    loopback transfer needs a genuine second agent. Both run in this process, which is
    enough for descriptor-level and single-node correctness; bandwidth and topology
    belong to the multi-node suite.

    **Order matters and is not obvious.** A VRAM region is registered on both agents
    *before* `add_remote_agent`, because the metadata blob records which backends serve
    which memory types. Exchanged too early, it advertises no VRAM backend, and a later
    remote `prep_xfer_dlist` fails with ``no available backends for mem type
    'VRAM_SEG'`` -- while the *local* prep still succeeds, so the fault looks like a
    descriptor problem rather than a handshake-ordering one.

    The keep-alive buffer below exists only to make that registration happen; tests get
    their own buffers from `registered_buffer`.

    **The progress thread must stay enabled.** With `enable_prog_thread=False` a
    transfer posts successfully and then never leaves ``PROC``: nothing advances the
    UCX worker, so `check_xfer_state` polls forever. The benchmark disables it safely
    only because it measures submission cost and never waits for completion.
    """
    config = _api.nixl_agent_config(enable_prog_thread=True)

    start = time.perf_counter()
    local = _api.nixl_agent("pytest-local", config)
    peer = _api.nixl_agent("pytest-peer", config)
    elapsed = time.perf_counter() - start

    if elapsed > AGENT_SETUP_BUDGET_S:
        pytest.skip(
            f"NIXL agent setup took {elapsed:.0f}s (budget {AGENT_SETUP_BUDGET_S:.0f}s); "
            f"interface count is {len(os.listdir('/sys/class/net'))}"
        )

    keepalive = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
    descs = np.zeros((1, 3), dtype=np.uint64)
    descs[0] = (keepalive.data_ptr(), keepalive.numel(), keepalive.device.index)
    reg_list = nixlBind.nixlRegDList(nixlBind.VRAM_SEG, descs)
    keepalive_handles = [agent.register_memory(reg_list) for agent in (local, peer)]

    # Only now is the metadata complete enough to describe a VRAM backend.
    local.add_remote_agent(peer.get_agent_metadata())
    peer.add_remote_agent(local.get_agent_metadata())

    yield local, peer

    # Only tear down what this fixture created; tests must not have removed peers.
    for agent, remote in ((local, "pytest-peer"), (peer, "pytest-local")):
        with contextlib.suppress(Exception):
            agent.remove_remote_agent(remote)
    for agent, handle in zip((local, peer), keepalive_handles):
        with contextlib.suppress(Exception):
            agent.deregister_memory(handle)


@pytest.fixture
def registered_buffer(agent_pair):
    """Allocate a GPU buffer, register it on both agents, and refresh peer metadata.

    Returns a factory so a test can take several buffers. Registration is per
    allocation (see `_record_region_registration` in the client), so one call here
    corresponds to one `register_memory`.

    **The metadata refresh is mandatory, not hygiene.** A region registered after
    `add_remote_agent` is invisible to the remote side: `prep_xfer_dlist` for that peer
    fails with ``failed to prepare the descriptors for any of the specified or potential
    backends``, because the cached metadata blob predates the registration. Re-calling
    `add_remote_agent` with fresh metadata fixes it.

    This is the mechanism behind design item 9c-c: incremental join cannot simply
    register and go, it has to re-publish metadata, and `getLocalPartialMD` exists so
    that re-publish costs ~400 bytes rather than the full blob.
    """
    local, peer = agent_pair
    created: list[tuple] = []

    def make(nbytes: int, device: int = 0):
        buffer = torch.zeros(nbytes, dtype=torch.uint8, device=f"cuda:{device}")
        descs = np.zeros((1, 3), dtype=np.uint64)
        descs[0] = (buffer.data_ptr(), nbytes, device)
        reg_list = nixlBind.nixlRegDList(nixlBind.VRAM_SEG, descs)
        handles = [agent.register_memory(reg_list) for agent in (local, peer)]
        created.append((handles, buffer))
        # Re-publish so each side can address the other's new region.
        local.add_remote_agent(peer.get_agent_metadata())
        peer.add_remote_agent(local.get_agent_metadata())
        return buffer

    yield make

    for handles, _buffer in created:
        for agent, handle in zip((local, peer), handles):
            with contextlib.suppress(Exception):
                agent.deregister_memory(handle)


def xfer_descs(agent, buffer: torch.Tensor, slices: list[tuple[int, int]]):
    """Build an Nx3 transfer descriptor list over ``(offset, length)`` pairs."""
    base = buffer.data_ptr()
    device = buffer.device.index
    array = np.zeros((len(slices), 3), dtype=np.uint64)
    for i, (offset, length) in enumerate(slices):
        array[i] = (base + offset, length, device)
    return agent.get_xfer_descs(array, mem_type="VRAM")
