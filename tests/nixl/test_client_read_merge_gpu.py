"""The client READ path merges a key's contiguous shards and releases its handles.

`test_xfer_merge.py` pins the merge and lifetime rules against a fake agent. This file
closes the gap to the real transport: it drives `NIXLStorageClient.client_read` with a
real agent over GPU memory, checks the destination byte-for-byte, and counts the
transfers the merge actually posted.

Two properties are load-bearing for the PS data plane:

* one NIXL request per (operation, target, key) instead of one per shard, which is what
  keeps descriptor submission O(#keys) as shard counts grow;
* every transfer and prepared descriptor list is released, so a long-running client
  does not accumulate NIXL handles across steps.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.nixl.conftest_agents import (  # noqa: F401  (fixtures)
    agent_pair,
    cuda_available,
    registered_buffer,
)

nixlBind = pytest.importorskip("nixl._bindings", reason="NIXL not installed")

SLICE_BYTES = 64 * 1024
TOTAL_BYTES = 4 * SLICE_BYTES


def _serialized_desc(agent, address: int, nbytes: int, device: int) -> bytes:
    descs = np.zeros((1, 3), dtype=np.uint64)
    descs[0] = (address, nbytes, device)
    xfer = agent.get_xfer_descs(descs, mem_type="VRAM")
    return agent.get_serialized_descs(xfer)


def _sharding(shard_count: int):
    from collections import OrderedDict

    from psrl.utils.nixl.nixl_spec import NIXLSharding

    return NIXLSharding(
        shard_mesh=OrderedDict([(0, shard_count)]),
        shard_indices=[(i,) for i in range(shard_count)],
    )


def _tensor_info(agent, address: int, nbytes: int, device: int, shard_count: int):
    from psrl.utils.nixl.nixl_spec import NIXLShardMetaInfo, NIXLTensorInfo

    shard_bytes = nbytes // shard_count
    sharding = _sharding(shard_count)
    meta = NIXLShardMetaInfo(
        dtype=torch.uint8,
        device=torch.device("cuda", device),
        shape=torch.Size((shard_bytes,)),
        stride=(1,),
        is_contiguous=True,
        version=1,
    )
    return NIXLTensorInfo(
        desc_bytes_list=[
            _serialized_desc(agent, address + i * shard_bytes, shard_bytes, device) for i in range(shard_count)
        ],
        temp_desc_bytes_list=[None] * shard_count,
        sharding=sharding,
        shard_meta_infos=[meta] * shard_count,
    )


def _client(local_agent, peer_agent, local_buffer, remote_buffer, shard_count=1, tag_client="pytest-peer-client"):
    """A client wired for one READ from `remote_buffer` into `local_buffer`."""
    from psrl.utils.nixl.client import NIXLStorageClient
    from psrl.utils.nixl.nixl_spec import NIXLClientInfo, NIXLClientType

    device = local_buffer.device.index or 0
    local_info = _tensor_info(local_agent, local_buffer.data_ptr(), local_buffer.numel(), device, shard_count)
    remote_info = _tensor_info(
        peer_agent, remote_buffer.data_ptr(), remote_buffer.numel(), remote_buffer.device.index or 0, shard_count
    )

    def info(name, agent_name, tensor_info):
        return NIXLClientInfo(
            name=name,
            node_ip="127.0.0.1",
            node_gpu_id=device,
            type=NIXLClientType.PUSH_SIDE,
            tensor_infos={"w": tensor_info},
            meta=b"",
            is_registered=True,
            agent_name=agent_name,
        )

    client = NIXLStorageClient.__new__(NIXLStorageClient)
    client.agent = local_agent
    client.client_name = "pytest-local-client"
    client.device = torch.device("cuda", device)
    client.merge_contiguous_xfer = True
    client.enable_prepared_dlist = True
    client.capture_telemetry = False
    client.enable_nixl_telemetry = False
    client.local_client_info = info("pytest-local-client", "pytest-local", local_info)
    client._all_client_infos = {tag_client: info(tag_client, "pytest-peer", remote_info)}
    client._target_client_connected = {tag_client: True}
    client._full_meta_loaded = set()
    client._comm_plan = None
    client._read_phase_us = {
        "deserialize": 0.0,
        "align": 0.0,
        "initialize": 0.0,
        "transfer": 0.0,
    }
    client._pinned_slot_running_read_xfer = {}
    client.xfer_handles = {}
    client._prepared_dlists = {}
    return client


def _fill(buffer: torch.Tensor) -> None:
    indices = torch.arange(buffer.numel(), dtype=torch.int64, device=buffer.device)
    buffer.copy_((indices % 251).to(torch.uint8))
    torch.cuda.synchronize(buffer.device)


class TestClientReadMergesShards:
    def test_two_contiguous_shards_are_one_prepared_transfer(self, agent_pair, registered_buffer):
        local_agent, peer_agent = agent_pair
        source = registered_buffer(TOTAL_BYTES)
        destination = registered_buffer(TOTAL_BYTES)
        _fill(source)
        destination.zero_()
        torch.cuda.synchronize(destination.device)

        client = _client(local_agent, peer_agent, destination, source, shard_count=2)
        calls = []
        original = type(local_agent).make_prepped_xfer

        def tracked(self, *args, **kwargs):
            calls.append(args[0])
            return original(self, *args, **kwargs)

        type(local_agent).make_prepped_xfer = tracked
        try:
            shards = client.client_read("pytest-peer", "pytest-peer-client", "w", "merge")
            client.wait("w", "merge", "READ", target_client="pytest-peer-client")
        finally:
            type(local_agent).make_prepped_xfer = original

        torch.cuda.synchronize(destination.device)
        assert shards == [(0,), (1,)]
        assert torch.equal(destination, source), "merged read delivered wrong bytes"
        assert calls == ["READ"], f"expected one merged request, posted {len(calls)}"

    def test_prepared_dlists_are_reused_and_released(self, agent_pair, registered_buffer):
        local_agent, peer_agent = agent_pair
        source = registered_buffer(TOTAL_BYTES)
        destination = registered_buffer(TOTAL_BYTES)
        _fill(source)

        client = _client(local_agent, peer_agent, destination, source, shard_count=2)
        for round_index in range(2):
            client.client_read("pytest-peer", "pytest-peer-client", "w", f"round{round_index}")
            client.wait("w", f"round{round_index}", "READ", target_client="pytest-peer-client")
        assert len(client._prepared_dlists) == 2, "local and remote dlists should be cached, not rebuilt"

        client._release_prepared_dlists()
        assert client._prepared_dlists == {}
        assert client.xfer_handles == {}, "completed transfers must release their handles"

        # A released dlist is rebuilt and still correct.
        client.client_read("pytest-peer", "pytest-peer-client", "w", "after-release")
        client.wait("w", "after-release", "READ", target_client="pytest-peer-client")
        torch.cuda.synchronize(destination.device)
        assert torch.equal(destination, source)
