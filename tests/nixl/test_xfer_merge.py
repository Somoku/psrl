#!/usr/bin/env python
"""CPU-only tests for the NIXL transfer merge and handle-lifecycle logic.

These tests never construct a real ``nixl_agent``: they build a client instance
with ``__new__`` and drive it with a fake agent, so they run in CI without a GPU
or NIXL plugins.
"""

from collections import OrderedDict

import nixl._bindings as nixlBind
import pytest
import torch
from psrl.utils.nixl.client import (
    NIXLStorageClient,
    make_xfer_tag,
)
from psrl.utils.nixl.nixl_spec import NIXLSharding, NIXLShardMetaInfo, NIXLTensorInfo

pytestmark = pytest.mark.cpu_test

_VRAM = nixlBind.VRAM_SEG


class _FakeDesc:
    """Single-entry transfer descriptor list stand-in."""

    def __init__(self, mem_type=_VRAM, addr: int = 0x1000, length: int = 16, device_id: int = 0):
        self._mem_type = mem_type
        self._entry = (addr, length, device_id)

    def getType(self):
        return self._mem_type

    def descCount(self):
        return 1

    def __getitem__(self, index):
        assert index == 0
        return self._entry


class _FakeAgent:
    def __init__(self, states=None):
        self.released_xfer = []
        self.released_dlists = []
        self.initialized = []
        self.transferred = []
        self._states = list(states or [])

    def deserialize_descs(self, _desc_bytes):
        return _FakeDesc()

    def initialize_xfer(self, op, local_descs, remote_descs, target_agent, tag):
        self.initialized.append((op, local_descs, remote_descs, target_agent, tag))
        return ("xfer-handle", tag)

    def transfer(self, handle):
        self.transferred.append(handle)
        return "DONE"

    def check_xfer_state(self, _handle):
        return self._states.pop(0)

    def release_xfer_handle(self, handle):
        self.released_xfer.append(handle)

    def release_dlist_handle(self, handle):
        self.released_dlists.append(handle)


def _make_client(agent):
    client = NIXLStorageClient.__new__(NIXLStorageClient)
    client.client_name = "client0"
    client.agent = agent
    client.xfer_handles = {}
    client._prepared_dlists = {}
    client.merge_contiguous_xfer = True
    client.enable_prepared_dlist = False
    client.enable_nixl_telemetry = False
    return client


def _make_tensor_info(shard_indices, mem_type=_VRAM):
    meta = NIXLShardMetaInfo(
        dtype=torch.float32,
        device=torch.device("cpu"),
        shape=torch.Size([4]),
        stride=(1,),
        is_contiguous=True,
    )
    return NIXLTensorInfo(
        desc_bytes_list=[repr(shard_idx).encode() for shard_idx in shard_indices],
        temp_desc_bytes_list=[None] * len(shard_indices),
        sharding=NIXLSharding(shard_mesh=OrderedDict([(0, len(shard_indices))]), shard_indices=list(shard_indices)),
        shard_meta_infos=[meta] * len(shard_indices),
    )


def test_make_xfer_tag_group_and_shard_are_distinct():
    group = make_xfer_tag("tag", "client0", "target1", "key")
    shard0 = make_xfer_tag("tag", "client0", "target1", "key", (0,))
    shard1 = make_xfer_tag("tag", "client0", "target1", "key", (1,))
    assert len({group, shard0, shard1}) == 3


def test_post_contiguous_group_posts_one_request_for_all_shards():
    agent = _FakeAgent()
    client = _make_client(agent)
    local_info = _make_tensor_info([(0,), (1,)])
    remote_info = _make_tensor_info([(0,), (1,)])

    handled = client._post_contiguous_group(
        "WRITE", "agentA", "target1", "key", "tag", [(0,), (1,)], local_info, remote_info
    )

    assert handled == {(0,), (1,)}
    assert len(agent.initialized) == 1
    op, local_descs, remote_descs, target_agent, tag = agent.initialized[0]
    assert op == "WRITE"
    assert local_descs.descCount() == 2
    assert remote_descs.descCount() == 2
    assert target_agent == "agentA"
    # The merged request is tracked under the no-shard (group) tag.
    assert make_xfer_tag("tag", "client0", "target1", "key") in client.xfer_handles
    assert make_xfer_tag("tag", "client0", "target1", "key", (0,)) not in client.xfer_handles
    assert len(agent.transferred) == 1


def test_post_contiguous_group_skips_non_contiguous_shards():
    agent = _FakeAgent()
    client = _make_client(agent)
    local_info = _make_tensor_info([(0,), (1,)])
    local_info.desc_bytes_list[0] = None  # non-contiguous shard
    remote_info = _make_tensor_info([(0,), (1,)])

    handled = client._post_contiguous_group(
        "READ", "agentA", "target1", "key", "tag", [(0,), (1,)], local_info, remote_info
    )

    assert handled == {(1,)}
    assert agent.initialized[0][1].descCount() == 1


def test_release_all_xfer_handles_releases_and_clears():
    agent = _FakeAgent()
    client = _make_client(agent)
    client.xfer_handles = {b"a": "handle-a", b"b": "handle-b"}

    client._release_all_xfer_handles("test")

    assert sorted(agent.released_xfer) == ["handle-a", "handle-b"]
    assert client.xfer_handles == {}


def test_release_prepared_dlists_filters_by_side_and_peer():
    agent = _FakeAgent()
    client = _make_client(agent)
    client._prepared_dlists = {
        ("local", "", "key", _VRAM): ("local-h", [(0,)]),
        ("remote", "peer-a", "key", _VRAM): ("remote-a", [(0,)]),
        ("remote", "peer-b", "key", _VRAM): ("remote-b", [(0,)]),
    }

    client._release_prepared_dlists(side="remote", target_client="peer-a")

    assert agent.released_dlists == ["remote-a"]
    assert ("remote", "peer-a", "key", _VRAM) not in client._prepared_dlists
    assert ("local", "", "key", _VRAM) in client._prepared_dlists
    assert ("remote", "peer-b", "key", _VRAM) in client._prepared_dlists


def test_await_and_release_releases_on_success():
    agent = _FakeAgent(states=["DONE"])
    client = _make_client(agent)
    handle_key = make_xfer_tag("tag", "client0", "target1", "key")
    client.xfer_handles[handle_key] = "handle"

    client._await_and_release(handle_key, "key", "tag", "WRITE", "target1", None, None, timeout=1.0)

    assert agent.released_xfer == ["handle"]
    assert client.xfer_handles == {}


def test_await_and_release_releases_even_on_error():
    agent = _FakeAgent(states=["ERR"])
    client = _make_client(agent)
    handle_key = make_xfer_tag("tag", "client0", "target1", "key")
    client.xfer_handles[handle_key] = "handle"

    with pytest.raises(RuntimeError):
        client._await_and_release(handle_key, "key", "tag", "WRITE", "target1", None, None, timeout=1.0)

    assert agent.released_xfer == ["handle"]
    assert client.xfer_handles == {}
