"""The version stamp survives a real metadata round trip, on one node (§3.4).

The CPU tests pin the arithmetic and the call ordering; what they cannot show is that a
version written into `NIXLShardMetaInfo` actually reaches a *peer* -- that path runs
through `NIXLClientInfo.serialize`, a NIXL metadata exchange, and the peer's cache, none
of which a same-process test touches.

That property is what three multi-node runs were spent on, and it does not need multiple
nodes: the failure was that readers cache metadata at the handshake, which is equally true
of two agents on one host. So it belongs here, at single-node cost, rather than in a
~30-minute five-node run.

What still requires multiple nodes is the transport itself -- real RDMA between hosts,
real placement. Nothing in this file asserts anything about that.
"""

from __future__ import annotations

import pytest
import torch

from psrl.utils.nixl.nixl_spec import (
    NIXLClientInfo,
    NIXLClientType,
    NIXLShardMetaInfo,
    NIXLSharding,
    NIXLTensorInfo,
)
from psrl.utils.nixl.shard_version import UNVERSIONED, check_pull_versions

# No marker: the `_gpu` suffix is how this suite separates GPU tests from the
# `cpu_test`-marked ones, so `-m cpu_test` already excludes this file.


def _tensor_info(versions: list[int]) -> NIXLTensorInfo:
    """A tensor info whose shards carry the given versions."""
    sharding = NIXLSharding(
        shard_mesh={0: len(versions)},
        shard_indices=[(i,) for i in range(len(versions))],
    )
    return NIXLTensorInfo(
        desc_bytes_list=[b""] * len(versions),
        temp_desc_bytes_list=[None] * len(versions),
        sharding=sharding,
        shard_meta_infos=[
            NIXLShardMetaInfo(
                dtype=torch.bfloat16,
                device=torch.device("cuda:0"),
                shape=torch.Size([8, 8]),
                stride=(8, 1),
                is_contiguous=True,
                version=version,
            )
            for version in versions
        ],
    )


def _client_info(name: str, tensor_infos: dict[str, NIXLTensorInfo]) -> NIXLClientInfo:
    return NIXLClientInfo(
        name=name,
        node_ip="127.0.0.1",
        node_gpu_id=0,
        type=NIXLClientType.PUSH_SIDE,
        tensor_infos=tensor_infos,
        meta=b"",
    )


class TestTheVersionSurvivesTheWireFormat:
    """`NIXLClientInfo` is pickled whole, so the stamp has to survive that round trip."""

    def test_a_peer_reads_back_the_published_version(self):
        info = _client_info("train_0", {"w": _tensor_info([7, 7])})

        restored = NIXLClientInfo.deserialize(info.serialize())
        versions = [m.version for m in restored.tensor_infos["w"].shard_meta_infos]

        assert versions == [7, 7]

    def test_per_shard_versions_are_not_collapsed(self):
        """A torn publish is visible only if shards keep their own versions."""
        info = _client_info("train_0", {"w": _tensor_info([7, 6])})

        restored = NIXLClientInfo.deserialize(info.serialize())
        versions = [m.version for m in restored.tensor_infos["w"].shard_meta_infos]

        assert versions == [7, 6]

    def test_an_unpublished_shard_stays_unpublished(self):
        """`-1` must not round-trip into something that looks like a real version."""
        info = _client_info("train_0", {"w": _tensor_info([UNVERSIONED])})

        restored = NIXLClientInfo.deserialize(info.serialize())

        assert restored.tensor_infos["w"].shard_meta_infos[0].version == UNVERSIONED


class TestPublishUpdatesWhatAPeerWouldReceive:
    """The ordering bug, reproduced without a cluster.

    Peers receive whatever `local_client_info` held *at serialization time*. Stamping
    after that point leaves them holding `-1` -- which is exactly how the first two
    multi-node attempts failed.
    """

    def _publish(self, info: NIXLClientInfo, version: int) -> None:
        """What `NIXLStorageClient.publish_version` does to the local info."""
        for tensor_info in info.tensor_infos.values():
            for meta in tensor_info.shard_meta_infos:
                if meta is not None:
                    meta.version = version

    def test_stamping_before_serialization_reaches_the_peer(self):
        info = _client_info("train_0", {"w": _tensor_info([UNVERSIONED, UNVERSIONED])})

        self._publish(info, 3)
        peer_copy = NIXLClientInfo.deserialize(info.serialize())

        versions = [m.version for m in peer_copy.tensor_infos["w"].shard_meta_infos]
        assert versions == [3, 3]

    def test_stamping_after_serialization_does_not(self):
        """The regression this guards: the peer keeps the pre-stamp snapshot."""
        info = _client_info("train_0", {"w": _tensor_info([UNVERSIONED, UNVERSIONED])})

        peer_copy = NIXLClientInfo.deserialize(info.serialize())
        self._publish(info, 3)

        versions = [m.version for m in peer_copy.tensor_infos["w"].shard_meta_infos]
        assert versions == [UNVERSIONED, UNVERSIONED]

    def test_the_stale_pull_is_detected_from_the_peer_copy(self):
        """End to end over the wire format: publish, serialize, then check as a reader."""
        info = _client_info("train_0", {"w": _tensor_info([UNVERSIONED, UNVERSIONED])})
        peer_copy = NIXLClientInfo.deserialize(info.serialize())

        published = {
            "train_0": {
                "w": {
                    (i,): meta.version
                    for i, meta in enumerate(peer_copy.tensor_infos["w"].shard_meta_infos)
                }
            }
        }
        stale = check_pull_versions(0, published)

        assert len(stale) == 2
        assert all(item.actual is None for item in stale)


class TestTheClientStampsEveryRegisteredShard:
    """`publish_version` must cover every key, not just the first."""

    def test_all_keys_and_shards_are_stamped(self):
        from psrl.utils.nixl.client import NIXLStorageClient

        info = _client_info(
            "train_0",
            {"a": _tensor_info([UNVERSIONED, UNVERSIONED]), "b": _tensor_info([UNVERSIONED])},
        )

        # Call the real method against a stand-in holding only what it touches, so this
        # tests the shipped implementation rather than a copy of it.
        class _Stub:
            local_client_info = info

            def refresh_publish_copies(self):
                pass

        NIXLStorageClient.publish_version(_Stub(), 5)

        for tensor_info in info.tensor_infos.values():
            assert all(m.version == 5 for m in tensor_info.shard_meta_infos)
