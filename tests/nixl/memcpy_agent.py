"""
A NIXL agent that executes transfers as host memcpy, for CPU tests of the pull path.

It is a real `nixl_agent` without backends, so descriptor lists, serialisation and the
client's registration code run unchanged. Only the operations that need a transport are
replaced: memory registration is recorded, a READ copies bytes inside this process, and
notifications go through a shared in-memory bus. Every agent in one test shares the
address space, which is what makes a remote address directly copyable.
"""

from __future__ import annotations

import ctypes
import threading
from collections import defaultdict

import numpy as np
from nixl._api import nixl_agent, nixl_agent_config


class NotifBus:
    """Mailboxes keyed by agent name, shared by the agents of one test."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._boxes: dict[str, dict[str, list[bytes]]] = defaultdict(lambda: defaultdict(list))
        # agent name -> registered (base, length) regions, which every descriptor must fall inside
        self.regions: dict[str, list[tuple[int, int]]] = defaultdict(list)

    def send(self, sender: str, receiver: str, payload: bytes) -> None:
        with self._lock:
            self._boxes[receiver][sender].append(payload)

    def drain(self, receiver: str) -> dict[str, list[bytes]]:
        with self._lock:
            box = self._boxes.pop(receiver, {})
        return {sender: list(payloads) for sender, payloads in box.items()}


class _Rows(np.ndarray):
    """Descriptor rows that remember which agent registered them, so a read can be failed by source."""

    owner: str | None = None

    def __new__(cls, rows: np.ndarray):
        return rows.view(cls)

    def __array_finalize__(self, obj):
        self.owner = getattr(obj, "owner", None)


class _Handle:
    def __init__(self, local: np.ndarray, remote: np.ndarray) -> None:
        self.local = local
        self.remote = remote
        self.posts = 0
        self.owner: str | None = None
        self.released = False


class MemcpyAgent(nixl_agent):
    """`nixl_agent` whose READ is `memmove` between addresses of this process."""

    def __init__(self, name: str, bus: NotifBus) -> None:
        super().__init__(name, nixl_agent_config(enable_prog_thread=False, backends=[]))
        self.agent_name = name
        self.bus = bus
        self.registered: list[object] = []
        self.descriptors_posted = 0
        self.bytes_read = 0
        # Agents whose reads fail, standing in for a dead train rank. `transfer` reports ERR
        # for them rather than moving bytes, which is what the transport does once a peer is gone.
        self.dead_sources: set[str] = set()

    def register_memory(self, reg_list, mem_type=None, backends=()):
        descs = self.get_reg_descs(reg_list, mem_type)
        for i in range(descs.descCount()):
            base, length, _dev, _meta = descs[i]
            self.bus.regions[self.agent_name].append((base, length))
        self.registered.append(reg_list)
        return reg_list

    def deregister_memory(self, dereg_list, backends=()):
        return None

    def get_agent_metadata(self) -> bytes:
        return self.agent_name.encode()

    def add_remote_agent(self, metadata: bytes) -> str:
        return metadata.decode()

    def prep_xfer_dlist(self, agent_name, xfer_list, mem_type=None, backends=()):
        rows = _Rows(np.asarray(xfer_list, dtype=np.uint64).copy())
        owner = self.agent_name if agent_name == "NIXL_INIT_AGENT" else agent_name
        rows.owner = None if agent_name == "NIXL_INIT_AGENT" else agent_name
        regions = self.bus.regions[owner]
        for address, length, _dev in rows.tolist():
            # Like the real transport, a descriptor must lie inside one registered region.
            assert any(base <= address and address + length <= base + size for base, size in regions), (
                f"Descriptor at {address} of {length} bytes is not inside one region registered by {owner}."
            )
        return rows

    def make_prepped_xfer(self, operation, local_xfer_side, local_indices, remote_xfer_side, remote_indices, *a, **k):
        assert operation == "READ", f"Only READ is modelled, got {operation!r}."
        handle = _Handle(local_xfer_side[np.asarray(local_indices)], remote_xfer_side[np.asarray(remote_indices)])
        handle.owner = getattr(remote_xfer_side, "owner", None)
        return handle

    def transfer(self, handle, notif_msg=b""):
        if handle.owner in self.dead_sources:
            return "ERR"
        for (dst, n, _), (src, m, _) in zip(handle.local.tolist(), handle.remote.tolist()):
            assert n == m, f"Paired descriptors disagree in length: {n} != {m}."
            ctypes.memmove(dst, src, n)
            self.bytes_read += n
        handle.posts += 1
        self.descriptors_posted += handle.local.shape[0]
        return "DONE"

    def check_xfer_state(self, handle):
        if handle.released:
            # NIXL deletes the request on release; polling it afterwards reads freed memory
            # and its binding reports the garbage status as BAD_STATUS. Model the raise, not
            # the read, so a cache invalidation that frees an in-flight handle fails here
            # instead of corrupting a cluster run.
            raise RuntimeError("BAD_STATUS")
        return "ERR" if handle.owner in self.dead_sources else "DONE"

    def release_xfer_handle(self, handle):
        handle.released = True

    def release_dlist_handle(self, handle):
        return None

    def send_notif(self, remote_agent_name, notif_msg):
        self.bus.send(self.agent_name, remote_agent_name, bytes(notif_msg))

    def get_new_notifs(self):
        return self.bus.drain(self.agent_name)
