"""
Rollout side weight pull: plan every read once, then post a handful of handles per pull.

A pull reads each planned `(source, key)` as byte runs paired between the source's read
layout and the local tensor. Runs that are adjacent on both sides merge, so a peer that
holds an identical tensor costs one descriptor per key. A side that is fragmented is read
through a bounded staging pool and scattered locally instead of paying a descriptor per
row. Same node peers form diffusion groups: per pull the first participating member of a
group reads the key upstream and the others read it from that member over NVLink.
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from psrl.utils.nixl.nixl_spec import NIXLClientType
from psrl.utils.nixl.read_geometry import coalesce, pair_runs, view_runs

if TYPE_CHECKING:
    from psrl.utils.nixl.client import NIXLStorageClient

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))

# Cost model used only to rank strategies. Posting one descriptor costs a few microseconds.
SECONDS_PER_DESCRIPTOR = 3e-6
BYTES_PER_SECOND = 25e9
# A key whose merged direct read needs at most this many descriptors per shard is read in place.
DIRECT_DESCRIPTORS_PER_SHARD = 64
# Staged items start on this alignment so every dtype view of the pool is legal.
_ALIGN = 256
# Signal chunks carried by the per pull notifications. Wave `w` is ready as chunk `_READY + w`.
_DONE, _READY = 0, 1

_NIXL_MEM = {"cuda": "VRAM", "cpu": "DRAM", "VRAM": "VRAM", "DRAM": "DRAM"}


@dataclass
class _Staged:
    """One shard read into the staging pool, then copied into its local view."""

    nbytes: int
    remote: np.ndarray
    rel: np.ndarray
    lengths: np.ndarray
    local_view: torch.Tensor
    view_shape: tuple[int, ...]
    view_strides: tuple[int, ...]


@dataclass
class _Spec:
    """Everything one `(source, key)` read moves, independent of when it runs."""

    source: str
    key: str
    direct_local: np.ndarray
    direct_remote: np.ndarray
    direct_len: np.ndarray
    direct_region: np.ndarray
    staged: list[_Staged]
    versions: set[int]
    nbytes: int
    # `(destination view, source view)` pairs for a source that shares this process's device.
    # Reading those through NIXL costs 7.7x a device copy, so they are copied, not transferred.
    ipc: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)


@dataclass
class _Handle:
    source: str
    handle: object
    descriptors: int
    nbytes: int
    posted_at: float = 0.0


@dataclass
class _Phase:
    """Posting order for one set of specs: direct handles, then staging rounds on two slots."""

    direct: list[_Handle] = field(default_factory=list)
    rounds: list[list[_Handle]] = field(default_factory=list)
    round_items: list[list[tuple[_Staged, int]]] = field(default_factory=list)
    versions: set[int] = field(default_factory=set)
    descriptors: int = 0
    nbytes: int = 0
    staged_bytes: int = 0
    ipc: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)
    ipc_bytes: int = 0


class ChunkUnavailable(RuntimeError):
    """
    Some keys have no live train source. Raised before any byte moves, so the rollout's weights are untouched.

    The caller keeps generating at its current version (design 3.5.7, D6).
    """

    def __init__(self, client: str, keys: list[str], dead: set[str]):
        self.keys, self.dead = keys, dead
        super().__init__(
            f"{client}: no live source holds {len(keys)} key(s), first {keys[0]!r}. "
            f"Unavailable sources: {sorted(dead)}."
        )


class TornPull(RuntimeError):
    """
    A source failed mid pull and no replica holds what it was serving.

    Earlier waves have already written the new version, so the weights mix two versions and
    the instance must not generate until a later pull completes.
    """


@dataclass
class PullReport:
    """What one pull did, for logs and for tests that pin the traffic it generated."""

    seconds: float
    upstream_seconds: float
    peer_seconds: float
    upstream_keys: int
    peer_keys: int
    fallback_keys: int
    upstream_bytes: int
    peer_bytes: int
    descriptors: int
    staged_bytes: int
    followers: int
    versions: set[int]
    # Per source `(bytes, first post, last completion)` of the upstream and peer reads.
    sources: dict[str, tuple[int, float, float]] = field(default_factory=dict)
    # Keys re-read from a DP replica because their planned source was unavailable or failed.
    takeover_keys: int = 0
    failed_sources: set[str] = field(default_factory=set)
    # Bytes copied from a source that shares this device, rather than transferred (3.7.2.1).
    ipc_bytes: int = 0


class NIXLPullEngine:
    """
    Pull engine bound to one rollout `NIXLStorageClient` after its handshake.

    Args:
        client (NIXLStorageClient): A PULL_SIDE client whose handshake has completed.
        staging_bytes (int): Size of the staging pool, split into two alternating slots.
        peer_timeout_s (float): How long a follower waits for its designate before reading upstream.
        waves (int): Upstream waves per pull. Peer reads of one wave overlap the next wave's upstream.
        read_timeout_s (float): How long one wave's reads may take before their sources count as failed.
    """

    def __init__(
        self,
        client: NIXLStorageClient,
        staging_bytes: int,
        peer_timeout_s: float,
        waves: int = 4,
        read_timeout_s: float = 300.0,
    ):
        self.client = client
        self.waves = max(1, int(waves))
        self.source_trace: dict[str, tuple[int, float, float]] = {}
        self.read_timeout_s = float(read_timeout_s)
        # Sources known to be gone (told by the coordinator, or learned from a failed read).
        self.unavailable: set[str] = set()
        # Sources that failed during the current pull.
        self.failed: set[str] = set()
        self.agent = client.agent
        self.staging_bytes = int(staging_bytes)
        self.peer_timeout_s = float(peer_timeout_s)
        self._specs: dict[tuple[str, str], _Spec] = {}
        self._phases: dict[tuple, _Phase] = {}
        self._pool: torch.Tensor | None = None
        self._pool_reg = None
        self._copy_stream: torch.cuda.Stream | None = None
        self._dlists: list[object] = []
        # (local allocation, remote allocation) to a small label, so coalescing never crosses one.
        self._region_labels: dict[tuple[int, int], int] = {}

    # ------------------------------------------------------------------ geometry

    def _local_view(self, key: str, shard: tuple[int, ...]) -> torch.Tensor:
        view = self.client.get_original_tensor_mapping().get((key, shard))
        assert view is not None, f"{self.client.client_name} holds no local view for {key!r} shard {shard}."
        return view

    def _spec(self, source: str, key: str, shards: list[tuple[int, ...]]) -> _Spec:
        cached = self._specs.get((source, key))
        if cached is not None:
            return cached
        remote_info = self.client.all_client_infos[source].tensor_infos[key]
        positions = {s: i for i, s in enumerate(remote_info.sharding.shard_indices)}
        cells = []
        for shard in shards:
            pos = positions[shard]
            meta = remote_info.shard_meta_infos[pos]
            local = self._local_view(key, shard)
            address = remote_info.read_addrs[pos] if remote_info.read_addrs else None
            assert address is not None, f"{source} exposes no readable bytes for {key!r} shard {shard}."
            assert meta.dtype == local.dtype and tuple(meta.shape) == tuple(local.shape), (
                f"{key!r} shard {shard} is {meta.dtype} {tuple(meta.shape)} at {source} "
                f"but {local.dtype} {tuple(local.shape)} locally."
            )
            itemsize = local.element_size()
            local_runs = view_runs(local.data_ptr(), tuple(local.shape), local.stride(), itemsize)
            remote_runs = view_runs(address, tuple(local.shape), tuple(remote_info.read_strides[pos]), itemsize)
            region = self._region_labels.setdefault(
                (local.untyped_storage().data_ptr(), remote_info.read_bases[pos]), len(self._region_labels)
            )
            cells.append((local, local_runs, remote_runs, tuple(remote_info.read_strides[pos]), meta.version, region))

        shared = self._ipc_cells(source, remote_info, shards, positions, cells)
        if shared is not None:
            empty = np.zeros(0, dtype=np.int64)
            spec = _Spec(
                source,
                key,
                empty,
                empty,
                empty,
                empty,
                [],
                {c[4] for c in cells},
                sum(c[1].total_bytes for c in cells),
                ipc=shared,
            )
            self._specs[(source, key)] = spec
            return spec

        # Whole-storage fast path (FINDINGS §2, design item D1).
        #
        # Peer diffusion (designate → follower) assigns byte-identical shards: both sides
        # hold the same gen-TP rank and lay the shard out in the same contiguous storage.
        # When local *and* remote are both contiguous we can issue one descriptor per shard
        # rather than one per contiguous run of each strided cell.  For a row-parallel key
        # cut by finest-mesh TP that collapses from O(hidden) descriptors to O(1).
        #
        # Safety: `meta.is_contiguous` is set by the source at registration time and means
        # the readable bytes form a single flat allocation starting at `read_addrs[pos]`.
        # `local.is_contiguous()` means the local destination is equally flat.  If either
        # fails the code falls through to the general strided path, which is always correct.
        if cells and all(
            c[0].is_contiguous()
            and remote_info.shard_meta_infos[positions[shard]].is_contiguous
            for c, shard in zip(cells, shards)
        ):
            local_arr = np.array([c[0].data_ptr() for c in cells], dtype=np.int64)
            remote_arr = np.array([remote_info.read_addrs[positions[shard]] for shard in shards], dtype=np.int64)
            lengths_arr = np.array([c[1].total_bytes for c in cells], dtype=np.int64)
            regions_arr = np.array([c[5] for c in cells], dtype=np.int64)
            spec = _Spec(
                source, key, local_arr, remote_arr, lengths_arr, regions_arr,
                [], {c[4] for c in cells}, int(lengths_arr.sum()),
            )
            self._specs[(source, key)] = spec
            return spec

        pairs = [pair_runs(c[1], c[2]) for c in cells]
        empty = np.zeros(0, dtype=np.int64)
        local = np.concatenate([p[0] for p in pairs]) if pairs else empty
        remote = np.concatenate([p[1] for p in pairs]) if pairs else empty
        lengths = np.concatenate([np.full(p[0].shape, p[2], dtype=np.int64) for p in pairs]) if pairs else empty
        regions = (
            np.concatenate([np.full(p[0].shape, c[5], dtype=np.int64) for p, c in zip(pairs, cells)])
            if pairs
            else empty
        )
        local, remote, lengths, regions = _coalesce(local, remote, lengths, regions)
        nbytes = int(lengths.sum())
        spec = _Spec(source, key, local, remote, lengths, regions, [], {c[4] for c in cells}, nbytes)
        if local.shape[0] > DIRECT_DESCRIPTORS_PER_SHARD * max(len(cells), 1):
            self._restage(spec, cells)
        self._specs[(source, key)] = spec
        return spec

    def _ipc_cells(self, source, remote_info, shards, positions, cells) -> list | None:
        """
        `(destination, source)` view pairs when `source` shares this device, else None.

        The source views are strided over the mapped region exactly as the remote descriptors
        would have addressed it, so the copy moves the same bytes the transfer would have.
        """
        info = self.client.all_client_infos[source]
        mapper = self.client.region_mapper()
        if not info.ipc_regions or info.device_uuid != mapper.uuid or self.client.device.type != "cuda":
            return None
        if not mapper.knows(source) and not mapper.adopt(source, info.ipc_regions):
            return None
        pairs = []
        for shard, (local, local_runs, _remote_runs, remote_strides, _version, _region) in zip(shards, cells):
            pos = positions[shard]
            address = remote_info.read_addrs[pos]
            itemsize = local.element_size()
            found = mapper.resolve(source, address, local_runs.total_bytes)
            if found is None or found[1] % itemsize:
                return None  # unmapped, or a view that would not start on an element boundary
            region, offset = found
            typed = region.view(local.dtype)
            view = torch.as_strided(typed, tuple(local.shape), tuple(remote_strides), offset // itemsize)
            pairs.append((local, view))
        return pairs

    def _restage(self, spec: _Spec, cells: list) -> None:
        """Pick, per shard, the cheapest of a direct read, a staged read, or a staged span read."""
        keep_local, keep_remote, keep_len, keep_region = [], [], [], []
        for local, lr, rr, remote_strides, _version, region in cells:
            nbytes = lr.total_bytes
            direct = nbytes // min(lr.run_bytes, rr.run_bytes)
            options = [(direct * SECONDS_PER_DESCRIPTOR + nbytes / BYTES_PER_SECOND, "direct")]
            if lr.count > 1:
                options.append((rr.count * SECONDS_PER_DESCRIPTOR + nbytes / BYTES_PER_SECOND, "stage"))
            span = int(rr.starts[-1] - rr.starts[0]) + rr.run_bytes
            if rr.count > 1:
                options.append((SECONDS_PER_DESCRIPTOR + span / BYTES_PER_SECOND, "span"))
            kind = min(options)[1]
            if kind == "direct":
                lo, ro, piece = pair_runs(lr, rr)
                keep_local.append(lo)
                keep_remote.append(ro)
                keep_len.append(np.full(lo.shape, piece, dtype=np.int64))
                keep_region.append(np.full(lo.shape, region, dtype=np.int64))
            elif kind == "stage":
                rel = np.arange(rr.count, dtype=np.int64) * rr.run_bytes
                spec.staged.append(
                    _Staged(
                        nbytes,
                        rr.starts,
                        rel,
                        np.full(rr.count, rr.run_bytes, dtype=np.int64),
                        local,
                        tuple(local.shape),
                        tuple(_dense(local.shape)),
                    )
                )
            else:
                spec.staged.append(
                    _Staged(
                        span,
                        rr.starts[:1].copy(),
                        np.zeros(1, dtype=np.int64),
                        np.array([span], dtype=np.int64),
                        local,
                        tuple(local.shape),
                        remote_strides,
                    )
                )
        if keep_local:
            spec.direct_local, spec.direct_remote, spec.direct_len, spec.direct_region = _coalesce(
                np.concatenate(keep_local),
                np.concatenate(keep_remote),
                np.concatenate(keep_len),
                np.concatenate(keep_region),
            )
        else:
            empty = np.zeros(0, dtype=np.int64)
            spec.direct_local, spec.direct_remote, spec.direct_len, spec.direct_region = empty, empty, empty, empty

    # ------------------------------------------------------------------ handles

    def _ensure_pool(self, need: int) -> None:
        size = max(self.staging_bytes, 2 * need)
        size = -(-size // (2 * _ALIGN)) * (2 * _ALIGN)
        if self._pool is not None and self._pool.numel() >= size:
            return
        # Prepared staging handles point into the old pool.
        self._release_handles()
        if self._pool_reg is not None:
            self.agent.deregister_memory(self._pool_reg)
        self._pool = torch.empty(size, dtype=torch.uint8, device=self.client.device)
        self._pool_reg = self.agent.register_memory([self._pool])

    def _make_handle(
        self,
        source: str,
        local: np.ndarray,
        remote: np.ndarray,
        lengths: np.ndarray,
        remote_mem: str,
        remote_dev: int,
    ) -> _Handle:
        count = int(local.shape[0])
        dev = self.client.device.index or 0
        local_mem = "VRAM" if self.client.device.type == "cuda" else "DRAM"
        local_rows = np.stack([local, lengths, np.full(count, dev, dtype=np.int64)], axis=1).astype(np.uint64)
        remote_rows = np.stack([remote, lengths, np.full(count, remote_dev, dtype=np.int64)], axis=1).astype(np.uint64)
        agent_name = self.client.agent_name_of(source)
        local_dlist = self.agent.prep_xfer_dlist("NIXL_INIT_AGENT", local_rows, mem_type=local_mem)
        remote_dlist = self.agent.prep_xfer_dlist(agent_name, remote_rows, mem_type=_NIXL_MEM[remote_mem])
        self._dlists.extend([local_dlist, remote_dlist])
        indices = np.arange(count, dtype=np.int32)
        handle = self.agent.make_prepped_xfer(
            "READ", local_dlist, indices, remote_dlist, indices, f"pull:{self.client.client_name}".encode()
        )
        return _Handle(source, handle, count, int(lengths.sum()))

    def _phase(self, specs: list[_Spec]) -> _Phase:
        signature = tuple(sorted((s.source, s.key) for s in specs))
        cached = self._phases.get(signature)
        if cached is not None:
            return cached
        phase = _Phase()
        by_source: dict[str, list[_Spec]] = defaultdict(list)
        for spec in specs:
            by_source[spec.source].append(spec)
            phase.versions |= spec.versions
        staged: list[tuple[str, _Staged]] = []
        for spec in specs:
            phase.ipc.extend(spec.ipc)
            if spec.ipc:
                phase.ipc_bytes += spec.nbytes
        for source, group in sorted(by_source.items()):
            info = self.client.all_client_infos[source].tensor_infos[group[0].key]
            local, remote, lengths, _ = _coalesce(
                np.concatenate([s.direct_local for s in group]),
                np.concatenate([s.direct_remote for s in group]),
                np.concatenate([s.direct_len for s in group]),
                np.concatenate([s.direct_region for s in group]),
            )
            if local.shape[0]:
                phase.direct.append(self._make_handle(source, local, remote, lengths, info.mem_type, info.device_id))
            staged.extend((source, item) for s in group for item in s.staged)

        if staged:
            self._ensure_pool(max(_aligned(item.nbytes) for _, item in staged))
            half = self._pool.numel() // 2
            rounds: list[list[tuple[str, _Staged, int]]] = [[]]
            used = 0
            for source, item in staged:
                if used + _aligned(item.nbytes) > half:
                    rounds.append([])
                    used = 0
                rounds[-1].append((source, item, used))
                used += _aligned(item.nbytes)
            base = self._pool.data_ptr()
            for index, members in enumerate(rounds):
                slot_base = base + (index % 2) * half
                handles = []
                per_source: dict[str, list[tuple[_Staged, int]]] = defaultdict(list)
                for source, item, offset in members:
                    per_source[source].append((item, offset))
                for source, items in sorted(per_source.items()):
                    info = self.client.all_client_infos[source].tensor_infos[by_source[source][0].key]
                    local = np.concatenate([slot_base + offset + item.rel for item, offset in items])
                    remote = np.concatenate([item.remote for item, _ in items])
                    lengths = np.concatenate([item.lengths for item, _ in items])
                    handles.append(self._make_handle(source, local, remote, lengths, info.mem_type, info.device_id))
                phase.rounds.append(handles)
                phase.round_items.append([(item, (index % 2) * half + offset) for _, item, offset in members])
                phase.staged_bytes += sum(item.nbytes for _, item, _ in members)
        phase.descriptors = sum(h.descriptors for h in phase.direct) + sum(
            h.descriptors for handles in phase.rounds for h in handles
        )
        phase.nbytes = sum(s.nbytes for s in specs)
        self._phases[signature] = phase
        return phase

    # ------------------------------------------------------------------ execution

    def _post(self, handles: list[_Handle]) -> list[_Handle]:
        posted = []
        for h in handles:
            h.posted_at = time.time()
            if h.source in self.failed:
                continue
            try:
                state = self.agent.transfer(h.handle)
            except Exception as exc:  # the transport may raise rather than return ERR for a lost peer
                psrl_logger.warning(f"{self.client.client_name}: posting a read from {h.source} raised {exc!r}.")
                state = "ERR"
            if state == "ERR":
                psrl_logger.warning(f"{self.client.client_name}: posting a read from {h.source} failed.")
                self.failed.add(h.source)
                continue
            posted.append(h)
        return posted

    def _wait(self, handles: list[_Handle], timeout: float | None = None) -> None:
        """
        Wait for every handle, polling them together and recording when each source finished.

        A handle that errors or outlives the read timeout marks its source failed rather than
        raising, so the other handles still complete and the caller can re-read that source's
        keys from a replica. A late completion from a merely slow source writes the same
        version's bytes, so it cannot corrupt the re-read.
        """
        timeout = self.read_timeout_s if timeout is None else timeout
        start = time.time()
        pending = list(handles)
        while pending:
            still = []
            for h in pending:
                state = self.agent.check_xfer_state(h.handle)
                if state == "DONE":
                    done = time.time()
                    bytes_, first, last = self.source_trace.get(h.source, (0, done, done))
                    self.source_trace[h.source] = (bytes_ + h.nbytes, min(first, h.posted_at), done)
                    continue
                if state == "ERR":
                    psrl_logger.warning(f"{self.client.client_name}: read of {h.nbytes} bytes from {h.source} failed.")
                    self.failed.add(h.source)
                    continue
                still.append(h)
            pending = still
            if pending:
                if time.time() - start > timeout:
                    late = sorted({h.source for h in pending})
                    psrl_logger.warning(f"{self.client.client_name}: reads from {late} exceeded {timeout}s.")
                    self.failed.update(late)
                    return
                time.sleep(5e-5)

    def _scatter(self, items: list[tuple[_Staged, int]]) -> None:
        with torch.cuda.stream(self._copy_stream) if self._copy_stream is not None else nullcontext():
            for item, offset in items:
                typed = self._pool.view(item.local_view.dtype)
                source = torch.as_strided(
                    typed, item.view_shape, item.view_strides, offset // item.local_view.element_size()
                )
                item.local_view.copy_(source, non_blocking=True)

    def _run(self, phase: _Phase) -> None:
        self._finish(self._start(phase))

    def _finish(self, direct: list[_Handle]) -> None:
        self._wait(direct)
        if self._copy_stream is not None:
            self._copy_stream.synchronize()

    def _start(self, phase: _Phase) -> list[_Handle]:
        """Post the direct reads, run the staged rounds to completion, and return the direct handles."""
        if self._copy_stream is None and self.client.device.type == "cuda":
            self._copy_stream = torch.cuda.Stream(device=self.client.device)
        direct = self._post(phase.direct)
        if phase.ipc:
            # Same device reads are copies, issued on the copy stream that `_finish` waits for.
            # The views are detached because a shard is often an output of `chunk`, which autograd
            # refuses to let anyone write into; the transfer path wrote raw addresses and never
            # met that rule. Weights are data here, so stepping outside autograd is the point.
            with (
                torch.no_grad(),
                torch.cuda.stream(self._copy_stream) if self._copy_stream is not None else nullcontext(),
            ):
                for destination, origin in phase.ipc:
                    destination.detach().copy_(origin.detach(), non_blocking=True)
        slot_free: list[torch.cuda.Event | None] = [None, None]
        in_flight: list[tuple[int, list[_Handle]]] = []

        def finish_oldest() -> None:
            index, handles = in_flight.pop(0)
            self._wait(handles)
            self._scatter(phase.round_items[index])
            if self._copy_stream is not None:
                event = torch.cuda.Event()
                event.record(self._copy_stream)
                slot_free[index % 2] = event

        for index, handles in enumerate(phase.rounds):
            if len(in_flight) == 2:
                finish_oldest()
            event = slot_free[index % 2]
            if event is not None:
                event.synchronize()
            in_flight.append((index, self._post(handles)))
        while in_flight:
            finish_oldest()
        return direct

    # ------------------------------------------------------------------ pull

    def mark_unavailable(self, sources) -> None:
        """Exclude train sources from every later pull, for example after the coordinator saw them die."""
        self.unavailable.update(sources)

    def _replica_of(self, key: str, shards: list, load: dict[str, int]) -> str | None:
        """
        A live train source holding every shard in `shards` of `key`, least loaded first.

        DP replicas of a train rank declare the same shard indices, so any of them serves the
        same bytes at the same version.
        """
        wanted = set(map(tuple, shards))
        candidates = []
        for name, info in self.client.all_client_infos.items():
            if name in self.unavailable or name in self.failed or info.type is not NIXLClientType.PUSH_SIDE:
                continue
            tensor_info = info.tensor_infos.get(key)
            if tensor_info is None or not wanted <= set(map(tuple, tensor_info.sharding.shard_indices)):
                continue
            candidates.append((load.get(name, 0), name))
        return min(candidates)[1] if candidates else None

    def _reroute(self, keys: dict[str, dict[str, list]]) -> tuple[dict[str, dict[str, list]], int, list[str]]:
        """
        Replace every unavailable or failed source in `keys` by a replica.

        Returns the rerouted reads, how many keys moved, and the keys no live source holds.
        """
        dead = self.unavailable | self.failed
        load: dict[str, int] = {}
        out: dict[str, dict[str, list]] = {}
        moved, missing = 0, []
        for key, per_source in keys.items():
            if not dead.intersection(per_source):
                out[key] = per_source
                continue
            rerouted: dict[str, list] = {}
            for source, shards in per_source.items():
                target = source
                if source in dead:
                    target = self._replica_of(key, shards, load)
                    if target is None:
                        missing.append(key)
                        break
                    load[target] = load.get(target, 0) + 1
                rerouted.setdefault(target, []).extend(shards)
            else:
                out[key] = {src: sorted(map(tuple, shards)) for src, shards in rerouted.items()}
                moved += 1
        return out, moved, missing

    def _assign(self, participates) -> tuple[dict[str, dict[str, list]], dict[str, list[str]], set[str]]:
        """Split this client's keys into upstream reads and reads from a designate."""
        me = self.client.client_name
        plan = self.client.comm_plan
        assert plan is not None, f"{me} has no communication plan."
        upstream: dict[str, dict[str, list]] = {}
        peer: dict[str, list[str]] = defaultdict(list)
        followers: set[str] = set()
        for key, per_source in plan.rollout_pull_from_ps_plan.get(me, {}).items():
            live = [w for w in plan.get_peer_group(me, key) if w == me or participates(w)]
            if len(live) >= 2:
                if live[0] != me:
                    peer[live[0]].append(key)
                    continue
                followers.update(live[1:])
            upstream[key] = per_source
        return upstream, dict(peer), followers

    def _upstream_specs(self, keys: dict[str, dict[str, list]]) -> list[_Spec]:
        return [self._spec(src, key, shards) for key, per_source in keys.items() for src, shards in per_source.items()]

    def _waves_of(self, keys: list[str]) -> dict[str, int]:
        """
        Wave of every key, cut in model order into byte balanced runs.

        Group members hold identical keys, so a designate and its followers derive the same waves.
        """
        info = self.client.local_client_info.tensor_infos
        sizes = [
            (key, sum(info[key].get_shard_size_bytes(i) for i in range(info[key].num_local_shards))) for key in keys
        ]
        total = max(sum(n for _, n in sizes), 1)
        waves, acc = {}, 0
        for key, nbytes in sorted(sizes):
            waves[key] = min(self.waves - 1, acc * self.waves // total)
            acc += nbytes
        return waves

    def pull(self, epoch: int, participants: set[str] | None = None) -> PullReport:
        """
        Read this client's weights for one sync.

        Upstream reads go in waves. After wave `w` lands the client tells its followers, and
        reads its own peer keys of wave `w - 1` over NVLink while wave `w` crosses the NIC.

        Args:
            epoch (int): Identifier of this sync, shared by every participant and never reused.
            participants (set[str] | None): Pull clients syncing together. None means all of them.
        """
        start = time.time()
        self.source_trace = {}
        self.failed = set()
        upstream, peer, followers = self._assign(lambda c: participants is None or c in participants)
        # Resolve known dead sources before any byte moves, so a missing key leaves the weights untouched.
        upstream, takeover, missing = self._reroute(upstream)
        if missing:
            raise ChunkUnavailable(self.client.client_name, missing, set(self.unavailable))
        publisher, waiter = self.client.chunk_completion_channel()
        all_keys = list(upstream) + [key for keys in peer.values() for key in keys]
        wave_of = self._waves_of(all_keys)
        up_waves = [{k: v for k, v in upstream.items() if wave_of[k] == w} for w in range(self.waves)]
        peer_waves = [
            {d: [k for k in keys if wave_of[k] == w] for d, keys in peer.items() if any(wave_of[k] == w for k in keys)}
            for w in range(self.waves)
        ]

        versions: set[int] = set()
        stats = {
            "upstream_bytes": 0,
            "peer_bytes": 0,
            "descriptors": 0,
            "staged": 0,
            "fallback": 0,
            "takeover": takeover,
            "ipc": 0,
        }

        def take_over(keys: dict[str, dict[str, list]]) -> None:
            """Re-read from replicas the keys of `keys` whose sources failed during this pull."""
            affected = {k: v for k, v in keys.items() if self.failed.intersection(v)}
            if not affected:
                return
            rerouted, moved, lost = self._reroute(affected)
            if lost:
                raise TornPull(
                    f"{self.client.client_name}: sources {sorted(self.failed)} failed mid pull and no replica "
                    f"holds {len(lost)} key(s), first {lost[0]!r}. The weights now mix two versions."
                )
            psrl_logger.warning(
                f"{self.client.client_name}: re-reading {moved} key(s) of failed sources {sorted(self.failed)} "
                f"from DP replicas in epoch {epoch}."
            )
            stats["takeover"] += moved
            phase = self._phase(self._upstream_specs(rerouted))
            versions.update(phase.versions)
            self._run(phase)
            take_over(rerouted)

        upstream_done = start

        def peer_wave(w: int) -> list[_Handle]:
            """Wait for this wave's designates, start reading them, and read any late designate's keys upstream."""
            wanted = peer_waves[w]
            if not wanted:
                return []
            ready = self._await(waiter, epoch, _READY + w, set(wanted))
            specs = [
                self._spec(d, key, list(self.client.local_client_info.tensor_infos[key].sharding.shard_indices))
                for d in sorted(ready)
                for key in wanted[d]
            ]
            handles = []
            if specs:
                phase = self._phase(specs)
                stats["peer_bytes"] += phase.nbytes
                stats["descriptors"] += phase.descriptors
                stats["ipc"] += phase.ipc_bytes
                handles = self._start(phase)
            late = set(wanted) - ready
            if late:
                psrl_logger.warning(
                    f"{self.client.client_name}: designates {sorted(late)} were not ready after "
                    f"{self.peer_timeout_s}s in epoch {epoch}, reading their keys upstream."
                )
                plan = self.client.comm_plan.rollout_pull_from_ps_plan[self.client.client_name]
                keys = {key: plan[key] for d in late for key in wanted[d]}
                stats["fallback"] += len(keys)
                fallback = self._phase(self._upstream_specs(keys))
                versions.update(fallback.versions)
                self._run(fallback)
            return handles

        pending_peer: list[_Handle] = []
        for w in range(self.waves):
            phase = self._phase(self._upstream_specs(up_waves[w])) if up_waves[w] else None
            direct = self._start(phase) if phase is not None else []
            if w > 0:
                pending_peer += peer_wave(w - 1)
            self._finish(direct)
            if phase is not None:
                versions.update(phase.versions)
                stats["upstream_bytes"] += phase.nbytes
                stats["descriptors"] += phase.descriptors
                stats["staged"] += phase.staged_bytes
                stats["ipc"] += phase.ipc_bytes
                # Before followers hear this wave is ready, so they never read a hole.
                take_over(up_waves[w])
            if self.client.device.type == "cuda":
                torch.cuda.synchronize(self.client.device)
            if followers:
                publisher.publish(epoch, _READY + w, sorted(followers))
        upstream_done = time.time()
        pending_peer += peer_wave(self.waves - 1)
        self._finish(pending_peer)
        lost_designates = self.failed.intersection(peer)
        if lost_designates:
            plan = self.client.comm_plan.rollout_pull_from_ps_plan[self.client.client_name]
            keys = {key: plan[key] for d in lost_designates for key in peer[d]}
            psrl_logger.warning(
                f"{self.client.client_name}: designates {sorted(lost_designates)} failed mid read, "
                f"reading their {len(keys)} key(s) upstream."
            )
            stats["fallback"] += len(keys)
            rerouted, moved, lost = self._reroute(keys)
            if lost:
                raise TornPull(
                    f"{self.client.client_name}: no live source holds {lost[0]!r} after its designate failed."
                )
            stats["takeover"] += moved
            fallback = self._phase(self._upstream_specs(rerouted))
            versions.update(fallback.versions)
            self._run(fallback)
        if self.client.device.type == "cuda":
            torch.cuda.synchronize(self.client.device)
        if peer:
            publisher.publish(epoch, _DONE, sorted(peer))
        if followers:
            done = self._await(waiter, epoch, _DONE, followers)
            if done != followers:
                psrl_logger.warning(
                    f"{self.client.client_name}: followers {sorted(followers - done)} did not confirm "
                    f"epoch {epoch} within {self.peer_timeout_s}s."
                )
        waiter.forget_versions_before(epoch)
        end = time.time()
        return PullReport(
            seconds=end - start,
            upstream_seconds=upstream_done - start,
            peer_seconds=end - upstream_done,
            upstream_keys=len(upstream),
            peer_keys=sum(len(v) for v in peer.values()),
            fallback_keys=stats["fallback"],
            upstream_bytes=stats["upstream_bytes"],
            peer_bytes=stats["peer_bytes"],
            descriptors=stats["descriptors"],
            staged_bytes=stats["staged"],
            followers=len(followers),
            versions=versions,
            sources=dict(self.source_trace),
            takeover_keys=stats["takeover"],
            ipc_bytes=stats["ipc"],
            failed_sources=set(self.failed),
        )

    def _await(self, waiter, epoch: int, chunk: int, sources: set[str]) -> set[str]:
        """Wait up to the peer timeout and return the sources that signalled."""
        deadline = time.time() + self.peer_timeout_s
        while True:
            waiter.absorb()
            arrived = {s for s in sources if waiter.seen(epoch, chunk, s)}
            if arrived == sources or time.time() > deadline:
                return arrived
            time.sleep(2e-4)

    def _release_handles(self) -> None:
        for phase in self._phases.values():
            for h in phase.direct + [h for handles in phase.rounds for h in handles]:
                self.agent.release_xfer_handle(h.handle)
        for dlist in self._dlists:
            self.agent.release_dlist_handle(dlist)
        self._phases.clear()
        self._dlists.clear()

    def release(self) -> None:
        """Drop every prepared handle and the staging pool. Required before memory is deregistered."""
        self._release_handles()
        self._specs.clear()
        if self._pool_reg is not None:
            self.agent.deregister_memory(self._pool_reg)
        self._pool = None
        self._pool_reg = None


def _coalesce(local, remote, lengths, regions):
    """`coalesce` that also returns the region label of every merged segment."""
    if local.shape[0] <= 1:
        return local, remote, lengths, regions
    order = np.lexsort((remote, local))
    merged = coalesce(local, remote, lengths, regions)
    heads = np.searchsorted(local[order], merged[0])
    return (*merged, regions[order][heads])


def _aligned(nbytes: int) -> int:
    return -(-nbytes // _ALIGN) * _ALIGN


def _dense(shape) -> list[int]:
    strides, acc = [], 1
    for extent in reversed(tuple(shape)):
        strides.append(acc)
        acc *= int(extent)
    return list(reversed(strides))
