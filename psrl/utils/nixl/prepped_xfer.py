"""Prepped-descriptor submission for the NIXL hot path (implementation item A1).

`initialize_xfer` re-serialises the whole descriptor list on every call, so its cost
grows with descriptor count: measured 0.027ms at 1024 descriptors and 0.399ms at 16384
(`tests/nixl/bench_nixl_prepped_xfer.result.json`). `prep_xfer_dlist` pays that once and
`make_prepped_xfer` then ships only an index array, which measures 6-28x cheaper and
widens with scale.

The cache here turns that API into something the transfer path can use. Two properties
make it work:

* **A stable index per descriptor.** A prepped handle is addressed by position, so the
  mapping from a descriptor to its index must not move between rounds. `_IndexedDlist`
  assigns indices on first sight and keeps them.
* **Subset submission.** Each round transfers only the shards its plan selected, which
  is a subset of the prepped set -- verified against real RDMA in
  `test_descriptor_correctness_gpu.py`, including rotated and partial index sets.

Indices are `int32` ndarrays, never Python lists: `make_prepped_xfer` accepts both, but
pybind11 converts a list element by element and reintroduces exactly the per-descriptor
cost the prepped path exists to remove (0.088ms vs 0.006ms at 4096 descriptors). The
type is therefore enforced here rather than left to callers.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))

#: A descriptor as NIXL hands it back: (address, length, device_id).
Descriptor = tuple[int, int, int]

#: `prep_xfer_dlist` names memory types with strings and looks them up in
#: `nixl_agent.nixl_mems`, but the merged-transfer path carries the `nixl_mem_t` enum
#: that `nixlXferDList.getType()` returns. Passing the enum raises a bare `KeyError`
#: from inside the binding, so translate here.
_MEM_TYPE_NAMES = {
    "DRAM_SEG": "DRAM",
    "VRAM_SEG": "VRAM",
    "BLK_SEG": "BLK",
    "OBJ_SEG": "OBJ",
    "FILE_SEG": "FILE",
}


@lru_cache(maxsize=16)
def mem_type_name(mem_type) -> str:
    """Return the string name `prep_xfer_dlist` expects for a memory type.

    Accepts the `nixl_mem_t` enum, its spelling ("VRAM_SEG"), or the short name
    ("VRAM"). Anything else raises rather than being passed through: an unknown string
    would surface as a bare `KeyError` from inside the binding, which is how the enum
    mismatch originally hid.

    Cached because it sits on the per-submission path twice (local and remote side) and
    measured 1us per call uncached -- half the entire submission cost at small
    descriptor counts, for what is a constant lookup.
    """
    if isinstance(mem_type, str):
        if mem_type in _MEM_TYPE_NAMES:
            return _MEM_TYPE_NAMES[mem_type]
        if mem_type in _MEM_TYPE_NAMES.values():
            return mem_type
        raise ValueError(
            f"unrecognised NIXL memory type {mem_type!r}; expected one of "
            f"{sorted(_MEM_TYPE_NAMES)} or {sorted(set(_MEM_TYPE_NAMES.values()))}"
        )

    name = getattr(mem_type, "name", None)
    if name is None:
        name = str(mem_type).rsplit(".", 1)[-1]
    if name not in _MEM_TYPE_NAMES:
        raise ValueError(
            f"unrecognised NIXL memory type {mem_type!r}; known names are "
            f"{sorted(_MEM_TYPE_NAMES)}"
        )
    return _MEM_TYPE_NAMES[name]


@dataclass
class _IndexedDlist:
    """One prepped dlist plus the descriptor -> index mapping that addresses it."""

    handle: object
    mem_type: object
    descriptors: list[Descriptor]
    index_of: dict[Descriptor, int] = field(default_factory=dict)
    #: The index array for "all descriptors, in prepped order", which is the overwhelmingly
    #: common submission. Held separately from the keyed cache so the common case needs no
    #: hashing at all -- see `indices_for`.
    _identity_indices: np.ndarray | None = None
    #: Index arrays for other layouts, keyed by the descriptor tuple they address.
    _index_cache: dict[tuple[Descriptor, ...], np.ndarray] = field(default_factory=dict)
    #: Bounds the cache so a long run with varying chunk layouts cannot grow unbounded.
    _max_cached_index_arrays: int = 8

    def indices_for(self, wanted: list[Descriptor]) -> np.ndarray | None:
        """Index array for ``wanted``, or None if any descriptor is not in this dlist.

        Returning None rather than raising lets the caller re-prep: the descriptor set
        changes legitimately when memory is re-registered after an offload, and that is
        a cache miss, not an error.

        Everything here is about keeping the common path O(1). Three costs had to go,
        each of which was linear in descriptor count and so reintroduced exactly what the
        prepped path exists to remove:

        1. building the index array -- ~0.085us/descriptor
        2. hashing ``tuple(wanted)`` to look the array up -- ~0.027us/descriptor
        3. comparing descriptor lists elementwise to detect "same as last time"

        The identity check below avoids all three: when the caller passes the dlist's own
        descriptor list object, which `_entry` arranges for the full-round case, the
        cached `arange` is returned after one pointer comparison.

        The returned array must not be mutated by the caller; it is shared.
        """
        if wanted is self.descriptors:
            if self._identity_indices is None:
                self._identity_indices = np.arange(len(self.descriptors), dtype=np.int32)
            return self._identity_indices

        cache_key = tuple(wanted)
        cached = self._index_cache.get(cache_key)
        if cached is not None:
            return cached

        indices = np.empty(len(wanted), dtype=np.int32)
        for position, descriptor in enumerate(wanted):
            index = self.index_of.get(descriptor)
            if index is None:
                return None
            indices[position] = index

        if len(self._index_cache) >= self._max_cached_index_arrays:
            # Steady state reuses one or two layouts; dropping an arbitrary entry is
            # fine because a miss only costs a rebuild.
            self._index_cache.pop(next(iter(self._index_cache)))
        self._index_cache[cache_key] = indices
        return indices


class PreppedXferCache:
    """Prepped dlists for one NIXL agent, keyed by transfer direction.

    A separate entry exists per ``(op_type, target_agent, mem_type)`` because a prepped
    handle is bound to one remote agent and one memory type. The local side of a READ is
    this agent's own memory, so it preps against ``NIXL_INIT_AGENT``.
    """

    #: Sentinel NIXL uses for "this agent's own memory".
    LOCAL_AGENT = "NIXL_INIT_AGENT"

    def __init__(self, agent, client_name: str):
        self.agent = agent
        self.client_name = client_name
        self._local: dict[tuple, _IndexedDlist] = {}
        self._remote: dict[tuple, _IndexedDlist] = {}
        #: Counts for assertions and reporting: how often the hot path had to re-prep.
        self.prep_count = 0
        self.submit_count = 0

    def _prep(self, agent_name: str, mem_type, descriptors: list[Descriptor]) -> _IndexedDlist:
        array = np.zeros((len(descriptors), 3), dtype=np.uint64)
        for i, (address, length, device) in enumerate(descriptors):
            array[i] = (address, length, device)
        handle = self.agent.prep_xfer_dlist(
            agent_name, array, mem_type=mem_type_name(mem_type)
        )
        if not handle:
            raise RuntimeError(
                f"{self.client_name}: prep_xfer_dlist returned no handle for "
                f"{len(descriptors)} descriptors to {agent_name!r}"
            )
        self.prep_count += 1
        return _IndexedDlist(
            handle=handle,
            mem_type=mem_type,
            descriptors=list(descriptors),
            index_of={descriptor: i for i, descriptor in enumerate(descriptors)},
        )

    def _entry(
        self,
        cache: dict[tuple, _IndexedDlist],
        cache_key: tuple,
        agent_name: str,
        mem_type,
        descriptors: list[Descriptor],
    ) -> tuple[_IndexedDlist, np.ndarray]:
        """Return a dlist covering ``descriptors`` and the indices addressing them.

        On a miss the dlist is prepped over the *union* of what it already held and
        what is now wanted. Successive rounds transfer different chunks of the same
        registered memory, so prepping only the current round would re-prep every
        round and lose the entire benefit -- which is what an earlier version did.
        """
        entry = cache.get(cache_key)
        if entry is not None and entry.mem_type == mem_type:
            # Hand the dlist its own list object when the request is the full prepped
            # set, so `indices_for` can take the O(1) identity path. Comparing lengths
            # first keeps this check itself O(1) in the common case.
            if len(descriptors) == len(entry.descriptors) and descriptors == entry.descriptors:
                descriptors = entry.descriptors
            indices = entry.indices_for(descriptors)
            if indices is not None:
                return entry, indices

            # Grow to cover both the old and new descriptors. Addresses that have gone
            # away are harmless to keep prepped: nothing will index them, and dropping
            # them would force a re-prep the next time a previous round repeats.
            known = set(entry.descriptors)
            superset = entry.descriptors + [d for d in descriptors if d not in known]
            self.release_entry(entry)
            entry = self._prep(agent_name, mem_type, superset)
            cache[cache_key] = entry
            indices = entry.indices_for(descriptors)
            assert indices is not None, "superset dlist must cover the wanted descriptors"
            return entry, indices

        if entry is not None:
            # Memory type changed; the old handle cannot serve it.
            self.release_entry(entry)

        entry = self._prep(agent_name, mem_type, descriptors)
        cache[cache_key] = entry
        indices = entry.indices_for(descriptors)
        assert indices is not None, "freshly prepped dlist must cover its own descriptors"
        return entry, indices

    def make_xfer(
        self,
        op_type: str,
        target_agent: str,
        local_mem_type,
        local_descriptors: list[Descriptor],
        remote_mem_type,
        remote_descriptors: list[Descriptor],
        notif: bytes,
    ):
        """Compose a transfer from prepped dlists, preparing them if needed.

        ``local_descriptors`` and ``remote_descriptors`` must be positionally paired:
        element i of one is transferred to or from element i of the other.
        """
        assert len(local_descriptors) == len(remote_descriptors), (
            f"{self.client_name}: {len(local_descriptors)} local vs "
            f"{len(remote_descriptors)} remote descriptors; pairing is positional"
        )

        local_key = (op_type, self.LOCAL_AGENT, local_mem_type)
        remote_key = (op_type, target_agent, remote_mem_type)
        local_entry, local_indices = self._entry(
            self._local, local_key, self.LOCAL_AGENT, local_mem_type, local_descriptors
        )
        remote_entry, remote_indices = self._entry(
            self._remote, remote_key, target_agent, remote_mem_type, remote_descriptors
        )

        handle = self.agent.make_prepped_xfer(
            op_type,
            local_entry.handle,
            local_indices,
            remote_entry.handle,
            remote_indices,
            notif,
        )
        if not handle:
            raise RuntimeError(
                f"{self.client_name}: make_prepped_xfer returned no handle for "
                f"{len(local_descriptors)} descriptors ({op_type} -> {target_agent})"
            )
        self.submit_count += 1
        return handle

    def release_entry(self, entry: _IndexedDlist) -> None:
        try:
            self.agent.release_dlist_handle(entry.handle)
        except Exception:  # pragma: no cover - teardown must not mask the real failure
            psrl_logger.debug("%s: releasing a prepped dlist failed", self.client_name)

    def clear(self) -> None:
        """Release every prepped handle. Required before memory is deregistered.

        A prepped dlist holds the addresses it was built from, so submitting against it
        after deregistration reads freed pages. Design 3.7.3.0a calls for an explicit
        ``registration_epoch`` for the same reason on the remote side.
        """
        for cache in (self._local, self._remote):
            for entry in cache.values():
                self.release_entry(entry)
            cache.clear()
