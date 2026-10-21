"""Bridge LMCache's off-GPU cache-event stream into SMG's KV-event stream.

SMG's cache-aware router scores two tiers independently: GPU-resident blocks
(reported by vLLM) and off-GPU blocks (LMCache L1/L2). The LMCache MP connector
does not surface its placements through vLLM's ``take_events``, so the off-GPU
tier has no data unless something forwards it. This module reads the node's
``/cache/events/stream`` NDJSON endpoint and converts each placement into the
same ``KvCacheEvent`` proto the GPU tier uses, with ``cache_level`` 1.

The router's indexer is positional: it hashes a request from position 0 and
extends the match only while the next block content also matches. A converted
store event therefore has to land its chunks at their real positions. LMCache
reports each chunk's ``token_offset``, and this bridge chains consecutive
stores for the same model and KV rank so a later turn's newly stored tail
continues from the prefix stored earlier.

The stream is advisory and lossy (LMCache drops batches for a slow consumer),
which matches the router's tolerance for missed events: a missing off-GPU hit
only costs a re-prefill. A malformed line is skipped for the same reason.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict

import aiohttp

from smg_grpc_proto.generated import common_pb2

psrl_logger = logging.getLogger(__file__)

# Off-GPU placements are cache level 1 in SMG's two-tier index.
_LMCACHE_CACHE_LEVEL = 1

# Keep the recent chunk-to-block-hash map bounded. Entries are per stored
# chunk, so this covers a multiple of the L1 chunk capacity of a large model
# while staying far below the event index's own memory.
_MAX_TRACKED_CHUNKS = 1 << 18

# Reconnect backoff when the node's MP server closes or refuses the stream.
_RECONNECT_DELAY_S = 1.0


def _stable_block_hash(chunk_hash_hex: str) -> int:
    """Map an LMCache chunk hash to the positive 63-bit id SMG uses.

    The value only has to be stable inside one stream and distinct per chunk:
    the router keeps the rolling content hash separately and uses this id to
    reverse deletions back to a stored block.
    """
    digest = hashlib.blake2b(chunk_hash_hex.encode("ascii"), digest_size=8).digest()
    return int.from_bytes(digest, "big") >> 1


class LMCacheKvEventBridge:
    """Convert one node's LMCache event stream into SMG `KvEventBatch` messages."""

    def __init__(self, base_url: str, current_version=None) -> None:
        """
        Initialize the bridge.

        Args:
            base_url: Management HTTP base URL of this node's MP server, e.g.
                ``http://10.0.0.1:41000``.
            current_version: Optional zero-argument callable returning the
                model version this replica currently serves. Entries tagged
                with another version are not indexed, and a version change
                resets the placement chain.
        """
        self._stream_url = f"{base_url.rstrip('/')}/cache/events/stream"
        self._current_version = current_version
        self._indexed_version: int | None = None
        # (model, kv_rank) -> {token_offset: (block_hash, position)}
        self._placements: dict[tuple[str, int], OrderedDict[int, tuple[int, int]]] = {}
        self._chunk_to_block: OrderedDict[str, int] = OrderedDict()

    def _version(self) -> int | None:
        """Return the version currently served, when the caller supplied one."""
        return self._current_version() if self._current_version is not None else None

    def _versions_match(self, entry: dict, version: int | None) -> bool:
        """Whether a placement belongs to the version this replica serves.

        Untagged deployments (no `lmcache.tag.model_version`) index every
        entry. A tagged entry is only visible to a request at its own version,
        so indexing it under another version would let the router select a
        worker whose L1 misses.
        """
        if version is None:
            return True
        for pair in entry.get("key", {}).get("tags") or ():
            if len(pair) == 2 and pair[0] == "lmcache.tag.model_version":
                return str(pair[1]) == str(version)
        return True

    def _reset_chain(self, version: int | None) -> None:
        """Drop placement state when the served model version changes."""
        if version == self._indexed_version:
            return
        self._indexed_version = version
        self._placements.clear()
        self._chunk_to_block.clear()

    def _parent_for(self, model: str, kv_rank: int, token_offset: int, block_size: int) -> int | None:
        """Return the block hash at the position just before ``token_offset``."""
        if token_offset < block_size:
            return None
        placements = self._placements.get((model, kv_rank))
        if not placements:
            return None
        previous = placements.get(token_offset - block_size)
        return previous[0] if previous else None

    def _remember(
        self,
        model: str,
        kv_rank: int,
        token_offset: int,
        block_size: int,
        chunk_hash_hex: str,
        block_hash: int,
    ) -> None:
        """Record a placement so the next turn's tail can chain onto it."""
        placements = self._placements.setdefault((model, kv_rank), OrderedDict())
        position = token_offset // block_size
        placements[token_offset] = (block_hash, position)
        placements.move_to_end(token_offset)
        while len(placements) > _MAX_TRACKED_CHUNKS:
            placements.popitem(last=False)
        self._chunk_to_block[chunk_hash_hex] = block_hash
        self._chunk_to_block.move_to_end(chunk_hash_hex)
        while len(self._chunk_to_block) > _MAX_TRACKED_CHUNKS:
            self._chunk_to_block.popitem(last=False)

    def convert(self, payload: dict, event_id_start: int) -> tuple[list[common_pb2.KvCacheEvent], int]:
        """
        Convert one ``CacheEventsRequest`` payload into SMG events.

        Args:
            payload: Decoded NDJSON object with a ``batches`` list.
            event_id_start: First event id to assign.

        Returns:
            tuple[list[common_pb2.KvCacheEvent], int]: Converted events and the
            next event id.
        """
        events: list[common_pb2.KvCacheEvent] = []
        event_id = event_id_start
        version = self._version()
        self._reset_chain(version)
        for batch in payload.get("batches", ()):
            event_type = str(batch.get("event_type", ""))
            if event_type not in ("store", "delete"):
                # `config` and `access` carry no placement for the router.
                continue
            entries = [entry for entry in batch.get("entries", ()) if entry.get("key")]
            if not entries:
                continue

            if event_type == "delete":
                block_hashes = []
                for entry in entries:
                    chunk_hash_hex = str(entry["key"].get("chunk_hash_hex", ""))
                    block_hash = self._chunk_to_block.pop(chunk_hash_hex, None)
                    if block_hash is not None:
                        block_hashes.append(block_hash)
                if block_hashes:
                    events.append(
                        common_pb2.KvCacheEvent(
                            event_id=event_id,
                            removed=common_pb2.KvBlocksRemoved(
                                block_hashes=block_hashes,
                                cache_level=_LMCACHE_CACHE_LEVEL,
                            ),
                        )
                    )
                    event_id += 1
                continue

            # A store batch is one request's newly committed chunks, ordered by
            # their position in the sequence. Split it into runs that are
            # contiguous in token offsets so each run is one parent chain.
            entries.sort(key=lambda entry: int(entry.get("token_offset", -1)))
            run: list[dict] = []
            run_parent: int | None = None
            for entry in entries:
                if not self._versions_match(entry, version):
                    continue
                token_ids = entry.get("token_ids") or []
                offset = int(entry.get("token_offset", -1))
                if not token_ids or offset < 0:
                    # Without both the tokens and the position the router
                    # cannot place the chunk; LMCache stamps tokens on the
                    # store path, so this only affects an evicted binding.
                    continue
                block_size = len(token_ids)
                if run and offset != int(run[-1].get("token_offset", -1)) + block_size:
                    events.append(self._stored_event(run, run_parent, event_id))
                    event_id += 1
                    run, run_parent = [], None
                if not run:
                    run_parent = self._parent_for(
                        str(entry["key"].get("model_name", "")),
                        int(entry["key"].get("kv_rank", 0)),
                        offset,
                        block_size,
                    )
                run.append(entry)
            if run:
                events.append(self._stored_event(run, run_parent, event_id))
                event_id += 1

        return events, event_id

    def _stored_event(
        self,
        entries: list[dict],
        parent_hash: int | None,
        event_id: int,
    ) -> common_pb2.KvCacheEvent:
        """Build one `KvBlocksStored` event and remember its placements."""
        blocks = []
        for entry in entries:
            token_ids = entry["token_ids"]
            key = entry["key"]
            chunk_hash_hex = str(key.get("chunk_hash_hex", ""))
            block_hash = _stable_block_hash(chunk_hash_hex)
            token_offset = int(entry.get("token_offset", 0))
            blocks.append(
                common_pb2.KvBlock(
                    block_hash=block_hash,
                    token_ids=token_ids,
                    block_size=len(token_ids),
                    cache_level=_LMCACHE_CACHE_LEVEL,
                )
            )
            self._remember(
                str(key.get("model_name", "")),
                int(key.get("kv_rank", 0)),
                token_offset,
                len(token_ids),
                chunk_hash_hex,
                block_hash,
            )
        stored = common_pb2.KvBlocksStored(blocks=blocks)
        if parent_hash is not None:
            stored.parent_block_hash = parent_hash
        return common_pb2.KvCacheEvent(event_id=event_id, stored=stored)

    async def stream(self, is_cancelled=None):
        """
        Yield converted `KvEventBatch` messages until cancelled.

        The MP server can restart independently of the rollout; reconnect
        instead of ending the router's subscription.

        Args:
            is_cancelled: Optional zero-argument callable returning whether the
                consumer has gone away.

        Yields:
            common_pb2.KvEventBatch: One converted batch per non-empty line.
        """
        sequence = 0
        event_id = 0
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=10.0, sock_read=None)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while is_cancelled is None or not is_cancelled():
                try:
                    async with session.get(self._stream_url) as response:
                        response.raise_for_status()
                        async for raw_line in response.content:
                            if is_cancelled is not None and is_cancelled():
                                return
                            line = raw_line.strip()
                            if not line:
                                continue
                            try:
                                payload = json.loads(line)
                            except json.JSONDecodeError:
                                psrl_logger.debug("[LMCache] Ignoring malformed event line.")
                                continue
                            events, event_id = self.convert(payload, event_id)
                            if not events:
                                continue
                            sequence += 1
                            yield common_pb2.KvEventBatch(
                                sequence_number=sequence,
                                timestamp=time.time(),
                                events=events,
                            )
                except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                    psrl_logger.warning(f"[LMCache] Event stream disconnected: {e}. Reconnecting.")
                if is_cancelled is not None and is_cancelled():
                    return
                await asyncio.sleep(_RECONNECT_DELAY_S)
