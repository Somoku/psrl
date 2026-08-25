"""Unit tests for KVCacheManager LMCache local snapshot query."""


class TestLmcacheLocalSnapshot:
    """Tests for get_lmcache_cache_info_local."""

    def _make_manager(self):
        from unittest.mock import MagicMock

        from psrl.utils.kv_cache.config import LMCacheConfig
        from psrl.utils.kv_cache.manager import KVCacheManager

        config = MagicMock(spec=LMCacheConfig)
        config.enable = True
        config.gpu_pin_block_budget = 100
        config.backend = "cpu"
        config.offload_size_gb = 1
        config.chunk_size = 4
        config.cache_policy = "lru"
        config.save_decode_cache = False
        config.save_unfull_chunk = False
        config.enable_async_loading = False
        config.clear_on_weight_update = False
        config.enable_p2p = False
        config.config_file = None
        config.lmcache_instance_id = "test_0"
        mgr = KVCacheManager(config)
        return mgr

    def _compute_chunk_hashes(self, tokens, chunk_size, hash_fn, none_hash):
        """Compute chunk hashes using the same algorithm as LMCache ChunkedTokenDatabase."""
        hashes = []
        prefix_hash = none_hash
        num_full_chunks = len(tokens) // chunk_size
        for i in range(num_full_chunks):
            start = i * chunk_size
            end = start + chunk_size
            tokens_tuple = tuple(tokens[start:end])
            prefix_hash = hash_fn((prefix_hash, tokens_tuple, ()))
            hashes.append(prefix_hash)
        return hashes

    def test_no_snapshot_returns_zeros(self):
        """Test that without a snapshot, cached tokens are 0."""
        mgr = self._make_manager()
        result = mgr.get_lmcache_cache_info_local([1, 2, 3, 4, 5, 6, 7, 8])
        assert result["lmcache_cached_tokens"] == 0
        assert result["lmcache_cached_chunks"] == 0

    def test_full_prefix_match(self):
        """Test that full prefix match counts all cached chunks."""
        mgr = self._make_manager()
        chunk_size = 4
        tokens = list(range(12))  # 3 full chunks

        from lmcache.v1.token_database import NONE_HASH
        from vllm.utils.hashing import sha256

        hash_fn = sha256

        hashes = self._compute_chunk_hashes(tokens, chunk_size, hash_fn, NONE_HASH)
        snapshot = {
            "chunk_hash_set": set(hashes),
            "chunk_size": chunk_size,
            "total_bytes": 1024,
            "chunk_bytes": 64,
        }
        mgr.update_lmcache_backend_snapshot(snapshot)
        # Pre-populate the cached functions to match expected hash fn
        mgr._lmcache_hash_fn = hash_fn
        mgr._lmcache_none_hash = NONE_HASH

        result = mgr.get_lmcache_cache_info_local(tokens)
        assert result["lmcache_cached_chunks"] == 3
        assert result["lmcache_cached_tokens"] == 12

    def test_partial_prefix_match(self):
        """Test that partial prefix match breaks at first missing hash."""
        mgr = self._make_manager()
        chunk_size = 4
        tokens = list(range(12))

        from lmcache.v1.token_database import NONE_HASH
        from vllm.utils.hashing import sha256

        hash_fn = sha256

        hashes = self._compute_chunk_hashes(tokens, chunk_size, hash_fn, NONE_HASH)
        snapshot = {
            "chunk_hash_set": set(hashes[:2]),  # only first 2 chunks
            "chunk_size": chunk_size,
            "total_bytes": 1024,
            "chunk_bytes": 64,
        }
        mgr.update_lmcache_backend_snapshot(snapshot)
        mgr._lmcache_hash_fn = hash_fn
        mgr._lmcache_none_hash = NONE_HASH

        result = mgr.get_lmcache_cache_info_local(tokens)
        assert result["lmcache_cached_chunks"] == 2
        assert result["lmcache_cached_tokens"] == 8

    def test_no_match_with_empty_snapshot(self):
        """Test that empty snapshot results in no cache hits."""
        mgr = self._make_manager()
        chunk_size = 4
        tokens = list(range(12))

        from lmcache.v1.token_database import NONE_HASH
        from vllm.utils.hashing import sha256

        hash_fn = sha256

        snapshot = {
            "chunk_hash_set": set(),
            "chunk_size": chunk_size,
            "total_bytes": 1024,
            "chunk_bytes": 64,
        }
        mgr.update_lmcache_backend_snapshot(snapshot)
        mgr._lmcache_hash_fn = hash_fn
        mgr._lmcache_none_hash = NONE_HASH

        result = mgr.get_lmcache_cache_info_local(tokens)
        assert result["lmcache_cached_chunks"] == 0
        assert result["lmcache_cached_tokens"] == 0

    def test_empty_tokens(self):
        """Test that empty token list returns 0 cached tokens."""
        mgr = self._make_manager()
        snapshot = {
            "chunk_hash_set": {b"test_hash"},
            "chunk_size": 4,
            "total_bytes": 1024,
            "chunk_bytes": 64,
        }
        mgr.update_lmcache_backend_snapshot(snapshot)
        result = mgr.get_lmcache_cache_info_local([])
        assert result["lmcache_cached_tokens"] == 0

    def test_hash_mismatch_disables_snapshot(self):
        """Test that hash function mismatch disables the snapshot."""
        mgr = self._make_manager()

        from lmcache.v1.token_database import NONE_HASH
        from vllm.utils.hashing import sha256

        hash_fn = sha256
        mgr._lmcache_hash_fn = hash_fn
        mgr._lmcache_none_hash = NONE_HASH

        snapshot = {
            "chunk_hash_set": {b"test_hash"},
            "chunk_size": 4,
            "total_bytes": 1024,
            "chunk_bytes": 64,
            "verification_tokens": list(range(4)),
            "verification_hash": b"wrong_hash",  # intentionally wrong
        }
        mgr.update_lmcache_backend_snapshot(snapshot)
        # Should have set mismatch flag, snapshot NOT stored
        assert getattr(mgr, "_lmcache_hash_mismatch", False) is True
        assert mgr._lmcache_chunk_hash_set is None

    def test_usage_percentage_calculation(self):
        """Test that cache usage percentage is calculated correctly."""
        mgr = self._make_manager()
        chunk_size = 4
        tokens = list(range(12))

        from lmcache.v1.token_database import NONE_HASH
        from vllm.utils.hashing import sha256

        hash_fn = sha256

        hashes = self._compute_chunk_hashes(tokens, chunk_size, hash_fn, NONE_HASH)
        total_bytes = 1024
        chunk_bytes = 64
        snapshot = {
            "chunk_hash_set": set(hashes[:2]),  # 2 out of 3 chunks
            "chunk_size": chunk_size,
            "total_bytes": total_bytes,
            "chunk_bytes": chunk_bytes,
        }
        mgr.update_lmcache_backend_snapshot(snapshot)
        mgr._lmcache_hash_fn = hash_fn
        mgr._lmcache_none_hash = NONE_HASH

        result = mgr.get_lmcache_cache_info_local(tokens)
        expected_bytes = 2 * chunk_bytes
        expected_pct = expected_bytes / total_bytes
        assert result["lmcache_bytes"] == expected_bytes
        assert abs(result["lmcache_usage_pct"] - expected_pct) < 1e-6
