"""Tests for response caching functionality."""

import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from time import sleep

from llm_game_utils.caching.response_cache import ResponseCache
from llm_game_utils.clients.base_client import LLMResponse
from llm_game_utils.exceptions import CacheError


class TestCacheInitialization:
    """Test response cache initialization."""

    def test_default_initialization(self, tmp_path):
        """Test initialization with default settings."""
        cache_dir = tmp_path / "cache"
        cache = ResponseCache(cache_dir=str(cache_dir))

        assert cache.cache_dir == cache_dir
        assert cache.enabled is True
        assert cache.ttl_hours == 24
        assert cache_dir.exists()

    def test_custom_ttl(self, tmp_path):
        """Test initialization with custom TTL."""
        cache = ResponseCache(
            cache_dir=str(tmp_path / "cache"),
            ttl_hours=48
        )
        assert cache.ttl_hours == 48

    def test_no_ttl(self, tmp_path):
        """Test initialization with no expiration."""
        cache = ResponseCache(
            cache_dir=str(tmp_path / "cache"),
            ttl_hours=None
        )
        assert cache.ttl_hours is None

    def test_disabled_cache(self, tmp_path):
        """Test initialization with caching disabled."""
        cache = ResponseCache(
            cache_dir=str(tmp_path / "cache"),
            enabled=False
        )
        assert cache.enabled is False


class TestCacheKey:
    """Test cache key generation."""

    def test_same_inputs_same_key(self, tmp_path):
        """Test that identical inputs produce same key."""
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"))

        key1 = cache.get_cache_key("model-1", "prompt", 0.7)
        key2 = cache.get_cache_key("model-1", "prompt", 0.7)

        assert key1 == key2

    def test_different_model_different_key(self, tmp_path):
        """Test that different models produce different keys."""
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"))

        key1 = cache.get_cache_key("model-1", "prompt", 0.7)
        key2 = cache.get_cache_key("model-2", "prompt", 0.7)

        assert key1 != key2

    def test_different_prompt_different_key(self, tmp_path):
        """Test that different prompts produce different keys."""
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"))

        key1 = cache.get_cache_key("model-1", "prompt1", 0.7)
        key2 = cache.get_cache_key("model-1", "prompt2", 0.7)

        assert key1 != key2

    def test_different_temperature_different_key(self, tmp_path):
        """Test that different temperatures produce different keys."""
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"))

        key1 = cache.get_cache_key("model-1", "prompt", 0.5)
        key2 = cache.get_cache_key("model-1", "prompt", 0.7)

        assert key1 != key2

    def test_system_prompt_affects_key(self, tmp_path):
        """Test that system prompt affects cache key."""
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"))

        key1 = cache.get_cache_key("model-1", "prompt", 0.7, system_prompt="sys1")
        key2 = cache.get_cache_key("model-1", "prompt", 0.7, system_prompt="sys2")
        key3 = cache.get_cache_key("model-1", "prompt", 0.7, system_prompt=None)

        assert key1 != key2
        assert key1 != key3

    def test_max_tokens_affects_key(self, tmp_path):
        """Test that max_tokens affects cache key."""
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"))

        key1 = cache.get_cache_key("model-1", "prompt", 0.7, max_tokens=100)
        key2 = cache.get_cache_key("model-1", "prompt", 0.7, max_tokens=200)
        key3 = cache.get_cache_key("model-1", "prompt", 0.7, max_tokens=None)

        assert key1 != key2
        assert key1 != key3


class TestCacheGetSet:
    """Test cache get and set operations."""

    def test_cache_miss(self, tmp_path):
        """Test cache miss returns None."""
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"))

        result = cache.get("model-1", "prompt", 0.7)
        assert result is None
        assert cache.misses == 1
        assert cache.hits == 0

    def test_cache_set_and_get(self, tmp_path):
        """Test setting and getting cached data."""
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"))

        response_data = {
            'response': 'Test response',
            'cost': 0.01,
            'tokens': 100
        }

        cache.set("model-1", "prompt", 0.7, response_data)

        retrieved = cache.get("model-1", "prompt", 0.7)
        assert retrieved == response_data
        assert cache.hits == 1

    def test_cache_with_llm_response(self, tmp_path):
        """Test caching LLMResponse objects."""
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"))

        llm_response = LLMResponse(
            model_id="model-1",
            model_name="Model 1",
            prompt="test prompt",
            response="test response",
            timestamp=datetime.now(),
            response_time=1.0,
            input_tokens=50,
            output_tokens=50,
            total_tokens=100,
            cost=0.01,
            metadata={}
        )

        cache.set("model-1", "test prompt", 0.7, llm_response)

        retrieved = cache.get("model-1", "test prompt", 0.7)
        assert retrieved is not None
        assert retrieved['response'] == "test response"
        assert retrieved['cost'] == 0.01

    def test_disabled_cache_get_returns_none(self, tmp_path):
        """Test that disabled cache always returns None."""
        cache = ResponseCache(
            cache_dir=str(tmp_path / "cache"),
            enabled=False
        )

        cache.set("model-1", "prompt", 0.7, {'data': 'test'})
        result = cache.get("model-1", "prompt", 0.7)

        assert result is None

    def test_disabled_cache_set_does_nothing(self, tmp_path):
        """Test that disabled cache doesn't write files."""
        cache_dir = tmp_path / "cache"
        cache = ResponseCache(
            cache_dir=str(cache_dir),
            enabled=False
        )

        cache.set("model-1", "prompt", 0.7, {'data': 'test'})

        # Cache directory shouldn't even be created
        if cache_dir.exists():
            assert len(list(cache_dir.glob("*.json"))) == 0


class TestCacheExpiration:
    """Test cache expiration functionality."""

    def test_expired_cache_returns_none(self, tmp_path):
        """Test that expired cache entries return None."""
        cache_dir = tmp_path / "cache"
        cache = ResponseCache(cache_dir=str(cache_dir), ttl_hours=1)

        # Create a cache entry
        cache.set("model-1", "prompt", 0.7, {'data': 'test'})

        # Manually modify the cached_at time to be old
        cache_files = list(cache_dir.glob("*.json"))
        assert len(cache_files) == 1

        with open(cache_files[0], 'r') as f:
            data = json.load(f)

        # Set cached_at to 2 hours ago (beyond the 1 hour TTL)
        old_time = datetime.now() - timedelta(hours=2)
        data['cached_at'] = old_time.isoformat()

        with open(cache_files[0], 'w') as f:
            json.dump(data, f)

        # Should return None because it's expired
        result = cache.get("model-1", "prompt", 0.7)
        assert result is None

        # The expired file should be deleted
        assert len(list(cache_dir.glob("*.json"))) == 0

    def test_no_expiration_with_none_ttl(self, tmp_path):
        """Test that cache never expires with ttl_hours=None."""
        cache_dir = tmp_path / "cache"
        cache = ResponseCache(cache_dir=str(cache_dir), ttl_hours=None)

        cache.set("model-1", "prompt", 0.7, {'data': 'test'})

        # Manually modify cached_at to be very old
        cache_files = list(cache_dir.glob("*.json"))
        with open(cache_files[0], 'r') as f:
            data = json.load(f)

        old_time = datetime.now() - timedelta(days=365)
        data['cached_at'] = old_time.isoformat()

        with open(cache_files[0], 'w') as f:
            json.dump(data, f)

        # Should still return the data
        result = cache.get("model-1", "prompt", 0.7)
        assert result is not None


class TestCacheClear:
    """Test cache clearing operations."""

    def test_clear_all(self, tmp_path):
        """Test clearing all cache entries."""
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"))

        # Add multiple entries
        cache.set("model-1", "prompt1", 0.7, {'data': '1'})
        cache.set("model-2", "prompt2", 0.7, {'data': '2'})
        cache.set("model-3", "prompt3", 0.7, {'data': '3'})

        deleted = cache.clear()
        assert deleted == 3

        # All should be gone
        assert cache.get("model-1", "prompt1", 0.7) is None
        assert cache.get("model-2", "prompt2", 0.7) is None
        assert cache.get("model-3", "prompt3", 0.7) is None

    def test_clear_expired_only(self, tmp_path):
        """Test clearing only expired entries."""
        cache_dir = tmp_path / "cache"
        cache = ResponseCache(cache_dir=str(cache_dir), ttl_hours=1)

        # Add entries
        cache.set("model-1", "prompt1", 0.7, {'data': '1'})
        cache.set("model-2", "prompt2", 0.7, {'data': '2'})

        # Make one entry expired
        cache_files = list(cache_dir.glob("*.json"))
        with open(cache_files[0], 'r') as f:
            data = json.load(f)

        old_time = datetime.now() - timedelta(hours=2)
        data['cached_at'] = old_time.isoformat()

        with open(cache_files[0], 'w') as f:
            json.dump(data, f)

        # Clear expired only
        deleted = cache.clear_expired()
        assert deleted == 1

        # One should remain
        remaining = len(list(cache_dir.glob("*.json")))
        assert remaining == 1

    def test_clear_empty_cache(self, tmp_path):
        """Test clearing empty cache."""
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"))
        deleted = cache.clear()
        assert deleted == 0


class TestCacheStatistics:
    """Test cache statistics."""

    def test_get_stats_empty(self, tmp_path):
        """Test statistics for empty cache."""
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"))

        stats = cache.get_stats()
        assert stats['total_entries'] == 0
        assert stats['cache_size_mb'] == 0.0
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['hit_rate'] == 0.0

    def test_get_stats_with_entries(self, tmp_path):
        """Test statistics with cache entries."""
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"))

        cache.set("model-1", "prompt1", 0.7, {'data': 'test1'})
        cache.set("model-2", "prompt2", 0.7, {'data': 'test2'})

        stats = cache.get_stats()
        assert stats['total_entries'] == 2
        assert stats['cache_size_mb'] > 0

    def test_hit_rate_calculation(self, tmp_path):
        """Test hit rate calculation."""
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"))

        # Set an entry
        cache.set("model-1", "prompt", 0.7, {'data': 'test'})

        # 2 hits
        cache.get("model-1", "prompt", 0.7)
        cache.get("model-1", "prompt", 0.7)

        # 1 miss
        cache.get("model-2", "other", 0.7)

        stats = cache.get_stats()
        assert stats['hits'] == 2
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 2.0 / 3.0  # 2 hits out of 3 total

    def test_repr(self, tmp_path):
        """Test string representation."""
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"))
        cache.set("model-1", "prompt", 0.7, {'data': 'test'})

        repr_str = repr(cache)
        assert 'ResponseCache' in repr_str
        assert 'entries=1' in repr_str


class TestCacheErrorHandling:
    """Test cache error handling."""

    def test_corrupted_cache_file(self, tmp_path):
        """Test handling of corrupted cache files."""
        cache_dir = tmp_path / "cache"
        cache = ResponseCache(cache_dir=str(cache_dir))

        # Set a valid entry
        cache.set("model-1", "prompt", 0.7, {'data': 'test'})

        # Corrupt the file
        cache_files = list(cache_dir.glob("*.json"))
        with open(cache_files[0], 'w') as f:
            f.write("corrupted data {{{")

        # Should handle gracefully and return None
        result = cache.get("model-1", "prompt", 0.7)
        assert result is None

        # Corrupted file should be deleted
        assert len(list(cache_dir.glob("*.json"))) == 0
