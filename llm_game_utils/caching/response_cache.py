"""Response caching to avoid redundant API calls during development.

This module provides the ResponseCache class to cache LLM responses to disk,
helping to save money and time during development by avoiding repeated calls
with identical parameters.
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from ..exceptions import CacheError

logger = logging.getLogger(__name__)


class ResponseCache:
    """Cache LLM responses to disk to avoid redundant API calls.

    This is especially useful during development when you're iterating on
    prompts or testing code. Responses are cached based on a hash of the
    model, prompt, and parameters.

    Example:
        ```python
        # Create a cache
        cache = ResponseCache(
            cache_dir=".llm_cache",
            enabled=True,
            ttl_hours=24  # Cache expires after 24 hours
        )

        # Check cache before making a call
        cached = cache.get(
            model_id="openai/gpt-4-turbo",
            prompt="What is 2+2?",
            temperature=0.7
        )

        if cached:
            print("Using cached response!")
            response_data = cached
        else:
            # Make API call
            response = client.query(...)
            # Store in cache
            cache.set(
                model_id="openai/gpt-4-turbo",
                prompt="What is 2+2?",
                temperature=0.7,
                response=response
            )
        ```
    """

    def __init__(
        self,
        cache_dir: str = ".llm_cache",
        enabled: bool = True,
        ttl_hours: Optional[int] = 24
    ):
        """Initialize the response cache.

        Args:
            cache_dir: Directory to store cached responses
            enabled: Whether caching is active (can disable without deleting cache)
            ttl_hours: Cache time-to-live in hours. None means cache never expires.

        Raises:
            CacheError: If cache directory cannot be created
        """
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self.ttl_hours = ttl_hours

        # Statistics
        self.hits = 0
        self.misses = 0

        if self.enabled:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"ResponseCache initialized: {self.cache_dir} (TTL: {ttl_hours}h)")
            except Exception as e:
                raise CacheError(f"Failed to create cache directory: {e}")
        else:
            logger.info("ResponseCache disabled")

    def get_cache_key(
        self,
        model_id: str,
        prompt: str,
        temperature: float,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate a unique cache key for this query.

        Uses SHA256 hash of model + prompt + parameters to create a unique
        but deterministic key for each unique query.

        Args:
            model_id: Model identifier
            prompt: User prompt
            temperature: Sampling temperature
            system_prompt: Optional system prompt
            max_tokens: Optional max tokens

        Returns:
            Hex string hash to use as cache key
        """
        # Create a string representation of all parameters that affect the response
        key_components = [
            f"model:{model_id}",
            f"prompt:{prompt}",
            f"temp:{temperature}",
            f"system:{system_prompt if system_prompt else ''}",
            f"max_tokens:{max_tokens if max_tokens else ''}"
        ]

        key_string = "|".join(key_components)

        # Generate hash
        hash_obj = hashlib.sha256(key_string.encode('utf-8'))
        return hash_obj.hexdigest()

    def get(
        self,
        model_id: str,
        prompt: str,
        temperature: float,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached response if it exists and hasn't expired.

        Args:
            model_id: Model identifier
            prompt: User prompt
            temperature: Sampling temperature
            system_prompt: Optional system prompt
            max_tokens: Optional max tokens

        Returns:
            Cached response data (dict) or None if not found/expired
        """
        if not self.enabled:
            return None

        cache_key = self.get_cache_key(model_id, prompt, temperature, system_prompt, max_tokens)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            self.misses += 1
            logger.debug(f"Cache miss: {cache_key[:16]}...")
            return None

        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)

            # Check if expired
            if self.ttl_hours is not None:
                cached_time = datetime.fromisoformat(cached_data['cached_at'])
                expiry_time = cached_time + timedelta(hours=self.ttl_hours)

                if datetime.now() > expiry_time:
                    logger.debug(f"Cache expired: {cache_key[:16]}...")
                    self.misses += 1
                    # Optionally delete expired cache
                    cache_file.unlink()
                    return None

            self.hits += 1
            logger.debug(f"Cache hit: {cache_key[:16]}...")
            return cached_data['response']

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Invalid cache file {cache_file}: {e}")
            self.misses += 1
            # Delete corrupted cache file
            try:
                cache_file.unlink()
            except Exception as delete_error:
                # Best-effort cleanup: log but do not fail if deletion of corrupted cache file fails
                logger.debug(f"Failed to delete corrupted cache file {cache_file}: {delete_error}")
            return None

        except Exception as e:
            logger.error(f"Error reading cache: {e}")
            self.misses += 1
            return None

    def set(
        self,
        model_id: str,
        prompt: str,
        temperature: float,
        response: Any,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> None:
        """Store a response in the cache.

        Args:
            model_id: Model identifier
            prompt: User prompt
            temperature: Sampling temperature
            response: Response to cache (LLMResponse object or dict)
            system_prompt: Optional system prompt
            max_tokens: Optional max tokens

        Raises:
            CacheError: If cache write fails
        """
        if not self.enabled:
            return

        cache_key = self.get_cache_key(model_id, prompt, temperature, system_prompt, max_tokens)
        cache_file = self.cache_dir / f"{cache_key}.json"

        # Convert LLMResponse to dict if needed
        if hasattr(response, '__dict__'):
            # It's an object, convert to dict
            response_dict = {}
            for key, value in response.__dict__.items():
                # Handle datetime objects
                if isinstance(value, datetime):
                    response_dict[key] = value.isoformat()
                else:
                    response_dict[key] = value
        elif isinstance(response, dict):
            response_dict = response
        else:
            logger.warning(f"Unsupported response type for caching: {type(response)}")
            return

        cache_data = {
            'cache_key': cache_key,
            'cached_at': datetime.now().isoformat(),
            'model_id': model_id,
            'prompt': prompt[:200],  # Store truncated prompt for reference
            'temperature': temperature,
            'response': response_dict
        }

        try:
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)

            logger.debug(f"Cached response: {cache_key[:16]}...")

        except Exception as e:
            raise CacheError(f"Failed to write cache: {e}")

    def clear(self) -> int:
        """Delete all cached responses.

        Returns:
            Number of cache files deleted
        """
        if not self.cache_dir.exists():
            return 0

        deleted = 0
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    cache_file.unlink()
                    deleted += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {cache_file}: {e}")

            logger.info(f"Cleared {deleted} cache entries")
            self.hits = 0
            self.misses = 0
            return deleted

        except Exception as e:
            raise CacheError(f"Failed to clear cache: {e}")

    def clear_expired(self) -> int:
        """Delete only expired cache entries.

        Returns:
            Number of expired cache files deleted
        """
        if not self.cache_dir.exists() or self.ttl_hours is None:
            return 0

        deleted = 0
        now = datetime.now()

        try:
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)

                    cached_time = datetime.fromisoformat(cached_data['cached_at'])
                    expiry_time = cached_time + timedelta(hours=self.ttl_hours)

                    if now > expiry_time:
                        cache_file.unlink()
                        deleted += 1

                except Exception as e:
                    logger.warning(f"Error checking {cache_file}: {e}")
                    # Delete corrupted files
                    try:
                        cache_file.unlink()
                        deleted += 1
                    except Exception as delete_error:
                        # Best-effort cleanup: log but do not fail if deletion fails
                        logger.debug(f"Failed to delete corrupted cache file {cache_file}: {delete_error}")

            logger.info(f"Cleared {deleted} expired cache entries")
            return deleted

        except Exception as e:
            raise CacheError(f"Failed to clear expired cache: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics:
            {
                "total_entries": int,
                "cache_size_mb": float,
                "hit_rate": float,  # 0.0 to 1.0
                "hits": int,
                "misses": int,
                "oldest_entry": datetime or None,
                "newest_entry": datetime or None
            }
        """
        stats = {
            "total_entries": 0,
            "cache_size_mb": 0.0,
            "hit_rate": 0.0,
            "hits": self.hits,
            "misses": self.misses,
            "oldest_entry": None,
            "newest_entry": None
        }

        if not self.cache_dir.exists():
            return stats

        try:
            total_size = 0
            oldest = None
            newest = None

            for cache_file in self.cache_dir.glob("*.json"):
                stats["total_entries"] += 1
                total_size += cache_file.stat().st_size

                try:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                    cached_time = datetime.fromisoformat(cached_data['cached_at'])

                    if oldest is None or cached_time < oldest:
                        oldest = cached_time
                    if newest is None or cached_time > newest:
                        newest = cached_time

                except Exception as e:
                    # Skip corrupted or unreadable cache files but log for debugging
                    logger.debug(f"Skipping cache file {cache_file} due to error while reading stats: {e}")

            stats["cache_size_mb"] = total_size / (1024 * 1024)
            stats["oldest_entry"] = oldest
            stats["newest_entry"] = newest

            # Calculate hit rate
            total_requests = self.hits + self.misses
            if total_requests > 0:
                stats["hit_rate"] = self.hits / total_requests

        except Exception as e:
            logger.error(f"Error calculating cache stats: {e}")

        return stats

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        stats = self.get_stats()
        return (
            f"ResponseCache(dir={self.cache_dir}, enabled={self.enabled}, "
            f"entries={stats['total_entries']}, size={stats['cache_size_mb']:.2f}MB, "
            f"hit_rate={stats['hit_rate']:.2%})"
        )
