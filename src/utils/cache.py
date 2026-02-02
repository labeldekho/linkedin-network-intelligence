"""
LLM Response Caching

Content-addressed caching for LLM responses using diskcache.
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class CacheEntry(BaseModel):
    """A cached LLM response."""
    key: str
    content: str
    provider: str
    model: str
    created_at: datetime
    expires_at: datetime
    metadata: dict = {}


class LLMCache:
    """Content-addressed LLM response cache.

    Uses SHA-256 hash of prompt + system + model as cache key.
    """

    def __init__(
        self,
        cache_path: str = ".cache/llm_responses.db",
        ttl_days: int = 7,
        max_size_mb: int = 500,
        enabled: bool = True,
    ):
        """Initialize cache.

        Args:
            cache_path: Path to cache database
            ttl_days: Time-to-live for cache entries
            max_size_mb: Maximum cache size in MB (0 = unlimited)
            enabled: Whether caching is enabled
        """
        self.enabled = enabled
        self.ttl_days = ttl_days
        self.cache_path = Path(cache_path)
        self.max_size_bytes = max_size_mb * 1024 * 1024 if max_size_mb > 0 else None

        self._cache = None

        if self.enabled:
            self._init_cache()

    def _init_cache(self) -> None:
        """Initialize the diskcache backend."""
        try:
            import diskcache

            # Create cache directory
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)

            self._cache = diskcache.Cache(
                str(self.cache_path),
                size_limit=self.max_size_bytes,
            )

            logger.debug(f"LLM cache initialized at {self.cache_path}")

        except ImportError:
            logger.warning("diskcache not installed, caching disabled")
            self.enabled = False

    def _generate_key(
        self,
        prompt: str,
        system: Optional[str],
        model: str,
        provider: str,
        schema: Optional[dict] = None,
    ) -> str:
        """Generate content-addressed cache key."""
        key_data = {
            "prompt": prompt,
            "system": system or "",
            "model": model,
            "provider": provider,
            "schema": schema,
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get(
        self,
        prompt: str,
        system: Optional[str],
        model: str,
        provider: str,
        schema: Optional[dict] = None,
    ) -> Optional[str]:
        """Retrieve cached response.

        Args:
            prompt: The user prompt
            system: System prompt
            model: Model name
            provider: Provider name
            schema: JSON schema for structured output

        Returns:
            Cached content or None if not found/expired
        """
        if not self.enabled or self._cache is None:
            return None

        key = self._generate_key(prompt, system, model, provider, schema)

        try:
            entry_data = self._cache.get(key)
            if entry_data is None:
                return None

            entry = CacheEntry.model_validate_json(entry_data)

            # Check expiration
            if datetime.now() > entry.expires_at:
                self._cache.delete(key)
                return None

            logger.debug(f"Cache hit for key {key[:16]}...")
            return entry.content

        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None

    def set(
        self,
        prompt: str,
        system: Optional[str],
        model: str,
        provider: str,
        content: str,
        schema: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Store response in cache.

        Args:
            prompt: The user prompt
            system: System prompt
            model: Model name
            provider: Provider name
            content: Response content to cache
            schema: JSON schema for structured output
            metadata: Optional metadata to store
        """
        if not self.enabled or self._cache is None:
            return

        key = self._generate_key(prompt, system, model, provider, schema)

        entry = CacheEntry(
            key=key,
            content=content,
            provider=provider,
            model=model,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=self.ttl_days),
            metadata=metadata or {},
        )

        try:
            self._cache.set(key, entry.model_dump_json())
            logger.debug(f"Cached response for key {key[:16]}...")

        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def invalidate(
        self,
        prompt: str,
        system: Optional[str],
        model: str,
        provider: str,
        schema: Optional[dict] = None,
    ) -> bool:
        """Remove a specific entry from cache.

        Returns:
            True if entry was removed, False if not found
        """
        if not self.enabled or self._cache is None:
            return False

        key = self._generate_key(prompt, system, model, provider, schema)

        try:
            return self._cache.delete(key)
        except Exception as e:
            logger.warning(f"Cache invalidation error: {e}")
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        if not self.enabled or self._cache is None:
            return

        try:
            self._cache.clear()
            logger.info("Cache cleared")
        except Exception as e:
            logger.warning(f"Cache clear error: {e}")

    def stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        if not self.enabled or self._cache is None:
            return {"enabled": False}

        try:
            return {
                "enabled": True,
                "path": str(self.cache_path),
                "size_bytes": self._cache.volume(),
                "count": len(self._cache),
                "ttl_days": self.ttl_days,
            }
        except Exception as e:
            logger.warning(f"Cache stats error: {e}")
            return {"enabled": True, "error": str(e)}

    def close(self) -> None:
        """Close the cache."""
        if self._cache is not None:
            try:
                self._cache.close()
            except Exception as e:
                logger.warning(f"Cache close error: {e}")
