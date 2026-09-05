"""Redis and no-op cache adapters.

Caching is an optimisation, never a correctness dependency: every Redis fault
is swallowed and logged so a cache outage degrades latency rather than
returning errors to callers.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence

import numpy as np
import redis

from app.domain.models import ScoredDoc

log = logging.getLogger(__name__)

_EMBEDDING_PREFIX = "emb:"
_RESULTS_PREFIX = "res:"


def cache_key(*parts: str) -> str:
    """Stable hash key. Hashing keeps keys bounded regardless of query length."""
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


class NullCache:
    """Cache that stores nothing — used when caching is disabled or unreachable."""

    def get_embedding(self, key: str) -> np.ndarray | None:
        return None

    def set_embedding(self, key: str, vector: np.ndarray) -> None:
        return None

    def get_results(self, key: str) -> list[ScoredDoc] | None:
        return None

    def set_results(self, key: str, results: Sequence[ScoredDoc]) -> None:
        return None


class RedisCache:
    """Redis-backed :class:`~app.ports.Cache`.

    Embeddings are stored as raw float32 bytes rather than JSON: a 768-dim
    vector is 3 KB packed versus ~15 KB as text, and decoding is a memcpy.
    """

    def __init__(self, client: redis.Redis, ttl_s: int = 3600) -> None:
        self._client = client
        self._ttl_s = ttl_s

    @classmethod
    def connect(cls, url: str, ttl_s: int = 3600) -> RedisCache | NullCache:
        """Connect, falling back to :class:`NullCache` if Redis is unavailable."""
        try:
            client = redis.from_url(url, socket_connect_timeout=2, socket_timeout=2)
            client.ping()
        except redis.RedisError as exc:
            log.warning("Redis unavailable at %s (%s); caching disabled", url, exc)
            return NullCache()
        log.info("Connected to Redis at %s", url)
        return cls(client, ttl_s)

    def get_embedding(self, key: str) -> np.ndarray | None:
        try:
            raw = self._client.get(_EMBEDDING_PREFIX + key)
        except redis.RedisError as exc:
            log.warning("Redis GET failed: %s", exc)
            return None
        if raw is None:
            return None
        return np.frombuffer(raw, dtype=np.float32)

    def set_embedding(self, key: str, vector: np.ndarray) -> None:
        try:
            self._client.setex(
                _EMBEDDING_PREFIX + key,
                self._ttl_s,
                np.ascontiguousarray(vector, dtype=np.float32).tobytes(),
            )
        except redis.RedisError as exc:
            log.warning("Redis SETEX failed: %s", exc)

    def get_results(self, key: str) -> list[ScoredDoc] | None:
        try:
            raw = self._client.get(_RESULTS_PREFIX + key)
        except redis.RedisError as exc:
            log.warning("Redis GET failed: %s", exc)
            return None
        if raw is None:
            return None
        try:
            return [ScoredDoc(**item) for item in json.loads(raw)]
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            log.warning("Discarding corrupt cache entry %s: %s", key, exc)
            return None

    def set_results(self, key: str, results: Sequence[ScoredDoc]) -> None:
        try:
            payload = json.dumps([doc.model_dump() for doc in results])
            self._client.setex(_RESULTS_PREFIX + key, self._ttl_s, payload)
        except redis.RedisError as exc:
            log.warning("Redis SETEX failed: %s", exc)
