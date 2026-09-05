"""Dense text encoders.

``SentenceTransformerEncoder`` owns the model; ``CachedTextEncoder`` decorates
any encoder with a query-embedding cache. Keeping the cache in a decorator
means index building (which should never consult a cache) and serving (which
should) use the same underlying encoder without conditional logic.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from app.adapters.redis_cache import cache_key
from app.ports import Cache, TextEncoder

log = logging.getLogger(__name__)


class SentenceTransformerEncoder:
    """Bi-encoder wrapper producing L2-normalised float32 embeddings.

    Normalising at encode time means inner product equals cosine similarity,
    which lets the FAISS index use ``IndexFlatIP`` directly.
    """

    def __init__(
        self,
        model_name: str,
        *,
        query_instruction: str = "",
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        log.info("Loading embedding model %s", model_name)
        self._model = SentenceTransformer(model_name, device=device)
        self._model_name = model_name
        self._query_instruction = query_instruction
        self._batch_size = batch_size

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def encode_query(self, query: str) -> np.ndarray:
        vector = self._model.encode(
            self._query_instruction + query,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vector.astype(np.float32, copy=False)

    def encode_documents(
        self, texts: Sequence[str], *, show_progress: bool = False
    ) -> np.ndarray:
        matrix = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return matrix.astype(np.float32, copy=False)


class CachedTextEncoder:
    """Decorates a :class:`~app.ports.TextEncoder` with a query-vector cache.

    Only queries are cached. Document embeddings are computed once during
    index building and persisted in the FAISS index, so caching them would
    duplicate storage for no gain.
    """

    def __init__(self, inner: TextEncoder, cache: Cache, model_name: str) -> None:
        self._inner = inner
        self._cache = cache
        self._model_name = model_name

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    def encode_query(self, query: str) -> np.ndarray:
        key = cache_key(self._model_name, query)
        cached = self._cache.get_embedding(key)
        if cached is not None and cached.size == self.dimension:
            return cached

        vector = self._inner.encode_query(query)
        self._cache.set_embedding(key, vector)
        return vector

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._inner.encode_documents(texts)
