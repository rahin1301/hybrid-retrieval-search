"""Composition root.

This is the only module that knows which concrete adapter satisfies each port.
Everything else depends on the protocols in :mod:`app.ports`.
"""

from __future__ import annotations

import logging

from fastapi import Request

from app.adapters.corpus_store import InMemoryDocumentStore
from app.adapters.cross_encoder import CrossEncoderReranker
from app.adapters.elastic_bm25 import ElasticBM25Retriever, create_client
from app.adapters.encoder import CachedTextEncoder, SentenceTransformerEncoder
from app.adapters.faiss_dense import FaissDenseRetriever
from app.adapters.redis_cache import NullCache, RedisCache
from app.config import Settings
from app.service import SearchService

log = logging.getLogger(__name__)


def build_search_service(settings: Settings, *, with_reranker: bool = True) -> SearchService:
    """Assemble the full retrieval stack.

    Args:
        settings: Resolved application settings.
        with_reranker: Load the cross-encoder. Disable to skip a ~90 MB model
            load when only first-stage retrieval is needed.

    Raises:
        IndexNotReadyError: If the dense index has not been built.
    """
    cache = (
        RedisCache.connect(settings.redis_url, settings.cache_ttl_s)
        if settings.cache_enabled
        else NullCache()
    )

    base_encoder = SentenceTransformerEncoder(
        settings.embed_model,
        query_instruction=settings.query_instruction,
        batch_size=settings.embed_batch_size,
    )
    encoder = CachedTextEncoder(base_encoder, cache, settings.embed_model)

    store = InMemoryDocumentStore.from_jsonl(settings.corpus_path)
    log.info("Document store holds %d documents", len(store))

    bm25 = ElasticBM25Retriever(create_client(settings.es_url), settings.index_name)
    dense = FaissDenseRetriever.load(
        settings.faiss_index_path, settings.faiss_ids_path, encoder
    )

    reranker = (
        CrossEncoderReranker(settings.rerank_model) if with_reranker else None
    )

    return SearchService(
        bm25=bm25,
        dense=dense,
        store=store,
        settings=settings,
        reranker=reranker,
        cache=cache,
    )


def get_search_service(request: Request) -> SearchService:
    """FastAPI dependency resolving the service built during startup."""
    return request.app.state.search_service


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings
