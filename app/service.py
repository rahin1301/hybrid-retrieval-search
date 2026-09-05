"""Retrieval orchestration.

The pipeline per mode:

* ``bm25``           — Elasticsearch only.
* ``dense``          — FAISS only.
* ``hybrid``         — both retrievers to ``candidate_k``, fused with RRF.
* ``hybrid_rerank``  — hybrid, then a cross-encoder rescores the top
  ``rerank_depth`` candidates.

Depends only on the port protocols, so every mode is testable with in-memory
fakes.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

from app.adapters.redis_cache import NullCache, cache_key
from app.config import Settings
from app.domain.errors import SearchError
from app.domain.fusion import reciprocal_rank_fusion
from app.domain.models import Document, ScoredDoc, SearchHit, SearchMode, SearchResponse
from app.ports import Cache, DocumentStore, Reranker, Retriever

log = logging.getLogger(__name__)


class SearchService:
    def __init__(
        self,
        *,
        bm25: Retriever,
        dense: Retriever,
        store: DocumentStore,
        settings: Settings,
        reranker: Reranker | None = None,
        cache: Cache | None = None,
    ) -> None:
        self._bm25 = bm25
        self._dense = dense
        self._store = store
        self._settings = settings
        self._reranker = reranker
        self._cache = cache or NullCache()

    @property
    def store(self) -> DocumentStore:
        """The corpus behind this service, for offline tooling that hydrates ids."""
        return self._store

    @property
    def supported_modes(self) -> list[SearchMode]:
        """Modes this instance can serve, given whether a reranker was wired."""
        modes = [SearchMode.BM25, SearchMode.DENSE, SearchMode.HYBRID]
        if self._reranker is not None:
            modes.append(SearchMode.HYBRID_RERANK)
        return modes

    def search(self, query: str, mode: SearchMode, k: int = 10) -> SearchResponse:
        """Run a query and return hydrated, ranked hits."""
        started = time.perf_counter()
        ranked, cached = self._ranked_with_cache(query, mode, k)
        hits = self._hydrate(ranked)
        return SearchResponse(
            query=query,
            mode=mode,
            took_ms=round((time.perf_counter() - started) * 1000, 2),
            cached=cached,
            hits=hits,
        )

    def rank(self, query: str, mode: SearchMode, k: int = 10) -> list[ScoredDoc]:
        """Ranked document ids without hydration — the entry point used by evaluation."""
        ranked, _ = self._ranked_with_cache(query, mode, k)
        return ranked

    # -- internals --------------------------------------------------------

    def _ranked_with_cache(
        self, query: str, mode: SearchMode, k: int
    ) -> tuple[list[ScoredDoc], bool]:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        key = cache_key(mode.value, str(k), query)
        if (cached := self._cache.get_results(key)) is not None:
            return cached, True

        ranked = self._execute(query, mode, k)
        self._cache.set_results(key, ranked)
        return ranked, False

    def _execute(self, query: str, mode: SearchMode, k: int) -> list[ScoredDoc]:
        match mode:
            case SearchMode.BM25:
                return self._bm25.search(query, k)
            case SearchMode.DENSE:
                return self._dense.search(query, k)
            case SearchMode.HYBRID:
                return self._fuse(query)[:k]
            case SearchMode.HYBRID_RERANK:
                return self._rerank(query, k)
            case _:  # pragma: no cover - StrEnum makes this unreachable
                raise SearchError(f"unsupported mode: {mode}")

    def _fuse(self, query: str) -> list[ScoredDoc]:
        depth = self._settings.candidate_k
        bm25_hits = self._bm25.search(query, depth)
        dense_hits = self._dense.search(query, depth)
        return reciprocal_rank_fusion(
            [[hit.doc_id for hit in bm25_hits], [hit.doc_id for hit in dense_hits]],
            k=self._settings.rrf_k,
        )

    def _rerank(self, query: str, k: int) -> list[ScoredDoc]:
        if self._reranker is None:
            raise SearchError(
                "hybrid_rerank requested but no reranker is configured"
            )
        candidates = self._fuse(query)[: self._settings.rerank_depth]
        documents = self._store.get_many(hit.doc_id for hit in candidates)
        if not documents:
            return []
        return self._reranker.rerank(query, documents, top_n=k)

    def _hydrate(self, ranked: Sequence[ScoredDoc]) -> list[SearchHit]:
        """Attach document fields to ranked ids, dropping ids missing from the store."""
        hits: list[SearchHit] = []
        for scored in ranked:
            document: Document | None = self._store.get(scored.doc_id)
            if document is None:
                log.warning("Ranked doc %s is absent from the store", scored.doc_id)
                continue
            hits.append(
                SearchHit(
                    doc_id=document.id,
                    score=scored.score,
                    title=document.title,
                    abstract=document.abstract,
                    url=document.url,
                    categories=document.categories,
                )
            )
        return hits
