import pytest

from app.adapters.corpus_store import InMemoryDocumentStore
from app.config import Settings
from app.domain.errors import SearchError
from app.domain.models import SearchMode
from app.service import SearchService
from tests.fakes import FakeCache, FakeReranker, FakeRetriever, make_documents

BM25_RANKING = ["d1", "d2", "d3", "d4"]
DENSE_RANKING = ["d3", "d5", "d1", "d6"]
ALL_IDS = ["d1", "d2", "d3", "d4", "d5", "d6"]


@pytest.fixture
def settings() -> Settings:
    return Settings(candidate_k=10, rerank_depth=3, rrf_k=60, cache_enabled=False)


@pytest.fixture
def store() -> InMemoryDocumentStore:
    return InMemoryDocumentStore(make_documents(ALL_IDS))


def build_service(settings, store, *, reranker=None, cache=None) -> SearchService:
    return SearchService(
        bm25=FakeRetriever("bm25", BM25_RANKING),
        dense=FakeRetriever("dense", DENSE_RANKING),
        store=store,
        settings=settings,
        reranker=reranker,
        cache=cache,
    )


def test_bm25_mode_returns_lexical_ranking(settings, store):
    service = build_service(settings, store)
    response = service.search("neural retrieval", SearchMode.BM25, k=3)
    assert [hit.doc_id for hit in response.hits] == ["d1", "d2", "d3"]


def test_dense_mode_returns_semantic_ranking(settings, store):
    service = build_service(settings, store)
    response = service.search("neural retrieval", SearchMode.DENSE, k=2)
    assert [hit.doc_id for hit in response.hits] == ["d3", "d5"]


def test_hybrid_promotes_documents_found_by_both_retrievers(settings, store):
    service = build_service(settings, store)
    ranked = service.rank("neural retrieval", SearchMode.HYBRID, k=6)
    ids = [doc.doc_id for doc in ranked]

    # d1 (rank 1 + rank 3) and d3 (rank 3 + rank 1) appear in both lists, so
    # they must outrank d2, which only BM25 returned at rank 2.
    assert set(ids[:2]) == {"d1", "d3"}
    assert ids.index("d2") > 1


def test_hybrid_queries_both_retrievers_at_candidate_depth(settings, store):
    bm25 = FakeRetriever("bm25", BM25_RANKING)
    dense = FakeRetriever("dense", DENSE_RANKING)
    service = SearchService(
        bm25=bm25, dense=dense, store=store, settings=settings, cache=None
    )

    service.rank("q", SearchMode.HYBRID, k=2)

    assert bm25.calls == [("q", settings.candidate_k)]
    assert dense.calls == [("q", settings.candidate_k)]


def test_rerank_receives_only_the_configured_depth(settings, store):
    reranker = FakeReranker()
    service = build_service(settings, store, reranker=reranker)

    service.rank("q", SearchMode.HYBRID_RERANK, k=2)

    query, candidate_ids, top_n = reranker.received[0]
    assert query == "q"
    assert len(candidate_ids) == settings.rerank_depth
    assert top_n == 2


def test_rerank_reorders_the_fused_candidates(settings, store):
    service = build_service(settings, store, reranker=FakeReranker())

    fused = [doc.doc_id for doc in service.rank("q", SearchMode.HYBRID, k=3)]
    reranked = [doc.doc_id for doc in service.rank("q", SearchMode.HYBRID_RERANK, k=3)]

    # FakeReranker reverses its input, so the orders must differ.
    assert reranked == list(reversed(fused))


def test_rerank_mode_without_a_reranker_is_rejected(settings, store):
    service = build_service(settings, store, reranker=None)
    with pytest.raises(SearchError, match="no reranker"):
        service.rank("q", SearchMode.HYBRID_RERANK, k=2)


def test_supported_modes_reflect_reranker_availability(settings, store):
    assert SearchMode.HYBRID_RERANK not in build_service(settings, store).supported_modes
    with_reranker = build_service(settings, store, reranker=FakeReranker())
    assert SearchMode.HYBRID_RERANK in with_reranker.supported_modes


def test_second_identical_query_is_served_from_cache(settings, store):
    cache = FakeCache()
    service = build_service(settings, store, cache=cache)

    first = service.search("q", SearchMode.BM25, k=2)
    second = service.search("q", SearchMode.BM25, k=2)

    assert first.cached is False
    assert second.cached is True
    assert [h.doc_id for h in first.hits] == [h.doc_id for h in second.hits]


def test_cache_key_separates_modes_and_depths(settings, store):
    cache = FakeCache()
    service = build_service(settings, store, cache=cache)

    service.search("q", SearchMode.BM25, k=2)
    service.search("q", SearchMode.DENSE, k=2)
    service.search("q", SearchMode.BM25, k=5)

    assert cache.misses == 3


def test_hits_missing_from_the_store_are_dropped(settings):
    partial_store = InMemoryDocumentStore(make_documents(["d1"]))
    service = build_service(settings, partial_store)

    response = service.search("q", SearchMode.BM25, k=3)

    assert [hit.doc_id for hit in response.hits] == ["d1"]


@pytest.mark.parametrize("query", ["", "   "])
def test_empty_query_is_rejected(settings, store, query):
    service = build_service(settings, store)
    with pytest.raises(ValueError, match="empty"):
        service.search(query, SearchMode.BM25, k=3)


def test_non_positive_k_is_rejected(settings, store):
    service = build_service(settings, store)
    with pytest.raises(ValueError, match="positive"):
        service.search("q", SearchMode.BM25, k=0)


def test_response_reports_latency(settings, store):
    service = build_service(settings, store)
    response = service.search("q", SearchMode.BM25, k=1)
    assert response.took_ms >= 0.0
    assert response.mode is SearchMode.BM25
