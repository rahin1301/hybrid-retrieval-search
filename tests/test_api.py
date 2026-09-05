"""Route-level tests.

The real service is replaced through FastAPI's dependency overrides, so these
exercise request validation, serialisation and error mapping without
Elasticsearch, FAISS, Redis or any model load. ``TestClient`` is deliberately
not used as a context manager: that would run the lifespan and build the real
stack.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_search_service
from app.api.main import create_app
from app.domain.errors import RetrievalError
from app.domain.models import SearchHit, SearchMode, SearchResponse


class StubService:
    def __init__(self, *, error: Exception | None = None) -> None:
        self._error = error
        self.calls: list[tuple[str, SearchMode, int]] = []

    @property
    def supported_modes(self) -> list[SearchMode]:
        return list(SearchMode)

    def search(self, query: str, mode: SearchMode, k: int = 10) -> SearchResponse:
        self.calls.append((query, mode, k))
        if self._error is not None:
            raise self._error
        return SearchResponse(
            query=query,
            mode=mode,
            took_ms=1.5,
            cached=False,
            hits=[
                SearchHit(
                    doc_id="2609.00001v1",
                    score=9.5,
                    title="A Paper",
                    abstract="An abstract.",
                    url="http://arxiv.org/abs/2609.00001v1",
                    categories=("cs.IR",),
                )
            ][:k],
        )


def make_client(service: StubService) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_search_service] = lambda: service
    return TestClient(app)


def test_health_lists_supported_modes():
    response = make_client(StubService()).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "hybrid_rerank" in response.json()["modes"]


def test_search_returns_hits():
    response = make_client(StubService()).get("/search", params={"q": "retrieval"})
    assert response.status_code == 200

    body = response.json()
    assert body["query"] == "retrieval"
    assert body["hits"][0]["doc_id"] == "2609.00001v1"
    assert body["took_ms"] >= 0


def test_search_defaults_to_the_full_pipeline():
    service = StubService()
    make_client(service).get("/search", params={"q": "retrieval"})
    assert service.calls == [("retrieval", SearchMode.HYBRID_RERANK, 10)]


@pytest.mark.parametrize("mode", [m.value for m in SearchMode])
def test_every_mode_is_accepted(mode):
    service = StubService()
    response = make_client(service).get("/search", params={"q": "x", "mode": mode})
    assert response.status_code == 200
    assert service.calls[0][1] == SearchMode(mode)


def test_unknown_mode_is_rejected():
    response = make_client(StubService()).get(
        "/search", params={"q": "x", "mode": "magic"}
    )
    assert response.status_code == 422


@pytest.mark.parametrize("params", [{}, {"q": ""}, {"q": "x", "k": 0}, {"q": "x", "k": 101}])
def test_invalid_parameters_are_rejected(params):
    assert make_client(StubService()).get("/search", params=params).status_code == 422


def test_backend_failure_maps_to_503():
    service = StubService(error=RetrievalError("elasticsearch is down"))
    response = make_client(service).get("/search", params={"q": "x"})
    assert response.status_code == 503
    assert "elasticsearch is down" in response.json()["detail"]


def test_service_level_value_error_maps_to_422():
    service = StubService(error=ValueError("query must not be empty"))
    response = make_client(service).get("/search", params={"q": "x"})
    assert response.status_code == 422
