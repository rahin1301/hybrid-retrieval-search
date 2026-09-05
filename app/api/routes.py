"""HTTP routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.dependencies import get_search_service
from app.domain.errors import SearchError
from app.domain.models import SearchMode, SearchResponse
from app.service import SearchService

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", tags=["ops"])
def health(service: SearchService = Depends(get_search_service)) -> dict:
    """Liveness and dependency check."""
    return {
        "status": "ok",
        "modes": [mode.value for mode in service.supported_modes],
    }


@router.get("/search", response_model=SearchResponse, tags=["search"])
def search(
    q: str = Query(..., min_length=1, max_length=512, description="Search query"),
    mode: SearchMode = Query(
        SearchMode.HYBRID_RERANK, description="Retrieval strategy"
    ),
    k: int = Query(10, ge=1, le=100, description="Number of results"),
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    """Run a search and return ranked results.

    Raises:
        HTTPException: 422 for an invalid query, 503 if a backend is unavailable.
    """
    try:
        return service.search(q, mode, k)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except SearchError as exc:
        log.exception("Search failed for query=%r mode=%s", q, mode)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
