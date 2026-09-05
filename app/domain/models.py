"""Core domain types.

These are deliberately free of any transport, storage or model-runtime
concerns so they can be constructed in tests without infrastructure.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class SearchMode(StrEnum):
    """Retrieval strategies the service can execute.

    The four modes correspond one-to-one with the systems compared in the
    evaluation harness, so adding a mode automatically extends the benchmark.
    """

    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


class Document(BaseModel):
    """A corpus document (an arXiv paper)."""

    model_config = {"frozen": True}

    id: str
    title: str
    abstract: str
    authors: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    published: str = ""
    url: str = ""

    def to_text(self) -> str:
        """Concatenated text used for embedding and cross-encoder scoring."""
        return f"{self.title}\n\n{self.abstract}"


class ScoredDoc(BaseModel):
    """A document id with the score assigned by one retrieval stage."""

    model_config = {"frozen": True}

    doc_id: str
    score: float


class SearchHit(BaseModel):
    """A fully hydrated result returned to API clients."""

    doc_id: str
    score: float
    title: str
    abstract: str
    url: str = ""
    categories: tuple[str, ...] = ()


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    took_ms: float
    cached: bool = False
    hits: list[SearchHit] = Field(default_factory=list)
