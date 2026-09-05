"""Port interfaces.

The service layer depends only on these protocols, never on Elasticsearch,
FAISS, Redis or sentence-transformers directly. Tests substitute in-memory
fakes, so the full retrieval pipeline is exercisable without infrastructure.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from app.domain.models import Document, ScoredDoc


@runtime_checkable
class Retriever(Protocol):
    """A first-stage retriever producing a ranked candidate list."""

    @property
    def name(self) -> str:
        """Short identifier used in logs and evaluation output."""
        ...

    def search(self, query: str, k: int) -> list[ScoredDoc]:
        """Return up to ``k`` documents ranked best-first."""
        ...


@runtime_checkable
class Reranker(Protocol):
    """A second-stage model that rescores a candidate set."""

    def rerank(
        self, query: str, documents: Sequence[Document], top_n: int
    ) -> list[ScoredDoc]:
        """Rescore ``documents`` against ``query``, returning the best ``top_n``."""
        ...


@runtime_checkable
class TextEncoder(Protocol):
    """Embeds text into a dense vector space."""

    @property
    def dimension(self) -> int:
        """Embedding dimensionality."""
        ...

    def encode_query(self, query: str) -> np.ndarray:
        """Embed a single query as a 1-D L2-normalised float32 vector."""
        ...

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed documents as a 2-D L2-normalised float32 matrix."""
        ...


@runtime_checkable
class DocumentStore(Protocol):
    """Read access to the corpus by document id."""

    def get(self, doc_id: str) -> Document | None:
        """Return a document, or None if the id is unknown."""
        ...

    def get_many(self, doc_ids: Iterable[str]) -> list[Document]:
        """Return documents for ``doc_ids``, skipping ids that are unknown."""
        ...

    def __len__(self) -> int:
        ...


@runtime_checkable
class Cache(Protocol):
    """Best-effort caching. Implementations must never raise on cache faults."""

    def get_embedding(self, key: str) -> np.ndarray | None:
        ...

    def set_embedding(self, key: str, vector: np.ndarray) -> None:
        ...

    def get_results(self, key: str) -> list[ScoredDoc] | None:
        ...

    def set_results(self, key: str, results: Sequence[ScoredDoc]) -> None:
        ...
