"""In-memory port implementations used across the test suite."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from app.domain.models import Document, ScoredDoc


class FakeRetriever:
    """Returns a fixed ranking, recording the depth it was asked for."""

    def __init__(self, name: str, ranking: Sequence[str]) -> None:
        self._name = name
        self._ranking = list(ranking)
        self.calls: list[tuple[str, int]] = []

    @property
    def name(self) -> str:
        return self._name

    def search(self, query: str, k: int) -> list[ScoredDoc]:
        self.calls.append((query, k))
        return [
            # Descending scores so the ordering is meaningful if inspected.
            ScoredDoc(doc_id=doc_id, score=1.0 / rank)
            for rank, doc_id in enumerate(self._ranking[:k], start=1)
        ]


class FakeReranker:
    """Reverses the candidate order, so reranking is observable in assertions."""

    def __init__(self) -> None:
        self.received: list[tuple[str, list[str], int]] = []

    def rerank(
        self, query: str, documents: Sequence[Document], top_n: int
    ) -> list[ScoredDoc]:
        self.received.append((query, [doc.id for doc in documents], top_n))
        reversed_docs = list(reversed(documents))
        return [
            ScoredDoc(doc_id=doc.id, score=float(len(reversed_docs) - i))
            for i, doc in enumerate(reversed_docs)
        ][:top_n]


class FakeCache:
    """Dict-backed cache that counts hits and misses."""

    def __init__(self) -> None:
        self.embeddings: dict[str, np.ndarray] = {}
        self.results: dict[str, list[ScoredDoc]] = {}
        self.hits = 0
        self.misses = 0

    def get_embedding(self, key: str) -> np.ndarray | None:
        return self.embeddings.get(key)

    def set_embedding(self, key: str, vector: np.ndarray) -> None:
        self.embeddings[key] = vector

    def get_results(self, key: str) -> list[ScoredDoc] | None:
        if key in self.results:
            self.hits += 1
            return self.results[key]
        self.misses += 1
        return None

    def set_results(self, key: str, results: Sequence[ScoredDoc]) -> None:
        self.results[key] = list(results)


def make_documents(doc_ids: Sequence[str]) -> list[Document]:
    return [
        Document(id=doc_id, title=f"Title {doc_id}", abstract=f"Abstract for {doc_id}")
        for doc_id in doc_ids
    ]
