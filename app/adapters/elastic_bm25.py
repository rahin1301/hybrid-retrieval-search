"""Elasticsearch BM25 retrieval.

Lexical retrieval is the sparse half of the hybrid system. It is strong on
exact terminology — model names, acronyms, author-coined phrases — which is
exactly where dense embeddings tend to be weakest.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import ApiError, TransportError

from app.domain.errors import RetrievalError
from app.domain.models import Document, ScoredDoc

log = logging.getLogger(__name__)

_ES_ERRORS = (ApiError, TransportError)

# The English analyzer applies stemming and stopword removal, which is what
# makes BM25 competitive on natural-language queries rather than exact-match.
INDEX_SETTINGS: dict = {
    "number_of_shards": 1,
    "number_of_replicas": 0,
    "analysis": {"analyzer": {"default": {"type": "english"}}},
    "similarity": {"default": {"type": "BM25", "k1": 1.2, "b": 0.75}},
}

INDEX_MAPPINGS: dict = {
    "properties": {
        "title": {"type": "text", "analyzer": "english"},
        "abstract": {"type": "text", "analyzer": "english"},
        "categories": {"type": "keyword"},
        "published": {"type": "keyword"},
        "url": {"type": "keyword", "index": False},
    }
}


def create_client(url: str, *, request_timeout: int = 30) -> Elasticsearch:
    return Elasticsearch(url, request_timeout=request_timeout, retry_on_timeout=True)


class ElasticIndexer:
    """Write path: index lifecycle and bulk loading.

    Kept separate from the retriever so the serving path holds no methods
    capable of deleting or mutating the index.
    """

    def __init__(self, client: Elasticsearch, index_name: str) -> None:
        self._client = client
        self._index = index_name

    def recreate_index(self) -> None:
        """Drop and recreate the index so a rebuild is idempotent."""
        if self._client.indices.exists(index=self._index):
            log.info("Deleting existing index %s", self._index)
            self._client.indices.delete(index=self._index)
        self._client.indices.create(
            index=self._index, settings=INDEX_SETTINGS, mappings=INDEX_MAPPINGS
        )
        log.info("Created index %s", self._index)

    def index_documents(
        self, documents: Iterable[Document], *, chunk_size: int = 500
    ) -> int:
        """Bulk-index documents, returning the number successfully written."""
        actions = (
            {
                "_index": self._index,
                "_id": doc.id,
                "_source": {
                    "title": doc.title,
                    "abstract": doc.abstract,
                    "categories": list(doc.categories),
                    "published": doc.published,
                    "url": doc.url,
                },
            }
            for doc in documents
        )
        succeeded, errors = helpers.bulk(
            self._client, actions, chunk_size=chunk_size, raise_on_error=False
        )
        if errors:
            log.warning("%d documents failed to index", len(errors))
        self._client.indices.refresh(index=self._index)
        return succeeded


class ElasticBM25Retriever:
    """Read path: BM25 candidate retrieval."""

    def __init__(self, client: Elasticsearch, index_name: str) -> None:
        self._client = client
        self._index = index_name

    @property
    def name(self) -> str:
        return "bm25"

    def search(self, query: str, k: int) -> list[ScoredDoc]:
        """Return the top ``k`` BM25 matches.

        Raises:
            RetrievalError: If Elasticsearch is unreachable or the query fails.
        """
        try:
            response = self._client.search(
                index=self._index,
                size=k,
                query={
                    "multi_match": {
                        "query": query,
                        # Title matches are a stronger relevance signal than
                        # body matches for short academic documents.
                        "fields": ["title^2", "abstract"],
                        "type": "best_fields",
                    }
                },
                source=False,
            )
        except _ES_ERRORS as exc:
            raise RetrievalError(f"BM25 retrieval failed: {exc}") from exc

        return [
            ScoredDoc(doc_id=hit["_id"], score=float(hit["_score"]))
            for hit in response["hits"]["hits"]
        ]

    def health(self) -> bool:
        try:
            return bool(self._client.indices.exists(index=self._index))
        except _ES_ERRORS:
            return False
