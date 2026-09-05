"""Domain-level errors.

Adapters wrap infrastructure exceptions in these types so the service and API
layers never need to import Elasticsearch, FAISS or Redis exception classes.
"""

from __future__ import annotations


class SearchError(Exception):
    """Base class for search failures."""


class RetrievalError(SearchError):
    """A retriever failed to return candidates (backend down, index missing)."""


class IndexNotReadyError(SearchError):
    """A required index has not been built yet."""
