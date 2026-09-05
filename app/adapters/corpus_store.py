"""JSONL-backed corpus access.

The corpus is a few thousand abstracts (~15 MB), so it is held in memory: an
external key-value store would add an operational dependency and a network hop
per hit hydration for no measurable benefit at this scale.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

from app.domain.models import Document

log = logging.getLogger(__name__)


def load_documents(path: Path) -> list[Document]:
    """Parse a corpus JSONL file into documents.

    Malformed lines are logged and skipped rather than aborting the load, so a
    single bad record from the upstream API cannot break indexing.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Corpus not found at {path}. Run: python scripts/fetch_corpus.py"
        )

    documents: list[Document] = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                documents.append(
                    Document(
                        id=raw["id"],
                        title=raw["title"],
                        abstract=raw["abstract"],
                        authors=tuple(raw.get("authors", ())),
                        categories=tuple(raw.get("categories", ())),
                        published=raw.get("published", ""),
                        url=raw.get("url", ""),
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                log.warning("Skipping malformed corpus line %d: %s", line_no, exc)

    log.info("Loaded %d documents from %s", len(documents), path)
    return documents


class InMemoryDocumentStore:
    """Dict-backed :class:`~app.ports.DocumentStore`."""

    def __init__(self, documents: Iterable[Document]) -> None:
        self._documents: dict[str, Document] = {doc.id: doc for doc in documents}

    @classmethod
    def from_jsonl(cls, path: Path) -> InMemoryDocumentStore:
        return cls(load_documents(path))

    def get(self, doc_id: str) -> Document | None:
        return self._documents.get(doc_id)

    def get_many(self, doc_ids: Iterable[str]) -> list[Document]:
        found = (self._documents.get(doc_id) for doc_id in doc_ids)
        return [doc for doc in found if doc is not None]

    def __len__(self) -> int:
        return len(self._documents)

    def __iter__(self) -> Iterator[Document]:
        return iter(self._documents.values())
