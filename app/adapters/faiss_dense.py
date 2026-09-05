"""FAISS dense retrieval.

``IndexFlatIP`` performs an exact inner-product scan. At ~5k documents an
exact index costs single-digit milliseconds per query, so the recall loss of
an approximate index (IVF/HNSW) would buy nothing. The interface is unchanged
if the corpus later grows enough to need one.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np

from app.domain.errors import IndexNotReadyError, RetrievalError
from app.domain.models import ScoredDoc
from app.ports import TextEncoder

log = logging.getLogger(__name__)


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """Build an exact cosine-similarity index from normalised embeddings.

    Raises:
        ValueError: If ``embeddings`` is not a 2-D float32 array.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {embeddings.shape}")

    vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def save_index(
    index: faiss.Index, doc_ids: list[str], index_path: Path, ids_path: Path
) -> None:
    """Persist the index alongside its row-to-doc-id mapping.

    The two files must stay in lockstep: FAISS returns row offsets, and the id
    list is the only thing that maps them back to documents.
    """
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    ids_path.write_text(json.dumps(doc_ids))
    log.info("Wrote FAISS index (%d vectors) to %s", index.ntotal, index_path)


class FaissDenseRetriever:
    """Semantic candidate retrieval over a persisted FAISS index."""

    def __init__(
        self, index: faiss.Index, doc_ids: list[str], encoder: TextEncoder
    ) -> None:
        if index.ntotal != len(doc_ids):
            raise ValueError(
                f"index has {index.ntotal} vectors but {len(doc_ids)} ids were given; "
                "the index and id map are out of sync — rebuild both"
            )
        self._index = index
        self._doc_ids = doc_ids
        self._encoder = encoder

    @classmethod
    def load(
        cls, index_path: Path, ids_path: Path, encoder: TextEncoder
    ) -> FaissDenseRetriever:
        """Load a persisted index.

        Raises:
            IndexNotReadyError: If either artifact is missing.
        """
        if not index_path.exists() or not ids_path.exists():
            raise IndexNotReadyError(
                f"Dense index missing at {index_path}. Run: python scripts/build_index.py"
            )
        index = faiss.read_index(str(index_path))
        doc_ids = json.loads(ids_path.read_text())
        log.info("Loaded FAISS index with %d vectors", index.ntotal)
        return cls(index, doc_ids, encoder)

    @property
    def name(self) -> str:
        return "dense"

    def search(self, query: str, k: int) -> list[ScoredDoc]:
        """Return the top ``k`` semantic matches.

        Raises:
            RetrievalError: If encoding or the index scan fails.
        """
        try:
            vector = self._encoder.encode_query(query).reshape(1, -1)
            scores, indices = self._index.search(
                np.ascontiguousarray(vector, dtype=np.float32), min(k, self._index.ntotal)
            )
        except Exception as exc:
            raise RetrievalError(f"Dense retrieval failed: {exc}") from exc

        results = []
        for score, row in zip(scores[0], indices[0], strict=True):
            # FAISS pads with -1 when fewer than k neighbours exist.
            if row < 0:
                continue
            results.append(
                ScoredDoc(doc_id=self._doc_ids[int(row)], score=float(score))
            )
        return results

    def __len__(self) -> int:
        return int(self._index.ntotal)
