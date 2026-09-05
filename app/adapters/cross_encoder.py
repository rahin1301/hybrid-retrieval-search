"""Cross-encoder reranking.

A bi-encoder embeds query and document independently, so it can never model
term interactions between them. A cross-encoder scores the pair jointly and is
markedly more accurate, but costs one forward pass per candidate — which is why
it runs only over the top ``rerank_depth`` fused candidates, not the corpus.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from sentence_transformers import CrossEncoder

from app.domain.models import Document, ScoredDoc

log = logging.getLogger(__name__)

# Cross-encoders truncate at their max sequence length; abstracts are clipped
# so the title and opening sentences (the highest-signal part) always survive.
_MAX_CHARS = 1800


class CrossEncoderReranker:
    """Pairwise reranker over a candidate set."""

    def __init__(
        self, model_name: str, *, batch_size: int = 32, device: str | None = None
    ) -> None:
        log.info("Loading cross-encoder %s", model_name)
        self._model = CrossEncoder(model_name, device=device, max_length=512)
        self._model_name = model_name
        self._batch_size = batch_size

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(
        self, query: str, documents: Sequence[Document], top_n: int
    ) -> list[ScoredDoc]:
        """Rescore ``documents`` jointly against ``query``.

        Returns the ``top_n`` highest-scoring documents, best first. An empty
        candidate list yields an empty result rather than an error.
        """
        if not documents:
            return []

        pairs = [(query, doc.to_text()[:_MAX_CHARS]) for doc in documents]
        scores = self._model.predict(
            pairs, batch_size=self._batch_size, show_progress_bar=False
        )

        scored = [
            ScoredDoc(doc_id=doc.id, score=float(score))
            for doc, score in zip(documents, scores, strict=True)
        ]
        scored.sort(key=lambda item: (-item.score, item.doc_id))
        return scored[:top_n]
