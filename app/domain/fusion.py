"""Rank fusion.

Pure functions over ranked id lists — no scores from the underlying retrievers
are needed, which is precisely why RRF is used here: BM25 scores and cosine
similarities are not on comparable scales, and normalising them requires
corpus-dependent tuning that does not transfer across queries.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.domain.models import ScoredDoc


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
    limit: int | None = None,
) -> list[ScoredDoc]:
    """Fuse ranked id lists using Reciprocal Rank Fusion.

    Each list contributes ``weight / (k + rank)`` to every document it ranks,
    where ``rank`` is 1-based. Documents missing from a list simply receive no
    contribution from it.

    Args:
        rankings: One ranked list of document ids per retriever, best first.
        k: Damping constant. Larger values flatten the contribution curve so
            that top ranks matter less. 60 is the value from Cormack et al.
        weights: Optional per-ranking weights, defaulting to 1.0 each.
        limit: Truncate the fused output to this many documents.

    Returns:
        Documents ordered by descending fused score. Ties break on doc id so
        the output is deterministic.

    Raises:
        ValueError: If ``weights`` is provided with a length mismatch, or ``k``
            is negative.
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    if weights is not None and len(weights) != len(rankings):
        raise ValueError(
            f"weights has length {len(weights)} but there are {len(rankings)} rankings"
        )

    effective_weights = list(weights) if weights is not None else [1.0] * len(rankings)

    fused: dict[str, float] = {}
    for ranking, weight in zip(rankings, effective_weights, strict=True):
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (k + rank)

    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    if limit is not None:
        ordered = ordered[:limit]
    return [ScoredDoc(doc_id=doc_id, score=score) for doc_id, score in ordered]
