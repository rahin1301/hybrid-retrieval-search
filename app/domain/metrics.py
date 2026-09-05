"""Ranking quality metrics.

Standard TREC-style definitions:

* Gain is exponential, ``2**rel - 1``, so a highly relevant document is worth
  meaningfully more than a marginally relevant one.
* Discount is ``log2(rank + 1)`` with 1-based ranks.
* nDCG is normalised against the ideal ordering of the *judged* documents, so
  a query whose relevant documents are all retrieved in the best possible
  order scores exactly 1.0.

Unjudged documents are treated as non-relevant, which is the usual convention
for a small hand-labelled pool. The evaluation README notes the bias this
introduces: systems that surface unjudged-but-relevant documents are penalised.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def dcg_at_k(ranked_ids: Sequence[str], relevance: Mapping[str, int], k: int) -> float:
    """Discounted cumulative gain of the top-k of ``ranked_ids``."""
    _validate_k(k)
    return sum(
        (2 ** relevance.get(doc_id, 0) - 1) / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked_ids[:k], start=1)
    )


def ndcg_at_k(ranked_ids: Sequence[str], relevance: Mapping[str, int], k: int) -> float:
    """Normalised DCG in ``[0, 1]``.

    Returns 0.0 when the query has no relevant documents, since no ranking can
    be distinguished from any other in that case.
    """
    _validate_k(k)
    ideal_ordering = sorted(
        (doc_id for doc_id, rel in relevance.items() if rel > 0),
        key=lambda doc_id: -relevance[doc_id],
    )
    ideal = dcg_at_k(ideal_ordering, relevance, k)
    if ideal == 0.0:
        return 0.0
    return dcg_at_k(ranked_ids, relevance, k) / ideal


def recall_at_k(
    ranked_ids: Sequence[str], relevance: Mapping[str, int], k: int
) -> float:
    """Fraction of all known-relevant documents appearing in the top k."""
    _validate_k(k)
    relevant = {doc_id for doc_id, rel in relevance.items() if rel > 0}
    if not relevant:
        return 0.0
    retrieved = set(ranked_ids[:k])
    return len(relevant & retrieved) / len(relevant)


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean, defined as 0.0 for an empty sequence."""
    return sum(values) / len(values) if values else 0.0


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
