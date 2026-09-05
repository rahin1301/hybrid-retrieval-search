import math

import pytest

from app.domain.metrics import dcg_at_k, mean, ndcg_at_k, recall_at_k

RELEVANCE = {"a": 3, "b": 2, "c": 1}


def test_ideal_ranking_scores_one():
    assert ndcg_at_k(["a", "b", "c"], RELEVANCE, 10) == pytest.approx(1.0)


def test_reversed_ranking_scores_below_ideal():
    assert ndcg_at_k(["c", "b", "a"], RELEVANCE, 10) < 1.0


def test_relevant_documents_below_the_cutoff_do_not_count():
    padded = ["x", "y", "z", "a"]
    assert ndcg_at_k(padded, RELEVANCE, 3) == 0.0
    assert ndcg_at_k(padded, RELEVANCE, 4) > 0.0


def test_dcg_applies_exponential_gain_and_log_discount():
    # Single document at rank 1: (2**3 - 1) / log2(2) == 7.0
    assert dcg_at_k(["a"], RELEVANCE, 1) == pytest.approx(7.0)
    # Same document at rank 2: 7 / log2(3)
    assert dcg_at_k(["x", "a"], RELEVANCE, 2) == pytest.approx(7.0 / math.log2(3))


def test_ndcg_is_zero_when_no_document_is_relevant():
    assert ndcg_at_k(["a", "b"], {"a": 0, "b": 0}, 10) == 0.0


def test_unjudged_documents_are_treated_as_non_relevant():
    assert ndcg_at_k(["unjudged", "a"], RELEVANCE, 10) < ndcg_at_k(
        ["a", "unjudged"], RELEVANCE, 10
    )


def test_recall_counts_distinct_relevant_documents_retrieved():
    assert recall_at_k(["a", "b", "c"], RELEVANCE, 10) == pytest.approx(1.0)
    assert recall_at_k(["a", "x", "y"], RELEVANCE, 10) == pytest.approx(1 / 3)
    assert recall_at_k(["x", "y"], RELEVANCE, 10) == 0.0


def test_recall_respects_the_cutoff():
    assert recall_at_k(["x", "x2", "a"], RELEVANCE, 2) == 0.0
    assert recall_at_k(["x", "x2", "a"], RELEVANCE, 3) == pytest.approx(1 / 3)


def test_recall_is_zero_without_relevant_documents():
    assert recall_at_k(["a"], {"a": 0}, 5) == 0.0


def test_mean_of_empty_sequence_is_zero():
    assert mean([]) == 0.0
    assert mean([1.0, 2.0]) == pytest.approx(1.5)


@pytest.mark.parametrize("metric", [dcg_at_k, ndcg_at_k, recall_at_k])
def test_non_positive_k_is_rejected(metric):
    with pytest.raises(ValueError, match="positive"):
        metric(["a"], RELEVANCE, 0)
