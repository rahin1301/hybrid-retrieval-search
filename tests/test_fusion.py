import pytest

from app.domain.fusion import reciprocal_rank_fusion


def ids(results):
    return [r.doc_id for r in results]


def test_single_ranking_preserves_order():
    fused = reciprocal_rank_fusion([["a", "b", "c"]])
    assert ids(fused) == ["a", "b", "c"]


def test_document_in_both_rankings_outranks_document_in_one():
    # 'b' is second in both lists; 'a' and 'c' are first in only one each.
    fused = reciprocal_rank_fusion([["a", "b"], ["c", "b"]], k=1)
    assert ids(fused)[0] == "b"


def test_scores_follow_the_rrf_formula():
    fused = reciprocal_rank_fusion([["a", "b"]], k=60)
    assert fused[0].score == pytest.approx(1 / 61)
    assert fused[1].score == pytest.approx(1 / 62)


def test_weights_shift_the_ordering():
    unweighted = reciprocal_rank_fusion([["a"], ["b"]])
    assert ids(unweighted) == ["a", "b"]  # tie, broken alphabetically

    weighted = reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0, 5.0])
    assert ids(weighted) == ["b", "a"]


def test_limit_truncates_output():
    fused = reciprocal_rank_fusion([["a", "b", "c", "d"]], limit=2)
    assert ids(fused) == ["a", "b"]


def test_ties_break_deterministically_on_doc_id():
    first = reciprocal_rank_fusion([["z"], ["a"]])
    second = reciprocal_rank_fusion([["a"], ["z"]])
    assert ids(first) == ids(second) == ["a", "z"]


def test_empty_input_yields_empty_output():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_negative_k_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        reciprocal_rank_fusion([["a"]], k=-1)


def test_weight_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="length"):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])
