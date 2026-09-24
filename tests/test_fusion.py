"""Unit tests for reciprocal-rank fusion (the RRF merge)."""
from langchain_core.documents import Document

from pennystock.retrieval.fusion import reciprocal_rank_fusion


def _d(text: str) -> Document:
    return Document(page_content=text)


def test_empty_inputs_return_empty():
    assert reciprocal_rank_fusion([], [], k=4) == []


def test_respects_k_limit():
    dense = [_d(str(i)) for i in range(10)]
    out = reciprocal_rank_fusion(dense, [], k=3)
    assert len(out) == 3


def test_preserves_rank_order_within_a_single_list():
    dense = [_d("first"), _d("second"), _d("third")]
    out = reciprocal_rank_fusion(dense, [], k=3)
    assert [d.page_content for d in out] == ["first", "second", "third"]


def test_doc_found_by_both_retrievers_ranks_first():
    # "B" is mid-rank in each list, but appears in BOTH -> fused score should win.
    dense = [_d("A"), _d("B"), _d("C")]
    sparse = [_d("D"), _d("B"), _d("E")]
    out = reciprocal_rank_fusion(dense, sparse, k=5)
    assert out[0].page_content == "B"


def test_deduplicates_identical_documents():
    dense = [_d("same"), _d("other")]
    sparse = [_d("same")]
    texts = [d.page_content for d in reciprocal_rank_fusion(dense, sparse, k=5)]
    assert texts.count("same") == 1               # merged, not duplicated
    assert set(texts) == {"same", "other"}
