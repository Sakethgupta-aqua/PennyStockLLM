"""Reciprocal Rank Fusion — merge two ranked document lists into one.

Pure function (no I/O, no model), so it is fully unit-testable. Each document earns
``1 / (rank + c)`` from each list it appears in; a doc found by *both* the dense and the
sparse retriever accumulates score from both and therefore ranks higher. Duplicates
(same text) are merged.
"""
from langchain_core.documents import Document


def reciprocal_rank_fusion(
    dense_docs: list[Document],
    sparse_docs: list[Document],
    k: int = 4,
    c: int = 60,
) -> list[Document]:
    """Fuse dense + sparse results; return the top ``k`` unique documents by fused score."""
    fused: dict[str, dict] = {}
    for docs in (dense_docs, sparse_docs):
        for rank, doc in enumerate(docs):
            entry = fused.setdefault(doc.page_content, {"doc": doc, "score": 0.0})
            entry["score"] += 1.0 / (rank + c)
    ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
    return [x["doc"] for x in ranked[:k]]


def reciprocal_rank_fusion_many(
    doc_lists: list[list[Document]],
    k: int | None = None,
    c: int = 60,
) -> list[Document]:
    """RRF over ANY number of ranked lists (one per sub-query); merges duplicates by text.

    Same formula as reciprocal_rank_fusion, generalized to N lists. A chunk that ranks high
    across several sub-queries accumulates score from each and rises to the top.
    """
    fused: dict[str, dict] = {}
    for docs in doc_lists:
        for rank, doc in enumerate(docs):
            entry = fused.setdefault(doc.page_content, {"doc": doc, "score": 0.0})
            entry["score"] += 1.0 / (rank + c)
    ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
    docs = [x["doc"] for x in ranked]
    return docs[:k] if k else docs
