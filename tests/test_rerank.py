from langchain_core.documents import Document

from pennystock.retrieval.rerank import cap_per_source, reorder_lost_in_middle


def _docs(spec):
    return [Document(page_content=f"{src}{i}", metadata={"source": src})
            for src, n in spec for i in range(n)]


def test_cap_forces_diversity():
    # 6 chunks from book A arrive first; B and C must still get in
    docs = _docs([("A", 6), ("B", 1), ("C", 1)])
    out = cap_per_source(docs, max_per_source=3, limit=5)
    srcs = [d.metadata["source"] for d in out]
    assert len(out) == 5
    assert srcs.count("A") == 3 and "B" in srcs and "C" in srcs


def test_cap_backfills_when_corpus_is_thin():
    # only one source available -> must still return `limit` docs, not fewer
    out = cap_per_source(_docs([("A", 6)]), max_per_source=2, limit=4)
    assert len(out) == 4


def test_cap_keeps_best_first_order():
    out = cap_per_source(_docs([("A", 2), ("B", 2)]), max_per_source=3, limit=4)
    assert [d.page_content for d in out] == ["A0", "A1", "B0", "B1"]


def test_reorder_puts_best_at_both_ends():
    docs = [Document(page_content=str(i)) for i in range(5)]
    assert [d.page_content for d in reorder_lost_in_middle(docs)] == ["0", "2", "4", "3", "1"]
