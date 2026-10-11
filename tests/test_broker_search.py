"""Layer 3: a narrow retriever over broker research - the ARGUMENT, for one company, attributed.

No model is loaded here: the vector store is replaced by an in-memory stand-in that implements the
three calls the code makes (get / delete / add_documents) with Chroma's semantics, so these tests
check what ends up IN the store, not which methods were called.
"""
import asyncio

import pytest
from langchain_core.documents import Document

from pennystock.retrieval import broker_search as bs
from pennystock.retrieval.broker_search import (drop_legacy_chunks, replace_document_chunks,
                                                search_broker_passages)


class MemoryStore:
    """Chroma's contract, minus the embeddings: upsert by id, delete by id, filter by metadata."""

    def __init__(self):
        self.rows: dict[str, Document] = {}

    def _match(self, md, where):
        if not where:
            return True
        if "$and" in where:
            return all(self._match(md, w) for w in where["$and"])
        (key, cond), = where.items()
        if isinstance(cond, dict) and "$in" in cond:
            return md.get(key) in cond["$in"]
        return md.get(key) == cond

    def get(self, where=None, include=None):
        hits = [(i, d) for i, d in self.rows.items() if self._match(d.metadata, where)]
        return {"ids": [i for i, _ in hits], "documents": [d.page_content for _, d in hits],
                "metadatas": [d.metadata for _, d in hits]}

    def delete(self, ids):
        for i in ids:
            self.rows.pop(i, None)

    def add_documents(self, docs, ids):
        self.rows.update(dict(zip(ids, docs)))


def chunks(n=3, kind="narrative", text="The bank grew loans 30% while holding credit cost at 1%."):
    base = {"source": "a.pdf", "company": "AU SF Bank", "broker": "ICICI Securities",
            "report_date": "2026-09-04", "rating": "BUY", "symbol": "AUBANK"}
    return [{"text": f"{text} [{i}]", "metadata": {**base, "chunk_type": kind}} for i in range(n)]


# --- writing: idempotent, research only ------------------------------------------------------
def test_re_ingesting_a_document_replaces_its_chunks_instead_of_duplicating_them():
    """The old ingest used random ids, so every re-run doubled the collection."""
    store = MemoryStore()
    replace_document_chunks(store, "doc1", chunks(3))
    replace_document_chunks(store, "doc1", chunks(2))
    assert sorted(store.rows) == ["doc1-0000", "doc1-0001"]


def test_only_the_argument_is_stored():
    """Tables embed badly (the measured cause of most financial-RAG misses) and numbers are served
    by get_broker_consensus, so table and estimate chunks never enter this store."""
    store = MemoryStore()
    n = replace_document_chunks(store, "doc1", chunks(2) + chunks(1, "table") +
                                chunks(1, "estimates") +
                                chunks(1, "recommendation", text="BUY, target 1,250 (unchanged)"))
    assert n == 3
    assert {d.metadata["chunk_type"] for d in store.rows.values()} == {"narrative", "recommendation"}


def test_a_passage_repeated_inside_one_report_is_stored_once():
    store = MemoryStore()
    twice = chunks(1) + chunks(1)
    assert replace_document_chunks(store, "doc1", twice) == 1


def test_every_stored_chunk_knows_its_document():
    store = MemoryStore()
    replace_document_chunks(store, "doc1", chunks(2))
    assert {d.metadata["doc_id"] for d in store.rows.values()} == {"doc1"}


def test_chunks_from_before_documents_had_ids_are_cleared():
    """The first ingest wrote 1,103 chunks with no symbol and no document id. They can never be
    reached by a company-filtered search, so they are dead weight that only dilutes retrieval."""
    store = MemoryStore()
    store.rows["legacy-1"] = Document(page_content="old", metadata={"company": "X"})
    replace_document_chunks(store, "doc1", chunks(1))
    assert drop_legacy_chunks(store) == 1
    assert list(store.rows) == ["doc1-0000"]


# --- reading: one company, newest first, at most N per report -------------------------------
@pytest.fixture
def no_models(monkeypatch):
    """Dense search and the cross-encoder are the model-backed steps: replace them with
    deterministic stand-ins (keyword overlap) so ordering and filtering can be checked exactly."""
    def overlap(q, d):
        return len(set(q.lower().split()) & set(d.page_content.lower().split()))

    def dense(store, question, k, where):
        docs = [Document(page_content=t, metadata=m)
                for t, m in zip(*[store.get(where=where)[x] for x in ("documents", "metadatas")])]
        return sorted(docs, key=lambda d: -overlap(question, d))[:k]

    def rerank(question, docs, top_n):
        return sorted(docs, key=lambda d: -overlap(question, d))[:top_n]

    monkeypatch.setattr(bs, "_dense", dense)
    monkeypatch.setattr(bs, "rerank", rerank)


def _passages(store, symbol, broker, date, texts, kind="narrative", doc=None):
    doc = doc or f"{symbol}-{broker}-{date}"
    replace_document_chunks(store, doc, [
        {"text": t, "metadata": {"source": f"{doc}.pdf", "symbol": symbol, "broker": broker,
                                 "report_date": date, "rating": "BUY", "chunk_type": kind}}
        for t in texts])


def test_search_never_returns_another_companys_research(no_models):
    """Metadata pre-filtering by company is the single biggest gain measured on financial RAG -
    and without it a question about AU SF Bank can be answered from a Muthoot note."""
    store = MemoryStore()
    _passages(store, "AUBANK", "ICICI", "2026-09-04", ["credit cost normalisation drives RoA"])
    _passages(store, "MUTHOOTFIN", "ICICI", "2026-09-04", ["credit cost normalisation in gold loans"])
    got = asyncio.run(search_broker_passages("AUBANK", "why will credit cost fall", store=store))
    assert {d.metadata["symbol"] for d in got} == {"AUBANK"}


def test_results_come_newest_report_first_and_no_report_floods_them(no_models, monkeypatch):
    monkeypatch.setattr(bs.settings, "broker_chunks_per_report", 2)
    store = MemoryStore()
    _passages(store, "AUBANK", "Axis", "2026-07-27", ["deposit growth credit cost view one",
                                                      "deposit growth credit cost view two",
                                                      "deposit growth credit cost view three"])
    _passages(store, "AUBANK", "ICICI", "2026-09-04", ["credit cost deposit growth newest"])
    got = asyncio.run(search_broker_passages("AUBANK", "credit cost deposit growth", store=store, top_n=3))
    dates = [d.metadata["report_date"] for d in got]
    assert dates == ["2026-09-04", "2026-07-27", "2026-07-27"]       # newest report first
    assert sum(1 for d in got if d.metadata["broker"] == "Axis") == 2  # the long note is capped


def test_a_company_with_no_research_returns_nothing(no_models):
    assert asyncio.run(search_broker_passages("NOBODY", "anything", store=MemoryStore())) == []
