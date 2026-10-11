"""Filings search: passages indexed once per file, searched by words and meaning, one company, point-in-time.

No model is loaded: the vector store is an in-memory stand-in with Chroma's contract (get / delete /
add_documents / similarity_search with $and, $in and $lte filters) that ranks by shared words.
"""
from __future__ import annotations

import asyncio
import re
from collections import Counter

import pytest
from langchain_core.documents import Document

from pennystock.data import documents
from pennystock.data import filing_text as ft
from pennystock.retrieval import filing_search as fs


class MemoryStore:
    def __init__(self):
        self.rows: dict[str, Document] = {}

    def _match(self, md, where):
        if not where:
            return True
        if "$and" in where:
            return all(self._match(md, w) for w in where["$and"])
        (key, cond), = where.items()
        if isinstance(cond, dict):
            if "$in" in cond:
                return md.get(key) in cond["$in"]
            if "$lte" in cond:
                return md.get(key) is not None and md.get(key) <= cond["$lte"]
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

    def similarity_search(self, query, k=4, filter=None):
        q = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored = [(len(q & set(re.findall(r"[a-z0-9]+", d.page_content.lower()))), i, d)
                  for i, d in self.rows.items() if self._match(d.metadata, filter)]
        return [d for s, _, d in sorted(scored, key=lambda x: (-x[0], x[1])) if s > 0][:k]


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(ft.settings, "filings_db", tmp_path / "filings.db")
    monkeypatch.setattr(ft.settings, "market_db", tmp_path / "market.db")
    monkeypatch.setattr(ft.settings, "project_root", tmp_path)
    monkeypatch.setattr(fs, "rerank", lambda q, docs, n: docs[:n])
    return tmp_path


@pytest.fixture
def store():
    return MemoryStore()


def _doc(key, doc_type, sha, date, stamp, symbol="AUBANK"):
    return documents.Document(doc_key=key, symbol=symbol, kind="filing", category=doc_type, doc_date=date,
                              title="t", source="nse",
                              source_url=f"https://nsearchives.nseindia.com/corporate/{symbol}{stamp}x.pdf",
                              local_path=f"Corporate_Filings/{symbol}/{doc_type}/{sha}.pdf", bytes=1,
                              sha256=sha, fetched_at="2026-10-05T00:00:00")


CALL_PAGES = {
    "c1": [("In the April call management said NIM guidance stays at 5.6% for the year. " * 8, False),
           ("Deposit costs and NIM guidance were discussed again by the CFO. " * 8, False)],
    "c2": [("In July management raised NIM guidance to 5.8% to 6.0% on lower deposit costs. " * 8, False),
           ("Credit cost and NIM guidance: the CFO repeated the NIM range. " * 8, False)],
}


def seed_two_calls(db):
    docs = [_doc("filing:nse:1", "earnings_call", "c1", "2026-04-25", "_25042026183000_"),
            _doc("filing:nse:2", "earnings_call", "c2", "2026-07-25", "_25072026183000_")]
    ft.index_texts(docs, extract=lambda path, doc_type: CALL_PAGES[path.stem])


def seed_results_with_a_table_page(db):
    pages = {"r1": [("Press release: profit rose 18% on loan growth and steady asset quality. " * 6, False),
                    ("Revenue from operations 12,345.67 11,234.56 45,678.90 Other income 123.45 98.76 "
                     "456.78 " * 8, False)]}
    ft.index_texts([_doc("filing:nse:9", "results", "r1", "2026-07-20", "_20072026170000_")],
                   extract=lambda path, doc_type: pages[path.stem])


def test_indexing_twice_writes_each_passage_once_and_resumes_after_a_crash(db, store):
    seed_two_calls(db)                                   # two AUBANK calls through index_texts
    assert fs.index_passages(store=store)["indexed"] > 0
    n = len(store.rows)
    assert fs.index_passages(store=store)["indexed"] == 0 and len(store.rows) == n

def test_search_stays_on_one_company_and_cites_type_date_and_page(db, store):
    seed_two_calls(db)
    fs.index_passages(store=store)
    hits = asyncio.run(fs.search_filings("AUBANK", "NIM guidance", store=store))
    assert hits and all(h.metadata["symbol"] == "AUBANK" for h in hits)
    m = hits[0].metadata
    assert {"doc_type", "doc_date", "page_from", "page_to", "path"} <= set(m)

def test_an_as_of_date_never_lets_a_later_filing_through(db, store):
    seed_two_calls(db)                                    # calls broadcast 2026-04-25 and 2026-07-25
    fs.index_passages(store=store)
    for mode in ("keyword", "dense", "fused", "fused_rerank"):
        hits = asyncio.run(fs.search_filings("AUBANK", "NIM guidance", as_of="2026-05-01T00:00", mode=mode,
                                             store=store))
        assert all(h.metadata["available_at"] <= "2026-05-01T00:00" for h in hits), mode

def test_no_document_fills_the_results_and_they_read_newest_first(db, store):
    seed_two_calls(db)
    fs.index_passages(store=store)
    hits = asyncio.run(fs.search_filings("AUBANK", "NIM", top_n=8, store=store))
    per_doc = Counter(h.metadata["content_sha"] for h in hits)
    assert max(per_doc.values()) <= 2
    when = [h.metadata["available_at"] for h in hits]
    assert when == sorted(when, reverse=True)

def test_a_crash_mid_indexing_leaves_no_file_marked_done_and_the_rerun_has_no_duplicates(db, store,
                                                                                           monkeypatch):
    seed_two_calls(db)
    real, calls = store.add_documents, []
    def flaky(docs, ids):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("killed")
        real(docs, ids)
    monkeypatch.setattr(store, "add_documents", flaky)
    with pytest.raises(RuntimeError):
        fs.index_passages(store=store)
    monkeypatch.setattr(store, "add_documents", real)
    fs.index_passages(store=store)
    ids = list(store.rows)
    assert len(ids) == len(set(ids)) and fs.index_passages(store=store)["indexed"] == 0

def test_a_table_page_is_found_by_keyword_but_never_embedded(db, store):
    seed_results_with_a_table_page(db)
    fs.index_passages(store=store)
    assert all(m.get("kind") != "table" for m in (d.metadata for d in store.rows.values()))
    hits = asyncio.run(fs.search_filings("AUBANK", "revenue from operations", mode="keyword", store=store))
    assert hits and hits[0].metadata["kind"] == "table"


def test_the_eval_can_read_results_in_ranked_order(db, store):
    seed_two_calls(db)
    fs.index_passages(store=store)
    question = "April call: NIM guidance stays at 5.6% for the year"          # the OLDER call answers it
    ranked = asyncio.run(fs.search_filings("AUBANK", question, mode="dense", newest_first=False, store=store))
    newest = asyncio.run(fs.search_filings("AUBANK", question, mode="dense", store=store))
    assert ranked[0].metadata["available_at"] == "2026-04-25T18:30"              # best match first
    assert newest[0].metadata["available_at"] == "2026-07-25T18:30"              # the analyst: newest first
    assert sorted(d.page_content for d in ranked) == sorted(d.page_content for d in newest)
