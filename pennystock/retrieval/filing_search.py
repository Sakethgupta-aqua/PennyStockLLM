"""Filings search: one company's filings, searched by exact words and by meaning, point-in-time.

Pages live in filings.db (keyword search over every page, `filing_text`); passages - the parts worth
reading, cut by structure (`filing_passages`) - live in the Chroma collection `filings`. A search:

  1. filters to ONE company (a shared store across companies answered 19% of FinanceBench, a
     per-document store 50%), optionally to document types and to what the market had seen by `as_of`
  2. keyword: BM25 over pages -> the passages covering those pages (a page that was never embedded -
     a statement table, annual-report boilerplate - comes back as the page itself)
  3. meaning: dense search over passages
  4. reciprocal-rank fusion -> cross-encoder rerank -> at most `filings_per_doc` per filing -> top N,
     read newest first

`mode` exists for the A/B eval: keyword | dense | fused | fused_rerank (the default is set from it).
Spec: docs/superpowers/specs/2026-10-05-filings-search-design.md
"""
from __future__ import annotations

import asyncio
import logging
from collections import Counter
from functools import lru_cache

from langchain_core.documents import Document

from pennystock.config import settings
from pennystock.data import filing_text as ft
from pennystock.retrieval.filing_passages import CHUNKER_VERSION, context_line, passages_for
from pennystock.retrieval.fusion import reciprocal_rank_fusion
from pennystock.retrieval.rerank import rerank

logger = logging.getLogger(__name__)

_BATCH = 256


@lru_cache(maxsize=1)
def filings_store():
    """The filings collection - separate from the books and the broker notes. Built lazily: importing
    this module loads no model."""
    from langchain_chroma import Chroma

    from pennystock.retrieval.store import _embeddings
    return Chroma(persist_directory=str(settings.vector_db_dir), embedding_function=_embeddings(),
                  collection_name=settings.filings_collection)


def _ts(when: str) -> int:
    """'YYYY-MM-DDTHH:MM' -> YYYYMMDDHHMM: Chroma compares numbers, not strings."""
    digits = "".join(ch for ch in when if ch.isdigit())
    return int((digits + "0000")[:12])


def _metadata(meta: dict, sha: str, page_from: int, page_to: int, section, speaker, kind: str) -> dict:
    # Chroma refuses None in metadata, so blanks are empty strings
    return {"symbol": meta["symbol"], "doc_type": meta["doc_type"], "doc_date": meta["doc_date"],
            "available_at": meta["available_at"], "available_ts": _ts(meta["available_at"]),
            "fiscal_period": meta.get("fiscal_period") or "", "page_from": page_from, "page_to": page_to,
            "section": section or "", "speaker": speaker or "", "content_sha": sha,
            "doc_key": meta["doc_key"], "path": meta["local_path"], "kind": kind}


def index_passages(store=None, reindex: bool = False, symbols: set[str] | None = None) -> dict:
    """Embed the passages of every extracted file not yet indexed at CHUNKER_VERSION (all, with
    reindex). A file's old passages are removed first and its version is written only after its new
    ones are stored, so a run killed half-way leaves nothing half-done and no duplicates."""
    store = store if store is not None else filings_store()
    with ft._connect() as con:
        rows = con.execute("SELECT content_sha, passages_version FROM filing_texts "
                           "WHERE extractor_version = ?", (ft.EXTRACTOR_VERSION,)).fetchall()
    tally: Counter = Counter(dict.fromkeys(("indexed", "current", "passages", "empty"), 0))
    for sha, version in rows:
        if not reindex and version == CHUNKER_VERSION:
            tally["current"] += 1
            continue
        meta = ft.doc_meta(sha)
        if not meta or (symbols and meta["symbol"] not in symbols):
            continue
        passages = passages_for(meta, ft.pages_of(sha))
        old = store.get(where={"content_sha": sha}, include=[])["ids"]
        if old:
            store.delete(ids=old)
        docs = [Document(page_content=p.text,
                         metadata=_metadata(meta, sha, p.page_from, p.page_to, p.section, p.speaker, "prose"))
                for p in passages]
        for i in range(0, len(docs), _BATCH):
            batch = docs[i:i + _BATCH]
            store.add_documents(batch, ids=[f"{sha[:16]}-{j:05d}" for j in range(i, i + len(batch))])
        with ft._connect() as con:
            con.execute("UPDATE filing_texts SET passages_version = ? WHERE content_sha = ?",
                        (CHUNKER_VERSION, sha))
        tally["indexed"] += 1
        tally["passages"] += len(docs)
        tally["empty"] += not docs
    return dict(tally)


def _keyword_docs(store, symbol: str, question: str, doc_types, as_of) -> list[Document]:
    """BM25 page hits as documents: the passages that cover each hit page, or - for a page that was
    never embedded - the page itself, with its context line."""
    out: list[Document] = []
    seen: set[str] = set()
    cache: dict[str, list[Document]] = {}
    for hit in ft.keyword_pages(symbol, question, doc_types, as_of, limit=settings.filings_fts_k):
        sha, page = hit["content_sha"], hit["page"]
        if sha not in cache:
            got = store.get(where={"content_sha": sha})
            cache[sha] = [Document(page_content=t, metadata=m)
                          for t, m in zip(got["documents"], got["metadatas"])]
        covering = [d for d in cache[sha] if d.metadata["page_from"] <= page <= d.metadata["page_to"]]
        if covering:
            for d in covering:
                key = d.page_content
                if key not in seen:
                    seen.add(key)
                    out.append(d)
            continue
        meta = ft.doc_meta(sha) or {**hit, "company": hit["symbol"]}
        line = context_line(meta.get("company") or hit["symbol"], hit["doc_type"], hit["doc_date"],
                            hit.get("fiscal_period"), page, page, hit.get("section"))
        text = f"{line}\n{hit['text'][:settings.filings_passage_chars]}"
        if text not in seen:
            seen.add(text)
            out.append(Document(page_content=text, metadata=_metadata(
                {**meta, **{k: hit[k] for k in ("symbol", "doc_type", "doc_date", "available_at", "doc_key")},
                 "local_path": hit["local_path"], "fiscal_period": hit.get("fiscal_period")},
                sha, page, page, hit.get("section"), None, hit["kind"])))
    return out


def _cap_per_filing(docs: list[Document], per_doc: int, limit: int) -> list[Document]:
    out, count = [], Counter()
    for d in docs:
        sha = d.metadata.get("content_sha")
        if count[sha] < per_doc:
            count[sha] += 1
            out.append(d)
        if len(out) >= limit:
            break
    return out


async def search_filings(symbol: str, question: str, doc_types: list[str] | None = None,
                         as_of: str | None = None, top_n: int | None = None, mode: str | None = None,
                         store=None, newest_first: bool = True) -> list[Document]:
    """The most relevant passages of ONE company's filings, each carrying type, date, period and page.
    Read newest first by default; `newest_first=False` keeps the ranked order (the eval scores rank)."""
    store = store if store is not None else filings_store()
    mode = mode or settings.filings_search_mode
    top_n = top_n or settings.filings_search_top_n
    conditions: list[dict] = [{"symbol": symbol.upper()}]
    if doc_types:
        conditions.append({"doc_type": {"$in": list(doc_types)}})
    if as_of:
        conditions.append({"available_ts": {"$lte": _ts(as_of)}})
    where = {"$and": conditions} if len(conditions) > 1 else conditions[0]
    dense: list[Document] = []
    sparse: list[Document] = []
    if mode != "keyword":
        dense = await asyncio.to_thread(store.similarity_search, question, k=settings.filings_dense_k,
                                        filter=where)
    if mode != "dense":
        sparse = await asyncio.to_thread(_keyword_docs, store, symbol, question, doc_types, as_of)
    if mode == "keyword":
        pool = sparse
    elif mode == "dense":
        pool = dense
    else:
        pool = reciprocal_rank_fusion(dense, sparse, k=settings.rerank_pool, c=settings.rrf_c)
    if mode == "fused_rerank" and pool:
        pool = await asyncio.to_thread(rerank, question, pool, top_n * settings.rerank_overfetch)
    picked = _cap_per_filing(pool, settings.filings_per_doc, top_n)
    if not newest_first:
        return picked
    return sorted(picked, key=lambda d: d.metadata.get("available_at", ""), reverse=True)
