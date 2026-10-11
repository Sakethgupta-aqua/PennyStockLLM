"""Layer 3: retrieval over broker research, narrowed to what retrieval is actually good at.

Measured on this collection (2026-09-30): asked "which brokers cut estimates recently?" it returned
a daily price newsletter; asked why ICICI downgraded UltraTech it returned the shareholding table.
Research on financial RAG finds the same failure everywhere - most misses come from tables that do
not embed as text - and that numeric questions belong to tools, narrative ones to retrieval.

So this retriever answers exactly one question - "what is the broker's ARGUMENT?" - and is narrow
in three ways:

  1. ONE COMPANY. Every search is pre-filtered on the resolved symbol. Metadata pre-filtering is
     the largest single gain measured on financial document QA, and it makes a Muthoot note
     answering an AU SF Bank question impossible rather than unlikely.
  2. PROSE ONLY. Only narrative and recommendation chunks are stored. Tables and the rendered
     estimates are numbers, and numbers come from get_broker_consensus (Layer 1 + 2).
  3. ATTRIBUTED AND DATED. Every chunk carries its broker, date and rating in its text, results come
     newest first, and no single long report may fill the context.
"""
from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

from langchain_core.documents import Document

from pennystock.config import settings
from pennystock.retrieval.fusion import reciprocal_rank_fusion
from pennystock.retrieval.rerank import cap_per_source, rerank

logger = logging.getLogger(__name__)

RESEARCH_CHUNK_TYPES = ("recommendation", "narrative")


@lru_cache(maxsize=1)
def broker_db():
    """The broker collection - separate from the book corpus, whose methodology must never be
    answered by sell-side optimism. Built lazily: importing this module loads no model."""
    from langchain_chroma import Chroma

    from pennystock.retrieval.store import _embeddings
    return Chroma(persist_directory=str(settings.vector_db_dir), embedding_function=_embeddings(),
                  collection_name=settings.broker_collection)


# ------------------------------------------------------------------ writing
def chunk_id(doc_id: str, i: int) -> str:
    return f"{doc_id}-{i:04d}"


def replace_document_chunks(store, doc_id: str, chunks: list[dict]) -> int:
    """Swap one document's chunks for a fresh set. Deterministic ids plus delete-before-add make a
    re-run a no-op instead of a duplicate - the old ingest's random ids doubled the collection."""
    old = store.get(where={"doc_id": doc_id}, include=[])["ids"]
    if old:
        store.delete(ids=old)
    docs: list[Document] = []
    seen: set[str] = set()
    for c in chunks:
        if c["metadata"].get("chunk_type") not in RESEARCH_CHUNK_TYPES:
            continue
        key = " ".join(c["text"].split())
        if key in seen:                         # the same paragraph printed twice in one note
            continue
        seen.add(key)
        docs.append(Document(page_content=c["text"], metadata={**c["metadata"], "doc_id": doc_id}))
    if docs:
        store.add_documents(docs, ids=[chunk_id(doc_id, i) for i in range(len(docs))])
    return len(docs)


def drop_legacy_chunks(store) -> int:
    """Remove chunks written before documents carried an id (and a symbol): no company-filtered
    search can ever reach them, so they only dilute the collection."""
    got = store.get(include=["metadatas"])
    legacy = [i for i, m in zip(got["ids"], got["metadatas"]) if not (m or {}).get("doc_id")]
    if legacy:
        store.delete(ids=legacy)
    return len(legacy)


# ------------------------------------------------------------------ reading
def _dense(store, question: str, k: int, where: dict) -> list[Document]:
    return store.similarity_search(question, k=k, filter=where)


def _bm25(docs: list[Document], question: str, k: int) -> list[Document]:
    """Keyword recall over the same filtered set: exact terms ("CASA", "Stage 3", "CGFMU") that an
    embedding can blur are what BM25 is for, and the set is small enough to index per query."""
    if len(docs) < 2:
        return list(docs)
    from pennystock.retrieval.bm25 import LangchainBM25sRetriever
    return LangchainBM25sRetriever.from_documents(docs, k=min(k, len(docs))).invoke(question)


async def search_broker_passages(symbol: str, question: str, top_n: int | None = None,
                                 store=None) -> list[Document]:
    """The most relevant research passages about ONE company: filter -> dense + BM25 -> RRF ->
    cross-encoder -> at most N per report -> newest report first."""
    store = store if store is not None else broker_db()
    top_n = top_n or settings.broker_search_top_n
    where = {"$and": [{"symbol": symbol.upper()},
                      {"chunk_type": {"$in": list(RESEARCH_CHUNK_TYPES)}}]}
    raw = await asyncio.to_thread(store.get, where=where)
    if not raw["ids"]:
        return []
    docs = [Document(page_content=t, metadata=m) for t, m in zip(raw["documents"], raw["metadatas"])]
    dense = await asyncio.to_thread(_dense, store, question, min(settings.rerank_pool, len(docs)), where)
    sparse = await asyncio.to_thread(_bm25, docs, question, settings.bm25_k)
    pool = reciprocal_rank_fusion(dense, sparse, k=settings.rerank_pool, c=settings.rrf_c)
    ranked = await asyncio.to_thread(rerank, question, pool, top_n * settings.rerank_overfetch)
    capped = cap_per_source(ranked, settings.broker_chunks_per_report, top_n)
    return sorted(capped, key=lambda d: str(d.metadata.get("report_date", "")), reverse=True)
