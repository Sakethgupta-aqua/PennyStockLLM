"""Vector store + hybrid retriever wiring.

Heavy objects (embeddings, Chroma, the BM25 index) are built lazily and cached, so simply
importing this module — or compiling the graph — stays fast and does not load the DB.
The DB is loaded on the first actual retrieval.
"""
import asyncio
from functools import lru_cache

from langchain_core.documents import Document

from pennystock.config import settings
from pennystock.retrieval.bm25 import LangchainBM25sRetriever
from pennystock.retrieval.fusion import reciprocal_rank_fusion


@lru_cache(maxsize=1)
def _embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    from pennystock.devices import torch_device, torch_dtype

    device = torch_device()
    emb = HuggingFaceEmbeddings(
        model_name=settings.embed_model,
        model_kwargs={"device": device, "model_kwargs": {"torch_dtype": torch_dtype(device)}},
        encode_kwargs={"normalize_embeddings": True},
    )
    # bge-m3 accepts 8,192 tokens, but a batch of 32 garbled 3,819-token passages asked the Mac's GPU for one
    # 29.9 GB buffer and aborted the filings index (2026-10-06); real passages stay under the cap
    emb._client.max_seq_length = settings.embed_max_tokens
    return emb


@lru_cache(maxsize=1)
def _vector_db():
    from langchain_chroma import Chroma

    return Chroma(
        persist_directory=str(settings.vector_db_dir),
        embedding_function=_embeddings(),
        collection_name=settings.collection,
    )


@lru_cache(maxsize=1)
def _retrievers():
    """Build the dense (MMR) + sparse (BM25) retrievers once, from the persisted DB."""
    vdb = _vector_db()
    dense = vdb.as_retriever(
        search_type="mmr", search_kwargs={"k": settings.dense_k, "fetch_k": settings.dense_fetch_k}
    )
    raw = vdb.get()
    docs = [Document(page_content=t, metadata=m) for t, m in zip(raw["documents"], raw["metadatas"])]
    sparse = LangchainBM25sRetriever.from_documents(docs, k=settings.bm25_k)
    return dense, sparse


async def hybrid_retrieve(query: str, k: int | None = None, c: int | None = None) -> list[Document]:
    """Run dense + sparse retrieval in parallel and fuse them (RRF)."""
    dense, sparse = _retrievers()
    dense_docs, sparse_docs = await asyncio.gather(
        dense.ainvoke(query),
        asyncio.to_thread(sparse.invoke, query),
    )
    return reciprocal_rank_fusion(dense_docs, sparse_docs, k=k or settings.rrf_k, c=c or settings.rrf_c)


async def advanced_retrieve(query: str) -> list[Document]:
    """Full pipeline: decompose -> hybrid -> RRF-fuse -> rerank -> per-source cap -> reorder.

    Broad recall (multi-query + big candidate pool) then high precision (reranker) then LLM-friendly
    ordering. The reranker scores against the ORIGINAL query (true intent), not the sub-queries.
    """
    from pennystock.retrieval.fusion import reciprocal_rank_fusion_many
    from pennystock.retrieval.query_expand import decompose_query
    from pennystock.retrieval.rerank import cap_per_source, rerank, reorder_lost_in_middle

    subqueries = await decompose_query(query)
    lists = await asyncio.gather(*(hybrid_retrieve(q, k=settings.candidate_k) for q in subqueries))
    pool = reciprocal_rank_fusion_many(lists, k=settings.rerank_pool)          # broad, deduped candidates
    deep = settings.rerank_top_n * settings.rerank_overfetch
    reranked = await asyncio.to_thread(rerank, query, pool, deep)          # precision (off-thread)
    diverse = cap_per_source(reranked, settings.max_chunks_per_source,     # no book monopolises
                             settings.rerank_top_n)
    return reorder_lost_in_middle(diverse)
