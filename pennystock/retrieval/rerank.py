"""Cross-encoder reranking + lost-in-the-middle reordering.

The reranker reads each (query, chunk) pair JOINTLY and re-scores by true relevance - the
single biggest precision lever after hybrid retrieval. Runs locally on MPS/CPU.
"""
from functools import lru_cache

from langchain_core.documents import Document

from pennystock.config import settings


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder

    from pennystock.devices import torch_device, torch_dtype

    device = torch_device()
    return CrossEncoder(settings.reranker_model, device=device, max_length=settings.rerank_max_tokens,
                        model_kwargs={"torch_dtype": torch_dtype(device)})


def rerank(query: str, docs: list[Document], top_n: int) -> list[Document]:
    """Score (query, chunk) pairs with the cross-encoder; return the top_n by true relevance.

    Blocking (model inference) - call via asyncio.to_thread from async code.
    """
    if not docs:
        return []
    scores = _reranker().predict([(query, d.page_content) for d in docs])
    ranked = sorted(zip(docs, scores), key=lambda p: p[1], reverse=True)
    out = []
    for d, s in ranked[:top_n]:
        d.metadata["rerank_score"] = round(float(s), 4)
        out.append(d)
    return out


def cap_per_source(docs: list[Document], max_per_source: int, limit: int) -> list[Document]:
    """Take the best `limit` docs, allowing at most `max_per_source` from any one book.

    Without this, a large book monopolises the context: on a Cipla run all 8 chunks came from
    The Intelligent Investor (38% of the corpus), crowding out a penny-stock passage that
    answered the question directly. Docs must arrive best-first. Overflow beyond the cap is
    kept aside and used to backfill only if there are not enough distinct sources to fill
    `limit`, so we never return fewer chunks than asked for.
    """
    kept: list[Document] = []
    spill: list[Document] = []
    counts: dict[str, int] = {}
    for d in docs:
        src = str(d.metadata.get("source", "?"))
        if counts.get(src, 0) < max_per_source:
            counts[src] = counts.get(src, 0) + 1
            kept.append(d)
            if len(kept) >= limit:
                return kept
        else:
            spill.append(d)
    for d in spill:                       # thin corpus -> backfill with the best overflow
        if len(kept) >= limit:
            break
        kept.append(d)
    return kept


def reorder_lost_in_middle(docs: list[Document]) -> list[Document]:
    """docs come in best->worst; put the strongest at BOTH ends, weakest in the middle
    (LLMs attend most to the start and end of a long context)."""
    front, back = [], []
    for i, d in enumerate(docs):
        (front if i % 2 == 0 else back).append(d)
    return front + back[::-1]
