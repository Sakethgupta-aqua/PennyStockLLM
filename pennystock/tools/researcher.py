"""FinancialAnalystAgent's tool: advanced RAG search over the book corpus.

Uses the full retrieval pipeline (multi-query -> hybrid -> RRF-fuse -> cross-encoder rerank ->
lost-in-the-middle reorder) so a multi-framework question reliably retrieves EVERY relevant book.
"""
import logging

from langchain_core.tools import tool

from pennystock.retrieval.store import advanced_retrieve

logger = logging.getLogger(__name__)


@tool
async def penny_stock_researcher(query: str) -> str:
    """Searches the book knowledge base and returns the most relevant passages (source-tagged)."""
    logger.info(f"Advanced RAG search for: {query}")
    docs = await advanced_retrieve(query)
    return "\n\n".join(f"[source: {d.metadata.get('source', '?')}] {d.page_content}" for d in docs)
