"""Query decomposition (a.k.a. RAG-Fusion / multi-query): split one multi-topic research
request into several FOCUSED sub-queries so each framework/book gets its own retrieval pass."""
import logging

from pydantic import BaseModel, Field

from pennystock.config import settings

log = logging.getLogger(__name__)


class SubQueries(BaseModel):
    queries: list[str] = Field(description="2-4 short, keyword-rich search queries, one per distinct topic")


async def decompose_query(query: str) -> list[str]:
    """Return up to research_subqueries focused sub-queries; fall back to [query] on any failure."""
    from pennystock.llms import decomposer_llm

    prompt = (
        f"Split the research request below into up to {settings.research_subqueries} FOCUSED search "
        "queries, ONE per distinct framework or topic (e.g. one for value/quality investing, one for "
        "behavioral finance, one for technical analysis). Each query must be short and keyword-rich. "
        f"Do NOT add topics the request didn't mention.\n\nREQUEST: {query}"
    )
    try:
        # constructed INSIDE the try: if Ollama is unreachable this must degrade to the
        # original query, not take retrieval down with it
        llm = decomposer_llm().with_structured_output(SubQueries, method="json_schema")
        result = await llm.ainvoke(prompt)
        subs = [q.strip() for q in result.queries if q.strip()][: settings.research_subqueries]
        # ALWAYS retrieve for the original query too. Measured failure: "What conditions confirm a
        # bearish marubozu?" was decomposed into ['value investing', 'behavioral finance',
        # 'technical analysis'] - the decomposer generalised the specific term away and the right
        # chunk was never retrieved. Keeping the original guarantees decomposition can only ADD
        # coverage, never lose it.
        merged = [query] + [q for q in subs if q.lower() != query.lower()]
        log.info(f"Decomposed into {len(subs)} sub-queries (+ original): {subs}")
        return merged
    except Exception as e:                       # never let decomposition break retrieval
        log.warning(f"decompose failed ({e}); using the original query")
        return [query]
