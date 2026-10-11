"""The company's own words, for the analyst: passages from its NSE filings, each with type, date and page.

Earnings-call answers, investor presentations, rating rationales, results commentary and the key parts
of annual reports - searched one company at a time (spec: docs/superpowers/specs/2026-10-05-filings-
search-design.md). Words, never numbers: figures come from the financial tools. A company whose filings
were never indexed is reported as NOT INDEXED, never as having disclosed nothing.
"""
from __future__ import annotations

import asyncio
import logging

from langchain_core.tools import tool

from pennystock.data.filing_text import indexed_count
from pennystock.retrieval.filing_search import search_filings as _search
from pennystock.tools.market_data import _resolve_ticker

logger = logging.getLogger(__name__)

DOC_TYPES = ("earnings_call", "investor_presentation", "credit_rating", "results", "annual_report",
             "material_event")
_ALIASES = {"concall": "earnings_call", "call": "earnings_call", "conference_call": "earnings_call",
            "transcript": "earnings_call", "earnings_call_transcript": "earnings_call",
            "presentation": "investor_presentation", "investor_deck": "investor_presentation",
            "rating": "credit_rating", "rating_rationale": "credit_rating", "rating_letter": "credit_rating",
            "result": "results", "financial_results": "results", "quarterly_results": "results",
            "ar": "annual_report", "announcement": "material_event", "event": "material_event"}


def _doc_type(name: str) -> str | None:
    key = "_".join(name.lower().replace("-", " ").split())
    for k in (key, key.removesuffix("s")):
        if k in DOC_TYPES:
            return k
        if k in _ALIASES:
            return _ALIASES[k]
    return None


def doc_type_filter(raw: str) -> tuple[list[str] | None, list[str]]:
    """The tool's comma list read forgivingly - case, spaces, plurals and common names ('concall', 'rating
    rationale'). Names it cannot place are handed back and dropped: a loosely written type must never
    filter every passage out and come back as 'not found in the filings'."""
    known: list[str] = []
    unknown: list[str] = []
    for part in (p.strip() for p in raw.split(",")):
        if not part:
            continue
        name = _doc_type(part)
        if name is None:
            unknown.append(part)
        elif name not in known:
            known.append(name)
    return known or None, unknown


HOW_TO_READ = (
    "Each passage is the company's own filing text; its 'source' line says company · document type · "
    "date · fiscal period · page(s) · section or speaker. Quote with the type and date (e.g. 'Q1 FY27 "
    "earnings call, 2026-07-25, p.7'). Never take a number from a passage - numbers come from the financial "
    "tools. Earnings-call passages are management's words, not facts. If nothing relevant comes back, say it "
    "was not found in the filings on file - not that the company never said it.")


def _view(doc) -> dict:
    head, _, body = doc.page_content.partition("\n")
    m = doc.metadata
    return {"source": head, "text": " ".join(body.split())[:1500], "type": m.get("doc_type"),
            "date": m.get("doc_date"), "pages": f"{m.get('page_from')}-{m.get('page_to')}",
            "file": m.get("path")}


@tool
async def search_filings(symbol: str, question: str, doc_types: str = "", exchange: str = "NSE") -> str:
    """WHAT THE COMPANY ITSELF FILED - passages from its NSE filings, each with type, date and page.

    Searches ONE company's earnings-call transcripts (management's answers to analysts), investor
    presentations, credit-rating rationales, results press releases and the key sections of its annual
    reports (management discussion, directors' report, risks, auditor's report, contingent liabilities,
    related parties). Use for 'what did management say about X', 'why was the rating cut', 'what did the
    company disclose about Y'. `doc_types` optionally narrows to a comma list of earnings_call,
    investor_presentation, credit_rating, results, annual_report, material_event (names it does not
    recognise are ignored and listed back).

    Quote with type and date. Never take a number from a passage. If nothing comes back, say it was not
    found in the filings on file. `symbol` may be a company name or a ticker."""
    try:
        ticker, name = _resolve_ticker(symbol, exchange)
        base = ticker.rsplit(".", 1)[0]
        on_file = await asyncio.to_thread(indexed_count, base)
        if not on_file:
            return str({"company": name, "symbol": base,
                        "filings": "NOT INDEXED - none of this company's filings are in the search index "
                                   "(scripts/index_filings.py). Say its filings could not be searched; do "
                                   "not imply it disclosed or said nothing."})
        types, ignored = doc_type_filter(doc_types)
        docs = await _search(base, question, doc_types=types)
        note = {"doc_types_not_understood": ignored} if ignored else {}
        if not docs:
            return str({"company": name, "symbol": base, "filings_searched": on_file, **note,
                        "passages": "NOT FOUND in the filings on file for this question",
                        "how_to_read": HOW_TO_READ})
        return str({"company": name, "symbol": base, "filings_searched": on_file, **note,
                    "passages": [_view(d) for d in docs], "how_to_read": HOW_TO_READ})
    except Exception as e:
        logger.warning(f"filings search failed for {symbol}: {type(e).__name__}: {e}")
        return f"Error searching the filings of {symbol}."
