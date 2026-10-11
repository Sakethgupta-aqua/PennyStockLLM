"""The analyst's filings search: passages attributed by type, date and page; NOT INDEXED is never
"nothing said"."""
import asyncio

from langchain_core.documents import Document

from pennystock.tools import filings as tool


def test_a_company_with_nothing_indexed_is_reported_as_such(monkeypatch):
    monkeypatch.setattr(tool, "_resolve_ticker", lambda s, e: ("XYZ.NS", "Xyz Ltd"))
    monkeypatch.setattr(tool, "indexed_count", lambda s: 0)
    out = asyncio.run(tool.search_filings.ainvoke({"symbol": "XYZ", "question": "capex plans"}))
    assert "NOT INDEXED" in out and "nothing disclosed" not in out.lower()

def test_passages_come_back_attributed_with_the_numbers_rule(monkeypatch):
    monkeypatch.setattr(tool, "_resolve_ticker", lambda s, e: ("AUBANK.NS", "AU Small Finance Bank"))
    monkeypatch.setattr(tool, "indexed_count", lambda s: 12)
    async def fake(*a, **k):
        return [Document(page_content="AU Small Finance Bank · earnings call · 2026-07-25 · Q1 FY27 · p.7\n"
                                      "NIM 5.8-6.0%",
                         metadata={"doc_type": "earnings_call", "doc_date": "2026-07-25", "page_from": 7,
                                   "page_to": 7})]
    monkeypatch.setattr(tool, "_search", fake)
    out = asyncio.run(tool.search_filings.ainvoke({"symbol": "AUBANK", "question": "NIM guidance"}))
    assert "2026-07-25" in out and "p.7" in out and "how_to_read" in out

def test_loosely_written_document_types_still_filter_and_unknown_ones_never_empty_the_search(monkeypatch):
    monkeypatch.setattr(tool, "_resolve_ticker", lambda s, e: ("AUBANK.NS", "AU Small Finance Bank"))
    monkeypatch.setattr(tool, "indexed_count", lambda s: 12)
    asked = []

    async def fake(symbol, question, doc_types=None):
        asked.append(doc_types)
        return []
    monkeypatch.setattr(tool, "_search", fake)
    loose = "Earnings Calls, concall, rating rationale, bogus"
    out = asyncio.run(tool.search_filings.ainvoke({"symbol": "AUBANK", "question": "NIM guidance",
                                                  "doc_types": loose}))
    assert asked[-1] == ["earnings_call", "credit_rating"]
    assert "bogus" in out                        # the analyst is told which names were not understood
    asyncio.run(tool.search_filings.ainvoke({"symbol": "AUBANK", "question": "NIM guidance",
                                             "doc_types": "bogus"}))
    assert asked[-1] is None                     # nothing understood -> every type is searched, never none
