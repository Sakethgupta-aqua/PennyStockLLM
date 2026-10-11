"""The analyst's two broker tools: NUMBERS from one (Layers 1+2), ARGUMENTS from the other (Layer 3).

The databases and models are replaced with real-shaped data built by the real reduction code
(summarize_calls, KpiView), so these tests check what the analyst is actually handed.
"""
import asyncio
from datetime import date

from langchain_core.documents import Document

from pennystock.data.broker_metrics import KpiView
from pennystock.data.consensus import Call, summarize_calls
from pennystock.tools import broker_research as br

AS_OF = date(2026, 10, 2)
CALLS = [Call("ICICI Securities", "2026-09-04", "Buy", 1, 1250.0),
         Call("Axis Direct", "2026-07-27", "Buy", 1, 1225.0)]
GNPA = KpiView(metric="gnpa_pct", period="Q1FY27", period_kind="actual", period_end="2026-06-30",
               median=2.1, low=2.1, high=2.1, n=1, brokers=["ICICI Securities"], labels=["gnpa"],
               latest_report="2026-09-04", status="single")


def _wire(monkeypatch, calls=CALLS, kpis=(GNPA,), passages=()):
    monkeypatch.setattr(br, "_resolve_ticker", lambda s, e="NSE": ("AUBANK.NS", "AU Small Finance Bank"))
    monkeypatch.setattr(br, "_live_price", lambda ticker: 1100.0)
    monkeypatch.setattr(br, "broker_consensus", lambda sym, price=None: {
        **summarize_calls(list(calls), as_of=AS_OF, price=price), "eps_revisions": None})
    monkeypatch.setattr(br, "kpi_snapshot", lambda sym, metrics=None: list(kpis))
    monkeypatch.setattr(br, "filed_figures", lambda ticker: ({}, None))
    monkeypatch.setattr(br, "anchored_estimates", lambda sym, filed, forecasts=2: [])

    async def search(sym, question, top_n=None, store=None):
        return list(passages)
    monkeypatch.setattr(br, "search_broker_passages", search)


def test_the_consensus_is_priced_at_todays_price(monkeypatch):
    """Implied upside is only meaningful against the live price, not the price on the report."""
    _wire(monkeypatch)
    out = asyncio.run(br.get_broker_consensus.ainvoke({"symbol": "AUBANK"}))
    assert "'implied_upside_pct': 12.5" in out                  # median 1,237.5 vs 1,100


def test_kpis_are_labelled_reported_or_forecast(monkeypatch):
    _wire(monkeypatch)
    out = asyncio.run(br.get_broker_consensus.ainvoke({"symbol": "AUBANK"}))
    assert "'kpi': 'GNPA %'" in out and "'kind': 'reported'" in out


def test_no_coverage_is_reported_as_unknown_not_as_silence(monkeypatch):
    """No calls and no KPIs must not read as 'brokers have no concerns'."""
    _wire(monkeypatch, calls=(), kpis=())
    out = asyncio.run(br.get_broker_consensus.ainvoke({"symbol": "AUBANK"}))
    assert "UNKNOWN" in out and "consensus" not in out.split("UNKNOWN")[0]


def test_search_says_when_no_research_is_on_file(monkeypatch):
    _wire(monkeypatch)
    out = asyncio.run(br.search_broker_research.ainvoke({"symbol": "AUBANK", "question": "why buy?"}))
    assert out.startswith("No broker research on file")


def test_search_passages_are_tagged_with_their_source(monkeypatch):
    """The '[source: X]' tag is what lets the trace check a quote against the passage it came from."""
    doc = Document(page_content="AU SF Bank | ICICI Securities | 2026-09-04 | rating BUY.\nCredit cost "
                                "should normalise below 1%.", metadata={"source": "a.pdf"})
    _wire(monkeypatch, passages=[doc])
    out = asyncio.run(br.search_broker_research.ainvoke({"symbol": "AUBANK", "question": "credit cost"}))
    assert out.startswith("[source: a.pdf] AU SF Bank | ICICI Securities")
