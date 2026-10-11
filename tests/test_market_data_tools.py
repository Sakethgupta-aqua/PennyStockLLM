"""The market-data tools compute what the analyst would otherwise compute in its head."""
import ast
import asyncio
from dataclasses import dataclass

from pennystock.tools import market_data as md


@dataclass
class _Fund:
    symbol: str = "ABC.NS"
    company: str = "Abc Ltd"
    pe_trailing: float | None = 37.35


def _fundamentals(monkeypatch, pe):
    class _Provider:
        def get_fundamentals(self, ticker):
            return _Fund(pe_trailing=pe)
    monkeypatch.setattr(md, "_resolve_ticker", lambda s, e: ("ABC.NS", "Abc Ltd"))
    monkeypatch.setattr(md, "get_provider", lambda: _Provider())
    monkeypatch.setattr(md, "snapshot_warnings", lambda fund: [])
    return ast.literal_eval(asyncio.run(md.get_stock_fundamentals.ainvoke({"symbol": "ABC"})))


def test_the_earnings_yield_is_computed_for_grahams_test(monkeypatch):
    # the live DIXON report worked out 1/37 = 2.68% itself - a number no tool had given it
    assert _fundamentals(monkeypatch, 37.35)["earnings_yield_pct"] == 2.68


def test_no_earnings_yield_without_a_positive_pe(monkeypatch):
    assert _fundamentals(monkeypatch, None)["earnings_yield_pct"] is None
    assert _fundamentals(monkeypatch, -12.0)["earnings_yield_pct"] is None
