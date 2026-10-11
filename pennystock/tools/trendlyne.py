"""Trendlyne's view of a stock, for the analyst: a third-party vendor's scores, attributed - never ours.

Trendlyne scores every stock 0-100 on Durability, Valuation and Momentum, runs a pass/fail checklist
and aggregates far more analysts than our broker notes (AU Small Finance Bank: 28 vs 2). The tool
hands over the latest Stock Report on file, what moved since the previous one, and how to read it.
With no report on file the answer is UNKNOWN - never a default.
Spec: docs/superpowers/specs/2026-10-04-stock-report-reader-design.md
"""
from __future__ import annotations

import asyncio
import logging

from langchain_core.tools import tool

from pennystock.data.trendlyne_scores import pe_zone_meaning, view
from pennystock.tools.market_data import _resolve_ticker

logger = logging.getLogger(__name__)

SOURCE = ("Trendlyne Stock Report - a third-party vendor's scores and analyst aggregate, "
          "NOT this system's analysis")
HOW_TO_READ = (
    "Scores are 0-100. Durability = financial strength (revenue, cash flow, debt) and Valuation = "
    "price against P/E and P/BV are LONG-term; Momentum (30+ technical indicators, recomputed daily) is "
    "SHORT-term. Bands: good when D > 55, V > 50, M >= 60; bad when D < 35, V < 30, M < 35. "
    "'classification' is Trendlyne's name for the mix. 'lean' is how this system's ledger grades "
    "Trendlyne's view (bullish = two good scores and no bad one). consensus/target_price/analysts are "
    "Trendlyne's aggregate of sell-side analysts - opinions. pe_zone_pct is the share of its history "
    "the stock's P/E spent BELOW today's P/E: low = cheap against its own past ('Strong Buy' zone), "
    "high = dear ('Sell'/'Strong Sell'); it assumes the P/E reverts to its mean. Trendlyne's "
    "claim that high scorers beat the index is unverified. Attribute every number to Trendlyne and "
    "the report date (as_of); never present it as your own view.")


@tool
async def get_trendlyne_view(symbol: str, exchange: str = "NSE") -> str:
    """TRENDLYNE'S SCORES for this stock - a third-party vendor's opinion, never this system's view.

    Returns the latest Trendlyne Stock Report on file: Durability (financial strength), Valuation
    (P/E, P/BV) and Momentum (technicals) scores 0-100 with Trendlyne's labels and their bands; the
    classification (e.g. 'Falling Comet'); the checklist pass rate; the consensus of N analysts with
    average target and implied upside; the P/E buy/sell zone; the lean this system's ledger grades;
    and what changed since the previous report.

    Use for 'what is the DVM score', 'what does Trendlyne say', and durability / valuation /
    momentum score questions, and as a second opinion beside your own analysis. Attribute every
    number to Trendlyne and the report date. If nothing comes back, say Trendlyne's view is
    UNKNOWN. `symbol` may be a company name or a ticker."""
    try:
        ticker, name = _resolve_ticker(symbol, exchange)
        base = ticker.rsplit(".", 1)[0]
        v = await asyncio.to_thread(view, base)
        if v is None:
            return str({"company": name, "symbol": base,
                        "trendlyne_view": "UNKNOWN - no Trendlyne Stock Report is on file for this "
                                          "company. Say Trendlyne's scores are unknown; do not guess them."})
        meaning = pe_zone_meaning(v.get("pe_zone"), v.get("pe_zone_pct"))
        if meaning:
            v = {**v, "pe_zone_meaning": meaning}
        return str({"company": name, "symbol": base, "source": SOURCE, "trendlyne_view": v,
                    "how_to_read": HOW_TO_READ})
    except Exception as e:
        logger.warning(f"Trendlyne view failed for {symbol}: {type(e).__name__}: {e}")
        return f"Error fetching Trendlyne's view for {symbol}."
