"""The evidence pack: the analyst's data, gathered and computed by code before the model writes a word.

Why (measured): one 16-tool question produced 23 raw tool results - 22,724 tokens, more than the window - and
even with room to spare the model converted units, lined up years and read percentiles wrongly (DIXON,
2026-10-11). Context engineering says give the model the smallest set of high-signal tokens, so code runs the
structured tools (the same tools the analyst has: one source of truth), and `fact_sheet.render_fact_sheet`
writes one sheet with every comparison made. The model explains and judges; it calls tools only for text
(filings, broker reasoning, books) or for what the sheet lists under DATA GAPS.
"""
from __future__ import annotations

import ast
import asyncio
import logging
import re
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime

from pennystock.agents.fact_sheet import render_fact_sheet
from pennystock.config import settings
from pennystock.data.integrated_filings import filed_year
from pennystock.tools import (get_annual_report_notes, get_broker_consensus, get_corporate_actions,
                              get_corporate_events, get_financials, get_forensics, get_management_commentary,
                              get_price_history, get_shareholding, get_stock_fundamentals, get_trendlyne_view,
                              get_valuation_context, screen_stocks)

logger = logging.getLogger(__name__)

PACK_TOOLS = (get_stock_fundamentals, get_price_history, get_financials, get_forensics, get_valuation_context,
              get_shareholding, get_corporate_events, get_corporate_actions, get_annual_report_notes,
              get_broker_consensus, get_trendlyne_view, get_management_commentary)
ALTERNATIVES = re.compile(r"\b(instead|alternatives?|shortlist|other\s+stocks|peers?\s+to\s+consider|"
                          r"similar\s+stocks|which\s+stocks)\b", re.I)


@dataclass
class PackCall:
    name: str
    args: dict
    output: str
    data: dict | None
    ok: bool
    seconds: float


@dataclass
class EvidencePack:
    symbol: str
    company: str | None
    fact_sheet: str
    calls: list[PackCall]
    built_at: str


def parse_output(text: str) -> dict | None:
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def wants_alternatives(question: str) -> bool:
    return bool(ALTERNATIVES.search(question or ""))


async def _timed(name: str, args: dict, run, sem: asyncio.Semaphore) -> PackCall:
    async with sem:
        started = time.time()
        try:
            out = str(await asyncio.wait_for(run(), timeout=settings.evidence_tool_timeout_s))
        except Exception as e:                       # one slow or broken source never sinks the pack
            out = f"Error: {name} failed ({type(e).__name__}: {e})"
        data = parse_output(out)
        return PackCall(name, args, out, data, data is not None and not out.lstrip().startswith("Error"),
                        round(time.time() - started, 1))


async def gather(symbol: str, question: str) -> list[PackCall]:
    sem = asyncio.Semaphore(settings.evidence_concurrency)
    jobs = [_timed(t.name, {"symbol": symbol}, lambda t=t: t.ainvoke({"symbol": symbol}), sem)
            for t in PACK_TOOLS]

    async def filed() -> str:
        fy = await asyncio.to_thread(filed_year, symbol)
        return str(asdict(fy)) if fy else "Error: no integrated filing on record"
    jobs.append(_timed("nse_integrated_filing", {"symbol": symbol}, filed, sem))
    calls = list(await asyncio.gather(*jobs))
    if wants_alternatives(question):
        fund = next((c.data for c in calls if c.name == "get_stock_fundamentals" and c.data), None) or {}
        args = {"sector": fund.get("sector") or "", "top_n": settings.evidence_alternatives}
        calls.append(await _timed("screen_stocks", args, lambda: screen_stocks.ainvoke(args), sem))
    return calls


async def build_pack(symbol: str, question: str) -> EvidencePack:
    calls = await gather(symbol, question)
    company = next((c.data.get("company") for c in calls if c.data and c.data.get("company")), None)
    sheet = render_fact_sheet(symbol, company, {c.name: c for c in calls}, today=date.today())
    logger.info(f"Evidence pack for {symbol}: {sum(c.ok for c in calls)}/{len(calls)} sources, "
                f"{len(sheet):,} chars, slowest {max(c.seconds for c in calls):.0f}s")
    return EvidencePack(symbol, company, sheet, calls, datetime.now().isoformat(timespec="seconds"))
