"""The analyst's broker-research tools: NUMBERS from one, ARGUMENTS from the other.

Route by question type, the rule financial-RAG research keeps arriving at - numeric questions to
structured data, narrative questions to retrieval:

  get_broker_consensus     ratings, targets, revisions, estimates, sector KPIs   (Layers 1 + 2)
  search_broker_research   "why" - the broker's thesis, in its own words        (Layer 3)

Both are symbol-scoped: the graph pins them to the resolved company like every per-stock tool, so a
broker note about one company can never answer a question about another.
"""
from __future__ import annotations

import asyncio
import logging

from langchain_core.tools import tool

from pennystock.config import settings
from pennystock.data.broker_metrics import anchored_estimates, kpi_snapshot
from pennystock.data.broker_reports import KPI_LABELS
from pennystock.data.consensus import broker_consensus
from pennystock.data.integrated_filings import filed_year
from pennystock.retrieval.broker_search import search_broker_passages
from pennystock.tools.market_data import _resolve_ticker

logger = logging.getLogger(__name__)

HOW_TO_READ = (
    "These are ATTRIBUTED OPINIONS from sell-side brokers, not facts and not this system's view. "
    "Sell-side ratings skew heavily to Buy, and only about 38% of 12-month targets are actually "
    "met. Name the broker and date for every number you use; say how many brokers stand behind a "
    "'consensus'; if `flags` mention dispersion, staleness or a single broker, say the consensus "
    "upside is unreliable. mean_score uses the I/B/E/S scale: 1 = Buy, 3 = Hold, 5 = Sell. "
    "sector_kpis are read from broker tables: 'reported' is a filed figure as the broker printed "
    "it, 'forecast' is the broker's estimate, and `basis` is the row label as printed - brokers "
    "compute ratios like NIM on different bases, so compare like with like. forward_estimates: for a reported "
    "year `filed` is the company's own figure; brokers that print a different figure are on an ADJUSTED basis - "
    "never mix the two, and compare a broker's forecast with its own basis (I/B/E/S majority basis).")


def _live_price(ticker: str) -> float | None:
    from pennystock.data.providers import get_provider
    try:
        return get_provider().get_snapshot(ticker).price
    except Exception as e:                       # upside is a nice-to-have, never a blocker
        logger.warning(f"live price unavailable for {ticker}: {type(e).__name__}: {e}")
        return None


def filed_figures(ticker: str) -> tuple[dict[tuple[str, str], float], str | None]:
    """{(metric, 'FY26'): value} from the company's own accounts: the NSE integrated filing for its year (audited,
    point-in-time), Yahoo's statements for earlier years."""
    from pennystock.data.providers import get_provider
    out: dict[tuple[str, str], float] = {}
    note = None
    try:
        fin = get_provider().get_financials(ticker, settings.financials_years)
        for y, rev, ni in zip(fin.fiscal_years, fin.revenue_cr or [], fin.net_income_cr or []):
            fy = f"FY{y % 100:02d}"
            if rev is not None:
                out[("revenue_cr", fy)] = rev
            if ni is not None:
                out[("net_profit_cr", fy)] = ni
        note = "Yahoo statements"
    except Exception as e:                         # anchoring is a bonus; the estimates still come back
        logger.warning(f"filed figures unavailable for {ticker}: {type(e).__name__}: {e}")
    fy = filed_year(ticker.rsplit(".", 1)[0])
    if fy:
        if fy.revenue_cr is not None:
            out[("revenue_cr", fy.fiscal_year)] = fy.revenue_cr
        if fy.profit_to_owners_cr is not None:
            out[("net_profit_cr", fy.fiscal_year)] = fy.profit_to_owners_cr
        note = f"NSE integrated filing {fy.fiscal_year} ({(fy.broadcast_at or '')[:11]}); Yahoo for earlier years"
    return out, note


def _view(v) -> dict:
    out = {"kpi": KPI_LABELS.get(v.metric, v.metric), "period": v.period,
           "kind": "reported" if v.period_kind == "actual" else "forecast", "value": v.median,
           "brokers": v.n, "agreement": v.status, "basis": v.labels, "as_of": v.latest_report}
    if v.n > 1:
        out["range"] = [v.low, v.high]
    return out


@tool
async def get_broker_consensus(symbol: str, exchange: str = "NSE") -> str:
    """WHAT THE SELL-SIDE SAYS about this stock, as NUMBERS - attributed opinions, never facts.

    Returns:
      consensus - each broker's LATEST call (one vote per broker; calls older than ~6 months are
        excluded as stale): rating, target, date, its previous call, and how often that broker
        rates anything bullish; the rating mix and mean (1 = Buy ... 5 = Sell); target median,
        range and DISPERSION, and implied upside at today's price; 90-day revisions (target
        raises/cuts, upgrades/downgrades), the drift in the consensus, EPS estimate revisions,
        and `flags`.
      forward_estimates - the brokers' fiscal-year forecasts (EPS, revenue, profit), median and range.
      sector_kpis - the operating metrics the sector is judged on (NIM, GNPA, CRAR for a lender),
        read from broker tables, marked REPORTED or FORECAST, with the basis each broker used.

    Use for 'what do brokers think', targets, ratings, estimates, upgrades/downgrades, and a
    lender's asset quality. Attribute every number (broker, date). Never present a broker target
    as your own forecast. If nothing comes back, say broker coverage is UNKNOWN - never that
    brokers are positive. `symbol` may be a company name or a ticker."""
    try:
        ticker, name = _resolve_ticker(symbol, exchange)
        base = ticker.rsplit(".", 1)[0]
        price = await asyncio.to_thread(_live_price, ticker)
        consensus = await asyncio.to_thread(broker_consensus, base, price)
        kpis = await asyncio.to_thread(kpi_snapshot, base)
        filed, filed_note = await asyncio.to_thread(filed_figures, ticker)
        estimates = await asyncio.to_thread(anchored_estimates, base, filed)
        if not (consensus["calls"] or consensus["stale_excluded"] or kpis or estimates):
            return str({"company": name, "symbol": base,
                        "broker_coverage": "UNKNOWN - no broker report or call is on file for "
                                           "this company. Say so; do not imply brokers are "
                                           "positive or negative."})
        return str({"company": name, "symbol": base, "price_used_for_upside": price,
                    "consensus": consensus,
                    "forward_estimates": estimates, "filed_figures_source": filed_note,
                    "sector_kpis": [_view(v) for v in kpis],
                    "how_to_read_this": HOW_TO_READ})
    except Exception as e:
        logger.warning(f"broker consensus failed for {symbol}: {type(e).__name__}: {e}")
        return f"Error fetching broker consensus for {symbol}."


@tool
async def search_broker_research(symbol: str, question: str, exchange: str = "NSE") -> str:
    """THE BROKERS' ARGUMENTS for this stock - why they rate it as they do, in their own words.

    Searches ONLY this company's broker research (never another company's, never the investing
    books) and returns the most relevant passages, newest report first, each stamped with broker,
    date and rating. Use for 'why' questions: the investment thesis, the risks a broker sees, what
    changed in a downgrade, what management told the broker.

    NEVER take a number from here - targets, ratings, estimates and KPIs come from
    get_broker_consensus. Quote a passage only with its broker and date. If nothing comes back,
    say no broker research is on file for this company. `question` is what you want explained."""
    try:
        ticker, name = _resolve_ticker(symbol, exchange)
        base = ticker.rsplit(".", 1)[0]
        docs = await search_broker_passages(base, question)
        if not docs:
            return (f"No broker research on file for {name or base}. Its broker notes have not "
                    "been downloaded or ingested; say so rather than describing a thesis.")
        return "\n\n".join(f"[source: {d.metadata.get('source', '?')}] {d.page_content}" for d in docs)
    except Exception as e:
        logger.warning(f"broker research search failed for {symbol}: {type(e).__name__}: {e}")
        return f"Error searching broker research for {symbol}."
