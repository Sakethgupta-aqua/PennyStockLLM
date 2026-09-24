"""Universe-level tool: find candidates, as opposed to analysing one named company.

Kept apart from market_data.py because it is a different KIND of tool. Everything there answers
"tell me about THIS company"; this answers "which companies are worth looking at at all", and it
therefore takes no symbol and is not symbol-scoped by the graph.
"""
import asyncio
import logging

from langchain_core.tools import tool

from pennystock.data.metrics import load_metrics
from pennystock.data.screener import screen

logger = logging.getLogger(__name__)
MAX_RESULTS = 25


def _run(sector: str | None, min_market_cap_cr: float, min_turnover_cr: float, top_n: int) -> dict:
    rows = load_metrics()
    if not rows:
        return {"screen": "unavailable",
                "note": "The screening cache is empty. It is populated offline by "
                        "scripts/ingest_metrics.py; say so rather than inventing candidates."}
    if sector:
        rows = [r for r in rows if (r.sector or "").lower() == sector.lower()]
        if not rows:
            return {"screen": "no cached stocks in that sector",
                    "sectors_available": sorted({r.sector for r in load_metrics() if r.sector})}
    f = screen(rows, min_market_cap_cr=min_market_cap_cr, min_turnover_cr=min_turnover_cr)
    return {
        "funnel": {"universe": f.universe, "tradeable": f.after_liquidity,
                   "survived_veto_gates": f.after_vetoes, "removed_by": f.veto_reasons},
        "candidates": [{
            "symbol": r.metrics.symbol, "company": r.metrics.company, "sector": r.metrics.sector,
            "score_out_of_100": r.score, "pillars": r.pillars,
            "market_cap_cr": r.metrics.market_cap_cr, "turnover_cr_per_day": r.metrics.turnover_cr,
            "roce_pct": r.metrics.roce_pct, "debt_to_ebitda": r.metrics.debt_to_ebitda,
            "pe": r.metrics.pe, "red_flag_count": r.metrics.red_flag_count,
            "data_completeness": r.completeness,
        } for r in f.results[: min(top_n, MAX_RESULTS)]],
        "how_to_read_this": "score_out_of_100 is a PERCENTILE RANK against the other survivors, "
                            "not an absolute quality mark - the top name in a weak universe is "
                            "still a weak name. Stocks failing a veto gate (severe promoter pledge, "
                            "interest cover under 1.5x, distress zone, manipulation flag, poor cash "
                            "conversion, debt/EBITDA over 4.5x) are REMOVED before scoring, which is "
                            "why the count drops. A shortlist is a starting point for research: "
                            "analyse individual names with the per-stock tools before saying "
                            "anything about them.",
    }


@tool
async def screen_stocks(sector: str = "", min_market_cap_cr: float = 500.0,
                        min_turnover_cr: float = 1.0, top_n: int = 15) -> str:
    """STOCK DISCOVERY - rank the cached universe when the user has NOT named a specific company.

    Use for "find me quality mid-caps", "which cheap stocks are safe", "screen for low debt". Do
    NOT use when the user asked about a named company - use the per-stock tools for that.

    Runs the funnel a professional Indian desk runs, in order:
      1. TRADEABLE - market cap and, more importantly, median daily TURNOVER in rupees. A stock you
         cannot exit is not an investment however good the business.
      2. VETO GATES - severe promoter pledge, interest cover under 1.5x, Altman EM distress zone,
         Beneish manipulation flag, weak cash conversion, debt/EBITDA over 4.5x. These REMOVE a
         name; they are not deductions, because no multiple is cheap enough to rescue accounts you
         cannot trust.
      3. RANK - survivors scored on quality 30%, safety 25%, governance 20%, valuation 15%,
         momentum 10%, each as a percentile against the other survivors.

    Report the funnel counts, not just the winners - how many names were removed and why is the
    most informative part of any screen. Args: sector (optional, e.g. "Basic Materials"),
    min_market_cap_cr, min_turnover_cr (Rs crore of median daily traded value), top_n."""
    try:
        out = await asyncio.to_thread(_run, sector or None, min_market_cap_cr,
                                      min_turnover_cr, top_n)
        return str(out)
    except Exception as e:
        logger.warning(f"screen failed: {type(e).__name__}: {e}")
        return "Error running the stock screen."
