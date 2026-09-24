"""Valuation in its OWN historical context - the second half of "versus what?".

Knowing a stock trades at 33x earnings is only half the question. Is 33x its normal multiple, or
a 10-year peak? Completely different conclusions from the same number. This builds a P/E and P/B
band from the last few fiscal years and places today's multiple inside it.

Indian fiscal years run April-March, so FY2026 means April 2025 to March 2026; we map each
year's EPS onto the prices that were actually trading during that year.
"""
from __future__ import annotations

import logging
from statistics import median

from pennystock.data.providers import Financials

logger = logging.getLogger(__name__)


def _fy_window(fy: int) -> tuple[str, str]:
    """FY2026 -> ('2025-04-01', '2026-03-31') - the Indian fiscal year."""
    return f"{fy - 1}-04-01", f"{fy}-03-31"


def valuation_bands(fin: Financials, price_history, current_pe: float | None = None,
                    current_pb: float | None = None, book_value: float | None = None) -> dict:
    """Historic P/E (and P/B where book value is known) vs today's multiple."""
    if price_history is None or getattr(price_history, "empty", True) or not fin.fiscal_years:
        return {"valuation_history": "unavailable", "note": "need price history and annual EPS"}

    closes = price_history["Close"].dropna()
    if closes.empty:
        return {"valuation_history": "unavailable", "note": "no usable price history"}

    yearly_pe: list[dict] = []
    for fy, eps in zip(fin.fiscal_years, fin.eps):
        if not eps or eps <= 0:                       # a loss year has no meaningful P/E
            continue
        start, end = _fy_window(fy)
        try:
            window = closes.loc[start:end]
        except Exception:
            continue
        if window.empty:
            continue
        yearly_pe.append({"fy": fy, "eps": eps,
                          "median_price": round(float(window.median()), 2),
                          "pe": round(float(window.median()) / eps, 1)})

    if len(yearly_pe) < 2:
        return {"valuation_history": "insufficient",
                "note": "fewer than 2 profitable years with overlapping price history"}

    pes = [y["pe"] for y in yearly_pe]
    lo, mid, hi = round(min(pes), 1), round(median(pes), 1), round(max(pes), 1)
    out = {"pe_band": {"low": lo, "median": mid, "high": hi, "years": len(pes)},
           "pe_by_year": yearly_pe}

    if current_pe:
        pos = ("BELOW its historical range" if current_pe < lo else
               "ABOVE its historical range" if current_pe > hi else
               "in the upper half of its range" if current_pe > mid else
               "in the lower half of its range")
        out["current_pe"] = round(current_pe, 1)
        out["vs_own_history_pct"] = round((current_pe / mid - 1) * 100, 1) if mid else None
        out["verdict"] = (f"Current P/E {round(current_pe, 1)} is {pos} "
                          f"(own {len(pes)}-year band {lo}-{hi}, median {mid}).")

    if current_pb and book_value:
        out["current_pb"] = round(current_pb, 2)
    return out
