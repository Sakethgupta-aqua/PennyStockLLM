"""Pure functions that turn raw financial statements into a compact, LLM-ready trend summary,
plus lightweight data-quality guards. Computed in code (never in the LLM's head), and
unit-tested. Parallels indicators.py, but for fundamentals instead of price."""
from __future__ import annotations

from pennystock.data.providers import Financials, Fundamentals


def _clean(series: list) -> list:
    return [v for v in series if v is not None]


def _fy(year: int) -> str:
    return f"FY{year % 100:02d}"


def _by_year(years: list, values: list | None) -> dict[str, float]:
    """Each value under its fiscal year, empty years dropped. Parallel lists whose first slot was empty made the
    analyst shift every year by one (DIXON, 2026-10-11: FY23's revenue reported as FY22's)."""
    return {_fy(y): v for y, v in zip(years, values or []) if v is not None}


def _growth(years: list, values: list | None) -> dict | None:
    """Compound annual growth between the first and last years that have a value, over the REAL year span (a gap
    year no longer shortens it), with both ends named so a '5-year CAGR' can never stand for three years."""
    points = [(y, v) for y, v in zip(years, values or []) if v is not None]
    if len(points) < 2:
        return None
    (y0, v0), (y1, v1) = points[0], points[-1]
    if v0 <= 0 or v1 <= 0 or y1 <= y0:
        return None
    return {"pct": round(((v1 / v0) ** (1 / (y1 - y0)) - 1) * 100, 1), "from": _fy(y0), "to": _fy(y1),
            "years": y1 - y0}


def _trend(series: list, turn_pct: float = 10.0) -> str:
    """Describe the SHAPE of a series, not just its endpoints.

    Comparing only first vs last hides a turn. Cipla's net income went
    2802 -> 4122 -> 5273 -> 3879: endpoints say "rising", but it actually peaked and fell
    26%. We therefore flag a latest value that is >= turn_pct away from the peak (for a
    rising series) or from the trough (for a falling one).
    """
    vals = _clean(series)
    if len(vals) < 2:
        return "n/a"
    first, last = vals[0], vals[-1]
    direction = "rising" if last > first else ("falling" if last < first else "flat")

    peak, trough = max(vals), min(vals)
    if direction == "rising" and peak and vals.index(peak) < len(vals) - 1:
        off_peak = (peak - last) / abs(peak) * 100
        if off_peak >= turn_pct:
            return f"rising overall but latest is {off_peak:.0f}% BELOW the peak"
    if direction == "falling" and trough and vals.index(trough) < len(vals) - 1:
        off_trough = (last - trough) / abs(trough) * 100
        if off_trough >= turn_pct:
            return f"falling overall but latest is {off_trough:.0f}% ABOVE the trough"
    return direction


def summarize_financials(fin: Financials) -> dict:
    """5-year statements -> compact trend summary."""
    if not fin.fiscal_years or not _clean(fin.revenue_cr):
        return {"financials": "unavailable"}
    ni = fin.net_income_cr
    counted = len(_clean(ni))
    profitable = sum(1 for v in ni if v is not None and v > 0)
    de_latest = None
    if fin.total_debt_cr and fin.equity_cr:
        d, e = fin.total_debt_cr[-1], fin.equity_cr[-1]
        if d is not None and e not in (None, 0):
            de_latest = round(d / e * 100, 1)

    # Quality metrics computed from the statements (yfinance's .info ROE is often None).
    ni_latest = ni[-1] if ni else None
    eq_latest = fin.equity_cr[-1] if fin.equity_cr else None
    ebit_latest = fin.operating_income_cr[-1] if fin.operating_income_cr else None
    debt_latest = fin.total_debt_cr[-1] if fin.total_debt_cr else None
    roe = round(ni_latest / eq_latest * 100, 1) if ni_latest is not None and eq_latest not in (None, 0) else None
    cap_employed = (eq_latest + debt_latest) if eq_latest is not None and debt_latest is not None else None
    roce = round(ebit_latest / cap_employed * 100, 1) if ebit_latest is not None and cap_employed not in (None, 0) else None

    years = fin.fiscal_years
    return {
        "latest_year": _fy(years[-1]),
        "revenue_cr": _by_year(years, fin.revenue_cr),
        "revenue_cagr": _growth(years, fin.revenue_cr),
        "revenue_trend": _trend(fin.revenue_cr),
        "net_income_cr": _by_year(years, ni),
        "net_income_trend": _trend(ni),
        "profitable_years": f"{profitable}/{counted}" if counted else "n/a",
        "eps": _by_year(years, fin.eps),
        "total_debt_cr": _by_year(years, fin.total_debt_cr),
        "debt_trend": _trend(fin.total_debt_cr),
        "debt_to_equity_pct_latest": de_latest,
        "roe_pct_latest": roe,          # net income / equity
        "roce_pct_latest": roce,        # EBIT / (equity + debt) - the quality bar (Coffee Can wants >=15%)
        "free_cash_flow_cr": _by_year(years, fin.free_cash_flow_cr),
    }


def snapshot_warnings(f: Fundamentals) -> list[str]:
    """Cheap contradiction checks on the single-point .info snapshot."""
    w = []
    if f.profit_margin is not None and f.eps_trailing is not None:
        if f.profit_margin > 0 and f.eps_trailing < 0:
            w.append("Positive profit margin but NEGATIVE trailing EPS - snapshot fundamentals "
                     "are inconsistent (often stale or mixed across a corporate action). Treat with caution.")
    if f.profit_margin is not None and f.return_on_equity is not None:
        if f.profit_margin > 0 and f.return_on_equity < 0:
            w.append("Positive profit margin but NEGATIVE ROE - verify fundamentals before relying on them.")
    return w
