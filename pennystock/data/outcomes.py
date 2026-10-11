"""Grading a prediction against what the market actually did.

The arithmetic here looks trivial and is not. Five things were measured on live data on
2026-09-27, and each one breaks the obvious implementation:

  1. THE PAST CHANGES. Yahoo applies dividends backwards; ITC's 2023-09-25 close reads
     426.19 raw and 379.91 adjusted. So the ENTRY price comes from the ledger, captured
     on the day, and is never re-fetched here. Returns are computed from ADJUSTED closes
     for stock and benchmark alike, so dividends are counted consistently on both sides.
  2. SYMBOLS VANISH. TATAMOTORS.NS returns 404 after the demerger. A missing series is
     labelled `unavailable`; it never crashes the run and never deletes the prediction.
  3. TRADING DAYS ARE NOT CALENDAR DAYS. +20 trading days measured out at +34 calendar
     days. Horizons are counted by walking the price series, never by date arithmetic.
  4. THE STOCK AND THE INDEX TRADE ON DIFFERENT DAYS. Over six months RELIANCE had 129
     sessions and Nifty 500 had 125. The two series are inner-joined; unmatched days are
     dropped, never interpolated.
  5. RAW RETURN LIES. RELIANCE fell 1.71% in a window where Nifty 500 rose 5.99%. Raw
     return says "down a bit"; the abnormal return of -7.71% says what really happened.

Model choice: the MARKET-ADJUSTED model (`abnormal = r_stock - r_benchmark`), not the
market model with an estimated beta. Event-study work finds the two perform similarly
over short windows, and skipping beta removes an estimation-window subsystem for no
measurable loss at 1-60 day horizons.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from functools import lru_cache

import pandas as pd

from pennystock.config import settings
from pennystock.data.ledger import Outcome, Prediction, load_predictions, pending, save_outcomes

logger = logging.getLogger(__name__)

# Only four NSE indices return usable history on this feed. Tested 2026-09-27: Nifty 500,
# Bank, IT and Pharma returned 23 rows over one month; AUTO, FMCG, METAL, ENERGY, REALTY,
# INFRA, MEDIA and PSUBANK returned ONE row each. A rulebook missing from this map simply
# gets no sector benchmark - a normal outcome, not an error. Expected to grow.
SECTOR_BENCHMARKS: dict[str, str] = {
    "financials": "^NSEBANK",
    "technology": "^CNXIT",
    "healthcare_pharma": "^CNXPHARMA",
}


def benchmark_for(rulebook: str | None) -> tuple[str, str | None]:
    """(broad, sector) benchmark symbols. Every prediction gets the broad one."""
    return settings.broad_benchmark, SECTOR_BENCHMARKS.get((rulebook or "").lower())


# ------------------------------------------------------------------ pure arithmetic
def _naive(s: pd.Series) -> pd.Series:
    """Strip timezone and time-of-day so two feeds can be joined on the calendar date."""
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return pd.Series(s.to_numpy(), index=idx.normalize())


def align(stock: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    """Inner-join two close series on the calendar date.

    Unmatched days are DROPPED, never interpolated: inventing a holiday price fabricates
    a trade that never happened, and comparing by position would line a stock's Tuesday
    up against the index's Thursday.
    """
    if stock is None or benchmark is None or stock.empty or benchmark.empty:
        return pd.DataFrame(columns=["stock", "benchmark"])
    joined = pd.concat({"stock": _naive(stock), "benchmark": _naive(benchmark)},
                       axis=1, join="inner").dropna()
    return joined[["stock", "benchmark"]] if not joined.empty else \
        pd.DataFrame(columns=["stock", "benchmark"])


def forward_return(closes: pd.Series, start, horizon_days: int):
    """Percent change over N TRADING days from the first session on or after `start`.

    Returns `(pct, outcome_date)`, or None when the horizon has not been reached yet -
    which is a different thing from zero, and the caller labels it `open`.
    """
    if closes is None or closes.empty:
        return None
    s = _naive(closes)
    start = pd.Timestamp(start).normalize()
    pos = int(s.index.searchsorted(start))                  # first session on or after
    end = pos + horizon_days
    if pos >= len(s) or end >= len(s):
        return None
    base = s.iloc[pos]
    if not base:
        return None
    return round((s.iloc[end] / base - 1) * 100, 4), s.index[end]


def abnormal_return(stock_pct: float | None, benchmark_pct: float | None) -> float | None:
    """The only number worth grading. A stock up 3% while the market ran 9% lost value."""
    if stock_pct is None or benchmark_pct is None:
        return None
    return round(stock_pct - benchmark_pct, 4)


def crossed_invalidation(closes: pd.Series, direction: str, level: float | None) -> bool:
    """Did any close in the window break the 'I am wrong if...' level?

    Checked across EVERY close in the window, not just the last one: a call that was
    stopped out mid-window and recovered by the horizon was still wrong on the day, and
    checking only the endpoint hides it.
    """
    if level is None or closes is None or closes.empty:
        return False
    if direction == "bullish":
        return bool((closes <= level).any())
    if direction == "bearish":
        return bool((closes >= level).any())
    return False


def label_for(direction: str, abnormal_pct: float | None, flat_band: float | None = None) -> str:
    """Turn an abnormal return into a verdict.

    The flat band exists because below roughly two points over twenty sessions you cannot
    separate skill from bid-ask friction, and counting those as hits flatters the record.
    """
    band = settings.flat_band_pct if flat_band is None else flat_band
    if abnormal_pct is None:
        return "unavailable"
    if abs(abnormal_pct) <= band:
        return "flat"
    if direction == "bullish":
        return "hit" if abnormal_pct > 0 else "miss"
    if direction == "bearish":
        return "hit" if abnormal_pct < 0 else "miss"
    return "miss"                       # neutral call, but the stock moved materially


# ------------------------------------------------------------------ the network edge
@lru_cache(maxsize=256)
def _fetch(symbol: str, period: str) -> pd.Series:
    """Adjusted closes. Cached per process - one benchmark serves every prediction."""
    import yfinance as yf
    try:
        h = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        if h is None or h.empty or "Close" not in h:
            return pd.Series(dtype=float)
        return h["Close"].dropna()
    except Exception as e:
        logger.warning(f"price fetch failed for {symbol}: {type(e).__name__}: {e}")
        return pd.Series(dtype=float)


def fetch_closes(symbol: str, period: str = "2y") -> pd.Series:
    """Adjusted closes for a stock or an index. Empty means delisted or unavailable."""
    return _fetch(symbol if symbol.startswith("^") else f"{symbol.upper()}.NS", period)


# ------------------------------------------------------------------ dating a call
def _sessions(closes: pd.Series) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(closes.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def actionable_date(report_date: str, closes: pd.Series | None, lag: int) -> str:
    """The first session you could actually have traded on this report.

    Trendlyne publishes after the broker's paying clients have already acted, so a report dated
    the 13th is priced from the next session, not the 13th. Walking the real price series rather
    than adding calendar days keeps weekends and NSE holidays out of the answer.

    When the series does not reach back to the report (listed later, or older than the feed),
    the report's own date is kept: jumping forward to the series' first session would re-date a
    2022 call to 2024. Grading then labels it `unavailable` rather than scoring the wrong window.

    When that session is not in the feed yet - a report from the feed's last session, or a weekend
    report after it - or there is no feed, the call is dated the calendar day after its report.
    Grading starts from the first session ON OR AFTER the as-of date, so that is exactly the first
    session you could have traded. Clamping to the last price instead dated a call on, or even
    before, the day it was published. (Past the feed, a lag above 1 counts as 1.)
    """
    start = pd.Timestamp(report_date).normalize()
    after = (start + pd.Timedelta(days=1)).date().isoformat() if lag > 0 else report_date
    if closes is None or closes.empty:
        return after
    idx = _sessions(closes)
    if idx[0] > start:
        return report_date
    pos = int(idx.searchsorted(start, side="right")) + lag - 1
    return idx[pos].date().isoformat() if pos < len(idx) else after


# ------------------------------------------------------------------ orchestration
def history_period(as_of: str, today: date | None = None) -> str:
    """Enough price history to reach back to a call: two years when that covers it, else all.
    A fixed two-year fetch silently started a 2022 call's window in 2024."""
    age = ((today or date.today()) - date.fromisoformat(str(as_of)[:10])).days
    return "2y" if age < 700 else "max"


def grade(pred: Prediction, horizon_days: int, period: str | None = None) -> Outcome:
    """Grade one prediction at one horizon. Never raises - a bad row is labelled, not lost."""
    broad_sym, sector_sym = benchmark_for(pred.rulebook)
    period = period or history_period(pred.as_of_date)
    stock = fetch_closes(pred.symbol, period)
    if stock.empty:                      # delisted, renamed, or halted - e.g. TATAMOTORS.NS
        return Outcome(prediction_id=pred.prediction_id, horizon_days=horizon_days,
                       label="unavailable", benchmark_symbol=broad_sym)

    joined = align(stock, fetch_closes(broad_sym, period))
    # A series that starts after the call (listed later, or history the feed does not have) would
    # be measured from its first session - a window that has nothing to do with the call.
    if joined.empty or joined.index[0] > pd.Timestamp(pred.as_of_date) + pd.Timedelta(days=7):
        return Outcome(prediction_id=pred.prediction_id, horizon_days=horizon_days,
                       label="unavailable", benchmark_symbol=broad_sym)

    s_fwd = forward_return(joined["stock"], pred.as_of_date, horizon_days)
    b_fwd = forward_return(joined["benchmark"], pred.as_of_date, horizon_days)
    if s_fwd is None or b_fwd is None:   # the horizon has not arrived yet
        return Outcome(prediction_id=pred.prediction_id, horizon_days=horizon_days,
                       label="open", benchmark_symbol=broad_sym)

    stock_pct, outcome_date = s_fwd
    bench_pct, _ = b_fwd
    abn = abnormal_return(stock_pct, bench_pct)

    # sector-relative, where an index that actually works exists for this rulebook
    sec_abn = None
    if sector_sym:
        sec_joined = align(stock, fetch_closes(sector_sym, period))
        sec_fwd = forward_return(sec_joined["benchmark"], pred.as_of_date, horizon_days) \
            if not sec_joined.empty else None
        if sec_fwd:
            sec_abn = abnormal_return(stock_pct, sec_fwd[0])

    window = joined["stock"].loc[
        pd.Timestamp(pred.as_of_date).normalize():pd.Timestamp(outcome_date)]
    label = ("invalidated"
             if crossed_invalidation(window, pred.direction, pred.invalidation_price)
             else label_for(pred.direction, abn))

    return Outcome(
        prediction_id=pred.prediction_id, horizon_days=horizon_days,
        outcome_date=pd.Timestamp(outcome_date).date().isoformat(),
        stock_return_pct=stock_pct, benchmark_symbol=broad_sym,
        benchmark_return_pct=bench_pct, abnormal_return_pct=abn,
        sector_benchmark_symbol=sector_sym if sec_abn is not None else None,
        sector_abnormal_return_pct=sec_abn, label=label)


def track(horizons: tuple[int, ...] | None = None, today: date | None = None,
          limit: int | None = None, dry_run: bool = False) -> list[Outcome]:
    """Grade every prediction whose horizon has arrived. Safe to run repeatedly."""
    horizons = horizons or tuple(settings.ledger_horizons)
    today = today or date.today()
    graded: list[Outcome] = []
    for h in horizons:
        due = pending(h, today)[:limit] if limit else pending(h, today)
        logger.info(f"horizon +{h}d: {len(due)} prediction(s) to grade")
        for p in due:
            try:
                graded.append(grade(p, h))
            except Exception as e:       # one bad row must never stop the nightly job
                logger.warning(f"grading {p.prediction_id} at +{h}d failed: "
                               f"{type(e).__name__}: {e}")
    if graded and not dry_run:
        save_outcomes(graded)
    return graded


def scorecard(source: str | None = None, horizon_days: int = 20) -> dict:
    """Hit rate at one horizon. Prints the sample size because it is the whole story."""
    from collections import Counter
    preds = {p.prediction_id: p for p in load_predictions(source=source)}
    rows = [o for o in _all_outcomes(horizon_days) if o.prediction_id in preds]
    tally = Counter(o.label for o in rows)
    decided = tally["hit"] + tally["miss"]
    scored = [o.abnormal_return_pct for o in rows if o.abnormal_return_pct is not None]
    return {
        "source": source or "all", "horizon_days": horizon_days,
        "predictions": len(preds), "graded": len(rows), "labels": dict(tally),
        "hit_rate": round(tally["hit"] / decided, 3) if decided else None,
        "mean_abnormal_pct": round(sum(scored) / len(scored), 2) if scored else None,
        "decided_sample": decided,
        "note": ("Too few decided outcomes to read anything into this."
                 if decided < 30 else "Sample is thin; treat as indicative."),
    }


def _all_outcomes(horizon_days: int) -> list[Outcome]:
    from pennystock.data.ledger import load_outcomes
    return load_outcomes(horizon_days=horizon_days)
