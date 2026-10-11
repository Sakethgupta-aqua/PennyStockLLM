"""Grading a prediction: what did the price actually do?

Every case here was measured on live data on 2026-09-27, and each one exists because
the obvious implementation is wrong:

  +20 trading days is +34 CALENDAR days, so date arithmetic silently grades the wrong day.
  RELIANCE had 129 trading days over six months where Nifty 500 had 125, so the two
  series must be joined, not zipped.
  RELIANCE fell 1.71% in a window where the market rose 5.99% - "down a bit" by raw
  return, badly beaten by the only number that matters.
"""
import pandas as pd
import pytest

from pennystock.data.outcomes import (SECTOR_BENCHMARKS, abnormal_return, align,
                                      benchmark_for, crossed_invalidation, forward_return,
                                      label_for)


def series(values, start="2026-01-01", skip=()):
    """A daily close series on business days, optionally missing some (holidays)."""
    days = pd.bdate_range(start=start, periods=len(values) + len(skip))
    days = [d for i, d in enumerate(days) if i not in skip][:len(values)]
    return pd.Series(values, index=pd.DatetimeIndex(days))


# --- aligning the stock with the market -----------------------------------------------
def test_alignment_keeps_only_days_both_series_traded():
    """RELIANCE had 129 trading days to Nifty 500's 125. Zipping them by position would
    silently compare a Tuesday's stock price to a Thursday's index."""
    stock = series([100, 101, 102, 103, 104])
    bench = series([200, 202, 204], skip=(1, 3))          # index missing two days
    joined = align(stock, bench)
    assert len(joined) == 3
    assert list(joined.columns) == ["stock", "benchmark"]


def test_alignment_never_invents_a_missing_price():
    """Interpolating a holiday would fabricate a trade that never happened."""
    stock = series([100, 101, 102])
    bench = series([200, 204], skip=(1,))
    assert not align(stock, bench).isna().any().any()


def test_alignment_of_disjoint_series_is_empty_not_an_error():
    a = series([1, 2, 3], start="2026-01-01")
    b = series([1, 2, 3], start="2027-01-01")
    assert align(a, b).empty


# --- counting trading days, not calendar days -----------------------------------------
def test_forward_return_counts_TRADING_days():
    """Measured: +20 trading days = +34 calendar days. Adding 20 days to the date would
    land four sessions early and grade the wrong bar."""
    s = series([100] * 10 + [110] + [120] * 10)           # jumps on the 11th session
    pct, on = forward_return(s, s.index[0], 10)
    assert pct == pytest.approx(10.0)                     # the 11th bar, not day+10
    assert on == s.index[10]


def test_forward_return_is_none_while_the_horizon_is_still_ahead():
    """Not yet knowable is not the same as zero. The caller labels this 'open'."""
    s = series([100, 101, 102])
    assert forward_return(s, s.index[0], 20) is None


def test_forward_return_from_a_date_the_stock_did_not_trade_uses_the_next_session():
    s = series([100, 105, 110])
    pct, _ = forward_return(s, pd.Timestamp("2025-12-31"), 1)   # before the series starts
    assert pct == pytest.approx(5.0)


def test_forward_return_on_an_empty_series_is_none():
    assert forward_return(pd.Series(dtype=float), pd.Timestamp("2026-01-01"), 5) is None


# --- the number that actually matters --------------------------------------------------
def test_abnormal_return_is_the_stock_minus_the_market():
    """The live RELIANCE window: -1.71% looks like a small dip until you see the market
    was +5.99%. The call underperformed by 7.70 points.

    The live run printed -7.71 because it rounded the result of UNROUNDED inputs.
    Rounding first and then subtracting does not compose, which is exactly why the
    stored return values are rounded once, at the edge, and never re-rounded.
    """
    assert abnormal_return(-1.71, 5.99) == pytest.approx(-7.70)


def test_a_rising_stock_can_still_be_a_miss():
    """Up 3% while the market ran 9% is a loss of relative value, and grading on raw
    return would have recorded it as a win."""
    assert abnormal_return(3.0, 9.0) < 0


# --- labelling ------------------------------------------------------------------------
FLAT = 2.0


@pytest.mark.parametrize("direction,abnormal,expected", [
    ("bullish", 7.7, "hit"),
    ("bullish", -7.7, "miss"),
    ("bearish", -7.7, "hit"),          # said it would fall; it fell
    ("bearish", 7.7, "miss"),
    ("bullish", 1.0, "flat"),          # inside the band
    ("bullish", -1.0, "flat"),
    ("bearish", 0.5, "flat"),
    ("neutral", 0.5, "flat"),
    ("neutral", 9.0, "miss"),          # said nothing would happen; something did
    ("neutral", -9.0, "miss"),
])
def test_every_label_branch(direction, abnormal, expected):
    assert label_for(direction, abnormal, flat_band=FLAT) == expected


def test_the_flat_band_exists_because_small_moves_are_noise():
    """Under ~2% over 20 days you cannot separate skill from bid-ask friction. Counting
    those as hits would flatter the record."""
    assert label_for("bullish", 1.99, flat_band=2.0) == "flat"
    assert label_for("bullish", 2.01, flat_band=2.0) == "hit"


def test_a_missing_abnormal_return_is_unavailable_not_a_guess():
    assert label_for("bullish", None, flat_band=FLAT) == "unavailable"


# --- invalidation ---------------------------------------------------------------------
def test_invalidation_fires_on_an_intermediate_close_not_just_the_last_one():
    """'I am wrong if it closes below 90' must trigger on the day it happened, even if
    the price recovered by the horizon. Checking only the endpoint hides the stop-out."""
    closes = series([100, 95, 88, 96, 102])               # dipped to 88 mid-window
    assert crossed_invalidation(closes, "bullish", 90.0) is True


def test_a_bullish_call_is_not_invalidated_by_a_price_that_never_fell_far_enough():
    assert crossed_invalidation(series([100, 95, 92, 99]), "bullish", 90.0) is False


def test_a_bearish_call_is_invalidated_by_a_price_going_UP_through_the_level():
    closes = series([100, 104, 112, 103])
    assert crossed_invalidation(closes, "bearish", 110.0) is True
    assert crossed_invalidation(closes, "bearish", 120.0) is False


def test_no_invalidation_price_means_never_invalidated():
    assert crossed_invalidation(series([100, 50, 10]), "bullish", None) is False


def test_an_empty_price_window_cannot_invalidate():
    assert crossed_invalidation(pd.Series(dtype=float), "bullish", 90.0) is False


# --- choosing the benchmark ------------------------------------------------------------
def test_every_prediction_gets_the_broad_market_benchmark():
    """Nifty 500 works for everything; it is the one index proven to return full history."""
    broad, sector = benchmark_for(rulebook=None)
    assert broad == "^CRSLDX"
    assert sector is None


def test_the_three_sectors_with_a_working_index_get_one_too():
    assert benchmark_for("financials")[1] == "^NSEBANK"
    assert benchmark_for("technology")[1] == "^CNXIT"
    assert benchmark_for("healthcare_pharma")[1] == "^CNXPHARMA"


def test_a_sector_without_a_working_index_simply_gets_none():
    """Eight of the twelve NSE sector indices return a single row on this feed. Absence
    of a sector benchmark is a normal outcome, not an error."""
    broad, sector = benchmark_for("materials_cyclical")
    assert broad == "^CRSLDX" and sector is None
    assert "materials_cyclical" not in SECTOR_BENCHMARKS


# --- calls older than the default two-year price window ---------------------------------
def _old_call(as_of="2022-09-26"):
    from pennystock.data.ledger import Prediction
    return Prediction(symbol="OLD", source="broker", as_of_date=as_of, direction="bullish",
                      prediction_id="p1", rulebook="industrials_infra")


def _fake_feed(monkeypatch, stock_start="2022-01-03", window_start="2024-10-01"):
    """Stock flat at 100 until 2022-09-26, then 110; index flat. A two-year request only reaches
    back to `window_start` - exactly what yfinance does with period='2y'."""
    from pennystock.data import outcomes as oc
    days = pd.bdate_range(stock_start, "2026-09-30")
    stock = pd.Series([100.0 if d <= pd.Timestamp("2022-09-26") else 110.0 for d in days], index=days)
    index = pd.Series(1000.0, index=pd.bdate_range("2015-01-01", "2026-09-30"))

    def fetch(symbol, period="2y"):
        s = index if symbol.startswith("^") else stock
        return s if period == "max" else s[s.index >= pd.Timestamp(window_start)]
    monkeypatch.setattr(oc, "fetch_closes", fetch)


def test_a_call_older_than_two_years_is_graded_on_its_own_window(monkeypatch):
    """With a fixed two-year fetch, a 2022 call was measured from the first session the feed
    returned - 2024 - so it was graded on a window that had nothing to do with it."""
    from pennystock.data.outcomes import grade
    _fake_feed(monkeypatch)
    out = grade(_old_call(), 1)
    assert (out.outcome_date, out.stock_return_pct, out.label) == ("2022-09-27", 10.0, "hit")


def test_a_call_dated_before_the_stock_listed_is_unavailable(monkeypatch):
    from pennystock.data.outcomes import grade
    _fake_feed(monkeypatch, stock_start="2024-01-01")
    assert grade(_old_call(), 1).label == "unavailable"


# --- dating a call: the first session you could have traded it ----------------------------------
from pennystock.data.outcomes import actionable_date  # noqa: E402


def closes_through(*days):
    return pd.Series(range(100, 100 + len(days)), index=pd.DatetimeIndex(days), dtype=float)


def test_a_call_is_dated_to_the_next_session_after_its_report():
    c = closes_through("2026-09-30", "2026-10-01", "2026-10-05")       # Oct 2 holiday, then a weekend
    assert actionable_date("2026-10-01", c, lag=1) == "2026-10-05"


def test_a_report_from_the_feeds_last_session_is_never_dated_on_that_session():
    # the next session has not happened yet; clamping to the last price would date the call to the
    # very day it was published - the look-ahead the lag exists to prevent
    c = closes_through("2026-09-30", "2026-10-01")
    assert actionable_date("2026-10-01", c, lag=1) == "2026-10-02"


def test_a_weekend_report_past_the_feed_is_never_dated_before_it_was_written():
    c = closes_through("2026-09-30", "2026-10-01")                     # feed ends Thursday
    assert actionable_date("2026-10-04", c, lag=1) == "2026-10-05"     # never Oct 1


def test_without_prices_a_call_still_waits_for_the_day_after_its_report():
    assert actionable_date("2026-10-01", None, lag=1) == "2026-10-02"
    assert actionable_date("2026-10-01", pd.Series(dtype=float), lag=1) == "2026-10-02"


def test_the_day_after_resolves_to_the_next_real_session_when_graded():
    # 2026-10-02 is a holiday: grading starts from the first session ON OR AFTER the as-of date
    c = closes_through("2026-10-01", "2026-10-05", "2026-10-06")
    assert forward_return(c, actionable_date("2026-10-01", c.iloc[:1], lag=1), 1)[1] == pd.Timestamp("2026-10-06")
