"""Pure technical-indicator functions — computed in code, never in the LLM's head.

The reference doc calls LLM numeric/time-series reasoning the 'retrieval-to-reasoning
bottleneck'. So we compute indicators here and hand the model finished values. Every
function takes a pandas OHLCV frame (from the provider) and is unit-tested.
"""
from __future__ import annotations

import pandas as pd


def sma(close: pd.Series, window: int) -> float | None:
    if len(close) < window:
        return None
    return round(float(close.rolling(window).mean().iloc[-1]), 2)


def rsi(close: pd.Series, window: int = 14) -> float | None:
    if len(close) < window + 1:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss
    return round(float((100 - 100 / (1 + rs)).iloc[-1]), 2)


def support_resistance(df: pd.DataFrame, lookback: int = 60) -> tuple[float, float]:
    window = df.tail(lookback)
    return round(float(window["Low"].min()), 2), round(float(window["High"].max()), 2)


def trend(close: pd.Series) -> str:
    s50, s200 = sma(close, 50), sma(close, 200)
    if s50 is None or s200 is None:
        return "insufficient history"
    return "uptrend" if s50 > s200 else "downtrend"


def summarize(df: pd.DataFrame, sr_lookback: int = 60, rsi_window: int = 14, recent_days: int = 252) -> dict:
    """Turn a raw OHLCV frame into a compact, LLM-ready technical summary.

    Drawdown (peak/low) is measured over the FULL series so the true multi-year peak shows,
    while the indicators (MAs/RSI/support-resistance) use only the recent window."""
    if df is None or df.empty:
        return {"error": "no price history available"}
    df = df[df["Close"].notna()]                 # drop blank rows (e.g. a market-holiday 'today')
    if df.empty:
        return {"error": "no price history available"}
    full = df["Close"]                           # whole series (for the real peak/low)
    recent = df.tail(recent_days)                # last ~1y (for the technical indicators)
    close = recent["Close"]
    last = round(float(close.iloc[-1]), 2)
    peak = round(float(full.max()), 2)           # the REAL multi-year peak
    support, resistance = support_resistance(recent, sr_lookback)
    return {
        "last_close": last,
        "high_1y": round(float(close.max()), 2),
        "peak_5y": peak,
        "pct_from_peak": round((last / peak - 1) * 100, 1),   # the TRUE drawdown
        "low_5y": round(float(full.min()), 2),
        "sma_50": sma(close, 50),
        "sma_200": sma(close, 200),
        "rsi_14": rsi(close, rsi_window),
        "trend": trend(close),
        "support": support,
        "resistance": resistance,
        "avg_volume_20d": int(recent["Volume"].tail(20).mean()) if "Volume" in recent else None,
    }


def detect_structural_break(df: pd.DataFrame, threshold_pct: float = 20.0) -> dict | None:
    """Flag a large single-day price jump - a strong hint of a split/bonus/demerger that
    yfinance's .splits may not record. Long-window indicators spanning that date are unreliable."""
    if df is None or df.empty or len(df) < 2:
        return None
    close = df["Close"].dropna()                  # ignore blank rows so gaps are real trading days
    if len(close) < 2:
        return None
    change = close.pct_change() * 100
    biggest = change.abs().max()
    if pd.isna(biggest) or biggest < threshold_pct:
        return None
    day = change.abs().idxmax()
    return {
        "date": str(day.date()),
        "single_day_move_pct": round(float(change[day]), 1),
        "note": ("Large single-day price gap detected - possible corporate action "
                 "(split/bonus/demerger). '% from high' and long-window indicators that span "
                 "this date may be distorted; interpret with caution."),
    }
