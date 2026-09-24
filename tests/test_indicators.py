import pandas as pd

from pennystock.data.indicators import rsi, sma, summarize, support_resistance, trend


def _df(prices):
    return pd.DataFrame({
        "Open": prices, "High": [p * 1.01 for p in prices],
        "Low": [p * 0.99 for p in prices], "Close": prices,
        "Volume": [1000] * len(prices),
    })


def test_sma_and_insufficient():
    df = _df(list(range(1, 21)))                 # 1..20
    assert sma(df["Close"], 5) == 18.0           # mean(16..20)
    assert sma(df["Close"], 50) is None          # not enough data -> None, not a guess


def test_rsi_all_up_is_high():
    assert rsi(_df(list(range(1, 40)))["Close"]) > 90


def test_trend_up():
    assert trend(_df(list(range(1, 260)))["Close"]) == "uptrend"


def test_support_resistance():
    lo, hi = support_resistance(_df([10, 12, 8, 15, 9, 11]), lookback=6)
    assert lo == 7.92 and hi == 15.15


def test_summarize_keys_and_empty():
    s = summarize(_df(list(range(1, 260))))
    assert {"last_close", "sma_50", "sma_200", "rsi_14", "trend", "support"} <= s.keys()
    assert "error" in summarize(pd.DataFrame())   # empty -> graceful


def test_summarize_drops_trailing_nan():
    df = _df(list(range(1, 260)))
    df.loc[df.index[-1], "Close"] = float("nan")   # simulate a market-holiday blank row
    s = summarize(df)
    assert s["last_close"] == 258.0                # last VALID close, never nan
