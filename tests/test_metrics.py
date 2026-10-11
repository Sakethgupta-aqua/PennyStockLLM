"""The screening metrics cache."""
import sqlite3
from datetime import datetime, timedelta

import pytest

from pennystock.data import metrics as mt
from pennystock.data.metrics import (StockMetrics, cached_symbols, is_stale, load_metrics,
                                     save_metrics)


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Every test gets its own market.db so the cache tests never touch the real one."""
    monkeypatch.setattr(mt.settings, "market_db", tmp_path / "test.db")
    yield


def test_round_trip_preserves_every_field():
    m = StockMetrics(symbol="ABC", company="ABC Ltd", sector="Materials", market_cap_cr=5000.0,
                     turnover_cr=12.5, roce_pct=18.4, altman_zem=3.1, piotroski=7,
                     beneish_m=-2.6, pledge_pct_of_promoter=0.0, pe=21.3, above_200dma=1,
                     red_flag_count=1, red_flags="one flag")
    assert save_metrics([m]) == 1
    got = load_metrics()[0]
    assert (got.symbol, got.roce_pct, got.piotroski, got.red_flags) == ("ABC", 18.4, 7, "one flag")
    assert got.fetched_at is not None                 # stamped on write, not by the caller


def test_saving_the_same_symbol_updates_rather_than_duplicates():
    save_metrics([StockMetrics(symbol="ABC", roce_pct=10.0)])
    save_metrics([StockMetrics(symbol="ABC", roce_pct=25.0)])
    rows = load_metrics()
    assert len(rows) == 1 and rows[0].roce_pct == 25.0


def test_saving_nothing_is_a_no_op():
    assert save_metrics([]) == 0


def test_cached_symbols_lists_what_is_stored():
    save_metrics([StockMetrics(symbol="A"), StockMetrics(symbol="B")])
    assert cached_symbols() == {"A", "B"}


def test_max_age_drops_rows_too_stale_to_act_on():
    """A screen built on year-old fundamentals is worse than no screen, because it looks current."""
    save_metrics([StockMetrics(symbol="FRESH")])
    old = (datetime.now() - timedelta(days=400)).isoformat(timespec="seconds")
    with sqlite3.connect(str(mt.settings.market_db)) as con:
        con.execute("INSERT INTO stock_metrics (symbol, fetched_at) VALUES ('OLD', ?)", (old,))
    assert len(load_metrics()) == 2
    assert [r.symbol for r in load_metrics(max_age_days=30)] == ["FRESH"]


def test_an_empty_cache_reads_as_stale():
    assert is_stale() is True


def test_a_freshly_written_cache_is_not_stale():
    save_metrics([StockMetrics(symbol="ABC")])
    assert is_stale() is False


# --- ROCE and growth, computed rather than taken on trust ----------------------------------
class _Fin:
    """Stand-in for Financials carrying only the lines _roce_and_growth reads."""
    def __init__(self, **kw):
        self.operating_income_cr = kw.get("ebit", [])
        self.total_assets_cr = kw.get("assets", [])
        self.current_liabilities_cr = kw.get("cl", [])
        self.revenue_cr = kw.get("rev", [])


def test_roce_uses_capital_employed_not_total_assets():
    """ROCE is EBIT / (total assets - current liabilities). Dividing by total assets instead
    understates every capital-light business and is the commonest way this ratio is got wrong."""
    roce, _ = mt._roce_and_growth(_Fin(ebit=[200], assets=[2000], cl=[1000]))
    assert roce == 20.0                               # 200 / (2000-1000), not 200/2000


def test_revenue_cagr_is_annualised_across_the_window():
    _, cagr = mt._roce_and_growth(_Fin(rev=[100, 110, 121]))     # two years of 10%
    assert cagr == pytest.approx(10.0, abs=0.1)


def test_roce_and_growth_survive_missing_lines():
    assert mt._roce_and_growth(_Fin()) == (None, None)


def test_roce_is_none_when_capital_employed_is_zero():
    """Working-capital-heavy filers can report current liabilities equal to total assets; that must
    return None rather than raise or produce an infinite ratio."""
    roce, _ = mt._roce_and_growth(_Fin(ebit=[200], assets=[1000], cl=[1000]))
    assert roce is None


# --- P/E sanity: a screen that ranks on price must sanity-check the price ------------------
class _FinNI:
    def __init__(self, net_income):
        self.net_income_cr = net_income


def test_pe_is_kept_when_it_agrees_with_the_filed_accounts():
    """Trailing-twelve-months vs last fiscal year differ legitimately - that must NOT be 'fixed'."""
    assert mt._sanity_check_pe("OK", 20.0, 2200.0, _FinNI([100.0])) == 20.0


def test_a_pe_the_accounts_contradict_is_dropped_not_guessed():
    """Live catch: yfinance reported ZFCVINDIA at P/E 2.8 while its accounts imply ~56, and a P/E
    of 2.8 is a top-decile valuation score - the bad number walked it into a top-25 shortlist.

    But we must DROP it, not substitute: INFY is the mirror image (its accounts arrive in USD, so
    the computed figure is nonsense and the reported 14.8 is right). Since we cannot tell which
    side broke, the stock is scored as unknown on valuation instead."""
    assert mt._sanity_check_pe("ZFCVINDIA", 2.8, 29022.0, _FinNI([517.15])) is None
    assert mt._sanity_check_pe("INFY", 14.8, 457684.0, _FinNI([331.3])) is None


def test_pe_check_ignores_loss_making_and_missing_cases():
    assert mt._sanity_check_pe("X", 20.0, 1000.0, _FinNI([-50.0])) == 20.0     # loss year
    assert mt._sanity_check_pe("X", None, 1000.0, _FinNI([100.0])) is None     # nothing reported
    assert mt._sanity_check_pe("X", 20.0, None, _FinNI([100.0])) == 20.0       # no market cap
    assert mt._sanity_check_pe("X", 20.0, 1000.0, _FinNI([])) == 20.0          # no profit data


# --- the 'Infinity' bug: a feed's non-number must never reach the ranking --------------------
def test_a_non_finite_feed_value_is_stored_as_unknown():
    """yfinance returns the STRING 'Infinity' for the P/E of a near-zero-earnings company. Four such
    rows crashed `screen.py --no-vetoes`: sorting mixed text and numbers raises TypeError."""
    save_metrics([StockMetrics(symbol="AVROIND", pe="Infinity", pb=float("inf"), roce_pct=12.0)])
    got = load_metrics()[0]
    assert (got.pe, got.pb, got.roce_pct) == (None, None, 12.0)


def test_text_already_stored_in_a_numeric_column_reads_back_as_unknown():
    """The four bad rows are already in market.db; reading must clean them without a re-fetch."""
    save_metrics([StockMetrics(symbol="ESSENTIA", pe=10.0)])
    with sqlite3.connect(mt.settings.market_db) as con:
        con.execute("UPDATE stock_metrics SET pe = 'Infinity', piotroski = 'n/a'")
    got = load_metrics()[0]
    assert (got.pe, got.piotroski) == (None, None)
