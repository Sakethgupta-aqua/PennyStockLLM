"""The screening funnel.

The behaviours pinned here are the ones that make a screen trustworthy rather than merely tidy:
a veto is a veto (cheapness never buys a name back in), missing data never disqualifies but never
flatters either, and the funnel can always explain what it removed.
"""
import pytest

from pennystock.data.metrics import StockMetrics
from pennystock.data.screener import (PILLARS, VETO_CASH_CONVERSION, _percentiles,
                                      is_financial, screen, score_universe,
                                      screening_caveats, veto_reasons)


def stock(symbol, **kw) -> StockMetrics:
    """A tradeable, unremarkable company. Tests override only what they are about."""
    base = dict(market_cap_cr=5000.0, turnover_cr=10.0, roce_pct=15.0, roe_pct=14.0,
                profit_margin_pct=8.0, revenue_cagr_pct=10.0, debt_to_equity_pct=40.0,
                debt_to_ebitda=1.5, interest_coverage=8.0, altman_zem=4.0, piotroski=6,
                beneish_m=-2.5, cash_conversion=1.0, pledge_pct_of_promoter=0.0,
                promoter_holding_pct=55.0, pe=20.0, pb=3.0, pct_from_peak_5y=-10.0, above_200dma=1)
    return StockMetrics(symbol=symbol, **(base | kw))


# --- the veto gates ------------------------------------------------------------------------
@pytest.mark.parametrize("field,value,expected", [
    ("pledge_pct_of_promoter", 58.45, "pledge"),        # the live SUZLON number
    ("interest_coverage", 1.2, "interest cover"),
    ("altman_zem", 0.5, "Altman"),
    ("beneish_m", -1.44, "Beneish"),                    # also live SUZLON
    ("cash_conversion", 0.33, "cash conversion"),
    ("debt_to_ebitda", 6.0, "debt/EBITDA"),

])
def test_each_gate_disqualifies_on_its_own(field, value, expected):
    rules = [rule for rule, _ in veto_reasons(stock("X", **{field: value}))]
    assert any(expected in r for r in rules), rules


def test_a_clean_company_trips_no_gate():
    assert veto_reasons(stock("CLEAN")) == []


def test_missing_data_never_disqualifies():
    """Absence of evidence is not evidence of a problem - a gap must be scored as unknown, not
    treated as a failure, or every thinly-covered small-cap silently disappears."""
    blank = StockMetrics(symbol="SPARSE")
    assert veto_reasons(blank) == []


def test_cheapness_cannot_rescue_a_vetoed_stock():
    """The whole reason vetoes run BEFORE scoring: no multiple is cheap enough to fix accounts you
    cannot trust. A stock at a P/E of 2 with a manipulation flag must not appear in the results."""
    universe = [stock(f"OK{i}", pe=40.0) for i in range(6)] + [stock("FRAUD", pe=2.0, beneish_m=-1.0)]
    f = screen(universe)
    assert "FRAUD" not in [r.metrics.symbol for r in f.results]
    assert f.after_vetoes == 6


# --- stage 1: tradeability -----------------------------------------------------------------
def test_illiquid_names_are_removed_before_anything_else():
    """10 lakh shares of a Rs 3 stock is Rs 30 lakh a day - unbuyable at any size."""
    f = screen([stock("BIG", turnover_cr=50.0), stock("THIN", turnover_cr=0.2)],
               min_turnover_cr=1.0)
    assert f.after_liquidity == 1


def test_market_cap_floor_applies():
    f = screen([stock("BIG"), stock("TINY", market_cap_cr=50.0)], min_market_cap_cr=500.0)
    assert f.after_liquidity == 1


# --- stage 3: percentile scoring -----------------------------------------------------------
def test_percentiles_run_worst_to_best_and_invert_for_lower_is_better():
    rows = [stock("A", roce_pct=5.0), stock("B", roce_pct=15.0), stock("C", roce_pct=25.0)]
    assert _percentiles(rows, "roce_pct", True) == {"A": 0.0, "B": 0.5, "C": 1.0}
    assert _percentiles(rows, "roce_pct", False) == {"A": 1.0, "B": 0.5, "C": 0.0}


def test_percentiles_need_at_least_two_datapoints():
    assert _percentiles([stock("A")], "roce_pct", True) == {}


def test_better_fundamentals_outrank_worse_ones():
    rows = [stock("GOOD", roce_pct=35.0, roe_pct=30.0, profit_margin_pct=25.0, revenue_cagr_pct=20.0),
            stock("MID"),
            stock("POOR", roce_pct=3.0, roe_pct=2.0, profit_margin_pct=1.0, revenue_cagr_pct=0.5)]
    ranked = score_universe(rows)
    assert [r.metrics.symbol for r in ranked] == ["GOOD", "MID", "POOR"]


def test_a_missing_pillar_scores_neutral_not_zero():
    """A company with no valuation data must not be pushed to the bottom for it - that would rank
    on data coverage instead of on the business."""
    rows = [stock("A"), stock("B"), stock("NOPRICE", pe=None, pb=None)]
    res = {r.metrics.symbol: r for r in score_universe(rows)}
    assert res["NOPRICE"].pillars["valuation"] == 0.5
    assert res["NOPRICE"].completeness < 1.0


def test_weights_are_a_full_hundred_percent_and_favour_quality_and_safety():
    """Quality + safety must outweigh valuation + momentum: the long-standing Indian ordering is
    'what is it, can it survive, then what does it cost'."""
    assert sum(w for w, _ in PILLARS.values()) == pytest.approx(1.0)
    assert PILLARS["quality"][0] + PILLARS["safety"][0] > PILLARS["valuation"][0] + PILLARS["momentum"][0]


def test_thinly_covered_names_are_dropped_from_the_output():
    rows = [stock(f"OK{i}") for i in range(4)] + [StockMetrics(
        symbol="SPARSE", market_cap_cr=5000.0, turnover_cr=10.0, pe=5.0)]
    f = screen(rows, min_completeness=0.5)
    assert "SPARSE" not in [r.metrics.symbol for r in f.results]


# --- the audit trail -----------------------------------------------------------------------
def test_funnel_reports_what_each_stage_removed():
    rows = [stock(f"OK{i}") for i in range(3)] + [
        stock("THIN", turnover_cr=0.1),
        stock("PLEDGED", pledge_pct_of_promoter=70.0),
        stock("NOCASH", cash_conversion=VETO_CASH_CONVERSION - 0.1)]
    f = screen(rows)
    assert (f.universe, f.after_liquidity, f.after_vetoes) == (6, 5, 3)
    assert sum(f.veto_reasons.values()) == 2
    text = f.describe()
    assert "universe" in text and "veto gates" in text


# --- lenders play by a different rulebook ---------------------------------------------------
def bank(symbol, **kw) -> StockMetrics:
    return stock(symbol, sector="Financial Services", industry="Banks - Regional", **kw)


@pytest.mark.parametrize("field,value", [
    ("cash_conversion", -3.99),        # BAJFINANCE, live
    ("interest_coverage", 0.4),
    ("altman_zem", 0.2),
    ("debt_to_ebitda", 9.0),
    ("beneish_m", -1.0),
])
def test_lenders_are_not_vetoed_by_uncalibrated_gates(field, value):
    """Growing a loan book IS an operating cash outflow, deposits are not Debt/EBITDA-style debt,
    and neither Altman nor Beneish was estimated on financial filers. Live proof this matters:
    the first Nifty-50 run threw out BAJFINANCE, BAJAJFINSV, SHRIRAMFIN and SBIN on artefacts."""
    assert veto_reasons(bank("LENDER", **{field: value})) == []


def test_promoter_pledge_still_applies_to_lenders():
    """Pledge is the one gate that means the same thing for every company, so it is NOT skipped."""
    rules = [r for r, _ in veto_reasons(bank("LENDER", pledge_pct_of_promoter=70.0))]
    assert rules == ["promoter pledge"]


def test_a_non_financial_is_still_vetoed_by_the_same_numbers():
    """The exemption must be about the KIND of company, not a way for anyone to dodge the gates."""
    assert veto_reasons(stock("FACTORY", cash_conversion=-3.99)) != []


@pytest.mark.parametrize("sector,industry", [
    ("Financial Services", "Credit Services"),
    ("Consumer Cyclical", "Insurance - Life"),        # sector mislabelled, industry gives it away
    ("", "Banks - Diversified"),
    ("Financial Services", ""),
])
def test_financials_are_recognised_by_sector_or_industry(sector, industry):
    assert is_financial(stock("X", sector=sector, industry=industry))


def test_an_industrial_is_not_mistaken_for_a_lender():
    assert not is_financial(stock("X", sector="Industrials", industry="Aerospace & Defense"))


def test_lenders_are_not_scored_on_metrics_that_do_not_apply():
    """SBIN ranked in the bottom third on governance because of a cash-conversion figure that means
    nothing for a bank. Skipping a check at the gate but still ranking on it is the same bug."""
    rows = [stock(f"IND{i}", cash_conversion=1.0 + i / 10) for i in range(4)]
    rows.append(bank("BANK", cash_conversion=0.07))
    res = {r.metrics.symbol: r for r in score_universe(rows)}
    bank_gov = res["BANK"].pillars["governance"]
    assert bank_gov > 0.0, "a bank must not be punished for a metric that does not apply to it"


def test_inapplicable_metrics_do_not_count_against_data_completeness():
    """Completeness measures how much is KNOWN, not how much of the rulebook happens to fit."""
    rows = [bank("BANK", debt_to_ebitda=None, interest_coverage=None, altman_zem=None,
                 cash_conversion=None, beneish_m=None, debt_to_equity_pct=None),
            stock("IND1"), stock("IND2")]
    res = {r.metrics.symbol: r for r in score_universe(rows)}
    assert res["BANK"].completeness == 1.0


def test_lenders_always_say_what_was_not_checked():
    """Silence about a skipped check reads as a pass - the most dangerous thing a screen can do."""
    notes = " ".join(screening_caveats(bank("LENDER")))
    assert "Skipped as uncalibrated for this sector" in notes
    assert "capital adequacy" in notes            # says what to judge a lender on instead
    assert "Promoter pledge WAS still checked" in notes


def test_negative_cash_flow_at_a_non_financial_is_flagged_as_ambiguous():
    """M&M and GRASIM consolidate lending arms and trip the gate. Nothing in a free feed separates
    that from real trouble, so the gate fires - but the ambiguity has to be stated."""
    notes = " ".join(screening_caveats(stock("CONGLOM", cash_conversion=-2.76)))
    assert "finance or lending subsidiary" in notes
