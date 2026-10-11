from pennystock.data.fundamentals import _growth, _trend, snapshot_warnings, summarize_financials
from pennystock.data.providers import Financials, Fundamentals


def test_growth_and_trend():
    assert _growth([2024, 2025, 2026], [100, 110, 121]) == {"pct": 10.0, "from": "FY24", "to": "FY26", "years": 2}
    assert _growth([2026], [100]) is None                     # too few points
    assert _growth([2025, 2026], [0, 50]) is None             # non-positive base guarded
    assert _growth([2025, 2026], [50, -5]) is None            # a loss at the end has no growth rate
    assert _trend([10, 20, 30]) == "rising"
    assert _trend([30, 10]) == "falling"


def test_summarize_and_unavailable():
    fin = Financials(fiscal_years=[2024, 2025], revenue_cr=[100, 121],
                     net_income_cr=[10, -5], total_debt_cr=[50, 40], equity_cr=[100, 100],
                     eps=[2.0, -1.0], operating_income_cr=[], free_cash_flow_cr=[])
    s = summarize_financials(fin)
    assert s["revenue_cagr"] == {"pct": 21.0, "from": "FY24", "to": "FY25", "years": 1}
    assert s["profitable_years"] == "1/2"
    assert s["debt_to_equity_pct_latest"] == 40.0
    assert summarize_financials(Financials()) == {"financials": "unavailable"}


def test_each_value_carries_its_year_and_growth_its_real_span():
    # DIXON 2026-10-11: Yahoo leaves FY22 empty, and parallel lists made the analyst call FY23's revenue FY22's
    fin = Financials(fiscal_years=[2022, 2023, 2024, 2025, 2026],
                     revenue_cr=[None, 12173.35, 17613.91, 38593.29, 48537.39],
                     net_income_cr=[None, 255.52, 367.75, 1095.54, 1438.64])
    s = summarize_financials(fin)
    assert s["revenue_cr"] == {"FY23": 12173.35, "FY24": 17613.91, "FY25": 38593.29, "FY26": 48537.39}
    assert s["revenue_cagr"] == {"pct": 58.6, "from": "FY23", "to": "FY26", "years": 3}
    assert s["net_income_cr"]["FY26"] == 1438.64 and "FY22" not in s["net_income_cr"]
    assert s["latest_year"] == "FY26"


def test_roe_roce_computed():
    fin = Financials(fiscal_years=[2025, 2026], revenue_cr=[100, 110],
                     net_income_cr=[18, 20], equity_cr=[90, 100],
                     operating_income_cr=[24, 25], total_debt_cr=[0, 25])
    s = summarize_financials(fin)
    assert s["roe_pct_latest"] == 20.0        # 20 / 100
    assert s["roce_pct_latest"] == 20.0       # 25 / (100 + 25)


def test_snapshot_contradiction():
    bad = Fundamentals(profit_margin=0.24, eps_trailing=-4.44, return_on_equity=-0.01)
    assert snapshot_warnings(bad)                 # flags the TMPV-style contradiction
    good = Fundamentals(profit_margin=0.08, eps_trailing=5.0, return_on_equity=0.15)
    assert snapshot_warnings(good) == []


def test_trend_detects_peak_and_decline():
    # Cipla FY23-FY26 net income: endpoints say "rising", but it peaked and fell 26%
    assert _trend([2801.91, 4121.55, 5272.52, 3879.23]) == "rising overall but latest is 26% BELOW the peak"
    # Cipla total debt: fell overall, but bottomed and is climbing again
    assert "ABOVE the trough" in _trend([803.12, 559.41, 438.19, 613.77])
    # a clean monotonic series keeps the simple label
    assert _trend([22473.18, 25446.63, 27145.4, 27711.69]) == "rising"
