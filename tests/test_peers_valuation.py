import pandas as pd

from pennystock.data.peers import _percentile
from pennystock.data.providers import Financials
from pennystock.data.valuation import valuation_bands


def test_percentile_orientation_is_always_higher_is_better():
    pop = [10, 20, 30, 40]
    # for P/E lower is better -> a low value must score HIGH
    assert _percentile(12, pop, lower_is_better=True) > 70
    assert _percentile(38, pop, lower_is_better=True) < 30
    # for ROE higher is better -> a high value scores high
    assert _percentile(38, pop, lower_is_better=False) > 70


def test_percentile_needs_a_population():
    assert _percentile(10, [5], lower_is_better=True) is None


def _prices(start, end, level):
    idx = pd.date_range(start, end, freq="D")
    return pd.DataFrame({"Close": [level] * len(idx)}, index=idx)


def test_valuation_band_places_current_pe_in_context():
    fin = Financials(fiscal_years=[2025, 2026], eps=[10.0, 10.0])
    hist = pd.concat([_prices("2024-04-01", "2025-03-31", 100),     # FY2025 -> P/E 10
                      _prices("2025-04-01", "2026-03-31", 150)])    # FY2026 -> P/E 15
    r = valuation_bands(fin, hist, current_pe=20.0)
    assert r["pe_band"]["low"] == 10.0 and r["pe_band"]["high"] == 15.0
    assert "ABOVE its historical range" in r["verdict"]


def test_valuation_skips_loss_years():
    fin = Financials(fiscal_years=[2025, 2026], eps=[-5.0, 10.0])
    hist = _prices("2024-04-01", "2026-03-31", 100)
    assert valuation_bands(fin, hist, 10.0)["valuation_history"] == "insufficient"


def test_valuation_handles_missing_inputs():
    assert valuation_bands(Financials(), None)["valuation_history"] == "unavailable"
