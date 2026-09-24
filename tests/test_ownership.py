from pennystock.data.ownership import Ownership, Quarter, pledge_band, summarize_ownership
from pennystock.data.ownership_providers import quarter_ord


def own(*quarters):
    return Ownership(symbol="TEST", quarters=list(quarters))


def test_pledge_bands():
    assert pledge_band(None) == "unknown"          # unknown is NOT zero
    assert pledge_band(0) == "none (ideal)"
    assert pledge_band(8) == "low"
    assert pledge_band(25) == "caution"
    assert pledge_band(45) == "high"
    assert pledge_band(80) == "severe"


def test_unknown_pledge_is_flagged_loudly():
    s = summarize_ownership(own(Quarter("Mar 2026", promoter_pct=55.0),
                                Quarter("Jun 2026", promoter_pct=55.0)))
    assert s["pledge_pct_of_promoter_holding"] is None and s["pledge_band"] == "unknown"
    assert any("PLEDGE UNKNOWN" in f for f in s["risk_flags"])


def test_severe_pledge_and_rising_trend():
    s = summarize_ownership(own(Quarter("Mar 2026", promoter_pct=50.0, pledge_pct=40.0),
                                Quarter("Jun 2026", promoter_pct=50.0, pledge_pct=62.0)))
    flags = " ".join(s["risk_flags"])
    assert "SEVERE" in flags and "RISING" in flags
    assert s["pledge_band"] == "severe"


def test_promoter_selling_down_is_flagged():
    s = summarize_ownership(own(Quarter("Sep 2025", promoter_pct=40.0),
                                Quarter("Dec 2025", promoter_pct=38.0),
                                Quarter("Mar 2026", promoter_pct=35.0),
                                Quarter("Jun 2026", promoter_pct=33.0)))
    assert any("Promoter stake FALLING" in f for f in s["risk_flags"])
    assert s["promoter_change_3q_pp"] == -7.0


def test_low_promoter_holding_and_institutions_exiting():
    s = summarize_ownership(own(Quarter("Mar 2026", promoter_pct=20.0, fii_pct=15.0, dii_pct=10.0),
                                Quarter("Jun 2026", promoter_pct=20.0, fii_pct=10.0, dii_pct=8.0)))
    flags = " ".join(s["risk_flags"])
    assert "Low promoter holding" in flags and "Institutions REDUCING" in flags
    assert s["institutional_pct"] == 18.0


def test_clean_company_has_no_flags():
    s = summarize_ownership(own(Quarter("Mar 2026", promoter_pct=60.0, fii_pct=15.0, dii_pct=10.0, pledge_pct=0.0),
                                Quarter("Jun 2026", promoter_pct=60.0, fii_pct=15.5, dii_pct=10.5, pledge_pct=0.0)))
    assert s["risk_flags"] == ["No ownership red flags detected in the available data."]


def test_empty_is_graceful():
    assert summarize_ownership(Ownership(symbol="X"))["ownership"] == "unavailable"


def test_quarter_ordering_is_chronological():
    qs = ["Sep 2023", "Mar 2024", "Dec 2023", "Jun 2026"]
    assert sorted(qs, key=quarter_ord) == ["Sep 2023", "Dec 2023", "Mar 2024", "Jun 2026"]


def test_both_pledge_denominators_are_reported():
    """NSE's headline % of TOTAL shares is far smaller than % of PROMOTER holding.
    Suzlon: 6.84% of total == 58.45% of the promoter stake ("low" vs "SEVERE")."""
    s = summarize_ownership(own(Quarter("Jun 2026", promoter_pct=11.71,
                                        pledge_pct=58.45, pledge_pct_of_total=6.84)))
    assert s["pledge_pct_of_promoter_holding"] == 58.45
    assert s["pledge_pct_of_total_shares"] == 6.84
    assert s["pledge_band"] == "severe"          # banded on the promoter-holding figure
