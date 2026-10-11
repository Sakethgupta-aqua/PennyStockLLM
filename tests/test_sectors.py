"""Sector rulebooks and sector-relative ranking.

The numbers quoted in these tests are medians measured on a live 500-name Indian universe. They
are the evidence that a single rulebook cannot work: ROCE spans 3.1x between Technology (21.1%)
and Utilities (6.8%), Debt/Equity spans 25x, and the median FINANCIAL sits at 5.55x Debt/EBITDA -
above a 4.5x veto written for manufacturers.
"""
import pytest

from pennystock.data.metrics import StockMetrics
from pennystock.data.sectors import DEFAULT_PROFILE, PROFILES, profile_for
from pennystock.data.screener import (MIN_SECTOR_PEERS, applies_to, pillars_for,
                                      score_universe, screening_caveats, sector_percentiles,
                                      veto_reasons)


def stock(symbol, sector="Industrials", industry="Auto Parts", **kw) -> StockMetrics:
    base = dict(market_cap_cr=5000.0, turnover_cr=10.0, roce_pct=15.0, roe_pct=14.0,
                profit_margin_pct=8.0, revenue_cagr_pct=10.0, debt_to_equity_pct=40.0,
                debt_to_ebitda=1.5, interest_coverage=8.0, altman_zem=4.0, piotroski=6,
                beneish_m=-2.5, cash_conversion=1.0, pledge_pct_of_promoter=0.0,
                promoter_holding_pct=55.0, pe=20.0, pb=3.0, pct_from_peak_5y=-10.0, above_200dma=1)
    return StockMetrics(symbol=symbol, sector=sector, industry=industry, **(base | kw))


# --- routing to the right rulebook -----------------------------------------------------------
@pytest.mark.parametrize("sector,industry,expected", [
    ("Financial Services", "Banks - Regional", "financials"),
    ("Financial Services", "Credit Services", "financials"),
    ("Utilities", "Utilities - Renewable", "utilities_power"),
    ("Real Estate", "Real Estate - Development", "real_estate"),
    ("Basic Materials", "Cement", "materials_cyclical"),
    ("Technology", "Information Technology Services", "technology"),
    ("Healthcare", "Drug Manufacturers - Specialty", "healthcare_pharma"),
    ("Consumer Defensive", "Packaged Foods", "consumer_staples"),
    ("Energy", "Oil & Gas Refining & Marketing", "energy"),
    ("Communication Services", "Telecom Services", "communication"),
])
def test_each_sector_reaches_its_rulebook(sector, industry, expected):
    assert profile_for(sector, industry).name == expected


def test_industry_beats_a_mislabelled_sector():
    """A life insurer filed under Consumer Cyclical is still a lender. The industry label is the
    more specific signal, so it is matched first."""
    assert profile_for("Consumer Cyclical", "Insurance - Life").name == "financials"


def test_an_unknown_business_gets_the_general_rulebook():
    assert profile_for("Widgets", "Widget Making") is DEFAULT_PROFILE


def test_every_profile_declares_its_blind_spots():
    """A screen that cannot say what it failed to measure invites the reader to assume it measured
    everything - which is how a screen gets someone hurt."""
    for p in (*PROFILES, DEFAULT_PROFILE):
        assert p.blind_spots, p.name


# --- gates move with the sector ---------------------------------------------------------------
def test_leverage_normal_for_a_utility_is_fatal_for_an_fmcg():
    """5.5x Debt/EBITDA against contracted, regulated cash flows is ordinary. The same number at a
    packaged-foods company is not."""
    util = stock("U", sector="Utilities", industry="Utilities - Regulated", debt_to_ebitda=5.5)
    fmcg = stock("F", sector="Consumer Defensive", industry="Packaged Foods", debt_to_ebitda=5.5)
    assert veto_reasons(util) == []
    assert [r for r, _ in veto_reasons(fmcg)] == ["debt/EBITDA"]


def test_a_utility_is_still_vetoed_when_leverage_is_genuinely_extreme():
    """The threshold moves; it is not removed."""
    util = stock("U", sector="Utilities", industry="Utilities - Regulated", debt_to_ebitda=9.0)
    assert [r for r, _ in veto_reasons(util)] == ["debt/EBITDA"]


def test_lenders_skip_the_gates_that_were_never_calibrated_for_them():
    bank = stock("B", sector="Financial Services", industry="Banks - Regional",
                 debt_to_ebitda=8.0, cash_conversion=-3.99, interest_coverage=0.4, beneish_m=-1.0)
    assert veto_reasons(bank) == []


def test_promoter_pledge_cannot_be_switched_off_by_any_rulebook():
    """Pledge means the same thing in every sector, so no profile is allowed to skip it."""
    for prof_sector, prof_industry in [("Financial Services", "Banks - Regional"),
                                       ("Utilities", "Utilities - Regulated"),
                                       ("Real Estate", "Real Estate - Development")]:
        m = stock("X", sector=prof_sector, industry=prof_industry, pledge_pct_of_promoter=70.0)
        assert "promoter pledge" in [r for r, _ in veto_reasons(m)]


# --- the ranking correction --------------------------------------------------------------------
def test_ranking_happens_inside_the_sector_not_across_the_market():
    """The core fix. A utility at 12% ROCE is excellent FOR A UTILITY; ranked against software at
    21% it looks mediocre. Both sector leaders must score 1.0 in their own sector."""
    utils = [stock(f"U{i}", sector="Utilities", industry="Utilities - Regulated",
                   roce_pct=5.0 + i) for i in range(MIN_SECTOR_PEERS)]
    tech = [stock(f"T{i}", sector="Technology", industry="Information Technology Services",
                  roce_pct=18.0 + i) for i in range(MIN_SECTOR_PEERS)]
    ranks = sector_percentiles(utils + tech, "roce_pct", True)
    best_util = f"U{MIN_SECTOR_PEERS - 1}"
    assert ranks[best_util] == 1.0            # top of its own sector despite a low absolute ROCE
    assert ranks["U0"] == 0.0
    assert ranks[f"T{MIN_SECTOR_PEERS - 1}"] == 1.0


def test_a_sector_too_small_to_rank_falls_back_to_the_whole_universe():
    """A percentile drawn from three peers is noise wearing a decimal point."""
    many = [stock(f"I{i}", roce_pct=10.0 + i) for i in range(MIN_SECTOR_PEERS + 4)]
    few = [stock(f"R{i}", sector="Real Estate", industry="Real Estate - Development",
                 roce_pct=50.0 + i) for i in range(3)]
    ranks = sector_percentiles(many + few, "roce_pct", True)
    # ranked against everyone, the tiny sector's high ROCE lands at the top rather than at 0/0.5/1
    assert ranks["R2"] == 1.0
    assert ranks["R0"] > 0.5


def test_metrics_that_do_not_apply_are_kept_out_of_the_ranking_pool():
    """A bank's Debt/Equity of 244% must not distort the distribution every industrial is
    measured against."""
    banks = [stock(f"B{i}", sector="Financial Services", industry="Banks - Regional",
                   debt_to_equity_pct=200.0 + i) for i in range(MIN_SECTOR_PEERS)]
    inds = [stock(f"I{i}", debt_to_equity_pct=10.0 + i) for i in range(MIN_SECTOR_PEERS)]
    ranks = sector_percentiles(banks + inds, "debt_to_equity_pct", False)
    assert not any(k.startswith("B") for k in ranks)
    assert len([k for k in ranks if k.startswith("I")]) == MIN_SECTOR_PEERS


def test_applies_to_respects_the_rulebook():
    bank = stock("B", sector="Financial Services", industry="Banks - Regional")
    assert not applies_to(bank, "debt_to_ebitda")
    assert applies_to(bank, "roe_pct")
    assert applies_to(stock("I"), "debt_to_ebitda")


# --- weights and the valuation yardstick --------------------------------------------------------
def test_a_lender_is_ranked_on_book_value_first():
    """Banks are bought and sold on book. P/E leads for an asset-light compounder; P/B for a bank."""
    bank = profile_for("Financial Services", "Banks - Regional")
    assert bank.valuation_metrics[0][0] == "pb"
    assert profile_for("Technology", "Information Technology Services").valuation_metrics[0][0] == "pe"


def test_pillar_weights_always_sum_to_one():
    """A rulebook may reweight the pillars, but the score must stay out of 100."""
    for p in (*PROFILES, DEFAULT_PROFILE):
        assert sum(w for w, _ in pillars_for(p).values()) == pytest.approx(1.0)


def test_a_rulebook_can_reweight_the_pillars():
    fin = pillars_for(profile_for("Financial Services", "Banks - Regional"))
    gen = pillars_for(DEFAULT_PROFILE)
    assert fin["governance"][0] > gen["governance"][0]     # asset quality is a governance question
    assert fin["safety"][0] < gen["safety"][0]             # leverage ratios do not apply


# --- caveats -------------------------------------------------------------------------------------
def test_the_caveat_names_the_metrics_the_sector_is_really_judged_on():
    text = " ".join(screening_caveats(stock("C", sector="Basic Materials", industry="Cement")))
    assert "EBITDA per tonne" in text and "capacity utilisation" in text


def test_a_pharma_caveat_warns_about_regulatory_risk_the_screen_cannot_see():
    text = " ".join(screening_caveats(
        stock("P", sector="Healthcare", industry="Drug Manufacturers - Specialty")))
    assert "USFDA" in text


def test_scoring_end_to_end_stays_within_bounds():
    rows = ([stock(f"U{i}", sector="Utilities", industry="Utilities - Regulated") for i in range(9)]
            + [stock(f"B{i}", sector="Financial Services", industry="Banks - Regional") for i in range(9)]
            + [stock(f"I{i}") for i in range(9)])
    for r in score_universe(rows):
        assert 0.0 <= r.score <= 100.0
        assert set(r.pillars) == {"quality", "safety", "governance", "valuation", "momentum"}


# --- Layer 1: blind spots that broker reports can now fill -----------------------------------
def test_every_mapped_blind_spot_is_a_real_one_with_real_kpis():
    """The map is keyed by blind-spot TEXT. Editing a rulebook's wording must break this test, not
    silently disconnect the broker data from the sector it was meant to fill."""
    from pennystock.data.broker_reports import KPI_METRICS
    from pennystock.data.sectors import BLIND_SPOT_KPIS
    declared = {b for p in PROFILES for b in p.blind_spots}
    assert set(BLIND_SPOT_KPIS) <= declared
    assert {k for ks in BLIND_SPOT_KPIS.values() for k in ks} <= KPI_METRICS


def test_a_blind_spot_counts_as_covered_only_when_one_of_its_kpis_has_data():
    from pennystock.data.sectors import blind_spot_coverage
    fin = profile_for("Financial Services", "Banks - Regional")
    covered, missing = blind_spot_coverage(fin, {"nim_pct", "gnpa_pct", "eps"})
    assert covered == ["net interest margin (NIM)", "GNPA/NNPA and slippage"]
    assert "CASA ratio" in missing and "net interest margin (NIM)" not in missing


def test_the_caveat_stops_claiming_ignorance_the_screen_no_longer_has():
    """'Not measured here: NIM' is false once NIM is on screen - and a screen that misstates what
    it measured is as misleading as one that hides what it skipped."""
    bank = stock("B", sector="Financial Services", industry="Banks - Regional")
    text = " ".join(screening_caveats(bank, covered={"nim_pct"}))
    assert "net interest margin (NIM)" in text.split("Read from broker reports")[1]
    assert "net interest margin (NIM)" not in text.split("Read from broker reports")[0]


def test_the_none_of_this_is_measured_note_disappears_once_something_is():
    bank = stock("B", sector="Financial Services", industry="Banks - Regional")
    assert "None of those inputs is in this screen" in " ".join(screening_caveats(bank))
    assert "None of those inputs" not in " ".join(screening_caveats(bank, covered={"gnpa_pct"}))
