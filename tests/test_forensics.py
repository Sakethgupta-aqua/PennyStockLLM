from pennystock.data.forensics import (altman_z_emerging, altman_z_score, beneish_m_score,
                                       cash_conversion, debt_to_ebitda, effective_tax_rate,
                                       interest_coverage, piotroski_f_score,
                                       summarize_forensics, working_capital_days)
from pennystock.data.providers import Financials


def improving() -> Financials:
    """A company getting better on every Piotroski axis."""
    return Financials(
        fiscal_years=[2025, 2026], revenue_cr=[1000, 1300], net_income_cr=[80, 130],
        operating_income_cr=[100, 160], total_assets_cr=[1000, 1100],
        current_assets_cr=[400, 600], current_liabilities_cr=[300, 300],
        total_liabilities_cr=[500, 500], retained_earnings_cr=[300, 400],
        long_term_debt_cr=[200, 100], gross_profit_cr=[300, 455],
        cost_of_revenue_cr=[700, 845], operating_cash_flow_cr=[120, 200],
        receivables_cr=[100, 120], inventory_cr=[150, 170], payables_cr=[80, 90],
        net_ppe_cr=[400, 420], depreciation_cr=[40, 42], sga_cr=[100, 120],
        shares_outstanding=[1000, 1000], equity_cr=[500, 600])


def test_piotroski_rewards_an_improving_company():
    r = piotroski_f_score(improving())
    assert r["out_of"] == 9 and r["f_score"] >= 8 and r["verdict"] == "STRONG"


def test_piotroski_reports_what_it_could_not_evaluate():
    thin = Financials(fiscal_years=[2026], net_income_cr=[10], total_assets_cr=[100])
    r = piotroski_f_score(thin)
    assert r["f_score"] is None or r["out_of"] < 9      # never pretends a partial score is complete


def test_altman_zones():
    safe = altman_z_score(improving(), market_cap_cr=5000)
    assert safe["z_score"] > 2.99 and safe["zone"] == "SAFE"
    weak = Financials(fiscal_years=[2026], total_assets_cr=[1000], current_assets_cr=[100],
                      current_liabilities_cr=[600], retained_earnings_cr=[-200],
                      operating_income_cr=[5], total_liabilities_cr=[900], revenue_cr=[200])
    assert altman_z_score(weak, market_cap_cr=50)["zone"] == "DISTRESS"


def test_altman_reports_missing_inputs_instead_of_guessing():
    r = altman_z_score(Financials(fiscal_years=[2026], total_assets_cr=[100]), market_cap_cr=None)
    assert r["z_score"] is None and r["missing"]


def test_beneish_computes_all_eight_components():
    r = beneish_m_score(improving())
    assert r["m_score"] is not None and set(r["components"]) == {
        "DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "TATA", "LVGI"}


def test_cash_conversion_flags_paper_profits():
    paper = Financials(fiscal_years=[2025, 2026], net_income_cr=[100, 120],
                       operating_cash_flow_cr=[20, 25])
    r = cash_conversion(paper)
    assert r["cfo_to_pat"] < 0.7 and "POOR" in r["verdict"]
    good = cash_conversion(improving())
    assert good["cfo_to_pat"] > 1 and good["verdict"] == "healthy"


def test_working_capital_days_reports_change():
    wc = working_capital_days(improving())
    assert wc["receivable_days"] is not None and wc["receivable_days_change"] is not None
    assert wc["cash_conversion_cycle_days"] is not None


def test_summary_collects_red_flags_and_degrades_gracefully():
    assert summarize_forensics(Financials())["forensics"] == "unavailable"
    paper = Financials(fiscal_years=[2025, 2026], revenue_cr=[100, 100],
                       net_income_cr=[50, 40], total_assets_cr=[100, 100],
                       operating_cash_flow_cr=[5, 5], current_assets_cr=[50, 40],
                       current_liabilities_cr=[30, 40], long_term_debt_cr=[10, 20],
                       gross_profit_cr=[40, 30], shares_outstanding=[100, 120])
    flags = " ".join(summarize_forensics(paper)["red_flags"])
    assert "WEAK Piotroski" in flags and "cash conversion" in flags


def test_emerging_market_altman_is_kinder_to_capital_heavy_firms():
    """The Indian variant drops the sales/assets term and uses BOOK equity, so an asset-heavy
    company is not branded distressed just for being asset-heavy. Measured on real data: the US
    model called Vedanta 1.09 (distress) while Z-EM scores it 5.02 (safe)."""
    heavy = Financials(fiscal_years=[2026], total_assets_cr=[10000], current_assets_cr=[3000],
                       current_liabilities_cr=[2000], retained_earnings_cr=[3000],
                       operating_income_cr=[1200], total_liabilities_cr=[5000],
                       equity_cr=[5000], revenue_cr=[3000])
    em = altman_z_emerging(heavy)
    us = altman_z_score(heavy, market_cap_cr=2000)
    assert em["z_em_score"] > us["z_score"]      # the same firm looks healthier on the right model
    assert em["zone"] == "SAFE"


def test_emerging_market_altman_still_catches_real_distress():
    broken = Financials(fiscal_years=[2026], total_assets_cr=[1000], current_assets_cr=[100],
                        current_liabilities_cr=[900], retained_earnings_cr=[-800],
                        operating_income_cr=[-200], total_liabilities_cr=[1400], equity_cr=[-400])
    assert altman_z_emerging(broken)["zone"] == "DISTRESS"


def test_interest_coverage_flags_the_indian_red_line():
    thin = Financials(fiscal_years=[2026], operating_income_cr=[100], interest_expense_cr=[80],
                      ebitda_cr=[160])
    r = interest_coverage(thin)
    assert r["interest_coverage"] == 1.25 and "DANGER" in r["verdict"]
    assert r["interest_coverage_ebitda"] == 2.0      # EBITDA variant, as Indian lenders use
    strong = Financials(fiscal_years=[2026], operating_income_cr=[800], interest_expense_cr=[80])
    assert interest_coverage(strong)["verdict"] == "comfortable"


def test_interest_coverage_handles_debt_free_companies():
    r = interest_coverage(Financials(fiscal_years=[2026], operating_income_cr=[100]))
    assert r["interest_coverage"] is None and "debt-free" in r["note"]

def _fin(**kw) -> Financials:
    """Minimal Financials carrying only the lines a single ratio needs. Series are OLDEST-FIRST,
    matching the provider, so the latest year is the LAST element."""
    n = max((len(v) for v in kw.values() if isinstance(v, list)), default=1)
    kw.setdefault("fiscal_years", list(range(2026 - n + 1, 2027)))
    return Financials(**kw)


# --- Debt/EBITDA: how an Indian lender actually sizes leverage -----------------------------
def test_debt_to_ebitda_flags_the_stressed_zone():
    out = debt_to_ebitda(_fin(total_debt_cr=[5000], ebitda_cr=[1000]))       # 5.0x
    assert out["debt_to_ebitda"] == 5.0
    assert out["verdict"].startswith("DANGER")


def test_debt_to_ebitda_comfortable_and_elevated_bands():
    assert debt_to_ebitda(_fin(total_debt_cr=[1000], ebitda_cr=[1000]))["verdict"] == "comfortable"
    assert debt_to_ebitda(_fin(total_debt_cr=[3500], ebitda_cr=[1000]))["verdict"] == "elevated"


def test_debt_to_ebitda_reads_the_latest_year_not_the_oldest():
    """Series are oldest-first; leverage must be judged on where the company is NOW."""
    out = debt_to_ebitda(_fin(total_debt_cr=[500, 5000], ebitda_cr=[1000, 1000]))
    assert out["debt_to_ebitda"] == 5.0


def test_debt_to_ebitda_never_reads_negative_ebitda_as_low_leverage():
    """A loss-making company has no meaningful multiple - the danger is a NEGATIVE ratio that
    sorts as 'cheap leverage' in any screen that consumes it."""
    out = debt_to_ebitda(_fin(total_debt_cr=[5000], ebitda_cr=[-200]))
    assert out["debt_to_ebitda"] is None
    assert "NEGATIVE" in out["verdict"]


def test_debt_to_ebitda_missing_data_returns_no_ratio():
    assert debt_to_ebitda(_fin(total_debt_cr=[1000]))["debt_to_ebitda"] is None


# --- effective tax rate: is the reported profit the kind the taxman recognises? -------------
def test_effective_tax_rate_normal_band_is_not_flagged():
    out = effective_tax_rate(_fin(tax_provision_cr=[248, 250], pretax_income_cr=[1000, 1000]))
    assert out["effective_tax_rate_pct"] == 25.0
    assert out["flag"] is False


def test_effective_tax_rate_flags_an_implausibly_low_rate():
    """Healthy profit paying almost no tax is the classic sign the profit may not be real."""
    out = effective_tax_rate(_fin(tax_provision_cr=[10], pretax_income_cr=[1000]))
    assert out["effective_tax_rate_pct"] == 1.0
    assert out["flag"] is True
    assert "below" in out["observations"][0]


def test_effective_tax_rate_flags_a_wild_swing_between_years():
    """Live check on VEDL showed a 42-point swing across four years - a real earnings-quality
    question that a single-year rate would have hidden completely."""
    out = effective_tax_rate(_fin(tax_provision_cr=[300, 50, 400, 250],
                                  pretax_income_cr=[1000, 1000, 1000, 1000]))
    assert out["swing_pct_points"] == 35.0
    assert out["flag"] is True


def test_effective_tax_rate_skips_loss_years_instead_of_going_negative():
    out = effective_tax_rate(_fin(tax_provision_cr=[5, 250], pretax_income_cr=[-800, 1000]))
    assert out["effective_tax_rate_pct"] == 25.0        # the loss year is skipped, not negated
    assert out["swing_pct_points"] is None              # only one usable year remains


def test_effective_tax_rate_with_no_profitable_year():
    out = effective_tax_rate(_fin(tax_provision_cr=[5], pretax_income_cr=[-800]))
    assert out["effective_tax_rate_pct"] is None
