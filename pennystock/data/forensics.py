"""Forensic accounting scores - "can I trust these numbers, and can this company survive?"

A veteran checks this BEFORE valuation: if the accounts are cooked or the balance sheet is
breaking, no P/E is cheap enough. These are the three screens the profession actually uses, all
computed here in pure code from statement lines we already fetch - no new data source:

  Piotroski F-Score (0-9)  is the business fundamentally IMPROVING?   8-9 strong, <=3 weak
  Altman Z-Score           could it go BANKRUPT within ~2 years?      >2.99 safe, <1.81 distress
  Beneish M-Score          are earnings being MANIPULATED?            > -1.78 = flag for review

plus two checks that need no model at all and catch most accounting games:
  Cash conversion (cumulative CFO / cumulative PAT) - profit that never becomes cash is paper
  Working-capital days - receivables or inventory growing faster than sales is a classic tell

Every function degrades to None when an input is missing; a score is always reported alongside
how many of its tests could actually be evaluated, so a partial score is never mistaken for a
clean bill of health.

CAVEATS worth stating in any report: the Beneish M-Score was calibrated on US filers and is
directional-only for India; Altman's original coefficients are for public manufacturers, so it
is unreliable for banks, NBFCs and other financials. Both are "investigate further" signals,
never proof of anything.
"""
from __future__ import annotations

from pennystock.data.providers import Financials

F_SCORE_STRONG, F_SCORE_WEAK = 8, 3
Z_SAFE, Z_DISTRESS = 2.99, 1.81            # original 1968 model (US public manufacturers)
ZEM_SAFE, ZEM_DISTRESS = 2.60, 1.10        # Altman Z'' / Emerging Market Score - used for India
ICR_WEAK, ICR_DANGER = 3.0, 1.5            # interest coverage; <1.5x is the Indian red line
DEBT_EBITDA_WATCH, DEBT_EBITDA_DANGER = 3.0, 4.5   # how Indian lenders size leverage
TAX_RATE_LOW, TAX_RATE_HIGH = 15.0, 40.0   # India's statutory band is ~22-30% post-2019
TAX_RATE_SWING = 15.0                      # year-on-year swing worth questioning
M_SCORE_FLAG = -1.78
CASH_CONVERSION_MIN = 0.7          # cumulative CFO should be >=70% of cumulative reported profit


def _at(series: list, idx: int):
    """Value at index, or None - tolerates short/ragged series."""
    try:
        v = series[idx]
    except (IndexError, TypeError):
        return None
    return v


def _div(a, b):
    if a is None or b in (None, 0):
        return None
    return a / b


def _gt(a, b) -> bool | None:
    return None if a is None or b is None else a > b


# ------------------------------------------------------------------ Piotroski
def piotroski_f_score(fin: Financials) -> dict:
    """9 binary tests of fundamental improvement (Piotroski 2000). Needs 2 years of data."""
    ta, ta0 = _at(fin.total_assets_cr, -1), _at(fin.total_assets_cr, -2)
    ni, ni0 = _at(fin.net_income_cr, -1), _at(fin.net_income_cr, -2)
    cfo = _at(fin.operating_cash_flow_cr, -1)
    roa, roa0 = _div(ni, ta), _div(ni0, ta0)
    cr = _div(_at(fin.current_assets_cr, -1), _at(fin.current_liabilities_cr, -1))
    cr0 = _div(_at(fin.current_assets_cr, -2), _at(fin.current_liabilities_cr, -2))
    ltd, ltd0 = _div(_at(fin.long_term_debt_cr, -1), ta), _div(_at(fin.long_term_debt_cr, -2), ta0)
    sh, sh0 = _at(fin.shares_outstanding, -1), _at(fin.shares_outstanding, -2)
    gm = _div(_at(fin.gross_profit_cr, -1), _at(fin.revenue_cr, -1))
    gm0 = _div(_at(fin.gross_profit_cr, -2), _at(fin.revenue_cr, -2))
    at_, at0 = _div(_at(fin.revenue_cr, -1), ta), _div(_at(fin.revenue_cr, -2), ta0)

    tests = {
        "positive_return_on_assets": _gt(roa, 0),
        "positive_operating_cash_flow": _gt(cfo, 0),
        "return_on_assets_improving": _gt(roa, roa0),
        "cash_flow_exceeds_profit": _gt(cfo, ni),               # accrual quality
        "long_term_debt_ratio_falling": _gt(ltd0, ltd),
        "current_ratio_improving": _gt(cr, cr0),
        "no_new_shares_issued": None if (sh is None or sh0 is None) else sh <= sh0 * 1.001,
        "gross_margin_improving": _gt(gm, gm0),
        "asset_turnover_improving": _gt(at_, at0),
    }
    scored = {k: v for k, v in tests.items() if v is not None}
    score = sum(1 for v in scored.values() if v)
    if not scored:
        return {"f_score": None, "note": "insufficient statement history"}
    verdict = ("STRONG" if score >= F_SCORE_STRONG else
               "WEAK" if score <= F_SCORE_WEAK else "MIDDLING")
    return {
        "f_score": score,
        "out_of": len(scored),
        "verdict": verdict,
        "failed_tests": sorted(k for k, v in scored.items() if not v),
    }


# ------------------------------------------------------------------ Altman
def altman_z_score(fin: Financials, market_cap_cr: float | None) -> dict:
    """Bankruptcy-distress score for public manufacturers (Altman 1968)."""
    ta = _at(fin.total_assets_cr, -1)
    ca, cl = _at(fin.current_assets_cr, -1), _at(fin.current_liabilities_cr, -1)
    re, ebit = _at(fin.retained_earnings_cr, -1), _at(fin.operating_income_cr, -1)
    tl, sales = _at(fin.total_liabilities_cr, -1), _at(fin.revenue_cr, -1)
    wc = None if ca is None or cl is None else ca - cl

    parts = {"working_capital/assets": _div(wc, ta), "retained_earnings/assets": _div(re, ta),
             "ebit/assets": _div(ebit, ta), "market_cap/liabilities": _div(market_cap_cr, tl),
             "sales/assets": _div(sales, ta)}
    weights = [1.2, 1.4, 3.3, 0.6, 1.0]
    values = list(parts.values())
    if any(v is None for v in values):
        return {"z_score": None, "note": "missing balance-sheet inputs",
                "missing": [k for k, v in parts.items() if v is None]}
    z = round(sum(w * v for w, v in zip(weights, values)), 2)
    zone = "SAFE" if z > Z_SAFE else ("DISTRESS" if z < Z_DISTRESS else "GREY")
    return {"z_score": z, "zone": zone,
            "note": "Altman's original model is for public MANUFACTURERS - unreliable for "
                    "banks/NBFCs/financials."}


def altman_z_emerging(fin: Financials) -> dict:
    """Altman Z'' / Emerging Market Score - the RIGHT variant for Indian companies.

    Altman published this specifically because the 1968 model travels badly. Two changes matter:
      * X4 uses BOOK equity / total liabilities, not MARKET cap - so it does not punish a company
        merely for trading on the lower multiples normal outside 1968 America.
      * The sales/assets term is DROPPED, which removes the bias against capital-intensive
        businesses. That bias is severe in India, where metals, power, cement and infrastructure
        all carry heavy asset bases by their nature.
    A +3.25 constant re-centres the scale. Thresholds: >2.60 safe, 1.10-2.60 grey, <1.10 distress.

    Measured on real Indian stocks, the original model called Vedanta (1.09), IRFC (0.34) and
    Hindalco (2.13) distressed-or-grey; the emerging-market model scores them 5.02, 3.64 and 5.76 -
    all safe - while still correctly flagging Vodafone Idea at -1.50. The original was producing
    false alarms, not insight.
    """
    ta = _at(fin.total_assets_cr, -1)
    ca, cl = _at(fin.current_assets_cr, -1), _at(fin.current_liabilities_cr, -1)
    re, ebit = _at(fin.retained_earnings_cr, -1), _at(fin.operating_income_cr, -1)
    tl, eq = _at(fin.total_liabilities_cr, -1), _at(fin.equity_cr, -1)
    wc = None if ca is None or cl is None else ca - cl
    parts = {"working_capital/assets": _div(wc, ta), "retained_earnings/assets": _div(re, ta),
             "ebit/assets": _div(ebit, ta), "book_equity/liabilities": _div(eq, tl)}
    if any(v is None for v in parts.values()):
        return {"z_em_score": None, "note": "missing balance-sheet inputs",
                "missing": [k for k, v in parts.items() if v is None]}
    z = round(3.25 + 6.56 * parts["working_capital/assets"] + 3.26 * parts["retained_earnings/assets"]
              + 6.72 * parts["ebit/assets"] + 1.05 * parts["book_equity/liabilities"], 2)
    zone = "SAFE" if z > ZEM_SAFE else ("DISTRESS" if z < ZEM_DISTRESS else "GREY")
    return {"z_em_score": z, "zone": zone,
            "model": "Altman Z''/EM (emerging markets) - the appropriate variant for India",
            "note": "Still unreliable for banks and NBFCs, whose balance sheets are structurally "
                    "different from operating companies."}


def interest_coverage(fin: Financials) -> dict:
    """EBIT / interest expense - the ratio Indian lenders and analysts lean on hardest.

    In a high-rate economy, servicing debt is what actually kills leveraged companies, long before
    a bankruptcy model notices. Below 1.5x the business is not covering its own interest bill from
    operating profit. Indian banks prefer the EBITDA variant for capital-intensive sectors (steel,
    cement, power, infrastructure) because depreciation there is enormous and non-cash.
    """
    ebit, ebitda = _at(fin.operating_income_cr, -1), _at(fin.ebitda_cr, -1)
    interest = _at(fin.interest_expense_cr, -1)
    if interest in (None, 0):
        return {"interest_coverage": None,
                "note": "no interest expense reported - either debt-free or not disclosed"}
    icr = _div(ebit, abs(interest))
    icr_ebitda = _div(ebitda, abs(interest))
    if icr is None:
        return {"interest_coverage": None, "note": "operating profit unavailable"}
    icr = round(icr, 2)
    verdict = ("DANGER - operating profit barely covers the interest bill" if icr < ICR_DANGER else
               "weak" if icr < ICR_WEAK else "comfortable")
    return {"interest_coverage": icr,
            "interest_coverage_ebitda": round(icr_ebitda, 2) if icr_ebitda else None,
            "interest_expense_cr": abs(interest),
            "verdict": verdict,
            "note": f"<{ICR_DANGER}x is the Indian red line. The EBITDA variant is the one lenders "
                    "use for capital-intensive sectors."}


def debt_to_ebitda(fin: Financials) -> dict:
    """Total debt / EBITDA - how an Indian lender actually sizes leverage.

    Preferred over debt-to-equity here because equity can be inflated by revaluation or eroded by
    losses, while EBITDA is the cash-ish number the loan is repaid out of. Above ~3x a bank starts
    asking questions; above ~4.5x refinancing becomes the company's main business risk.
    """
    debt, ebitda = _at(fin.total_debt_cr, -1), _at(fin.ebitda_cr, -1)
    if debt is None or ebitda in (None, 0):
        return {"debt_to_ebitda": None, "note": "debt or EBITDA unavailable"}
    if ebitda < 0:
        return {"debt_to_ebitda": None, "verdict": "EBITDA is NEGATIVE - leverage cannot be "
                                                   "serviced from operations at all"}
    ratio = round(debt / ebitda, 2)
    verdict = ("DANGER - refinancing risk dominates" if ratio > DEBT_EBITDA_DANGER else
               "elevated" if ratio > DEBT_EBITDA_WATCH else "comfortable")
    return {"debt_to_ebitda": ratio, "total_debt_cr": debt, "ebitda_cr": ebitda, "verdict": verdict,
            "note": f">{DEBT_EBITDA_WATCH}x draws lender attention, >{DEBT_EBITDA_DANGER}x is stressed."}


def effective_tax_rate(fin: Financials) -> dict:
    """Tax paid / pre-tax profit, per year - a classic Indian check on whether profits are real.

    India's statutory corporate rate is roughly 22-30% since the 2019 cuts. A company reporting
    healthy profits while paying almost no tax is either sitting on genuine exemptions (SEZ, tax
    holidays, carried-forward losses - all legitimate and disclosable) or reporting profits the tax
    authority does not recognise. Either way it demands an explanation, and a rate that JUMPS
    around year to year is the bigger tell.
    """
    rates, years = [], []
    for i, fy in enumerate(fin.fiscal_years):
        tax, pbt = _at(fin.tax_provision_cr, i), _at(fin.pretax_income_cr, i)
        if tax is None or pbt in (None, 0) or pbt < 0:      # a loss year has no meaningful rate
            continue
        rates.append(round(tax / pbt * 100, 1))
        years.append(fy)
    if not rates:
        return {"effective_tax_rate_pct": None, "note": "no profitable year with tax data"}
    latest = rates[-1]
    swing = round(max(rates) - min(rates), 1) if len(rates) > 1 else None
    notes = []
    if latest < TAX_RATE_LOW:
        notes.append(f"latest effective rate {latest}% is well below India's statutory band - "
                     "check for exemptions, tax holidays or carried-forward losses")
    if latest > TAX_RATE_HIGH:
        notes.append(f"latest effective rate {latest}% is unusually high - look for one-off "
                     "provisions or disputed demands")
    if swing is not None and swing > TAX_RATE_SWING:
        notes.append(f"rate swung {swing} points across {len(rates)} years - inconsistent tax "
                     "treatment is a classic earnings-quality question")
    return {"effective_tax_rate_pct": latest, "by_year": dict(zip(years, rates)),
            "swing_pct_points": swing, "flag": bool(notes),   # boolean for the screener to score on
            "observations": notes or ["rate is within the normal band"]}


# ------------------------------------------------------------------ Beneish
def beneish_m_score(fin: Financials) -> dict:
    """8-variable earnings-manipulation probability (Beneish 1999). Needs 2 years."""
    def two(attr):
        s = getattr(fin, attr)
        return _at(s, -1), _at(s, -2)

    rec, rec0 = two("receivables_cr")
    sales, sales0 = two("revenue_cr")
    gp, gp0 = two("gross_profit_cr")
    ta, ta0 = two("total_assets_cr")
    ca, ca0 = two("current_assets_cr")
    ppe, ppe0 = two("net_ppe_cr")
    dep, dep0 = two("depreciation_cr")
    sga, sga0 = two("sga_cr")
    cl, cl0 = two("current_liabilities_cr")
    ltd, ltd0 = two("long_term_debt_cr")
    ni, cfo = _at(fin.net_income_cr, -1), _at(fin.operating_cash_flow_cr, -1)

    dsri = _div(_div(rec, sales), _div(rec0, sales0))
    gmi = _div(_div(gp0, sales0), _div(gp, sales))
    aqi = _div(_div(None if None in (ca, ppe, ta) else ta - ca - ppe, ta),
               _div(None if None in (ca0, ppe0, ta0) else ta0 - ca0 - ppe0, ta0))
    sgi = _div(sales, sales0)
    depi = _div(_div(dep0, None if None in (dep0, ppe0) else dep0 + ppe0),
                _div(dep, None if None in (dep, ppe) else dep + ppe))
    sgai = _div(_div(sga, sales), _div(sga0, sales0))
    tata = _div(None if None in (ni, cfo) else ni - cfo, ta)
    lvgi = _div(_div(None if None in (cl, ltd, ta) else cl + ltd, ta),
                _div(None if None in (cl0, ltd0, ta0) else cl0 + ltd0, ta0))

    idx = {"DSRI": dsri, "GMI": gmi, "AQI": aqi, "SGI": sgi,
           "DEPI": depi, "SGAI": sgai, "TATA": tata, "LVGI": lvgi}
    missing = [k for k, v in idx.items() if v is None]
    if missing:
        return {"m_score": None, "note": "missing inputs", "missing": missing}
    m = round(-4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi
              + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi, 2)
    return {"m_score": m,
            "verdict": "FLAGGED - investigate" if m > M_SCORE_FLAG else "no manipulation signal",
            "components": {k: round(v, 2) for k, v in idx.items()},
            "note": "Calibrated on US filers; directional only for India. Not proof of fraud."}


# ------------------------------------------------------------------ simple, powerful checks
def cash_conversion(fin: Financials) -> dict:
    """Cumulative operating cash flow vs cumulative reported profit.

    The single most useful fraud tell: a company can fake profit far more easily than cash.
    """
    cfo = [v for v in fin.operating_cash_flow_cr if v is not None]
    ni = [v for v in fin.net_income_cr if v is not None]
    n = min(len(cfo), len(ni))
    if n < 2:
        return {"cfo_to_pat": None, "note": "insufficient history"}
    tot_cfo, tot_ni = sum(cfo[-n:]), sum(ni[-n:])
    ratio = _div(tot_cfo, tot_ni)
    return {"cfo_to_pat": round(ratio, 2) if ratio is not None else None,
            "years": n,
            "cumulative_cfo_cr": round(tot_cfo, 0),
            "cumulative_profit_cr": round(tot_ni, 0),
            "verdict": ("healthy" if ratio and ratio >= 1
                        else "acceptable" if ratio and ratio >= CASH_CONVERSION_MIN
                        else "POOR - profits are not turning into cash")}


def working_capital_days(fin: Financials) -> dict:
    """Receivable / inventory / payable days and the cash conversion cycle, latest vs prior."""
    def days(num_attr, den_attr, idx):
        num, den = _at(getattr(fin, num_attr), idx), _at(getattr(fin, den_attr), idx)
        r = _div(num, den)
        return round(r * 365, 1) if r is not None else None

    out = {}
    for label, num, den in (("receivable_days", "receivables_cr", "revenue_cr"),
                            ("inventory_days", "inventory_cr", "cost_of_revenue_cr"),
                            ("payable_days", "payables_cr", "cost_of_revenue_cr")):
        now, prior = days(num, den, -1), days(num, den, -2)
        out[label] = now
        out[f"{label}_prior"] = prior
        out[f"{label}_change"] = (None if now is None or prior is None else round(now - prior, 1))
    if out["receivable_days"] is not None and out["inventory_days"] is not None:
        ccc = out["receivable_days"] + out["inventory_days"] - (out["payable_days"] or 0)
        out["cash_conversion_cycle_days"] = round(ccc, 1)
    return out


# ------------------------------------------------------------------ the report
def summarize_forensics(fin: Financials, market_cap_cr: float | None = None) -> dict:
    """One compact forensic picture plus an explicit red-flag list."""
    if not fin.fiscal_years:
        return {"forensics": "unavailable", "note": "no financial statements for this symbol"}

    f = piotroski_f_score(fin)
    z = altman_z_emerging(fin)                     # India-appropriate variant
    z_us = altman_z_score(fin, market_cap_cr)      # original, kept for reference only
    m = beneish_m_score(fin)
    cc, wc, icr = cash_conversion(fin), working_capital_days(fin), interest_coverage(fin)
    de, tax = debt_to_ebitda(fin), effective_tax_rate(fin)

    flags: list[str] = []
    if f.get("f_score") is not None and f["f_score"] <= F_SCORE_WEAK:
        flags.append(f"WEAK Piotroski F-Score {f['f_score']}/{f['out_of']} - fundamentals "
                     f"deteriorating. Failed: {', '.join(f['failed_tests'][:4])}.")
    if z.get("z_em_score") is not None and z["zone"] == "DISTRESS":
        flags.append(f"Altman Z-EM {z['z_em_score']} is in the DISTRESS zone (<{ZEM_DISTRESS}) - "
                     "elevated risk of financial distress within ~2 years.")
    if de.get("debt_to_ebitda") is not None and de["debt_to_ebitda"] > DEBT_EBITDA_DANGER:
        flags.append(f"Debt/EBITDA {de['debt_to_ebitda']}x (>{DEBT_EBITDA_DANGER}x) - leverage is at "
                     "the level where refinancing becomes the main risk.")
    for obs in tax.get("observations", []):
        if "below" in obs or "swung" in obs:
            flags.append(f"Effective tax rate: {obs}.")
    if icr.get("interest_coverage") is not None and icr["interest_coverage"] < ICR_DANGER:
        flags.append(f"Interest coverage only {icr['interest_coverage']}x (<{ICR_DANGER}x) - "
                     "operating profit barely covers the interest bill.")
    if m.get("m_score") is not None and m["m_score"] > M_SCORE_FLAG:
        flags.append(f"Beneish M-Score {m['m_score']} exceeds {M_SCORE_FLAG} - statistical "
                     "earnings-manipulation signal. Investigate; not proof.")
    if cc.get("cfo_to_pat") is not None and cc["cfo_to_pat"] < CASH_CONVERSION_MIN:
        flags.append(f"POOR cash conversion: {cc['years']}-year cumulative operating cash flow is "
                     f"only {cc['cfo_to_pat']}x reported profit - earnings are not becoming cash.")
    for label, limit in (("receivable_days", 20.0), ("inventory_days", 30.0)):
        chg = wc.get(f"{label}_change")
        if chg is not None and chg > limit:
            flags.append(f"{label.replace('_', ' ').title()} rose by {chg} days year on year - "
                         "working capital is deteriorating faster than sales.")

    return {
        "piotroski": f, "altman_emerging_markets": z, "altman_original_us_model": z_us,
        "beneish": m, "interest_coverage": icr, "debt_to_ebitda": de,
        "effective_tax_rate": tax,
        "cash_conversion": cc, "working_capital": wc,
        "red_flags": flags or ["No forensic red flags detected in the available statements."],
        "years_analysed": fin.fiscal_years,
    }
