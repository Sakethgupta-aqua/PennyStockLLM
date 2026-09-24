"""The screening funnel: many thousands of listed names down to a handful worth reading.

This is deliberately built the way an Indian institutional desk has worked for decades, and the
ORDER is the part that matters:

  STAGE 1  TRADEABLE      Can a real position be built and exited? Size and, far more importantly,
                          daily TURNOVER. A stock you cannot sell is not an investment.
  STAGE 2  VETO GATES     Disqualifiers, not deductions. A heavily pledged promoter, interest the
                          company cannot cover, or a manipulation flag REMOVES a name outright. No
                          multiple is cheap enough to rescue accounts you cannot trust - which is
                          why cheapness is never allowed to vote until after this stage.
  STAGE 3  RANK           Only survivors are scored, on five pillars weighted the way a long-term
                          Indian investor weights them: business quality and balance-sheet safety
                          together outweigh valuation and price momentum combined.

Scoring uses PERCENTILE RANK within the surviving universe rather than absolute cut-offs. Absolute
thresholds rot - a P/E of 15 meant something different in 2003, 2013 and today - while "cheaper
than 80% of what survived the gates" stays meaningful across cycles.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from pennystock.data.forensics import (DEBT_EBITDA_DANGER, ICR_DANGER, M_SCORE_FLAG, ZEM_DISTRESS)
from pennystock.data.metrics import StockMetrics
from pennystock.data.ownership import PLEDGE_SEVERE
from pennystock.data.sectors import (DEFAULT_PROFILE, PROFILES, SectorProfile,
                                     profile_for)

# A sector needs at least this many names before ranking inside it is meaningful; below it the
# percentile is noise and the whole-universe rank is the lesser evil.
MIN_SECTOR_PEERS = 8

# Veto thresholds. Most simply reuse the analysis thresholds, so a rule cannot drift between the
# single-stock report and the screen. The one deliberate exception is cash conversion: forensics
# FLAGS below 0.7, but a genuinely good company can dip under that in a heavy-capex year, so the
# screen only DISQUALIFIES below 0.5 - a gate has to be harder to trip than a note.
VETO_PLEDGE_PCT = PLEDGE_SEVERE
VETO_ICR = ICR_DANGER
VETO_ZEM = ZEM_DISTRESS
VETO_BENEISH = M_SCORE_FLAG
VETO_DEBT_EBITDA = DEBT_EBITDA_DANGER
VETO_CASH_CONVERSION = 0.5

# Banks, NBFCs and insurers are measured on a DIFFERENT rulebook, and always have been. For a
# lender, growing the loan book is an operating cash OUTFLOW, so cash conversion goes deeply
# negative precisely when the business is doing well - a live Nifty-50 run vetoed BAJFINANCE
# (-3.99), BAJAJFINSV (-6.68), SHRIRAMFIN (-3.21) and SBIN (0.07) on exactly that artefact.
# Deposits are not "debt" in the Debt/EBITDA sense, interest is a lender's cost of goods rather
# than a burden to cover, and Altman and Beneish were never calibrated on financial filers. So the
# leverage and cash gates are SKIPPED for them and the omission is reported, never hidden.
FINANCIAL_SECTORS = {"financial services", "financials", "financial"}
FINANCIAL_INDUSTRY_HINTS = ("bank", "insur", "credit services", "capital markets",
                            "asset management", "financial", "mortgage", "brokerage")
# skipped for lenders; promoter pledge is NOT among them - it means the same thing for any company
SKIPPED_FOR_FINANCIALS = ("interest cover", "Altman distress", "cash conversion",
                          "debt/EBITDA", "Beneish")
# The same metrics must also be excluded from SCORING, not just from the gates. Skipping a check
# at the gate and then quietly ranking on it is the same mistake in a different place: SBIN scored
# in the bottom decile of governance on a cash-conversion figure that means nothing for a bank.
# Excluding them from the ranking POOL as well keeps a lender's balance sheet from distorting the
# percentile distribution every industrial company is measured against.
INVALID_FOR_FINANCIALS = {"debt_to_ebitda", "interest_coverage", "altman_zem",
                          "cash_conversion", "beneish_m", "debt_to_equity_pct"}


def is_financial(m: StockMetrics) -> bool:
    """Is this a lender/insurer, for which the standard solvency rulebook does not apply?"""
    if (m.sector or "").strip().lower() in FINANCIAL_SECTORS:
        return True
    industry = (m.industry or "").lower()
    return any(h in industry for h in FINANCIAL_INDUSTRY_HINTS)


def screening_caveats(m: StockMetrics) -> list[str]:
    """What this screen could NOT judge about a name. Silence about a skipped check reads as a
    pass, which is the most dangerous thing a screen can do - so every sector states the metrics
    it is genuinely judged on that this screen does not have."""
    prof = profile_for(m.sector, m.industry)
    out = []
    if prof.blind_spots:
        out.append(f"[{prof.name}] not measured here: " + "; ".join(prof.blind_spots) +
                   (f". {prof.note}" if prof.note else ""))
    if prof.skip_metrics:
        out.append("Skipped as uncalibrated for this sector: " +
                   ", ".join(sorted(prof.skip_metrics)) + ". Promoter pledge WAS still checked.")
    if not prof.skip_metrics and m.cash_conversion is not None and m.cash_conversion < 0:
        out.append(f"cumulative operating cash flow is negative (CFO/PAT {m.cash_conversion:.2f}). "
                   "If this company consolidates a finance or lending subsidiary that is an "
                   "accounting artefact, not a red flag - verify before dismissing it.")
    return out


# pillar -> (default weight, [(metric, higher_is_better), ...]). A sector rulebook may override
# the weights and the valuation yardstick; see pillars_for().
PILLARS: dict[str, tuple[float, list[tuple[str, bool]]]] = {
    "quality":    (0.30, [("roce_pct", True), ("roe_pct", True),
                          ("profit_margin_pct", True), ("revenue_cagr_pct", True)]),
    "safety":     (0.25, [("debt_to_ebitda", False), ("interest_coverage", True),
                          ("altman_zem", True), ("debt_to_equity_pct", False)]),
    "governance": (0.20, [("pledge_pct_of_promoter", False), ("promoter_holding_pct", True),
                          ("cash_conversion", True), ("piotroski", True), ("beneish_m", False)]),
    "valuation":  (0.15, [("pe", False), ("pb", False)]),
    "momentum":   (0.10, [("pct_from_peak_5y", True), ("above_200dma", True)]),
}


@dataclass
class ScreenResult:
    metrics: StockMetrics
    score: float
    pillars: dict[str, float]
    completeness: float                       # share of scored metrics actually present
    notes: list[str] = field(default_factory=list)


@dataclass
class Funnel:
    """How many names each stage removed - the audit trail that makes a screen defensible."""
    universe: int = 0
    after_liquidity: int = 0
    after_vetoes: int = 0
    veto_reasons: dict[str, int] = field(default_factory=dict)
    results: list[ScreenResult] = field(default_factory=list)

    def describe(self) -> str:
        lines = [f"universe (cached metrics)      {self.universe:>6}",
                 f"tradeable (size + turnover)    {self.after_liquidity:>6}",
                 f"survived the veto gates        {self.after_vetoes:>6}"]
        for reason, n in sorted(self.veto_reasons.items(), key=lambda kv: -kv[1]):
            lines.append(f"    removed by {reason:<34} {n:>5}")
        return "\n".join(lines)


def veto_reasons(m: StockMetrics) -> list[tuple[str, str]]:
    """Hard disqualifiers, as (RULE, detail) pairs, applied through the company's SECTOR RULEBOOK.

    A single set of thresholds disqualifies whole sectors by construction. Measured on a live
    500-name universe the median financial-sector company sits at 5.55x Debt/EBITDA and the median
    utility at 4.31x - both above a 4.5x veto written for manufacturers. Leverage there is the
    business model, not a warning, so each profile can skip a gate outright or move its threshold.

    Every gate still fails OPEN on missing data: absence of evidence is not evidence of a problem.
    """
    prof = profile_for(m.sector, m.industry)
    thr = lambda k, default: prof.veto_overrides.get(k, default)   # noqa: E731
    out: list[tuple[str, str]] = []

    def gate(rule: str) -> bool:
        return rule not in prof.skip_vetoes

    # Promoter pledge is the one rule that means the same thing in every sector, so no profile
    # is allowed to switch it off.
    if m.pledge_pct_of_promoter is not None and m.pledge_pct_of_promoter > VETO_PLEDGE_PCT:
        out.append(("promoter pledge", f"{m.pledge_pct_of_promoter:.0f}% of promoter stake pledged"))
    if gate("Beneish manipulation flag") and m.beneish_m is not None and m.beneish_m > VETO_BENEISH:
        out.append(("Beneish manipulation flag", f"M-score {m.beneish_m:.2f} > {VETO_BENEISH}"))
    if gate("interest cover") and m.interest_coverage is not None:
        limit = thr("interest_coverage", VETO_ICR)
        if m.interest_coverage < limit:
            out.append(("interest cover", f"{m.interest_coverage:.1f}x, below {limit}x"))
    if gate("Altman EM distress zone") and m.altman_zem is not None and m.altman_zem < thr("altman_zem", VETO_ZEM):
        out.append(("Altman EM distress zone", f"Z-EM {m.altman_zem:.2f}"))
    if gate("cash conversion") and m.cash_conversion is not None:
        limit = thr("cash_conversion", VETO_CASH_CONVERSION)
        if m.cash_conversion < limit:
            out.append(("cash conversion", f"cumulative CFO/PAT {m.cash_conversion:.2f}, below {limit}"))
    if gate("debt/EBITDA") and m.debt_to_ebitda is not None:
        limit = thr("debt_to_ebitda", VETO_DEBT_EBITDA)
        if m.debt_to_ebitda > limit:
            out.append(("debt/EBITDA", f"{m.debt_to_ebitda:.1f}x, above {limit}x"))
    return out


def _percentiles(rows: list[StockMetrics], metric: str, higher_is_better: bool) -> dict[str, float]:
    """Rank one metric across the rows given, 0.0 (worst) to 1.0 (best). Ties share a rank."""
    present = sorted([(v, r.symbol) for r in rows if (v := getattr(r, metric)) is not None])
    if len(present) < 2:
        return {}
    n = len(present) - 1
    ranked = {sym: i / n for i, (_, sym) in enumerate(present)}
    return ranked if higher_is_better else {s: 1.0 - p for s, p in ranked.items()}


def sector_percentiles(rows: list[StockMetrics], metric: str,
                       higher_is_better: bool) -> dict[str, float]:
    """Percentile rank WITHIN the company's own sector rulebook, not across the whole market.

    This is the single most important correction in the screen. Ranking ROCE across every sector at
    once does not measure quality, it measures asset intensity: software is structurally at 21%
    and a regulated utility at 7%, so the best utility in the country loses to a mediocre IT
    company every single time. Debt/Equity is worse - a 25x structural spread - so every bank and
    utility scores near zero on safety by construction.

    Sectors with too few names to rank meaningfully fall back to the whole applicable universe;
    a percentile drawn from four peers is noise wearing a decimal point.
    """
    pool = [r for r in rows if applies_to(r, metric)]
    groups: dict[str, list[StockMetrics]] = defaultdict(list)
    for r in pool:
        groups[profile_for(r.sector, r.industry).name].append(r)

    out: dict[str, float] = {}
    orphans: list[StockMetrics] = []
    for group in groups.values():
        if len(group) >= MIN_SECTOR_PEERS:
            out.update(_percentiles(group, metric, higher_is_better))
        else:
            orphans.extend(group)
    if orphans:
        wide = _percentiles(pool, metric, higher_is_better)
        out.update({r.symbol: wide[r.symbol] for r in orphans if r.symbol in wide})
    return out


def pillars_for(profile: SectorProfile) -> dict[str, tuple[float, list[tuple[str, bool]]]]:
    """The pillar definition this company is scored on: its rulebook's weights, and its rulebook's
    valuation yardstick. Banks are bought and sold on book value, so P/B leads for a lender while
    P/E leads for an asset-light compounder."""
    spec = {}
    for name, (weight, metrics) in PILLARS.items():
        if name == "valuation" and profile.valuation_metrics:
            metrics = list(profile.valuation_metrics)
        spec[name] = (profile.weights.get(name, weight), metrics)
    total = sum(w for w, _ in spec.values()) or 1.0
    return {k: (w / total, m) for k, (w, m) in spec.items()}      # weights must sum to 1


def applies_to(m: StockMetrics, metric: str) -> bool:
    """Is this metric meaningful for this kind of business at all?"""
    return metric not in profile_for(m.sector, m.industry).skip_metrics


def score_universe(rows: list[StockMetrics]) -> list[ScreenResult]:
    """Percentile-score every row against ITS OWN SECTOR, on its own rulebook's weights."""
    metrics_used = {m for _, ms in PILLARS.values() for m, _ in ms}
    metrics_used |= {m for p in (*PROFILES, DEFAULT_PROFILE) for m, _ in p.valuation_metrics}
    hib = {m: h for _, ms in PILLARS.values() for m, h in ms}
    hib.update({m: h for p in (*PROFILES, DEFAULT_PROFILE) for m, h in p.valuation_metrics})
    ranks = {m: sector_percentiles(rows, m, hib[m]) for m in metrics_used}

    results = []
    for r in rows:
        prof = profile_for(r.sector, r.industry)
        spec = pillars_for(prof)
        pillars, present, total = {}, 0, 0
        for name, (_, metrics) in spec.items():
            applicable = [m for m, _ in metrics if applies_to(r, m)]
            got = [ranks[m][r.symbol] for m in applicable if r.symbol in ranks.get(m, {})]
            # completeness measures DATA availability, so metrics that do not apply to this kind
            # of company are excluded from the denominator rather than counted as gaps
            total += len(applicable)
            present += len(got)
            # a pillar with no data scores 0.5 - neutral, so a gap neither rewards nor punishes
            pillars[name] = round(sum(got) / len(got), 3) if got else 0.5
        score = round(sum(w * pillars[n] for n, (w, _) in spec.items()) * 100, 1)
        notes = screening_caveats(r)
        if r.red_flag_count:
            notes.append(f"{r.red_flag_count} forensic red flag(s)")
        if r.pledge_pct_of_promoter:
            notes.append(f"pledge {r.pledge_pct_of_promoter:.0f}% of promoter stake")
        results.append(ScreenResult(metrics=r, score=score, pillars=pillars,
                                    completeness=round(present / total, 2) if total else 0.0,
                                    notes=notes))
    return sorted(results, key=lambda x: -x.score)


def balance_by_sector(results: list[ScreenResult], top_n: int) -> list[ScreenResult]:
    """Round-robin across sector rulebooks: the best name from each sector, then the second best
    from each, and so on until the shortlist is full.

    A straight top-N is dominated by whichever sectors happen to rank well in the current cycle,
    which is exactly the concentration risk a shortlist is supposed to help you avoid. Taking the
    leader of every sector first guarantees the list spans the market. Sectors that run out simply
    stop contributing, so a thin sector never pads the list with weak names.
    """
    groups: dict[str, list[ScreenResult]] = defaultdict(list)
    for r in results:                                  # results arrive already sorted by score
        groups[profile_for(r.metrics.sector, r.metrics.industry).name].append(r)
    picked: list[ScreenResult] = []
    depth = 0
    while len(picked) < top_n and any(len(g) > depth for g in groups.values()):
        for name in sorted(groups):                    # alphabetical = deterministic runs
            if len(groups[name]) > depth and len(picked) < top_n:
                picked.append(groups[name][depth])
        depth += 1
    return sorted(picked, key=lambda r: -r.score)


def screen(rows: list[StockMetrics], min_market_cap_cr: float = 500.0,
           min_turnover_cr: float = 1.0, min_completeness: float = 0.5,
           apply_vetoes: bool = True) -> Funnel:
    """Run the whole funnel and return both the survivors AND the audit trail of what was removed."""
    f = Funnel(universe=len(rows))

    tradeable = [r for r in rows
                 if (r.market_cap_cr or 0) >= min_market_cap_cr
                 and (r.turnover_cr or 0) >= min_turnover_cr]
    f.after_liquidity = len(tradeable)

    survivors = []
    for r in tradeable:
        reasons = veto_reasons(r) if apply_vetoes else []
        if reasons:
            for rule, _detail in reasons:            # bucket by RULE, never by the measured value
                f.veto_reasons[rule] = f.veto_reasons.get(rule, 0) + 1
            continue
        survivors.append(r)
    f.after_vetoes = len(survivors)

    scored = score_universe(survivors)
    # a name scored on almost no data is noise dressed as a ranking, so drop it from the OUTPUT
    # rather than from the universe - it still counted towards everyone else's percentiles
    f.results = [s for s in scored if s.completeness >= min_completeness]
    return f
