"""Ownership analysis - promoter holding, institutional flows, and promoter PLEDGE.

Why this matters: for Indian small/micro-caps, promoter pledging is the single most reliable
red flag. When the stock falls, lenders issue margin calls and dump the pledged shares, which
pushes the price lower and triggers more margin calls - a forced-selling feedback loop that has
repeatedly wiped out minority shareholders.

All scoring here is PURE and computed in code (never by the LLM), using published practitioner
thresholds: 0% ideal, <=10% low, <=30% caution, <=50% high, >50% severe. Crucially the TREND
matters as much as the level - pledge rising across consecutive quarters is a red flag at any
level, and a steadily falling promoter stake is a warning in its own right.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# (upper bound %, label) - first band whose bound is >= the value wins
PLEDGE_SEVERE = 50.0        # above half the promoter stake pledged: a disqualifier, not a demerit
PLEDGE_BANDS = ((0.0, "none (ideal)"), (10.0, "low"), (30.0, "caution"),
                (PLEDGE_SEVERE, "high"), (float("inf"), "severe"))

PROMOTER_DROP_PP = 1.0      # promoter stake falling this many points is worth flagging
INSTITUTIONAL_DROP_PP = 2.0 # FII+DII exiting this much is worth flagging
BLOCKING_STAKE_PCT = 26.0   # below this, promoters cannot block a special resolution


@dataclass
class Quarter:
    """One quarterly shareholding filing (SEBI LODR Reg 31 - filed 4x a year)."""
    quarter: str                                   # e.g. "Jun 2026"
    promoter_pct: float | None = None
    fii_pct: float | None = None
    dii_pct: float | None = None
    public_pct: float | None = None
    government_pct: float | None = None
    shareholders: int | None = None
    pledge_pct: float | None = None                # % of PROMOTER holding - the industry convention
    pledge_pct_of_total: float | None = None       # % of TOTAL shares - what NSE reports as headline
    pledge_source: str | None = None               # "nse" | "manual" | ... provenance matters


@dataclass
class Ownership:
    symbol: str
    quarters: list[Quarter] = field(default_factory=list)   # oldest -> newest
    fetched_at: str | None = None


def pledge_band(pct: float | None) -> str:
    if pct is None:
        return "unknown"
    for ceiling, label in PLEDGE_BANDS:
        if pct <= ceiling:
            return label
    return "severe"


def _change(values: list[float | None]) -> float | None:
    """Change from the oldest to the newest non-null value, in percentage POINTS."""
    vals = [v for v in values if v is not None]
    return round(vals[-1] - vals[0], 2) if len(vals) >= 2 else None


def summarize_ownership(own: Ownership, lookback: int = 4) -> dict:
    """Turn quarterly filings into a compact, LLM-ready ownership picture + explicit risk flags."""
    qs = own.quarters
    if not qs:
        return {"ownership": "unavailable",
                "note": "No shareholding data cached for this symbol. Run scripts/ingest_ownership.py."}

    latest = qs[-1]
    window = qs[-(lookback + 1):]                  # N+1 points give an N-quarter change
    span = len(window) - 1

    def inst(q: Quarter) -> float | None:
        if q.fii_pct is None and q.dii_pct is None:
            return None
        return round((q.fii_pct or 0) + (q.dii_pct or 0), 2)

    promoter_chg = _change([q.promoter_pct for q in window])
    inst_chg = _change([inst(q) for q in window])
    pledge_chg = _change([q.pledge_pct for q in window])

    flags: list[str] = []
    if latest.pledge_pct is None:
        flags.append("PLEDGE UNKNOWN - promoter pledge is not available from the configured free "
                     "sources. Do NOT assume it is zero; state that it is unverified.")
    else:
        if latest.pledge_pct > 50:
            flags.append(f"SEVERE: {latest.pledge_pct}% of PROMOTER HOLDING is pledged - "
                         "forced-selling / margin-call risk if the price falls.")
        elif latest.pledge_pct > 30:
            flags.append(f"HIGH RISK: {latest.pledge_pct}% of promoter holding is pledged.")
        elif latest.pledge_pct > 10:
            flags.append(f"CAUTION: {latest.pledge_pct}% of promoter holding is pledged.")
        if pledge_chg is not None and pledge_chg > 0:
            flags.append(f"Pledge RISING (+{pledge_chg}pp over {span} quarter(s)) - "
                         "an increasing trend is a red flag at any absolute level.")

    if promoter_chg is not None and promoter_chg <= -PROMOTER_DROP_PP:
        flags.append(f"Promoter stake FALLING ({promoter_chg}pp over {span} quarter(s)) - "
                     "promoters reducing skin in the game.")
    if latest.promoter_pct is not None and latest.promoter_pct < BLOCKING_STAKE_PCT:
        flags.append(f"Low promoter holding ({latest.promoter_pct}%) - below the "
                     f"{BLOCKING_STAKE_PCT}% blocking-stake level.")
    if inst_chg is not None and inst_chg <= -INSTITUTIONAL_DROP_PP:
        flags.append(f"Institutions REDUCING ({inst_chg}pp of FII+DII over {span} quarter(s)).")

    return {
        "symbol": own.symbol,
        "as_of_quarter": latest.quarter,
        "quarters_available": len(qs),
        "promoter_pct": latest.promoter_pct,
        f"promoter_change_{span}q_pp": promoter_chg,
        "fii_pct": latest.fii_pct,
        "dii_pct": latest.dii_pct,
        "institutional_pct": inst(latest),
        f"institutional_change_{span}q_pp": inst_chg,
        "public_pct": latest.public_pct,
        "shareholders": latest.shareholders,
        "pledge_pct_of_promoter_holding": latest.pledge_pct,
        "pledge_pct_of_total_shares": latest.pledge_pct_of_total,
        "pledge_band": pledge_band(latest.pledge_pct),
        "pledge_source": latest.pledge_source,
        "risk_flags": flags or ["No ownership red flags detected in the available data."],
        "history": [{"q": q.quarter, "promoter": q.promoter_pct, "pledge": q.pledge_pct}
                    for q in qs[-8:]],
    }
