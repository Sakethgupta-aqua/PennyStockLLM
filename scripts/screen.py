"""
Rank the cached universe and show the FUNNEL that produced the shortlist.

Reads the stock_metrics cache (populate it with scripts/ingest_metrics.py) and applies the same
three stages a professional Indian screen has always applied: tradeable -> veto gates -> rank.

  python scripts/screen.py --top 25
  python scripts/screen.py --top 20 --min-turnover 5      # only properly liquid names
  python scripts/screen.py --sector "Basic Materials"
  python scripts/screen.py --explain HINDALCO             # why one stock scored what it did
  python scripts/screen.py --no-vetoes                    # see what the gates removed

The funnel is printed every time on purpose: a shortlist you cannot explain is not research.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s | %(message)s")

from pennystock.data.metrics import load_metrics                              # noqa: E402
from pennystock.data.screener import (applies_to, balance_by_sector, pillars_for,
                                      screen, screening_caveats, veto_reasons)
from pennystock.data.sectors import profile_for            # noqa: E402


def _f(v, nd=1, dash="-"):
    return dash if v is None else f"{v:.{nd}f}"


def explain(symbol: str, rows, args) -> int:
    symbol = symbol.upper()
    m = next((r for r in rows if r.symbol == symbol), None)
    if m is None:
        print(f"{symbol} is not in the metrics cache. "
              f"Run: python scripts/ingest_metrics.py {symbol}")
        return 1
    prof = profile_for(m.sector, m.industry)
    print(f"\n{symbol} - {m.company or ''}\n{'=' * 74}")
    print(f"  sector {m.sector} / {m.industry}")
    print(f"  rulebook: {prof.name}  (ranked against other {prof.name} names, not the whole market)")
    print(f"  market cap {_f(m.market_cap_cr)}cr | median turnover {_f(m.turnover_cr, 2)}cr/day")
    vetoes = veto_reasons(m)
    if vetoes:
        print("\n  VETO GATES: FAILS")
        for rule, detail in vetoes:
            print(f"    x {rule:<26} {detail}")
    else:
        print("\n  VETO GATES: passes all")
    for c in screening_caveats(m):
        print(f"\n  CAVEAT: {c}")
    if prof.blind_spots:
        print(f"\n  THIS SECTOR IS REALLY JUDGED ON (and none of it is in this screen):")
        for b in prof.blind_spots:
            print(f"    - {b}")

    f = screen(rows, args.min_market_cap, args.min_turnover, apply_vetoes=False)
    res = next((r for r in f.results if r.metrics.symbol == symbol), None)
    if res:
        peers = sum(1 for x in rows if profile_for(x.sector, x.industry).name == prof.name)
        print(f"\n  SCORE {res.score}/100   (percentile rank against the {peers} "
              f"{prof.name} names in the cache, NOT the whole market;"
              f" data completeness {res.completeness:.0%})")
        for name, (w, metrics) in pillars_for(prof).items():
            # mark what does not apply, rather than printing a value that had no vote
            parts = [f"{mm}={getattr(m, mm)}" if applies_to(m, mm) else f"{mm}=n/a"
                     for mm, _ in metrics]
            print(f"    {name:<11} {res.pillars[name]:.2f}  x{w:<5}  {', '.join(parts)}"[:158])
    if m.red_flags:
        print(f"\n  FORENSIC RED FLAGS ({m.red_flag_count}):")
        for flag in m.red_flags.split(" | "):
            print(f"    - {flag[:110]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Screen the cached universe.")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--min-market-cap", type=float, default=500.0, help="Rs crore")
    ap.add_argument("--min-turnover", type=float, default=1.0, help="Rs crore of median daily value")
    ap.add_argument("--min-completeness", type=float, default=0.5)
    ap.add_argument("--sector", help="restrict to one sector (yfinance label)")
    ap.add_argument("--balanced", action="store_true",
                    help="give every sector equal weight in the shortlist (round-robin by rulebook)")
    ap.add_argument("--rulebook", help="restrict to one sector rulebook, e.g. financials, "
                                       "materials_cyclical, technology")
    ap.add_argument("--no-vetoes", action="store_true", help="skip the disqualifying gates")
    ap.add_argument("--explain", help="show one stock's scorecard and exit")
    ap.add_argument("--max-age-days", type=int, help="ignore cached rows older than this")
    args = ap.parse_args()

    rows = load_metrics(args.max_age_days)
    if not rows:
        print("No cached metrics. Run: python scripts/ingest_metrics.py --index totalmarket")
        return 1
    if args.explain:
        return explain(args.explain, rows, args)
    if args.sector:
        rows = [r for r in rows if (r.sector or "").lower() == args.sector.lower()]
    if args.rulebook:
        rows = [r for r in rows
                if profile_for(r.sector, r.industry).name == args.rulebook.lower()]

    f = screen(rows, args.min_market_cap, args.min_turnover, args.min_completeness,
               apply_vetoes=not args.no_vetoes)
    shown = balance_by_sector(f.results, args.top) if args.balanced else f.results[: args.top]
    print(f"\nSCREENING FUNNEL\n{'-' * 74}\n{f.describe()}")
    print(f"scored and shown                  {len(shown):>6}"
          f"   (of {len(f.results)} with enough data"
          f"{', balanced across sectors' if args.balanced else ''})\n")

    hdr = (f"  {'#':<4}{'SYMBOL':<13}{'RULEBOOK':<19}{'SCORE':>6}{'QUAL':>6}{'SAFE':>6}"
           f"{'GOV':>6}{'VAL':>6}{'MOM':>6}   {'ROCE':>7}{'PE':>8}  FLAGS")
    print(hdr + "\n" + "-" * len(hdr))
    for i, r in enumerate(shown, 1):
        m, p = r.metrics, r.pillars
        print(f"  {i:<4}{m.symbol:<13}{profile_for(m.sector, m.industry).name:<19}"
              f"{r.score:>6.1f}{p['quality']:>6.2f}{p['safety']:>6.2f}"
              f"{p['governance']:>6.2f}{p['valuation']:>6.2f}{p['momentum']:>6.2f}   "
              f"{_f(m.roce_pct):>7}{_f(m.pe):>8}  "
              f"{m.red_flag_count if m.red_flag_count is not None else '-'}")
    print("\nScores are PERCENTILE RANKS WITHIN EACH SECTOR'S OWN RULEBOOK - a utility is ranked")
    print("against utilities, a bank against banks. Comparing a bank's ROCE to a software firm's")
    print("measures asset intensity, not quality. Run --explain to see what a sector is really")
    print("judged on and which of those metrics this screen does NOT have.")
    print("A shortlist is where research starts - run demo.py on the names that interest you.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
