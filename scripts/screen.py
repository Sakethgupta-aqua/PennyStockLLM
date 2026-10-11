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
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s | %(message)s")

from pennystock.config import settings                                       # noqa: E402
from pennystock.data.broker_metrics import kpi_snapshot, screening_kpis      # noqa: E402
from pennystock.data.broker_reports import KPI_LABELS                         # noqa: E402
from pennystock.data.consensus import broker_consensus                        # noqa: E402
from pennystock.data.metrics import load_metrics                              # noqa: E402
from pennystock.data.screener import (BROKER_KPI_METRICS, applies_to,         # noqa: E402
                                      balance_by_sector, kpi_coverage, pillars_for, screen,
                                      screening_caveats, veto_reasons)
from pennystock.data.sectors import blind_spot_coverage, profile_for         # noqa: E402
from pennystock.data import event_signals, trendlyne_scores                  # noqa: E402


def _f(v, nd=1, dash="-"):
    return dash if v is None else f"{v:.{nd}f}"


def _kpi_name(metric: str) -> str:
    return KPI_LABELS.get(metric, metric)


def _broker_kpis() -> dict:
    """Sector KPIs for scoring. A missing or locked broker table must never stop the screen."""
    try:
        return screening_kpis(BROKER_KPI_METRICS)
    except Exception as e:
        logging.warning(f"broker KPIs unavailable ({type(e).__name__}: {e}); screening without them")
        return {}


def print_kpi_panel(symbol: str, rows, kpis: dict, prof) -> set[str]:
    """Layer 1: what broker reports say this company's sector KPIs are. Returns the KPIs found."""
    snap = kpi_snapshot(symbol)
    if not snap:
        print("\n  SECTOR KPIs FROM BROKER REPORTS: none parsed for this company "
              "(run scripts/ingest_broker_reports.py once its reports are downloaded)")
        return set()
    stale_before = (date.today() - timedelta(days=settings.broker_max_age_days)).isoformat()
    print("\n  SECTOR KPIs FROM BROKER REPORTS  (what each broker printed, on its own basis - "
          "an attributed opinion, not an audited figure)")
    shown = None
    for v in snap:
        value = _f(v.median, 2) + (f" ({_f(v.low, 2)}-{_f(v.high, 2)})" if v.n > 1 else "")
        note = "STALE " if v.latest_report < stale_before else ""
        print(f"    {_kpi_name(v.metric) if v.metric != shown else '':<22}{v.period:<8}"
              f"{v.period_kind:<9}{value:<20}n={v.n:<3}{v.status:<8}"
              f"{note}{v.latest_report}  basis: {', '.join(v.labels)[:48]}")
        shown = v.metric
    have = {v.metric for v in snap}
    covered, missing = blind_spot_coverage(prof, have)
    if covered:
        print(f"    blind spots now covered: {'; '.join(covered)}")
    if missing:
        print(f"    still not measured:      {'; '.join(missing)}")
    if prof.kpi_pillars:
        cov = kpi_coverage(rows, kpis, prof.name)
        need = settings.broker_kpi_min_companies
        voting = sorted(m for m, n in cov.items() if n >= need)
        print(f"    in the score: {', '.join(voting)}" if voting else
              f"    in the score: not yet - a KPI votes once {need} {prof.name} names carry it "
              f"(best now: {max(cov.values()) if cov else 0})")
    return have


def print_consensus_panel(symbol: str) -> None:
    """Layer 2: what the brokers say, how much they disagree, which way they are moving."""
    try:
        c = broker_consensus(symbol)
    except Exception as e:                      # the ledger is a bonus here, never a blocker
        print(f"\n  WHAT THE BROKERS SAY: unavailable ({type(e).__name__}: {e})")
        return
    if not c["calls"] and not c["stale_excluded"]:
        print("\n  WHAT THE BROKERS SAY: no broker calls on record "
              "(python scripts/backfill_broker_predictions.py)")
        return
    print(f"\n  WHAT THE BROKERS SAY  ({c['brokers']} current call(s), one per broker, "
          f"latest within {settings.broker_max_age_days} days - opinions, not forecasts)")
    for x in c["calls"]:
        prev = x["previous"]
        was = f"   was {prev['rating']} @ {_f(prev['target'], 0)} on {prev['date']}" if prev else ""
        habit = (f"   [rates {x['broker_bullish_share']:.0%} of {x['broker_calls_on_record']} calls bullish]"
                 if x.get("broker_bullish_share") is not None else "")
        print(f"    {x['date']}  {x['broker'][:26]:<28}{x['rating']:<12}target {_f(x['target'], 0):>8}"
              f"{was}{habit}")
    if c["brokers"]:
        mix, t = c["rating_mix"], c["target"]
        print(f"    mix: {mix['bullish']} bullish / {mix['neutral']} neutral / {mix['bearish']} bearish"
              f"   mean rating {c['mean_score']} (1 = Buy ... 5 = Sell)")
        if t:
            spread = f"   spread {t['dispersion_pct']}% of median" if t["dispersion_pct"] is not None else ""
            print(f"    targets: median {_f(t['median'], 0)}  range {_f(t['low'], 0)}-{_f(t['high'], 0)}{spread}")
        r, d = c["revisions"], c["drift"]
        print(f"    last {r['window_days']} days: targets raised {r['target_raised']}, cut {r['target_cut']};"
              f" upgrades {r['upgrades']}, downgrades {r['downgrades']}"
              + (f"; consensus {d['direction']} ({d['mean_score_then']} -> {d['mean_score_now']})" if d else ""))
    eps = c.get("eps_revisions")
    print(f"    EPS estimate revisions: {eps['raised']} raised / {eps['cut']} cut / "
          f"{eps['unchanged']} unchanged" if eps else
          "    EPS estimate revisions: n/a (needs two parsed reports from the same broker)")
    for f in c["flags"]:
        print(f"    ! {f}")


def _rs(v) -> str:
    if not v:
        return "-"
    return f"Rs {v / 1e7:,.1f} cr" if v >= 1e7 else f"Rs {v / 1e5:,.1f} lakh"


def print_events_panel(symbol: str) -> None:
    """The last 12 months of NSE disclosures that move stocks, and how each type has done in the ledger."""
    since = (date.today() - timedelta(days=365)).isoformat()
    try:
        events = event_signals.load_events(symbol, since=since)
    except Exception as e:                      # a bonus panel, never a blocker
        print(f"\n  EVENTS: unavailable ({type(e).__name__}: {e})")
        return
    if not events:
        print("\n  EVENTS: no stored events in the last 12 months (scripts/ingest_events.py)")
        return
    print("\n  EVENTS  (NSE disclosures, last 12 months - dated when the market learned them)")
    for e in events[-15:]:
        d = e.details
        who = (f"{d.get('person')} ({d.get('category')}) {d.get('mode', '').lower()} {_rs(d.get('value_rs'))}"
               if d.get("person") else
               f"{d.get('agency')}: {d.get('rating')} - {d.get('text', '')[:60]}" if d.get("agency") else
               d.get("subject", "")[:70] + (f" - {d.get('text', '')[:60]}" if d.get("text") else ""))
        mark = {"veto": "!!", "signal": " >", "info": "  "}[e.severity]
        print(f"   {mark} {e.signal_at[:10]}  {e.event_type:<24}{who}"[:150])
    if len(events) > 15:
        print(f"      ... {len(events) - 15} earlier event(s)")
    buys = [e for e in events if e.event_type == "insider_buy"]
    sells = [e for e in events if e.event_type == "insider_sell"]
    if buys or sells:
        print(f"    insiders: {len(buys)} buy(s) {_rs(sum(e.details.get('value_rs') or 0 for e in buys))}"
              f" / {len(sells)} sell(s) {_rs(sum(e.details.get('value_rs') or 0 for e in sells))}")
    records = []
    for etype in sorted({e.event_type for e in events if e.severity != "info"}):
        t = event_signals.track_record(etype)
        rate = f"{t['hit_rate']:.0%}" if t["hit_rate"] is not None else "n/a"
        records.append(f"{etype} {rate} of {t['decided_sample']} decided ({t['companies']} cos)")
    if records:
        print("    track record at 20 sessions: " + "  |  ".join(records))


def print_trendlyne_panel(symbol: str, ours: dict) -> None:
    """Trendlyne's scores beside ours: a third-party opinion, what moved, and where it disagrees."""
    try:
        hist = trendlyne_scores.history(symbol)
    except Exception as e:                      # a bonus panel, never a blocker
        print(f"\n  TRENDLYNE'S VIEW: unavailable ({type(e).__name__}: {e})")
        return
    if not hist:
        print("\n  TRENDLYNE'S VIEW: no Trendlyne Stock Report on file "
              "(scripts/fetch_stock_reports.py, then scripts/parse_stock_reports.py)")
        return
    _, s = hist[-1]
    direction, conviction = trendlyne_scores.lean(s)
    print(f"\n  TRENDLYNE'S VIEW  (third-party scores - an opinion, not this analysis; data of {s.as_of})")
    print("    " + "  |  ".join(
        f"{k.capitalize()} {_f(getattr(s, k), 0)} {getattr(s, k + '_label') or ''} "
        f"[{trendlyne_scores.band_of(s, k) or 'n/a'}]" for k in trendlyne_scores.KINDS))
    checked = (s.checklist_passed or 0) + (s.checklist_failed or 0)
    print(f"    {s.classification or '(no classification printed)'}; graded in the ledger as {direction} "
          f"(conviction {conviction}); checklist {_f(s.checklist_pass_pct)}% passed"
          + (f" ({s.checklist_passed} of {checked})" if checked else ""))
    if s.analysts or s.target_price:
        upside = f" ({s.upside_pct:+.1f}%)" if s.upside_pct is not None else ""
        spread = f", estimates {_f(s.est_low, 0)}-{_f(s.est_high, 0)}" if s.est_low is not None else ""
        print(f"    analysts: {s.consensus or 'no consensus word'} from {s.analysts or '?'}, "
              f"average target {_f(s.target_price, 0)}{upside}{spread}")
    if s.pe_zone:
        print(f"    P/E in its own {s.pe_zone} zone "
              f"(spent {_f(s.pe_zone_pct)}% of its history below today's P/E)")
    if len(hist) > 1:
        prev = hist[-2][1]
        moved = trendlyne_scores.changes(prev, s)
        said = [f"{k} {v:+g}" if isinstance(v, (int, float)) else f"{k} {v}" for k, v in moved.items()]
        print(f"    since {prev.as_of}: " + (", ".join(said) or "no change"))
    print("    vs our pillars: " + "  |  ".join(
        f"{r['ours']} {_f(r['our_score'], 2)} ~ {r['theirs']} {r['their_band'] or 'n/a'}: {r['verdict']}"
        for r in trendlyne_scores.agreement(ours, s)))


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
    vetoes = veto_reasons(m) + event_signals.veto_map().get(symbol, [])
    if vetoes:
        print("\n  VETO GATES: FAILS")
        for rule, detail in vetoes:
            print(f"    x {rule:<26} {detail}")
    else:
        print("\n  VETO GATES: passes all")
    kpis = _broker_kpis()
    have = print_kpi_panel(symbol, rows, kpis, prof)
    print_consensus_panel(symbol)
    for c in screening_caveats(m, covered=have):
        print(f"\n  CAVEAT: {c}")
    if prof.blind_spots:
        _, missing = blind_spot_coverage(prof, have)
        if missing:
            print("\n  THIS SECTOR IS ALSO JUDGED ON (none of it is in this screen):")
            for b in missing:
                print(f"    - {b}")

    f = screen(rows, args.min_market_cap, args.min_turnover, apply_vetoes=False, kpis=kpis)
    res = next((r for r in f.results if r.metrics.symbol == symbol), None)
    if res:
        peers = sum(1 for x in rows if profile_for(x.sector, x.industry).name == prof.name)
        print(f"\n  SCORE {res.score}/100   (percentile rank against the {peers} "
              f"{prof.name} names in the cache, NOT the whole market;"
              f" data completeness {res.completeness:.0%})")
        for name, (w, metrics) in pillars_for(prof).items():
            # mark what does not apply, rather than printing a value that had no vote
            parts = [f"{mm}={getattr(m, mm, kpis.get(symbol, {}).get(mm))}" if applies_to(m, mm)
                     else f"{mm}=n/a" for mm, _ in metrics]
            print(f"    {name:<11} {res.pillars[name]:.2f}  x{w:<5}  {', '.join(parts)}"[:158])
    print_trendlyne_panel(symbol, res.pillars if res else {})
    print_events_panel(symbol)
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
               apply_vetoes=not args.no_vetoes, kpis=_broker_kpis(), events=event_signals.veto_map())
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
